__all__ = ["NonstationaryTransformer"]

from math import sqrt
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast.common._modules import (
    AttentionLayer,
    DataEmbedding,
    TriangularCausalMask,
)

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class DSAttention(nn.Module):
    """De-stationary full attention with learned scale and bias factors."""

    def __init__(
        self,
        mask_flag: bool = True,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        bsz, q_len, _, head_dim = queries.shape
        _, k_len, _, _ = keys.shape
        scale = self.scale or 1.0 / sqrt(head_dim)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(bsz, q_len, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attn, values)
        return output.contiguous(), (attn if self.output_attention else None)


class Projector(nn.Module):
    """Learns sequence-wise de-stationary factors from raw series statistics."""

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        hidden_dims: list[int],
        output_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        dims = [2 * enc_in, *hidden_dims, output_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        batch_size = x.shape[0]
        series_feat = self.series_conv(x)
        x = torch.cat([series_feat, stats], dim=1).reshape(batch_size, -1)
        return self.backbone(x)


class NSTransEncoderLayer(nn.Module):
    def __init__(self, attention, hidden_size, conv_hidden_size, dropout, activation):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(hidden_size, conv_hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(conv_hidden_size, hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class NSTransEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class NSTransDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, hidden_size, conv_hidden_size, dropout, activation):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(hidden_size, conv_hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(conv_hidden_size, hidden_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0]
        )
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class NSTransDecoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class NonstationaryTransformer(BaseModel):
    """NonstationaryTransformer.

    Direct forecasting transformer with learned de-stationary attention factors.

    References:
        - [Yong Liu, Haixu Wu, Haoran Zhang, et al. "Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting".](https://arxiv.org/abs/2205.14415)
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y: bool = False,
        decoder_input_size_multiplier: float = 0.5,
        hidden_size: int = 128,
        dropout: float = 0.05,
        n_head: int = 4,
        conv_hidden_size: int = 128,
        activation: str = "gelu",
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        p_hidden_dims: Optional[list[int]] = None,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        self.label_len = int(np.ceil(input_size * decoder_input_size_multiplier))
        if not 0 < self.label_len < input_size:
            raise ValueError(
                f"decoder_input_size_multiplier={decoder_input_size_multiplier} must keep label_len in (0, input_size)"
            )
        if activation not in {"relu", "gelu"}:
            raise ValueError(f"Unsupported activation={activation}")

        self.hidden_size = hidden_size
        self.c_out = self.loss.outputsize_multiplier
        self.enc_in = 1
        self.dec_in = 1
        self.output_attention = False
        hidden_dims = p_hidden_dims or [hidden_size, hidden_size]

        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
            exog_input_size=0,
            hidden_size=hidden_size,
            pos_embedding=True,
            dropout=dropout,
        )
        self.dec_embedding = DataEmbedding(
            c_in=self.dec_in,
            exog_input_size=0,
            hidden_size=hidden_size,
            pos_embedding=True,
            dropout=dropout,
        )

        self.encoder = NSTransEncoder(
            [
                NSTransEncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=self.output_attention,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size=hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(encoder_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size),
        )
        self.decoder = NSTransDecoder(
            [
                NSTransDecoderLayer(
                    AttentionLayer(
                        DSAttention(
                            mask_flag=True,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    AttentionLayer(
                        DSAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size=hidden_size,
                    conv_hidden_size=conv_hidden_size,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(decoder_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size),
            projection=nn.Linear(hidden_size, self.c_out, bias=True),
        )

        self.tau_learner = Projector(
            enc_in=self.enc_in,
            seq_len=input_size,
            hidden_dims=hidden_dims,
            output_dim=1,
        )
        self.delta_learner = Projector(
            enc_in=self.enc_in,
            seq_len=input_size,
            hidden_dims=hidden_dims,
            output_dim=input_size,
        )

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        x_raw = insample_y.detach().clone()

        mean_enc = insample_y.mean(1, keepdim=True).detach()
        x_enc = insample_y - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        tau = self.tau_learner(x_raw, std_enc).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        x_dec = torch.cat(
            [x_enc[:, -self.label_len :, :], torch.zeros_like(x_enc[:, -self.h :, :])],
            dim=1,
        )

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec, None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)
        forecast = dec_out[:, -self.h :, :]
        return forecast * std_enc + mean_enc
