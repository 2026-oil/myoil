__all__ = ["DeepEDM"]

import math
from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe.unsqueeze(0).unsqueeze(0), requires_grad=True)

    def forward(self, x, offset: int = 0):
        return self.pe[:, :, offset : offset + x.size(2)]


class InputEncoder(nn.Module):
    def __init__(self, mlp_layers, lookback_len, pred_len, activation, dropout):
        super().__init__()
        layers = []
        for layer_idx in range(mlp_layers):
            in_features = lookback_len if layer_idx == 0 else pred_len
            layers.append(nn.Linear(in_features, pred_len))
            if layer_idx < mlp_layers - 1:
                layers.extend([nn.Dropout(dropout), activation])
        self.mlp_projection = nn.Sequential(*layers)

    def forward(self, x):
        skip = self.mlp_projection(x)
        return x, skip, skip


class EDMBlock(nn.Module):
    def __init__(
        self,
        lookback_len: int,
        out_pred_len: int,
        delay: int,
        time_delay_stride: int,
        theta: float,
        dropout: float,
        projection_dim: int,
        layer_norm: bool,
        add_pe: bool,
        activation: nn.Module,
    ):
        super().__init__()
        self.delay = delay
        self.time_delay_stride = time_delay_stride
        unfolded_len = ((lookback_len + out_pred_len) // time_delay_stride) - delay + 1
        self.unfolded_lookback_len = int((lookback_len / (lookback_len + out_pred_len)) * unfolded_len)
        self.unfolded_pred_len = max(1, unfolded_len - self.unfolded_lookback_len)
        self.theta = theta
        self.projection = nn.Sequential(
            nn.Linear(delay, projection_dim),
            nn.Dropout(dropout),
            activation,
        )
        self.pe = LearnablePositionalEmbedding(
            hidden_size=projection_dim,
            max_len=max(1024, lookback_len + out_pred_len),
        ) if add_pe else None
        self.norm = nn.LayerNorm(projection_dim) if layer_norm else nn.Identity()
        self.undelay = nn.Sequential(
            nn.Linear(delay * self.unfolded_pred_len, out_pred_len),
            nn.Dropout(dropout),
            activation,
            nn.Linear(out_pred_len, out_pred_len),
        )

    def forward(self, x, focal_points):
        x = torch.cat([x, focal_points], dim=-1)
        x_td = x.unfold(-1, self.delay, self.time_delay_stride)
        focal = x_td[:, :, -self.unfolded_pred_len - 1 :, :]
        x_td = x_td[:, :, : -self.unfolded_pred_len - 1, :]

        keys = self.projection(x_td[:, :, :-1, :])
        query = self.projection(focal[:, :, :-1, :])
        values = x_td[:, :, 1:, :]

        if self.pe is not None and keys.size(-1) % 2 == 0:
            keys = keys + self.pe(keys)
            query = query + self.pe(query, offset=keys.size(2))

        keys = self.norm(keys)
        query = self.norm(query)

        scale = (1.0 / math.sqrt(keys.size(-1))) * self.theta
        attn_values = nn.functional.scaled_dot_product_attention(
            query,
            keys,
            values,
            dropout_p=0.1 if self.training else 0.0,
            scale=scale,
        )
        pred = attn_values.reshape(attn_values.size(0), attn_values.size(1), -1)
        return self.undelay(pred)


class DeepEDM(BaseModel):
    """DeepEDM.

    Deep empirical dynamic modeling with iterative delay-embedding attention blocks.

    References:
        - [Abrar Majeedi, et al. "DeepEDM: A Deep Learning Approach for Empirical Dynamic Modeling in Time Series Forecasting".](https://arxiv.org/abs/2501.15295)
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
        n_edm_blocks: int = 2,
        mlp_layers: int = 2,
        delay: int = 4,
        time_delay_stride: int = 1,
        theta: float = 1.0,
        projection_dim: int = 64,
        layer_norm: bool = True,
        add_pe: bool = True,
        dropout: float = 0.1,
        activation: str = "gelu",
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 5000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 128,
        inference_windows_batch_size: int = 128,
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

        activation_map = {
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation={activation}")
        activation_layer = activation_map[activation]

        max_valid_delay = max(2, min(delay, max(2, (input_size + h) // 2)))
        self.delay = max_valid_delay
        self.time_delay_stride = time_delay_stride
        self.encoder = InputEncoder(
            mlp_layers=mlp_layers,
            lookback_len=input_size,
            pred_len=h,
            activation=activation_layer,
            dropout=dropout,
        )
        self.edm_blocks = nn.ModuleList(
            [
                EDMBlock(
                    lookback_len=input_size,
                    out_pred_len=h,
                    delay=self.delay,
                    time_delay_stride=time_delay_stride,
                    theta=theta,
                    dropout=dropout,
                    projection_dim=projection_dim,
                    layer_norm=layer_norm,
                    add_pe=add_pe,
                    activation=activation_layer,
                )
                for _ in range(n_edm_blocks)
            ]
        )
        self.gate = nn.Linear(h, 1)

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"].squeeze(-1)
        x = insample_y.unsqueeze(1)

        means = x.mean(dim=-1, keepdim=True).detach()
        x = x - means
        std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / std

        curr_lookback, focal_points, skip = self.encoder(x)
        edm_pred = focal_points
        for edm_block in self.edm_blocks:
            edm_pred = edm_block(curr_lookback, focal_points)
            focal_points = edm_pred

        gate_prob = self.gate(edm_pred).sigmoid()
        pred = edm_pred * gate_prob + skip
        pred = pred * std + means
        return pred.transpose(1, 2)
