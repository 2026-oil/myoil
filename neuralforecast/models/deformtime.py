from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE

__all__ = ["DeformTime"]


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, std=std)
    return nn.init.normal_(tensor, std=std)


def _num_patches(seq_len: int, patch_len: int, stride: int) -> int:
    if seq_len < patch_len:
        raise ValueError("seq_len must be >= patch_len")
    return (seq_len - patch_len) // stride + 1


def _grid_sample_1d(feats: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    feats = feats.unsqueeze(-1)
    zeros = torch.zeros_like(grid)
    grid = torch.stack((grid, zeros), dim=-1).unsqueeze(2)
    out = F.grid_sample(
        feats,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return out.squeeze(-1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class DeformTemporalEmbedding(nn.Module):
    def __init__(self, d_inp: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(d_inp, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class LocalTemporalEmbedding(nn.Module):
    def __init__(
        self,
        d_inp: int,
        d_model: int,
        padding: int,
        sub_groups: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sub_seqlen = d_inp
        self.padding = padding
        self.sub_groups = sub_groups
        self.d_model = d_model
        d_out = math.ceil(d_model / sub_groups)
        self.value_embedding = nn.Linear(d_inp, d_out, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = F.pad(x, (0, self.padding))
        x = x.unfold(dimension=-1, size=self.sub_seqlen, step=self.sub_seqlen)
        # [B, L, G, sub_seqlen]
        num_groups = x.shape[2]
        x = x.permute(0, 2, 1, 3).reshape(bsz * num_groups, seq_len, self.sub_seqlen)
        x = self.value_embedding(x)
        x = x.reshape(bsz, num_groups, seq_len, -1).permute(0, 2, 1, 3)
        x = x.reshape(bsz, seq_len, -1)[:, :, : self.d_model]
        x = x + self.position_embedding(x)
        return self.dropout(x)


class DeformAtten1D(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        kernel: int = 3,
        n_groups: int = 4,
        no_off: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_model % n_groups != 0:
            raise ValueError("d_model must be divisible by n_groups")
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_group_channels = d_model // n_groups
        self.n_heads = n_heads
        self.n_head_channels = d_model // n_heads
        self.scale_factor = d_model ** -0.5

        self.proj_q = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_k = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_v = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj_out = nn.Linear(d_model, d_model)
        pad_size = kernel // 2 if kernel != 1 else 0
        self.proj_offset = nn.Sequential(
            nn.Conv1d(
                self.n_group_channels,
                self.n_group_channels,
                kernel_size=kernel,
                stride=1,
                padding=pad_size,
            ),
            nn.Conv1d(self.n_group_channels, 1, kernel_size=1, bias=False),
        )
        self.relative_position_bias_table = nn.Parameter(torch.zeros(1, 1, seq_len))
        _trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _group(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, seq_len = x.shape
        return x.reshape(bsz, self.n_groups, self.n_group_channels, seq_len).reshape(
            bsz * self.n_groups, self.n_group_channels, seq_len
        )

    def _ungroup(self, x: torch.Tensor, bsz: int) -> torch.Tensor:
        _, _, seq_len = x.shape
        return x.reshape(bsz, self.n_groups, self.n_group_channels, seq_len).reshape(
            bsz, self.d_model, seq_len
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.permute(0, 2, 1)

        q = self.proj_q(x)
        grouped_queries = self._group(q)
        offset = self.proj_offset(grouped_queries).squeeze(1)
        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        grouped_x = self._group(x)
        if self.no_off:
            x_sampled = F.avg_pool1d(grouped_x, kernel_size=1, stride=1)
        else:
            grid = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(0)
            grid = grid.expand(offset.shape[0], -1)
            vgrid = grid + offset
            vgrid_scaled = 2.0 * vgrid / max(seq_len - 1, 1) - 1.0
            x_sampled = _grid_sample_1d(grouped_x, vgrid_scaled)

        if not self.no_off:
            x_sampled = self._ungroup(x_sampled, bsz)

        q = q.reshape(bsz * self.n_heads, self.n_head_channels, seq_len)
        k = self.proj_k(x_sampled).reshape(
            bsz * self.n_heads, self.n_head_channels, seq_len
        )
        v = self.proj_v(x_sampled).reshape(
            bsz * self.n_heads, self.n_head_channels, seq_len
        )
        v = v + self.relative_position_bias_table[..., :seq_len]

        scores = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale_factor
        attention = torch.softmax(scores, dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attention, v)
        out = out.reshape(bsz, self.d_model, seq_len).transpose(1, 2)
        return self.proj_out(out)


class DeformAtten2D(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        kernel: int = 3,
        n_groups: int = 1,
        no_off: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_model % n_groups != 0:
            raise ValueError("d_model must be divisible by n_groups")
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_groups = n_groups
        self.n_group_channels = d_model // n_groups
        self.n_heads = n_heads
        self.n_head_channels = d_model // n_heads
        self.scale_factor = d_model ** -0.5

        self.proj_q = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.proj_k = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.proj_v = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.proj_out = nn.Linear(d_model, d_model)
        pad_size = kernel // 2 if kernel != 1 else 0
        self.proj_offset = nn.Sequential(
            nn.Conv2d(
                self.n_group_channels,
                self.n_group_channels,
                kernel_size=kernel,
                stride=1,
                padding=pad_size,
            ),
            nn.Conv2d(self.n_group_channels, 2, kernel_size=1, bias=False),
        )
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(1, 1, seq_len, 1)
        )
        _trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, height, width, channels = x.shape
        x = x.permute(0, 3, 1, 2)
        q = self.proj_q(x)
        offset = self.proj_offset(q)
        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=1, stride=1)
        else:
            grid_y = torch.arange(height, device=x.device, dtype=x.dtype)
            grid_x = torch.arange(width, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
            grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(bsz, -1, -1, -1)
            vgrid = grid + offset
            vgrid_x = 2.0 * vgrid[:, 0] / max(width - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, 1] / max(height - 1, 1) - 1.0
            vgrid_scaled = torch.stack([vgrid_x, vgrid_y], dim=-1)
            x_sampled = F.grid_sample(
                x,
                vgrid_scaled,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[:, :, :height, :width]

        q = q.reshape(bsz * self.n_heads, height, width)
        k = self.proj_k(x_sampled).reshape(bsz * self.n_heads, height, width)
        v = self.proj_v(x_sampled) + self.relative_position_bias_table
        v = v.reshape(bsz * self.n_heads, height, width)

        scores = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale_factor
        attention = torch.softmax(scores, dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attention, v)
        return self.proj_out(out.reshape(bsz, height, width, channels))


class CrossDeformAttn(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        droprate: float,
        n_days: int = 1,
        window_size: int = 3,
        patch_len: int = 4,
        stride: int = 2,
        no_off: bool = False,
    ) -> None:
        super().__init__()
        self.n_days = n_days
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = _num_patches(seq_len, patch_len, stride)
        self.norm = nn.LayerNorm(d_model)
        self.deform_1d = DeformAtten1D(
            seq_len=(seq_len + n_days - 1) // n_days,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            kernel=window_size,
            no_off=no_off,
        )
        self.deform_2d = DeformAtten2D(
            seq_len=patch_len,
            d_model=1,
            n_heads=1,
            dropout=dropout,
            kernel=window_size,
            n_groups=1,
            no_off=no_off,
        )
        self.write_out = nn.Linear(self.num_patches * patch_len, seq_len)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        bsz, seq_len, d_model = x.shape

        pad_len = (self.n_days - (seq_len % self.n_days)) % self.n_days
        if pad_len > 0:
            pad = x[:, :1, :].expand(-1, pad_len, -1)
            x_pad = torch.cat([x, pad], dim=1)
        else:
            x_pad = x
        seg_len = x_pad.shape[1] // self.n_days
        x_1d = x_pad.reshape(bsz, seg_len, self.n_days, d_model)
        x_1d = x_1d.permute(0, 2, 1, 3).reshape(bsz * self.n_days, seg_len, d_model)
        x_1d = self.deform_1d(x_1d)
        x_1d = x_1d.reshape(bsz, self.n_days, seg_len, d_model)
        x_1d = x_1d.permute(0, 2, 1, 3).reshape(bsz, seg_len * self.n_days, d_model)
        x_1d = x_1d[:, :seq_len, :]

        x_2d = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [B, num_patches, C, patch_len]
        x_2d = x_2d.permute(0, 1, 3, 2).reshape(
            bsz * self.num_patches, self.patch_len, d_model, 1
        )
        x_2d = self.deform_2d(x_2d)
        x_2d = x_2d.reshape(bsz, self.num_patches * self.patch_len, d_model, 1)
        x_2d = x_2d.squeeze(-1)
        x_2d = self.write_out(x_2d.permute(0, 2, 1)).permute(0, 2, 1)

        x = torch.cat([x_1d, x_2d], dim=-1)
        return self.proj(x)


class DeformTime(BaseModel):
    """DeformTime

    CPU-safe, self-contained NeuralForecast port of the official deformable
    attention + GRU forecasting path.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True
    RECURRENT = False

    def __init__(
        self,
        h,
        input_size,
        n_series,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y: bool = False,
        d_model: int = 16,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        n_reshape: int = 4,
        window_size: int = 3,
        patch_len: int = 4,
        stride: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=32,
        inference_windows_batch_size=32,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        trainer_kwargs = dict(trainer_kwargs)
        trainer_kwargs.setdefault("accelerator", "cpu")
        trainer_kwargs.setdefault("devices", 1)
        trainer_kwargs.setdefault("logger", False)
        trainer_kwargs.setdefault("enable_checkpointing", False)
        trainer_kwargs.setdefault("enable_progress_bar", False)

        super().__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
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
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        if n_series <= 0:
            raise ValueError("n_series must be positive")
        if input_size < patch_len:
            raise ValueError("input_size must be >= patch_len")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_series = n_series
        self.seq_len = input_size
        self.pred_len = h
        self.pred_len_out = h * self.loss.outputsize_multiplier
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.use_norm = use_norm

        if n_series == 1:
            self.enc_value_embedding = DeformTemporalEmbedding(
                1, d_model, dropout=dropout
            )
        else:
            sub_groups = 4
            padded_n_series = math.ceil(n_series / sub_groups) * sub_groups
            padding = padded_n_series - n_series
            self.enc_value_embedding = LocalTemporalEmbedding(
                math.ceil(n_series / sub_groups),
                d_model,
                padding,
                sub_groups=sub_groups,
                dropout=dropout,
            )

        self.pre_norm = nn.LayerNorm(d_model)
        n_days = [1] + [n_reshape] * max(e_layers - 1, 0)
        self.encoder = nn.ModuleList(
            [
                CrossDeformAttn(
                    seq_len=input_size,
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    droprate=dropout,
                    n_days=n_days[i],
                    window_size=window_size,
                    patch_len=patch_len,
                    stride=stride,
                )
                for i in range(e_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(
            d_model,
            d_model,
            d_layers,
            batch_first=True,
            dropout=dropout if d_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, self.pred_len_out),
        )
        self.projection = nn.Linear(d_model, n_series)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_enc = x_enc / stdev

        x_enc = self.enc_value_embedding(x_enc)
        x_enc = self.pre_norm(x_enc)
        for layer in self.encoder:
            x_enc = x_enc + layer(x_enc)
        x_enc = self.encoder_norm(x_enc)

        h0 = torch.zeros(
            self.d_layers, x_enc.size(0), self.d_model, device=x_enc.device
        )
        out, _ = self.gru(x_enc, h0)
        out = self.fc(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.projection(out)

        if self.use_norm:
            out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len_out, 1)
            out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len_out, 1)

        return out

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        y_pred = self.forecast(insample_y)
        return y_pred.reshape(insample_y.shape[0], self.h, -1)
