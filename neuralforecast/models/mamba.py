__all__ = ["MambaBlock", "Mamba"]


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class MambaBlock(nn.Module):
    def __init__(self, hidden_size, expand_ratio=2, kernel_size=3, dropout=0.0):
        super().__init__()
        inner_size = hidden_size * expand_ratio
        self.norm = nn.LayerNorm(hidden_size)
        self.in_proj = nn.Linear(hidden_size, inner_size * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_size,
            inner_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=inner_size,
        )
        self.out_proj = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = value.transpose(1, 2)
        value = self.depthwise_conv(value)[..., : x.size(1)].transpose(1, 2)
        value = F.silu(value)
        x = value * torch.sigmoid(gate)
        x = self.out_proj(self.dropout(x))
        return residual + x


class Mamba(BaseModel):
    """Mamba

    Lightweight repo-native Mamba-style forecaster that mixes local state-space
    dynamics with gated residual updates over the temporal dimension.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h,
        input_size,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y=False,
        hidden_size=64,
        n_block=2,
        expand_ratio=2,
        kernel_size=3,
        dropout=0.0,
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
        super(Mamba, self).__init__(
            h=h,
            input_size=input_size,
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

        self.input_proj = nn.Linear(1, hidden_size)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    hidden_size=hidden_size,
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(n_block)
            ]
        )
        self.temporal_projection = nn.Linear(input_size, h)
        self.output_projection = nn.Linear(hidden_size, self.loss.outputsize_multiplier)

    def forward(self, windows_batch):
        x = windows_batch["insample_y"]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.temporal_projection(x.transpose(1, 2)).transpose(1, 2)
        return self.output_projection(x)
