__all__ = ["SeriesMambaMixer", "SMamba"]


from typing import Optional

import torch.nn as nn

from ..common._base_model import BaseModel
from ..common._modules import RevINMultivariate
from ..losses.pytorch import MAE
from .mamba import MambaBlock


class SeriesMambaMixer(nn.Module):
    def __init__(self, n_series, hidden_size, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.series_proj = nn.Linear(n_series, n_series)
        self.hidden_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.series_proj(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.hidden_proj(x)
        return residual + self.dropout(x)


class SMamba(BaseModel):
    """SMamba

    Spatial Mamba variant for multivariate forecasting with shared temporal
    Mamba blocks followed by explicit cross-series mixing.
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
        exclude_insample_y=False,
        hidden_size=64,
        n_block=2,
        expand_ratio=2,
        kernel_size=3,
        dropout=0.0,
        revin=True,
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
        super(SMamba, self).__init__(
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
        self.revin = revin
        if self.revin:
            self.norm = RevINMultivariate(num_features=n_series, affine=True)

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
        self.series_mixer = SeriesMambaMixer(
            n_series=n_series, hidden_size=hidden_size, dropout=dropout
        )
        self.temporal_projection = nn.Linear(input_size, h)
        self.output_projection = nn.Linear(hidden_size, self.loss.outputsize_multiplier)

    def forward(self, windows_batch):
        x = windows_batch["insample_y"]
        batch_size, seq_len, n_series = x.shape

        if self.revin:
            x = self.norm(x, "norm")

        x = x.permute(0, 2, 1).reshape(batch_size * n_series, seq_len, 1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(batch_size, n_series, seq_len, -1).permute(0, 2, 1, 3)
        x = self.series_mixer(x)
        x = x.permute(0, 2, 3, 1).reshape(batch_size * n_series, -1, seq_len)
        x = self.temporal_projection(x).reshape(batch_size, n_series, -1, self.h)
        x = x.permute(0, 3, 1, 2)
        x = self.output_projection(x)
        x = x.reshape(batch_size, self.h, self.n_series * self.loss.outputsize_multiplier)

        if self.revin and self.loss.outputsize_multiplier == 1:
            x = self.norm(x.reshape(batch_size, self.h, self.n_series), "denorm")

        return x
