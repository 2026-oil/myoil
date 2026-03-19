__all__ = ['DUETBlock', 'DUET']


from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..common._modules import RevINMultivariate, SeriesDecomp
from ..losses.pytorch import MAE
from .tsmixer import MixingLayer


class DUETBlock(nn.Module):
    """Dual-branch update block with temporal mixing and inter-series attention."""

    def __init__(
        self,
        n_series: int,
        input_size: int,
        hidden_size: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.temporal_mixer = MixingLayer(
            n_series=n_series,
            input_size=input_size,
            dropout=dropout,
            ff_dim=ff_dim,
        )
        self.series_projection = nn.Linear(input_size, hidden_size)
        self.series_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.series_back_projection = nn.Linear(hidden_size, input_size)
        self.gate = nn.Sequential(
            nn.Linear(2 * n_series, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, n_series),
            nn.Sigmoid(),
        )

    def forward(self, x):
        temporal_state = self.temporal_mixer(x)
        series_tokens = x.permute(0, 2, 1)
        series_tokens = self.series_projection(series_tokens)
        attended_state, _ = self.series_attention(
            series_tokens, series_tokens, series_tokens
        )
        attended_state = self.series_back_projection(attended_state).permute(0, 2, 1)
        fusion_gate = self.gate(torch.cat([temporal_state, attended_state], dim=-1))
        return fusion_gate * temporal_state + (1.0 - fusion_gate) * attended_state


class DUET(BaseModel):
    """DUET

    DUET is a multivariate forecaster that decomposes the input trajectory into
    seasonal and trend components, then updates each branch with temporal mixing
    plus lightweight inter-series attention before recombining the outputs.
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
        n_block: int = 2,
        hidden_size: int = 64,
        ff_dim: int = 128,
        moving_avg_window: int = 5,
        dropout: float = 0.1,
        revin: bool = True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
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
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super(DUET, self).__init__(
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

        self.revin = revin
        if self.revin:
            self.norm = RevINMultivariate(num_features=n_series, affine=True)

        self.decomposition = SeriesDecomp(moving_avg_window)
        self.seasonal_blocks = nn.Sequential(
            *[
                DUETBlock(
                    n_series=n_series,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_block)
            ]
        )
        self.trend_blocks = nn.Sequential(
            *[
                DUETBlock(
                    n_series=n_series,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_block)
            ]
        )
        self.output = nn.Linear(
            input_size, h * self.loss.outputsize_multiplier
        )

    def forward(self, windows_batch):
        x = windows_batch["insample_y"]
        batch_size = x.shape[0]

        if self.revin:
            x = self.norm(x, "norm")

        seasonal_state, trend_state = self.decomposition(x)
        seasonal_state = self.seasonal_blocks(seasonal_state)
        trend_state = self.trend_blocks(trend_state)
        x = seasonal_state + trend_state
        x = x.permute(0, 2, 1)
        x = self.output(x)
        x = x.permute(0, 2, 1)

        if self.revin:
            x = self.norm(x, "denorm")

        x = x.reshape(
            batch_size, self.h, self.loss.outputsize_multiplier * self.n_series
        )
        return x
