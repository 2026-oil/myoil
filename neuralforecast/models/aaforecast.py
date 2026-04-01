__all__ = ["AAForecast"]

from typing import Optional

import torch
import torch.nn as nn

from plugins.aa_forecast import CriticalSparseAttention, STARFeatureExtractor

from ..common._base_model import BaseModel
from ..common._modules import MLP
from ..losses.pytorch import MAE


class AAForecast(BaseModel):
    """AAForecast

    PyTorch/neuralforecast-style adaptation of the AA-Forecast paper:
    STAR-style decomposition + anomaly/event-aware sparse attention over a GRU
    encoder + dense multi-horizon forecast head.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        inference_input_size: Optional[int] = None,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 128,
        encoder_dropout: float = 0.1,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        attention_hidden_size: Optional[int] = None,
        season_length: int = 12,
        trend_kernel_size: int = 5,
        anomaly_threshold: float = 3.5,
        hist_exog_list=None,
        futr_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y: bool = False,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 128,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "robust",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            inference_input_size=inference_input_size,
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

        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_n_layers = encoder_n_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.season_length = season_length
        self.trend_kernel_size = trend_kernel_size
        self.anomaly_threshold = anomaly_threshold
        self.exclude_insample_y = exclude_insample_y

        feature_size = (0 if exclude_insample_y else 1) + self.hist_exog_size + 4
        self.star = STARFeatureExtractor(
            season_length=season_length,
            trend_kernel_size=trend_kernel_size,
            anomaly_threshold=anomaly_threshold,
        )
        self.encoder = nn.GRU(
            input_size=feature_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_n_layers,
            batch_first=True,
            dropout=encoder_dropout if encoder_n_layers > 1 else 0.0,
        )
        self.attention = CriticalSparseAttention(
            hidden_size=encoder_hidden_size,
            attention_hidden_size=attention_hidden_size,
        )
        self.sequence_adapter = (
            nn.Linear(self.input_size, self.h) if self.h > self.input_size else None
        )
        self.decoder = MLP(
            in_features=2 * encoder_hidden_size,
            out_features=self.loss.outputsize_multiplier,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            activation="ReLU",
            dropout=encoder_dropout,
        )

    def _align_horizon(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.h > self.input_size:
            hidden = hidden.permute(0, 2, 1)
            hidden = self.sequence_adapter(hidden)
            hidden = hidden.permute(0, 2, 1)
            return hidden
        return hidden[:, -self.h :]

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        hist_exog = windows_batch["hist_exog"]

        star = self.star(insample_y)
        encoder_parts = []
        if not self.exclude_insample_y:
            encoder_parts.append(insample_y)
        if self.hist_exog_size > 0:
            encoder_parts.append(hist_exog)
            event_signal = hist_exog[:, :, :1]
        else:
            event_signal = torch.zeros_like(insample_y)
        encoder_parts.extend(
            [
                star["trend"],
                star["seasonal"],
                star["anomalies"],
                star["residual"],
            ]
        )
        encoder_input = torch.cat(encoder_parts, dim=2)

        hidden_states, _ = self.encoder(encoder_input)
        critical_mask = star["critical_mask"] | (event_signal.abs() > 0)
        attended_states, _ = self.attention(hidden_states, critical_mask)

        hidden_aligned = self._align_horizon(hidden_states)
        attended_aligned = self._align_horizon(attended_states)
        decoder_input = torch.cat([hidden_aligned, attended_aligned], dim=-1)
        output = self.decoder(decoder_input)
        return output[:, -self.h :]
