__all__ = ["AAForecast"]

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from plugins.aa_forecast import CriticalSparseAttention, STARFeatureExtractor

from ..common._base_model import BaseModel
from ..common._modules import MLP
from ..losses.pytorch import MAE


class AAForecast(BaseModel):
    """AAForecast

    PyTorch/neuralforecast adaptation of the AA-Forecast architecture:
    STAR decomposition + anomaly/event-aware sparse attention over a GRU encoder
    + stochastic-dropout uncertainty inference at prediction time.
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
        trend_kernel_size: int | None = None,
        lowess_frac: float = 0.6,
        lowess_delta: float = 0.01,
        anomaly_threshold: float = 3.5,
        star_hist_exog_list=None,
        non_star_hist_exog_list=None,
        uncertainty_enabled: bool = False,
        uncertainty_dropout_candidates=None,
        uncertainty_sample_count: int = 5,
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
        self.lowess_frac = lowess_frac
        self.lowess_delta = lowess_delta
        self.anomaly_threshold = anomaly_threshold
        self.exclude_insample_y = exclude_insample_y
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.uncertainty_dropout_candidates = tuple(
            uncertainty_dropout_candidates or ()
        )
        self.uncertainty_sample_count = int(uncertainty_sample_count)
        self._stochastic_inference_enabled = False
        self._stochastic_dropout_p = float(encoder_dropout)

        self.star_hist_exog_list = tuple(star_hist_exog_list or ())
        self.non_star_hist_exog_list = tuple(non_star_hist_exog_list or ())
        self.star_hist_exog_indices, self.non_star_hist_exog_indices = (
            self._resolve_hist_exog_groups()
        )

        feature_size = (
            (0 if exclude_insample_y else 1)
            + self.hist_exog_size
            + 4 * (1 + len(self.star_hist_exog_list))
        )
        self.star = STARFeatureExtractor(
            season_length=season_length,
            lowess_frac=lowess_frac,
            lowess_delta=lowess_delta,
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

    def configure_stochastic_inference(
        self,
        *,
        enabled: bool,
        dropout_p: float | None = None,
    ) -> None:
        self._stochastic_inference_enabled = bool(enabled)
        if dropout_p is not None:
            self._stochastic_dropout_p = float(dropout_p)

    def _apply_stochastic_dropout(self, tensor: torch.Tensor) -> torch.Tensor:
        training = self.training or self._stochastic_inference_enabled
        if not training:
            return tensor
        p = self.encoder_dropout if self.training else self._stochastic_dropout_p
        if p <= 0:
            return tensor
        return F.dropout(tensor, p=p, training=True)

    def _align_horizon(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.h > self.input_size:
            hidden = hidden.permute(0, 2, 1)
            hidden = self.sequence_adapter(hidden)
            hidden = hidden.permute(0, 2, 1)
            return hidden
        return hidden[:, -self.h :]

    def _resolve_hist_exog_groups(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.hist_exog_size == 0:
            if self.star_hist_exog_list or self.non_star_hist_exog_list:
                raise ValueError(
                    "AAForecast received STAR/non-STAR hist exog groups without hist_exog_list"
                )
            return (), ()

        if not self.star_hist_exog_list:
            raise ValueError(
                "AAForecast requires star_hist_exog_list when hist_exog_list is non-empty"
            )

        hist_lookup = {name: idx for idx, name in enumerate(self.hist_exog_list)}
        all_names = self.star_hist_exog_list + self.non_star_hist_exog_list
        if len(set(self.star_hist_exog_list)) != len(self.star_hist_exog_list):
            raise ValueError("AAForecast star_hist_exog_list must not contain duplicates")
        if len(set(self.non_star_hist_exog_list)) != len(self.non_star_hist_exog_list):
            raise ValueError(
                "AAForecast non_star_hist_exog_list must not contain duplicates"
            )
        overlap = sorted(set(self.star_hist_exog_list).intersection(self.non_star_hist_exog_list))
        if overlap:
            raise ValueError(
                "AAForecast hist exog groups must be disjoint: " + ", ".join(overlap)
            )
        unknown = sorted(set(all_names).difference(hist_lookup))
        if unknown:
            raise ValueError(
                "AAForecast hist exog groups contain unknown column(s): "
                + ", ".join(unknown)
            )

        resolved_star = tuple(
            name for name in self.hist_exog_list if name in self.star_hist_exog_list
        )
        resolved_non_star = tuple(
            name for name in self.hist_exog_list if name in self.non_star_hist_exog_list
        )
        if resolved_star != self.star_hist_exog_list:
            raise ValueError(
                "AAForecast star_hist_exog_list must follow hist_exog_list order exactly"
            )
        if resolved_non_star != self.non_star_hist_exog_list:
            raise ValueError(
                "AAForecast non_star_hist_exog_list must follow hist_exog_list order exactly"
            )
        if resolved_star + resolved_non_star != tuple(self.hist_exog_list):
            raise ValueError(
                "AAForecast hist exog groups must cover hist_exog_list exactly"
            )

        return (
            tuple(hist_lookup[name] for name in self.star_hist_exog_list),
            tuple(hist_lookup[name] for name in self.non_star_hist_exog_list),
        )

    @staticmethod
    def _select_hist_exog(hist_exog: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor | None:
        if not indices:
            return None
        index_tensor = torch.as_tensor(indices, device=hist_exog.device, dtype=torch.long)
        return torch.index_select(hist_exog, dim=2, index=index_tensor)

    @staticmethod
    def _reduce_critical_mask(mask: torch.Tensor | None, *, template: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return torch.zeros_like(template, dtype=torch.bool)
        if mask.ndim != 3:
            raise ValueError("AAForecast critical mask must be rank-3")
        if mask.size(-1) == 1:
            return mask.bool()
        return mask.any(dim=2, keepdim=True)

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        hist_exog = windows_batch["hist_exog"]

        target_star = self.star(insample_y)
        star_hist_exog = self._select_hist_exog(hist_exog, self.star_hist_exog_indices)
        star_hist_outputs = self.star(star_hist_exog) if star_hist_exog is not None else None
        encoder_parts = []
        if not self.exclude_insample_y:
            encoder_parts.append(insample_y)
        if self.hist_exog_size > 0:
            encoder_parts.append(hist_exog)
        encoder_parts.extend(
            [
                target_star["trend"],
                target_star["seasonal"],
                target_star["anomalies"],
                target_star["residual"],
            ]
        )
        if star_hist_outputs is not None:
            encoder_parts.extend(
                [
                    star_hist_outputs["trend"],
                    star_hist_outputs["seasonal"],
                    star_hist_outputs["anomalies"],
                    star_hist_outputs["residual"],
                ]
            )
        encoder_input = torch.cat(encoder_parts, dim=2)

        hidden_states, _ = self.encoder(encoder_input)
        target_mask = self._reduce_critical_mask(
            target_star["critical_mask"],
            template=insample_y,
        )
        star_hist_mask = self._reduce_critical_mask(
            None if star_hist_outputs is None else star_hist_outputs["critical_mask"],
            template=insample_y,
        )
        critical_mask = target_mask | star_hist_mask
        attended_states, _ = self.attention(hidden_states, critical_mask)

        hidden_aligned = self._align_horizon(hidden_states)
        attended_aligned = self._align_horizon(attended_states)
        hidden_aligned = self._apply_stochastic_dropout(hidden_aligned)
        attended_aligned = self._apply_stochastic_dropout(attended_aligned)
        decoder_input = torch.cat([hidden_aligned, attended_aligned], dim=-1)
        decoder_input = self._apply_stochastic_dropout(decoder_input)
        return self.decoder(decoder_input)[:, -self.h :]
