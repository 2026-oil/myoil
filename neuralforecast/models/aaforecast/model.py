from __future__ import annotations

__all__ = ["AAForecast"]

import os
from typing import Optional

import torch
import torch.nn as nn

from plugins.aa_forecast import (
    CriticalSparseAttention,
    ITransformerTokenSparseAttention,
    STARFeatureExtractor,
    TimeXerTokenSparseAttention,
)

from ...common._base_model import BaseModel
from ...common._modules import MLP
from ...losses.pytorch import MAE
from .backbones import AA_SUPPORTED_BACKBONES, build_aaforecast_backbone
from .gru import _align_horizon, _apply_stochastic_dropout
from .models.base import AATimeXerTokenStates
from ..timexer import FlattenHead


class InformerHorizonAwareHead(nn.Module):
    """Informer-only horizon-aware decoder head with event gating.

    Keeps the decode specialization local to the Informer path while letting the
    shared event summary influence each horizon differently.
    """

    def __init__(
        self,
        *,
        h: int,
        in_features: int,
        event_features: int,
        hidden_size: int,
        out_features: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.h = int(h)
        self.base_in_features = int(in_features)
        self.event_features = int(event_features)
        self.hidden_size = int(hidden_size)
        self.horizon_embeddings = nn.Embedding(
            num_embeddings=self.h,
            embedding_dim=self.hidden_size,
        )
        self.event_gate = nn.Sequential(
            nn.Linear(self.event_features + self.hidden_size, self.event_features),
            nn.Sigmoid(),
        )
        self.shared_trunk = MLP(
            in_features=(
                self.base_in_features + self.hidden_size + (2 * self.event_features)
            ),
            out_features=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            activation="ReLU",
            dropout=dropout,
        )
        head_num_layers = max(1, int(num_layers) - 1)
        self.horizon_heads = nn.ModuleList(
            [
                MLP(
                    in_features=hidden_size,
                    out_features=out_features,
                    hidden_size=hidden_size,
                    num_layers=head_num_layers,
                    activation="ReLU",
                    dropout=dropout,
                )
                for _ in range(self.h)
            ]
        )

    def build_horizon_context(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        horizon_ids = torch.arange(self.h, device=device)
        horizon_context = self.horizon_embeddings(horizon_ids)
        return horizon_context.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

    def forward(
        self,
        decoder_input: torch.Tensor,
        event_summary: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_input.ndim != 3:
            raise ValueError(
                "InformerHorizonAwareHead decoder_input must be rank-3 [B, h, features]"
            )
        if decoder_input.shape[1] != self.h:
            raise ValueError(
                f"InformerHorizonAwareHead expected horizon dimension {self.h}, "
                f"got {decoder_input.shape[1]}"
            )
        if event_summary.ndim != 2:
            raise ValueError(
                "InformerHorizonAwareHead event_summary must be rank-2 [B, features]"
            )
        if event_summary.shape[0] != decoder_input.shape[0]:
            raise ValueError(
                "InformerHorizonAwareHead event_summary batch must match decoder_input batch"
            )
        if event_summary.shape[1] != self.event_features:
            raise ValueError(
                "InformerHorizonAwareHead event_summary width must match event_features"
            )
        horizon_context = self.build_horizon_context(
            batch_size=decoder_input.shape[0],
            device=decoder_input.device,
            dtype=decoder_input.dtype,
        )
        repeated_event = event_summary.unsqueeze(1).expand(-1, self.h, -1).to(
            dtype=decoder_input.dtype
        )
        event_gate = self.event_gate(torch.cat([repeated_event, horizon_context], dim=-1))
        conditioned = torch.cat(
            [
                decoder_input,
                horizon_context,
                repeated_event,
                repeated_event * event_gate,
            ],
            dim=-1,
        )
        trunk_features = self.shared_trunk(conditioned)
        return torch.cat(
            [
                head(trunk_features[:, horizon_idx : horizon_idx + 1, :])
                for horizon_idx, head in enumerate(self.horizon_heads)
            ],
            dim=1,
        )


class AAForecast(BaseModel):
    """AAForecast

    PyTorch/neuralforecast adaptation of the AA-Forecast architecture:
    STAR decomposition + anomaly/event-aware sparse attention over a selectable
    sequence backbone + stochastic-dropout uncertainty inference at prediction time.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = False
    MULTIVARIATE = False
    RECURRENT = False
    EVENT_SUMMARY_SIZE = 9

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        inference_input_size: Optional[int] = None,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 128,
        encoder_dropout: float = 0.1,
        backbone: str = "gru",
        hidden_size: int = 128,
        n_head: int = 4,
        n_heads: int = 4,
        encoder_layers: int = 2,
        dropout: float = 0.1,
        linear_hidden_size: Optional[int] = None,
        factor: int = 3,
        attn_dropout: float = 0.0,
        patch_len: int = 4,
        stride: int = 2,
        e_layers: int = 2,
        d_ff: int = 256,
        use_norm: bool = True,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        attention_hidden_size: Optional[int] = None,
        season_length: int = 12,
        trend_kernel_size: int | None = None,
        lowess_frac: float = 0.6,
        lowess_delta: float = 0.01,
        thresh: float = 3.5,
        star_hist_exog_list=None,
        non_star_hist_exog_list=None,
        star_hist_exog_tail_modes=None,
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

        self.backbone = str(backbone).strip().lower()
        if self.backbone not in AA_SUPPORTED_BACKBONES:
            supported = ", ".join(sorted(AA_SUPPORTED_BACKBONES))
            raise ValueError(f"AAForecast backbone must be one of: {supported}")
        self.encoder_hidden_size = (
            encoder_hidden_size if self.backbone == "gru" else hidden_size
        )
        self.encoder_n_layers = encoder_n_layers
        self.encoder_dropout = (
            encoder_dropout if self.backbone == "gru" else dropout
        )
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_heads = n_heads
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.linear_hidden_size = linear_hidden_size
        self.factor = factor
        self.attn_dropout = attn_dropout
        self.patch_len = patch_len
        self.stride = stride
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.use_norm = use_norm
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.season_length = season_length
        self.trend_kernel_size = trend_kernel_size
        self.lowess_frac = lowess_frac
        self.lowess_delta = lowess_delta
        self.thresh = float(thresh)
        self.exclude_insample_y = exclude_insample_y
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.uncertainty_dropout_candidates = tuple(
            uncertainty_dropout_candidates or ()
        )
        self.uncertainty_sample_count = int(uncertainty_sample_count)
        self._stochastic_inference_enabled = False
        self._stochastic_dropout_p = float(self.encoder_dropout)
        self._star_precompute_enabled = True
        self._star_precompute_fold_key: str | None = None
        self._star_phase_cache: dict[str, dict[str, object]] = {}

        self.star_hist_exog_list = tuple(star_hist_exog_list or ())
        self.non_star_hist_exog_list = tuple(non_star_hist_exog_list or ())
        self.star_hist_exog_tail_modes = tuple(star_hist_exog_tail_modes or ())
        self.star_hist_exog_indices, self.non_star_hist_exog_indices = (
            self._resolve_hist_exog_groups()
        )
        self.star_hist_exog_tail_modes = self._resolve_star_hist_exog_tail_modes()
        self.target_token_indices = self._resolve_target_token_indices()

        feature_size = (
            (0 if exclude_insample_y else 1)
            + len(self.non_star_hist_exog_list)
            + 4
            + 4 * len(self.star_hist_exog_list)
        )
        self.star = STARFeatureExtractor(
            season_length=season_length,
            lowess_frac=lowess_frac,
            lowess_delta=lowess_delta,
            thresh=thresh,
        )
        self.encoder = build_aaforecast_backbone(
            self.backbone,
            feature_size=feature_size,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            encoder_hidden_size=encoder_hidden_size,
            encoder_n_layers=encoder_n_layers,
            encoder_dropout=encoder_dropout,
            n_head=n_head,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            dropout=dropout,
            linear_hidden_size=linear_hidden_size,
            factor=factor,
            attn_dropout=attn_dropout,
            patch_len=patch_len,
            stride=stride,
            e_layers=e_layers,
            d_ff=d_ff,
            use_norm=use_norm,
        )
        if self.backbone == "timexer":
            self.attention = TimeXerTokenSparseAttention(
                hidden_size=self.encoder_hidden_size,
                attention_hidden_size=attention_hidden_size,
            )
        elif self.backbone == "itransformer":
            self.attention = ITransformerTokenSparseAttention(
                hidden_size=self.encoder_hidden_size,
                attention_hidden_size=attention_hidden_size,
            )
        else:
            self.attention = CriticalSparseAttention(
                hidden_size=self.encoder_hidden_size,
                attention_hidden_size=attention_hidden_size,
            )
        self.sequence_adapter = (
            nn.Linear(self.input_size, self.h) if self.h > self.input_size else None
        )
        if self.backbone == "timexer":
            self.timexer_decoder = FlattenHead(
                feature_size,
                (self.encoder.patch_num + 1) * (2 * self.encoder_hidden_size),
                self.h * self.loss.outputsize_multiplier,
                head_dropout=self.encoder_dropout,
            )
            self.decoder = None
            self.itransformer_decoder = None
            self.informer_decoder = None
            self.event_summary_projector = None
        elif self.backbone == "itransformer":
            self.itransformer_decoder = nn.Linear(
                2 * self.encoder_hidden_size,
                self.h * self.loss.outputsize_multiplier,
            )
            self.decoder = None
            self.timexer_decoder = None
            self.informer_decoder = None
            self.event_summary_projector = None
        elif self.backbone == "informer":
            self.event_summary_projector = MLP(
                in_features=self.EVENT_SUMMARY_SIZE,
                out_features=self.encoder_hidden_size,
                hidden_size=max(self.encoder_hidden_size, decoder_hidden_size),
                num_layers=2,
                activation="ReLU",
                dropout=self.encoder_dropout,
            )
            self.informer_decoder = InformerHorizonAwareHead(
                h=self.h,
                in_features=2 * self.encoder_hidden_size,
                event_features=self.encoder_hidden_size,
                hidden_size=decoder_hidden_size,
                out_features=self.loss.outputsize_multiplier,
                num_layers=decoder_layers,
                dropout=self.encoder_dropout,
            )
            self.decoder = None
            self.timexer_decoder = None
            self.itransformer_decoder = None
        else:
            self.decoder = MLP(
                in_features=2 * self.encoder_hidden_size,
                out_features=self.loss.outputsize_multiplier,
                hidden_size=decoder_hidden_size,
                num_layers=decoder_layers,
                activation="ReLU",
                dropout=self.encoder_dropout,
            )
            self.timexer_decoder = None
            self.itransformer_decoder = None
            self.informer_decoder = None
            self.event_summary_projector = None

    def configure_stochastic_inference(
        self,
        *,
        enabled: bool,
        dropout_p: float | None = None,
    ) -> None:
        self._stochastic_inference_enabled = bool(enabled)
        if dropout_p is not None:
            self._stochastic_dropout_p = float(dropout_p)

    def set_star_precompute_context(
        self,
        *,
        enabled: bool = True,
        fold_key: str | None = None,
    ) -> None:
        self._star_precompute_enabled = bool(enabled) and (
            os.environ.get("NEURALFORECAST_AA_STAR_PRECOMPUTE", "1") != "0"
        )
        if fold_key != self._star_precompute_fold_key:
            self._star_phase_cache.clear()
        self._star_precompute_fold_key = fold_key

    def _compute_star_outputs(
        self,
        insample_y: torch.Tensor,
        hist_exog: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        target_star = self.star(insample_y, tail_modes=("two_sided",))
        star_hist_exog = self._select_hist_exog(hist_exog, self.star_hist_exog_indices)
        star_hist_outputs = (
            self.star(
                star_hist_exog,
                tail_modes=self.star_hist_exog_tail_modes,
            )
            if star_hist_exog is not None
            else None
        )
        target_count = self._count_active_channels(
            target_star["critical_mask"],
            template=insample_y,
        )
        star_hist_count = self._count_active_channels(
            None if star_hist_outputs is None else star_hist_outputs["critical_mask"],
            template=insample_y,
        )
        combined_count = target_count + star_hist_count
        target_activity = target_star["ranking_score"] * target_star["critical_mask"].to(
            dtype=insample_y.dtype
        )
        star_hist_activity = (
            star_hist_outputs["ranking_score"]
            * star_hist_outputs["critical_mask"].to(dtype=insample_y.dtype)
            if star_hist_outputs is not None
            else insample_y.new_empty((insample_y.size(0), insample_y.size(1), 0))
        )
        target_signed_score = target_star["robust_score_signed"]
        star_hist_signed_score = (
            star_hist_outputs["robust_score_signed"]
            if star_hist_outputs is not None
            else insample_y.new_empty((insample_y.size(0), insample_y.size(1), 0))
        )
        event_summary = self._build_event_summary_from_payload(
            {
                "critical_mask": combined_count > 0,
                "count_active_channels": combined_count,
                "channel_activity": torch.cat([target_activity, star_hist_activity], dim=2),
                "target_activity": target_activity,
                "star_hist_activity": star_hist_activity,
                "target_signed_score": target_signed_score,
                "star_hist_signed_score": star_hist_signed_score,
            },
            dtype=insample_y.dtype,
            device=insample_y.device,
        )
        return {
            "target_trend": target_star["trend"],
            "target_seasonal": target_star["seasonal"],
            "target_anomalies": target_star["anomalies"],
            "target_residual": target_star["residual"],
            "target_activity": target_activity,
            "target_signed_score": target_signed_score,
            "star_hist_trend": (
                star_hist_outputs["trend"]
                if star_hist_outputs is not None
                else insample_y.new_empty(
                    (insample_y.size(0), insample_y.size(1), 0)
                )
            ),
            "star_hist_seasonal": (
                star_hist_outputs["seasonal"]
                if star_hist_outputs is not None
                else insample_y.new_empty(
                    (insample_y.size(0), insample_y.size(1), 0)
                )
            ),
            "star_hist_anomalies": (
                star_hist_outputs["anomalies"]
                if star_hist_outputs is not None
                else insample_y.new_empty(
                    (insample_y.size(0), insample_y.size(1), 0)
                )
            ),
            "star_hist_residual": (
                star_hist_outputs["residual"]
                if star_hist_outputs is not None
                else insample_y.new_empty(
                    (insample_y.size(0), insample_y.size(1), 0)
                )
            ),
            "critical_mask": combined_count > 0,
            "count_active_channels": combined_count,
            "star_hist_activity": star_hist_activity,
            "star_hist_signed_score": star_hist_signed_score,
            "channel_activity": torch.cat([target_activity, star_hist_activity], dim=2),
            "event_summary": event_summary,
        }

    def _build_star_phase_cache(
        self,
        *,
        batch,
        phase: str,
    ) -> dict[str, object]:
        if phase == "predict":
            return {"window_ids": torch.empty(0, dtype=torch.long), "payload": {}}
        scaler_state = {}
        had_shift = hasattr(self.scaler, "x_shift")
        had_scale = hasattr(self.scaler, "x_scale")
        if had_shift:
            scaler_state["x_shift"] = self.scaler.x_shift.detach().clone()
        if had_scale:
            scaler_state["x_scale"] = self.scaler.x_scale.detach().clone()
        windows_temporal, static, static_cols, final_condition = self._create_windows(
            batch, step=phase
        )
        if len(final_condition) == 0:
            return {"window_ids": torch.empty(0, dtype=torch.long), "payload": {}}
        temporal_cols = batch["temporal_cols"]
        w_idxs = torch.arange(len(final_condition), device=windows_temporal.device)
        windows = self._sample_windows(
            windows_temporal=windows_temporal,
            static=static,
            static_cols=static_cols,
            temporal_cols=temporal_cols,
            w_idxs=w_idxs,
            final_condition=final_condition,
        )
        try:
            windows = self._normalization(windows=windows, y_idx=batch["y_idx"])
            (
                insample_y,
                _insample_mask,
                _outsample_y,
                _outsample_mask,
                hist_exog,
                _futr_exog,
                _stat_exog,
            ) = self._parse_windows(batch, windows)
            payload = self._compute_star_outputs(insample_y, hist_exog)
            cached_payload = {
                name: value.detach().cpu()
                for name, value in payload.items()
            }
            return {
                "window_ids": windows["window_ids"].detach().cpu(),
                "id_to_pos": {
                    int(window_id): pos
                    for pos, window_id in enumerate(
                        windows["window_ids"].detach().cpu().tolist()
                    )
                },
                "payload": cached_payload,
            }
        finally:
            if had_shift:
                self.scaler.x_shift = scaler_state["x_shift"]
            elif hasattr(self.scaler, "x_shift"):
                delattr(self.scaler, "x_shift")
            if had_scale:
                self.scaler.x_scale = scaler_state["x_scale"]
            elif hasattr(self.scaler, "x_scale"):
                delattr(self.scaler, "x_scale")

    def get_star_precomputed(
        self,
        *,
        batch,
        phase: str,
        window_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor] | None:
        if not self._star_precompute_enabled or phase == "predict":
            return None
        cache = self._star_phase_cache.get(phase)
        if cache is None:
            cache = self._build_star_phase_cache(batch=batch, phase=phase)
            self._star_phase_cache[phase] = cache
        payload = cache["payload"]
        if not payload:
            return None
        positions = [cache["id_to_pos"][int(window_id)] for window_id in window_ids.detach().cpu().tolist()]
        result: dict[str, torch.Tensor] = {}
        for name, value in payload.items():
            selected = value[positions]
            result[name] = selected.to(device=device, dtype=dtype)
        return result

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

    def _resolve_star_hist_exog_tail_modes(self) -> tuple[str, ...]:
        if not self.star_hist_exog_list:
            if self.star_hist_exog_tail_modes:
                raise ValueError(
                    "AAForecast received star_hist_exog_tail_modes without STAR hist exog"
                )
            return ()
        if len(self.star_hist_exog_tail_modes) != len(self.star_hist_exog_list):
            raise ValueError(
                "AAForecast star_hist_exog_tail_modes must align with star_hist_exog_list"
            )
        invalid = sorted(
            set(self.star_hist_exog_tail_modes).difference({"two_sided", "upward"})
        )
        if invalid:
            raise ValueError(
                "AAForecast star_hist_exog_tail_modes contain unsupported value(s): "
                + ", ".join(invalid)
            )
        return self.star_hist_exog_tail_modes

    def _resolve_target_token_indices(self) -> tuple[int, ...]:
        target_indices: list[int] = []
        if not self.exclude_insample_y:
            target_indices.append(0)
        target_star_offset = (0 if self.exclude_insample_y else 1) + len(
            self.non_star_hist_exog_list
        )
        target_indices.extend(range(target_star_offset, target_star_offset + 4))
        return tuple(target_indices)

    @staticmethod
    def _select_hist_exog(hist_exog: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor | None:
        if not indices:
            return None
        index_tensor = torch.as_tensor(indices, device=hist_exog.device, dtype=torch.long)
        return torch.index_select(hist_exog, dim=2, index=index_tensor)

    @staticmethod
    def _count_active_channels(
        mask: torch.Tensor | None,
        *,
        template: torch.Tensor,
    ) -> torch.Tensor:
        if mask is None:
            return torch.zeros_like(template)
        if mask.ndim != 3:
            raise ValueError("AAForecast critical mask must be rank-3")
        return mask.bool().to(dtype=template.dtype).sum(dim=2, keepdim=True)

    @staticmethod
    def _reduce_critical_mask(mask: torch.Tensor | None, *, template: torch.Tensor) -> torch.Tensor:
        return AAForecast._count_active_channels(mask, template=template) > 0

    @staticmethod
    def _mean_feature(
        values: torch.Tensor,
        *,
        keepdim: bool = True,
    ) -> torch.Tensor:
        if values.numel() == 0 or values.shape[-1] == 0:
            shape = (values.shape[0], 1) if keepdim else (values.shape[0],)
            return values.new_zeros(shape)
        return values.mean(dim=(1, 2), keepdim=keepdim)

    @staticmethod
    def _weighted_mean_feature(
        values: torch.Tensor,
        *,
        weights: torch.Tensor,
        keepdim: bool = True,
    ) -> torch.Tensor:
        if values.numel() == 0 or values.shape[-1] == 0:
            shape = (values.shape[0], 1) if keepdim else (values.shape[0],)
            return values.new_zeros(shape)
        denom = weights.sum().clamp_min(1e-6) * values.shape[-1]
        reduced = (values * weights).sum(dim=(1, 2), keepdim=keepdim) / denom
        return reduced

    def _build_event_summary_from_payload(
        self,
        payload: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        critical_mask = payload["critical_mask"].to(device=device, dtype=dtype)
        count_active_channels = payload["count_active_channels"].to(
            device=device,
            dtype=dtype,
        )
        channel_activity = payload["channel_activity"].to(device=device, dtype=dtype).clamp_min(0.0)
        seq_len = critical_mask.shape[1]
        time_weights = torch.linspace(
            0.25,
            1.0,
            steps=seq_len,
            device=device,
            dtype=dtype,
        ).view(1, seq_len, 1)

        target_activity = payload.get("target_activity")
        if target_activity is None:
            target_activity = channel_activity[:, :, :1]
        else:
            target_activity = target_activity.to(device=device, dtype=dtype).clamp_min(0.0)

        star_hist_activity = payload.get("star_hist_activity")
        if star_hist_activity is None:
            star_hist_activity = channel_activity[:, :, 1:]
        else:
            star_hist_activity = star_hist_activity.to(device=device, dtype=dtype).clamp_min(0.0)

        target_signed_score = payload.get("target_signed_score")
        if target_signed_score is None:
            target_positive = target_activity
        else:
            target_positive = target_signed_score.to(device=device, dtype=dtype).clamp_min(0.0)

        star_hist_signed_score = payload.get("star_hist_signed_score")
        if star_hist_signed_score is None:
            hist_positive = star_hist_activity
        else:
            hist_positive = star_hist_signed_score.to(device=device, dtype=dtype).clamp_min(0.0)

        density = critical_mask.to(dtype=dtype).mean(dim=1)
        recent_density = (
            critical_mask.to(dtype=dtype) * time_weights
        ).sum(dim=1) / time_weights.sum().clamp_min(1e-6)
        mean_count = torch.log1p(count_active_channels.mean(dim=1))
        recent_activity = torch.log1p(
            self._weighted_mean_feature(
                channel_activity,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        target_up_mass = torch.log1p(
            self._mean_feature(target_positive, keepdim=False).unsqueeze(-1)
        )
        target_up_recent = torch.log1p(
            self._weighted_mean_feature(
                target_positive,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        hist_up_mass = torch.log1p(
            self._mean_feature(hist_positive, keepdim=False).unsqueeze(-1)
        )
        hist_up_recent = torch.log1p(
            self._weighted_mean_feature(
                hist_positive,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        peak_activity = torch.log1p(
            channel_activity.amax(dim=(1, 2), keepdim=False).unsqueeze(-1)
        )
        summary = torch.cat(
            [
                density,
                recent_density,
                mean_count,
                recent_activity,
                target_up_mass,
                target_up_recent,
                hist_up_mass,
                hist_up_recent,
                peak_activity,
            ],
            dim=1,
        )
        summary = torch.nan_to_num(summary, nan=0.0, posinf=10.0, neginf=-10.0)
        return summary.clamp(min=-10.0, max=10.0)

    def _reduce_time_signal_to_timexer_patches(
        self,
        signal: torch.Tensor,
        *,
        reduce: str,
    ) -> torch.Tensor:
        if self.backbone != "timexer":
            raise ValueError("TimeXer patch reduction is only available for the timexer backbone")
        if signal.ndim != 3:
            raise ValueError("TimeXer patch reduction expects rank-3 [B, time, channel]")
        if signal.shape[1] != self.input_size:
            raise ValueError(
                "TimeXer patch reduction requires signal length to match input_size; "
                f"got signal_len={signal.shape[1]}, input_size={self.input_size}"
            )
        reshaped = signal.reshape(
            signal.shape[0],
            self.encoder.patch_num,
            self.patch_len,
            signal.shape[2],
        )
        if reduce == "any":
            return reshaped.bool().any(dim=2)
        if reduce == "sum":
            return reshaped.sum(dim=2)
        raise ValueError(f"Unsupported TimeXer patch reduction mode: {reduce}")

    def _aggregate_timexer_attention_signals(
        self,
        *,
        critical_mask: torch.Tensor,
        count_active_channels: torch.Tensor,
        channel_activity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        time_activity = channel_activity.sum(dim=2, keepdim=True)
        patch_mask = self._reduce_time_signal_to_timexer_patches(
            critical_mask.to(dtype=torch.bool),
            reduce="any",
        ).to(dtype=torch.bool)
        patch_count = self._reduce_time_signal_to_timexer_patches(
            count_active_channels,
            reduce="sum",
        )
        patch_activity = self._reduce_time_signal_to_timexer_patches(
            time_activity,
            reduce="sum",
        )
        global_mask = critical_mask.bool().any(dim=1, keepdim=True)
        global_count = count_active_channels.sum(dim=1, keepdim=True)
        global_activity = time_activity.sum(dim=1, keepdim=True)
        return {
            "patch_mask": patch_mask,
            "patch_count": patch_count,
            "patch_activity": patch_activity,
            "global_mask": global_mask,
            "global_count": global_count,
            "global_activity": global_activity,
        }

    def _aggregate_itransformer_attention_signals(
        self,
        *,
        star_payload: dict[str, torch.Tensor],
        template: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero_non_star = template.new_zeros(
            template.shape[0],
            template.shape[1],
            len(self.non_star_hist_exog_list),
        )
        activity_parts = []
        if not self.exclude_insample_y:
            activity_parts.append(star_payload["target_activity"])
        if zero_non_star.size(-1) > 0:
            activity_parts.append(zero_non_star)
        activity_parts.extend([star_payload["target_activity"]] * 4)
        if star_payload["star_hist_activity"].size(-1) > 0:
            activity_parts.extend([star_payload["star_hist_activity"]] * 4)
        full_token_activity = torch.cat(activity_parts, dim=2)
        token_activity = full_token_activity.clamp_min(0.0).sum(dim=1, keepdim=False)
        token_count = (full_token_activity > 0).sum(dim=1, keepdim=False).to(
            dtype=full_token_activity.dtype
        )
        return {
            "token_mask": token_count.unsqueeze(-1) > 0,
            "token_count": token_count.unsqueeze(-1),
            "token_activity": token_activity.unsqueeze(-1),
        }

    def _decode_timexer_forecast(
        self,
        *,
        raw_states: AATimeXerTokenStates,
        attended_states: AATimeXerTokenStates,
    ) -> torch.Tensor:
        if self.timexer_decoder is None:
            raise ValueError("TimeXer decoder is not initialized")
        raw_tokens = raw_states.combined()
        attended_tokens = attended_states.combined()
        raw_tokens = _apply_stochastic_dropout(
            raw_tokens,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        attended_tokens = _apply_stochastic_dropout(
            attended_tokens,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        decoder_input = torch.cat([raw_tokens, attended_tokens], dim=-1).permute(0, 1, 3, 2)
        decoder_input = _apply_stochastic_dropout(
            decoder_input,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        decoded = self.timexer_decoder(decoder_input)
        target_forecast = decoded[:, :1, :]
        return target_forecast.transpose(1, 2).reshape(raw_tokens.shape[0], self.h, -1)

    def _decode_itransformer_forecast(
        self,
        *,
        raw_tokens: torch.Tensor,
        attended_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.itransformer_decoder is None:
            raise ValueError("iTransformer decoder is not initialized")
        raw_tokens = _apply_stochastic_dropout(
            raw_tokens,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        attended_tokens = _apply_stochastic_dropout(
            attended_tokens,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        decoder_input = torch.cat([raw_tokens, attended_tokens], dim=-1)
        decoder_input = _apply_stochastic_dropout(
            decoder_input,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        decoded = self.itransformer_decoder(decoder_input)
        target_tokens = decoded[:, self.target_token_indices, :]
        pooled = target_tokens.reshape(
            decoded.shape[0],
            len(self.target_token_indices),
            self.h,
            self.loss.outputsize_multiplier,
        ).mean(dim=1)
        return pooled.reshape(decoded.shape[0], self.h, -1)

    def _build_time_decoder_features(
        self,
        *,
        hidden_states: torch.Tensor,
        attended_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_aligned = _align_horizon(
            hidden_states,
            h=self.h,
            input_size=self.input_size,
            sequence_adapter=self.sequence_adapter,
        )
        attended_aligned = _align_horizon(
            attended_states,
            h=self.h,
            input_size=self.input_size,
            sequence_adapter=self.sequence_adapter,
        )
        hidden_aligned = _apply_stochastic_dropout(
            hidden_aligned,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        attended_aligned = _apply_stochastic_dropout(
            attended_aligned,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        return hidden_aligned, attended_aligned

    def _project_event_summary(
        self,
        event_summary: torch.Tensor,
    ) -> torch.Tensor:
        if self.event_summary_projector is None:
            raise ValueError("Event summary projector is only available for informer backbone")
        event_summary = torch.nan_to_num(
            event_summary,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(min=-10.0, max=10.0)
        event_latent = self.event_summary_projector(event_summary)
        event_latent = torch.tanh(
            torch.nan_to_num(event_latent, nan=0.0, posinf=10.0, neginf=-10.0)
        )
        return _apply_stochastic_dropout(
            event_latent,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )

    def _build_time_decoder_input(
        self,
        *,
        hidden_states: torch.Tensor,
        attended_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_aligned, attended_aligned = self._build_time_decoder_features(
            hidden_states=hidden_states,
            attended_states=attended_states,
        )
        decoder_input = torch.cat([hidden_aligned, attended_aligned], dim=-1)
        return _apply_stochastic_dropout(
            decoder_input,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )

    def _decode_informer_forecast(
        self,
        *,
        hidden_states: torch.Tensor,
        attended_states: torch.Tensor,
        event_summary: torch.Tensor,
    ) -> torch.Tensor:
        if self.informer_decoder is None:
            raise ValueError("Informer decoder is not initialized")
        hidden_aligned, attended_aligned = self._build_time_decoder_features(
            hidden_states=hidden_states,
            attended_states=attended_states,
        )
        event_context = self._project_event_summary(event_summary)
        decoder_input = torch.cat([hidden_aligned, attended_aligned], dim=-1)
        decoder_input = _apply_stochastic_dropout(
            decoder_input,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        return self.informer_decoder(
            decoder_input,
            event_context,
        )

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        hist_exog = windows_batch["hist_exog"]
        star_payload = windows_batch.get("star_precomputed")
        if star_payload is None:
            star_payload = self._compute_star_outputs(insample_y, hist_exog)
        encoder_parts = []
        if not self.exclude_insample_y:
            encoder_parts.append(insample_y)
        non_star_hist_exog = self._select_hist_exog(
            hist_exog, self.non_star_hist_exog_indices
        )
        if non_star_hist_exog is not None:
            encoder_parts.append(non_star_hist_exog)
        encoder_parts.extend(
            [
                star_payload["target_trend"],
                star_payload["target_seasonal"],
                star_payload["target_anomalies"],
                star_payload["target_residual"],
            ]
        )
        if star_payload["star_hist_anomalies"].size(-1) > 0:
            encoder_parts.extend(
                [
                    star_payload["star_hist_trend"],
                    star_payload["star_hist_seasonal"],
                    star_payload["star_hist_anomalies"],
                    star_payload["star_hist_residual"],
                ]
            )
        encoder_input = torch.cat(encoder_parts, dim=2)

        backbone_states = self.encoder(encoder_input)
        if self.backbone == "timexer":
            if not isinstance(backbone_states, AATimeXerTokenStates):
                raise ValueError("AAForecast timexer backbone must return AATimeXerTokenStates")
            timexer_signals = self._aggregate_timexer_attention_signals(
                critical_mask=star_payload["critical_mask"],
                count_active_channels=star_payload["count_active_channels"],
                channel_activity=star_payload["channel_activity"],
            )
            (attended_patch, attended_global), _ = self.attention(
                backbone_states.patch_states,
                backbone_states.global_states,
                timexer_signals["patch_mask"],
                timexer_signals["patch_count"],
                timexer_signals["patch_activity"],
                timexer_signals["global_mask"],
                timexer_signals["global_count"],
                timexer_signals["global_activity"],
            )
            attended_states = AATimeXerTokenStates(
                patch_states=attended_patch,
                global_states=attended_global,
            )
            return self._decode_timexer_forecast(
                raw_states=backbone_states,
                attended_states=attended_states,
            )
        if self.backbone == "itransformer":
            token_signals = self._aggregate_itransformer_attention_signals(
                star_payload={
                    "target_activity": star_payload["target_activity"].to(
                        device=encoder_input.device,
                        dtype=encoder_input.dtype,
                    ),
                    "star_hist_activity": star_payload["star_hist_activity"].to(
                        device=encoder_input.device,
                        dtype=encoder_input.dtype,
                    ),
                },
                template=encoder_input,
            )
            attended_tokens, _ = self.attention(
                backbone_states,
                token_signals["token_mask"],
                token_signals["token_count"].to(
                    device=backbone_states.device,
                    dtype=backbone_states.dtype,
                ),
                token_signals["token_activity"].to(
                    device=backbone_states.device,
                    dtype=backbone_states.dtype,
                ),
            )
            return self._decode_itransformer_forecast(
                raw_tokens=backbone_states,
                attended_tokens=attended_tokens,
            )

        hidden_states = self.encoder.project_to_time_states(backbone_states)
        critical_mask = star_payload["critical_mask"].bool()
        count_active_channels = star_payload["count_active_channels"].to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        channel_activity = star_payload["channel_activity"].to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        event_summary = star_payload.get("event_summary")
        if event_summary is None:
            event_summary = self._build_event_summary_from_payload(
                star_payload,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            event_summary = event_summary.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        attended_states, _ = self.attention(
            hidden_states,
            critical_mask,
            count_active_channels,
            channel_activity,
        )
        if self.backbone == "informer":
            return self._decode_informer_forecast(
                hidden_states=hidden_states,
                attended_states=attended_states,
                event_summary=event_summary,
            )
        if self.decoder is None:
            raise ValueError("Shared decoder is not initialized")
        decoder_input = self._build_time_decoder_input(
            hidden_states=hidden_states,
            attended_states=attended_states,
        )
        return self.decoder(decoder_input)[:, -self.h :]
