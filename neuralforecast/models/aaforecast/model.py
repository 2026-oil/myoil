from __future__ import annotations

__all__ = ["AAForecast"]

import os
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Informer-only path-aware MIMO decoder head.

    The previous Informer path used per-horizon heads after a shared trunk.
    That separated horizons, but it still treated each horizon mostly like an
    independent scalar. This head keeps horizon conditioning while decoding the
    whole trajectory jointly so h1/h2 can share one path representation.
    """

    def __init__(
        self,
        *,
        h: int,
        in_features: int,
        event_features: int,
        path_features: int,
        regime_features: int,
        pooled_features: int,
        hidden_size: int,
        out_features: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.h = int(h)
        self.base_in_features = int(in_features)
        self.event_features = int(event_features)
        self.path_features = int(path_features)
        self.regime_features = int(regime_features)
        self.pooled_features = int(pooled_features)
        self.hidden_size = int(hidden_size)
        self.horizon_embeddings = nn.Embedding(
            num_embeddings=self.h,
            embedding_dim=self.hidden_size,
        )
        self.event_gate = nn.Sequential(
            nn.Linear(self.event_features + self.hidden_size, self.event_features),
            nn.Sigmoid(),
        )
        self.path_gate = nn.Sequential(
            nn.Linear(self.path_features + self.hidden_size, self.path_features),
            nn.Sigmoid(),
        )
        self.regime_projector = MLP(
            in_features=self.regime_features,
            out_features=self.hidden_size,
            hidden_size=max(self.hidden_size, self.regime_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.regime_gate = nn.Sequential(
            nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
        )
        context_feature_size = self.event_features + self.path_features + self.regime_features
        film_hidden_size = max(self.base_in_features, context_feature_size)
        self.decoder_scale = MLP(
            in_features=context_feature_size,
            out_features=self.base_in_features,
            hidden_size=film_hidden_size,
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.decoder_shift = MLP(
            in_features=context_feature_size,
            out_features=self.base_in_features,
            hidden_size=film_hidden_size,
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.shared_trunk = MLP(
            in_features=(
                self.base_in_features
                + self.hidden_size
                + (2 * self.event_features)
                + (2 * self.path_features)
                + (2 * self.hidden_size)
            ),
            out_features=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            activation="ReLU",
            dropout=dropout,
        )
        attention_heads = 4
        while attention_heads > 1 and (hidden_size % attention_heads) != 0:
            attention_heads -= 1
        self.path_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.path_mixer = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.memory_transport_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.memory_transport_projector = nn.Linear(self.pooled_features, hidden_size)
        self.local_head = nn.Linear(hidden_size, out_features)
        joint_hidden_size = max(hidden_size, self.h * hidden_size)
        self.level_head = MLP(
            in_features=(
                (self.h * hidden_size)
                + self.event_features
                + self.path_features
                + self.pooled_features
            ),
            out_features=out_features,
            hidden_size=joint_hidden_size,
            num_layers=max(2, int(num_layers)),
            activation="ReLU",
            dropout=dropout,
        )
        self.global_head = MLP(
            in_features=(
                (self.h * hidden_size)
                + self.event_features
                + self.path_features
                + self.pooled_features
            ),
            out_features=self.h * out_features,
            hidden_size=joint_hidden_size,
            num_layers=max(2, int(num_layers)),
            activation="ReLU",
            dropout=dropout,
        )
        self.delta_head = MLP(
            in_features=(
                (self.h * hidden_size)
                + self.event_features
                + self.path_features
                + self.pooled_features
            ),
            out_features=self.h * out_features,
            hidden_size=joint_hidden_size,
            num_layers=max(2, int(num_layers)),
            activation="ReLU",
            dropout=dropout,
        )
        shock_hidden_size = max(
            hidden_size,
            self.event_features + self.path_features + self.pooled_features,
        )
        self.event_bias_head = MLP(
            in_features=self.event_features + self.path_features + self.pooled_features,
            out_features=self.h * out_features,
            hidden_size=shock_hidden_size,
            num_layers=max(2, int(num_layers)),
            activation="ReLU",
            dropout=dropout,
        )
        self.event_delta_head = MLP(
            in_features=self.event_features + self.path_features + self.pooled_features,
            out_features=self.h * out_features,
            hidden_size=shock_hidden_size,
            num_layers=max(2, int(num_layers)),
            activation="ReLU",
            dropout=dropout,
        )
        self.event_delta_gate = MLP(
            self.event_features + self.path_features + self.pooled_features,
            out_features,
            hidden_size=shock_hidden_size,
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.path_amplitude_head = MLP(
            in_features=(
                self.event_features
                + self.path_features
                + self.regime_features
                + self.pooled_features
            ),
            out_features=out_features,
            hidden_size=max(
                shock_hidden_size,
                self.regime_features + self.path_features + self.pooled_features,
            ),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.level_shift_head = MLP(
            in_features=(
                self.event_features
                + self.path_features
                + self.regime_features
                + self.pooled_features
            ),
            out_features=out_features,
            hidden_size=max(
                shock_hidden_size,
                self.regime_features + self.path_features + self.pooled_features,
            ),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.normal_expert_head = MLP(
            in_features=(
                (self.h * hidden_size)
                + self.event_features
                + self.path_features
                + self.pooled_features
            ),
            out_features=self.h * out_features,
            hidden_size=joint_hidden_size,
            num_layers=max(2, int(num_layers)),
            activation="ReLU",
            dropout=dropout,
        )
        self.spike_expert_head = MLP(
            in_features=(
                (self.h * hidden_size)
                + self.event_features
                + self.path_features
                + self.pooled_features
            ),
            out_features=self.h * out_features,
            hidden_size=joint_hidden_size,
            num_layers=max(2, int(num_layers)),
            activation="ReLU",
            dropout=dropout,
        )
        trajectory_context_features = (
            self.event_features
            + self.path_features
            + self.regime_features
            + self.pooled_features
        )
        self.trajectory_seed_head = MLP(
            in_features=trajectory_context_features,
            out_features=hidden_size,
            hidden_size=max(hidden_size, trajectory_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.trajectory_memory_seed_head = MLP(
            in_features=self.pooled_features,
            out_features=hidden_size,
            hidden_size=max(hidden_size, self.pooled_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.trajectory_input_head = MLP(
            in_features=self.hidden_size + trajectory_context_features,
            out_features=hidden_size,
            hidden_size=max(hidden_size, trajectory_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.trajectory_cell = nn.GRUCell(hidden_size, hidden_size)
        self.trajectory_output_head = nn.Linear(hidden_size, out_features)
        self.trajectory_gate_head = MLP(
            in_features=trajectory_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, trajectory_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        prototype_context_features = (
            trajectory_context_features + self.pooled_features + out_features + 1
        )
        self.family_blend_gate_head = MLP(
            in_features=prototype_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, prototype_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.prototype_query_head = MLP(
            in_features=prototype_context_features,
            out_features=hidden_size,
            hidden_size=max(hidden_size, prototype_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.prototype_gain_head = MLP(
            in_features=prototype_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, prototype_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.prototype_level_head = MLP(
            in_features=prototype_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, prototype_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.prototype_key_bank = nn.Parameter(
            0.05 * torch.randn(6, hidden_size)
        )
        self.prototype_increment_bank = nn.Parameter(
            0.05 * torch.randn(6, self.h, out_features)
        )
        self.memory_transport_gate_head = MLP(
            in_features=trajectory_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, trajectory_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        semantic_baseline_context_features = (
            (self.h * hidden_size) + self.event_features + self.regime_features
        )
        self.semantic_baseline_level_head = MLP(
            in_features=semantic_baseline_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, semantic_baseline_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.semantic_baseline_delta_head = MLP(
            in_features=semantic_baseline_context_features,
            out_features=self.h * out_features,
            hidden_size=max(hidden_size, semantic_baseline_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        semantic_spike_context_features = (
            trajectory_context_features + self.pooled_features + out_features + 1
        )
        self.semantic_spike_seed_head = MLP(
            in_features=semantic_spike_context_features,
            out_features=hidden_size,
            hidden_size=max(hidden_size, semantic_spike_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.semantic_spike_step_head = MLP(
            in_features=(
                hidden_size
                + hidden_size
                + self.hidden_size
                + self.path_features
                + self.regime_features
            ),
            out_features=hidden_size,
            hidden_size=max(
                hidden_size,
                self.path_features + self.regime_features + self.hidden_size,
            ),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.semantic_spike_cell = nn.GRUCell(hidden_size, hidden_size)
        self.semantic_spike_pos_out_head = nn.Linear(hidden_size, out_features)
        self.semantic_spike_neg_out_head = nn.Linear(hidden_size, out_features)
        self.semantic_spike_gate_head = MLP(
            in_features=semantic_spike_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, semantic_spike_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.semantic_spike_gain_head = MLP(
            in_features=semantic_spike_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, semantic_spike_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.semantic_spike_direction_head = MLP(
            in_features=semantic_spike_context_features,
            out_features=out_features,
            hidden_size=max(hidden_size, semantic_spike_context_features),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )
        self.expert_gate = MLP(
            in_features=(
                self.event_features
                + self.path_features
                + self.regime_features
                + self.pooled_features
            ),
            out_features=out_features,
            hidden_size=max(
                shock_hidden_size,
                self.regime_features + self.path_features + self.pooled_features,
            ),
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
        )

    def _validate_inputs(
        self,
        decoder_input: torch.Tensor,
        event_summary: torch.Tensor,
        event_path: torch.Tensor,
        raw_regime: torch.Tensor,
        pooled_context: torch.Tensor | None,
        memory_token: torch.Tensor | None,
        memory_bank: torch.Tensor | None,
    ) -> None:
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
        if event_path.ndim != 2:
            raise ValueError(
                "InformerHorizonAwareHead event_path must be rank-2 [B, features]"
            )
        if event_path.shape[0] != decoder_input.shape[0]:
            raise ValueError(
                "InformerHorizonAwareHead event_path batch must match decoder_input batch"
            )
        if event_path.shape[1] != self.path_features:
            raise ValueError(
                "InformerHorizonAwareHead event_path width must match path_features"
            )
        if raw_regime.ndim != 2:
            raise ValueError(
                "InformerHorizonAwareHead raw_regime must be rank-2 [B, features]"
            )
        if raw_regime.shape[0] != decoder_input.shape[0]:
            raise ValueError(
                "InformerHorizonAwareHead raw_regime batch must match decoder_input batch"
            )
        if raw_regime.shape[1] != self.regime_features:
            raise ValueError(
                "InformerHorizonAwareHead raw_regime width must match regime_features"
            )
        if pooled_context is not None:
            if pooled_context.ndim != 2:
                raise ValueError(
                    "InformerHorizonAwareHead pooled_context must be rank-2 [B, features]"
                )
            if pooled_context.shape[0] != decoder_input.shape[0]:
                raise ValueError(
                    "InformerHorizonAwareHead pooled_context batch must match decoder_input batch"
                )
            if pooled_context.shape[1] != self.pooled_features:
                raise ValueError(
                    "InformerHorizonAwareHead pooled_context width must match pooled_features"
                )
        if memory_token is not None:
            if memory_token.ndim != 2:
                raise ValueError(
                    "InformerHorizonAwareHead memory_token must be rank-2 [B, features]"
                )
            if memory_token.shape[0] != decoder_input.shape[0]:
                raise ValueError(
                    "InformerHorizonAwareHead memory_token batch must match decoder_input batch"
                )
            if memory_token.shape[1] != self.pooled_features:
                raise ValueError(
                    "InformerHorizonAwareHead memory_token width must match pooled_features"
                )
        if memory_bank is not None:
            if memory_bank.ndim != 3:
                raise ValueError(
                    "InformerHorizonAwareHead memory_bank must be rank-3 [B, steps, features]"
                )
            if memory_bank.shape[0] != decoder_input.shape[0]:
                raise ValueError(
                    "InformerHorizonAwareHead memory_bank batch must match decoder_input batch"
                )
            if memory_bank.shape[2] != self.pooled_features:
                raise ValueError(
                    "InformerHorizonAwareHead memory_bank width must match pooled_features"
                )

    def _build_conditioned_features(
        self,
        *,
        decoder_input: torch.Tensor,
        event_summary: torch.Tensor,
        event_path: torch.Tensor,
        raw_regime: torch.Tensor,
        pooled_context: torch.Tensor | None,
    ) -> torch.Tensor:
        horizon_context = self.build_horizon_context(
            batch_size=decoder_input.shape[0],
            device=decoder_input.device,
            dtype=decoder_input.dtype,
        )
        repeated_event = event_summary.unsqueeze(1).expand(-1, self.h, -1).to(
            dtype=decoder_input.dtype
        )
        repeated_path = event_path.unsqueeze(1).expand(-1, self.h, -1).to(
            dtype=decoder_input.dtype
        )
        context_features = torch.cat(
            [
                event_summary.to(dtype=decoder_input.dtype),
                event_path.to(dtype=decoder_input.dtype),
                raw_regime.to(dtype=decoder_input.dtype),
            ],
            dim=-1,
        )
        decoder_scale = (
            1.0 + torch.tanh(self.decoder_scale(context_features)).unsqueeze(1)
        )
        decoder_shift = self.decoder_shift(context_features).unsqueeze(1)
        modulated_decoder_input = (decoder_input * decoder_scale) + decoder_shift
        regime_latent = self.regime_projector(
            raw_regime.to(dtype=decoder_input.dtype)
        )
        regime_latent = F.gelu(
            torch.nan_to_num(regime_latent, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
                min=-20.0,
                max=20.0,
            )
        )
        repeated_regime = regime_latent.unsqueeze(1).expand(-1, self.h, -1)
        event_gate = self.event_gate(torch.cat([repeated_event, horizon_context], dim=-1))
        path_gate = self.path_gate(torch.cat([repeated_path, horizon_context], dim=-1))
        regime_gate = self.regime_gate(torch.cat([repeated_regime, horizon_context], dim=-1))
        return torch.cat(
            [
                modulated_decoder_input,
                horizon_context,
                repeated_event,
                repeated_event * event_gate,
                repeated_path,
                repeated_path * path_gate,
                repeated_regime,
                repeated_regime * regime_gate,
            ],
            dim=-1,
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
        event_path: torch.Tensor,
        raw_regime: torch.Tensor,
        pooled_context: torch.Tensor | None = None,
        memory_signal: torch.Tensor | None = None,
        anchor_value: torch.Tensor | None = None,
        memory_token: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(
            decoder_input,
            event_summary,
            event_path,
            raw_regime,
            pooled_context,
            memory_token,
            memory_bank,
        )
        conditioned = self._build_conditioned_features(
            decoder_input=decoder_input,
            event_summary=event_summary,
            event_path=event_path,
            raw_regime=raw_regime,
            pooled_context=pooled_context,
        )
        trunk_features = self.shared_trunk(conditioned)
        attended_path, _ = self.path_attention(
            trunk_features,
            trunk_features,
            trunk_features,
            need_weights=False,
        )
        trunk_features = trunk_features + attended_path
        mixed_path, _ = self.path_mixer(trunk_features)
        mixed_path = mixed_path + trunk_features
        if pooled_context is None:
            pooled_context = decoder_input.new_zeros(
                (decoder_input.shape[0], self.pooled_features)
            )
        else:
            pooled_context = pooled_context.to(dtype=decoder_input.dtype)
        if memory_signal is None:
            memory_signal = decoder_input.new_zeros((decoder_input.shape[0], 1))
        else:
            memory_signal = memory_signal.to(dtype=decoder_input.dtype)
        if memory_token is None:
            memory_token = decoder_input.new_zeros(
                (decoder_input.shape[0], self.pooled_features)
            )
        else:
            memory_token = memory_token.to(dtype=decoder_input.dtype)
        if memory_bank is None:
            memory_bank = memory_token.unsqueeze(1)
        else:
            memory_bank = memory_bank.to(dtype=decoder_input.dtype)
        memory_bank = self.memory_transport_projector(memory_bank)
        if anchor_value is None:
            anchor_value = decoder_input.new_zeros(
                (decoder_input.shape[0], self.local_head.out_features)
            )
        else:
            anchor_value = anchor_value.to(dtype=decoder_input.dtype)
        pooled_for_baseline = decoder_input.new_zeros(
            (decoder_input.shape[0], self.pooled_features)
        )
        baseline_context = torch.cat(
            [
                mixed_path.reshape(mixed_path.shape[0], -1),
                event_summary.to(dtype=decoder_input.dtype),
                event_path.to(dtype=decoder_input.dtype),
                pooled_for_baseline,
            ],
            dim=-1,
        )
        spike_context = torch.cat(
            [
                mixed_path.reshape(mixed_path.shape[0], -1),
                event_summary.to(dtype=decoder_input.dtype),
                event_path.to(dtype=decoder_input.dtype),
                pooled_context,
            ],
            dim=-1,
        )
        global_path = self.global_head(baseline_context).reshape(
            decoder_input.shape[0], self.h, -1
        )
        level = self.level_head(baseline_context).unsqueeze(1)
        delta_path = self.delta_head(baseline_context).reshape(
            decoder_input.shape[0], self.h, -1
        )
        delta_path = torch.cumsum(delta_path, dim=1)
        shock_context = torch.cat(
            [
                event_summary.to(dtype=decoder_input.dtype),
                event_path.to(dtype=decoder_input.dtype),
                pooled_context,
            ],
            dim=-1,
        )
        event_bias = self.event_bias_head(shock_context).reshape(
            decoder_input.shape[0], self.h, -1
        )
        event_delta = self.event_delta_head(shock_context).reshape(
            decoder_input.shape[0], self.h, -1
        )
        event_delta = torch.cumsum(F.softplus(event_delta), dim=1)
        event_delta_gate = (
            1.0 + F.softplus(self.event_delta_gate(shock_context))
        ).unsqueeze(1)
        baseline_amplitude_context = torch.cat(
            [
                event_summary.to(dtype=decoder_input.dtype),
                event_path.to(dtype=decoder_input.dtype),
                raw_regime.to(dtype=decoder_input.dtype),
                pooled_for_baseline,
            ],
            dim=-1,
        )
        spike_amplitude_context = torch.cat(
            [
                event_summary.to(dtype=decoder_input.dtype),
                event_path.to(dtype=decoder_input.dtype),
                raw_regime.to(dtype=decoder_input.dtype),
                pooled_context,
            ],
            dim=-1,
        )
        level_shift = self.level_shift_head(baseline_amplitude_context).unsqueeze(1)
        path_amplitude = (
            1.0 + F.softplus(self.path_amplitude_head(spike_amplitude_context))
        ).unsqueeze(1)
        normal_expert = self.normal_expert_head(baseline_context).reshape(
            decoder_input.shape[0], self.h, -1
        )
        spike_expert = self.spike_expert_head(spike_context).reshape(
            decoder_input.shape[0], self.h, -1
        )
        spike_expert = torch.cumsum(F.softplus(spike_expert), dim=1)
        expert_gate = torch.sigmoid(
            self.expert_gate(spike_amplitude_context)
            + (0.5 * memory_signal)
        ).unsqueeze(1)
        spike_uplift = expert_gate * spike_expert
        expert_residual = normal_expert + spike_uplift
        local_path = self.local_head(mixed_path)
        residual_path = (
            global_path
            + delta_path
            + local_path
            + event_bias
            + (event_delta * event_delta_gate)
            + expert_residual
        )
        trajectory_context = torch.cat(
            [
                event_summary.to(dtype=decoder_input.dtype),
                event_path.to(dtype=decoder_input.dtype),
                raw_regime.to(dtype=decoder_input.dtype),
                pooled_context,
            ],
            dim=-1,
        )
        trajectory_hidden = self.trajectory_seed_head(trajectory_context)
        trajectory_hidden = F.gelu(
            torch.nan_to_num(
                trajectory_hidden,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ).clamp(min=-20.0, max=20.0)
        )
        memory_seed = self.trajectory_memory_seed_head(
            pooled_context.to(dtype=decoder_input.dtype)
        )
        memory_seed = F.gelu(
            torch.nan_to_num(
                memory_seed,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ).clamp(min=-20.0, max=20.0)
        )
        trajectory_hidden = trajectory_hidden + (0.5 * memory_seed)
        horizon_context = self.build_horizon_context(
            batch_size=decoder_input.shape[0],
            device=decoder_input.device,
            dtype=decoder_input.dtype,
        )
        trajectory_steps: list[torch.Tensor] = []
        for step_idx in range(self.h):
            step_context = torch.cat(
                [
                    horizon_context[:, step_idx, :],
                    trajectory_context,
                ],
                dim=-1,
            )
            step_input = self.trajectory_input_head(step_context)
            trajectory_hidden = self.trajectory_cell(step_input, trajectory_hidden)
            trajectory_steps.append(0.15 * torch.sigmoid(self.trajectory_output_head(trajectory_hidden)))
        trajectory_shock = torch.cumsum(torch.stack(trajectory_steps, dim=1), dim=1)
        trajectory_gate = (
            1.0 + torch.sigmoid(self.trajectory_gate_head(trajectory_context))
        ).unsqueeze(1)
        anchor_scale = torch.log1p(anchor_value.abs().clamp_min(1.0)).unsqueeze(1)
        trajectory_component = trajectory_shock * trajectory_gate * anchor_scale
        semantic_baseline_context = torch.cat(
            [
                mixed_path.reshape(mixed_path.shape[0], -1),
                event_summary.to(dtype=decoder_input.dtype),
                raw_regime.to(dtype=decoder_input.dtype),
            ],
            dim=-1,
        )
        semantic_baseline_level = (
            0.1
            * torch.tanh(
                self.semantic_baseline_level_head(semantic_baseline_context)
            ).unsqueeze(1)
            * anchor_scale
        )
        memory_confidence = getattr(self, "_latest_memory_confidence", None)
        if memory_confidence is None:
            memory_confidence = decoder_input.new_zeros((decoder_input.shape[0], 1))
        else:
            memory_confidence = memory_confidence.to(dtype=decoder_input.dtype)
        prototype_context = torch.cat(
            [
                trajectory_context,
                memory_token,
                anchor_value,
                memory_signal,
            ],
            dim=-1,
        )
        family_gate = torch.sigmoid(
            self.family_blend_gate_head(prototype_context)
        ).unsqueeze(1)
        prototype_query = self.prototype_query_head(prototype_context)
        prototype_logits = torch.matmul(
            prototype_query, self.prototype_key_bank.t()
        )
        prototype_weights = torch.softmax(prototype_logits, dim=1)
        prototype_level = (
            self.prototype_level_head(prototype_context).unsqueeze(1) * anchor_scale
        )
        prototype_increments = torch.einsum(
            "bp,pho->bho",
            prototype_weights,
            self.prototype_increment_bank,
        )
        prototype_gain = torch.sigmoid(
            self.prototype_gain_head(prototype_context)
        ).unsqueeze(1)
        prototype_curve = prototype_increments * prototype_gain * anchor_scale
        semantic_spike_context = torch.cat(
            [
                trajectory_context,
                memory_token,
                anchor_value,
                memory_signal,
            ],
            dim=-1,
        )
        semantic_baseline_delta = self.semantic_baseline_delta_head(
            semantic_baseline_context
        ).reshape(decoder_input.shape[0], self.h, -1)
        semantic_baseline_curve = torch.cumsum(
            0.1 * torch.tanh(semantic_baseline_delta),
            dim=1,
        ) * anchor_scale
        semantic_spike_hidden = self.semantic_spike_seed_head(semantic_spike_context)
        semantic_spike_hidden = F.gelu(
            torch.nan_to_num(
                semantic_spike_hidden,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ).clamp(min=-20.0, max=20.0)
        )
        semantic_spike_pos_steps: list[torch.Tensor] = []
        semantic_spike_neg_steps: list[torch.Tensor] = []
        for step_idx in range(self.h):
            semantic_query = semantic_spike_hidden.unsqueeze(1)
            semantic_memory_step, _ = self.memory_transport_attention(
                semantic_query,
                memory_bank,
                memory_bank,
                need_weights=False,
            )
            semantic_step_features = torch.cat(
                [
                    semantic_spike_hidden,
                    semantic_memory_step.squeeze(1),
                    horizon_context[:, step_idx, :],
                    event_path.to(dtype=decoder_input.dtype),
                    raw_regime.to(dtype=decoder_input.dtype),
                ],
                dim=-1,
            )
            semantic_step_input = self.semantic_spike_step_head(
                semantic_step_features
            )
            semantic_spike_hidden = self.semantic_spike_cell(
                semantic_step_input,
                semantic_spike_hidden,
            )
            semantic_spike_pos_steps.append(
                F.softplus(self.semantic_spike_pos_out_head(semantic_spike_hidden))
            )
            semantic_spike_neg_steps.append(
                F.softplus(self.semantic_spike_neg_out_head(semantic_spike_hidden))
            )
        semantic_spike_pos_curve = torch.stack(semantic_spike_pos_steps, dim=1)
        semantic_spike_neg_curve = torch.stack(semantic_spike_neg_steps, dim=1)
        semantic_spike_gate = torch.sigmoid(
            self.semantic_spike_gate_head(semantic_spike_context)
            + (0.5 * memory_signal)
        ).unsqueeze(1)
        semantic_spike_gain = torch.sigmoid(
            self.semantic_spike_gain_head(semantic_spike_context)
        ).unsqueeze(1)
        semantic_spike_direction = torch.sigmoid(
            self.semantic_spike_direction_head(semantic_spike_context)
            + (0.5 * memory_signal)
        ).unsqueeze(1)
        semantic_negative_weight = 0.9 * (1.0 - semantic_spike_direction).pow(2)
        semantic_spike_curve = (
            (semantic_spike_direction * semantic_spike_pos_curve)
            - (semantic_negative_weight * semantic_spike_neg_curve)
        )
        semantic_spike_component = (
            semantic_spike_curve * semantic_spike_gate * semantic_spike_gain * anchor_scale
        )
        memory_transport_states, _ = self.memory_transport_attention(
            mixed_path,
            memory_bank,
            memory_bank,
            need_weights=False,
        )
        prototype_memory_curve = (
            0.1 * torch.tanh(self.local_head(memory_transport_states)) * anchor_scale
        )
        prototype_memory_confidence = torch.sqrt(
            memory_confidence.clamp_min(0.0)
        ).unsqueeze(1)
        prototype_component = (
            prototype_level
            + prototype_curve
            + (prototype_memory_curve * prototype_memory_confidence)
        ) * family_gate * memory_confidence.unsqueeze(1)
        memory_transport = torch.cumsum(
            F.softplus(self.local_head(memory_transport_states)),
            dim=1,
        )
        memory_transport_gate = (
            1.0 + torch.sigmoid(self.memory_transport_gate_head(trajectory_context))
        ).unsqueeze(1)
        analogue_component = trajectory_component + (
            memory_transport * memory_transport_gate * anchor_scale
        )
        final_output = (
            semantic_baseline_level
            + semantic_spike_component
            + prototype_component
        )
        def _summary_tensor(value: torch.Tensor) -> torch.Tensor:
            if value.ndim >= 3:
                return value.detach().mean(dim=0)
            if value.ndim == 2:
                return value.detach().mean(dim=0, keepdim=True)
            return value.detach()
        self.latest_debug = {
            "level": _summary_tensor(level),
            "level_shift": _summary_tensor(level_shift),
            "global_path": _summary_tensor(global_path),
            "delta_path": _summary_tensor(delta_path),
            "local_path": _summary_tensor(local_path),
            "event_bias": _summary_tensor(event_bias),
            "event_delta": _summary_tensor(event_delta),
            "event_delta_gate": _summary_tensor(event_delta_gate),
            "normal_expert": _summary_tensor(normal_expert),
            "spike_expert": _summary_tensor(spike_expert),
            "expert_gate": _summary_tensor(expert_gate),
            "spike_uplift": _summary_tensor(spike_uplift),
            "expert_residual": _summary_tensor(expert_residual),
            "trajectory_shock": _summary_tensor(trajectory_shock),
            "trajectory_gate": _summary_tensor(trajectory_gate),
            "memory_seed": _summary_tensor(memory_seed),
            "anchor_scale": _summary_tensor(anchor_scale),
            "semantic_baseline_level": _summary_tensor(semantic_baseline_level),
            "semantic_baseline_curve": _summary_tensor(semantic_baseline_curve),
            "semantic_spike_pos_curve": _summary_tensor(semantic_spike_pos_curve),
            "semantic_spike_neg_curve": _summary_tensor(semantic_spike_neg_curve),
            "semantic_spike_curve": _summary_tensor(semantic_spike_curve),
            "semantic_spike_gate": _summary_tensor(semantic_spike_gate),
            "semantic_spike_gain": _summary_tensor(semantic_spike_gain),
            "semantic_spike_direction": _summary_tensor(semantic_spike_direction),
            "semantic_negative_weight": _summary_tensor(semantic_negative_weight),
            "semantic_spike_component": _summary_tensor(semantic_spike_component),
            "trajectory_component": _summary_tensor(trajectory_component),
            "memory_transport": _summary_tensor(memory_transport),
            "memory_transport_gate": _summary_tensor(memory_transport_gate),
            "analogue_component": _summary_tensor(analogue_component),
            "path_amplitude": _summary_tensor(path_amplitude),
            "residual_path": _summary_tensor(residual_path),
            "final_output": _summary_tensor(final_output),
            "baseline_context_norm": baseline_context.detach().norm(dim=-1).mean(),
            "spike_context_norm": spike_context.detach().norm(dim=-1).mean(),
            "event_context_norm": event_summary.detach().norm(dim=-1).mean(),
            "event_path_norm": event_path.detach().norm(dim=-1).mean(),
            "pooled_context_norm": pooled_context.detach().norm(dim=-1).mean(),
            "raw_regime_norm": raw_regime.detach().norm(dim=-1).mean(),
        }
        return final_output


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
    EVENT_SUMMARY_SIZE = 21
    EVENT_TRAJECTORY_SIZE = 23
    NON_STAR_REGIME_SIZE = 8

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
        (
            self.non_star_market_regime_indices,
            self.non_star_policy_regime_indices,
        ) = self._resolve_non_star_regime_groups()
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
            self.regime_time_projector = None
            self.memory_query_projector = None
            self.memory_key_projector = None
            self.memory_value_projector = None
            self.memory_token_shock_head = None
            self.memory_token_shock_gate = None
            self.event_trajectory_projector = None
        elif self.backbone == "itransformer":
            self.itransformer_decoder = nn.Linear(
                2 * self.encoder_hidden_size,
                self.h * self.loss.outputsize_multiplier,
            )
            self.decoder = None
            self.timexer_decoder = None
            self.informer_decoder = None
            self.event_summary_projector = None
            self.regime_time_projector = None
            self.memory_query_projector = None
            self.memory_key_projector = None
            self.memory_value_projector = None
            self.memory_token_shock_head = None
            self.memory_token_shock_gate = None
            self.event_trajectory_projector = None
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
            self.regime_time_projector = None
            self.memory_query_projector = None
            self.memory_key_projector = None
            self.memory_value_projector = None
            self.memory_token_shock_head = None
            self.memory_token_shock_gate = None
            self.event_trajectory_projector = None

        self.transformer_anomaly_projection = None

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
        non_star_hist_exog = self._select_hist_exog(hist_exog, self.non_star_hist_exog_indices)
        star_hist_outputs = (
            self.star(
                star_hist_exog,
                tail_modes=self.star_hist_exog_tail_modes,
            )
            if star_hist_exog is not None
            else None
        )
        non_star_star_outputs = (
            self.star(
                non_star_hist_exog,
                tail_modes=tuple("two_sided" for _ in self.non_star_hist_exog_list),
            )
            if non_star_hist_exog is not None
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
        non_star_star_count = self._count_active_channels(
            None if non_star_star_outputs is None else non_star_star_outputs["critical_mask"],
            template=insample_y,
        )
        regime_activity, regime_count = self._build_non_star_regime_activity(
            non_star_hist_exog,
            dtype=insample_y.dtype,
            device=insample_y.device,
            batch_size=insample_y.size(0),
            seq_len=insample_y.size(1),
        )
        regime_intensity = regime_activity.sum(dim=2, keepdim=True)
        non_star_channels = max(len(self.non_star_hist_exog_list), 1)
        regime_density = regime_count / float(non_star_channels)
        combined_count = target_count + star_hist_count + non_star_star_count + regime_count
        target_activity = target_star["ranking_score"] * target_star["critical_mask"].to(
            dtype=insample_y.dtype
        )
        star_hist_activity = (
            star_hist_outputs["ranking_score"]
            * star_hist_outputs["critical_mask"].to(dtype=insample_y.dtype)
            if star_hist_outputs is not None
            else insample_y.new_empty((insample_y.size(0), insample_y.size(1), 0))
        )
        non_star_star_activity = (
            non_star_star_outputs["ranking_score"]
            * non_star_star_outputs["critical_mask"].to(dtype=insample_y.dtype)
            if non_star_star_outputs is not None
            else insample_y.new_empty((insample_y.size(0), insample_y.size(1), 0))
        )
        non_star_regime = (
            self._build_non_star_regime_descriptor(
                non_star_hist_exog,
                dtype=insample_y.dtype,
                device=insample_y.device,
            )
            if non_star_hist_exog is not None
            else insample_y.new_zeros((insample_y.size(0), self.NON_STAR_REGIME_SIZE))
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
                "channel_activity": torch.cat(
                    [target_activity, star_hist_activity, non_star_star_activity, regime_activity],
                    dim=2,
                ),
                "target_activity": target_activity,
                "star_hist_activity": star_hist_activity,
                "target_signed_score": target_signed_score,
                "star_hist_signed_score": star_hist_signed_score,
                "non_star_regime": non_star_regime,
            },
            dtype=insample_y.dtype,
            device=insample_y.device,
        )
        event_trajectory = self._build_event_trajectory_from_payload(
            {
                "critical_mask": combined_count > 0,
                "target_activity": target_activity,
                "star_hist_activity": star_hist_activity,
                "channel_activity": torch.cat(
                    [target_activity, star_hist_activity, non_star_star_activity, regime_activity],
                    dim=2,
                ),
                "target_signed_score": target_signed_score,
                "star_hist_signed_score": star_hist_signed_score,
                "non_star_star_activity": non_star_star_activity,
                "non_star_regime": non_star_regime,
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
            "target_count": target_count,
            "star_hist_count": star_hist_count,
            "star_hist_activity": star_hist_activity,
            "star_hist_signed_score": star_hist_signed_score,
            "channel_activity": torch.cat(
                [target_activity, star_hist_activity, non_star_star_activity, regime_activity],
                dim=2,
            ),
            "event_summary": event_summary,
            "event_trajectory": event_trajectory,
            "non_star_regime": non_star_regime,
            "non_star_star_activity": non_star_star_activity,
            "non_star_regime_activity": regime_activity,
            "non_star_star_count": non_star_star_count,
            "non_star_regime_count": regime_count,
            "regime_intensity": regime_intensity,
            "regime_density": regime_density,
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
        if set(all_names) != set(self.hist_exog_list) or len(all_names) != len(
            self.hist_exog_list
        ):
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

    def _resolve_non_star_regime_groups(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not self.non_star_hist_exog_list:
            return (), ()
        market_names = {
            "BS_Core_Index_A",
            "BS_Core_Index_C",
            "Idx_OVX",
            "Com_LMEX",
            "Com_BloombergCommodity_BCOM",
        }
        policy_names = {
            "GPRD_THREAT",
            "GPRD",
            "GPRD_ACT",
            "BS_Core_Index_B",
            "Idx_DxyUSD",
        }
        market_idx = tuple(
            idx
            for idx, name in enumerate(self.non_star_hist_exog_list)
            if name in market_names
        )
        policy_idx = tuple(
            idx
            for idx, name in enumerate(self.non_star_hist_exog_list)
            if name in policy_names
        )
        if not market_idx:
            market_idx = tuple(range(len(self.non_star_hist_exog_list)))
        if not policy_idx:
            policy_idx = tuple(range(len(self.non_star_hist_exog_list)))
        return market_idx, policy_idx

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
        non_star_star_activity = payload.get("non_star_star_activity")
        if non_star_star_activity is None:
            non_star_star_activity = channel_activity[:, :, 1:]
        else:
            non_star_star_activity = (
                non_star_star_activity.to(device=device, dtype=dtype).clamp_min(0.0)
            )

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
        non_star_star_density = (
            (non_star_star_activity > 0)
            .to(dtype=dtype)
            .mean(dim=(1, 2), keepdim=False)
            .unsqueeze(-1)
        )
        non_star_star_recent_density = torch.log1p(
            self._weighted_mean_feature(
                (non_star_star_activity > 0).to(dtype=dtype),
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        non_star_star_recent_mass = torch.log1p(
            self._weighted_mean_feature(
                non_star_star_activity,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        non_star_star_peak = torch.log1p(
            non_star_star_activity.amax(dim=(1, 2), keepdim=False).unsqueeze(-1)
        )
        peak_activity = torch.log1p(
            channel_activity.amax(dim=(1, 2), keepdim=False).unsqueeze(-1)
        )
        non_star_regime = payload.get("non_star_regime")
        if non_star_regime is None:
            non_star_regime = critical_mask.new_zeros(
                (critical_mask.shape[0], self.NON_STAR_REGIME_SIZE)
            )
        else:
            non_star_regime = non_star_regime.to(device=device, dtype=dtype)
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
                non_star_star_density,
                non_star_star_recent_density,
                non_star_star_recent_mass,
                non_star_star_peak,
                non_star_regime,
            ],
            dim=1,
        )
        summary = torch.nan_to_num(summary, nan=0.0, posinf=10.0, neginf=-10.0)
        return summary.clamp(min=-10.0, max=10.0)

    def _build_event_trajectory_from_payload(
        self,
        payload: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        critical_mask = payload["critical_mask"].to(device=device, dtype=dtype)
        target_signed_score = payload.get("target_signed_score")
        if target_signed_score is None:
            target_signed_score = payload.get("target_activity")
        if target_signed_score is None:
            target_signed_score = payload["channel_activity"][:, :, :1]
        target_signed_score = target_signed_score.to(device=device, dtype=dtype)

        star_hist_signed_score = payload.get("star_hist_signed_score")
        if star_hist_signed_score is None:
            star_hist_signed_score = payload.get("star_hist_activity")
        if star_hist_signed_score is None:
            star_hist_signed_score = payload["channel_activity"][:, :, 1:]
        star_hist_signed_score = star_hist_signed_score.to(device=device, dtype=dtype)

        seq_len = critical_mask.shape[1]
        recent_steps = max(1, seq_len // 2)
        time_weights = torch.linspace(
            0.35,
            1.0,
            steps=seq_len,
            device=device,
            dtype=dtype,
        ).view(1, seq_len, 1)

        target_positive = target_signed_score.clamp_min(0.0)
        hist_positive = star_hist_signed_score.clamp_min(0.0)
        non_star_star_activity = payload.get("non_star_star_activity")
        if non_star_star_activity is None:
            fallback_activity = payload.get("channel_activity")
            if fallback_activity is None:
                non_star_star_activity = critical_mask.new_zeros((critical_mask.shape[0], seq_len, 0))
            else:
                non_star_star_activity = fallback_activity[:, :, 1:]
        non_star_star_activity = (
            non_star_star_activity.to(device=device, dtype=dtype).clamp_min(0.0)
        )

        earlier_target_positive = target_positive[:, : seq_len - recent_steps, :]
        earlier_hist_positive = hist_positive[:, : seq_len - recent_steps, :]
        earlier_non_star_star = non_star_star_activity[:, : seq_len - recent_steps, :]

        recent_up_mass = torch.log1p(
            self._weighted_mean_feature(
                target_positive,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        earlier_up_mass = torch.log1p(
            self._mean_feature(earlier_target_positive, keepdim=False).unsqueeze(-1)
        )
        up_shift = recent_up_mass - earlier_up_mass
        terminal_up_mass = torch.log1p(
            target_positive[:, -1, :].mean(dim=1, keepdim=True)
        )
        recent_signed = torch.tanh(
            self._weighted_mean_feature(
                target_signed_score,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        terminal_signed = torch.tanh(
            target_signed_score[:, -1, :].mean(dim=1, keepdim=True)
        )
        recent_hist_up_mass = torch.log1p(
            self._weighted_mean_feature(
                hist_positive,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        earlier_hist_up_mass = torch.log1p(
            self._mean_feature(earlier_hist_positive, keepdim=False).unsqueeze(-1)
        )
        hist_up_shift = recent_hist_up_mass - earlier_hist_up_mass
        terminal_hist_up_mass = torch.log1p(
            hist_positive[:, -1, :].mean(dim=1, keepdim=True)
        )
        hist_recent_signed = torch.tanh(
            self._weighted_mean_feature(
                star_hist_signed_score,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        target_hist_gap = recent_signed - hist_recent_signed
        target_persistence = critical_mask[:, -recent_steps:, :].mean(
            dim=(1, 2), keepdim=False
        ).unsqueeze(-1)
        non_star_star_recent_mass = torch.log1p(
            self._weighted_mean_feature(
                non_star_star_activity,
                weights=time_weights,
                keepdim=False,
            ).unsqueeze(-1)
        )
        non_star_star_earlier_mass = torch.log1p(
            self._mean_feature(earlier_non_star_star, keepdim=False).unsqueeze(-1)
        )
        non_star_star_up_shift = non_star_star_recent_mass - non_star_star_earlier_mass
        non_star_star_terminal_mass = torch.log1p(
            non_star_star_activity[:, -1, :].mean(dim=1, keepdim=True)
        )
        non_star_star_persistence = (
            (non_star_star_activity[:, -recent_steps:, :] > 0)
            .to(dtype=dtype)
            .mean(dim=(1, 2), keepdim=False)
            .unsqueeze(-1)
        )
        non_star_regime = payload.get("non_star_regime")
        if non_star_regime is None:
            non_star_regime = critical_mask.new_zeros(
                (critical_mask.shape[0], self.NON_STAR_REGIME_SIZE)
            )
        else:
            non_star_regime = non_star_regime.to(device=device, dtype=dtype)

        trajectory = torch.cat(
            [
                recent_up_mass,
                up_shift,
                terminal_up_mass,
                recent_signed,
                terminal_signed,
                recent_hist_up_mass,
                hist_up_shift,
                terminal_hist_up_mass,
                hist_recent_signed,
                target_hist_gap,
                target_persistence,
                non_star_star_recent_mass,
                non_star_star_up_shift,
                non_star_star_terminal_mass,
                non_star_star_persistence,
                non_star_regime,
            ],
            dim=1,
        )
        trajectory = torch.nan_to_num(trajectory, nan=0.0, posinf=10.0, neginf=-10.0)
        return trajectory.clamp(min=-10.0, max=10.0)

    def _build_non_star_regime_descriptor(
        self,
        non_star_hist_exog: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if non_star_hist_exog is None or non_star_hist_exog.numel() == 0:
            return torch.zeros((1, self.NON_STAR_REGIME_SIZE), device=device, dtype=dtype)
        values = non_star_hist_exog.to(device=device, dtype=dtype)
        batch_size, seq_len, _ = values.shape
        recent_steps = max(1, seq_len // 8)
        def _group_descriptor(group_indices: tuple[int, ...]) -> torch.Tensor:
            if not group_indices:
                return values.new_zeros((batch_size, 4))
            group = values[:, :, list(group_indices)]
            recent = group[:, -recent_steps:, :]
            baseline = group.mean(dim=1, keepdim=True)
            baseline_scale = group.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
            recent_shift = (recent.mean(dim=1, keepdim=True) - baseline) / baseline_scale
            terminal_z = (group[:, -1:, :] - baseline) / baseline_scale
            return torch.cat(
                [
                    recent_shift.mean(dim=2, keepdim=False),
                    recent_shift.amax(dim=2, keepdim=False),
                    terminal_z.mean(dim=2, keepdim=False),
                    terminal_z.amax(dim=2, keepdim=False),
                ],
                dim=1,
            )
        descriptor = torch.cat(
            [
                _group_descriptor(self.non_star_market_regime_indices),
                _group_descriptor(self.non_star_policy_regime_indices),
            ],
            dim=1,
        )
        descriptor = torch.nan_to_num(descriptor, nan=0.0, posinf=10.0, neginf=-10.0)
        return descriptor.clamp(min=-10.0, max=10.0).reshape(
            batch_size, self.NON_STAR_REGIME_SIZE
        )

    def _build_non_star_regime_activity(
        self,
        non_star_hist_exog: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if non_star_hist_exog is None or non_star_hist_exog.numel() == 0:
            empty = torch.zeros((batch_size, seq_len, 0), device=device, dtype=dtype)
            return empty, torch.zeros((batch_size, seq_len, 1), device=device, dtype=dtype)
        values = non_star_hist_exog.to(device=device, dtype=dtype)
        recent_steps = max(1, seq_len // 8)
        earlier = values[:, : max(1, seq_len - recent_steps), :]
        baseline = earlier.mean(dim=1, keepdim=True)
        scale = earlier.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-2)
        positive_score = ((values - baseline) / scale).clamp_min(0.0)
        active_threshold = max(self.thresh * 0.15, 0.5)
        regime_excess = (positive_score - active_threshold).clamp_min(0.0)
        regime_activity = F.softplus(regime_excess) - math.log(2.0)
        regime_count = (regime_excess > 0).sum(dim=2, keepdim=True).to(dtype=dtype)
        return regime_activity, regime_count

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
        event_latent = F.gelu(
            torch.nan_to_num(event_latent, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
                min=-20.0,
                max=20.0,
            )
        )
        return _apply_stochastic_dropout(
            event_latent,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )

    def _project_event_trajectory(
        self,
        event_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        if self.event_trajectory_projector is None:
            raise ValueError(
                "Event trajectory projector is only available for informer backbone"
            )
        event_trajectory = torch.nan_to_num(
            event_trajectory,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(min=-10.0, max=10.0)
        event_path = self.event_trajectory_projector(event_trajectory)
        event_path = F.gelu(
            torch.nan_to_num(event_path, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
                min=-20.0,
                max=20.0,
            )
        )
        return _apply_stochastic_dropout(
            event_path,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )

    def _project_regime_time_context(
        self,
        regime_intensity: torch.Tensor,
        regime_density: torch.Tensor,
    ) -> torch.Tensor:
        if self.regime_time_projector is None:
            raise ValueError(
                "Regime time projector is only available for informer backbone"
            )
        regime_context = torch.cat(
            [regime_intensity, regime_density],
            dim=-1,
        )
        regime_context = torch.nan_to_num(
            regime_context,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(min=-10.0, max=10.0)
        projected = self.regime_time_projector(regime_context)
        projected = F.gelu(
            torch.nan_to_num(projected, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
                min=-20.0,
                max=20.0,
            )
        )
        return _apply_stochastic_dropout(
            projected,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )

    def _build_memory_pooled_context(
        self,
        *,
        hidden_states: torch.Tensor,
        attended_states: torch.Tensor,
        event_context: torch.Tensor,
        event_path: torch.Tensor,
        non_star_regime: torch.Tensor,
        regime_intensity: torch.Tensor,
        regime_density: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.memory_query_projector is None
            or self.memory_key_projector is None
            or self.memory_value_projector is None
        ):
            raise ValueError("Informer memory retriever is not initialized")
        query_context = torch.cat(
            [
                event_context,
                event_path,
                non_star_regime.to(dtype=hidden_states.dtype),
            ],
            dim=-1,
        )
        query = self.memory_query_projector(query_context)
        memory_states = hidden_states + attended_states
        keys = self.memory_key_projector(memory_states)
        values = self.memory_value_projector(memory_states)
        logits = torch.einsum("bh,bth->bt", query, keys) / math.sqrt(
            float(hidden_states.shape[-1])
        )
        regime_signal = torch.log1p(regime_intensity.clamp_min(0.0)).squeeze(-1)
        regime_signal = regime_signal + torch.log1p(regime_density.clamp_min(0.0)).squeeze(-1)
        logits = logits + regime_signal
        top_k = min(4, logits.shape[1])
        top_indices = None
        top_values = None
        if top_k < logits.shape[1]:
            top_values, top_indices = torch.topk(logits, k=top_k, dim=1)
            masked_logits = torch.full_like(logits, -1e9)
            masked_logits.scatter_(1, top_indices, top_values)
            logits = masked_logits
        else:
            top_values, top_indices = torch.topk(logits, k=top_k, dim=1)
        logits = logits - logits.amax(dim=1, keepdim=True)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (weights * values).sum(dim=1)
        self._latest_memory_debug = {
            "top_indices": top_indices.detach().cpu(),
            "top_values": top_values.detach().cpu(),
            "weight_mass_top1": weights.detach().amax(dim=1).mean().cpu(),
            "pooled_norm": pooled.detach().norm(dim=-1).mean().cpu(),
        }
        batch_idx = torch.arange(values.shape[0], device=values.device)
        self._latest_memory_token = values[batch_idx, top_indices[:, 0]]
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        self._latest_memory_bank = values.gather(1, gather_index)
        self._latest_memory_signal = torch.log1p(top_values.mean(dim=1, keepdim=True).clamp_min(0.0))
        self._latest_memory_confidence = weights.detach().amax(dim=1)
        return _apply_stochastic_dropout(
            pooled,
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
        event_trajectory: torch.Tensor,
        non_star_regime: torch.Tensor,
        anchor_level: torch.Tensor,
        regime_intensity: torch.Tensor,
        regime_density: torch.Tensor,
        count_active_channels: torch.Tensor,
    ) -> torch.Tensor:
        if self.informer_decoder is None:
            raise ValueError("Informer decoder is not initialized")
        hidden_aligned, attended_aligned = self._build_time_decoder_features(
            hidden_states=hidden_states,
            attended_states=attended_states,
        )
        if self.transformer_anomaly_projection is not None:
            count_aligned = _align_horizon(
                count_active_channels,
                h=self.h,
                input_size=self.input_size,
                sequence_adapter=self.sequence_adapter,
            )
            attended_aligned = attended_aligned + self.transformer_anomaly_projection(
                count_aligned
            )
        regime_time_latent = self._project_regime_time_context(
            regime_intensity,
            regime_density,
        )
        regime_time_aligned = _align_horizon(
            regime_time_latent,
            h=self.h,
            input_size=self.input_size,
            sequence_adapter=self.sequence_adapter,
        )
        event_context = self._project_event_summary(event_summary)
        event_path = self._project_event_trajectory(event_trajectory)
        pooled_context = self._build_memory_pooled_context(
            hidden_states=hidden_states,
            attended_states=attended_states,
            event_context=event_context,
            event_path=event_path,
            non_star_regime=non_star_regime,
            regime_intensity=regime_intensity,
            regime_density=regime_density,
        )
        memory_token = getattr(self, "_latest_memory_token", None)
        if memory_token is None:
            memory_token = pooled_context
        memory_bank = getattr(self, "_latest_memory_bank", None)
        decoder_input = torch.cat(
            [
                hidden_aligned + regime_time_aligned,
                attended_aligned + regime_time_aligned,
            ],
            dim=-1,
        )
        decoder_input = _apply_stochastic_dropout(
            decoder_input,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        delta_forecast = self.informer_decoder(
            decoder_input,
            event_context,
            event_path,
            non_star_regime,
            pooled_context,
            getattr(self, "_latest_memory_signal", None),
            anchor_level[:, -1, :],
            memory_token,
            memory_bank,
        )
        latest_decoder_debug = getattr(self.informer_decoder, "latest_debug", None)
        latest_memory_debug = getattr(self, "_latest_memory_debug", None)
        if isinstance(latest_decoder_debug, dict) and isinstance(latest_memory_debug, dict):
            merged_debug = dict(latest_decoder_debug)
            merged_debug.update(latest_memory_debug)
            self._latest_decoder_debug = merged_debug
        else:
            self._latest_decoder_debug = latest_decoder_debug
        anchor = anchor_level[:, -1:, :].to(dtype=delta_forecast.dtype)
        return anchor + delta_forecast

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        hist_exog = windows_batch["hist_exog"]
        self._latest_encoding_export = None
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
        if self.backbone == "informer" and bool(
            getattr(self, "_capture_encoding_export", False)
        ):
            self._latest_encoding_export = {
                "backbone_states": backbone_states.detach().cpu(),
                "hidden_states": hidden_states.detach().cpu(),
                "time_axis": 1,
            }
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
        event_trajectory = star_payload.get("event_trajectory")
        if event_trajectory is None:
            event_trajectory = self._build_event_trajectory_from_payload(
                star_payload,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            event_trajectory = event_trajectory.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        non_star_regime = star_payload.get("non_star_regime")
        if non_star_regime is None:
            non_star_regime = hidden_states.new_zeros(
                (hidden_states.shape[0], self.NON_STAR_REGIME_SIZE)
            )
        else:
            non_star_regime = non_star_regime.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        regime_intensity = star_payload.get("regime_intensity")
        if regime_intensity is None:
            regime_intensity = hidden_states.new_zeros((hidden_states.shape[0], self.input_size, 1))
        else:
            regime_intensity = regime_intensity.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        regime_density = star_payload.get("regime_density")
        if regime_density is None:
            regime_density = hidden_states.new_zeros((hidden_states.shape[0], self.input_size, 1))
        else:
            regime_density = regime_density.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        attended_states, _ = self.attention(
            hidden_states,
            critical_mask,
            count_active_channels,
            channel_activity,
        )
        if self.decoder is None:
            raise ValueError("Shared decoder is not initialized")
        decoder_input = self._build_time_decoder_input(
            hidden_states=hidden_states,
            attended_states=attended_states,
        )
        return self.decoder(decoder_input)[:, -self.h :]
