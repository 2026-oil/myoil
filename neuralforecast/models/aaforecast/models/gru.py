from __future__ import annotations

import torch
import torch.nn as nn

from ..gru import _build_encoder
from .base import AABackboneAdapter, AABackboneEvidence


class GRUBackboneAdapter(AABackboneAdapter):
    evidence = AABackboneEvidence(
        backbone="gru",
        reused_components=("neuralforecast.models.aaforecast.gru._build_encoder",),
        aa_bridge_steps=("none: encoder already returns per-timestep hidden states",),
        unavoidable_divergences=(
            "standalone GRU forecast head remains outside the AA encoder adapter",
        ),
    )

    def __init__(
        self,
        *,
        feature_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = feature_size
        self.encoder = _build_encoder(
            feature_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.encoder(inputs)
        return hidden_states


def build_gru_backbone(
    *,
    feature_size: int,
    encoder_hidden_size: int,
    encoder_n_layers: int,
    encoder_dropout: float,
    **_: object,
) -> nn.Module:
    return GRUBackboneAdapter(
        feature_size=feature_size,
        hidden_size=encoder_hidden_size,
        num_layers=encoder_n_layers,
        dropout=encoder_dropout,
    )
