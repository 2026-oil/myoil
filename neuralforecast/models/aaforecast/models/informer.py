from __future__ import annotations

import torch
import torch.nn as nn

from ...informer import InformerEncoderOnly
from .base import AABackboneAdapter, AABackboneEvidence, validate_attention_heads


class InformerBackboneAdapter(AABackboneAdapter):
    evidence = AABackboneEvidence(
        backbone="informer",
        reused_components=(
            "neuralforecast.models.informer.InformerEncoderOnly",
            "neuralforecast.models.informer.ProbAttention",
            "neuralforecast.common._modules.DataEmbedding",
            "neuralforecast.common._modules.TransEncoder",
        ),
        aa_bridge_steps=(
            "treat AA encoder input as the autoregressive channel set consumed by the encoder-only Informer path",
        ),
        unavoidable_divergences=(
            "Informer distillation is disabled to preserve per-timestep alignment required by AA attention",
            "standalone decoder/head remains outside the AA adapter",
        ),
    )

    def __init__(
        self,
        *,
        feature_size: int,
        hidden_size: int,
        n_head: int,
        encoder_layers: int,
        dropout: float,
        linear_hidden_size: int | None,
        factor: int,
    ) -> None:
        super().__init__()
        validate_attention_heads(hidden_size, n_head, field_name="n_head")
        self.hidden_size = hidden_size
        self.encoder_only = InformerEncoderOnly(
            c_in=feature_size,
            exog_input_size=0,
            hidden_size=hidden_size,
            factor=factor,
            n_head=n_head,
            conv_hidden_size=linear_hidden_size or hidden_size * 4,
            activation="gelu",
            encoder_layers=encoder_layers,
            dropout=dropout,
            distil=False,
        )
        self.encoder = self.encoder_only.encoder
        self.enc_embedding = self.encoder_only.enc_embedding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder_only(inputs, None)


def build_informer_backbone(
    *,
    feature_size: int,
    hidden_size: int,
    n_head: int,
    encoder_layers: int,
    dropout: float,
    linear_hidden_size: int | None,
    factor: int,
    **_: object,
) -> nn.Module:
    return InformerBackboneAdapter(
        feature_size=feature_size,
        hidden_size=hidden_size,
        n_head=n_head,
        encoder_layers=encoder_layers,
        dropout=dropout,
        linear_hidden_size=linear_hidden_size,
        factor=factor,
    )
