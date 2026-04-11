from __future__ import annotations

import torch
import torch.nn as nn

from ...vanillatransformer import VanillaTransformerEncoderOnly
from .base import AABackboneAdapter, AABackboneEvidence, validate_attention_heads


class VanillaTransformerBackboneAdapter(AABackboneAdapter):
    evidence = AABackboneEvidence(
        backbone="vanillatransformer",
        reused_components=(
            "neuralforecast.models.vanillatransformer.VanillaTransformerEncoderOnly",
            "neuralforecast.common._modules.DataEmbedding",
            "neuralforecast.common._modules.TransEncoder",
        ),
        aa_bridge_steps=(
            "treat AA encoder input as the autoregressive channel set consumed by the standalone encoder-only path",
        ),
        unavoidable_divergences=(
            "standalone decoder/head remains outside the AA adapter because AA consumes encoder hidden states directly",
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
    ) -> None:
        super().__init__()
        validate_attention_heads(hidden_size, n_head, field_name="n_head")
        self.hidden_size = hidden_size
        self.encoder_only = VanillaTransformerEncoderOnly(
            c_in=feature_size,
            exog_input_size=0,
            hidden_size=hidden_size,
            n_head=n_head,
            conv_hidden_size=linear_hidden_size or hidden_size * 4,
            activation="gelu",
            encoder_layers=encoder_layers,
            dropout=dropout,
        )
        self.encoder = self.encoder_only.encoder
        self.enc_embedding = self.encoder_only.enc_embedding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder_only(inputs, None)


def build_vanillatransformer_backbone(
    *,
    feature_size: int,
    hidden_size: int,
    n_head: int,
    encoder_layers: int,
    dropout: float,
    linear_hidden_size: int | None,
    **_: object,
) -> nn.Module:
    return VanillaTransformerBackboneAdapter(
        feature_size=feature_size,
        hidden_size=hidden_size,
        n_head=n_head,
        encoder_layers=encoder_layers,
        dropout=dropout,
        linear_hidden_size=linear_hidden_size,
    )
