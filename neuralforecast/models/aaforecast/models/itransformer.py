from __future__ import annotations

import torch
import torch.nn as nn

from ...itransformer import ITransformerTokenEncoderOnly
from .base import AABackboneAdapter, AABackboneEvidence, validate_attention_heads


class ITransformerBackboneAdapter(AABackboneAdapter):
    evidence = AABackboneEvidence(
        backbone="itransformer",
        reused_components=(
            "neuralforecast.models.itransformer.ITransformerTokenEncoderOnly",
            "neuralforecast.common._modules.DataEmbedding_inverted",
            "neuralforecast.common._modules.TransEncoder",
        ),
        aa_bridge_steps=(
            "project token states back to time states",
            "add a residual input projection for AA hidden-state stability",
        ),
        unavoidable_divergences=(
            "standalone projector/head remains outside the AA adapter",
            "token-to-time projection is the AA-specific bridge needed to recover per-timestep hidden states",
        ),
    )

    def __init__(
        self,
        *,
        feature_size: int,
        input_size: int,
        hidden_size: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
        d_ff: int,
        factor: int,
        use_norm: bool,
    ) -> None:
        super().__init__()
        validate_attention_heads(hidden_size, n_heads, field_name="n_heads")
        self.hidden_size = hidden_size
        self.encoder_only = ITransformerTokenEncoderOnly(
            input_size=input_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            use_norm=use_norm,
        )
        self.embedding = self.encoder_only.enc_embedding
        self.encoder = self.encoder_only.encoder
        self.token_to_time = nn.Linear(feature_size, input_size)
        self.input_projection = nn.Linear(feature_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size) if use_norm else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_tokens = self.encoder_only(inputs)
        time_states = self.token_to_time(encoded_tokens.transpose(1, 2)).transpose(1, 2)
        residual = self.input_projection(inputs)
        return self.output_norm(time_states + residual)


def build_itransformer_backbone(
    *,
    feature_size: int,
    input_size: int,
    hidden_size: int,
    n_heads: int,
    e_layers: int,
    dropout: float,
    d_ff: int,
    factor: int,
    use_norm: bool,
    **_: object,
) -> nn.Module:
    return ITransformerBackboneAdapter(
        feature_size=feature_size,
        input_size=input_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
        d_ff=d_ff,
        factor=factor,
        use_norm=use_norm,
    )
