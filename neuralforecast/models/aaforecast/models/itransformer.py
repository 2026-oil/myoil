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
            "preserve raw token states through the iTransformer encoder path",
            "late-project token states back to time states immediately before AA sparse attention",
        ),
        unavoidable_divergences=(
            "standalone projector/head remains outside the AA adapter",
            "token-to-time late projection is still required because current AA sparse attention consumes timestep states",
        ),
        required_output="[B, token, hidden]",
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
        self.late_token_projection = nn.Linear(feature_size, input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder_only(inputs)

    def project_to_time_states(self, states: torch.Tensor) -> torch.Tensor:
        return self.late_token_projection(states.transpose(1, 2)).transpose(1, 2)


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
