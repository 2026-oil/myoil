from __future__ import annotations

import torch
import torch.nn as nn

from ...timexer import TimeXerEncoderOnly
from .base import (
    AABackboneAdapter,
    AABackboneEvidence,
    scatter_patch_tokens_to_time_states,
    validate_attention_heads,
)


class TimeXerBackboneAdapter(AABackboneAdapter):
    evidence = AABackboneEvidence(
        backbone="timexer",
        reused_components=(
            "neuralforecast.models.timexer.TimeXerEncoderOnly",
            "neuralforecast.models.timexer.EnEmbedding",
            "neuralforecast.models.timexer.Encoder",
            "neuralforecast.common._modules.DataEmbedding_inverted",
        ),
        aa_bridge_steps=(
            "split TimeXer patch tokens from the global token",
            "scatter patch tokens back to time positions",
            "broadcast averaged global token context across time states",
        ),
        unavoidable_divergences=(
            "standalone forecast head remains outside the AA adapter",
            "the AA bridge reconstructs per-timestep hidden states from patch/global token outputs",
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
        patch_len: int,
        use_norm: bool,
    ) -> None:
        super().__init__()
        validate_attention_heads(hidden_size, n_heads, field_name="n_heads")
        self.hidden_size = hidden_size
        self.patch_len = patch_len
        self.encoder_only = TimeXerEncoderOnly(
            input_size=input_size,
            n_series=feature_size,
            patch_len=patch_len,
            hidden_size=hidden_size,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            use_norm=use_norm,
        )
        self.en_embedding = self.encoder_only.en_embedding
        self.ex_embedding = self.encoder_only.ex_embedding
        self.encoder = self.encoder_only.encoder
        self.output_norm = nn.LayerNorm(hidden_size) if use_norm else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_only(inputs)
        patch_tokens = encoded[..., :-1].mean(dim=1).permute(0, 2, 1)
        global_context = encoded[..., -1].mean(dim=1)
        hidden = scatter_patch_tokens_to_time_states(
            patch_tokens,
            seq_len=inputs.shape[1],
            patch_len=self.patch_len,
            stride=self.patch_len,
            template=inputs,
            global_context=global_context,
        )
        return self.output_norm(hidden)


def build_timexer_backbone(
    *,
    feature_size: int,
    input_size: int,
    hidden_size: int,
    n_heads: int,
    e_layers: int,
    dropout: float,
    d_ff: int,
    factor: int,
    patch_len: int,
    use_norm: bool,
    **_: object,
) -> nn.Module:
    return TimeXerBackboneAdapter(
        feature_size=feature_size,
        input_size=input_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
        d_ff=d_ff,
        factor=factor,
        patch_len=patch_len,
        use_norm=use_norm,
    )
