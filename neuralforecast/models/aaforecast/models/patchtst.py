from __future__ import annotations

import torch
import torch.nn as nn

from ...patchtst import PatchTSTEncoderOnly
from .base import (
    AABackboneAdapter,
    AABackboneEvidence,
    scatter_patch_tokens_to_time_states,
    validate_attention_heads,
)


class PatchTSTBackboneAdapter(AABackboneAdapter):
    evidence = AABackboneEvidence(
        backbone="patchtst",
        reused_components=(
            "neuralforecast.models.patchtst.PatchTSTEncoderOnly",
            "neuralforecast.models.patchtst._patchtst_create_patches",
            "neuralforecast.models.patchtst._patchtst_patch_num",
            "neuralforecast.models.patchtst.TSTiEncoder",
        ),
        aa_bridge_steps=(
            "average per-variable patch states",
            "scatter patch states back to time positions",
            "layer-normalize time states for AA attention consumption",
        ),
        unavoidable_divergences=(
            "PatchTST forecast head is replaced by an AA-specific patch-to-time bridge",
            "RevIN stays disabled because AA routing already owns scaling semantics",
        ),
    )

    def __init__(
        self,
        *,
        feature_size: int,
        input_size: int,
        hidden_size: int,
        n_heads: int,
        encoder_layers: int,
        dropout: float,
        linear_hidden_size: int | None,
        attn_dropout: float,
        patch_len: int,
        stride: int,
    ) -> None:
        super().__init__()
        validate_attention_heads(hidden_size, n_heads, field_name="n_heads")
        if patch_len <= 0:
            raise ValueError(f"PatchTST patch_len must be positive, got {patch_len}")
        if patch_len > input_size:
            raise ValueError(
                "PatchTST patch_len must not exceed input_size; "
                f"got patch_len={patch_len}, input_size={input_size}"
            )
        if stride <= 0:
            raise ValueError(f"PatchTST stride must be positive, got {stride}")
        self.hidden_size = hidden_size
        self.patch_len = patch_len
        self.stride = stride
        self.encoder_only = PatchTSTEncoderOnly(
            c_in=feature_size,
            input_size=input_size,
            patch_len=patch_len,
            stride=stride,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            linear_hidden_size=linear_hidden_size or hidden_size * 4,
            attn_dropout=attn_dropout,
            dropout=dropout,
            revin=False,
            padding_patch="end",
        )
        self.encoder = self.encoder_only.encoder
        self.patch_num = self.encoder_only.patch_num
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_only(inputs)
        patch_tokens = encoded.mean(dim=1).permute(0, 2, 1)
        hidden = scatter_patch_tokens_to_time_states(
            patch_tokens,
            seq_len=inputs.shape[1],
            patch_len=self.patch_len,
            stride=self.stride,
            template=inputs,
        )
        return self.output_norm(hidden)


def build_patchtst_backbone(
    *,
    feature_size: int,
    input_size: int,
    hidden_size: int,
    n_heads: int,
    encoder_layers: int,
    dropout: float,
    linear_hidden_size: int | None,
    attn_dropout: float,
    patch_len: int,
    stride: int,
    **_: object,
) -> nn.Module:
    return PatchTSTBackboneAdapter(
        feature_size=feature_size,
        input_size=input_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        encoder_layers=encoder_layers,
        dropout=dropout,
        linear_hidden_size=linear_hidden_size,
        attn_dropout=attn_dropout,
        patch_len=patch_len,
        stride=stride,
    )
