from __future__ import annotations

import torch
import torch.nn as nn

from ...timexer import TimeXerEncoderOnly
from .base import (
    AABackboneAdapter,
    AABackboneEvidence,
    AATimeXerTokenStates,
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
            "preserve per-series patch tokens and per-series global tokens from the TimeXer encoder",
            "run AA sparse attention directly over the patch/global token structure",
        ),
        unavoidable_divergences=(
            "standalone forecast head remains outside the AA adapter",
            "AA still injects STAR-driven anomaly priors that are aggregated from timesteps to tokens",
        ),
        required_output="{patch:[B, channel, patch, hidden], global:[B, channel, 1, hidden]}",
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
        target_token_indices: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__()
        validate_attention_heads(hidden_size, n_heads, field_name="n_heads")
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.patch_len = patch_len
        self.patch_num = input_size // patch_len
        self.target_token_indices = tuple(target_token_indices or (0,))
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

    def forward(self, inputs: torch.Tensor) -> AATimeXerTokenStates:
        encoded = self.encoder_only(inputs)
        patch_states = encoded[..., :-1].permute(0, 1, 3, 2)
        global_states = encoded[..., -1:].permute(0, 1, 3, 2)
        return AATimeXerTokenStates(
            patch_states=patch_states,
            global_states=global_states,
        )

    def _select_target_channels(self, token_states: torch.Tensor) -> torch.Tensor:
        if token_states.ndim != 4:
            raise ValueError("TimeXer token states must be rank-4 [B, channel, token, hidden]")
        index_tensor = torch.as_tensor(
            self.target_token_indices,
            device=token_states.device,
            dtype=torch.long,
        )
        selected = torch.index_select(token_states, dim=1, index=index_tensor)
        return selected.mean(dim=1)

    def project_to_time_states(self, states: AATimeXerTokenStates) -> torch.Tensor:
        if not isinstance(states, AATimeXerTokenStates):
            raise TypeError("TimeXer adapter expects AATimeXerTokenStates for time projection")
        patch_tokens = self._select_target_channels(states.patch_states)
        global_tokens = self._select_target_channels(states.global_states).squeeze(1)
        template = patch_tokens.new_zeros(
            patch_tokens.shape[0],
            self.input_size,
            self.hidden_size,
        )
        return scatter_patch_tokens_to_time_states(
            patch_tokens,
            seq_len=self.input_size,
            patch_len=self.patch_len,
            stride=self.patch_len,
            template=template,
            global_context=global_tokens,
        )


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
    target_token_indices: tuple[int, ...] = (0,),
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
        target_token_indices=target_token_indices,
    )
