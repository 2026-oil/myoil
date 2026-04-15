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
            "route the target plus selected AA STAR channels through Informer c_in and preserve the remaining AA channels as Informer exogenous marks",
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
        signal_channel_indices: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
        validate_attention_heads(hidden_size, n_head, field_name="n_head")
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        normalized_signal = tuple(
            dict.fromkeys(
                int(index)
                for index in (
                    signal_channel_indices if signal_channel_indices is not None else (0,)
                )
            )
        )
        if not normalized_signal:
            raise ValueError("InformerBackboneAdapter requires at least one signal channel")
        invalid_signal = tuple(
            index for index in normalized_signal if index < 0 or index >= feature_size
        )
        if invalid_signal:
            raise ValueError(
                "InformerBackboneAdapter received out-of-range signal channel index/indices: "
                + ", ".join(str(index) for index in invalid_signal)
            )
        self.signal_channel_indices = normalized_signal
        self.signal_input_size = len(self.signal_channel_indices)
        self.exog_channel_indices = tuple(
            index for index in range(feature_size) if index not in self.signal_channel_indices
        )
        self.exog_input_size = len(self.exog_channel_indices)
        self.encoder_only = InformerEncoderOnly(
            c_in=self.signal_input_size,
            exog_input_size=self.exog_input_size,
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

    def _split_inputs(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        signal = inputs[..., list(self.signal_channel_indices)]
        exog = (
            inputs[..., list(self.exog_channel_indices)]
            if self.exog_input_size > 0
            else None
        )
        return signal, exog

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        signal, exog = self._split_inputs(inputs)
        return self.encoder_only(signal, exog)


def build_informer_backbone(
    *,
    feature_size: int,
    hidden_size: int,
    n_head: int,
    encoder_layers: int,
    dropout: float,
    linear_hidden_size: int | None,
    factor: int,
    signal_channel_indices: tuple[int, ...] | list[int] | None = None,
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
        signal_channel_indices=signal_channel_indices,
    )
