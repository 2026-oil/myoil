from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_encoder(
    *,
    feature_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bias: bool = True,
) -> nn.GRU:
    return nn.GRU(
        input_size=feature_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bias=bias,
        dropout=dropout if num_layers > 1 else 0.0,
    )


def _align_horizon(
    hidden: torch.Tensor,
    *,
    h: int,
    input_size: int,
    sequence_adapter: nn.Linear | None,
) -> torch.Tensor:
    if h > input_size:
        if sequence_adapter is None:
            raise ValueError("AAForecast sequence_adapter is required when h > input_size")
        hidden = hidden.permute(0, 2, 1)
        hidden = sequence_adapter(hidden)
        hidden = hidden.permute(0, 2, 1)
        return hidden
    return hidden[:, -h:]


def _apply_stochastic_dropout(
    tensor: torch.Tensor,
    *,
    training: bool,
    stochastic_inference_enabled: bool,
    train_dropout_p: float,
    inference_dropout_p: float,
) -> torch.Tensor:
    enabled = training or stochastic_inference_enabled
    if not enabled:
        return tensor
    p = train_dropout_p if training else inference_dropout_p
    if p <= 0:
        return tensor
    return F.dropout(tensor, p=p, training=True)
