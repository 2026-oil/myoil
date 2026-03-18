from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .plugins import LSTMResidualPlugin
from .plugins_base import ResidualPlugin


def build_residual_plugin(config: Any) -> ResidualPlugin:
    if is_dataclass(config):
        config = asdict(config)
    name = str(config.get('model', 'lstm')).lower()
    if name != 'lstm':
        raise ValueError(f'Unsupported residual model: {name}')
    params = dict(config.get('params', {}))
    return LSTMResidualPlugin(
        lookback=int(params.get('lookback', 4)),
        hidden_size=int(params.get('hidden_size', 8)),
        num_layers=int(params.get('num_layers', 1)),
        epochs=int(params.get('epochs', 20)),
        learning_rate=float(params.get('learning_rate', 0.01)),
    )
