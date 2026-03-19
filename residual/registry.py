from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .plugins import XGBoostResidualPlugin
from .plugins_base import ResidualPlugin


def build_residual_plugin(config: Any) -> ResidualPlugin:
    if is_dataclass(config):
        config = asdict(config)
    name = str(config.get('model', 'xgboost')).lower()
    if name != 'xgboost':
        raise ValueError(f'Unsupported residual model: {name}')
    params = dict(config.get('params', {}))
    return XGBoostResidualPlugin(
        n_estimators=int(params.get('n_estimators', 32)),
        max_depth=int(params.get('max_depth', 3)),
        learning_rate=float(params.get('learning_rate', 0.01)),
        subsample=float(params.get('subsample', 1.0)),
        colsample_bytree=float(params.get('colsample_bytree', 1.0)),
    )
