from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .plugins import XGBoostResidualPlugin
from .optuna_spaces import DEFAULT_RESIDUAL_PARAMS
from .plugins_base import ResidualPlugin


def build_residual_plugin(config: Any) -> ResidualPlugin:
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    name = str(config.get("model", "xgboost")).lower()
    if name != "xgboost":
        raise ValueError(f"Unsupported residual model: {name}")
    params = {**DEFAULT_RESIDUAL_PARAMS, **dict(config.get("params", {}))}
    return XGBoostResidualPlugin(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
    )
