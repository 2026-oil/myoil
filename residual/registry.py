from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .plugins import (
    LightGBMResidualPlugin,
    RandomForestResidualPlugin,
    XGBoostResidualPlugin,
)
from .optuna_spaces import DEFAULT_RESIDUAL_PARAMS_BY_MODEL
from .plugins_base import ResidualPlugin


def build_residual_plugin(config: Any) -> ResidualPlugin:
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    name = str(config.get("model", "xgboost")).lower()
    if name not in DEFAULT_RESIDUAL_PARAMS_BY_MODEL:
        raise ValueError(f"Unsupported residual model: {name}")
    params = {**DEFAULT_RESIDUAL_PARAMS_BY_MODEL[name], **dict(config.get("params", {}))}
    if name == "xgboost":
        return XGBoostResidualPlugin(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
        )
    if name == "randomforest":
        return RandomForestResidualPlugin(
            n_estimators=int(params["n_estimators"]),
            max_depth=(None if params["max_depth"] is None else int(params["max_depth"])),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
        )
    if name == "lightgbm":
        return LightGBMResidualPlugin(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            feature_fraction=float(params["feature_fraction"]),
        )
    raise ValueError(f"Unsupported residual model: {name}")
