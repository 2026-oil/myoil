from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from tuning.search_space import DEFAULT_RESIDUAL_PARAMS_BY_MODEL
from residual.plugins_base import (
    PluginDefinition,
    PluginRegistry,
    ResidualPlugin,
)

from plugins.residual.backends import (
    LightGBMResidualPlugin,
    RandomForestResidualPlugin,
    XGBoostResidualPlugin,
)

BACKEND_PLUGIN_CATEGORY = "backend"
PLUGIN_EXTENSION_RULES = (
    "Add residual backends under residual/models/backends/ and register them here.",
    "Residual registry is backend-only; bs_preforcast has its own top-level registry.",
)

_PLUGIN_REGISTRY = PluginRegistry(
    (
        PluginDefinition(
            category=BACKEND_PLUGIN_CATEGORY,
            name="xgboost",
            factory=XGBoostResidualPlugin,
            description="Residual correction backend implemented with xgboost.",
        ),
        PluginDefinition(
            category=BACKEND_PLUGIN_CATEGORY,
            name="randomforest",
            factory=RandomForestResidualPlugin,
            description="Residual correction backend implemented with sklearn random forest.",
        ),
        PluginDefinition(
            category=BACKEND_PLUGIN_CATEGORY,
            name="lightgbm",
            factory=LightGBMResidualPlugin,
            description="Residual correction backend implemented with lightgbm.",
        ),
    )
)


def plugin_registry() -> PluginRegistry:
    return _PLUGIN_REGISTRY


def available_plugins(category: str | None = None) -> tuple[str, ...]:
    return _PLUGIN_REGISTRY.names(category)


def build_residual_plugin(config: Any) -> ResidualPlugin:
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    name = str(config.get("model", "xgboost")).lower()
    if name not in DEFAULT_RESIDUAL_PARAMS_BY_MODEL:
        raise ValueError(f"Unsupported residual model: {name}")
    params = {**DEFAULT_RESIDUAL_PARAMS_BY_MODEL[name], **dict(config.get("params", {}))}
    cpu_threads = config.get("cpu_threads")
    return _PLUGIN_REGISTRY.create(
        BACKEND_PLUGIN_CATEGORY,
        name,
        cpu_threads=(None if cpu_threads is None else int(cpu_threads)),
        **params,
    )
