from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import (
    BsPreforcastConfig,
    BsPreforcastRoutingConfig,
    LoadedConfig,
    load_app_config,
)


def is_bs_preforcast_enabled(loaded: LoadedConfig) -> bool:
    return bool(loaded.config.bs_preforcast.enabled)


def resolve_bs_preforcast_route_path(
    repo_root: Path,
    bs_preforcast: BsPreforcastConfig,
) -> Path:
    selected = bs_preforcast.routing.selected_config_path
    if not selected:
        raise ValueError("bs_preforcast routing did not resolve a selected config path")
    route_path = Path(selected)
    if not route_path.is_absolute():
        route_path = (repo_root / route_path).resolve()
    if not route_path.exists():
        raise FileNotFoundError(f"bs_preforcast selected config not found: {route_path}")
    return route_path


def load_bs_preforcast_stage1_config(
    repo_root: Path,
    loaded: LoadedConfig,
) -> LoadedConfig:
    route_path = resolve_bs_preforcast_route_path(repo_root, loaded.config.bs_preforcast)
    stage1_loaded = load_app_config(
        repo_root,
        config_path=route_path,
        model_search_space_key="bs_preforcast_models",
        training_search_space_key="bs_preforcast_training",
    )
    normalized = dict(stage1_loaded.normalized_payload)
    normalized["bs_preforcast_parent_config_path"] = str(loaded.source_path.resolve())
    normalized["bs_preforcast_target_columns"] = list(
        loaded.config.bs_preforcast.target_columns
    )
    return replace(stage1_loaded, normalized_payload=normalized)


def stage1_route_metadata(loaded: LoadedConfig) -> dict[str, object]:
    routing: BsPreforcastRoutingConfig = loaded.config.bs_preforcast.routing
    return {
        "enabled": loaded.config.bs_preforcast.enabled,
        "using_futr_exog": loaded.config.bs_preforcast.using_futr_exog,
        "target_columns": list(loaded.config.bs_preforcast.target_columns),
        "multivariable": loaded.config.bs_preforcast.task.multivariable,
        "selected_config_path": routing.selected_config_path,
        "univariable_config": routing.univariable_config,
        "multivariable_config": routing.multivariable_config,
    }
