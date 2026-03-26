from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .search_space import rewrite_search_space_error, stage_search_space_payload

if TYPE_CHECKING:
    from residual.config import AppConfig, LoadedConfig, SearchSpaceContract

BS_PREFORCAST_KEYS = {
    "enabled",
    "config_path",
    "using_futr_exog",
    "target_columns",
    "task",
}
BS_PREFORCAST_TASK_KEYS = {"multivariable"}


@dataclass(frozen=True)
class BsPreforcastTaskConfig:
    multivariable: bool = False


@dataclass(frozen=True)
class BsPreforcastConfig:
    enabled: bool = False
    config_path: str | None = None
    using_futr_exog: bool = False
    target_columns: tuple[str, ...] = field(default_factory=tuple)
    task: BsPreforcastTaskConfig = field(default_factory=BsPreforcastTaskConfig)


@dataclass(frozen=True)
class BsPreforcastStageLoadedConfig:
    config: AppConfig
    source_path: Path
    source_type: str
    normalized_payload: dict[str, Any]
    input_hash: str
    resolved_hash: str
    search_space_path: Path | None
    search_space_hash: str | None
    search_space_payload: dict[str, Any] | None


def normalize_bs_preforcast_config(
    value: Any,
    *,
    unknown_keys: Any,
    coerce_bool: Any,
    coerce_optional_path_string: Any,
    coerce_name_tuple: Any,
) -> BsPreforcastConfig:
    if value is None:
        return BsPreforcastConfig()
    if not isinstance(value, dict):
        raise ValueError("bs_preforcast must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=BS_PREFORCAST_KEYS, section="bs_preforcast")

    task_payload = dict(payload.get("task") or {})
    if not isinstance(payload.get("task", {}), (dict, type(None))):
        raise ValueError("bs_preforcast.task must be a mapping")
    unknown_keys(
        task_payload,
        allowed=BS_PREFORCAST_TASK_KEYS,
        section="bs_preforcast.task",
    )

    enabled = coerce_bool(
        payload.get("enabled"),
        field_name="bs_preforcast.enabled",
        default=False,
    )
    using_futr_exog = coerce_bool(
        payload.get("using_futr_exog"),
        field_name="bs_preforcast.using_futr_exog",
        default=False,
    )
    config_path = coerce_optional_path_string(
        payload.get("config_path"),
        field_name="bs_preforcast.config_path",
    )
    target_columns = coerce_name_tuple(
        payload.get("target_columns"),
        field_name="bs_preforcast.target_columns",
    )
    multivariable = coerce_bool(
        task_payload.get("multivariable"),
        field_name="bs_preforcast.task.multivariable",
        default=False,
    )
    if enabled:
        if config_path is None:
            config_path = "bs_preforcast.yaml"
        if not target_columns:
            raise ValueError(
                "bs_preforcast.target_columns must be non-empty when bs_preforcast.enabled is true"
            )
    return BsPreforcastConfig(
        enabled=enabled,
        config_path=config_path,
        using_futr_exog=using_futr_exog,
        target_columns=target_columns,
        task=BsPreforcastTaskConfig(multivariable=multivariable),
    )


def resolve_bs_preforcast_route_path(
    repo_root: Path,
    bs_preforcast: BsPreforcastConfig,
) -> Path:
    selected = bs_preforcast.config_path
    if not selected:
        raise ValueError("bs_preforcast config_path did not resolve a selected config path")
    route_path = Path(selected)
    if not route_path.is_absolute():
        route_path = (repo_root / route_path).resolve()
    if not route_path.exists():
        raise FileNotFoundError(f"bs_preforcast selected config not found: {route_path}")
    return route_path


def merge_stage_payload(
    stage_payload: dict[str, Any],
    *,
    multivariable: bool,
) -> dict[str, Any]:
    if not any(key in stage_payload for key in ("common", "univariable", "multivariable")):
        return stage_payload
    common_payload = stage_payload.get("common", {})
    if common_payload is None:
        common_payload = {}
    if not isinstance(common_payload, dict):
        raise ValueError("bs_preforcast.common must be a mapping")
    variant_key = "multivariable" if multivariable else "univariable"
    variant_payload = stage_payload.get(variant_key, {})
    if variant_payload is None:
        variant_payload = {}
    if not isinstance(variant_payload, dict):
        raise ValueError(f"bs_preforcast.{variant_key} must be a mapping")
    merged = json.loads(json.dumps(common_payload))
    for key, value in variant_payload.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_bs_preforcast_stage1(
    repo_root: Path,
    *,
    source_path: Path,
    source_type: str,
    bs_preforcast: BsPreforcastConfig,
    search_space_contract: SearchSpaceContract | None,
) -> BsPreforcastStageLoadedConfig:
    from residual.config import (
        _hash_text,
        _load_document,
        _normalize_payload,
        _resolve_jobs_reference,
        _resolve_relative_config_reference,
    )

    selected_config_path = bs_preforcast.config_path
    if not bs_preforcast.enabled or selected_config_path is None:
        raise ValueError("bs_preforcast stage1 loading requires an enabled config_path")
    stage_source_path = _resolve_relative_config_reference(
        repo_root,
        source_path,
        selected_config_path,
    )
    if not stage_source_path.exists():
        raise FileNotFoundError(
            f"bs_preforcast selected route does not exist: {stage_source_path}"
        )
    stage_source_type = source_type
    if stage_source_path.suffix.lower() == ".toml":
        stage_source_type = "toml"
    elif stage_source_path.suffix.lower() in {".yaml", ".yml"}:
        stage_source_type = "yaml"
    stage_raw_text = stage_source_path.read_text(encoding="utf-8")
    raw_stage_payload = _load_document(stage_source_path, stage_source_type)
    stage_payload = merge_stage_payload(
        raw_stage_payload,
        multivariable=bs_preforcast.task.multivariable,
    )
    if stage_payload.get("bs_preforcast") not in (None, {}):
        raise ValueError("bs_preforcast routed YAML must not define its own bs_preforcast block")
    stage_search_space = stage_search_space_payload(
        search_space_contract.payload if search_space_contract else None
    )
    stage_payload = dict(stage_payload)
    stage_payload["jobs"] = _resolve_jobs_reference(
        repo_root,
        source_path=stage_source_path,
        jobs_value=stage_payload.get("jobs", []),
    )
    stage_dataset_path = Path(stage_payload.get("dataset", {}).get("path", "df.csv"))
    if stage_dataset_path.is_absolute():
        stage_base_dir = stage_source_path.parent
    else:
        repo_candidate = (repo_root / stage_dataset_path).resolve()
        local_candidate = (stage_source_path.parent / stage_dataset_path).resolve()
        stage_base_dir = (
            repo_root
            if repo_candidate.exists() or not local_candidate.exists()
            else stage_source_path.parent
        )
    try:
        stage_base_config = _normalize_payload(
            stage_payload,
            stage_base_dir,
            search_space=None,
            allow_missing_search_space=True,
            model_search_space_key="bs_preforcast_models",
            training_search_space_key="bs_preforcast_training",
            stage_scope="bs_preforcast",
        )
        stage_config = (
            _normalize_payload(
                stage_payload,
                stage_base_dir,
                search_space=stage_search_space,
                model_search_space_key="bs_preforcast_models",
                training_search_space_key="bs_preforcast_training",
                stage_scope="bs_preforcast",
            )
            if stage_search_space is not None
            else stage_base_config
        )
    except ValueError as exc:
        raise ValueError(rewrite_search_space_error(str(exc))) from exc
    stage_normalized_payload = stage_config.to_dict()
    stage_normalized_payload["bs_preforcast"] = {
        "enabled": True,
        "config_path": bs_preforcast.config_path,
        "using_futr_exog": bs_preforcast.using_futr_exog,
        "target_columns": list(bs_preforcast.target_columns),
        "task": {"multivariable": bs_preforcast.task.multivariable},
        "selected_config_path": str(stage_source_path),
    }
    if search_space_contract is not None:
        stage_normalized_payload["search_space_path"] = str(search_space_contract.path)
        stage_normalized_payload["search_space_sha256"] = search_space_contract.sha256
    resolved_text = json.dumps(
        stage_normalized_payload, sort_keys=True, ensure_ascii=False
    )
    return BsPreforcastStageLoadedConfig(
        config=stage_config,
        source_path=stage_source_path,
        source_type=stage_source_type,
        normalized_payload=stage_normalized_payload,
        input_hash=_hash_text(stage_raw_text),
        resolved_hash=_hash_text(resolved_text),
        search_space_path=search_space_contract.path if search_space_contract else None,
        search_space_hash=search_space_contract.sha256 if search_space_contract else None,
        search_space_payload=stage_search_space,
    )


def is_bs_preforcast_enabled(loaded: LoadedConfig) -> bool:
    return bool(loaded.config.bs_preforcast.enabled)


def load_bs_preforcast_stage1_config(
    repo_root: Path,
    loaded: LoadedConfig,
) -> LoadedConfig:
    from residual.config import load_app_config

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
    return {
        "enabled": loaded.config.bs_preforcast.enabled,
        "config_path": loaded.config.bs_preforcast.config_path,
        "using_futr_exog": loaded.config.bs_preforcast.using_futr_exog,
        "target_columns": list(loaded.config.bs_preforcast.target_columns),
        "multivariable": loaded.config.bs_preforcast.task.multivariable,
        "selected_config_path": loaded.config.bs_preforcast.config_path,
    }
