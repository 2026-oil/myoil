from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .search_space import rewrite_search_space_error, stage_search_space_payload

if TYPE_CHECKING:
    from residual.config import AppConfig, LoadedConfig, SearchSpaceContract

BS_PREFORCAST_MAIN_KEYS = {
    "enabled",
    "config_path",
}
BS_PREFORCAST_LINKED_KEYS = {
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
) -> BsPreforcastConfig:
    if value is None:
        return BsPreforcastConfig()
    if not isinstance(value, dict):
        raise ValueError("bs_preforcast must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=BS_PREFORCAST_MAIN_KEYS, section="bs_preforcast")

    enabled = coerce_bool(
        payload.get("enabled"),
        field_name="bs_preforcast.enabled",
        default=False,
    )
    config_path = coerce_optional_path_string(
        payload.get("config_path"),
        field_name="bs_preforcast.config_path",
    )
    if enabled:
        if config_path is None:
            config_path = "bs_preforcast.yaml"
    return BsPreforcastConfig(
        enabled=enabled,
        config_path=config_path,
    )


def normalize_linked_bs_preforcast_config(
    value: Any,
    *,
    unknown_keys: Any,
    coerce_bool: Any,
    coerce_name_tuple: Any,
) -> BsPreforcastConfig:
    if value is None:
        raise ValueError(
            "bs_preforcast routed YAML must define a top-level bs_preforcast block"
        )
    if not isinstance(value, dict):
        raise ValueError("bs_preforcast routed YAML block must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=BS_PREFORCAST_LINKED_KEYS, section="bs_preforcast")

    task_payload = dict(payload.get("task") or {})
    if not isinstance(payload.get("task", {}), (dict, type(None))):
        raise ValueError("bs_preforcast.task must be a mapping")
    unknown_keys(
        task_payload,
        allowed=BS_PREFORCAST_TASK_KEYS,
        section="bs_preforcast.task",
    )

    using_futr_exog = coerce_bool(
        payload.get("using_futr_exog"),
        field_name="bs_preforcast.using_futr_exog",
        default=False,
    )
    target_columns = coerce_name_tuple(
        payload.get("target_columns"),
        field_name="bs_preforcast.target_columns",
    )
    if not target_columns:
        raise ValueError(
            "bs_preforcast.target_columns must be non-empty in routed bs_preforcast config"
        )
    multivariable = coerce_bool(
        task_payload.get("multivariable"),
        field_name="bs_preforcast.task.multivariable",
        default=False,
    )
    return BsPreforcastConfig(
        enabled=True,
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
        _coerce_bool,
        _coerce_name_tuple,
        _hash_text,
        _load_document,
        _normalize_payload,
        _unknown_keys,
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
    routed_bs_preforcast = normalize_linked_bs_preforcast_config(
        raw_stage_payload.get("bs_preforcast"),
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
        coerce_name_tuple=_coerce_name_tuple,
    )
    routed_bs_preforcast = replace(
        routed_bs_preforcast,
        config_path=bs_preforcast.config_path,
    )
    routed_payload_without_owner = {
        key: value
        for key, value in raw_stage_payload.items()
        if key != "bs_preforcast"
    }
    stage_payload = merge_stage_payload(
        routed_payload_without_owner,
        multivariable=routed_bs_preforcast.task.multivariable,
    )
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
    stage_config = replace(stage_config, bs_preforcast=routed_bs_preforcast)
    stage_normalized_payload = stage_config.to_dict()
    stage_normalized_payload["bs_preforcast"]["selected_config_path"] = str(
        stage_source_path
    )
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
    if loaded.bs_preforcast_stage1 is None:
        raise ValueError("bs_preforcast stage1 config is not loaded")
    normalized = dict(loaded.bs_preforcast_stage1.normalized_payload)
    normalized["bs_preforcast_parent_config_path"] = str(loaded.source_path.resolve())
    normalized["bs_preforcast_target_columns"] = list(
        loaded.bs_preforcast_stage1.config.bs_preforcast.target_columns
    )
    return LoadedConfig(
        config=loaded.bs_preforcast_stage1.config,
        source_path=loaded.bs_preforcast_stage1.source_path,
        source_type=loaded.bs_preforcast_stage1.source_type,
        normalized_payload=normalized,
        input_hash=loaded.bs_preforcast_stage1.input_hash,
        resolved_hash=loaded.bs_preforcast_stage1.resolved_hash,
        search_space_path=loaded.bs_preforcast_stage1.search_space_path,
        search_space_hash=loaded.bs_preforcast_stage1.search_space_hash,
        search_space_payload=loaded.bs_preforcast_stage1.search_space_payload,
    )


def stage1_route_metadata(loaded: LoadedConfig) -> dict[str, object]:
    return {
        "enabled": loaded.config.bs_preforcast.enabled,
        "config_path": loaded.config.bs_preforcast.config_path,
        "using_futr_exog": loaded.config.bs_preforcast.using_futr_exog,
        "target_columns": list(loaded.config.bs_preforcast.target_columns),
        "multivariable": loaded.config.bs_preforcast.task.multivariable,
        "selected_config_path": loaded.config.bs_preforcast.config_path,
    }
