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
    "target_columns",
    "task",
    "exog_columns",
}
BS_PREFORCAST_TASK_KEYS = {"multivariable"}


@dataclass(frozen=True)
class BsPreforcastTaskConfig:
    multivariable: bool = False


@dataclass(frozen=True)
class BsPreforcastConfig:
    enabled: bool = False
    config_path: str | None = None
    target_columns: tuple[str, ...] = field(default_factory=tuple)
    exog_columns: tuple[str, ...] = field(default_factory=tuple)
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

    target_columns = coerce_name_tuple(
        payload.get("target_columns"),
        field_name="bs_preforcast.target_columns",
    )
    exog_columns = coerce_name_tuple(
        payload.get("exog_columns"),
        field_name="bs_preforcast.exog_columns",
    )
    if not target_columns:
        raise ValueError(
            "bs_preforcast.target_columns must be non-empty in routed bs_preforcast config"
        )
    overlap = sorted(set(target_columns).intersection(exog_columns))
    if overlap:
        raise ValueError(
            "bs_preforcast.exog_columns cannot overlap target_columns: "
            + ", ".join(overlap)
        )
    multivariable = coerce_bool(
        task_payload.get("multivariable"),
        field_name="bs_preforcast.task.multivariable",
        default=False,
    )
    return BsPreforcastConfig(
        enabled=True,
        target_columns=target_columns,
        exog_columns=exog_columns,
        task=BsPreforcastTaskConfig(multivariable=multivariable),
    )


def _validate_plugin_jobs(jobs: list[dict[str, Any]]) -> None:
    if not jobs:
        raise ValueError(
            "bs_preforcast plugin YAML must define at least one fixed-param job"
        )
    for job in jobs:
        model_name = str(job.get("model", "")).strip()
        if model_name == "AutoARIMA":
            raise ValueError(
                "bs_preforcast stage no longer supports AutoARIMA; use ARIMA instead"
            )
        params = job.get("params", {})
        if not isinstance(params, dict) or not params:
            raise ValueError(
                "bs_preforcast plugin YAML must define only fixed-param jobs"
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
        _effective_shared_settings_for_source,
        _load_shared_settings_for_yaml_app_config,
        _uses_repo_shared_settings,
        _merge_shared_settings_into_payload,
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
        key: value for key, value in raw_stage_payload.items() if key != "bs_preforcast"
    }
    allowed_plugin_keys = {"jobs"}
    unexpected = sorted(set(routed_payload_without_owner).difference(allowed_plugin_keys))
    if unexpected:
        raise ValueError(
            "bs_preforcast plugin YAML contains unsupported key(s): "
            + ", ".join(unexpected)
        )
    main_payload = _load_document(source_path, source_type)
    if source_type == "yaml" and _uses_repo_shared_settings(repo_root, source_path):
        shared_settings_payload, _, _ = _load_shared_settings_for_yaml_app_config(
            repo_root
        )
        if shared_settings_payload is not None:
            effective_shared_settings, effective_owned_paths = (
                _effective_shared_settings_for_source(
                    repo_root, source_path, shared_settings_payload
                )
            )
            main_payload = _merge_shared_settings_into_payload(
                main_payload,
                effective_shared_settings,
                owned_paths=effective_owned_paths,
            )
    stage_payload = {
        "task": json.loads(json.dumps(main_payload.get("task", {}))),
        "dataset": json.loads(json.dumps(main_payload.get("dataset", {}))),
        "runtime": json.loads(json.dumps(main_payload.get("runtime", {}))),
        "training": json.loads(json.dumps(main_payload.get("training", {}))),
        "cv": json.loads(json.dumps(main_payload.get("cv", {}))),
        "scheduler": json.loads(json.dumps(main_payload.get("scheduler", {}))),
        "residual": json.loads(json.dumps(main_payload.get("residual", {}))),
        "jobs": _resolve_jobs_reference(
            repo_root,
            source_path=stage_source_path,
            jobs_value=routed_payload_without_owner.get("jobs", []),
        ),
    }
    _validate_plugin_jobs(stage_payload["jobs"])
    stage_payload["training"].pop("train_protocol", None)
    stage_payload["cv"].pop("n_windows", None)
    stage_payload["cv"].pop("step_size", None)
    stage_payload["dataset"]["target_col"] = routed_bs_preforcast.target_columns[0]
    if routed_bs_preforcast.task.multivariable:
        stage_payload["dataset"]["hist_exog_cols"] = list(
            dict.fromkeys(
                [
                    *routed_bs_preforcast.target_columns[1:],
                    *routed_bs_preforcast.exog_columns,
                ]
            )
        )
    else:
        stage_payload["dataset"]["hist_exog_cols"] = list(
            routed_bs_preforcast.exog_columns
        )
    stage_payload["dataset"]["futr_exog_cols"] = []
    stage_payload["dataset"]["static_exog_cols"] = []
    stage_search_space = stage_search_space_payload(
        search_space_contract.payload if search_space_contract else None
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
    if any(job.validated_mode != "learned_fixed" for job in stage_config.jobs):
        raise ValueError(
            "bs_preforcast plugin YAML must define only fixed-param jobs"
        )
    if stage_config.training_search.validated_mode != "training_fixed":
        raise ValueError(
            "bs_preforcast plugin-only mode does not support training_auto"
        )
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
        "target_columns": list(loaded.config.bs_preforcast.target_columns),
        "exog_columns": list(loaded.config.bs_preforcast.exog_columns),
        "multivariable": loaded.config.bs_preforcast.task.multivariable,
        "selected_config_path": loaded.config.bs_preforcast.config_path,
    }
