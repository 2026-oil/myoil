from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast
import hashlib
import json
import math
import tomllib

import yaml
from plugin_contracts.stage_registry import (
    _ensure_plugins_loaded,
    get_active_stage_plugin,
    get_stage_plugin,
    get_stage_plugin_for_payload,
)
from tuning.search_space import (
    BASELINE_MODEL_NAMES,
    DEFAULT_TRAINING_LR_SCHEDULER,
    DEFAULT_TRAINING_PARAMS,
    FIXED_TRAINING_KEYS,
    LEGACY_TRAINING_SELECTOR_TO_CONFIG_FIELD,
    ResidualMode,
    SearchSpaceContract,
    load_search_space_contract,
    normalize_search_space_payload,
    SUPPORTED_AUTO_MODEL_NAMES,
    SUPPORTED_RESIDUAL_MODELS,
)

SHARED_SETTINGS_RELATIVE_PATH = Path("yaml/setting/setting.yaml")
DEFAULT_MANIFEST_VERSION = "1"
DEFAULT_ARTIFACT_SCHEMA_VERSION = "1"
DEFAULT_EVALUATION_PROTOCOL_VERSION = "2"
SUPPORTED_LOSSES = {"mse", "exloss"}
SUPPORTED_TRAINING_OPTIMIZERS = ("adamw", "ademamix", "mars", "soap")
CENTRALIZED_TRAINING_KEYS = {
    "train_protocol",
    "input_size",
    "batch_size",
    "valid_batch_size",
    "windows_batch_size",
    "inference_windows_batch_size",
    "lr_scheduler",
    "scaler_type",
    "model_step_size",
    "max_steps",
    "val_size",
    "val_check_steps",
    "early_stop_patience_steps",
    "loss",
    "loss_params",
    "optimizer",
    "accelerator",
    "devices",
    "strategy",
    "precision",
    "dataloader_kwargs",
} | set(FIXED_TRAINING_KEYS)
FORBIDDEN_JOB_PARAM_KEYS = frozenset({"optimizer", "optimizer_kwargs"})
LEGACY_SHARED_JOB_TRAINING_KEYS = {
    "scaler_type",
    "step_size",
    "early_stop_patience_steps",
}
SHARED_SETTINGS_OWNED_DOTTED_PATHS = (
    "runtime.random_seed",
    "training.input_size",
    "training.batch_size",
    "training.valid_batch_size",
    "training.windows_batch_size",
    "training.inference_windows_batch_size",
    "training.lr_scheduler",
    "training.max_steps",
    "training.val_size",
    "training.val_check_steps",
    "training.model_step_size",
    "training.early_stop_patience_steps",
    "training.loss",
    "training.optimizer",
    "cv.gap",
    "cv.horizon",
    "cv.step_size",
    "cv.n_windows",
    "cv.max_train_size",
    "cv.overlap_eval_policy",
    "scheduler.gpu_ids",
    "scheduler.max_concurrent_jobs",
    "scheduler.worker_devices",
)
SHARED_SETTINGS_MAPPING_DOTTED_PATHS = frozenset(
    {"training.lr_scheduler", "training.optimizer"}
)
RESIDUAL_FEATURE_KEYS = {
    "include_base_prediction",
    "include_horizon_step",
    "include_date_features",
    "lag_features",
    "exog_sources",
}
RESIDUAL_LAG_FEATURE_KEYS = {"enabled", "sources", "steps", "transforms"}
RESIDUAL_EXOG_SOURCE_KEYS = {"hist", "futr", "static"}
SUPPORTED_RESIDUAL_LAG_TRANSFORMS = {"raw"}
SUPPORTED_TRAINER_ACCELERATORS = {"auto", "cpu", "gpu"}
SUPPORTED_DATALOADER_KWARGS = {
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
}
FORBIDDEN_RESIDUAL_LAG_SOURCES = {
    "y",
    "residual_target",
    "cutoff_day",
    "ds_day",
}
ResidualTargetMode = Literal["level", "delta"]
RuntimeTransformationMode = Literal["diff"]

@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    target_col: str
    dt_col: str = "dt"
    freq: str | None = None
    hist_exog_cols: tuple[str, ...] = field(default_factory=tuple)
    futr_exog_cols: tuple[str, ...] = field(default_factory=tuple)
    static_exog_cols: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int = 1
    opt_n_trial: int | None = None
    transformations_target: RuntimeTransformationMode | None = None
    transformations_exog: RuntimeTransformationMode | None = None


@dataclass(frozen=True)
class TaskConfig:
    name: str | None = None


@dataclass(frozen=True)
class TrainingLossParams:
    up_th: float = 0.9
    down_th: float = 0.1
    lamda_underestimate: float = 1.2
    lamda_overestimate: float = 1.0
    lamda: float = 1.0


@dataclass(frozen=True)
class TrainingLRSchedulerConfig:
    name: Literal["OneCycleLR"] = "OneCycleLR"
    max_lr: float = DEFAULT_TRAINING_LR_SCHEDULER["max_lr"]
    pct_start: float = DEFAULT_TRAINING_LR_SCHEDULER["pct_start"]
    div_factor: float = DEFAULT_TRAINING_LR_SCHEDULER["div_factor"]
    final_div_factor: float = DEFAULT_TRAINING_LR_SCHEDULER["final_div_factor"]
    anneal_strategy: Literal["cos", "linear"] = DEFAULT_TRAINING_LR_SCHEDULER[
        "anneal_strategy"
    ]
    three_phase: bool = DEFAULT_TRAINING_LR_SCHEDULER["three_phase"]
    cycle_momentum: bool = DEFAULT_TRAINING_LR_SCHEDULER["cycle_momentum"]


@dataclass(frozen=True)
class TrainingOptimizerConfig:
    name: Literal["adamw", "ademamix", "mars", "soap"] = "adamw"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    train_protocol: str = "expanding_window_tscv"
    input_size: int = DEFAULT_TRAINING_PARAMS["input_size"]
    batch_size: int = DEFAULT_TRAINING_PARAMS["batch_size"]
    valid_batch_size: int = DEFAULT_TRAINING_PARAMS["valid_batch_size"]
    windows_batch_size: int = DEFAULT_TRAINING_PARAMS["windows_batch_size"]
    inference_windows_batch_size: int = DEFAULT_TRAINING_PARAMS[
        "inference_windows_batch_size"
    ]
    lr_scheduler: TrainingLRSchedulerConfig = field(
        default_factory=TrainingLRSchedulerConfig
    )
    scaler_type: str | None = DEFAULT_TRAINING_PARAMS["scaler_type"]
    model_step_size: int = DEFAULT_TRAINING_PARAMS["model_step_size"]
    max_steps: int = DEFAULT_TRAINING_PARAMS["max_steps"]
    val_size: int = DEFAULT_TRAINING_PARAMS["val_size"]
    val_check_steps: int = DEFAULT_TRAINING_PARAMS["val_check_steps"]
    early_stop_patience_steps: int = DEFAULT_TRAINING_PARAMS[
        "early_stop_patience_steps"
    ]
    loss: str = "mse"
    loss_params: TrainingLossParams | None = None
    optimizer: TrainingOptimizerConfig = field(default_factory=TrainingOptimizerConfig)
    accelerator: str | None = None
    devices: int | None = None
    strategy: str | None = None
    precision: str | int | None = None
    dataloader_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSearchConfig:
    requested_mode: Literal["training_fixed", "training_auto_requested"] = (
        "training_fixed"
    )
    validated_mode: Literal["training_fixed", "training_auto"] = "training_fixed"
    selected_search_params: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CVConfig:
    horizon: int = 12
    step_size: int = 4
    n_windows: int = 24
    gap: int = 0
    max_train_size: int | None = None
    overlap_eval_policy: Literal["by_cutoff_mean"] = "by_cutoff_mean"


@dataclass(frozen=True)
class SchedulerConfig:
    gpu_ids: tuple[int, ...] = (0, 1)
    max_concurrent_jobs: int = 2
    worker_devices: int = 1
    parallelize_single_job_tuning: bool = True


@dataclass(frozen=True)
class ResidualLagFeatureConfig:
    enabled: bool = False
    sources: tuple[str, ...] = field(default_factory=tuple)
    steps: tuple[int, ...] = field(default_factory=tuple)
    transforms: tuple[str, ...] = ("raw",)


@dataclass(frozen=True)
class ResidualExogSourceConfig:
    hist: tuple[str, ...] = field(default_factory=tuple)
    futr: tuple[str, ...] = field(default_factory=tuple)
    static: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResidualFeatureConfig:
    include_base_prediction: bool = True
    include_horizon_step: bool = True
    include_date_features: bool = False
    lag_features: ResidualLagFeatureConfig = field(
        default_factory=ResidualLagFeatureConfig
    )
    exog_sources: ResidualExogSourceConfig = field(
        default_factory=ResidualExogSourceConfig
    )


@dataclass(frozen=True)
class ResidualConfig:
    enabled: bool = True
    model: str = "xgboost"
    target: ResidualTargetMode = "level"
    cpu_threads: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    features: ResidualFeatureConfig = field(default_factory=ResidualFeatureConfig)
    requested_mode: ResidualMode = "residual_fixed"
    validated_mode: ResidualMode = "residual_fixed"
    selected_search_params: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class JobConfig:
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    requested_mode: Literal["baseline_fixed", "learned_fixed", "learned_auto_requested"] = "learned_fixed"
    validated_mode: Literal["baseline_fixed", "learned_fixed", "learned_auto"] = "learned_fixed"
    selected_search_params: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AppConfig:
    task: TaskConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
    training: TrainingConfig
    training_search: TrainingSearchConfig
    cv: CVConfig
    scheduler: SchedulerConfig
    residual: ResidualConfig
    jobs: tuple[JobConfig, ...]
    stage_plugin_config: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("stage_plugin_config", None)
        if self.stage_plugin_config is not None:
            _ensure_plugins_loaded()
            result = get_active_stage_plugin(self)
            if result is not None:
                plugin, _ = result
                serialized = plugin.config_to_dict(self.stage_plugin_config)
                if serialized is not None:
                    payload[plugin.config_key] = serialized
        if self.task.name is None:
            payload.pop("task", None)
        payload["dataset"]["path"] = str(self.dataset.path)
        for key in ("transformations_target", "transformations_exog"):
            if payload["runtime"].get(key) is None:
                payload["runtime"].pop(key, None)
        payload["training_search"]["selected_search_params"] = list(
            payload["training_search"]["selected_search_params"]
        )
        if payload["training"].get("loss") != "exloss":
            payload["training"].pop("loss_params", None)
        payload["jobs"] = list(payload["jobs"])
        for job in payload["jobs"]:
            job["selected_search_params"] = list(job["selected_search_params"])
        payload["residual"]["selected_search_params"] = list(
            payload["residual"]["selected_search_params"]
        )
        feature_payload = payload["residual"].get("features")
        if feature_payload is not None:
            lag_payload = feature_payload["lag_features"]
            lag_payload["sources"] = list(lag_payload["sources"])
            lag_payload["steps"] = list(lag_payload["steps"])
            lag_payload["transforms"] = list(lag_payload["transforms"])
            exog_payload = feature_payload["exog_sources"]
            for key in ("hist", "futr", "static"):
                exog_payload[key] = list(exog_payload[key])
        return payload


@dataclass(frozen=True)
class LoadedConfig:
    config: AppConfig
    source_path: Path
    source_type: str
    normalized_payload: dict[str, Any]
    input_hash: str
    resolved_hash: str
    search_space_path: Path | None
    search_space_hash: str | None
    search_space_payload: dict[str, Any] | None
    shared_settings_path: Path | None = None
    shared_settings_hash: str | None = None
    stage_plugin_loaded: Any = None
    jobs_fanout_specs: tuple["JobsFanoutSpec", ...] = field(default_factory=tuple)
    active_jobs_route_slug: str | None = None


@dataclass(frozen=True)
class JobsFanoutSpec:
    reference: str
    resolved_path: Path
    route_slug: str
    jobs_payload: tuple[dict[str, Any], ...]
    stage_jobs_reference: str | None = None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_MISSING = object()


def _lookup_dotted_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _set_dotted_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _validate_shared_settings_fragment(
    payload: dict[str, Any],
    *,
    owned_paths: tuple[str, ...],
    section_label: str,
) -> None:
    owned = set(owned_paths)
    mapping_roots = SHARED_SETTINGS_MAPPING_DOTTED_PATHS

    def _walk(node: dict[str, Any], prefix: str = "") -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if path in owned:
                if path in mapping_roots and not isinstance(value, dict):
                    raise ValueError(f"{section_label} {path} must be a mapping")
                continue
            if any(owned_path.startswith(path + ".") for owned_path in owned):
                if not isinstance(value, dict):
                    raise ValueError(f"{section_label} {path} must be a mapping")
                _walk(value, path)
                continue
            if any(path.startswith(root + ".") for root in mapping_roots):
                continue
            raise ValueError(f"{section_label} contains unsupported key: {path}")

    _walk(payload)


def _validate_shared_settings_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("yaml/setting/setting.yaml must be a mapping")
    _validate_shared_settings_fragment(
        dict(payload),
        owned_paths=SHARED_SETTINGS_OWNED_DOTTED_PATHS,
        section_label="yaml/setting/setting.yaml",
    )
    return dict(payload)


def _load_shared_settings_from_path(
    path: Path,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    path = path.resolve()
    if not path.exists():
        return None, None, None
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    validated = _validate_shared_settings_payload(payload)
    return validated, path, _hash_text(text)


def _load_shared_settings_for_yaml_app_config(
    repo_root: Path,
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    return _load_shared_settings_from_path(repo_root / SHARED_SETTINGS_RELATIVE_PATH)


def _resolve_shared_settings_reference(
    repo_root: Path,
    shared_settings_path: str | Path,
) -> Path:
    candidate = Path(shared_settings_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _uses_repo_shared_settings(repo_root: Path, source_path: Path) -> bool:
    try:
        relative_path = source_path.resolve().relative_to(repo_root.resolve())
    except AttributeError:
        source_resolved = source_path.resolve()
        repo_resolved = repo_root.resolve()
        if not str(source_resolved).startswith(str(repo_resolved) + "/"):
            return False
        relative_path = Path(str(source_resolved)[len(str(repo_resolved)) + 1 :])
    except ValueError:
        return False
    return bool(relative_path.parts) and relative_path.parts[0] == "yaml"


def _overlay_shared_fragment(
    base_payload: dict[str, Any],
    fragment: dict[str, Any],
    *,
    owned_paths: tuple[str, ...],
) -> dict[str, Any]:
    merged = deepcopy(base_payload)
    for dotted_path in owned_paths:
        shared_value = _lookup_dotted_path(fragment, dotted_path)
        if shared_value is _MISSING:
            continue
        _set_dotted_path(merged, dotted_path, deepcopy(shared_value))
    return merged


def _effective_shared_settings_for_source(
    shared_settings_or_repo_root: dict[str, Any] | Path,
    source_path: Path | None = None,
    shared_settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if shared_settings is None:
        shared_settings_payload = cast(
            dict[str, Any], shared_settings_or_repo_root
        )
    else:
        shared_settings_payload = shared_settings
    del source_path
    effective = _overlay_shared_fragment(
        {},
        shared_settings_payload,
        owned_paths=SHARED_SETTINGS_OWNED_DOTTED_PATHS,
    )
    return effective, SHARED_SETTINGS_OWNED_DOTTED_PATHS


def _merge_shared_settings_into_payload(
    payload: dict[str, Any],
    shared_settings: dict[str, Any],
    *,
    owned_paths: tuple[str, ...] = SHARED_SETTINGS_OWNED_DOTTED_PATHS,
) -> dict[str, Any]:
    merged = deepcopy(payload)
    duplicates: list[str] = []
    for dotted_path in owned_paths:
        shared_value = _lookup_dotted_path(shared_settings, dotted_path)
        if shared_value is _MISSING:
            continue
        if _lookup_dotted_path(merged, dotted_path) is not _MISSING:
            duplicates.append(dotted_path)
            continue
        _set_dotted_path(merged, dotted_path, deepcopy(shared_value))
    if duplicates:
        raise ValueError(
            "config repeats shared setting path(s): " + ", ".join(sorted(duplicates))
        )
    return merged


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


def _unknown_keys(
    payload: dict[str, Any], *, allowed: set[str], section: str
) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(
            f"{section} contains unsupported key(s): {', '.join(unknown)}"
        )


def _coerce_bool(value: Any, *, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _coerce_name_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    names = _as_tuple(value)
    if any(not name.strip() for name in names):
        raise ValueError(f"{field_name} contains an empty value")
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"{field_name} contains duplicate value(s): {', '.join(duplicated)}"
        )
    return names


def _coerce_optional_path_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized




def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _normalize_training_loss_params(value: Any) -> TrainingLossParams:
    if value is None:
        return TrainingLossParams()
    if not isinstance(value, dict):
        raise ValueError("training.loss_params must be a mapping")
    payload = dict(value)
    allowed = {
        "up_th",
        "down_th",
        "lamda_underestimate",
        "lamda_overestimate",
        "lamda",
    }
    _unknown_keys(payload, allowed=allowed, section="training.loss_params")
    params = TrainingLossParams(
        up_th=_coerce_float(payload.get("up_th", 0.9), field_name="training.loss_params.up_th"),
        down_th=_coerce_float(payload.get("down_th", 0.1), field_name="training.loss_params.down_th"),
        lamda_underestimate=_coerce_float(
            payload.get("lamda_underestimate", 1.2),
            field_name="training.loss_params.lamda_underestimate",
        ),
        lamda_overestimate=_coerce_float(
            payload.get("lamda_overestimate", 1.0),
            field_name="training.loss_params.lamda_overestimate",
        ),
        lamda=_coerce_float(payload.get("lamda", 1.0), field_name="training.loss_params.lamda"),
    )
    if not 0 < params.down_th < params.up_th < 1:
        raise ValueError("training.loss_params thresholds must satisfy 0 < down_th < up_th < 1")
    if params.lamda_underestimate < 0:
        raise ValueError("training.loss_params.lamda_underestimate must be >= 0")
    if params.lamda_overestimate < 0:
        raise ValueError("training.loss_params.lamda_overestimate must be >= 0")
    if params.lamda < 0:
        raise ValueError("training.loss_params.lamda must be >= 0")
    return params


def _normalize_training_lr_scheduler(value: Any) -> TrainingLRSchedulerConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("training.lr_scheduler must be a mapping")
    payload = {**DEFAULT_TRAINING_LR_SCHEDULER, **dict(value)}
    _unknown_keys(
        payload,
        allowed={
            "name",
            "max_lr",
            "pct_start",
            "div_factor",
            "final_div_factor",
            "anneal_strategy",
            "three_phase",
            "cycle_momentum",
        },
        section="training.lr_scheduler",
    )
    name = str(payload.get("name", "")).strip()
    if name != "OneCycleLR":
        raise ValueError("training.lr_scheduler.name must be 'OneCycleLR'")
    max_lr = _coerce_float(payload["max_lr"], field_name="training.lr_scheduler.max_lr")
    if max_lr <= 0:
        raise ValueError("training.lr_scheduler.max_lr must be > 0")
    pct_start = _coerce_float(
        payload["pct_start"], field_name="training.lr_scheduler.pct_start"
    )
    if not 0 < pct_start < 1:
        raise ValueError("training.lr_scheduler.pct_start must satisfy 0 < value < 1")
    div_factor = _coerce_float(
        payload["div_factor"], field_name="training.lr_scheduler.div_factor"
    )
    if div_factor <= 1:
        raise ValueError("training.lr_scheduler.div_factor must be > 1")
    final_div_factor = _coerce_float(
        payload["final_div_factor"],
        field_name="training.lr_scheduler.final_div_factor",
    )
    if final_div_factor <= 1:
        raise ValueError("training.lr_scheduler.final_div_factor must be > 1")
    anneal_strategy = str(payload["anneal_strategy"]).strip().lower()
    if anneal_strategy not in {"cos", "linear"}:
        raise ValueError(
            "training.lr_scheduler.anneal_strategy must be one of: cos, linear"
        )
    return TrainingLRSchedulerConfig(
        name="OneCycleLR",
        max_lr=max_lr,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        anneal_strategy=cast(Literal["cos", "linear"], anneal_strategy),
        three_phase=_coerce_bool(
            payload["three_phase"],
            field_name="training.lr_scheduler.three_phase",
            default=False,
        ),
        cycle_momentum=_coerce_bool(
            payload["cycle_momentum"],
            field_name="training.lr_scheduler.cycle_momentum",
            default=False,
        ),
    )


def _normalize_training_optimizer(value: Any) -> TrainingOptimizerConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("training.optimizer must be a mapping")
    payload = dict(value)
    _unknown_keys(payload, allowed={"name", "kwargs"}, section="training.optimizer")
    name = str(payload.get("name", "adamw")).strip().lower()
    if name not in SUPPORTED_TRAINING_OPTIMIZERS:
        raise ValueError(
            "training.optimizer.name must be one of: "
            + ", ".join(SUPPORTED_TRAINING_OPTIMIZERS)
        )
    kwargs = payload.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("training.optimizer.kwargs must be a mapping")
    normalized_kwargs: dict[str, Any] = {}
    for key, item in kwargs.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "training.optimizer.kwargs keys must be non-empty strings"
            )
        normalized_kwargs[key] = deepcopy(item)
    return TrainingOptimizerConfig(
        name=cast(Literal["adamw", "ademamix", "mars", "soap"], name),
        kwargs=normalized_kwargs,
    )


def _coerce_positive_int_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of positive integers")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field_name} must contain only positive integers")
        if item <= 0:
            raise ValueError(f"{field_name} must contain only positive integers")
        out.append(item)
    duplicated = sorted({item for item in out if out.count(item) > 1})
    if duplicated:
        raise ValueError(
            f"{field_name} contains duplicate step(s): {', '.join(str(item) for item in duplicated)}"
        )
    return tuple(out)


def _coerce_nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return int(value)


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return int(value)


def _normalize_dataloader_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("training.dataloader_kwargs must be a mapping")
    payload = dict(value)
    _unknown_keys(
        payload,
        allowed=SUPPORTED_DATALOADER_KWARGS,
        section="training.dataloader_kwargs",
    )
    normalized: dict[str, Any] = {}
    if "num_workers" in payload:
        normalized["num_workers"] = _coerce_nonnegative_int(
            payload["num_workers"],
            field_name="training.dataloader_kwargs.num_workers",
        )
    if "pin_memory" in payload:
        normalized["pin_memory"] = _coerce_bool(
            payload["pin_memory"],
            field_name="training.dataloader_kwargs.pin_memory",
            default=False,
        )
    if "persistent_workers" in payload:
        normalized["persistent_workers"] = _coerce_bool(
            payload["persistent_workers"],
            field_name="training.dataloader_kwargs.persistent_workers",
            default=False,
        )
    if "prefetch_factor" in payload:
        normalized["prefetch_factor"] = _coerce_positive_int(
            payload["prefetch_factor"],
            field_name="training.dataloader_kwargs.prefetch_factor",
        )
    num_workers = int(normalized.get("num_workers", 0))
    if num_workers == 0 and normalized.get("persistent_workers"):
        raise ValueError(
            "training.dataloader_kwargs.persistent_workers requires num_workers > 0"
        )
    if num_workers == 0 and "prefetch_factor" in normalized:
        raise ValueError(
            "training.dataloader_kwargs.prefetch_factor requires num_workers > 0"
        )
    return normalized


def _normalize_residual_feature_config(
    value: Any,
    *,
    hist_exog_cols: tuple[str, ...],
    futr_exog_cols: tuple[str, ...],
    static_exog_cols: tuple[str, ...],
) -> ResidualFeatureConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("residual.features must be a mapping")
    features_payload = dict(value)
    _unknown_keys(
        features_payload, allowed=RESIDUAL_FEATURE_KEYS, section="residual.features"
    )

    raw_lag_payload = features_payload.get("lag_features")
    raw_exog_payload = features_payload.get("exog_sources")
    if not isinstance(raw_lag_payload, (dict, type(None))):
        raise ValueError("residual.features.lag_features must be a mapping")
    if not isinstance(raw_exog_payload, (dict, type(None))):
        raise ValueError("residual.features.exog_sources must be a mapping")
    lag_payload = dict(raw_lag_payload or {})
    exog_payload = dict(raw_exog_payload or {})
    _unknown_keys(
        lag_payload,
        allowed=RESIDUAL_LAG_FEATURE_KEYS,
        section="residual.features.lag_features",
    )
    _unknown_keys(
        exog_payload,
        allowed=RESIDUAL_EXOG_SOURCE_KEYS,
        section="residual.features.exog_sources",
    )

    hist_selected = _coerce_name_tuple(
        exog_payload.get("hist", hist_exog_cols),
        field_name="residual.features.exog_sources.hist",
    )
    futr_selected = _coerce_name_tuple(
        exog_payload.get("futr"), field_name="residual.features.exog_sources.futr"
    )
    static_selected = _coerce_name_tuple(
        exog_payload.get("static"), field_name="residual.features.exog_sources.static"
    )
    for field_name, selected, available in (
        (
            "residual.features.exog_sources.hist",
            hist_selected,
            hist_exog_cols,
        ),
        (
            "residual.features.exog_sources.futr",
            futr_selected,
            futr_exog_cols,
        ),
        (
            "residual.features.exog_sources.static",
            static_selected,
            static_exog_cols,
        ),
    ):
        unknown = sorted(set(selected).difference(available))
        if unknown:
            dataset_field = field_name.rsplit(".", 1)[-1]
            raise ValueError(
                f"{field_name} must be selected from dataset.{dataset_field}_exog_cols; "
                f"unknown value(s): {', '.join(unknown)}"
            )

    lag_enabled = _coerce_bool(
        lag_payload.get("enabled"),
        field_name="residual.features.lag_features.enabled",
        default=False,
    )
    lag_sources = _coerce_name_tuple(
        lag_payload.get("sources"), field_name="residual.features.lag_features.sources"
    )
    lag_steps = _coerce_positive_int_tuple(
        lag_payload.get("steps"), field_name="residual.features.lag_features.steps"
    )
    lag_transforms = tuple(
        item.lower()
        for item in _coerce_name_tuple(
            lag_payload.get("transforms", ("raw",)),
            field_name="residual.features.lag_features.transforms",
        )
    )
    unsupported_transforms = sorted(
        set(lag_transforms).difference(SUPPORTED_RESIDUAL_LAG_TRANSFORMS)
    )
    if unsupported_transforms:
        raise ValueError(
            "residual.features.lag_features.transforms contains unsupported transform(s): "
            + ", ".join(unsupported_transforms)
        )
    forbidden_sources = sorted(set(lag_sources).intersection(FORBIDDEN_RESIDUAL_LAG_SOURCES))
    if forbidden_sources:
        raise ValueError(
            "residual.features.lag_features.sources contains forbidden source(s): "
            + ", ".join(forbidden_sources)
        )
    allowed_lag_sources = {"y_hat_base", *hist_selected, *futr_selected}
    unknown_lag_sources = sorted(set(lag_sources).difference(allowed_lag_sources))
    if unknown_lag_sources:
        raise ValueError(
            "residual.features.lag_features.sources must be y_hat_base or selected "
            "hist/futr exog columns; unknown value(s): "
            + ", ".join(unknown_lag_sources)
        )
    if lag_enabled:
        if not lag_sources:
            raise ValueError(
                "residual.features.lag_features.sources must be non-empty when lag_features.enabled is true"
            )
        if not lag_steps:
            raise ValueError(
                "residual.features.lag_features.steps must be non-empty when lag_features.enabled is true"
            )
    elif lag_sources or lag_steps or lag_transforms != ("raw",):
        raise ValueError(
            "residual.features.lag_features.enabled must be true when sources, steps, or non-default transforms are provided"
        )

    return ResidualFeatureConfig(
        include_base_prediction=_coerce_bool(
            features_payload.get("include_base_prediction"),
            field_name="residual.features.include_base_prediction",
            default=True,
        ),
        include_horizon_step=_coerce_bool(
            features_payload.get("include_horizon_step"),
            field_name="residual.features.include_horizon_step",
            default=True,
        ),
        include_date_features=_coerce_bool(
            features_payload.get("include_date_features"),
            field_name="residual.features.include_date_features",
            default=False,
        ),
        lag_features=ResidualLagFeatureConfig(
            enabled=lag_enabled,
            sources=lag_sources,
            steps=lag_steps,
            transforms=lag_transforms,
        ),
        exog_sources=ResidualExogSourceConfig(
            hist=hist_selected,
            futr=futr_selected,
            static=static_selected,
        ),
    )

def resolve_config_path(
    repo_root: Path,
    config_path: str | Path | None = None,
    config_toml_path: str | Path | None = None,
) -> tuple[Path, str]:
    if config_toml_path is not None:
        path = Path(config_toml_path)
        if not path.is_absolute():
            path = repo_root / path
        return path, "toml"
    if config_path is not None:
        path = Path(config_path)
        if not path.is_absolute():
            path = repo_root / path
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return path, "yaml"
        if suffix == ".toml":
            return path, "toml"
        raise ValueError(f"Unsupported config extension: {path}")
    raise ValueError(
        "config path is required; pass --config/--config-path or --config-toml"
    )


def _load_document(path: Path, source_type: str) -> Any:
    text = path.read_text(encoding="utf-8")
    if source_type == "toml":
        return tomllib.loads(text)
    payload = yaml.safe_load(text)
    return {} if payload is None else payload


def _requested_job_mode(
    model_name: str, params: dict[str, Any]
) -> Literal["baseline_fixed", "learned_fixed", "learned_auto_requested"]:
    if model_name in BASELINE_MODEL_NAMES:
        return "baseline_fixed"
    if params:
        return "learned_fixed"
    return "learned_auto_requested"


def _requested_residual_mode(enabled: bool, params: dict[str, Any]) -> ResidualMode:
    if not enabled:
        return "residual_disabled"
    if params:
        return "residual_fixed"
    return "residual_auto_requested"


def _normalize_job(
    job: dict[str, Any],
    *,
    search_space: dict[str, Any] | None,
    allow_missing_search_space: bool = False,
    model_search_space_key: str = "models",
    stage_scope: str | None = None,
) -> JobConfig:
    model_name = str(job["model"])
    _ensure_plugins_loaded()
    stage_plugin = get_stage_plugin(stage_scope) if stage_scope else None
    if stage_plugin is not None:
        stage_plugin.validate_model(model_name)
    params = dict(job.get("params", {}))
    supported_auto_models = (
        stage_plugin.supported_models()
        if stage_plugin is not None
        else SUPPORTED_AUTO_MODEL_NAMES
    )
    requested_mode = _requested_job_mode(model_name, params)
    validated_mode: Literal["baseline_fixed", "learned_fixed", "learned_auto"]
    selected: tuple[str, ...] = ()
    if requested_mode == "baseline_fixed":
        validated_mode = "baseline_fixed"
    elif requested_mode == "learned_fixed":
        validated_mode = "learned_fixed"
    else:
        if model_name not in supported_auto_models:
            raise ValueError(
                f"jobs[{model_name}] uses empty params but has no supported learned_auto Optuna mapping"
            )
        search_space_models = None
        fallback_key = (
            stage_plugin.model_search_space_fallback_key()
            if stage_plugin is not None
            else None
        )
        if search_space is not None:
            search_space_models = search_space.get(model_search_space_key)
            if search_space_models is None and fallback_key is not None:
                search_space_models = search_space.get(fallback_key)
        if search_space_models is None or model_name not in search_space_models:
            if allow_missing_search_space:
                validated_mode = "learned_fixed"
                return JobConfig(
                    model=model_name,
                    params=params,
                    requested_mode=requested_mode,
                    validated_mode=validated_mode,
                    selected_search_params=selected,
                )
            raise ValueError(
                f"jobs[{model_name}] requires search_space.{model_search_space_key}.{model_name} for learned_auto execution"
            )
        selected = tuple(search_space_models[model_name])
        validated_mode = "learned_auto"
    return JobConfig(
        model=model_name,
        params=params,
        requested_mode=requested_mode,
        validated_mode=validated_mode,
        selected_search_params=selected,
    )


def _normalize_payload(
    payload: dict[str, Any],
    base_dir: Path,
    *,
    search_space: dict[str, Any] | None,
    allow_missing_search_space: bool = False,
    model_search_space_key: str = "models",
    training_search_space_key: str = "training",
    stage_scope: str | None = None,
    repo_root: Path | None = None,
    source_path: Path | None = None,
) -> AppConfig:
    task = dict(payload.get("task", {}))
    dataset = dict(payload.get("dataset", {}))
    runtime = dict(payload.get("runtime", {}))
    training = dict(payload.get("training", {}))
    training.pop("train_protocol", None)
    training.pop("season_length", None)
    if "learning_rate" in training:
        raise ValueError(
            "legacy fixed-lr training key has been removed; use training.lr_scheduler.max_lr instead"
        )
    if "num_lr_decays" in training:
        raise ValueError(
            "legacy num_lr_decays training key has been removed; learned-model scheduling is fixed through training.lr_scheduler"
        )
    for selector, field_name in LEGACY_TRAINING_SELECTOR_TO_CONFIG_FIELD.items():
        if selector in training:
            if field_name in training:
                raise ValueError(
                    f"training.{selector} and training.{field_name} cannot both be set"
                )
            training[field_name] = training.pop(selector)
    cv = dict(payload.get("cv", {}))
    if "final_holdout" in cv:
        raise ValueError(
            "cv.final_holdout has been removed; evaluation now runs entirely through config-driven TSCV folds"
        )

    scheduler = dict(payload.get("scheduler", {}))
    residual = dict(payload.get("residual", {}))
    _ensure_plugins_loaded()
    stage_plugin = (
        get_stage_plugin(stage_scope) if stage_scope else None
    ) or get_stage_plugin_for_payload(payload)
    stage_plugin_config = (
        stage_plugin.normalize_config(
            payload.get(stage_plugin.config_key),
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )
        if stage_plugin is not None
        else None
    )

    target_col = str(dataset.get("target_col", "")).strip()
    if not target_col:
        raise ValueError("dataset.target_col is required")
    hist_exog_cols = _as_tuple(dataset.get("hist_exog_cols"))
    futr_exog_cols = _as_tuple(dataset.get("futr_exog_cols"))
    static_exog_cols = _as_tuple(dataset.get("static_exog_cols"))

    dataset_path = Path(dataset.get("path", "df.csv"))
    if not dataset_path.is_absolute():
        dataset_path = (base_dir / dataset_path).resolve()

    if runtime.get("opt_n_trial") is not None:
        runtime["opt_n_trial"] = int(runtime["opt_n_trial"])
        if runtime["opt_n_trial"] <= 0:
            raise ValueError("runtime.opt_n_trial must be a positive integer")
    if "transformations" in runtime:
        raise ValueError(
            "runtime.transformations is no longer supported; "
            "use runtime.transformations_target and/or "
            "runtime.transformations_exog"
        )
    for key in ("transformations_target", "transformations_exog"):
        if key not in runtime:
            continue
        value = runtime.get(key)
        if value is None:
            runtime.pop(key, None)
        elif not isinstance(value, str):
            raise ValueError(f"runtime.{key} must be the string 'diff'")
        else:
            normalized_transformation = value.strip().lower()
            if normalized_transformation != "diff":
                raise ValueError(f"runtime.{key} must be the string 'diff'")
            runtime[key] = normalized_transformation

    training.setdefault("loss", "mse")
    loss = str(training["loss"]).lower()
    if loss not in SUPPORTED_LOSSES:
        raise ValueError(f"Unsupported common loss: {loss}")
    training["loss"] = loss
    if loss == "exloss":
        training["loss_params"] = _normalize_training_loss_params(
            training.get("loss_params")
        )
    elif "loss_params" in training:
        raise ValueError(
            "training.loss_params is only supported when training.loss == exloss"
        )
    if "accelerator" in training:
        value = training.get("accelerator")
        if value is None:
            training["accelerator"] = None
        elif not isinstance(value, str):
            raise ValueError("training.accelerator must be a string")
        else:
            normalized_accelerator = value.strip().lower()
            if normalized_accelerator not in SUPPORTED_TRAINER_ACCELERATORS:
                raise ValueError(
                    "training.accelerator must be one of: auto, cpu, gpu"
                )
            training["accelerator"] = normalized_accelerator
    if "devices" in training:
        value = training.get("devices")
        if value is None:
            training["devices"] = None
        else:
            training["devices"] = _coerce_positive_int(
                value,
                field_name="training.devices",
            )
    if "strategy" in training:
        value = training.get("strategy")
        if value is None:
            training["strategy"] = None
        elif not isinstance(value, str):
            raise ValueError("training.strategy must be a string")
        else:
            training["strategy"] = value.strip() or None
    if "precision" in training:
        value = training.get("precision")
        if value is None:
            training["precision"] = None
        elif isinstance(value, (int, str)) and not isinstance(value, bool):
            training["precision"] = value
        else:
            raise ValueError("training.precision must be an int or string")
    training["dataloader_kwargs"] = _normalize_dataloader_kwargs(
        training.get("dataloader_kwargs")
    )
    training["optimizer"] = _normalize_training_optimizer(training.get("optimizer"))
    training["lr_scheduler"] = _normalize_training_lr_scheduler(
        training.get("lr_scheduler")
    )

    scheduler.setdefault("worker_devices", 1)
    scheduler.setdefault("parallelize_single_job_tuning", True)
    scheduler["gpu_ids"] = tuple(
        int(item) for item in scheduler.get("gpu_ids", (0, 1))
    )
    scheduler["worker_devices"] = _coerce_positive_int(
        int(scheduler["worker_devices"]),
        field_name="scheduler.worker_devices",
    )
    if not scheduler["gpu_ids"]:
        raise ValueError("scheduler.gpu_ids must contain at least one GPU id")
    if scheduler["worker_devices"] > len(scheduler["gpu_ids"]):
        raise ValueError(
            "scheduler.worker_devices cannot exceed the number of configured gpu_ids"
        )
    if len(scheduler["gpu_ids"]) % scheduler["worker_devices"] != 0:
        raise ValueError(
            "scheduler.gpu_ids must be evenly divisible by scheduler.worker_devices"
        )
    scheduler["parallelize_single_job_tuning"] = _coerce_bool(
        scheduler["parallelize_single_job_tuning"],
        field_name="scheduler.parallelize_single_job_tuning",
        default=True,
    )

    residual.setdefault("enabled", True)
    residual.setdefault("model", "xgboost")
    residual.setdefault("target", "level")
    if "cpu_threads" in residual and residual["cpu_threads"] is not None:
        residual["cpu_threads"] = _coerce_positive_int(
            residual["cpu_threads"],
            field_name="residual.cpu_threads",
        )
    residual.setdefault("params", {})
    if "train_source" in residual:
        raise ValueError(
            "residual.train_source has been removed; residual training now "
            "always uses fold-specific CV residual panels"
        )
    residual_model = str(residual["model"]).lower()
    if residual_model not in SUPPORTED_RESIDUAL_MODELS:
        raise ValueError(f"Unsupported residual model: {residual_model}")
    residual_target = str(residual["target"]).lower()
    if residual_target not in {"level", "delta"}:
        raise ValueError(
            "residual.target must be one of: level, delta"
        )
    residual["model"] = residual_model
    residual["target"] = residual_target
    residual["params"] = dict(residual.get("params", {}))
    if "learning_rate" in residual["params"]:
        raise ValueError(
            "legacy residual optimizer-rate key has been removed; residual optimizer rates are now internal-only"
        )
    residual["features"] = _normalize_residual_feature_config(
        residual.get("features"),
        hist_exog_cols=hist_exog_cols,
        futr_exog_cols=futr_exog_cols,
        static_exog_cols=static_exog_cols,
    )
    residual_requested_mode = _requested_residual_mode(
        bool(residual["enabled"]), residual["params"]
    )
    residual_selected: tuple[str, ...] = ()
    if residual_requested_mode == "residual_auto_requested":
        if residual_model not in SUPPORTED_RESIDUAL_MODELS:
            raise ValueError(
                f"residual[{residual_model}] has no supported auto-tuning mapping"
            )
        if search_space is None or residual_model not in search_space["residual"]:
            if allow_missing_search_space:
                residual_validated_mode = "residual_fixed"
                residual_selected = ()
            else:
                raise ValueError(
                    f"residual[{residual_model}] requires search_space.residual.{residual_model} for auto tuning"
                )
        else:
            residual_selected = tuple(search_space["residual"][residual_model])
            residual_validated_mode = "residual_auto"
    else:
        residual_validated_mode = residual_requested_mode

    if repo_root is None or source_path is None:
        jobs_payload = _resolve_jobs_reference(
            base_dir,
            source_path=base_dir,
            jobs_value=payload.get("jobs", []),
        )
    else:
        jobs_payload = _resolve_jobs_reference(
            repo_root,
            source_path=source_path,
            jobs_value=payload.get("jobs", []),
        )

    for selector in LEGACY_SHARED_JOB_TRAINING_KEYS:
        field_name = LEGACY_TRAINING_SELECTOR_TO_CONFIG_FIELD.get(selector, selector)
        legacy_values = [
            job["params"][selector]
            for job in jobs_payload
            if selector in job["params"]
        ]
        if not legacy_values:
            continue
        canonical_value = legacy_values[0]
        if any(value != canonical_value for value in legacy_values[1:]):
            raise ValueError(
                f"jobs use conflicting legacy centralized training key values for {selector}; "
                "move one shared value under training."
            )
        if field_name in training and training[field_name] != canonical_value:
            raise ValueError(
                f"training.{field_name} conflicts with legacy jobs[*].params.{selector}; "
                "keep the shared value under training only."
            )
        training.setdefault(field_name, canonical_value)
        for job in jobs_payload:
            job["params"].pop(selector, None)

    jobs: tuple[JobConfig, ...] = tuple(
        _normalize_job(
            job,
            search_space=search_space,
            allow_missing_search_space=allow_missing_search_space,
            model_search_space_key=model_search_space_key,
            stage_scope=stage_scope,
        )
        for job in jobs_payload
    )
    if not jobs:
        raise ValueError("Config must define at least one job")
    models = [job.model for job in jobs]
    if len(models) != len(set(models)):
        raise ValueError("jobs.model values must be unique")
    normalized_jobs = cast(tuple[JobConfig, ...], jobs)
    for normalized_job in normalized_jobs:
        forbidden_optimizer_keys = FORBIDDEN_JOB_PARAM_KEYS.intersection(
            normalized_job.params
        )
        if forbidden_optimizer_keys:
            forbidden_keys = ", ".join(sorted(forbidden_optimizer_keys))
            raise ValueError(
                f"jobs[{normalized_job.model}] contains forbidden optimizer override key(s): {forbidden_keys}. "
                "Move optimizer selection under training.optimizer."
            )
        duplicated = CENTRALIZED_TRAINING_KEYS.intersection(normalized_job.params)
        if duplicated:
            duplicated_keys = ", ".join(sorted(duplicated))
            raise ValueError(
                f"jobs[{normalized_job.model}] repeats centralized training key(s): {duplicated_keys}. "
                "Move these settings under training."
            )
    training_selected = ()
    if any(job.validated_mode == "learned_auto" for job in jobs):
        training_search_payload = None
        if search_space is not None:
            training_search_payload = search_space.get(training_search_space_key)
            if training_search_payload is None and stage_plugin is not None:
                fallback = stage_plugin.training_search_space_fallback_key()
                if (
                    fallback is not None
                    and training_search_space_key != fallback
                ):
                    training_search_payload = search_space.get(fallback)
        training_selected = (
            tuple(training_search_payload["global"])
            if training_search_payload is not None
            else ()
        )
    training_requested_mode = cast(
        Literal["training_fixed", "training_auto_requested"],
        "training_auto_requested" if training_selected else "training_fixed"
    )
    training_validated_mode = cast(
        Literal["training_fixed", "training_auto"],
        "training_auto" if training_selected else "training_fixed"
    )

    return AppConfig(
        task=TaskConfig(name=str(task["name"]).strip() or None)
        if "name" in task
        else TaskConfig(),
        dataset=DatasetConfig(
            path=dataset_path,
            target_col=target_col,
            dt_col=str(dataset.get("dt_col", "dt")),
            freq=dataset.get("freq"),
            hist_exog_cols=hist_exog_cols,
            futr_exog_cols=futr_exog_cols,
            static_exog_cols=static_exog_cols,
        ),
        runtime=RuntimeConfig(**runtime),
        training=TrainingConfig(**training),
        training_search=TrainingSearchConfig(
            requested_mode=training_requested_mode,
            validated_mode=training_validated_mode,
            selected_search_params=training_selected,
        ),
        cv=CVConfig(**cv),
        scheduler=SchedulerConfig(**scheduler),
        residual=ResidualConfig(
            **residual,
            requested_mode=residual_requested_mode,
            validated_mode=residual_validated_mode,
            selected_search_params=residual_selected,
        ),
        jobs=jobs,
        stage_plugin_config=stage_plugin_config,
    )


def _resolve_relative_config_reference(
    repo_root: Path,
    source_path: Path,
    reference: str,
) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate.resolve()
    repo_candidate = (repo_root / candidate).resolve()
    local_candidate = (source_path.parent / candidate).resolve()
    if local_candidate.exists():
        return local_candidate
    return repo_candidate


def _load_jobs_from_path(jobs_path: Path) -> list[Any]:
    suffix = jobs_path.suffix.lower()
    if suffix == ".toml":
        jobs_source_type = "toml"
    elif suffix in {".yaml", ".yml"}:
        jobs_source_type = "yaml"
    else:
        raise ValueError(
            "jobs route must reference a yaml, yml, or toml file: "
            f"{jobs_path}"
        )
    jobs_payload = _load_document(jobs_path, jobs_source_type)
    resolved_jobs: Any
    if isinstance(jobs_payload, list):
        resolved_jobs = jobs_payload
    elif isinstance(jobs_payload, dict):
        resolved_jobs = jobs_payload.get("jobs")
    else:
        resolved_jobs = None
    if not isinstance(resolved_jobs, list):
        raise ValueError(
            f"jobs route must resolve to a list or a mapping with 'jobs': {jobs_path}"
        )
    return resolved_jobs


def _resolve_jobs_path_reference(
    repo_root: Path,
    *,
    source_path: Path,
    jobs_value: str,
) -> tuple[Path, list[Any]]:
    jobs_path = _resolve_relative_config_reference(
        repo_root,
        source_path,
        jobs_value,
    )
    if not jobs_path.exists():
        raise FileNotFoundError(f"jobs route does not exist: {jobs_path}")
    return jobs_path, _load_jobs_from_path(jobs_path)


def _jobs_fanout_specs_from_reference(
    repo_root: Path,
    *,
    source_path: Path,
    reference: str,
    seen_route_slugs: set[str],
) -> list[JobsFanoutSpec]:
    resolved_path, jobs_payload = _resolve_jobs_path_reference(
        repo_root,
        source_path=source_path,
        jobs_value=reference,
    )
    if all(isinstance(job, dict) for job in jobs_payload):
        route_slug = resolved_path.stem
        if route_slug in seen_route_slugs:
            raise ValueError(
                "jobs path list must produce unique filename stems; "
                f"duplicate stem: {route_slug}"
            )
        seen_route_slugs.add(route_slug)
        return [
            JobsFanoutSpec(
                reference=reference,
                resolved_path=resolved_path,
                route_slug=route_slug,
                jobs_payload=tuple(dict(job) for job in jobs_payload),
            )
        ]
    if all(isinstance(job, str) for job in jobs_payload):
        nested_specs: list[JobsFanoutSpec] = []
        for nested_reference in jobs_payload:
            assert isinstance(nested_reference, str)
            nested_specs.extend(
                _jobs_fanout_specs_from_reference(
                    repo_root,
                    source_path=resolved_path,
                    reference=nested_reference,
                    seen_route_slugs=seen_route_slugs,
                )
            )
        return nested_specs
    raise ValueError(
        "jobs route must resolve to either inline job mappings or repo-relative "
        f"path strings: {resolved_path}"
    )


def _resolve_jobs_fanout_specs(
    repo_root: Path,
    *,
    source_path: Path,
    jobs_value: Any,
) -> tuple[JobsFanoutSpec, ...]:
    if not isinstance(jobs_value, list):
        return ()
    if not jobs_value:
        return ()
    if all(isinstance(item, dict) for item in jobs_value):
        return ()
    if not all(isinstance(item, str) for item in jobs_value):
        raise ValueError(
            "jobs list must contain either inline job mappings or repo-relative path strings"
        )
    specs: list[JobsFanoutSpec] = []
    seen_route_slugs: set[str] = set()
    for reference in jobs_value:
        assert isinstance(reference, str)
        specs.extend(
            _jobs_fanout_specs_from_reference(
                repo_root,
                source_path=source_path,
                reference=reference,
                seen_route_slugs=seen_route_slugs,
            )
        )
    return tuple(specs)


def _resolve_jobs_reference(
    repo_root: Path,
    *,
    source_path: Path,
    jobs_value: Any,
) -> list[dict[str, Any]]:
    if isinstance(jobs_value, list):
        if not jobs_value:
            return []
        if all(isinstance(item, dict) for item in jobs_value):
            return jobs_value
        if all(isinstance(item, str) for item in jobs_value):
            # Fanout path: load_app_config resolves multi-file jobs via
            # _resolve_jobs_fanout_specs; here we only need the first route's
            # jobs for _normalize_payload's validation pass.
            _, resolved_jobs = _resolve_jobs_path_reference(
                repo_root,
                source_path=source_path,
                jobs_value=jobs_value[0],
            )
            return resolved_jobs
        raise ValueError(
            "jobs list must contain either inline job mappings or repo-relative path strings"
        )
    if jobs_value is None:
        return []
    if isinstance(jobs_value, str):
        _, resolved_jobs = _resolve_jobs_path_reference(
            repo_root,
            source_path=source_path,
            jobs_value=jobs_value,
        )
        return resolved_jobs
    raise ValueError(
        "jobs must be an inline list, a repo-relative yaml path string, or a list of repo-relative yaml path strings"
    )


def _stage_payload_requests_search_space(
    repo_root: Path,
    *,
    source_path: Path,
    stage_payload: dict[str, Any],
) -> bool:
    for job in _resolve_jobs_reference(
        repo_root,
        source_path=source_path,
        jobs_value=stage_payload.get("jobs", []),
    ):
        if str(job.get("model")) not in BASELINE_MODEL_NAMES and not dict(
            job.get("params", {})
        ):
            return True
    residual_payload = dict(stage_payload.get("residual", {}))
    if residual_payload.get("enabled", True) and not dict(
        residual_payload.get("params", {})
    ):
        return True
    return False


def _resolve_dataset_base_dir(
    repo_root: Path,
    *,
    source_path: Path,
    payload: dict[str, Any],
) -> Path:
    dataset_path = Path(payload.get("dataset", {}).get("path", "df.csv"))
    if dataset_path.is_absolute():
        return source_path.parent
    repo_candidate = (repo_root / dataset_path).resolve()
    local_candidate = (source_path.parent / dataset_path).resolve()
    return (
        repo_root
        if repo_candidate.exists() or not local_candidate.exists()
        else source_path.parent
    )


def load_app_config(
    repo_root: Path,
    *,
    config_path: str | Path | None = None,
    config_toml_path: str | Path | None = None,
    shared_settings_path: str | Path | None = None,
    model_search_space_key: str = "models",
    training_search_space_key: str = "training",
) -> LoadedConfig:
    source_path, source_type = resolve_config_path(
        repo_root,
        config_path=config_path,
        config_toml_path=config_toml_path,
    )
    raw_text = source_path.read_text(encoding="utf-8")
    payload = _load_document(source_path, source_type)
    shared_settings_payload: dict[str, Any] | None = None
    resolved_shared_settings_path: Path | None = None
    shared_settings_hash: str | None = None
    if shared_settings_path is not None:
        explicit_shared_settings_path = _resolve_shared_settings_reference(
            repo_root, shared_settings_path
        )
        (
            shared_settings_payload,
            resolved_shared_settings_path,
            shared_settings_hash,
        ) = _load_shared_settings_from_path(explicit_shared_settings_path)
        if resolved_shared_settings_path is None:
            raise FileNotFoundError(
                "Shared settings file does not exist: "
                f"{explicit_shared_settings_path}"
            )
    elif source_type == "yaml" and _uses_repo_shared_settings(repo_root, source_path):
        (
            shared_settings_payload,
            resolved_shared_settings_path,
            shared_settings_hash,
        ) = _load_shared_settings_for_yaml_app_config(repo_root)
    if shared_settings_payload is not None:
        effective_shared_settings, effective_owned_paths = (
            _effective_shared_settings_for_source(shared_settings_payload)
        )
        payload = _merge_shared_settings_into_payload(
            payload,
            effective_shared_settings,
            owned_paths=effective_owned_paths,
        )
    jobs_fanout_specs = _resolve_jobs_fanout_specs(
        repo_root,
        source_path=source_path,
        jobs_value=payload.get("jobs", []),
    )
    _ensure_plugins_loaded()
    stage_plugin = get_stage_plugin_for_payload(payload)
    stage_source_path: Path | None = None
    stage_payload_probe: dict[str, Any] | None = None
    if stage_plugin is not None:
        raw_stage_config = stage_plugin.normalize_config(
            payload.get(stage_plugin.config_key),
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )
        if stage_plugin.is_enabled(raw_stage_config):
            stage_payload_probe = stage_plugin.validate_route(
                repo_root,
                source_path,
                raw_stage_config,
                load_document=_load_document,
                unknown_keys=_unknown_keys,
                coerce_bool=_coerce_bool,
                coerce_name_tuple=_coerce_name_tuple,
            )
    search_space_contract: SearchSpaceContract | None = None
    requested_search_space = False
    payload = dict(payload)
    if jobs_fanout_specs:
        payload["jobs"] = [dict(job) for job in jobs_fanout_specs[0].jobs_payload]
        jobs_payload_candidates = [
            [dict(job) for job in spec.jobs_payload] for spec in jobs_fanout_specs
        ]
    else:
        payload["jobs"] = _resolve_jobs_reference(
            repo_root,
            source_path=source_path,
            jobs_value=payload.get("jobs", []),
        )
        jobs_payload_candidates = [payload["jobs"]]
        stage_fanout_probe = (
            tuple(stage_payload_probe.get("jobs_fanout_specs", ()))
            if isinstance(stage_payload_probe, dict)
            else ()
        )
        if stage_fanout_probe:
            jobs_fanout_specs = tuple(
                JobsFanoutSpec(
                    reference=spec.reference,
                    resolved_path=spec.resolved_path,
                    route_slug=spec.route_slug,
                    jobs_payload=tuple(dict(job) for job in payload["jobs"]),
                    stage_jobs_reference=(
                        str(spec.resolved_path.relative_to(repo_root.resolve()))
                        if str(spec.resolved_path).startswith(str(repo_root.resolve()) + "/")
                        else str(spec.resolved_path)
                    ),
                )
                for spec in stage_fanout_probe
            )
            jobs_payload_candidates = [
                [dict(job) for job in spec.jobs_payload] for spec in jobs_fanout_specs
            ]
    for jobs_candidate in jobs_payload_candidates:
        for job in jobs_candidate:
            if str(job.get("model")) not in BASELINE_MODEL_NAMES and not dict(job.get("params", {})):
                requested_search_space = True
                break
        if requested_search_space:
            break
    residual_payload = dict(payload.get("residual", {}))
    if (
        residual_payload.get("enabled", True)
        and not dict(residual_payload.get("params", {}))
    ):
        requested_search_space = True
    if stage_payload_probe is not None:
        stage_jobs = stage_payload_probe.get("jobs", [])
        for sjob in stage_jobs:
            if str(sjob.get("model")) not in BASELINE_MODEL_NAMES and not dict(
                sjob.get("params", {})
            ):
                requested_search_space = True
                break
    if requested_search_space:
        search_space_contract = load_search_space_contract(repo_root)
    search_space = (
        normalize_search_space_payload(search_space_contract.payload)
        if search_space_contract is not None
        else None
    )
    dataset_base_dir = _resolve_dataset_base_dir(
        repo_root,
        source_path=source_path,
        payload=payload,
    )
    config = _normalize_payload(
        payload,
        dataset_base_dir,
        search_space=search_space,
        allow_missing_search_space=(search_space is None),
        model_search_space_key=model_search_space_key,
        training_search_space_key=training_search_space_key,
    )
    stage_plugin_loaded = None
    if (
        stage_plugin is not None
        and config.stage_plugin_config is not None
        and stage_plugin.is_enabled(config.stage_plugin_config)
        and getattr(config.stage_plugin_config, "config_path", None)
    ):
        stage_source_path = _resolve_relative_config_reference(
            repo_root,
            source_path,
            config.stage_plugin_config.config_path,
        )
        if stage_source_path.exists():
            stage_search_space_contract = search_space_contract
            if stage_search_space_contract is None:
                candidate = (repo_root / "yaml/HPO/search_space.yaml").resolve()
                if candidate.exists():
                    stage_search_space_contract = load_search_space_contract(repo_root)
            stage_plugin_loaded = stage_plugin.load_stage(
                repo_root,
                source_path=source_path,
                source_type=source_type,
                config=config.stage_plugin_config,
                search_space_contract=stage_search_space_contract,
            )
            config = stage_plugin.apply_stage_to_config(config, stage_plugin_loaded)
            if search_space_contract is None:
                search_space_contract = stage_search_space_contract
        elif Path(config.stage_plugin_config.config_path).is_absolute():
            raise FileNotFoundError(
                f"Stage plugin selected route does not exist: {stage_source_path}"
            )
    normalized_payload = config.to_dict()
    if stage_plugin is not None and stage_plugin_loaded is not None:
        for key, value in stage_plugin.stage_normalized_payload(
            config, stage_plugin_loaded
        ).items():
            normalized_payload.setdefault(key, {}).update(value)
    normalized_payload["search_space_path"] = (
        str(search_space_contract.path) if search_space_contract else None
    )
    normalized_payload["search_space_sha256"] = (
        search_space_contract.sha256 if search_space_contract else None
    )
    normalized_payload["shared_settings_path"] = (
        str(resolved_shared_settings_path)
        if resolved_shared_settings_path is not None
        else None
    )
    normalized_payload["shared_settings_sha256"] = shared_settings_hash
    resolved_text = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False)
    return LoadedConfig(
        config=config,
        source_path=source_path,
        source_type=source_type,
        normalized_payload=normalized_payload,
        input_hash=_hash_text(raw_text),
        resolved_hash=_hash_text(resolved_text),
        search_space_path=search_space_contract.path if search_space_contract else None,
        search_space_hash=search_space_contract.sha256 if search_space_contract else None,
        search_space_payload=search_space_contract.payload if search_space_contract else None,
        shared_settings_path=resolved_shared_settings_path,
        shared_settings_hash=shared_settings_hash,
        stage_plugin_loaded=stage_plugin_loaded,
        jobs_fanout_specs=jobs_fanout_specs,
    )


def loaded_config_for_jobs_fanout(
    repo_root: Path,
    loaded: LoadedConfig,
    spec: JobsFanoutSpec,
    *,
    model_search_space_key: str = "models",
    training_search_space_key: str = "training",
) -> LoadedConfig:
    base_keys = {
        "task", "dataset", "runtime", "training", "cv",
        "scheduler", "residual", "jobs",
    }
    _ensure_plugins_loaded()
    stage_plugin = get_stage_plugin_for_payload(loaded.normalized_payload)
    if stage_plugin is not None:
        base_keys |= stage_plugin.fanout_config_keys()
    payload = {
        key: deepcopy(value)
        for key, value in loaded.normalized_payload.items()
        if key in base_keys
    }
    if stage_plugin is not None:
        sp_key = stage_plugin.config_key
        if isinstance(payload.get(sp_key), dict):
            payload[sp_key] = stage_plugin.fanout_filter_payload(payload[sp_key])
    if isinstance(payload.get("residual"), dict):
        payload["residual"] = {
            key: value
            for key, value in payload["residual"].items()
            if key
            in {
                "enabled",
                "model",
                "target",
                "cpu_threads",
                "params",
                "features",
            }
        }
    payload["jobs"] = [dict(job) for job in spec.jobs_payload]
    dataset_base_dir = _resolve_dataset_base_dir(
        repo_root,
        source_path=loaded.source_path,
        payload=payload,
    )
    config = _normalize_payload(
        payload,
        dataset_base_dir,
        search_space=loaded.search_space_payload,
        model_search_space_key=model_search_space_key,
        training_search_space_key=training_search_space_key,
        repo_root=repo_root,
        source_path=loaded.source_path,
    )
    if (
        spec.stage_jobs_reference is not None
        and getattr(config, "stage_plugin_config", None) is not None
        and hasattr(config.stage_plugin_config, "jobs_config_path")
    ):
        config = replace(
            config,
            stage_plugin_config=replace(
                config.stage_plugin_config,
                jobs_config_path=spec.stage_jobs_reference,
            ),
        )
    stage_plugin_loaded = loaded.stage_plugin_loaded
    if (
        stage_plugin is not None
        and config.stage_plugin_config is not None
        and stage_plugin.is_enabled(config.stage_plugin_config)
        and getattr(config.stage_plugin_config, "config_path", None)
    ):
        stage_search_space_contract = (
            SearchSpaceContract(
                path=loaded.search_space_path,
                payload=loaded.search_space_payload,
                sha256=loaded.search_space_hash,
            )
            if loaded.search_space_path is not None
            and loaded.search_space_hash is not None
            and loaded.search_space_payload is not None
            else None
        )
        stage_source_path = _resolve_relative_config_reference(
            repo_root,
            loaded.source_path,
            config.stage_plugin_config.config_path,
        )
        if stage_source_path.exists():
            stage_plugin_loaded = stage_plugin.load_stage(
                repo_root,
                source_path=loaded.source_path,
                source_type=loaded.source_type,
                config=config.stage_plugin_config,
                search_space_contract=stage_search_space_contract,
            )
            config = stage_plugin.apply_stage_to_config(config, stage_plugin_loaded)
    normalized_payload = config.to_dict()
    if stage_plugin is not None:
        if stage_plugin_loaded is not None:
            extra = stage_plugin.stage_normalized_payload(config, stage_plugin_loaded)
        else:
            extra = stage_plugin.fanout_stage_payload(loaded)
        if extra is not None:
            for k, v in extra.items():
                normalized_payload.setdefault(k, {}).update(v)
    normalized_payload["search_space_path"] = (
        str(loaded.search_space_path) if loaded.search_space_path else None
    )
    normalized_payload["search_space_sha256"] = loaded.search_space_hash
    normalized_payload["shared_settings_path"] = (
        str(loaded.shared_settings_path) if loaded.shared_settings_path else None
    )
    normalized_payload["shared_settings_sha256"] = loaded.shared_settings_hash
    resolved_text = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False)
    return replace(
        loaded,
        config=config,
        normalized_payload=normalized_payload,
        resolved_hash=_hash_text(resolved_text),
        stage_plugin_loaded=stage_plugin_loaded,
        jobs_fanout_specs=(),
        active_jobs_route_slug=spec.route_slug,
    )
