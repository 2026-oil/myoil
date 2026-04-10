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
    SearchSpaceContract,
    load_search_space_contract,
    normalize_search_space_payload,
    SUPPORTED_MODEL_AUTO_MODEL_NAMES,
    SUPPORTED_AUTO_MODEL_NAMES,
)

SHARED_SETTINGS_RELATIVE_PATH = Path("yaml/setting/setting.yaml")
DEFAULT_MANIFEST_VERSION = "1"
DEFAULT_ARTIFACT_SCHEMA_VERSION = "1"
DEFAULT_EVALUATION_PROTOCOL_VERSION = "2"
SUPPORTED_LOSSES = {
    "mae",
    "mse",
    "exloss",
    "latehorizonweightedmape",
    "huberlatemape",
    "quantilelatemape",
}
LOSSES_WITH_PARAMS = frozenset(
    {
        "exloss",
        "latehorizonweightedmape",
        "huberlatemape",
        "quantilelatemape",
    }
)
SUPPORTED_TRAINING_OPTIMIZERS = (
    "adamw",
    "ademamix",
    "mars",
    "soap",
    "rmsprop",
    "radam",
)
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
    "min_steps_before_early_stop",
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
    "runtime.opt_n_trial",
    "runtime.opt_study_count",
    "runtime.opt_selected_study",
    "runtime.transformations_target",
    "runtime.transformations_exog",
    "training.input_size",
    "training.batch_size",
    "training.valid_batch_size",
    "training.windows_batch_size",
    "training.inference_windows_batch_size",
    "training.lr_scheduler",
    "training.scaler_type",
    "training.max_steps",
    "training.val_size",
    "training.val_check_steps",
    "training.min_steps_before_early_stop",
    "training.model_step_size",
    "training.early_stop_patience_steps",
    "training.loss",
    "training.loss_params",
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
SUPPORTED_TRAINER_ACCELERATORS = {"auto", "cpu", "gpu"}
SUPPORTED_DATALOADER_KWARGS = {
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
}
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
    opt_study_count: int = 1
    opt_selected_study: int | None = None
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
    horizon: int = 8
    ramp_power: float = 1.5
    base_weight: float = 1.0
    late_multiplier: float = 3.0
    late_start: int = 6
    delta: float = 1.0
    late_weight: float = 3.0
    q_under: float = 0.7
    q_over: float = 0.3
    late_factor: float = 2.0


@dataclass(frozen=True)
class TrainingLRSchedulerConfig:
    name: Literal["OneCycleLR", "ReduceLROnPlateau"] = cast(
        Literal["OneCycleLR", "ReduceLROnPlateau"],
        DEFAULT_TRAINING_LR_SCHEDULER["name"],
    )
    max_lr: float = DEFAULT_TRAINING_LR_SCHEDULER["max_lr"]
    pct_start: float | None = None
    div_factor: float | None = None
    final_div_factor: float | None = None
    anneal_strategy: Literal["cos", "linear"] | None = None
    three_phase: bool | None = None
    cycle_momentum: bool | None = None
    mode: Literal["min", "max"] | None = cast(
        Literal["min", "max"] | None,
        DEFAULT_TRAINING_LR_SCHEDULER.get("mode"),
    )
    factor: float | None = DEFAULT_TRAINING_LR_SCHEDULER.get("factor")
    patience: int | None = DEFAULT_TRAINING_LR_SCHEDULER.get("patience")
    threshold: float | None = DEFAULT_TRAINING_LR_SCHEDULER.get("threshold")
    threshold_mode: Literal["rel", "abs"] | None = cast(
        Literal["rel", "abs"] | None,
        DEFAULT_TRAINING_LR_SCHEDULER.get("threshold_mode"),
    )
    cooldown: int | None = DEFAULT_TRAINING_LR_SCHEDULER.get("cooldown")
    min_lr: float | None = DEFAULT_TRAINING_LR_SCHEDULER.get("min_lr")
    eps: float | None = DEFAULT_TRAINING_LR_SCHEDULER.get("eps")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "max_lr": self.max_lr,
        }
        if self.name == "OneCycleLR":
            payload.update(
                {
                    "pct_start": self.pct_start,
                    "div_factor": self.div_factor,
                    "final_div_factor": self.final_div_factor,
                    "anneal_strategy": self.anneal_strategy,
                    "three_phase": self.three_phase,
                    "cycle_momentum": self.cycle_momentum,
                }
            )
        elif self.name == "ReduceLROnPlateau":
            payload.update(
                {
                    "mode": self.mode,
                    "factor": self.factor,
                    "patience": self.patience,
                    "threshold": self.threshold,
                    "threshold_mode": self.threshold_mode,
                    "cooldown": self.cooldown,
                    "min_lr": self.min_lr,
                    "eps": self.eps,
                }
            )
        return payload


@dataclass(frozen=True)
class TrainingOptimizerConfig:
    name: Literal["adamw", "ademamix", "mars", "soap", "rmsprop", "radam"] = "adamw"
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
    min_steps_before_early_stop: int = DEFAULT_TRAINING_PARAMS[
        "min_steps_before_early_stop"
    ]
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


def _normalize_training_search_config(value: Any) -> dict[str, bool]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("training_search must be a mapping")
    payload = dict(value)
    _unknown_keys(payload, allowed={"enabled"}, section="training_search")
    return {
        "enabled": _coerce_bool(
            payload.get("enabled", True),
            field_name="training_search.enabled",
            default=True,
        )
    }


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
class JobConfig:
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    requested_mode: Literal[
        "baseline_fixed", "learned_fixed", "learned_auto_requested"
    ] = "learned_fixed"
    validated_mode: Literal["baseline_fixed", "learned_fixed", "learned_auto"] = (
        "learned_fixed"
    )
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
        payload["training"]["lr_scheduler"] = self.training.lr_scheduler.to_dict()
        if payload["runtime"].get("opt_selected_study") is None:
            payload["runtime"].pop("opt_selected_study", None)
        for key in ("transformations_target", "transformations_exog"):
            if payload["runtime"].get(key) is None:
                payload["runtime"].pop(key, None)
        payload["training_search"]["selected_search_params"] = list(
            payload["training_search"]["selected_search_params"]
        )
        if payload["training"].get("loss") not in LOSSES_WITH_PARAMS:
            payload["training"].pop("loss_params", None)
        payload["jobs"] = list(payload["jobs"])
        for job in payload["jobs"]:
            job["selected_search_params"] = list(job["selected_search_params"])
        return payload


def _resolve_aa_forecast_stage_plugin_config(config: AppConfig) -> AppConfig:
    result = get_active_stage_plugin(config)
    if result is None:
        return config
    plugin, stage_config = result
    if plugin.config_key != "aa_forecast":
        return config

    from plugins.aa_forecast.config import (
        AAForecastPluginConfig,
        resolve_aa_forecast_hist_exog,
    )

    if not isinstance(stage_config, AAForecastPluginConfig):
        return config
    resolved = resolve_aa_forecast_hist_exog(
        stage_config,
        hist_exog_cols=config.dataset.hist_exog_cols,
    )
    if resolved == stage_config:
        return config
    return replace(config, stage_plugin_config=resolved)


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
        shared_settings_payload = cast(dict[str, Any], shared_settings_or_repo_root)
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


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _reject_legacy_or_mixed_aaforecast_jobs(payload: dict[str, Any]) -> None:
    jobs_value = payload.get("jobs")
    if not isinstance(jobs_value, list) or not jobs_value:
        return
    models = [
        str(job.get("model")).strip()
        for job in jobs_value
        if isinstance(job, dict) and job.get("model") is not None
    ]
    if "AAForecast" in models:
        raise ValueError(
            "Direct top-level AAForecast jobs are no longer supported; use aa_forecast.enabled with aa_forecast.config_path"
        )
    aa_forecast_payload = payload.get("aa_forecast")
    if isinstance(aa_forecast_payload, dict) and bool(aa_forecast_payload.get("enabled")):
        raise ValueError(
            "aa_forecast plugin route cannot be combined with top-level jobs; split AAForecast into its own plugin-routed config"
        )


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in value)


def _unknown_keys(payload: dict[str, Any], *, allowed: set[str], section: str) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"{section} contains unsupported key(s): {', '.join(unknown)}")


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
        "horizon",
        "ramp_power",
        "base_weight",
        "late_multiplier",
        "late_start",
        "delta",
        "late_weight",
        "q_under",
        "q_over",
        "late_factor",
    }
    _unknown_keys(payload, allowed=allowed, section="training.loss_params")
    params = TrainingLossParams(
        up_th=_coerce_float(
            payload.get("up_th", 0.9), field_name="training.loss_params.up_th"
        ),
        down_th=_coerce_float(
            payload.get("down_th", 0.1), field_name="training.loss_params.down_th"
        ),
        lamda_underestimate=_coerce_float(
            payload.get("lamda_underestimate", 1.2),
            field_name="training.loss_params.lamda_underestimate",
        ),
        lamda_overestimate=_coerce_float(
            payload.get("lamda_overestimate", 1.0),
            field_name="training.loss_params.lamda_overestimate",
        ),
        lamda=_coerce_float(
            payload.get("lamda", 1.0), field_name="training.loss_params.lamda"
        ),
        horizon=_coerce_nonnegative_int(
            payload.get("horizon", 8), field_name="training.loss_params.horizon"
        ),
        ramp_power=_coerce_float(
            payload.get("ramp_power", 1.5), field_name="training.loss_params.ramp_power"
        ),
        base_weight=_coerce_float(
            payload.get("base_weight", 1.0),
            field_name="training.loss_params.base_weight",
        ),
        late_multiplier=_coerce_float(
            payload.get("late_multiplier", 3.0),
            field_name="training.loss_params.late_multiplier",
        ),
        late_start=_coerce_nonnegative_int(
            payload.get("late_start", 6), field_name="training.loss_params.late_start"
        ),
        delta=_coerce_float(
            payload.get("delta", 1.0), field_name="training.loss_params.delta"
        ),
        late_weight=_coerce_float(
            payload.get("late_weight", 3.0),
            field_name="training.loss_params.late_weight",
        ),
        q_under=_coerce_float(
            payload.get("q_under", 0.7), field_name="training.loss_params.q_under"
        ),
        q_over=_coerce_float(
            payload.get("q_over", 0.3), field_name="training.loss_params.q_over"
        ),
        late_factor=_coerce_float(
            payload.get("late_factor", 2.0),
            field_name="training.loss_params.late_factor",
        ),
    )
    if not 0 < params.down_th < params.up_th < 1:
        raise ValueError(
            "training.loss_params thresholds must satisfy 0 < down_th < up_th < 1"
        )
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
            "mode",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "min_lr",
            "eps",
        },
        section="training.lr_scheduler",
    )
    name = str(payload.get("name", "")).strip()
    if name not in {"OneCycleLR", "ReduceLROnPlateau"}:
        raise ValueError(
            "training.lr_scheduler.name must be one of: OneCycleLR, ReduceLROnPlateau"
        )
    max_lr = _coerce_float(payload["max_lr"], field_name="training.lr_scheduler.max_lr")
    if max_lr <= 0:
        raise ValueError("training.lr_scheduler.max_lr must be > 0")
    if name == "OneCycleLR":
        pct_start = _coerce_float(
            payload["pct_start"], field_name="training.lr_scheduler.pct_start"
        )
        if not 0 < pct_start < 1:
            raise ValueError(
                "training.lr_scheduler.pct_start must satisfy 0 < value < 1"
            )
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

    mode = str(payload["mode"]).strip().lower()
    if mode not in {"min", "max"}:
        raise ValueError("training.lr_scheduler.mode must be one of: min, max")
    factor = _coerce_float(payload["factor"], field_name="training.lr_scheduler.factor")
    if not 0 < factor < 1:
        raise ValueError("training.lr_scheduler.factor must satisfy 0 < value < 1")
    patience = _coerce_nonnegative_int(
        payload["patience"], field_name="training.lr_scheduler.patience"
    )
    threshold = _coerce_float(
        payload["threshold"], field_name="training.lr_scheduler.threshold"
    )
    if threshold < 0:
        raise ValueError("training.lr_scheduler.threshold must be >= 0")
    threshold_mode = str(payload["threshold_mode"]).strip().lower()
    if threshold_mode not in {"rel", "abs"}:
        raise ValueError(
            "training.lr_scheduler.threshold_mode must be one of: rel, abs"
        )
    cooldown = _coerce_nonnegative_int(
        payload["cooldown"], field_name="training.lr_scheduler.cooldown"
    )
    min_lr = _coerce_float(payload["min_lr"], field_name="training.lr_scheduler.min_lr")
    if min_lr < 0:
        raise ValueError("training.lr_scheduler.min_lr must be >= 0")
    eps = _coerce_float(payload["eps"], field_name="training.lr_scheduler.eps")
    if eps <= 0:
        raise ValueError("training.lr_scheduler.eps must be > 0")
    return TrainingLRSchedulerConfig(
        name="ReduceLROnPlateau",
        max_lr=max_lr,
        mode=cast(Literal["min", "max"], mode),
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=cast(Literal["rel", "abs"], threshold_mode),
        cooldown=cooldown,
        min_lr=min_lr,
        eps=eps,
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
            raise ValueError("training.optimizer.kwargs keys must be non-empty strings")
        normalized_kwargs[key] = deepcopy(item)
    return TrainingOptimizerConfig(
        name=cast(
            Literal["adamw", "ademamix", "mars", "soap", "rmsprop", "radam"], name
        ),
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


def resolve_config_path(
    repo_root: Path,
    config_path: str | Path | None = None,
    config_toml_path: str | Path | None = None,
) -> tuple[Path, str]:
    def _normalize_explicit_path(path_like: str | Path) -> Path:
        raw_text = str(path_like)
        normalized_text = raw_text.replace("\\", "/")
        candidate = Path(normalized_text)
        if candidate.is_absolute():
            return candidate
        if "/" in normalized_text:
            return candidate
        repo_candidate = repo_root / candidate
        if repo_candidate.exists():
            return candidate
        suffix = candidate.suffix.lower()
        if suffix not in {".yaml", ".yml", ".toml"}:
            return candidate
        matches = [
            resolved.relative_to(repo_root)
            for resolved in repo_root.rglob(f"*{suffix}")
            if resolved.is_file()
            and resolved.relative_to(repo_root).as_posix().replace("/", "")
            == normalized_text
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                "Config path matched multiple repo files after shell path normalization; "
                "use forward slashes or quote the path explicitly: "
                f"{raw_text} -> {matches}"
            )
        return candidate

    if config_toml_path is not None:
        path = _normalize_explicit_path(config_toml_path)
        if not path.is_absolute():
            path = repo_root / path
        return path, "toml"
    if config_path is not None:
        path = _normalize_explicit_path(config_path)
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


def _normalize_job(
    job: dict[str, Any],
    *,
    search_space: dict[str, Any] | None,
    allow_missing_search_space: bool = False,
    model_search_space_key: str = "models",
    stage_scope: str | None = None,
) -> JobConfig:
    model_name = str(job["model"])
    if stage_scope != "aa_forecast" and model_name == "AAForecast":
        raise ValueError(
            "Direct top-level AAForecast jobs are no longer supported; use aa_forecast.enabled with aa_forecast.config_path"
        )
    _ensure_plugins_loaded()
    stage_plugin = get_stage_plugin(stage_scope) if stage_scope else None
    if stage_plugin is not None:
        stage_plugin.validate_model(model_name)
    stage_plugin_owns_job = False
    if stage_plugin is not None:
        owns_top_level_job = getattr(stage_plugin, "owns_top_level_job", None)
        stage_plugin_owns_job = bool(
            callable(owns_top_level_job) and owns_top_level_job(model_name)
        )
    params = dict(job.get("params", {}))
    if model_name == "AAForecast" and "anomaly_threshold" in params:
        raise ValueError(
            "AAForecast.anomaly_threshold has been removed; migrate to aa_forecast.thresh and aa_forecast.star_anomaly_tails"
        )
    params_for_mode = params
    if stage_scope == "aa_forecast" and model_name == "AAForecast":
        structural_keys = {
            "thresh",
            "lowess_frac",
            "lowess_delta",
            "uncertainty_enabled",
            "uncertainty_dropout_candidates",
            "uncertainty_sample_count",
        }
        params_for_mode = {
            key: value for key, value in params.items() if key not in structural_keys
        }
    supported_auto_models = (
        stage_plugin.supported_models()
        if stage_plugin is not None
        else SUPPORTED_MODEL_AUTO_MODEL_NAMES
    )
    requested_mode = _requested_job_mode(model_name, params_for_mode)
    if stage_plugin_owns_job and not params_for_mode and model_name != "AAForecast":
        requested_mode = "learned_fixed"
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


def _relative_config_reference(repo_root: Path, target_path: Path) -> str:
    resolved_repo_root = repo_root.resolve()
    resolved_target = target_path.resolve()
    if str(resolved_target).startswith(str(resolved_repo_root) + "/"):
        return str(resolved_target.relative_to(resolved_repo_root))
    return str(resolved_target)


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
    training_search = _normalize_training_search_config(payload.get("training_search"))
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
    if "residual" in payload:
        raise ValueError(
            "legacy residual config is no longer supported; remove the top-level residual section"
        )
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
    if "opt_study_count" in runtime:
        if runtime["opt_study_count"] is None:
            runtime.pop("opt_study_count", None)
        else:
            runtime["opt_study_count"] = int(runtime["opt_study_count"])
            if runtime["opt_study_count"] <= 0:
                raise ValueError("runtime.opt_study_count must be a positive integer")
    if runtime.get("opt_selected_study") is not None:
        runtime["opt_selected_study"] = int(runtime["opt_selected_study"])
        if runtime["opt_selected_study"] <= 0:
            raise ValueError(
                "runtime.opt_selected_study must be a positive integer"
            )
    if (
        runtime.get("opt_selected_study") is not None
        and runtime["opt_selected_study"]
        > runtime.get("opt_study_count", RuntimeConfig.opt_study_count)
    ):
        raise ValueError(
            "runtime.opt_selected_study cannot exceed runtime.opt_study_count"
        )
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
    if loss in LOSSES_WITH_PARAMS:
        training["loss_params"] = _normalize_training_loss_params(
            training.get("loss_params")
        )
    elif "loss_params" in training:
        raise ValueError(
            f"training.loss_params is only supported when training.loss in {LOSSES_WITH_PARAMS}"
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
                raise ValueError("training.accelerator must be one of: auto, cpu, gpu")
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
    scheduler["gpu_ids"] = tuple(int(item) for item in scheduler.get("gpu_ids", (0, 1)))
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
            if isinstance(job.get("params"), dict) and selector in job["params"]
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
            if isinstance(job.get("params"), dict):
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
    if training_search["enabled"] and any(
        job.validated_mode == "learned_auto" and job.model in SUPPORTED_AUTO_MODEL_NAMES
        for job in jobs
    ):
        training_search_payload = None
        if search_space is not None:
            training_search_payload = search_space.get(training_search_space_key)
            if training_search_payload is None and stage_plugin is not None:
                fallback = stage_plugin.training_search_space_fallback_key()
                if fallback is not None and training_search_space_key != fallback:
                    training_search_payload = search_space.get(fallback)
        training_selected = (
            tuple(training_search_payload["global"])
            if training_search_payload is not None
            else ()
        )
    training_requested_mode = cast(
        Literal["training_fixed", "training_auto_requested"],
        "training_auto_requested" if training_selected else "training_fixed",
    )
    training_validated_mode = cast(
        Literal["training_fixed", "training_auto"],
        "training_auto" if training_selected else "training_fixed",
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
            f"jobs route must reference a yaml, yml, or toml file: {jobs_path}"
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
                f"Shared settings file does not exist: {explicit_shared_settings_path}"
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
    payload = dict(payload)
    _reject_legacy_or_mixed_aaforecast_jobs(payload)
    jobs_fanout_specs = _resolve_jobs_fanout_specs(
        repo_root,
        source_path=source_path,
        jobs_value=payload.get("jobs", []),
    )
    _ensure_plugins_loaded()
    stage_plugin = get_stage_plugin_for_payload(payload)

    def _active_stage_owns_top_level_job(model_name: str) -> bool:
        if stage_plugin is None:
            return False
        owns_top_level_job = getattr(stage_plugin, "owns_top_level_job", None)
        return bool(callable(owns_top_level_job) and owns_top_level_job(model_name))

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
        if (
            stage_plugin is not None
            and getattr(stage_plugin, "config_key", None) == "aa_forecast"
            and raw_stage_config is not None
            and stage_plugin.is_enabled(raw_stage_config)
            and payload["jobs"]
        ):
            raise ValueError(
                "aa_forecast plugin route cannot be combined with top-level jobs; split AAForecast into its own plugin-routed config"
            )
    else:
        payload["jobs"] = _resolve_jobs_reference(
            repo_root,
            source_path=source_path,
            jobs_value=payload.get("jobs", []),
        )
        jobs_payload_candidates = [payload["jobs"]]
        if (
            stage_plugin is not None
            and getattr(stage_plugin, "config_key", None) == "aa_forecast"
            and raw_stage_config is not None
            and stage_plugin.is_enabled(raw_stage_config)
            and payload["jobs"]
        ):
            raise ValueError(
                "aa_forecast plugin route cannot be combined with top-level jobs; split AAForecast into its own plugin-routed config"
            )
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
                        if str(spec.resolved_path).startswith(
                            str(repo_root.resolve()) + "/"
                        )
                        else str(spec.resolved_path)
                    ),
                )
                for spec in stage_fanout_probe
            )
            jobs_payload_candidates = [
                [dict(job) for job in spec.jobs_payload] for spec in jobs_fanout_specs
            ]
        elif isinstance(stage_payload_probe, dict) and "jobs" in stage_payload_probe:
            payload["jobs"] = [dict(job) for job in stage_payload_probe["jobs"]]
            jobs_payload_candidates = [payload["jobs"]]
    if isinstance(stage_payload_probe, dict):
        if stage_payload_probe.get("requires_search_space"):
            requested_search_space = True
        if "training_search" in stage_payload_probe:
            payload["training_search"] = dict(stage_payload_probe["training_search"])
        if stage_payload_probe.get("residual"):
            raise ValueError(
                "stage plugins may no longer inject residual config; residual support has been removed"
            )
    for jobs_candidate in jobs_payload_candidates:
        for job in jobs_candidate:
            model_name = str(job.get("model"))
            if (
                model_name not in BASELINE_MODEL_NAMES
                and not (
                    _active_stage_owns_top_level_job(model_name)
                    and model_name != "AAForecast"
                )
                and not dict(job.get("params", {}))
            ):
                requested_search_space = True
                break
        if requested_search_space:
            break
    if stage_payload_probe is not None:
        stage_jobs = stage_payload_probe.get("jobs", [])
        for sjob in stage_jobs:
            model_name = str(sjob.get("model"))
            if (
                model_name not in BASELINE_MODEL_NAMES
                and not _active_stage_owns_top_level_job(model_name)
                and not dict(sjob.get("params", {}))
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
        stage_scope=stage_plugin.config_key if stage_plugin is not None else None,
        repo_root=repo_root,
        source_path=source_path,
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
    config = _resolve_aa_forecast_stage_plugin_config(config)
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
    loaded_search_space_payload = (
        search_space_contract.payload if search_space_contract else None
    )
    if (
        stage_plugin is not None
        and stage_plugin.config_key == "aa_forecast"
        and stage_plugin_loaded is not None
        and stage_plugin_loaded.search_space_payload is not None
    ):
        loaded_search_space_payload = stage_plugin_loaded.search_space_payload
    return LoadedConfig(
        config=config,
        source_path=source_path,
        source_type=source_type,
        normalized_payload=normalized_payload,
        input_hash=_hash_text(raw_text),
        resolved_hash=_hash_text(resolved_text),
        search_space_path=search_space_contract.path if search_space_contract else None,
        search_space_hash=search_space_contract.sha256
        if search_space_contract
        else None,
        search_space_payload=loaded_search_space_payload,
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
        "task",
        "dataset",
        "runtime",
        "training",
        "cv",
        "scheduler",
        "jobs",
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
        if not stage_source_path.exists():
            raise FileNotFoundError(
                f"stage plugin config_path does not exist: {stage_source_path}"
            )
        stage_plugin_loaded = stage_plugin.load_stage(
            repo_root,
            source_path=loaded.source_path,
            source_type=loaded.source_type,
            config=config.stage_plugin_config,
            search_space_contract=stage_search_space_contract,
        )
        config = stage_plugin.apply_stage_to_config(config, stage_plugin_loaded)
    config = _resolve_aa_forecast_stage_plugin_config(config)
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
