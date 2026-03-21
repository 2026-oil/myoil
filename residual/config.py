from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import hashlib
import json
import tomllib

import yaml

from .optuna_spaces import (
    BASELINE_MODEL_NAMES,
    DEFAULT_TRAINING_PARAMS,
    FIXED_TRAINING_KEYS,
    SUPPORTED_AUTO_MODEL_NAMES,
    SUPPORTED_RESIDUAL_MODELS,
    MODEL_PARAM_REGISTRY,
    RESIDUAL_PARAM_REGISTRY,
    TRAINING_PARAM_REGISTRY,
    TRAINING_SELECTOR_TO_CONFIG_FIELD,
    ResidualMode,
    SearchSpaceContract,
    load_search_space_contract,
)

CONFIG_FILENAMES = ("config.yaml", "config.yml", "config.toml")
DEFAULT_MANIFEST_VERSION = "1"
DEFAULT_ARTIFACT_SCHEMA_VERSION = "1"
DEFAULT_EVALUATION_PROTOCOL_VERSION = "2"
SUPPORTED_LOSSES = {"mse"}
CENTRALIZED_TRAINING_KEYS = {
    "train_protocol",
    "input_size",
    "batch_size",
    "valid_batch_size",
    "windows_batch_size",
    "inference_windows_batch_size",
    "learning_rate",
    "scaler_type",
    "step_size",
    "max_steps",
    "val_size",
    "val_check_steps",
    "early_stop_patience_steps",
    "num_lr_decays",
    "loss",
} | set(FIXED_TRAINING_KEYS)
LEGACY_SHARED_JOB_TRAINING_KEYS = {
    "scaler_type",
    "step_size",
    "early_stop_patience_steps",
    "num_lr_decays",
}
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
FORBIDDEN_RESIDUAL_LAG_SOURCES = {
    "y",
    "residual_target",
    "cutoff_day",
    "ds_day",
}
ResidualTargetMode = Literal["level", "delta"]


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


@dataclass(frozen=True)
class TaskConfig:
    name: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    train_protocol: str = "expanding_window_tscv"
    input_size: int = DEFAULT_TRAINING_PARAMS["input_size"]
    season_length: int = DEFAULT_TRAINING_PARAMS["season_length"]
    batch_size: int = DEFAULT_TRAINING_PARAMS["batch_size"]
    valid_batch_size: int = DEFAULT_TRAINING_PARAMS["valid_batch_size"]
    windows_batch_size: int = DEFAULT_TRAINING_PARAMS["windows_batch_size"]
    inference_windows_batch_size: int = DEFAULT_TRAINING_PARAMS[
        "inference_windows_batch_size"
    ]
    learning_rate: float = DEFAULT_TRAINING_PARAMS["learning_rate"]
    scaler_type: str | None = DEFAULT_TRAINING_PARAMS["scaler_type"]
    model_step_size: int = DEFAULT_TRAINING_PARAMS["model_step_size"]
    max_steps: int = DEFAULT_TRAINING_PARAMS["max_steps"]
    val_size: int = DEFAULT_TRAINING_PARAMS["val_size"]
    val_check_steps: int = DEFAULT_TRAINING_PARAMS["val_check_steps"]
    early_stop_patience_steps: int = DEFAULT_TRAINING_PARAMS[
        "early_stop_patience_steps"
    ]
    num_lr_decays: int = DEFAULT_TRAINING_PARAMS["num_lr_decays"]
    loss: str = "mse"


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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.task.name is None:
            payload.pop("task", None)
        payload["dataset"]["path"] = str(self.dataset.path)
        if "model_step_size" in payload["training"]:
            payload["training"]["step_size"] = payload["training"].pop(
                "model_step_size"
            )
        payload["training_search"]["selected_search_params"] = list(
            payload["training_search"]["selected_search_params"]
        )
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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        exog_payload.get("hist"), field_name="residual.features.exog_sources.hist"
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
    for name in CONFIG_FILENAMES:
        candidate = repo_root / name
        if candidate.exists():
            return candidate, "yaml" if candidate.suffix in {
                ".yaml",
                ".yml",
            } else "toml"
    raise FileNotFoundError("No config file found in repo root (config.yaml/yml/toml)")


def _load_document(path: Path, source_type: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if source_type == "toml":
        return tomllib.loads(text)
    payload = yaml.safe_load(text)
    return {} if payload is None else payload


def _coerce_param_name_list(value: Any, *, section: str, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"search_space.{section}.{name} must be a list of canonical parameter names"
        )
    out = tuple(str(item) for item in value)
    if any(not item.strip() for item in out):
        raise ValueError(
            f"search_space.{section}.{name} contains an empty parameter name"
        )
    return out


def _validate_search_space_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not payload:
        raise ValueError("search_space.yaml is empty")
    required_sections = {"models", "residual"}
    missing = required_sections.difference(payload)
    if missing:
        raise ValueError(
            "search_space.yaml must contain top-level sections: models and residual"
        )
    normalized: dict[str, Any] = {
        "models": {},
        "training": (),
        "residual": {},
    }
    for section in ("models", "residual"):
        section_payload = payload.get(section)
        if section_payload is None:
            continue
        if not isinstance(section_payload, dict):
            raise ValueError(f"search_space.{section} must be a mapping")
        for name, value in section_payload.items():
            normalized[section][str(name)] = _coerce_param_name_list(
                value, section=section, name=str(name)
            )
    training_payload = payload.get("training")
    if training_payload is not None:
        normalized["training"] = _coerce_param_name_list(
            training_payload, section="training", name="selectors"
        )
    unknown_models = sorted(
        set(normalized["models"]).difference(SUPPORTED_AUTO_MODEL_NAMES)
    )
    if unknown_models:
        raise ValueError(
            "search_space.models contains unsupported learned model(s): "
            + ", ".join(unknown_models)
        )
    unknown_residual = sorted(
        set(normalized["residual"]).difference(SUPPORTED_RESIDUAL_MODELS)
    )
    if unknown_residual:
        raise ValueError(
            "search_space.residual contains unsupported residual model(s): "
            + ", ".join(unknown_residual)
        )
    for model_name, param_names in normalized["models"].items():
        unknown = sorted(set(param_names).difference(MODEL_PARAM_REGISTRY[model_name]))
        if unknown:
            raise ValueError(
                f"search_space.models.{model_name} contains unknown parameter(s): {', '.join(unknown)}"
            )
    fixed_training = sorted(
        set(normalized["training"]).intersection(FIXED_TRAINING_KEYS)
    )
    if fixed_training:
        raise ValueError(
            "search_space.training contains fixed, non-tunable parameter(s): "
            + ", ".join(fixed_training)
        )
    unknown_training = sorted(set(normalized["training"]).difference(TRAINING_PARAM_REGISTRY))
    if unknown_training:
        raise ValueError(
            "search_space.training contains unknown parameter(s): "
            + ", ".join(unknown_training)
        )
    for model_name, param_names in normalized["residual"].items():
        unknown = sorted(
            set(param_names).difference(RESIDUAL_PARAM_REGISTRY[model_name])
        )
        if unknown:
            raise ValueError(
                f"search_space.residual.{model_name} contains unknown parameter(s): {', '.join(unknown)}"
            )
    if "learning_rate" in normalized["training"]:
        overlaps = sorted(
            model_name
            for model_name, param_names in normalized["models"].items()
            if "learning_rate" in param_names
        )
        if overlaps:
            raise ValueError(
                "search_space.training.learning_rate overlaps with model-level "
                "learning_rate selector(s): "
                + ", ".join(overlaps)
            )
    return normalized


def _requested_job_mode(model_name: str, params: dict[str, Any]) -> str:
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
    search_space: dict[str, dict[str, tuple[str, ...]]] | None,
    allow_missing_search_space: bool = False,
) -> JobConfig:
    model_name = str(job["model"])
    params = dict(job.get("params", {}))
    requested_mode = _requested_job_mode(model_name, params)
    selected: tuple[str, ...] = ()
    if requested_mode == "baseline_fixed":
        validated_mode = "baseline_fixed"
    elif requested_mode == "learned_fixed":
        validated_mode = "learned_fixed"
    else:
        if model_name not in SUPPORTED_AUTO_MODEL_NAMES:
            raise ValueError(
                f"jobs[{model_name}] uses empty params but has no supported learned_auto Optuna mapping"
            )
        if search_space is None or model_name not in search_space["models"]:
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
                f"jobs[{model_name}] requires search_space.models.{model_name} for learned_auto execution"
            )
        selected = search_space["models"][model_name]
        unknown = sorted(set(selected).difference(MODEL_PARAM_REGISTRY[model_name]))
        if unknown:
            raise ValueError(
                f"search_space.models.{model_name} contains unknown parameter(s): {', '.join(unknown)}"
            )
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
    search_space: dict[str, dict[str, tuple[str, ...]]] | None,
    allow_missing_search_space: bool = False,
) -> AppConfig:
    task = dict(payload.get("task", {}))
    dataset = dict(payload.get("dataset", {}))
    runtime = dict(payload.get("runtime", {}))
    training = dict(payload.get("training", {}))
    for selector, field_name in TRAINING_SELECTOR_TO_CONFIG_FIELD.items():
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

    training.setdefault("loss", "mse")
    loss = str(training["loss"]).lower()
    if loss not in SUPPORTED_LOSSES:
        raise ValueError(f"Unsupported common loss: {loss}")
    training["loss"] = loss

    scheduler.setdefault("worker_devices", 1)
    scheduler["gpu_ids"] = tuple(int(item) for item in scheduler.get("gpu_ids", (0, 1)))
    if int(scheduler["worker_devices"]) != 1:
        raise ValueError("worker_devices must remain 1 for scheduler-launched jobs")

    residual.setdefault("enabled", True)
    residual.setdefault("model", "xgboost")
    residual.setdefault("target", "level")
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
            residual_selected = search_space["residual"][residual_model]
            unknown = sorted(
                set(residual_selected).difference(RESIDUAL_PARAM_REGISTRY[residual_model])
            )
            if unknown:
                raise ValueError(
                    f"search_space.residual.{residual_model} contains unknown parameter(s): {', '.join(unknown)}"
                )
            residual_validated_mode = "residual_auto"
    else:
        residual_validated_mode = residual_requested_mode

    jobs_payload = []
    for raw_job in payload.get("jobs", []):
        normalized_job = dict(raw_job)
        normalized_job["params"] = dict(raw_job.get("params", {}))
        jobs_payload.append(normalized_job)

    for selector in LEGACY_SHARED_JOB_TRAINING_KEYS:
        field_name = TRAINING_SELECTOR_TO_CONFIG_FIELD.get(selector, selector)
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

    jobs = tuple(
        _normalize_job(
            job,
            search_space=search_space,
            allow_missing_search_space=allow_missing_search_space,
        )
        for job in jobs_payload
    )
    if not jobs:
        raise ValueError("Config must define at least one job")
    models = [job.model for job in jobs]
    if len(models) != len(set(models)):
        raise ValueError("jobs.model values must be unique")
    for job in jobs:
        duplicated = CENTRALIZED_TRAINING_KEYS.intersection(job.params)
        if duplicated:
            duplicated_keys = ", ".join(sorted(duplicated))
            raise ValueError(
                f"jobs[{job.model}] repeats centralized training key(s): {duplicated_keys}. "
                "Move these settings under training."
            )
    training_selected = ()
    if any(job.validated_mode == "learned_auto" for job in jobs):
        training_selected = (
            search_space["training"] if search_space is not None else ()
        )
    training_requested_mode = (
        "training_auto_requested" if training_selected else "training_fixed"
    )
    training_validated_mode = (
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
    )


def load_app_config(
    repo_root: Path,
    *,
    config_path: str | Path | None = None,
    config_toml_path: str | Path | None = None,
) -> LoadedConfig:
    source_path, source_type = resolve_config_path(
        repo_root,
        config_path=config_path,
        config_toml_path=config_toml_path,
    )
    raw_text = source_path.read_text(encoding="utf-8")
    payload = _load_document(source_path, source_type)
    search_space_contract: SearchSpaceContract | None = None
    requested_search_space = False
    jobs_payload = payload.get("jobs", [])
    for job in jobs_payload:
        if str(job.get("model")) not in BASELINE_MODEL_NAMES and not dict(job.get("params", {})):
            requested_search_space = True
            break
    residual_payload = dict(payload.get("residual", {}))
    if (
        residual_payload.get("enabled", True)
        and not dict(residual_payload.get("params", {}))
    ):
        requested_search_space = True
    if requested_search_space:
        search_space_contract = load_search_space_contract(repo_root)
    search_space = (
        _validate_search_space_payload(search_space_contract.payload)
        if search_space_contract is not None
        else None
    )
    dataset_path = Path(payload.get("dataset", {}).get("path", "df.csv"))
    if dataset_path.is_absolute():
        dataset_base_dir = source_path.parent
    else:
        repo_candidate = (repo_root / dataset_path).resolve()
        local_candidate = (source_path.parent / dataset_path).resolve()
        dataset_base_dir = (
            repo_root if repo_candidate.exists() or not local_candidate.exists() else source_path.parent
        )
    base_config = _normalize_payload(
        payload,
        dataset_base_dir,
        search_space=None,
        allow_missing_search_space=True,
    )
    config = (
        _normalize_payload(payload, dataset_base_dir, search_space=search_space)
        if search_space is not None
        else base_config
    )
    normalized_payload = config.to_dict()
    normalized_payload["search_space_path"] = (
        str(search_space_contract.path) if search_space_contract else None
    )
    normalized_payload["search_space_sha256"] = (
        search_space_contract.sha256 if search_space_contract else None
    )
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
    )
