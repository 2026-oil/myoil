from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import hashlib
import json
import math
import tomllib

import yaml

from .optuna_spaces import (
    BASELINE_MODEL_NAMES,
    DEFAULT_TRAINING_PARAMS,
    FIXED_TRAINING_KEYS,
    LEGACY_TRAINING_SELECTOR_TO_CONFIG_FIELD,
    ResidualMode,
    SearchSpaceContract,
    load_search_space_contract,
    normalize_search_space_payload,
    SUPPORTED_AUTO_MODEL_NAMES,
    SUPPORTED_BS_PREFORCAST_MODELS,
    SUPPORTED_RESIDUAL_MODELS,
)

CONFIG_FILENAMES = ("config.yaml", "config.yml", "config.toml")
DEFAULT_MANIFEST_VERSION = "1"
DEFAULT_ARTIFACT_SCHEMA_VERSION = "1"
DEFAULT_EVALUATION_PROTOCOL_VERSION = "2"
SUPPORTED_LOSSES = {"mse", "exloss"}
CENTRALIZED_TRAINING_KEYS = {
    "train_protocol",
    "input_size",
    "batch_size",
    "valid_batch_size",
    "windows_batch_size",
    "inference_windows_batch_size",
    "learning_rate",
    "scaler_type",
    "model_step_size",
    "max_steps",
    "val_size",
    "val_check_steps",
    "early_stop_patience_steps",
    "num_lr_decays",
    "loss",
    "loss_params",
    "accelerator",
    "devices",
    "strategy",
    "precision",
    "dataloader_kwargs",
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
BS_PREFORCAST_KEYS = {
    "enabled",
    "config_path",
    "using_futr_exog",
    "target_columns",
    "task",
    "routing",
}
BS_PREFORCAST_TASK_KEYS = {"multivariable"}
BS_PREFORCAST_ROUTING_KEYS = {"univariable_config", "multivariable_config"}
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
class BsPreforcastTaskConfig:
    multivariable: bool = False


@dataclass(frozen=True)
class BsPreforcastRoutingConfig:
    univariable_config: str | None = None
    multivariable_config: str | None = None
    selected_config_path: str | None = None


@dataclass(frozen=True)
class BsPreforcastConfig:
    enabled: bool = False
    config_path: str | None = None
    using_futr_exog: bool = False
    target_columns: tuple[str, ...] = field(default_factory=tuple)
    task: BsPreforcastTaskConfig = field(default_factory=BsPreforcastTaskConfig)
    routing: BsPreforcastRoutingConfig = field(default_factory=BsPreforcastRoutingConfig)



@dataclass(frozen=True)
class TrainingLossParams:
    up_th: float = 0.9
    down_th: float = 0.1
    lamda_underestimate: float = 1.2
    lamda_overestimate: float = 1.0
    lamda: float = 1.0


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
    loss_params: TrainingLossParams | None = None
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
    bs_preforcast: BsPreforcastConfig
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
        if not payload["bs_preforcast"].get("enabled", False):
            payload.pop("bs_preforcast", None)
        else:
            payload["bs_preforcast"]["target_columns"] = list(
                payload["bs_preforcast"]["target_columns"]
            )
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
    bs_preforcast_stage1: BsPreforcastStageLoadedConfig | None = None


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


def _normalize_bs_preforcast_config(value: Any) -> BsPreforcastConfig:
    if value is None:
        return BsPreforcastConfig()
    if not isinstance(value, dict):
        raise ValueError("bs_preforcast must be a mapping")
    payload = dict(value)
    _unknown_keys(payload, allowed=BS_PREFORCAST_KEYS, section="bs_preforcast")

    task_payload = dict(payload.get("task") or {})
    routing_payload = dict(payload.get("routing") or {})
    if not isinstance(payload.get("task", {}), (dict, type(None))):
        raise ValueError("bs_preforcast.task must be a mapping")
    if not isinstance(payload.get("routing", {}), (dict, type(None))):
        raise ValueError("bs_preforcast.routing must be a mapping")
    _unknown_keys(
        task_payload,
        allowed=BS_PREFORCAST_TASK_KEYS,
        section="bs_preforcast.task",
    )
    _unknown_keys(
        routing_payload,
        allowed=BS_PREFORCAST_ROUTING_KEYS,
        section="bs_preforcast.routing",
    )

    enabled = _coerce_bool(
        payload.get("enabled"),
        field_name="bs_preforcast.enabled",
        default=False,
    )
    using_futr_exog = _coerce_bool(
        payload.get("using_futr_exog"),
        field_name="bs_preforcast.using_futr_exog",
        default=False,
    )
    config_path = _coerce_optional_path_string(
        payload.get("config_path"),
        field_name="bs_preforcast.config_path",
    )
    target_columns = _coerce_name_tuple(
        payload.get("target_columns"),
        field_name="bs_preforcast.target_columns",
    )
    multivariable = _coerce_bool(
        task_payload.get("multivariable"),
        field_name="bs_preforcast.task.multivariable",
        default=False,
    )
    routing = BsPreforcastRoutingConfig(
        univariable_config=_coerce_optional_path_string(
            routing_payload.get("univariable_config"),
            field_name="bs_preforcast.routing.univariable_config",
        ),
        multivariable_config=_coerce_optional_path_string(
            routing_payload.get("multivariable_config"),
            field_name="bs_preforcast.routing.multivariable_config",
        ),
    )
    selected_config_path = config_path or (
        routing.multivariable_config if multivariable else routing.univariable_config
    )
    if enabled:
        if not target_columns:
            raise ValueError(
                "bs_preforcast.target_columns must be non-empty when bs_preforcast.enabled is true"
            )
        if selected_config_path is None:
            raise ValueError(
                "bs_preforcast.config_path or a matching bs_preforcast.routing.* path is required when bs_preforcast.enabled is true"
            )
    return BsPreforcastConfig(
        enabled=enabled,
        config_path=config_path,
        using_futr_exog=using_futr_exog,
        target_columns=target_columns,
        task=BsPreforcastTaskConfig(multivariable=multivariable),
        routing=BsPreforcastRoutingConfig(
            univariable_config=routing.univariable_config,
            multivariable_config=routing.multivariable_config,
            selected_config_path=selected_config_path if enabled else None,
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


def _validate_search_space_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    return normalize_search_space_payload(payload)


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
    search_space: dict[str, Any] | None,
    allow_missing_search_space: bool = False,
    model_search_space_key: str = "models",
) -> JobConfig:
    model_name = str(job["model"])
    params = dict(job.get("params", {}))
    supported_auto_models = (
        SUPPORTED_BS_PREFORCAST_MODELS
        if model_search_space_key == "bs_preforcast_models"
        else SUPPORTED_AUTO_MODEL_NAMES
    )
    requested_mode = _requested_job_mode(model_name, params)
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
        if search_space is not None:
            search_space_models = search_space.get(model_search_space_key)
            if (
                search_space_models is None
                and model_search_space_key == "bs_preforcast_models"
            ):
                search_space_models = search_space.get("models")
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
) -> AppConfig:
    task = dict(payload.get("task", {}))
    dataset = dict(payload.get("dataset", {}))
    runtime = dict(payload.get("runtime", {}))
    training = dict(payload.get("training", {}))
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
    bs_preforcast = _normalize_bs_preforcast_config(payload.get("bs_preforcast"))

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

    jobs_payload = []
    for raw_job in payload.get("jobs", []):
        normalized_job = dict(raw_job)
        normalized_job["params"] = dict(raw_job.get("params", {}))
        jobs_payload.append(normalized_job)

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

    jobs = tuple(
        _normalize_job(
            job,
            search_space=search_space,
            allow_missing_search_space=allow_missing_search_space,
            model_search_space_key=model_search_space_key,
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
        if (
            (
                stage_scope == "bs_preforcast"
                or (
                    search_space is not None
                    and search_space.get("__scope__") == "bs_preforcast"
                )
            )
            and job.model in {"AutoARIMA", "ES"}
        ):
            duplicated = duplicated.difference({"season_length"})
        if duplicated:
            duplicated_keys = ", ".join(sorted(duplicated))
            raise ValueError(
                f"jobs[{job.model}] repeats centralized training key(s): {duplicated_keys}. "
                "Move these settings under training."
            )
    training_selected = ()
    if any(job.validated_mode == "learned_auto" for job in jobs):
        training_search_payload = None
        if search_space is not None:
            training_search_payload = search_space.get(training_search_space_key)
            if (
                training_search_payload is None
                and training_search_space_key == "bs_preforcast_training"
            ):
                training_search_payload = search_space.get("training")
        training_selected = (
            tuple(training_search_payload["global"])
            if training_search_payload is not None
            else ()
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
        bs_preforcast=bs_preforcast,
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


def _bs_preforcast_stage_search_space(
    search_space_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if search_space_payload is None:
        return None
    model_payload = search_space_payload.get("bs_preforcast_models", {})
    training_payload = search_space_payload.get(
        "bs_preforcast_training", {"global": {}, "per_model": {}}
    )
    return {
        "__scope__": "bs_preforcast",
        "models": model_payload,
        "training": training_payload,
        "bs_preforcast_models": model_payload,
        "bs_preforcast_training": training_payload,
        "residual": search_space_payload.get("residual", {}),
    }


def _stage_payload_requests_search_space(stage_payload: dict[str, Any]) -> bool:
    for job in stage_payload.get("jobs", []):
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


def _rewrite_bs_preforcast_search_space_error(message: str) -> str:
    return (
        message.replace("search_space.models.", "search_space.bs_preforcast_models.")
        .replace("search_space.training.", "search_space.bs_preforcast_training.")
        .replace("search_space.models ", "search_space.bs_preforcast_models ")
        .replace("search_space.training ", "search_space.bs_preforcast_training ")
    )


def _merge_bs_preforcast_stage_payload(
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
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _load_bs_preforcast_stage1(
    repo_root: Path,
    *,
    source_path: Path,
    source_type: str,
    bs_preforcast: BsPreforcastConfig,
    search_space_contract: SearchSpaceContract | None,
) -> BsPreforcastStageLoadedConfig:
    selected_config_path = bs_preforcast.routing.selected_config_path
    if not bs_preforcast.enabled or selected_config_path is None:
        raise ValueError("bs_preforcast stage1 loading requires an enabled selected route")
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
    stage_payload = _merge_bs_preforcast_stage_payload(
        raw_stage_payload,
        multivariable=bs_preforcast.task.multivariable,
    )
    if stage_payload.get("bs_preforcast") not in (None, {}):
        raise ValueError("bs_preforcast routed YAML must not define its own bs_preforcast block")
    stage_search_space = _bs_preforcast_stage_search_space(
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
        raise ValueError(_rewrite_bs_preforcast_search_space_error(str(exc))) from exc
    stage_normalized_payload = stage_config.to_dict()
    stage_normalized_payload["bs_preforcast"] = {
        "enabled": True,
        "using_futr_exog": bs_preforcast.using_futr_exog,
        "target_columns": list(bs_preforcast.target_columns),
        "task": {"multivariable": bs_preforcast.task.multivariable},
        "routing": {
            "selected_config_path": str(stage_source_path),
        },
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


def load_app_config(
    repo_root: Path,
    *,
    config_path: str | Path | None = None,
    config_toml_path: str | Path | None = None,
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
    raw_bs_preforcast = _normalize_bs_preforcast_config(payload.get("bs_preforcast"))
    stage_source_path: Path | None = None
    stage_payload_probe: dict[str, Any] | None = None
    if raw_bs_preforcast.enabled:
        selected_config_path = raw_bs_preforcast.routing.selected_config_path
        if selected_config_path is None:
            raise ValueError("bs_preforcast enabled but selected route was not resolved")
        stage_source_path = _resolve_relative_config_reference(
            repo_root,
            source_path,
            selected_config_path,
        )
        if not stage_source_path.exists():
            raise FileNotFoundError(
                f"bs_preforcast selected route does not exist: {stage_source_path}"
            )
        stage_source_type = (
            "toml" if stage_source_path.suffix.lower() == ".toml" else "yaml"
        )
        stage_payload_probe = _load_document(stage_source_path, stage_source_type)
        if stage_payload_probe.get("bs_preforcast") not in (None, {}):
            raise ValueError(
                "bs_preforcast routed YAML must not define its own bs_preforcast block"
            )
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
    if stage_payload_probe is not None and _stage_payload_requests_search_space(
        stage_payload_probe
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
        model_search_space_key=model_search_space_key,
        training_search_space_key=training_search_space_key,
    )
    config = (
        _normalize_payload(
            payload,
            dataset_base_dir,
            search_space=search_space,
            model_search_space_key=model_search_space_key,
            training_search_space_key=training_search_space_key,
        )
        if search_space is not None
        else base_config
    )
    normalized_payload = config.to_dict()
    bs_preforcast_stage1 = None
    if config.bs_preforcast.enabled and config.bs_preforcast.routing.selected_config_path:
        stage_source_path = _resolve_relative_config_reference(
            repo_root,
            source_path,
            config.bs_preforcast.routing.selected_config_path,
        )
        if stage_source_path.exists():
            stage_search_space_contract = search_space_contract
            if stage_search_space_contract is None:
                candidate = (repo_root / "search_space.yaml").resolve()
                if candidate.exists():
                    stage_search_space_contract = load_search_space_contract(repo_root)
            bs_preforcast_stage1 = _load_bs_preforcast_stage1(
                repo_root,
                source_path=source_path,
                source_type=source_type,
                bs_preforcast=config.bs_preforcast,
                search_space_contract=stage_search_space_contract,
            )
            normalized_payload.setdefault("bs_preforcast", {})["stage1"] = {
                "source_path": str(bs_preforcast_stage1.source_path),
                "source_type": bs_preforcast_stage1.source_type,
                "config_input_sha256": bs_preforcast_stage1.input_hash,
                "config_resolved_sha256": bs_preforcast_stage1.resolved_hash,
                "search_space_path": (
                    str(bs_preforcast_stage1.search_space_path)
                    if bs_preforcast_stage1.search_space_path is not None
                    else None
                ),
                "search_space_sha256": bs_preforcast_stage1.search_space_hash,
                "target_columns": list(config.bs_preforcast.target_columns),
            }
            if search_space_contract is None:
                search_space_contract = stage_search_space_contract
        elif Path(config.bs_preforcast.routing.selected_config_path).is_absolute():
            raise FileNotFoundError(
                f"bs_preforcast selected route does not exist: {stage_source_path}"
            )
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
        search_space_payload=search_space_contract.payload if search_space_contract else None,
        bs_preforcast_stage1=bs_preforcast_stage1,
    )
