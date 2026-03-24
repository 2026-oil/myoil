from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Literal

import optuna
import yaml

SEARCH_SPACE_FILENAME = "search_space.yaml"
BASELINE_MODEL_NAMES = {"Naive", "SeasonalNaive", "HistoricAverage"}
EXCLUDED_AUTO_MODEL_NAMES = {"HINT"}
SUPPORTED_AUTO_MODEL_NAMES = {
    "RNN",
    "GRU",
    "LSTM",
    "TCN",
    "DeepAR",
    "DilatedRNN",
    "BiTCN",
    "xLSTM",
    "MLP",
    "NBEATS",
    "NBEATSx",
    "NHITS",
    "DLinear",
    "NLinear",
    "TiDE",
    "DeepNPTS",
    "DeformTime",
    "DeformableTST",
    "KAN",
    "TFT",
    "VanillaTransformer",
    "Informer",
    "Autoformer",
    "FEDformer",
    "PatchTST",
    "iTransformer",
    "TimeLLM",
    "TimeXer",
    "TimesNet",
    "StemGNN",
    "TSMixer",
    "TSMixerx",
    "MLPMultivariate",
    "SOFTS",
    "TimeMixer",
    "ModernTCN",
    "DUET",
    "Mamba",
    "SMamba",
    "CMamba",
    "xLSTMMixer",
    "RMoK",
    "XLinear",
}
SUPPORTED_RESIDUAL_MODELS = {"xgboost", "randomforest", "lightgbm"}
DEFAULT_OPTUNA_NUM_TRIALS = 20
DEFAULT_OPTUNA_STUDY_DIRECTION = "minimize"
DEFAULT_RESIDUAL_PARAMS_BY_MODEL = {
    "xgboost": {
        "n_estimators": 32,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
    "randomforest": {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "lightgbm": {
        "n_estimators": 64,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 1.0,
    },
}
DEFAULT_RESIDUAL_PARAMS = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["xgboost"].copy()
RESIDUAL_DEFAULTS = {
    name: params.copy() for name, params in DEFAULT_RESIDUAL_PARAMS_BY_MODEL.items()
}
DEFAULT_TRAINING_PARAMS = {
    "input_size": 64,
    "season_length": 52,
    "batch_size": 32,
    "valid_batch_size": 32,
    "windows_batch_size": 1024,
    "inference_windows_batch_size": 1024,
    "learning_rate": 0.001,
    "scaler_type": None,
    "model_step_size": 1,
    "max_steps": 1000,
    "val_size": 8,
    "val_check_steps": 100,
    "early_stop_patience_steps": -1,
    "num_lr_decays": -1,
}
LEGACY_TRAINING_SELECTOR_TO_CONFIG_FIELD = {"step_size": "model_step_size"}
FIXED_TRAINING_VALUES = {
    "season_length": 52,
    "max_steps": 1000,
    "val_size": 8,
    "val_check_steps": 100,
}
FIXED_TRAINING_KEYS = tuple(FIXED_TRAINING_VALUES)
GLOBAL_TRAINING_RANGE_SOURCE = "global_fallback"

ExecutionMode = Literal[
    "baseline_fixed",
    "learned_fixed",
    "learned_auto_requested",
    "learned_auto",
]
ResidualMode = Literal[
    "residual_disabled",
    "residual_fixed",
    "residual_auto_requested",
    "residual_auto",
]
SearchParamType = Literal["categorical", "int", "float"]
SearchParamSpec = dict[str, Any]


@dataclass(frozen=True)
class SearchSpaceContract:
    path: Path
    payload: dict[str, Any]
    sha256: str

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else payload


def optuna_num_trials(runtime_opt_n_trial: int | None = None) -> int:
    if runtime_opt_n_trial is not None:
        return int(runtime_opt_n_trial)
    return int(
        os.environ.get("NEURALFORECAST_OPTUNA_NUM_TRIALS", DEFAULT_OPTUNA_NUM_TRIALS)
    )


def optuna_seed(default: int) -> int:
    return int(os.environ.get("NEURALFORECAST_OPTUNA_SEED", default))


def build_optuna_sampler(seed: int) -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(seed=seed)


def _normalize_choices(value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError("categorical search-space choices must be a non-empty list")
    return deepcopy(value)


def _normalize_param_spec(spec: Any, *, section: str, owner: str, name: str) -> SearchParamSpec:
    if not isinstance(spec, dict):
        raise ValueError(
            f"search_space.{section}.{owner}.{name} must be a mapping with type metadata"
        )
    spec_type = str(spec.get("type", "")).strip().lower()
    if spec_type == "categorical":
        return {"type": "categorical", "choices": _normalize_choices(spec.get("choices"))}
    if spec_type == "int":
        low = spec.get("low")
        high = spec.get("high")
        step = spec.get("step", 1)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (low, high, step)):
            raise ValueError(
                f"search_space.{section}.{owner}.{name} int spec requires integer low/high/step"
            )
        if low > high or step <= 0:
            raise ValueError(
                f"search_space.{section}.{owner}.{name} int spec must satisfy low <= high and step > 0"
            )
        return {"type": "int", "low": int(low), "high": int(high), "step": int(step)}
    if spec_type == "float":
        low = spec.get("low")
        high = spec.get("high")
        log = bool(spec.get("log", False))
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (low, high)):
            raise ValueError(
                f"search_space.{section}.{owner}.{name} float spec requires numeric low/high"
            )
        low = float(low)
        high = float(high)
        if low > high:
            raise ValueError(
                f"search_space.{section}.{owner}.{name} float spec must satisfy low <= high"
            )
        if log and low <= 0:
            raise ValueError(
                f"search_space.{section}.{owner}.{name} float log spec requires low > 0"
            )
        normalized: SearchParamSpec = {"type": "float", "low": low, "high": high}
        if log:
            normalized["log"] = True
        return normalized
    raise ValueError(
        f"search_space.{section}.{owner}.{name} must declare type one of: categorical, int, float"
    )


MODEL_PARAM_REGISTRY: dict[str, dict[str, SearchParamSpec]]
TRAINING_PARAM_REGISTRY: dict[str, SearchParamSpec]
TRAINING_PARAM_REGISTRY_BY_MODEL: dict[str, dict[str, SearchParamSpec]]
RESIDUAL_PARAM_REGISTRY: dict[str, dict[str, SearchParamSpec]]


def _coerce_legacy_param_name_list(value: Any, *, section: str, owner: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"search_space.{section}.{owner} must be a mapping of param specs (or legacy list of names)"
        )
    out = tuple(str(item) for item in value)
    if any(not item.strip() for item in out):
        raise ValueError(f"search_space.{section}.{owner} contains an empty parameter name")
    return out


def _normalize_selector_specs(
    value: Any,
    *,
    section: str,
    owner: str,
    fallback_specs: dict[str, SearchParamSpec] | None,
) -> dict[str, SearchParamSpec]:
    if isinstance(value, list):
        names = _coerce_legacy_param_name_list(value, section=section, owner=owner)
        if fallback_specs is None:
            raise ValueError(
                f"search_space.{section}.{owner} legacy selector lists require default fallback specs"
            )
        missing = sorted(set(names).difference(fallback_specs))
        if missing:
            raise ValueError(
                f"search_space.{section}.{owner} contains unknown parameter(s): {', '.join(missing)}"
            )
        return {name: deepcopy(fallback_specs[name]) for name in names}
    if not isinstance(value, dict):
        raise ValueError(
            f"search_space.{section}.{owner} must be a mapping of parameter specs"
        )
    return {
        str(name): _normalize_param_spec(spec, section=section, owner=owner, name=str(name))
        for name, spec in value.items()
    }


def normalize_search_space_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        raise ValueError("search_space.yaml is empty")
    required_sections = {"models", "residual"}
    missing = required_sections.difference(payload)
    if missing:
        raise ValueError(
            "search_space.yaml must contain top-level sections: models and residual"
        )
    model_fallback = globals().get("MODEL_PARAM_REGISTRY")
    residual_fallback = globals().get("RESIDUAL_PARAM_REGISTRY")
    training_global_fallback = globals().get("TRAINING_PARAM_REGISTRY")

    models_payload = payload.get("models")
    if not isinstance(models_payload, dict):
        raise ValueError("search_space.models must be a mapping")
    models = {
        str(model_name): _normalize_selector_specs(
            model_specs,
            section="models",
            owner=str(model_name),
            fallback_specs=(None if model_fallback is None else model_fallback.get(str(model_name))),
        )
        for model_name, model_specs in models_payload.items()
    }
    unknown_models = sorted(set(models).difference(SUPPORTED_AUTO_MODEL_NAMES))
    if unknown_models:
        raise ValueError(
            "search_space.models contains unsupported learned model(s): "
            + ", ".join(unknown_models)
        )

    residual_payload = payload.get("residual")
    if not isinstance(residual_payload, dict):
        raise ValueError("search_space.residual must be a mapping")
    residual = {
        str(model_name): _normalize_selector_specs(
            model_specs,
            section="residual",
            owner=str(model_name),
            fallback_specs=(None if residual_fallback is None else residual_fallback.get(str(model_name))),
        )
        for model_name, model_specs in residual_payload.items()
    }
    unknown_residual = sorted(set(residual).difference(SUPPORTED_RESIDUAL_MODELS))
    if unknown_residual:
        raise ValueError(
            "search_space.residual contains unsupported residual model(s): "
            + ", ".join(unknown_residual)
        )

    training_payload = payload.get("training")
    if training_payload is None:
        training = {"global": {}, "per_model": {}}
    elif isinstance(training_payload, list):
        training = {
            "global": _normalize_selector_specs(
                training_payload,
                section="training",
                owner="global",
                fallback_specs=training_global_fallback,
            ),
            "per_model": {},
        }
    elif isinstance(training_payload, dict):
        if "global" in training_payload or "per_model" in training_payload:
            global_payload = training_payload.get("global", {})
            per_model_payload = training_payload.get("per_model", {})
        else:
            global_payload = training_payload
            per_model_payload = {}
        global_specs = _normalize_selector_specs(
            global_payload,
            section="training",
            owner="global",
            fallback_specs=training_global_fallback,
        )
        if not isinstance(per_model_payload, dict):
            raise ValueError("search_space.training.per_model must be a mapping")
        per_model_specs = {
            str(model_name): _normalize_selector_specs(
                spec_payload,
                section="training.per_model",
                owner=str(model_name),
                fallback_specs=global_specs,
            )
            for model_name, spec_payload in per_model_payload.items()
        }
        training = {"global": global_specs, "per_model": per_model_specs}
    else:
        raise ValueError("search_space.training must be a mapping or legacy list")

    fixed_training = sorted(set(training["global"]).intersection(FIXED_TRAINING_KEYS))
    if fixed_training:
        raise ValueError(
            "search_space.training.global contains fixed, non-tunable parameter(s): "
            + ", ".join(fixed_training)
        )
    unknown_training = sorted(
        set(training["global"]).difference({
            key for key in (training_global_fallback or {})
        })
    )
    if training_global_fallback is not None and unknown_training:
        raise ValueError(
            "search_space.training.global contains unknown parameter(s): "
            + ", ".join(unknown_training)
        )
    unknown_training_models = sorted(
        set(training["per_model"]).difference(SUPPORTED_AUTO_MODEL_NAMES)
    )
    if unknown_training_models:
        raise ValueError(
            "search_space.training.per_model contains unsupported model(s): "
            + ", ".join(unknown_training_models)
        )
    missing_training_models = sorted(
        set(SUPPORTED_AUTO_MODEL_NAMES).difference(training["per_model"])
    )
    for model_name in missing_training_models:
        training["per_model"][model_name] = deepcopy(training["global"])
    for model_name, specs in training["per_model"].items():
        spec_keys = set(specs)
        global_keys = set(training["global"])
        extra = sorted(spec_keys.difference(global_keys))
        missing = sorted(global_keys.difference(spec_keys))
        if extra:
            raise ValueError(
                f"search_space.training.per_model.{model_name} cannot introduce new parameter(s): {', '.join(extra)}"
            )
        if missing:
            raise ValueError(
                f"search_space.training.per_model.{model_name} must define every training selector explicitly; missing: {', '.join(missing)}"
            )
    if "learning_rate" in training["global"]:
        overlaps = sorted(
            model_name for model_name, specs in models.items() if "learning_rate" in specs
        )
        if overlaps:
            raise ValueError(
                "search_space.training.global.learning_rate overlaps with model-level learning_rate selector(s): "
                + ", ".join(overlaps)
            )

    return {
        "models": models,
        "training": training,
        "residual": residual,
    }


def load_search_space_contract(repo_root: Path) -> SearchSpaceContract:
    path = (repo_root / SEARCH_SPACE_FILENAME).resolve()
    text = path.read_text(encoding="utf-8")
    payload = normalize_search_space_payload(_read_yaml(path))
    return SearchSpaceContract(path=path, payload=payload, sha256=_hash_text(text))


def _default_search_space_contract() -> SearchSpaceContract:
    return load_search_space_contract(Path(__file__).resolve().parents[1])


_DEFAULT_CONTRACT = _default_search_space_contract()
MODEL_PARAM_REGISTRY = {
    model_name: deepcopy(specs)
    for model_name, specs in _DEFAULT_CONTRACT.payload["models"].items()
}
TRAINING_PARAM_REGISTRY = deepcopy(_DEFAULT_CONTRACT.payload["training"]["global"])
TRAINING_PARAM_REGISTRY_BY_MODEL = {
    model_name: deepcopy(_DEFAULT_CONTRACT.payload["training"]["per_model"][model_name])
    for model_name in sorted(SUPPORTED_AUTO_MODEL_NAMES)
}
RESIDUAL_PARAM_REGISTRY = {
    model_name: deepcopy(specs)
    for model_name, specs in _DEFAULT_CONTRACT.payload["residual"].items()
}


def training_param_registry_for_model(
    model_name: str | None,
    *,
    search_space_payload: dict[str, Any] | None = None,
) -> dict[str, SearchParamSpec]:
    if search_space_payload is None:
        if model_name is None:
            return deepcopy(TRAINING_PARAM_REGISTRY)
        return deepcopy(TRAINING_PARAM_REGISTRY_BY_MODEL[model_name])
    global_specs = search_space_payload["training"]["global"]
    if model_name is None:
        return deepcopy(global_specs)
    return deepcopy(search_space_payload["training"]["per_model"][model_name])


def training_range_source_for_model(
    model_name: str | None,
    *,
    search_space_payload: dict[str, Any] | None = None,
) -> str:
    if model_name is None:
        return GLOBAL_TRAINING_RANGE_SOURCE
    if search_space_payload is None:
        if model_name not in TRAINING_PARAM_REGISTRY_BY_MODEL:
            return GLOBAL_TRAINING_RANGE_SOURCE
        return f"model_override:{model_name}"
    if model_name not in search_space_payload["training"]["per_model"]:
        return GLOBAL_TRAINING_RANGE_SOURCE
    return f"model_override:{model_name}"


def _suggest_from_spec(
    trial: optuna.Trial,
    name: str,
    spec: SearchParamSpec,
) -> Any:
    spec_type = spec["type"]
    if spec_type == "categorical":
        return trial.suggest_categorical(name, deepcopy(spec["choices"]))
    if spec_type == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if spec_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    raise ValueError(f"Unsupported search-space type: {spec_type}")


def suggest_model_params(
    model_name: str,
    selected_names: tuple[str, ...],
    trial: optuna.Trial,
    *,
    param_specs: dict[str, SearchParamSpec] | None = None,
    name_prefix: str = "",
) -> dict[str, Any]:
    registry = MODEL_PARAM_REGISTRY[model_name] if param_specs is None else param_specs
    return {
        name: _suggest_from_spec(trial, f"{name_prefix}{name}", registry[name])
        for name in selected_names
    }


def suggest_residual_params(
    model_name: str,
    selected_names: tuple[str, ...],
    trial: optuna.Trial,
    *,
    param_specs: dict[str, SearchParamSpec] | None = None,
    name_prefix: str = "",
) -> dict[str, Any]:
    registry = RESIDUAL_PARAM_REGISTRY[model_name] if param_specs is None else param_specs
    suggested = DEFAULT_RESIDUAL_PARAMS_BY_MODEL[model_name].copy()
    for name in selected_names:
        suggested[name] = _suggest_from_spec(trial, f"{name_prefix}{name}", registry[name])
    return suggested


def suggest_training_params(
    selected_names: tuple[str, ...],
    trial: optuna.Trial,
    *,
    model_name: str | None = None,
    param_specs: dict[str, SearchParamSpec] | None = None,
    name_prefix: str = "",
) -> dict[str, Any]:
    fixed = sorted(set(selected_names).intersection(FIXED_TRAINING_KEYS))
    if fixed:
        raise ValueError(
            "selected training parameter(s) are fixed and non-tunable: "
            + ", ".join(fixed)
        )
    registry = (
        training_param_registry_for_model(model_name)
        if param_specs is None
        else param_specs
    )
    return {
        name: _suggest_from_spec(trial, f"{name_prefix}{name}", registry[name])
        for name in selected_names
    }
