from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Literal

import optuna
import yaml

import bs_preforcast.search_space as bs_preforcast_search_space

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
    "batch_size": 32,
    "valid_batch_size": 64,
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
    "max_steps": 1000,
    "val_size": 8,
    "val_check_steps": 100,
}
FIXED_TRAINING_KEYS = tuple(FIXED_TRAINING_VALUES)
GLOBAL_TRAINING_RANGE_SOURCE = "global_fallback"
SUPPORTED_BS_PREFORCAST_MODELS = bs_preforcast_search_space.SUPPORTED_BS_PREFORCAST_MODELS
BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY = (
    bs_preforcast_search_space.BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY
)

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


def _auto_default_param_specs(model_name: str) -> dict[str, SearchParamSpec] | None:
    if model_name == "TFT":
        return {
            "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
            "dropout": {"type": "categorical", "choices": [0.0, 0.1, 0.2, 0.3, 0.5]},
            "n_head": {"type": "categorical", "choices": [4, 8]},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "scaler_type": {"type": "categorical", "choices": [None, "robust", "standard"]},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
            "windows_batch_size": {"type": "categorical", "choices": [128, 256, 512, 1024]},
            "input_size_multiplier": {"type": "categorical", "choices": [1, 2, 3, 4, 5]},
        }
    try:
        import neuralforecast.auto as nf_auto
    except Exception:
        return None
    auto_name = f"Auto{model_name}"
    auto_cls = getattr(nf_auto, auto_name, None)
    default_config = getattr(auto_cls, "default_config", None)
    if not isinstance(default_config, dict):
        return None

    specs: dict[str, SearchParamSpec] = {}
    for key, value in default_config.items():
        if key in {
            "h",
            "loss",
            "valid_loss",
            "random_seed",
            "max_steps",
            "val_check_steps",
            "early_stop_patience_steps",
            "callbacks",
            "cpus",
            "gpus",
            "verbose",
            "alias",
        }:
            continue
        if value is None:
            specs[str(key)] = {"type": "categorical", "choices": [None]}
            continue
        if isinstance(value, list) and value:
            specs[str(key)] = {"type": "categorical", "choices": deepcopy(value)}
            continue
        categories = getattr(value, "categories", None)
        if categories is not None:
            specs[str(key)] = {
                "type": "categorical",
                "choices": deepcopy(list(categories)),
            }
            continue
        lower = getattr(value, "lower", None)
        upper = getattr(value, "upper", None)
        if isinstance(lower, int) and isinstance(upper, int):
            specs[str(key)] = {
                "type": "int",
                "low": int(lower),
                "high": int(upper),
                "step": int(getattr(value, "q", 1) or 1),
            }
            continue
        if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
            sampler_name = type(getattr(value, "sampler", None)).__name__.lower()
            spec: SearchParamSpec = {
                "type": "float",
                "low": float(lower),
                "high": float(upper),
            }
            if "log" in sampler_name:
                spec["log"] = True
            specs[str(key)] = spec
    return specs or None

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


def _known_model_param_names(model_name: str) -> set[str]:
    stage_only = globals().get("BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY") or {}
    if model_name in stage_only:
        return set(stage_only[model_name])
    try:
        import neuralforecast.models as nf_models
    except Exception:
        return set()
    model_cls = getattr(nf_models, model_name, None)
    if model_cls is None:
        return set()
    try:
        return {
            name
            for name in inspect.signature(model_cls.__init__).parameters
            if name != "self"
        }
    except (TypeError, ValueError):
        return set()


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
            known_params = _known_model_param_names(owner)
            truly_unknown = sorted(set(missing).difference(known_params))
            if truly_unknown:
                raise ValueError(
                    f"search_space.{section}.{owner} contains unknown parameter(s): {', '.join(truly_unknown)}"
                )
        resolved_specs = {
            name: deepcopy(fallback_specs.get(name, {"type": "categorical", "choices": [None]}))
            for name in names
        }
        return resolved_specs
    if not isinstance(value, dict):
        raise ValueError(
            f"search_space.{section}.{owner} must be a mapping of parameter specs"
        )
    return {
        str(name): _normalize_param_spec(spec, section=section, owner=owner, name=str(name))
        for name, spec in value.items()
    }


def _normalize_model_section(
    payload: Any,
    *,
    section: str,
    allowed_models: set[str] | None = None,
) -> dict[str, dict[str, SearchParamSpec]]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"search_space.{section} must be a mapping")
    model_fallback = globals().get("MODEL_PARAM_REGISTRY")
    residual_fallback = globals().get("RESIDUAL_PARAM_REGISTRY")
    stage_only_fallback = globals().get("BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY")
    models = {
        str(model_name): _normalize_selector_specs(
            model_specs,
            section=section,
            owner=str(model_name),
            fallback_specs=(
                (
                    None
                    if model_fallback is None
                    else model_fallback.get(str(model_name))
                )
                or (
                    residual_fallback.get(str(model_name))
                    if section == "bs_preforcast_models"
                    and residual_fallback is not None
                    else None
                )
                or (
                    stage_only_fallback.get(str(model_name))
                    if section == "bs_preforcast_models"
                    and stage_only_fallback is not None
                    else None
                )
                or _auto_default_param_specs(str(model_name))
            ),
        )
        for model_name, model_specs in payload.items()
    }
    supported = SUPPORTED_AUTO_MODEL_NAMES if allowed_models is None else allowed_models
    unknown_models = sorted(set(models).difference(supported))
    if unknown_models:
        raise ValueError(
            f"search_space.{section} contains unsupported learned model(s): "
            + ", ".join(unknown_models)
        )
    return models


def _normalize_training_section(
    payload: Any,
    *,
    section: str,
    allowed_models: set[str] | None = None,
) -> dict[str, dict[str, dict[str, SearchParamSpec]]]:
    training_global_fallback = globals().get("TRAINING_PARAM_REGISTRY")
    supported = SUPPORTED_AUTO_MODEL_NAMES if allowed_models is None else allowed_models
    if payload is None:
        training = {"global": {}, "per_model": {}}
    elif isinstance(payload, list):
        training = {
            "global": _normalize_selector_specs(
                payload,
                section=section,
                owner="global",
                fallback_specs=training_global_fallback,
            ),
            "per_model": {},
        }
    elif isinstance(payload, dict):
        if "global" in payload or "per_model" in payload:
            global_payload = payload.get("global", {})
            per_model_payload = payload.get("per_model", {})
        else:
            global_payload = payload
            per_model_payload = {}
        global_specs = _normalize_selector_specs(
            global_payload,
            section=section,
            owner="global",
            fallback_specs=training_global_fallback,
        )
        if not isinstance(per_model_payload, dict):
            raise ValueError(f"search_space.{section}.per_model must be a mapping")
        per_model_specs = {
            str(model_name): _normalize_selector_specs(
                spec_payload,
                section=f"{section}.per_model",
                owner=str(model_name),
                fallback_specs=global_specs,
            )
            for model_name, spec_payload in per_model_payload.items()
        }
        training = {"global": global_specs, "per_model": per_model_specs}
    else:
        raise ValueError(f"search_space.{section} must be a mapping or legacy list")

    fixed_training = sorted(set(training["global"]).intersection(FIXED_TRAINING_KEYS))
    if fixed_training:
        raise ValueError(
            f"search_space.{section}.global contains fixed, non-tunable parameter(s): "
            + ", ".join(fixed_training)
        )
    unknown_training = sorted(
        set(training["global"]).difference({
            key for key in (training_global_fallback or {})
        })
    )
    if training_global_fallback is not None and unknown_training:
        raise ValueError(
            f"search_space.{section}.global contains unknown parameter(s): "
            + ", ".join(unknown_training)
        )
    unknown_training_models = sorted(set(training["per_model"]).difference(supported))
    if unknown_training_models:
        raise ValueError(
            f"search_space.{section}.per_model contains unsupported model(s): "
            + ", ".join(unknown_training_models)
        )
    missing_training_models = sorted(set(supported).difference(training["per_model"]))
    for model_name in missing_training_models:
        training["per_model"][model_name] = deepcopy(training["global"])
    for model_name, specs in training["per_model"].items():
        spec_keys = set(specs)
        global_keys = set(training["global"])
        extra = sorted(spec_keys.difference(global_keys))
        missing = sorted(global_keys.difference(spec_keys))
        if extra:
            raise ValueError(
                f"search_space.{section}.per_model.{model_name} cannot introduce new parameter(s): {', '.join(extra)}"
            )
        if missing:
            raise ValueError(
                f"search_space.{section}.per_model.{model_name} must define every training selector explicitly; missing: {', '.join(missing)}"
            )
    if "learning_rate" in training["global"]:
        return training
    return training


def normalize_search_space_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        raise ValueError("search_space.yaml is empty")
    required_sections = {"models", "residual"}
    missing = required_sections.difference(payload)
    if missing:
        raise ValueError(
            "search_space.yaml must contain top-level sections: models and residual"
        )
    residual_fallback = globals().get("RESIDUAL_PARAM_REGISTRY")

    models = _normalize_model_section(payload.get("models"), section="models")

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

    training = _normalize_training_section(payload.get("training"), section="training")
    if "learning_rate" in training["global"]:
        overlaps = sorted(
            model_name for model_name, specs in models.items() if "learning_rate" in specs
        )
        if overlaps:
            raise ValueError(
                "search_space.training.global.learning_rate overlaps with model-level learning_rate selector(s): "
                + ", ".join(overlaps)
            )

    bs_preforcast_models, bs_preforcast_training = bs_preforcast_search_space.normalize_bs_preforcast_sections(
        payload,
        normalize_model_section=_normalize_model_section,
        normalize_training_section=_normalize_training_section,
    )

    return {
        "models": models,
        "training": training,
        "residual": residual,
        "bs_preforcast_models": bs_preforcast_models,
        "bs_preforcast_training": bs_preforcast_training,
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
    section: str = "training",
) -> dict[str, SearchParamSpec]:
    if search_space_payload is None:
        if model_name is None:
            return deepcopy(TRAINING_PARAM_REGISTRY)
        return deepcopy(TRAINING_PARAM_REGISTRY_BY_MODEL[model_name])
    global_specs = search_space_payload[section]["global"]
    if model_name is None:
        return deepcopy(global_specs)
    return deepcopy(search_space_payload[section]["per_model"][model_name])


def training_range_source_for_model(
    model_name: str | None,
    *,
    search_space_payload: dict[str, Any] | None = None,
    section: str = "training",
) -> str:
    if model_name is None:
        return GLOBAL_TRAINING_RANGE_SOURCE
    if search_space_payload is None:
        if model_name not in TRAINING_PARAM_REGISTRY_BY_MODEL:
            return GLOBAL_TRAINING_RANGE_SOURCE
        return f"model_override:{model_name}"
    if model_name not in search_space_payload[section]["per_model"]:
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
