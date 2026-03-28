from __future__ import annotations

from typing import Any, Callable

SUPPORTED_BS_PREFORCAST_MODELS = {
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
    "Naive",
    "xgboost",
    "lightgbm",
    "ARIMA",
    "ES",
}

BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "ARIMA": {
        "order": {
            "type": "categorical",
            "choices": ["[1, 0, 0]", "[1, 1, 0]", "[2, 1, 0]"],
        },
        "include_mean": {"type": "categorical", "choices": [True, False]},
        "include_drift": {"type": "categorical", "choices": [False, True]},
    },
    "ES": {
        "trend": {"type": "categorical", "choices": [None, "add"]},
        "damped_trend": {"type": "categorical", "choices": [False, True]},
    },
    "xgboost": {
        "lags": {
            "type": "categorical",
            "choices": ["[1, 2, 3]", "[1, 2, 3, 6, 12]"],
        },
        "n_estimators": {"type": "categorical", "choices": [16, 32, 64]},
        "max_depth": {"type": "int", "low": 2, "high": 6, "step": 1},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2},
    },
    "lightgbm": {
        "lags": {
            "type": "categorical",
            "choices": ["[1, 2, 3]", "[1, 2, 3, 6, 12]"],
        },
        "n_estimators": {"type": "categorical", "choices": [32, 64, 96]},
        "max_depth": {"type": "categorical", "choices": [4, 6, -1]},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2},
        "num_leaves": {"type": "categorical", "choices": [15, 31, 63]},
        "min_child_samples": {"type": "categorical", "choices": [10, 20, 40]},
        "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
    },
}


def stage_search_space_payload(
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


def rewrite_search_space_error(message: str) -> str:
    return (
        message.replace("search_space.models.", "search_space.bs_preforcast_models.")
        .replace("search_space.training.", "search_space.bs_preforcast_training.")
        .replace("search_space.models ", "search_space.bs_preforcast_models ")
        .replace("search_space.training ", "search_space.bs_preforcast_training ")
    )


def normalize_bs_preforcast_sections(
    payload: dict[str, Any],
    *,
    normalize_model_section: Callable[..., dict[str, Any]],
    normalize_training_section: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bs_preforcast_models = normalize_model_section(
        payload.get("bs_preforcast_models"),
        section="bs_preforcast_models",
        allowed_models=SUPPORTED_BS_PREFORCAST_MODELS,
    )
    bs_preforcast_training = normalize_training_section(
        payload.get("bs_preforcast_training"),
        section="bs_preforcast_training",
        allowed_models=SUPPORTED_BS_PREFORCAST_MODELS,
    )
    if "learning_rate" in bs_preforcast_training["global"]:
        overlaps = sorted(
            model_name
            for model_name, specs in bs_preforcast_models.items()
            if "learning_rate" in specs
            and model_name not in {"xgboost", "lightgbm", "ARIMA", "ES"}
        )
        if overlaps:
            raise ValueError(
                "search_space.bs_preforcast_training.global.learning_rate overlaps with model-level learning_rate selector(s): "
                + ", ".join(overlaps)
            )
    return bs_preforcast_models, bs_preforcast_training
