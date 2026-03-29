from __future__ import annotations

from typing import Any

DIRECT_STAGE_MODEL_NAMES = frozenset({"ARIMA", "ES", "xgboost", "lightgbm"})

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
    *DIRECT_STAGE_MODEL_NAMES,
}

BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "ARIMA": {
        "order": {
            "type": "categorical",
            "choices": [[1, 0, 0], [1, 1, 0], [2, 1, 0]],
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
            "choices": [[1, 2, 3], [1, 2, 3, 6, 12]],
        },
        "n_estimators": {"type": "categorical", "choices": [16, 32, 64]},
        "max_depth": {"type": "int", "low": 2, "high": 6, "step": 1},
    },
    "lightgbm": {
        "lags": {
            "type": "categorical",
            "choices": [[1, 2, 3], [1, 2, 3, 6, 12]],
        },
        "n_estimators": {"type": "categorical", "choices": [32, 64, 96]},
        "max_depth": {"type": "categorical", "choices": [4, 6, -1]},
        "num_leaves": {"type": "categorical", "choices": [15, 31, 63]},
        "min_child_samples": {"type": "categorical", "choices": [10, 20, 40]},
        "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
    },
}


def is_direct_stage_model(model_name: str) -> bool:
    return model_name in DIRECT_STAGE_MODEL_NAMES
