from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Literal
import hashlib

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


def default_residual_params(model_name: str) -> dict[str, Any]:
    normalized = str(model_name).lower()
    if normalized not in RESIDUAL_DEFAULTS:
        raise KeyError(f"Unsupported residual model defaults: {normalized}")
    return RESIDUAL_DEFAULTS[normalized].copy()

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


def load_search_space_contract(repo_root: Path) -> SearchSpaceContract:
    path = (repo_root / SEARCH_SPACE_FILENAME).resolve()
    text = path.read_text(encoding="utf-8")
    payload = _read_yaml(path)
    return SearchSpaceContract(path=path, payload=payload, sha256=_hash_text(text))


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


def _categorical(options: list[Any]):
    def _apply(trial: optuna.Trial, name: str) -> Any:
        return trial.suggest_categorical(name, options)

    return _apply


def _int(low: int, high: int, step: int = 1):
    def _apply(trial: optuna.Trial, name: str) -> int:
        return trial.suggest_int(name, low, high, step=step)

    return _apply


def _float(low: float, high: float, *, log: bool = False):
    def _apply(trial: optuna.Trial, name: str) -> float:
        return trial.suggest_float(name, low, high, log=log)

    return _apply


MODEL_PARAM_REGISTRY = {
    "RNN": {
        "encoder_hidden_size": _categorical([16, 32, 64, 128]),
        "encoder_n_layers": _int(1, 3),
        "context_size": _categorical([5, 10, 50]),
        "decoder_hidden_size": _categorical([16, 32, 64, 128]),
    },
    "GRU": {
        "encoder_hidden_size": _categorical([16, 32, 64, 128]),
        "encoder_n_layers": _int(1, 3),
        "context_size": _categorical([5, 10, 50]),
        "decoder_hidden_size": _categorical([16, 32, 64, 128]),
    },
    "TFT": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "dropout": _float(0.0, 0.3),
        "n_head": _categorical([4, 8]),
    },
    "VanillaTransformer": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "dropout": _float(0.0, 0.3),
        "encoder_layers": _int(1, 4),
        "n_head": _categorical([4, 8]),
    },
    "Informer": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "dropout": _float(0.0, 0.3),
        "factor": _int(1, 5),
        "n_head": _categorical([4, 8]),
    },
    "Autoformer": {
        "hidden_size": _categorical([64, 128, 256]),
        "n_head": _categorical([4, 8]),
        "encoder_layers": _int(1, 3),
        "decoder_layers": _int(1, 2),
        "factor": _categorical([1, 3, 5]),
        "MovingAvg_window": _categorical([5, 13, 25]),
        "conv_hidden_size": _categorical([32, 64, 128]),
        "dropout": _categorical([0.0, 0.1, 0.2, 0.3]),
    },
    "FEDformer": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "dropout": _float(0.0, 0.3),
        "modes": _categorical([16, 32, 64]),
        "n_head": _categorical([4, 8]),
    },
    "PatchTST": {
        "hidden_size": _categorical([64, 128]),
        "n_heads": _categorical([4, 8]),
        "encoder_layers": _int(2, 3),
        "linear_hidden_size": _categorical([128, 256, 512]),
        "patch_len": _categorical([4, 8, 16]),
        "stride": _categorical([2, 4, 8]),
        "dropout": _categorical([0.0, 0.2, 0.3]),
        "fc_dropout": _categorical([0.0, 0.2]),
        "attn_dropout": _categorical([0.0, 0.1]),
        "revin": _categorical([True, False]),
    },
    "LSTM": {
        "encoder_hidden_size": _categorical([64, 128, 256]),
        "encoder_n_layers": _int(1, 3),
        "inference_input_size": _categorical([-1, 24, 48, 96]),
        "encoder_dropout": _categorical([0.1, 0.2, 0.3]),
        "decoder_hidden_size": _categorical([32, 64, 128]),
        "decoder_layers": _int(1, 3),
    },
    "TCN": {
        "encoder_hidden_size": _categorical([16, 32, 64, 128]),
        "context_size": _categorical([5, 10, 50]),
        "decoder_hidden_size": _categorical([16, 32, 64, 128]),
        "kernel_size": _categorical([2, 3, 5]),
    },
    "DeepAR": {
        "lstm_hidden_size": _categorical([16, 32, 64, 128]),
        "lstm_n_layers": _int(1, 3),
        "lstm_dropout": _float(0.0, 0.3),
        "decoder_hidden_size": _categorical([16, 32, 64]),
    },
    "DilatedRNN": {
        "cell_type": _categorical(["GRU", "LSTM"]),
        "encoder_hidden_size": _categorical([16, 32, 64, 128]),
        "context_size": _categorical([5, 10, 50]),
        "decoder_hidden_size": _categorical([16, 32, 64, 128]),
    },
    "BiTCN": {
        "hidden_size": _categorical([16, 32, 64, 128]),
        "dropout": _float(0.0, 0.3),
    },
    "xLSTM": {
        "encoder_hidden_size": _categorical([16, 32, 64, 128]),
        "encoder_n_blocks": _int(1, 3),
        "decoder_hidden_size": _categorical([16, 32, 64, 128]),
        "encoder_dropout": _float(0.0, 0.3),
    },
    "MLP": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "num_layers": _int(1, 4),
    },
    "NBEATS": {
        "n_blocks": _categorical([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        "mlp_units": _categorical(
            [
                [[64, 64], [64, 64], [64, 64]],
                [[128, 128], [128, 128], [128, 128]],
            ]
        ),
        "dropout_prob_theta": _float(0.0, 0.3),
    },
    "NBEATSx": {
        "n_blocks": _categorical([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        "mlp_units": _categorical(
            [
                [[64, 64], [64, 64], [64, 64]],
                [[128, 128], [128, 128], [128, 128]],
            ]
        ),
        "dropout_prob_theta": _float(0.0, 0.3),
    },
    "NHITS": {
        "n_pool_kernel_size": _categorical(
            [[2, 2, 1], [4, 2, 1], [8, 4, 1]]
        ),
        "n_freq_downsample": _categorical(
            [[4, 2, 1], [8, 4, 1], [12, 3, 1]]
        ),
        "n_blocks": _categorical([[1, 1, 1], [1, 2, 2], [2, 2, 2]]),
        "mlp_units": _categorical(
            [
                [[128, 128], [128, 128], [128, 128]],
                [[256, 256], [256, 256], [256, 256]],
                [[512, 512], [512, 512], [512, 512]],
            ]
        ),
        "dropout_prob_theta": _categorical([0.0, 0.1, 0.2]),
        "activation": _categorical(
            ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]
        ),
    },
    "DLinear": {
        "moving_avg_window": _categorical([5, 9, 13, 25, 51]),
    },
    "NLinear": {
        "learning_rate": _float(1e-4, 1e-1, log=True),
    },
    "TiDE": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "decoder_output_dim": _categorical([8, 16, 32]),
        "temporal_decoder_dim": _categorical([8, 16, 32]),
        "num_encoder_layers": _int(1, 3),
    },
    "DeepNPTS": {
        "hidden_size": _categorical([16, 32, 64, 128]),
        "dropout": _float(0.0, 0.3),
        "n_layers": _int(1, 4),
    },
    "KAN": {
        "grid_size": _categorical([3, 5, 8]),
        "spline_order": _categorical([2, 3, 4]),
        "hidden_size": _categorical([16, 32, 64, 128]),
    },
    "DeformTime": {
        "d_model": _categorical([16, 32, 64]),
        "n_heads": _categorical([2, 4, 8]),
        "e_layers": _int(1, 3),
        "patch_len": _categorical([4, 8, 16]),
        "dropout": _float(0.0, 0.3),
    },
    "DeformableTST": {
        "dims": _categorical([[64, 128, 256, 512], [32, 64, 128, 256]]),
        "depths": _categorical([[1, 1, 3, 1], [1, 1, 2, 1]]),
        "drop": _float(0.0, 0.3),
        "heads": _categorical([[4, 8, 16, 32], [2, 4, 8, 16]]),
    },
    "iTransformer": {
        "hidden_size": _categorical([64, 128]),
        "n_heads": _categorical([4, 8]),
        "e_layers": _int(1, 2),
        "d_ff": _categorical([128, 256, 512]),
        "d_layers": _int(1, 2),
        "factor": _categorical([1, 3]),
        "dropout": _categorical([0.0, 0.2, 0.3]),
        "use_norm": _categorical([True, False]),
    },
    "TimeLLM": {
        "patch_len": _categorical([8, 16, 24]),
        "stride": _categorical([4, 8]),
        "d_ff": _categorical([64, 128, 256]),
        "top_k": _categorical([3, 5, 7]),
        "d_model": _categorical([16, 32, 64]),
        "n_heads": _categorical([4, 8]),
        "dropout": _float(0.0, 0.3),
    },
    "TimeXer": {
        "patch_len": _categorical([8, 16, 24]),
        "hidden_size": _categorical([32, 64, 128, 256]),
        "n_heads": _categorical([4, 8]),
        "e_layers": _int(1, 4),
    },
    "TimesNet": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "conv_hidden_size": _categorical([32, 64, 128]),
        "top_k": _categorical([3, 5, 7]),
        "encoder_layers": _int(1, 4),
    },
    "StemGNN": {
        "n_stacks": _categorical([1, 2, 3]),
        "multi_layer": _categorical([3, 5, 7]),
        "dropout_rate": _float(0.0, 0.3),
    },
    "TSMixer": {
        "n_block": _categorical([1, 2, 3]),
        "ff_dim": _categorical([16, 32, 64, 128]),
        "dropout": _float(0.0, 0.3),
    },
    "TSMixerx": {
        "n_block": _categorical([1, 2]),
        "ff_dim": _categorical([32, 64, 128]),
        "dropout": _categorical([0.05, 0.1, 0.2]),
    },
    "MLPMultivariate": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "num_layers": _int(1, 4),
    },
    "SOFTS": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "d_core": _categorical([16, 32, 64]),
        "e_layers": _int(1, 4),
        "d_ff": _categorical([64, 128, 256]),
    },
    "TimeMixer": {
        "d_model": _categorical([16, 32, 64, 128]),
        "d_ff": _categorical([32, 64, 128, 256]),
        "down_sampling_layers": _categorical([1, 2, 3]),
        "top_k": _categorical([3, 5, 7]),
    },
    "ModernTCN": {
        "patch_size": _categorical([8, 16, 24]),
        "patch_stride": _categorical([4, 8, 12]),
        "ffn_ratio": _categorical([1, 2, 4]),
        "large_size": _categorical([(5, 5, 3, 3), (7, 7, 5, 5)]),
        "dims": _categorical([(8, 8, 8, 8), (16, 16, 16, 16)]),
    },
    "DUET": {
        "n_block": _categorical([1, 2, 3]),
        "hidden_size": _categorical([32, 64, 128]),
        "ff_dim": _categorical([64, 128, 256]),
        "moving_avg_window": _categorical([3, 5, 7]),
        "dropout": _float(0.0, 0.3),
        "revin": _categorical([True, False]),
    },
    "Mamba": {
        "hidden_size": _categorical([16, 32, 64, 128]),
        "n_block": _categorical([1, 2, 3]),
        "expand_ratio": _categorical([1, 2, 4]),
        "kernel_size": _categorical([3, 5]),
        "dropout": _float(0.0, 0.3),
    },
    "SMamba": {
        "hidden_size": _categorical([16, 32, 64, 128]),
        "n_block": _categorical([1, 2, 3]),
        "expand_ratio": _categorical([1, 2, 4]),
        "kernel_size": _categorical([3, 5]),
        "dropout": _float(0.0, 0.3),
        "revin": _categorical([True, False]),
    },
    "CMamba": {
        "hidden_size": _categorical([16, 32, 64, 128]),
        "ff_dim": _categorical([32, 64, 128, 256]),
        "n_block": _categorical([1, 2, 3]),
        "expand_ratio": _categorical([1, 2, 4]),
        "kernel_size": _categorical([3, 5]),
        "dropout": _float(0.0, 0.3),
        "revin": _categorical([True, False]),
    },
    "xLSTMMixer": {
        "hidden_size": _categorical([16, 32, 64, 128]),
        "n_block": _categorical([1, 2, 3]),
        "ff_dim": _categorical([32, 64, 128, 256]),
        "dropout": _float(0.0, 0.3),
        "encoder_n_blocks": _categorical([1, 2, 3]),
        "encoder_dropout": _float(0.0, 0.3),
        "revin": _categorical([True, False]),
    },
    "RMoK": {
        "taylor_order": _categorical([2, 3, 4]),
        "jacobi_degree": _categorical([2, 3, 4]),
        "wavelet_function": _categorical(["haar", "db2"]),
        "dropout": _float(0.0, 0.3),
    },
    "XLinear": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "temporal_ff": _categorical([64, 128, 256]),
        "channel_ff": _categorical([64, 128, 256]),
        "temporal_dropout": _float(0.0, 0.3),
    },
}

RESIDUAL_PARAM_REGISTRY = {
    "xgboost": {
        "n_estimators": _categorical([16, 32, 64, 128]),
        "max_depth": _int(2, 6),
        "learning_rate": _float(1e-3, 0.3, log=True),
        "subsample": _float(0.5, 1.0),
        "colsample_bytree": _float(0.5, 1.0),
    },
    "randomforest": {
        "n_estimators": _categorical([64, 128, 200, 300]),
        "max_depth": _categorical([4, 6, 8, 12, None]),
        "min_samples_leaf": _categorical([1, 2, 4, 8]),
        "max_features": _categorical(["sqrt", "log2", 1.0]),
    },
    "lightgbm": {
        "n_estimators": _categorical([32, 64, 96, 128]),
        "max_depth": _categorical([4, 6, 8, -1]),
        "learning_rate": _float(1e-3, 0.1, log=True),
        "num_leaves": _categorical([15, 31, 63]),
        "min_child_samples": _categorical([10, 20, 40]),
        "feature_fraction": _float(0.6, 1.0),
    },
}

TRAINING_PARAM_REGISTRY = {
    "input_size": _categorical([24, 36, 48, 64, 72, 96]),
    "batch_size": _categorical([16, 32, 64, 128]),
    "valid_batch_size": _categorical([16, 32, 64, 128]),
    "windows_batch_size": _categorical([256, 512, 1024, 2048]),
    "inference_windows_batch_size": _categorical([256, 512, 1024, 2048]),
    "learning_rate": _float(3e-4, 1e-2, log=True),
    "scaler_type": _categorical([None, "robust", "standard", "identity"]),
    "model_step_size": _categorical([1, 4, 8, 12]),
}

GLOBAL_TRAINING_RANGE_SOURCE = "global_fallback"
_PATCHTST_TRAINING_PARAM_REGISTRY = {
    **TRAINING_PARAM_REGISTRY,
    "input_size": _categorical([24, 36, 48, 64]),
    "batch_size": _categorical([16, 32, 64]),
    "valid_batch_size": _categorical([16, 32, 64]),
    "windows_batch_size": _categorical([256, 512, 1024]),
    "inference_windows_batch_size": _categorical([256, 512, 1024]),
    "learning_rate": _float(3e-4, 9e-3, log=True),
}
_TSMIXERX_TRAINING_PARAM_REGISTRY = {
    **TRAINING_PARAM_REGISTRY,
    "input_size": _categorical([48, 64]),
    "batch_size": _categorical([16, 32]),
    "valid_batch_size": _categorical([32, 64, 128]),
    "windows_batch_size": _categorical([512]),
    "inference_windows_batch_size": _categorical([256, 512, 1024]),
    "learning_rate": _float(4e-4, 1e-3, log=True),
    "scaler_type": _categorical(["robust"]),
    "model_step_size": _categorical([1]),
}
_ITRANSFORMER_TRAINING_PARAM_REGISTRY = {
    **TRAINING_PARAM_REGISTRY,
    "input_size": _categorical([24, 36, 48, 64, 72]),
    "batch_size": _categorical([16, 32, 64]),
    "valid_batch_size": _categorical([16, 32, 64]),
    "windows_batch_size": _categorical([256, 512, 1024]),
    "inference_windows_batch_size": _categorical([256, 512, 1024]),
    "learning_rate": _float(4e-4, 7e-3, log=True),
    "scaler_type": _categorical([None, "robust", "standard"]),
}
_LSTM_TRAINING_PARAM_REGISTRY = {
    **TRAINING_PARAM_REGISTRY,
    "input_size": _categorical([24, 48, 64, 96]),
    "batch_size": _categorical([16, 32, 64]),
    "valid_batch_size": _categorical([16, 32, 64]),
    "windows_batch_size": _categorical([128, 256, 512, 1024]),
    "inference_windows_batch_size": _categorical([256, 512, 1024]),
    "learning_rate": _float(1e-3, 1e-2, log=True),
    "scaler_type": _categorical(["robust", "standard", "identity"]),
    "model_step_size": _categorical([4, 8, 12]),
}
TRAINING_PARAM_REGISTRY_BY_MODEL = {
    model_name: TRAINING_PARAM_REGISTRY.copy()
    for model_name in sorted(SUPPORTED_AUTO_MODEL_NAMES)
}
TRAINING_PARAM_REGISTRY_BY_MODEL.update(
    {
        "PatchTST": _PATCHTST_TRAINING_PARAM_REGISTRY,
        "TSMixerx": _TSMIXERX_TRAINING_PARAM_REGISTRY,
        "iTransformer": _ITRANSFORMER_TRAINING_PARAM_REGISTRY,
        "LSTM": _LSTM_TRAINING_PARAM_REGISTRY,
    }
)


def training_param_registry_for_model(
    model_name: str | None,
) -> dict[str, Any]:
    if model_name is None:
        return TRAINING_PARAM_REGISTRY
    return TRAINING_PARAM_REGISTRY_BY_MODEL.get(model_name, TRAINING_PARAM_REGISTRY)


def training_range_source_for_model(model_name: str | None) -> str:
    if model_name is None:
        return GLOBAL_TRAINING_RANGE_SOURCE
    if model_name in TRAINING_PARAM_REGISTRY_BY_MODEL:
        return f"model_override:{model_name}"
    return GLOBAL_TRAINING_RANGE_SOURCE


def suggest_model_params(
    model_name: str, selected_names: tuple[str, ...], trial: optuna.Trial
) -> dict[str, Any]:
    registry = MODEL_PARAM_REGISTRY[model_name]
    return {name: registry[name](trial, name) for name in selected_names}


def suggest_residual_params(
    model_name: str, selected_names: tuple[str, ...], trial: optuna.Trial
) -> dict[str, Any]:
    registry = RESIDUAL_PARAM_REGISTRY[model_name]
    suggested = DEFAULT_RESIDUAL_PARAMS_BY_MODEL[model_name].copy()
    for name in selected_names:
        suggested[name] = registry[name](trial, name)
    return suggested


def suggest_training_params(
    selected_names: tuple[str, ...],
    trial: optuna.Trial,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    fixed = sorted(set(selected_names).intersection(FIXED_TRAINING_KEYS))
    if fixed:
        raise ValueError(
            "selected training parameter(s) are fixed and non-tunable: "
            + ", ".join(fixed)
        )
    suggested = {}
    registry = training_param_registry_for_model(model_name)
    for name in selected_names:
        suggested[name] = registry[name](trial, name)
    return suggested
