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
    "NonstationaryTransformer",
    "StemGNN",
    "TSMixer",
    "TSMixerx",
    "MLPMultivariate",
    "SOFTS",
    "TimeMixer",
    "Mamba",
    "SMamba",
    "CMamba",
    "xLSTMMixer",
    "RMoK",
    "XLinear",
}
SUPPORTED_RESIDUAL_MODELS = {"xgboost"}
DEFAULT_OPTUNA_NUM_TRIALS = 5
DEFAULT_OPTUNA_STUDY_DIRECTION = "minimize"

DEFAULT_RESIDUAL_PARAMS = {
    "n_estimators": 32,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

DEFAULT_TRAINING_PARAMS = {
    "input_size": 64,
    "season_length": 52,
    "batch_size": 32,
    "valid_batch_size": 32,
    "windows_batch_size": 1024,
    "inference_windows_batch_size": 1024,
    "learning_rate": 0.001,
    "max_steps": 1000,
    "val_size": 12,
    "val_check_steps": 100,
}

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


def optuna_num_trials() -> int:
    return int(os.environ.get("NEURALFORECAST_OPTUNA_NUM_TRIALS", DEFAULT_OPTUNA_NUM_TRIALS))


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
        "hidden_size": _categorical([32, 64, 128, 256]),
        "dropout": _float(0.0, 0.3),
        "factor": _int(1, 5),
        "n_head": _categorical([4, 8]),
    },
    "FEDformer": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "dropout": _float(0.0, 0.3),
        "modes": _categorical([16, 32, 64]),
        "n_head": _categorical([4, 8]),
    },
    "PatchTST": {
        "hidden_size": _categorical([16, 64, 128, 256]),
        "n_heads": _categorical([4, 8, 16]),
        "encoder_layers": _int(1, 4),
        "patch_len": _categorical([8, 16, 24]),
        "dropout": _float(0.0, 0.3),
    },
    "LSTM": {
        "encoder_hidden_size": _categorical([16, 32, 64, 128]),
        "decoder_hidden_size": _categorical([16, 32, 64, 128]),
        "encoder_n_layers": _int(1, 3),
        "context_size": _categorical([5, 10, 50]),
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
            [[2, 2, 1], [1, 1, 1], [2, 2, 2], [4, 2, 1]]
        ),
        "n_freq_downsample": _categorical(
            [[24, 12, 1], [60, 8, 1], [40, 20, 1], [1, 1, 1]]
        ),
        "dropout_prob_theta": _float(0.0, 0.3),
    },
    "DLinear": {
        "moving_avg_window": _categorical([3, 5, 7, 25]),
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
    "iTransformer": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "n_heads": _categorical([4, 8]),
        "e_layers": _int(1, 4),
        "d_ff": _categorical([128, 256, 512]),
        "dropout": _float(0.0, 0.3),
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
    "NonstationaryTransformer": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "dropout": _float(0.0, 0.3),
        "n_head": _categorical([4, 8]),
        "conv_hidden_size": _categorical([64, 128, 256]),
        "encoder_layers": _int(1, 4),
        "decoder_layers": _int(1, 4),
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
        "n_block": _categorical([1, 2, 3]),
        "ff_dim": _categorical([16, 32, 64, 128]),
        "dropout": _float(0.0, 0.3),
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
    }
}

TRAINING_PARAM_REGISTRY = {
    "input_size": _categorical([16, 32, 64, 128]),
    "season_length": _categorical([12, 24, 52]),
    "batch_size": _categorical([16, 32, 64]),
    "valid_batch_size": _categorical([16, 32, 64]),
    "windows_batch_size": _categorical([256, 512, 1024]),
    "inference_windows_batch_size": _categorical([256, 512, 1024]),
    "learning_rate": _float(1e-4, 1e-1, log=True),
    "max_steps": _categorical([100, 300, 500, 1000]),
    "val_check_steps": _categorical([25, 50, 100, 200]),
}


def suggest_model_params(
    model_name: str, selected_names: tuple[str, ...], trial: optuna.Trial
) -> dict[str, Any]:
    registry = MODEL_PARAM_REGISTRY[model_name]
    return {name: registry[name](trial, name) for name in selected_names}


def suggest_residual_params(
    model_name: str, selected_names: tuple[str, ...], trial: optuna.Trial
) -> dict[str, Any]:
    registry = RESIDUAL_PARAM_REGISTRY[model_name]
    suggested = DEFAULT_RESIDUAL_PARAMS.copy()
    for name in selected_names:
        suggested[name] = registry[name](trial, name)
    return suggested


def suggest_training_params(
    selected_names: tuple[str, ...], trial: optuna.Trial
) -> dict[str, Any]:
    suggested = {}
    for name in selected_names:
        suggested[name] = TRAINING_PARAM_REGISTRY[name](trial, name)
    return suggested
