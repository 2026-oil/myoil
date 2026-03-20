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
FIRST_CUT_AUTO_MODEL_NAMES = {
    "TFT",
    "VanillaTransformer",
    "Informer",
    "Autoformer",
    "FEDformer",
    "PatchTST",
    "LSTM",
    "NHITS",
    "iTransformer",
}
FIRST_CUT_RESIDUAL_MODELS = {"xgboost"}
DEFAULT_OPTUNA_NUM_TRIALS = 5
DEFAULT_OPTUNA_STUDY_DIRECTION = "minimize"

DEFAULT_RESIDUAL_PARAMS = {
    "n_estimators": 32,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
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
    "NHITS": {
        "n_pool_kernel_size": _categorical(
            [[2, 2, 1], [1, 1, 1], [2, 2, 2], [4, 2, 1]]
        ),
        "n_freq_downsample": _categorical(
            [[24, 12, 1], [60, 8, 1], [40, 20, 1], [1, 1, 1]]
        ),
        "dropout_prob_theta": _float(0.0, 0.3),
    },
    "iTransformer": {
        "hidden_size": _categorical([32, 64, 128, 256]),
        "n_heads": _categorical([4, 8]),
        "e_layers": _int(1, 4),
        "d_ff": _categorical([128, 256, 512]),
        "dropout": _float(0.0, 0.3),
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
