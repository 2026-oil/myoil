from __future__ import annotations

import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from pytorch_lightning.strategies import DDPStrategy
from neuralforecast.models import (
    Autoformer,
    FEDformer,
    Informer,
    LSTM,
    NHITS,
    PatchTST,
    TFT,
    VanillaTransformer,
)

TRANSFORMER_MODEL_NAMES = [
    "TFT",
    "VanillaTransformer",
    "Informer",
    "Autoformer",
    "FEDformer",
    "PatchTST",
]
LEARNED_MODEL_NAMES = [*TRANSFORMER_MODEL_NAMES, "LSTM", "NHITS"]
BASELINE_MODEL_NAMES = ["Naive", "SeasonalNaive", "HistoricAverage"]
ALL_MODEL_NAMES = [*LEARNED_MODEL_NAMES, *BASELINE_MODEL_NAMES]
TARGET_COLUMNS = ["Com_CrudeOil", "Com_BrentCrudeOil"]
TRAINER_KWARG_NAMES = {"accelerator", "devices", "strategy", "enable_checkpointing"}
GLOO_SINGLE_GPU_MODEL_NAMES = {"FEDformer"}
MODEL_REGISTRY = {
    "TFT": TFT,
    "VanillaTransformer": VanillaTransformer,
    "Informer": Informer,
    "Autoformer": Autoformer,
    "FEDformer": FEDformer,
    "PatchTST": PatchTST,
    "LSTM": LSTM,
    "NHITS": NHITS,
}
STALE_DISTRIBUTED_ENV_KEYS = (
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "NODE_RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)

DEFAULT_MODEL_OVERRIDES = {
    "TFT": {"hidden_size": 64, "n_head": 4},
    "VanillaTransformer": {"hidden_size": 64, "n_head": 4},
    "Informer": {"hidden_size": 64, "n_head": 4},
    "Autoformer": {"hidden_size": 64, "n_head": 4},
    "FEDformer": {"hidden_size": 64, "n_head": 8, "modes": 16},
    "PatchTST": {"hidden_size": 64, "patch_len": 16},
    "LSTM": {
        "encoder_hidden_size": 64,
        "decoder_hidden_size": 64,
        "encoder_n_layers": 2,
        "decoder_layers": 2,
    },
    "NHITS": {
        "mlp_units": 3 * [[64, 64]],
        "n_pool_kernel_size": [2, 2, 1],
        "n_freq_downsample": [4, 2, 1],
        "dropout_prob_theta": 0.0,
        "activation": "ReLU",
    },
}


@dataclass(slots=True)
class Phase1Config:
    data_path: Path
    output_root: Path
    targets: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))
    model_names: list[str] = field(default_factory=lambda: list(ALL_MODEL_NAMES))
    freq: str | None = None
    horizon: int = 12
    step_size: int = 4
    n_windows: int = 24
    final_holdout: int = 12
    input_size: int = 48
    season_length: int = 52
    max_steps: int = 200
    val_size: int = 12
    val_check_steps: int = 20
    early_stop_patience_steps: int = 5
    batch_size: int = 32
    valid_batch_size: int = 32
    windows_batch_size: int = 128
    inference_windows_batch_size: int = 128
    learning_rate: float = 1e-3
    random_seed: int = 1
    require_gpu_count: int = 1
    allow_fewer_gpus: bool = False
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    config_path: str | None = None

    @property
    def cv_test_size(self) -> int:
        return self.horizon + self.step_size * (self.n_windows - 1)


@dataclass(slots=True)
class RunArtifacts:
    run_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    partials_dir: Path

    @classmethod
    def create(cls, output_root: Path) -> "RunArtifacts":
        run_dir = output_root
        checkpoints_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        predictions_dir = run_dir / "predictions"
        partials_dir = run_dir / "partials"
        for path in (run_dir, checkpoints_dir, metrics_dir, predictions_dir, partials_dir):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            metrics_dir=metrics_dir,
            predictions_dir=predictions_dir,
            partials_dir=partials_dir,
        )


def load_source_frame(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def infer_frequency(df: pd.DataFrame) -> str:
    inferred = pd.infer_freq(df["dt"])
    if inferred is None:
        raise ValueError("Could not infer a regular frequency from df.csv dt column.")
    return inferred


def make_target_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found in source data.")
    target_df = df[["dt", target]].rename(columns={"dt": "ds", target: "y"}).copy()
    target_df["unique_id"] = target
    target_df = target_df[["unique_id", "ds", "y"]]
    if target_df["y"].isna().any():
        raise ValueError(f"Target {target} contains missing values; phase1 expects dense history.")
    return target_df


def split_final_holdout(target_df: pd.DataFrame, final_holdout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(target_df) <= final_holdout:
        raise ValueError("Series is shorter than the requested final holdout.")
    train_df = target_df.iloc[:-final_holdout].reset_index(drop=True)
    holdout_df = target_df.iloc[-final_holdout:].reset_index(drop=True)
    return train_df, holdout_df


def assert_cv_capacity(train_df: pd.DataFrame, config: Phase1Config) -> None:
    minimum_rows = config.cv_test_size + max(config.input_size, config.val_size) + 1
    if len(train_df) < minimum_rows:
        raise ValueError(
            f"Series length {len(train_df)} is insufficient for the requested phase1 CV setup; "
            f"need at least {minimum_rows} rows after removing the final holdout."
        )


def ensure_gpu_policy(config: Phase1Config) -> int:
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available < config.require_gpu_count and not config.allow_fewer_gpus:
        raise RuntimeError(
            f"Phase1 requires {config.require_gpu_count} GPUs, but only {available} were detected."
        )
    return min(max(available, 1), config.require_gpu_count if available else 1)


def compute_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual_arr = np.asarray(list(actual), dtype=float)
    pred_arr = np.asarray(list(predicted), dtype=float)
    if actual_arr.shape != pred_arr.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    errors = pred_arr - actual_arr
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))
    safe_actual = np.where(np.isclose(actual_arr, 0.0), np.finfo(float).eps, actual_arr)
    mape = float(np.mean(np.abs(errors / safe_actual)) * 100.0)
    scale = float(actual_arr.max() - actual_arr.min())
    if np.isclose(scale, 0.0):
        scale = float(np.mean(np.abs(actual_arr))) or 1.0
    nrmse = float(rmse / scale)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "NRMSE": nrmse}


def rank_leaderboard(metrics_df: pd.DataFrame) -> pd.DataFrame:
    leaderboard = metrics_df.copy()
    metric_columns = ["RMSE", "MAE", "MAPE", "NRMSE"]
    for metric in metric_columns:
        leaderboard[f"rank_{metric}"] = leaderboard[metric].rank(method="min")
    leaderboard["rank_mean"] = leaderboard[[f"rank_{metric}" for metric in metric_columns]].mean(axis=1)
    return leaderboard.sort_values(["target", "rank_mean", "RMSE", "MAE", "MAPE", "NRMSE", "model"]).reset_index(drop=True)


def _baseline_window_predictions(values: np.ndarray, horizon: int, season_length: int) -> dict[str, np.ndarray]:
    last_value = float(values[-1])
    naive = np.repeat(last_value, horizon)
    if len(values) >= season_length:
        seasonal = values[-season_length:][:horizon]
        if len(seasonal) < horizon:
            seasonal = np.resize(seasonal, horizon)
    else:
        seasonal = naive.copy()
    historic = np.repeat(float(np.mean(values)), horizon)
    return {
        "Naive": naive,
        "SeasonalNaive": seasonal,
        "HistoricAverage": historic,
    }


def baseline_cross_validation(target_df: pd.DataFrame, config: Phase1Config, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = target_df["y"].to_numpy(dtype=float)
    ds = target_df["ds"].to_numpy()
    rows: list[dict[str, Any]] = []
    test_size = config.cv_test_size
    start = len(values) - test_size
    for window_idx in range(config.n_windows):
        train_end = start + window_idx * config.step_size
        train_values = values[:train_end]
        forecast_ds = ds[train_end : train_end + config.horizon]
        actual = values[train_end : train_end + config.horizon]
        preds = _baseline_window_predictions(train_values, config.horizon, config.season_length)
        cutoff = ds[train_end - 1]
        for model_name, forecast in preds.items():
            for ts, y_true, y_hat in zip(forecast_ds, actual, forecast, strict=True):
                rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "cutoff": cutoff,
                        "ds": ts,
                        "y": float(y_true),
                        "y_hat": float(y_hat),
                        "split": "cv",
                    }
                )
    predictions = pd.DataFrame(rows)
    metrics = (
        predictions.groupby(["target", "model"], as_index=False)
        .apply(lambda frame: pd.Series(compute_metrics(frame["y"], frame["y_hat"])), include_groups=False)
        .reset_index(drop=True)
    )
    return metrics, predictions


def baseline_holdout_predictions(train_df: pd.DataFrame, holdout_df: pd.DataFrame, config: Phase1Config, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = _baseline_window_predictions(train_df["y"].to_numpy(dtype=float), len(holdout_df), config.season_length)
    rows: list[dict[str, Any]] = []
    for model_name, forecast in preds.items():
        for (_, actual_row), y_hat in zip(holdout_df.iterrows(), forecast, strict=True):
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "cutoff": train_df["ds"].iloc[-1],
                    "ds": actual_row["ds"],
                    "y": float(actual_row["y"]),
                    "y_hat": float(y_hat),
                    "split": "holdout",
                }
            )
    predictions = pd.DataFrame(rows)
    metrics = (
        predictions.groupby(["target", "model"], as_index=False)
        .apply(lambda frame: pd.Series(compute_metrics(frame["y"], frame["y_hat"])), include_groups=False)
        .reset_index(drop=True)
    )
    return metrics, predictions


def _filter_kwargs(model_class: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(model_class.__init__).parameters
    accepted = {
        name
        for name, param in parameters.items()
        if name != "self"
        and param.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    return {
        key: value
        for key, value in kwargs.items()
        if key in accepted or key in TRAINER_KWARG_NAMES
    }


def _trainer_strategy(gpu_devices: int) -> str | DDPStrategy:
    if not torch.cuda.is_available() or gpu_devices <= 1:
        return "auto"

    backend = os.environ.get("PHASE1_DISTRIBUTED_BACKEND", "gloo")
    start_method = os.environ.get("PHASE1_DDP_START_METHOD", "popen")
    return DDPStrategy(
        process_group_backend=backend,
        start_method=start_method,
    )


def _resolved_gpu_devices(model_name: str, gpu_devices: int) -> int:
    if not torch.cuda.is_available() or gpu_devices <= 1:
        return gpu_devices

    backend = os.environ.get("PHASE1_DISTRIBUTED_BACKEND", "gloo")
    if backend == "gloo" and model_name in GLOO_SINGLE_GPU_MODEL_NAMES:
        return 1
    return gpu_devices


def build_learned_model(model_name: str, gpu_devices: int, config: Phase1Config):
    model_class = MODEL_REGISTRY[model_name]
    trainer_devices = _resolved_gpu_devices(model_name, gpu_devices)
    common = {
        "h": config.horizon,
        "input_size": config.input_size,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "val_check_steps": config.val_check_steps,
        "early_stop_patience_steps": config.early_stop_patience_steps,
        "batch_size": config.batch_size,
        "valid_batch_size": config.valid_batch_size,
        "windows_batch_size": config.windows_batch_size,
        "inference_windows_batch_size": config.inference_windows_batch_size,
        "random_seed": config.random_seed,
        "step_size": 1,
        "alias": model_name,
        "loss": MAE(),
        "valid_loss": MAE(),
        "enable_checkpointing": True,
        "accelerator": "gpu" if torch.cuda.is_available() and trainer_devices >= 1 else "cpu",
        "devices": trainer_devices if torch.cuda.is_available() and trainer_devices >= 1 else 1,
        "strategy": _trainer_strategy(trainer_devices),
        "scaler_type": "robust",
        "dropout": 0.1,
        "hidden_size": 64,
        "n_head": 4,
        "n_heads": 4,
        "factor": 1,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 256,
        "conv_hidden_size": 32,
        "patch_len": 16,
        "n_series": 1,
        "encoder_hidden_size": 64,
        "decoder_hidden_size": 64,
        "context_size": config.horizon,
        "recurrent": False,
    }
    model_kwargs = {**common, **DEFAULT_MODEL_OVERRIDES.get(model_name, {})}
    model_kwargs.update(config.model_overrides.get(model_name, {}))
    model_kwargs["alias"] = model_name
    model_kwargs = _filter_kwargs(model_class, model_kwargs)
    return model_class(**model_kwargs)


def _extract_prediction_column(frame: pd.DataFrame, model_name: str) -> str:
    if model_name in frame.columns:
        return model_name
    for column in frame.columns:
        if column.startswith(model_name):
            return column
    raise KeyError(f"Could not find prediction column for {model_name} in {frame.columns.tolist()}.")


def _destroy_process_group_if_needed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@contextmanager
def _single_process_distributed_env_scope(devices: int):
    if devices > 1:
        yield
        return

    saved = {key: os.environ.pop(key) for key in STALE_DISTRIBUTED_ENV_KEYS if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def learned_cross_validation(
    target_df: pd.DataFrame,
    target: str,
    freq: str,
    model_name: str,
    config: Phase1Config,
    gpu_devices: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        trainer_devices = _resolved_gpu_devices(model_name, gpu_devices)
        with _single_process_distributed_env_scope(trainer_devices):
            model = build_learned_model(model_name, gpu_devices=gpu_devices, config=config)
            nf = NeuralForecast(models=[model], freq=freq)
            cv = nf.cross_validation(
                df=target_df,
                h=config.horizon,
                n_windows=config.n_windows,
                step_size=config.step_size,
                val_size=config.val_size,
                refit=False,
                verbose=False,
            )
            pred_column = _extract_prediction_column(cv, model_name)
            predictions = cv[["cutoff", "ds", "y", pred_column]].rename(columns={pred_column: "y_hat"}).copy()
            predictions.insert(0, "model", model_name)
            predictions.insert(0, "target", target)
            predictions["split"] = "cv"
            metrics = pd.DataFrame([
                {"target": target, "model": model_name, **compute_metrics(predictions["y"], predictions["y_hat"])}
            ])
            return metrics, predictions
    finally:
        _destroy_process_group_if_needed()


def learned_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target: str,
    freq: str,
    model_name: str,
    config: Phase1Config,
    gpu_devices: int,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        trainer_devices = _resolved_gpu_devices(model_name, gpu_devices)
        with _single_process_distributed_env_scope(trainer_devices):
            model = build_learned_model(model_name, gpu_devices=gpu_devices, config=config)
            nf = NeuralForecast(models=[model], freq=freq)
            nf.fit(df=train_df, val_size=config.val_size)
            forecasts = nf.predict(futr_df=holdout_df[["unique_id", "ds"]])
            pred_column = _extract_prediction_column(forecasts, model_name)
            merged = holdout_df.merge(
                forecasts[["unique_id", "ds", pred_column]],
                on=["unique_id", "ds"],
                how="left",
            )
            if merged[pred_column].isna().any():
                debug_dir = checkpoint_dir.parent / "debug_holdout"
                debug_dir.mkdir(parents=True, exist_ok=True)
                forecasts_path = debug_dir / f"{target}_{model_name}_forecasts.csv"
                merged_path = debug_dir / f"{target}_{model_name}_merged.csv"
                forecasts.to_csv(forecasts_path, index=False)
                merged.to_csv(merged_path, index=False)
                missing_rows = int(merged[pred_column].isna().sum())
                raise ValueError(
                    f"Missing holdout predictions for {target}/{model_name}: "
                    f"{missing_rows} rows had null predictions. "
                    f"Debug artifacts: {forecasts_path} ; {merged_path}"
                )
            predictions = merged[["ds", "y", pred_column]].rename(columns={pred_column: "y_hat"}).copy()
            predictions.insert(0, "cutoff", train_df["ds"].iloc[-1])
            predictions.insert(0, "model", model_name)
            predictions.insert(0, "target", target)
            predictions["split"] = "holdout"
            checkpoint_path = checkpoint_dir / target / model_name
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            nf.save(path=str(checkpoint_path), overwrite=True, save_dataset=False)
            metrics = pd.DataFrame([
                {"target": target, "model": model_name, **compute_metrics(predictions["y"], predictions["y_hat"])}
            ])
            return metrics, predictions
    finally:
        _destroy_process_group_if_needed()


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _partial_file(artifacts: RunArtifacts, split: str, target: str, model: str, kind: str) -> Path:
    safe_target = target.replace("/", "_")
    safe_model = model.replace("/", "_")
    return artifacts.partials_dir / f"{split}__{safe_target}__{safe_model}__{kind}.csv"


def _save_partial_result(
    artifacts: RunArtifacts,
    split: str,
    target: str,
    model: str,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    write_frame(metrics, _partial_file(artifacts, split, target, model, "metrics"))
    write_frame(predictions, _partial_file(artifacts, split, target, model, "predictions"))


def _load_partial_result(
    artifacts: RunArtifacts,
    split: str,
    target: str,
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    metrics_path = _partial_file(artifacts, split, target, model, "metrics")
    predictions_path = _partial_file(artifacts, split, target, model, "predictions")
    if not metrics_path.exists() or not predictions_path.exists():
        return None
    return pd.read_csv(metrics_path), pd.read_csv(predictions_path)


def _recover_holdout_from_checkpoint(
    checkpoint_path: Path,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    model_name: str,
    target: str,
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not checkpoint_path.exists():
        return None

    loaded = NeuralForecast.load(path=str(checkpoint_path))
    forecasts = loaded.predict(df=train_df, futr_df=holdout_df[["unique_id", "ds"]])
    pred_column = _extract_prediction_column(forecasts, model_name)
    merged = holdout_df.merge(
        forecasts[["unique_id", "ds", pred_column]],
        on=["unique_id", "ds"],
        how="left",
    )
    if merged[pred_column].isna().any():
        return None
    predictions = merged[["ds", "y", pred_column]].rename(columns={pred_column: "y_hat"}).copy()
    predictions.insert(0, "cutoff", train_df["ds"].iloc[-1])
    predictions.insert(0, "model", model_name)
    predictions.insert(0, "target", target)
    predictions["split"] = "holdout"
    metrics = pd.DataFrame([
        {"target": target, "model": model_name, **compute_metrics(predictions["y"], predictions["y_hat"])}
    ])
    return metrics, predictions


def run_phase1(config: Phase1Config) -> dict[str, Any]:
    source_df = load_source_frame(config.data_path)
    freq = config.freq or infer_frequency(source_df)
    gpu_devices = ensure_gpu_policy(config)
    artifacts = RunArtifacts.create(config.output_root)

    cv_metrics_parts: list[pd.DataFrame] = []
    holdout_metrics_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []

    for target in config.targets:
        target_df = make_target_frame(source_df, target)
        train_df, holdout_df = split_final_holdout(target_df, config.final_holdout)
        assert_cv_capacity(train_df, config)

        missing_baseline_partials = any(
            _load_partial_result(artifacts, "cv", target, model_name) is None
            or _load_partial_result(artifacts, "holdout", target, model_name) is None
            for model_name in BASELINE_MODEL_NAMES
            if model_name in config.model_names
        )
        if missing_baseline_partials:
            baseline_metrics, baseline_predictions = baseline_cross_validation(train_df, config, target)
            for model_name in BASELINE_MODEL_NAMES:
                if model_name not in config.model_names:
                    continue
                model_metrics = baseline_metrics[baseline_metrics["model"] == model_name].reset_index(drop=True)
                model_predictions = baseline_predictions[baseline_predictions["model"] == model_name].reset_index(drop=True)
                _save_partial_result(artifacts, "cv", target, model_name, model_metrics, model_predictions)

            baseline_holdout_metrics, baseline_holdout_predictions_df = baseline_holdout_predictions(
                train_df, holdout_df, config, target
            )
            for model_name in BASELINE_MODEL_NAMES:
                if model_name not in config.model_names:
                    continue
                model_metrics = baseline_holdout_metrics[baseline_holdout_metrics["model"] == model_name].reset_index(drop=True)
                model_predictions = baseline_holdout_predictions_df[
                    baseline_holdout_predictions_df["model"] == model_name
                ].reset_index(drop=True)
                _save_partial_result(artifacts, "holdout", target, model_name, model_metrics, model_predictions)

        for model_name in BASELINE_MODEL_NAMES:
            if model_name not in config.model_names:
                continue
            cv_partial = _load_partial_result(artifacts, "cv", target, model_name)
            holdout_partial = _load_partial_result(artifacts, "holdout", target, model_name)
            assert cv_partial is not None and holdout_partial is not None
            cv_metrics_parts.append(cv_partial[0])
            prediction_parts.append(cv_partial[1])
            holdout_metrics_parts.append(holdout_partial[0])
            prediction_parts.append(holdout_partial[1])

        learned_names = [name for name in config.model_names if name in LEARNED_MODEL_NAMES]
        for model_name in learned_names:
            cv_partial = _load_partial_result(artifacts, "cv", target, model_name)
            if cv_partial is None:
                metrics_df, predictions_df = learned_cross_validation(
                    target_df=train_df,
                    target=target,
                    freq=freq,
                    model_name=model_name,
                    config=config,
                    gpu_devices=gpu_devices,
                )
                _save_partial_result(artifacts, "cv", target, model_name, metrics_df, predictions_df)
            else:
                metrics_df, predictions_df = cv_partial
            cv_metrics_parts.append(metrics_df)
            prediction_parts.append(predictions_df)
            holdout_partial = _load_partial_result(artifacts, "holdout", target, model_name)
            if holdout_partial is None:
                checkpoint_path = artifacts.checkpoints_dir / target / model_name
                holdout_partial = _recover_holdout_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    train_df=train_df,
                    holdout_df=holdout_df,
                    model_name=model_name,
                    target=target,
                    freq=freq,
                )
            if holdout_partial is None:
                holdout_metrics_df, holdout_predictions_df = learned_holdout(
                    train_df=train_df,
                    holdout_df=holdout_df,
                    target=target,
                    freq=freq,
                    model_name=model_name,
                    config=config,
                    gpu_devices=gpu_devices,
                    checkpoint_dir=artifacts.checkpoints_dir,
                )
                _save_partial_result(artifacts, "holdout", target, model_name, holdout_metrics_df, holdout_predictions_df)
            else:
                holdout_metrics_df, holdout_predictions_df = holdout_partial
            holdout_metrics_parts.append(holdout_metrics_df)
            prediction_parts.append(holdout_predictions_df)

    cv_metrics = pd.concat(cv_metrics_parts, ignore_index=True)
    cv_metrics = cv_metrics[cv_metrics["model"].isin(config.model_names)].reset_index(drop=True)
    holdout_metrics = pd.concat(holdout_metrics_parts, ignore_index=True)
    holdout_metrics = holdout_metrics[holdout_metrics["model"].isin(config.model_names)].reset_index(drop=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = predictions[predictions["model"].isin(config.model_names)].reset_index(drop=True)
    cv_leaderboard = rank_leaderboard(cv_metrics)
    holdout_leaderboard = rank_leaderboard(holdout_metrics)

    write_frame(cv_metrics, artifacts.metrics_dir / "cv_metrics.csv")
    write_frame(holdout_metrics, artifacts.metrics_dir / "holdout_metrics.csv")
    write_frame(cv_leaderboard, artifacts.metrics_dir / "cv_leaderboard.csv")
    write_frame(holdout_leaderboard, artifacts.metrics_dir / "holdout_leaderboard.csv")
    write_frame(predictions, artifacts.predictions_dir / "all_predictions.csv")

    summary = {
        "config": {
            **asdict(config),
            "data_path": str(config.data_path),
            "output_root": str(config.output_root),
            "freq": freq,
            "detected_gpu_devices": gpu_devices,
            "available_cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "artifacts": {
            "run_dir": str(artifacts.run_dir),
            "checkpoints_dir": str(artifacts.checkpoints_dir),
            "metrics_dir": str(artifacts.metrics_dir),
            "predictions_dir": str(artifacts.predictions_dir),
        },
    }
    (artifacts.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
