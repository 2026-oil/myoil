from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class ScalerState:
    kind: str
    center: float
    scale: float


@dataclass
class FoldSpec:
    fold_index: int
    train_end: int
    val_end: int
    test_start: int
    regime_cell: str
    regime_info: Dict[str, float]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _detect_time_column(df: pd.DataFrame, target_column: str) -> str | None:
    candidates = ["dt", "date", "Date", "week", "Week", "time", "Time", "timestamp", "Timestamp", "ds"]
    for name in candidates:
        if name in df.columns and name != target_column:
            return name
    for name in df.columns:
        if name == target_column:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[name]):
            return name
    for name in df.columns:
        if name == target_column:
            continue
        parsed = pd.to_datetime(df[name], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return name
    return None


def load_authoritative_series(dataset_path: str | Path, target_column: str) -> Tuple[np.ndarray, pd.DataFrame]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found in {path}")
    time_col = _detect_time_column(df, target_column)
    if time_col is not None:
        sort_key = pd.to_datetime(df[time_col], errors="coerce")
        df = df.assign(_sort_key=sort_key).sort_values("_sort_key", kind="mergesort", na_position="last")
        df = df.drop(columns=["_sort_key"]).reset_index(drop=True)
    target = pd.to_numeric(df[target_column], errors="coerce").interpolate(limit_direction="both").ffill().bfill()
    values = target.to_numpy(dtype=np.float32)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Target series is empty or contains non-finite values")
    return values, df


def fit_scaler(values: np.ndarray, kind: str = "robust") -> ScalerState:
    values = np.asarray(values, dtype=np.float32)
    if kind == "standard":
        center = float(np.mean(values))
        scale = float(np.std(values, ddof=1) if values.size > 1 else 1.0)
    else:
        center = float(np.median(values))
        q75 = float(np.percentile(values, 75))
        q25 = float(np.percentile(values, 25))
        scale = q75 - q25
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        scale = float(np.std(values, ddof=1) if values.size > 1 else 1.0)
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        scale = 1.0
    return ScalerState(kind=kind, center=center, scale=scale)


def transform_values(values: np.ndarray, scaler: ScalerState) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - scaler.center) / scaler.scale


def inverse_transform_values(values: np.ndarray, scaler: ScalerState) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values * scaler.scale + scaler.center


def _ols_slope(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=np.float32)
    x = x - float(np.mean(x))
    y = arr - float(np.mean(arr))
    denom = float(np.sum(x * x))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def regime_cell_for_fold(
    training_values: np.ndarray,
    *,
    vol_window: int = 8,
    slope_window: int = 4,
) -> Tuple[str, Dict[str, float]]:
    train = np.asarray(training_values, dtype=np.float32)
    if train.size < max(vol_window, slope_window) + 1:
        return "low/negative_or_flat", {"vol_threshold": 0.0, "recent_vol": 0.0, "slope": 0.0}
    rolling = pd.Series(train).rolling(vol_window).std(ddof=1).dropna().to_numpy(dtype=np.float32)
    if rolling.size == 0:
        rolling = np.asarray([float(np.std(train, ddof=1))], dtype=np.float32)
    vol_threshold = float(np.median(rolling))
    recent_window = train[-vol_window:]
    recent_vol = float(np.std(recent_window, ddof=1) if recent_window.size > 1 else 0.0)
    vol_label = "high" if recent_vol > vol_threshold else "low"
    slope = _ols_slope(train[-slope_window:])
    slope_label = "positive" if slope > 0 else "negative_or_flat"
    return f"{vol_label}/{slope_label}", {
        "vol_threshold": vol_threshold,
        "recent_vol": recent_vol,
        "slope": slope,
    }


def build_fold_specs(
    values: np.ndarray,
    *,
    folds: int,
    horizon: int,
    validation_size: int,
    context_length: int,
    season_length: int,
    terminal_gap: int = 0,
) -> List[FoldSpec]:
    n = len(values)
    min_test_start = context_length + validation_size + season_length + horizon
    max_test_start = n - horizon - max(0, terminal_gap)
    if max_test_start <= min_test_start + folds - 1:
        raise ValueError(
            f"Not enough observations ({n}) for {folds} folds, horizon={horizon}, "
            f"validation_size={validation_size}, context_length={context_length}"
        )
    raw_starts = np.rint(np.linspace(min_test_start, max_test_start, folds)).astype(int)
    test_starts: List[int] = []
    for idx, candidate in enumerate(raw_starts):
        min_allowed = test_starts[-1] + 1 if test_starts else min_test_start
        max_allowed = max_test_start - (folds - idx - 1)
        start = int(min(max(candidate, min_allowed), max_allowed))
        test_starts.append(start)
    specs: List[FoldSpec] = []
    for fold_index, test_start in enumerate(test_starts):
        train_end = test_start - validation_size
        val_end = test_start
        if train_end <= context_length:
            raise ValueError(f"Fold {fold_index} leaves too little training history")
        cell, info = regime_cell_for_fold(values[:train_end])
        specs.append(
            FoldSpec(
                fold_index=fold_index,
                train_end=train_end,
                val_end=val_end,
                test_start=test_start,
                regime_cell=cell,
                regime_info=info,
            )
        )
    return specs


def make_window_arrays(
    values: np.ndarray,
    context_length: int,
    horizon: int,
    start_index: int,
    end_exclusive: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for target_start in range(start_index, end_exclusive - horizon + 1):
        context_start = target_start - context_length
        if context_start < 0:
            continue
        xs.append(np.asarray(values[context_start:target_start], dtype=np.float32))
        ys.append(np.asarray(values[target_start:target_start + horizon], dtype=np.float32))
    if not xs:
        return (
            np.zeros((0, context_length), dtype=np.float32),
            np.zeros((0, horizon), dtype=np.float32),
        )
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def compute_router_features_from_window(window: Sequence[float]) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    recent = window[-8:] if window.size >= 8 else window
    vol = float(np.std(recent, ddof=1) if recent.size > 1 else 0.0)
    slope = _ols_slope(window[-4:] if window.size >= 4 else window)
    recent_mean = float(np.mean(window[-4:] if window.size >= 4 else window))
    return np.asarray([vol, slope, recent_mean], dtype=np.float32)


def joint_gate_admissible(pred: Sequence[float], actual: Sequence[float], tol: float = 0.10) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    if pred.size < 2 or actual.size < 2:
        return 0.0
    eps = 1e-8
    pct = np.abs(pred[:2] - actual[:2]) / np.maximum(np.abs(actual[:2]), eps)
    return float((pred[1] > pred[0]) and np.all(pct <= tol))


def noncompliance_rate(pred: Sequence[float], actual: Sequence[float], tol: float = 0.10) -> float:
    return 1.0 - joint_gate_admissible(pred, actual, tol=tol)


def mean_absolute_percentage_error(preds: np.ndarray, actuals: np.ndarray) -> float:
    preds = np.asarray(preds, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    eps = 1e-8
    pct = np.abs(preds - actuals) / np.maximum(np.abs(actuals), eps)
    return float(np.mean(pct) * 100.0)


def normalized_rmse(preds: np.ndarray, actuals: np.ndarray) -> float:
    preds = np.asarray(preds, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    denom = float(np.std(actuals, ddof=0) + 1e-8)
    return float(rmse / denom)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float] | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 5:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[samples].mean(axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))
