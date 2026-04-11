"""
Dataset used and loading:
- Exact authoritative dataset: /home/sonet/.openclaw/workspace/research/neuralforecast/data/df.csv
- Loaded with pandas, sorted by the detected dt/date-like column when present, and reduced to the
  target column Com_BrentCrudeOil. Missing target values are interpolated, forward-filled, and
  back-filled.

Distribution shift / corruption:
- No synthetic corruption is injected.
- Fold-local regime variation is derived only from the training slice via rolling 8-week volatility
  and trailing 4-week slope labels.

Model architecture:
- Baseline NLinear family: compact lag-window MLPs over a context length of 16 with ReLU activations.
- Order-aware variant: predicts h1 and a softplus delta so h2 = h1 + delta.
- PatchTST baseline: patch embedding plus a shallow Transformer encoder.
- iTransformer baseline: time-axis mixing plus a shallow Transformer encoder.
- Volatility-gated ensemble: two NLinear experts (trend and mean-reversion) blended by a router MLP.

Training protocol:
- Optimizer: AdamW
- Max epochs: 25, pilot epochs: 4
- Batch size: 16
- LR schedule: cosine annealing
- Early stopping patience: 3
- Gradient clipping: 1.0
- Validation-only affine calibration before terminal scoring
- Deterministic seeds: [0, 1, 2]
- Calibrated terminal gap: 27 observations reserved after the last claim fold so the final fold is
  still chronological but not trivially impossible.

Evaluation protocol:
- 4 expanding-window folds, horizon = 2, fold 4 reserved as the terminal claim fold.
- Primary claim uses only the last fold and the admissibility gate:
  y_hat2 > y_hat1 and both horizons within ±10% of actual values.
- Secondary metrics: last-fold admissibility/order violation rates, per-regime compliance, full-CV
  MAPE/NRMSE, regime compliance gap, success rate, and paired seed-wise comparisons.

METRIC DEF:
- METRIC NAME: primary_metric
- DIRECTION: minimize
- UNITS/SCALE: fraction in [0, 1]
- FORMULA: primary_metric = 1 - 1{y_hat2 > y_hat1 and |y_hat1-y1|/|y1| <= 0.10 and |y_hat2-y2|/|y2| <= 0.10}
- AGGREGATION: last-fold binary value, then mean/std across seeds
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from data_utils import (
    FoldSpec,
    bootstrap_mean_ci,
    build_fold_specs,
    compute_router_features_from_window,
    joint_gate_admissible,
    load_authoritative_series,
    mean_absolute_percentage_error,
    noncompliance_rate,
    normalized_rmse,
    set_all_seeds,
)
from experiment_harness import ExperimentHarness
from models import (
    DirectNLinearTwoHeadForecaster,
    FoldwiseRobustNLinearTwoHeadForecaster,
    ITransformerTwoStepForecaster,
    OrderAwareNLinearDeltaTwoHeadForecaster,
    PatchTSTTwoStepForecaster,
    StandardScaledNLinearTwoHeadForecaster,
    UniformRouterNLinearEnsemble,
    VolatilityGatedNLinearEnsemble,
)


HYPERPARAMETERS = {
    "dataset_path": "/home/sonet/.openclaw/workspace/research/neuralforecast/data/df.csv",
    "target_column": "Com_BrentCrudeOil",
    "time_budget_seconds": 300,
    "context_length": 16,
    "trend_context_length": 8,
    "horizon": 2,
    "season_length": 4,
    "folds": 4,
    "validation_size": 4,
    "terminal_gap": 27,
    "batch_size": 16,
    "max_epochs": 25,
    "pilot_epochs": 4,
    "patience": 3,
    "scheduler_type": "cosine",
    "learning_rate": 0.001,
    "learning_rate_order": 0.0003,
    "learning_rate_patch": 0.0005,
    "learning_rate_transformer": 0.0005,
    "learning_rate_router": 0.0005,
    "weight_decay": 0.0001,
    "hidden_dim": 32,
    "delta_hidden_dim": 32,
    "patch_length": 4,
    "patch_stride": 1,
    "patch_d_model": 32,
    "patch_n_heads": 4,
    "patch_n_layers": 2,
    "patch_dropout": 0.1,
    "transformer_d_model": 32,
    "transformer_d_ff": 64,
    "transformer_n_heads": 2,
    "transformer_n_layers": 2,
    "transformer_dropout": 0.1,
    "router_hidden_dim": 8,
    "router_temperature": 0.7,
    "lambda_order": 1.0,
    "order_margin": 0.0,
    "lambda_balance": 0.05,
    "lambda_entropy": 0.01,
    "lambda_expert": 0.5,
    "calibration_weight": 0.05,
    "gradient_clip_norm": 1.0,
    "bootstrap_samples": 1000,
    "seed_count": 3,
    "seeds": [0, 1, 2],
}

SEEDS = [0, 1, 2]
EXPECTED_CONDITIONS = [
    "nlinear_two_head_mae_forecaster_with_foldwise_robust_scaling",
    "order_aware_nlinear_nonnegative_delta_two_head_forecaster",
    "patchtst_multihorizon_forecaster",
    "itransformer_multihorizon_forecaster",
    "volatility_gated_trend_meanreversion_two_expert_ensemble",
    "order_aware_nlinear_without_nonnegative_delta_constraint",
    "nlinear_two_head_with_fixed_standard_scaling",
    "volatility_gated_two_expert_ensemble_with_uniform_router",
]


@dataclass
class ConditionSpec:
    name: str
    factory: Callable[[], Any]
    family: str
    comparator: Optional[List[str]] = None


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_results_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)


def safe_report_metric(harness: ExperimentHarness, metric_name: str, value: float) -> None:
    value = float(value)
    if harness.check_value(value, metric_name):
        harness.report_metric(metric_name, value)
    else:
        print("SKIP: NaN/Inf detected")


def build_condition_registry() -> OrderedDict[str, ConditionSpec]:
    registry: "OrderedDict[str, ConditionSpec]" = OrderedDict()

    registry["nlinear_two_head_mae_forecaster_with_foldwise_robust_scaling"] = ConditionSpec(
        name="nlinear_two_head_mae_forecaster_with_foldwise_robust_scaling",
        family="baseline",
        factory=lambda: FoldwiseRobustNLinearTwoHeadForecaster(
            context_length=HYPERPARAMETERS["context_length"],
            horizon=HYPERPARAMETERS["horizon"],
            hidden_dim=HYPERPARAMETERS["hidden_dim"],
            batch_size=HYPERPARAMETERS["batch_size"],
            max_epochs=HYPERPARAMETERS["max_epochs"],
            patience=HYPERPARAMETERS["patience"],
            learning_rate=HYPERPARAMETERS["learning_rate"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
            gradient_clip_norm=HYPERPARAMETERS["gradient_clip_norm"],
            calibration_weight=HYPERPARAMETERS["calibration_weight"],
            scheduler_type=HYPERPARAMETERS["scheduler_type"],
        ),
    )
    registry["order_aware_nlinear_nonnegative_delta_two_head_forecaster"] = ConditionSpec(
        name="order_aware_nlinear_nonnegative_delta_two_head_forecaster",
        family="proposed",
        comparator=["nlinear_two_head_mae_forecaster_with_foldwise_robust_scaling"],
        factory=lambda: OrderAwareNLinearDeltaTwoHeadForecaster(
            context_length=HYPERPARAMETERS["context_length"],
            horizon=HYPERPARAMETERS["horizon"],
            hidden_dim=HYPERPARAMETERS["hidden_dim"],
            batch_dim=HYPERPARAMETERS["delta_hidden_dim"],
            batch_size=HYPERPARAMETERS["batch_size"],
            max_epochs=HYPERPARAMETERS["max_epochs"],
            patience=HYPERPARAMETERS["patience"],
            learning_rate=HYPERPARAMETERS["learning_rate_order"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
            gradient_clip_norm=HYPERPARAMETERS["gradient_clip_norm"],
            calibration_weight=HYPERPARAMETERS["calibration_weight"],
            lambda_order=HYPERPARAMETERS["lambda_order"],
            order_margin=HYPERPARAMETERS["order_margin"],
            scheduler_type=HYPERPARAMETERS["scheduler_type"],
        ),
    )
    registry["patchtst_multihorizon_forecaster"] = ConditionSpec(
        name="patchtst_multihorizon_forecaster",
        family="baseline",
        factory=lambda: PatchTSTTwoStepForecaster(
            context_length=HYPERPARAMETERS["context_length"],
            horizon=HYPERPARAMETERS["horizon"],
            patch_length=HYPERPARAMETERS["patch_length"],
            stride=HYPERPARAMETERS["patch_stride"],
            d_model=HYPERPARAMETERS["patch_d_model"],
            n_heads=HYPERPARAMETERS["patch_n_heads"],
            n_layers=HYPERPARAMETERS["patch_n_layers"],
            dropout=HYPERPARAMETERS["patch_dropout"],
            batch_size=HYPERPARAMETERS["batch_size"],
            max_epochs=HYPERPARAMETERS["max_epochs"],
            patience=HYPERPARAMETERS["patience"],
            learning_rate=HYPERPARAMETERS["learning_rate_patch"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
            gradient_clip_norm=HYPERPARAMETERS["gradient_clip_norm"],
            calibration_weight=HYPERPARAMETERS["calibration_weight"],
            scheduler_type=HYPERPARAMETERS["scheduler_type"],
        ),
    )
    registry["itransformer_multihorizon_forecaster"] = ConditionSpec(
        name="itransformer_multihorizon_forecaster",
        family="baseline",
        factory=lambda: ITransformerTwoStepForecaster(
            context_length=HYPERPARAMETERS["context_length"],
            horizon=HYPERPARAMETERS["horizon"],
            d_model=HYPERPARAMETERS["transformer_d_model"],
            d_ff=HYPERPARAMETERS["transformer_d_ff"],
            n_heads=HYPERPARAMETERS["transformer_n_heads"],
            n_layers=HYPERPARAMETERS["transformer_n_layers"],
            dropout=HYPERPARAMETERS["transformer_dropout"],
            batch_size=HYPERPARAMETERS["batch_size"],
            max_epochs=HYPERPARAMETERS["max_epochs"],
            patience=HYPERPARAMETERS["patience"],
            learning_rate=HYPERPARAMETERS["learning_rate_transformer"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
            gradient_clip_norm=HYPERPARAMETERS["gradient_clip_norm"],
            calibration_weight=HYPERPARAMETERS["calibration_weight"],
            scheduler_type=HYPERPARAMETERS["scheduler_type"],
        ),
    )
    registry["volatility_gated_trend_meanreversion_two_expert_ensemble"] = ConditionSpec(
        name="volatility_gated_trend_meanreversion_two_expert_ensemble",
        family="proposed",
        comparator=["patchtst_multihorizon_forecaster", "itransformer_multihorizon_forecaster"],
        factory=lambda: VolatilityGatedNLinearEnsemble(
            context_length=HYPERPARAMETERS["context_length"],
            trend_context_length=HYPERPARAMETERS["trend_context_length"],
            horizon=HYPERPARAMETERS["horizon"],
            hidden_dim=HYPERPARAMETERS["hidden_dim"],
            router_hidden_dim=HYPERPARAMETERS["router_hidden_dim"],
            router_temperature=HYPERPARAMETERS["router_temperature"],
            lambda_balance=HYPERPARAMETERS["lambda_balance"],
            lambda_entropy=HYPERPARAMETERS["lambda_entropy"],
            lambda_expert=HYPERPARAMETERS["lambda_expert"],
            batch_size=HYPERPARAMETERS["batch_size"],
            max_epochs=HYPERPARAMETERS["max_epochs"],
            patience=HYPERPARAMETERS["patience"],
            learning_rate=HYPERPARAMETERS["learning_rate"],
            learning_rate_router=HYPERPARAMETERS["learning_rate_router"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
            gradient_clip_norm=HYPERPARAMETERS["gradient_clip_norm"],
            calibration_weight=HYPERPARAMETERS["calibration_weight"],
            scheduler_type=HYPERPARAMETERS["scheduler_type"],
        ),
    )
    registry["order_aware_nlinear_without_nonnegative_delta_constraint"] = ConditionSpec(
        name="order_aware_nlinear_without_nonnegative_delta_constraint",
        family="ablation",
        comparator=["order_aware_nlinear_nonnegative_delta_two_head_forecaster"],
        factory=lambda: DirectNLinearTwoHeadForecaster(
            context_length=HYPERPARAMETERS["context_length"],
            horizon=HYPERPARAMETERS["horizon"],
            hidden_dim=HYPERPARAMETERS["hidden_dim"],
            batch_dim=HYPERPARAMETERS["delta_hidden_dim"],
            batch_size=HYPERPARAMETERS["batch_size"],
            max_epochs=HYPERPARAMETERS["max_epochs"],
            patience=HYPERPARAMETERS["patience"],
            learning_rate=HYPERPARAMETERS["learning_rate_order"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
            gradient_clip_norm=HYPERPARAMETERS["gradient_clip_norm"],
            calibration_weight=HYPERPARAMETERS["calibration_weight"],
            lambda_order=HYPERPARAMETERS["lambda_order"],
            order_margin=HYPERPARAMETERS["order_margin"],
            scheduler_type=HYPERPARAMETERS["scheduler_type"],
        ),
    )
    registry["nlinear_two_head_with_fixed_standard_scaling"] = ConditionSpec(
        name="nlinear_two_head_with_fixed_standard_scaling",
        family="ablation",
        comparator=["nlinear_two_head_mae_forecaster_with_foldwise_robust_scaling"],
        factory=lambda: StandardScaledNLinearTwoHeadForecaster(
            context_length=HYPERPARAMETERS["context_length"],
            horizon=HYPERPARAMETERS["horizon"],
            hidden_dim=HYPERPARAMETERS["hidden_dim"],
            batch_size=HYPERPARAMETERS["batch_size"],
            max_epochs=HYPERPARAMETERS["max_epochs"],
            patience=HYPERPARAMETERS["patience"],
            learning_rate=HYPERPARAMETERS["learning_rate"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
            gradient_clip_norm=HYPERPARAMETERS["gradient_clip_norm"],
            calibration_weight=HYPERPARAMETERS["calibration_weight"],
            scheduler_type=HYPERPARAMETERS["scheduler_type"],
        ),
    )
    registry["volatility_gated_two_expert_ensemble_with_uniform_router"] = ConditionSpec(
        name="volatility_gated_two_expert_ensemble_with_uniform_router",
        family="ablation",
        comparator=["volatility_gated_trend_meanreversion_two_expert_ensemble"],
        factory=lambda: UniformRouterNLinearEnsemble(
            context_length=HYPERPARAMETERS["context_length"],
            trend_context_length=HYPERPARAMETERS["trend_context_length"],
            horizon=HYPERPARAMETERS["horizon"],
            hidden_dim=HYPERPARAMETERS["hidden_dim"],
            router_hidden_dim=HYPERPARAMETERS["router_hidden_dim"],
            router_temperature=HYPERPARAMETERS["router_temperature"],
            lambda_balance=HYPERPARAMETERS["lambda_balance"],
            lambda_entropy=HYPERPARAMETERS["lambda_entropy"],
            lambda_expert=HYPERPARAMETERS["lambda_expert"],
            batch_size=HYPERPARAMETERS["batch_size"],
            max_epochs=HYPERPARAMETERS["max_epochs"],
            patience=HYPERPARAMETERS["patience"],
            learning_rate=HYPERPARAMETERS["learning_rate"],
            learning_rate_router=HYPERPARAMETERS["learning_rate_router"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
            gradient_clip_norm=HYPERPARAMETERS["gradient_clip_norm"],
            calibration_weight=HYPERPARAMETERS["calibration_weight"],
            scheduler_type=HYPERPARAMETERS["scheduler_type"],
        ),
    )
    return registry


def compute_fold_metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    eps = 1e-8
    pct = np.abs(pred[:2] - actual[:2]) / np.maximum(np.abs(actual[:2]), eps)
    admissible = float((pred[1] > pred[0]) and np.all(pct <= 0.10))
    order_violation = float(pred[1] <= pred[0])
    h1_within = float(pct[0] <= 0.10)
    h2_within = float(pct[1] <= 0.10)
    mape = float(np.mean(pct) * 100.0)
    rmse = float(np.sqrt(np.mean((pred[:2] - actual[:2]) ** 2)))
    denom = float(np.std(actual[:2], ddof=0) + eps)
    nrmse = float(rmse / denom)
    return {
        "joint_admissible": admissible,
        "primary_metric": 1.0 - admissible,
        "order_violation": order_violation,
        "h1_within_10pct": h1_within,
        "h2_within_10pct": h2_within,
        "mape": mape,
        "nrmse": nrmse,
        "pred_h1": float(pred[0]),
        "pred_h2": float(pred[1]),
        "actual_h1": float(actual[0]),
        "actual_h2": float(actual[1]),
    }


def new_seed_state() -> Dict[str, Any]:
    return {"folds": [], "failures": [], "success": True, "last_fold": None}


def new_condition_state() -> Dict[str, Any]:
    return {
        "seed_records": {},
        "regime_cells": defaultdict(lambda: {
            "primary_metric": [],
            "admissible": [],
            "order_violation": [],
            "h1_within_10pct": [],
            "h2_within_10pct": [],
        }),
        "full_cv_preds": [],
        "full_cv_actuals": [],
        "family": None,
        "comparator_name": None,
    }


def run_single_fold(
    condition_name: str,
    seed: int,
    fold: FoldSpec,
    values: np.ndarray,
    registry: Dict[str, ConditionSpec],
    *,
    max_epochs: int,
    stop_fn: Callable[[], bool],
):
    set_all_seeds(seed)
    model = registry[condition_name].factory()
    model.fit(
        values,
        train_end=fold.train_end,
        val_end=fold.val_end,
        seed=seed,
        max_epochs_override=max_epochs,
        stop_fn=stop_fn,
    )
    pred = model.predict(values, test_start=fold.test_start)
    actual = np.asarray(values[fold.test_start : fold.test_start + HYPERPARAMETERS["horizon"]], dtype=np.float32)
    metrics = compute_fold_metrics(pred, actual)
    return metrics, np.asarray(pred, dtype=np.float32), actual


def run_ablation_checks(values: np.ndarray, fold: FoldSpec, registry: Dict[str, ConditionSpec], stop_fn: Callable[[], bool]) -> None:
    probe_pairs = [
        (
            "order_aware_nlinear_nonnegative_delta_two_head_forecaster",
            "order_aware_nlinear_without_nonnegative_delta_constraint",
        ),
        (
            "nlinear_two_head_mae_forecaster_with_foldwise_robust_scaling",
            "nlinear_two_head_with_fixed_standard_scaling",
        ),
        (
            "volatility_gated_trend_meanreversion_two_expert_ensemble",
            "volatility_gated_two_expert_ensemble_with_uniform_router",
        ),
    ]
    for left_name, right_name in probe_pairs:
        left_metrics, left_pred, _ = run_single_fold(
            left_name,
            0,
            fold,
            values,
            registry,
            max_epochs=1,
            stop_fn=stop_fn,
        )
        right_metrics, right_pred, _ = run_single_fold(
            right_name,
            0,
            fold,
            values,
            registry,
            max_epochs=1,
            stop_fn=stop_fn,
        )
        outputs_differ = not np.allclose(left_pred, right_pred, atol=1e-6, rtol=1e-6)
        print(f"ABLATION_CHECK: {left_name} vs {right_name} outputs_differ={outputs_differ}")
        if not outputs_differ:
            raise AssertionError(f"Ablation broken: {left_name} and {right_name} produced identical outputs")


def run_pilot_calibration(values: np.ndarray, fold: FoldSpec, registry: Dict[str, ConditionSpec], stop_fn: Callable[[], bool]) -> Tuple[float, float, int]:
    pilot_conditions = [
        "nlinear_two_head_mae_forecaster_with_foldwise_robust_scaling",
        "order_aware_nlinear_nonnegative_delta_two_head_forecaster",
    ]
    pilot_values: List[float] = []
    success_count = 0
    total = 0
    pilot_start = time.perf_counter()
    for condition_name in pilot_conditions:
        for seed in SEEDS:
            total += 1
            try:
                metrics, _, _ = run_single_fold(
                    condition_name,
                    seed,
                    fold,
                    values,
                    registry,
                    max_epochs=HYPERPARAMETERS["pilot_epochs"],
                    stop_fn=stop_fn,
                )
                pilot_values.append(float(metrics["primary_metric"]))
                success_count += 1
            except Exception as exc:
                print(f"CONDITION_FAILED: {condition_name} seed={seed} pilot_fold={fold.fold_index + 1} {exc}")
                pilot_values.append(1.0)
    pilot_elapsed = time.perf_counter() - pilot_start
    pilot_success_rate = success_count / max(1, total)
    pilot_std = float(np.std(np.asarray(pilot_values, dtype=np.float64), ddof=1)) if len(pilot_values) > 1 else 0.0
    print(
        f"CALIBRATION: regime={fold.regime_cell} "
        f"pilot_success_rate={pilot_success_rate:.4f} "
        f"pilot_primary_metric_std={pilot_std:.4f}"
    )
    return pilot_elapsed, pilot_success_rate, total


def compute_hypothesis_support_rate(seed_records: Dict[int, Dict[str, Any]], all_primary: Dict[str, Dict[int, float]], comparator_names: Sequence[str]) -> float:
    if not comparator_names:
        return float("nan")
    if len(comparator_names) == 1:
        comp_values = all_primary.get(comparator_names[0], {})
        support = []
        for seed, seed_state in seed_records.items():
            if seed_state["last_fold"] is None:
                continue
            method_value = float(seed_state["last_fold"]["primary_metric"]) if seed_state["success"] else 1.0
            comparator_value = float(comp_values.get(seed, 1.0))
            support.append(method_value < comparator_value)
        return float(np.mean(support)) if support else float("nan")
    support = []
    for seed, seed_state in seed_records.items():
        if seed_state["last_fold"] is None:
            continue
        method_value = float(seed_state["last_fold"]["primary_metric"]) if seed_state["success"] else 1.0
        comparator_value = min(float(all_primary.get(name, {}).get(seed, 1.0)) for name in comparator_names)
        support.append(method_value < comparator_value)
    return float(np.mean(support)) if support else float("nan")


def paired_analysis(method_name: str, method_values: Dict[int, float], baseline_name: str, baseline_values: Dict[int, float]) -> Dict[str, Any]:
    common_seeds = sorted(set(method_values.keys()) & set(baseline_values.keys()))
    if not common_seeds:
        print(f"PAIRED: {method_name} vs {baseline_name} insufficient_pairs=True")
        return {
            "method": method_name,
            "baseline": baseline_name,
            "n_pairs": 0,
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "effect_size_cohen_d": float("nan"),
            "rank_biserial": float("nan"),
            "bootstrap_ci95": None,
        }
    method_arr = np.asarray([method_values[s] for s in common_seeds], dtype=np.float64)
    baseline_arr = np.asarray([baseline_values[s] for s in common_seeds], dtype=np.float64)
    diffs = method_arr - baseline_arr
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    cohen_d = mean_diff / (std_diff + 1e-12)
    try:
        if np.allclose(diffs, 0.0):
            p_value = 1.0
        else:
            p_value = float(stats.wilcoxon(method_arr, baseline_arr, zero_method="wilcox", alternative="two-sided", mode="auto").pvalue)
    except Exception:
        p_value = float("nan")
    t_stat = float("nan")
    if len(diffs) >= 10:
        try:
            t_stat = float(stats.ttest_rel(method_arr, baseline_arr).statistic)
        except Exception:
            t_stat = float("nan")
    nonzero = diffs[~np.isclose(diffs, 0.0)]
    if len(nonzero):
        pos = float(np.sum(nonzero > 0))
        neg = float(np.sum(nonzero < 0))
        rank_biserial = float((pos - neg) / len(nonzero))
    else:
        rank_biserial = 0.0
    ci = bootstrap_mean_ci(diffs, n_boot=HYPERPARAMETERS["bootstrap_samples"], seed=0) if len(diffs) >= 5 else None
    ci_str = "skipped_n_lt_5" if ci is None else f"[{ci[0]:.6f}, {ci[1]:.6f}]"
    print(
        f"PAIRED: {method_name} vs {baseline_name} "
        f"mean_diff={mean_diff:.6f} std_diff={std_diff:.6f} "
        f"t_stat={t_stat:.6f} p_value={p_value:.6f} "
        f"effect_size_cohen_d={cohen_d:.6f} rank_biserial={rank_biserial:.6f} "
        f"bootstrap_ci95={ci_str}"
    )
    return {
        "method": method_name,
        "baseline": baseline_name,
        "n_pairs": len(common_seeds),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "effect_size_cohen_d": cohen_d,
        "rank_biserial": rank_biserial,
        "bootstrap_ci95": None if ci is None else [float(ci[0]), float(ci[1])],
    }


def summarize_condition(condition_name: str, spec: ConditionSpec, state: Dict[str, Any], harness: ExperimentHarness, all_primary: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
    seed_records = state["seed_records"]
    successful_primary: List[float] = []
    unconditional_primary: List[float] = []
    successful_admissible: List[float] = []
    successful_order_violation: List[float] = []
    successful_h1: List[float] = []
    successful_h2: List[float] = []
    successful_seeds = 0
    full_cv_preds: List[np.ndarray] = []
    full_cv_actuals: List[np.ndarray] = []
    regime_report: Dict[str, Dict[str, float]] = {}

    for seed in SEEDS:
        seed_state = seed_records.get(seed)
        if seed_state is None:
            unconditional_primary.append(1.0)
            continue
        if seed_state["success"] and seed_state["last_fold"] is not None:
            successful_seeds += 1
            successful_primary.append(float(seed_state["last_fold"]["primary_metric"]))
            successful_admissible.append(float(seed_state["last_fold"]["joint_admissible"]))
            successful_order_violation.append(float(seed_state["last_fold"]["order_violation"]))
            successful_h1.append(float(seed_state["last_fold"]["h1_within_10pct"]))
            successful_h2.append(float(seed_state["last_fold"]["h2_within_10pct"]))
            unconditional_primary.append(float(seed_state["last_fold"]["primary_metric"]))
        else:
            unconditional_primary.append(1.0)

    for cell in ["low/negative_or_flat", "low/positive", "high/negative_or_flat", "high/positive"]:
        cell_stats = state["regime_cells"].get(cell, {
            "primary_metric": [],
            "admissible": [],
            "order_violation": [],
            "h1_within_10pct": [],
            "h2_within_10pct": [],
        })
        count = len(cell_stats["primary_metric"])
        cell_primary = float(np.mean(cell_stats["primary_metric"])) if count else float("nan")
        cell_adm = float(np.mean(cell_stats["admissible"])) if count else float("nan")
        cell_order = float(np.mean(cell_stats["order_violation"])) if count else float("nan")
        cell_h1 = float(np.mean(cell_stats["h1_within_10pct"])) if count else float("nan")
        cell_h2 = float(np.mean(cell_stats["h2_within_10pct"])) if count else float("nan")
        regime_report[cell] = {
            "count": count,
            "primary_metric": cell_primary,
            "admissible": cell_adm,
            "order_violation": cell_order,
            "h1_within_10pct": cell_h1,
            "h2_within_10pct": cell_h2,
        }
        print(
            f"condition={condition_name} regime={cell} primary_metric: {cell_primary:.6f} "
            f"last_fold_joint_admissible_rate: {cell_adm:.6f} "
            f"last_fold_order_violation_rate: {cell_order:.6f} "
            f"last_fold_h1_within_10pct_rate: {cell_h1:.6f} "
            f"last_fold_h2_within_10pct_rate: {cell_h2:.6f} "
            f"count={count}"
        )

    for seed, seed_state in seed_records.items():
        seed_primary = float(seed_state["last_fold"]["primary_metric"]) if seed_state["last_fold"] is not None else 1.0
        print(f"condition={condition_name} seed={seed} primary_metric: {seed_primary:.6f}")

    for pred in state["full_cv_preds"]:
        full_cv_preds.append(np.asarray(pred, dtype=np.float64))
    for actual in state["full_cv_actuals"]:
        full_cv_actuals.append(np.asarray(actual, dtype=np.float64))
    if full_cv_preds and full_cv_actuals:
        preds = np.concatenate(full_cv_preds, axis=0)
        actuals = np.concatenate(full_cv_actuals, axis=0)
        full_cv_mape = mean_absolute_percentage_error(preds, actuals)
        full_cv_nrmse = normalized_rmse(preds, actuals)
    else:
        full_cv_mape = float("nan")
        full_cv_nrmse = float("nan")

    cell_values = [v["admissible"] for v in regime_report.values() if np.isfinite(v["admissible"])]
    regime_gap = float(np.max(cell_values) - np.min(cell_values)) if len(cell_values) >= 2 else 0.0

    primary_mean = float(np.mean(successful_primary)) if successful_primary else float("nan")
    primary_std = float(np.std(successful_primary, ddof=1)) if len(successful_primary) > 1 else 0.0
    admissible_mean = float(np.mean(successful_admissible)) if successful_admissible else float("nan")
    order_mean = float(np.mean(successful_order_violation)) if successful_order_violation else float("nan")
    h1_mean = float(np.mean(successful_h1)) if successful_h1 else float("nan")
    h2_mean = float(np.mean(successful_h2)) if successful_h2 else float("nan")
    unconditional_mean = float(np.mean(unconditional_primary)) if unconditional_primary else float("nan")
    success_rate = f"{successful_seeds}/{len(SEEDS)}"

    print(f"condition={condition_name} primary_metric_mean: {primary_mean:.6f} primary_metric_std: {primary_std:.6f}")
    print(f"condition={condition_name} last_fold_joint_admissible_rate: {admissible_mean:.6f}")
    print(f"condition={condition_name} last_fold_order_violation_rate: {order_mean:.6f}")
    print(f"condition={condition_name} last_fold_h1_within_10pct_rate: {h1_mean:.6f}")
    print(f"condition={condition_name} last_fold_h2_within_10pct_rate: {h2_mean:.6f}")
    print(f"condition={condition_name} success_rate: {success_rate}")
    print(f"condition={condition_name} unconditional_primary_metric_mean: {unconditional_mean:.6f}")
    print(f"condition={condition_name} full_cv_mape: {full_cv_mape:.6f}")
    print(f"condition={condition_name} full_cv_nrmse: {full_cv_nrmse:.6f}")
    print(f"condition={condition_name} regime_sliced_compliance_gap: {regime_gap:.6f}")

    if spec.family == "proposed" and spec.comparator:
        support_rate = compute_hypothesis_support_rate(seed_records, all_primary, spec.comparator)
        print(f"condition={condition_name} hypothesis_support_rate: {support_rate:.6f}")
    else:
        support_rate = float("nan")

    safe_report_metric(harness, f"{condition_name}/primary_metric_mean", primary_mean if np.isfinite(primary_mean) else 1.0)
    safe_report_metric(harness, f"{condition_name}/unconditional_primary_metric_mean", unconditional_mean if np.isfinite(unconditional_mean) else 1.0)
    safe_report_metric(harness, f"{condition_name}/full_cv_mape", full_cv_mape if np.isfinite(full_cv_mape) else 1e6)
    safe_report_metric(harness, f"{condition_name}/full_cv_nrmse", full_cv_nrmse if np.isfinite(full_cv_nrmse) else 1e6)

    return {
        "primary_metric_mean": primary_mean,
        "primary_metric_std": primary_std,
        "unconditional_primary_metric_mean": unconditional_mean,
        "last_fold_joint_admissible_rate": admissible_mean,
        "last_fold_order_violation_rate": order_mean,
        "last_fold_h1_within_10pct_rate": h1_mean,
        "last_fold_h2_within_10pct_rate": h2_mean,
        "success_rate": success_rate,
        "success_count": successful_seeds,
        "full_cv_mape": full_cv_mape,
        "full_cv_nrmse": full_cv_nrmse,
        "regime_sliced_compliance_gap": regime_gap,
        "hypothesis_support_rate": support_rate,
        "regime_report": regime_report,
    }


def summarize_all_conditions(registry: OrderedDict[str, ConditionSpec], all_states: Dict[str, Any]) -> Dict[str, float]:
    summary = {}
    for name in registry.keys():
        seed_records = all_states[name]["seed_records"]
        values = []
        for seed in SEEDS:
            seed_state = seed_records.get(seed)
            if seed_state is None or seed_state["last_fold"] is None:
                values.append(1.0)
            else:
                values.append(float(seed_state["last_fold"]["primary_metric"]))
        summary[name] = float(np.mean(values)) if values else float("nan")
    return summary


def maybe_print_degeneracy_warning(summary: Dict[str, float]) -> None:
    vals = [v for v in summary.values() if np.isfinite(v)]
    if len(vals) >= 2 and np.allclose(vals, vals[0]):
        print(f"WARNING: DEGENERATE_METRICS all conditions have same mean={vals[0]:.6f}")


def main() -> int:
    harness = ExperimentHarness(time_budget=HYPERPARAMETERS["time_budget_seconds"])
    registry = build_condition_registry()
    print(f"REGISTERED_CONDITIONS: {', '.join(registry.keys())}")
    for name in EXPECTED_CONDITIONS:
        if name not in registry:
            print(f"MISSING_CONDITION: {name}")
    print(
        "METRIC_DEF: primary_metric | direction=lower | "
        "desc=last-fold joint noncompliance rate; fraction of terminal forecasts failing order and/or ±10% gate"
    )
    print(
        f"SEED_COUNT: {len(SEEDS)} (fixed minimum, budget={HYPERPARAMETERS['time_budget_seconds']}s, "
        f"conditions={len(registry)})"
    )
    print("SEED_WARNING: only 3 seeds used due to time budget")

    values, raw_df = load_authoritative_series(HYPERPARAMETERS["dataset_path"], HYPERPARAMETERS["target_column"])
    fold_specs = build_fold_specs(
        values,
        folds=HYPERPARAMETERS["folds"],
        horizon=HYPERPARAMETERS["horizon"],
        validation_size=HYPERPARAMETERS["validation_size"],
        context_length=HYPERPARAMETERS["context_length"],
        season_length=HYPERPARAMETERS["season_length"],
        terminal_gap=HYPERPARAMETERS["terminal_gap"],
    )

    results_payload: Dict[str, Any] = {
        "hyperparameters": HYPERPARAMETERS,
        "metadata": {
            "dataset_path": HYPERPARAMETERS["dataset_path"],
            "target_column": HYPERPARAMETERS["target_column"],
            "n_observations": int(len(values)),
            "fold_specs": [asdict(spec) for spec in fold_specs],
        },
        "pilot": {},
        "conditions": {},
        "metrics": {},
    }

    start_time = time.perf_counter()
    deadline = start_time + 0.8 * HYPERPARAMETERS["time_budget_seconds"]
    stop_announced = False

    def should_stop() -> bool:
        return bool(harness.should_stop() or time.perf_counter() >= deadline)

    run_ablation_checks(values, fold_specs[0], registry, should_stop)
    pilot_elapsed, pilot_success_rate, pilot_runs = run_pilot_calibration(values, fold_specs[0], registry, should_stop)
    pilot_scale = (len(registry) * len(SEEDS) * len(fold_specs) * HYPERPARAMETERS["max_epochs"]) / max(1, pilot_runs * HYPERPARAMETERS["pilot_epochs"])
    estimated_total = pilot_elapsed * pilot_scale
    print(f"TIME_ESTIMATE: {estimated_total:.2f}s")

    results_payload["pilot"] = {
        "elapsed_seconds": float(pilot_elapsed),
        "success_rate": float(pilot_success_rate),
        "run_count": int(pilot_runs),
        "estimated_total_seconds": float(estimated_total),
    }

    all_states: Dict[str, Dict[str, Any]] = {name: new_condition_state() for name in registry.keys()}
    all_primary: Dict[str, Dict[int, float]] = {name: {} for name in registry.keys()}
    for name, spec in registry.items():
        all_states[name]["family"] = spec.family
        all_states[name]["comparator_name"] = spec.comparator

    print("FOLD_PROGRESS: starting expanding-window evaluation")

    try:
        for seed_idx, seed in enumerate(SEEDS):
            for condition_name, spec in registry.items():
                if should_stop():
                    if not stop_announced:
                        print("TIME_GUARD: stopping at 80% budget")
                        stop_announced = True
                    break
                condition_state = all_states[condition_name]
                seed_state = condition_state["seed_records"].setdefault(seed, new_seed_state())
                try:
                    for fold in fold_specs:
                        metrics, pred, actual = run_single_fold(
                            condition_name,
                            seed,
                            fold,
                            values,
                            registry,
                            max_epochs=HYPERPARAMETERS["max_epochs"],
                            stop_fn=should_stop,
                        )
                        seed_state["folds"].append({"fold_index": fold.fold_index, **metrics})
                        condition_state["full_cv_preds"].append(pred)
                        condition_state["full_cv_actuals"].append(actual)
                        regime_stats = condition_state["regime_cells"][fold.regime_cell]
                        regime_stats["primary_metric"].append(metrics["primary_metric"])
                        regime_stats["admissible"].append(metrics["joint_admissible"])
                        regime_stats["order_violation"].append(metrics["order_violation"])
                        regime_stats["h1_within_10pct"].append(metrics["h1_within_10pct"])
                        regime_stats["h2_within_10pct"].append(metrics["h2_within_10pct"])
                        if fold.fold_index == len(fold_specs) - 1:
                            seed_state["last_fold"] = metrics
                            all_primary[condition_name][seed] = float(metrics["primary_metric"])
                            print(f"condition={condition_name} seed={seed} primary_metric: {metrics['primary_metric']:.6f}")
                    if seed_state["last_fold"] is None:
                        seed_state["last_fold"] = {
                            "primary_metric": 1.0,
                            "joint_admissible": 0.0,
                            "order_violation": 1.0,
                            "h1_within_10pct": 0.0,
                            "h2_within_10pct": 0.0,
                        }
                        all_primary[condition_name][seed] = 1.0
                        seed_state["success"] = False
                except Exception as exc:
                    seed_state["success"] = False
                    seed_state["failures"].append(str(exc))
                    seed_state["last_fold"] = {
                        "primary_metric": 1.0,
                        "joint_admissible": 0.0,
                        "order_violation": 1.0,
                        "h1_within_10pct": 0.0,
                        "h2_within_10pct": 0.0,
                    }
                    all_primary[condition_name][seed] = 1.0
                    print(f"CONDITION_FAILED: {condition_name} seed={seed} {exc}")
                    print(f"condition={condition_name} seed={seed} primary_metric: 1.000000")

                save_results_json(Path("results.json"), {
                    "hyperparameters": HYPERPARAMETERS,
                    "metadata": results_payload["metadata"],
                    "pilot": results_payload["pilot"],
                    "conditions": results_payload["conditions"],
                    "metrics": results_payload["metrics"],
                })

        summary = summarize_all_conditions(registry, all_states)
        maybe_print_degeneracy_warning(summary)
        results_payload["metrics"]["summary"] = {name: float(val) for name, val in summary.items()}

        for condition_name, spec in registry.items():
            condition_state = all_states[condition_name]
            condition_summary = summarize_condition(condition_name, spec, condition_state, harness, all_primary)
            results_payload["conditions"][condition_name] = {
                "family": spec.family,
                "comparator": spec.comparator,
                "seed_records": json_safe(condition_state["seed_records"]),
                "regime_report": json_safe(condition_summary["regime_report"]),
                "aggregates": json_safe(condition_summary),
            }

        pairwise_results = {}
        for condition_name, spec in registry.items():
            if not spec.comparator:
                continue
            if len(spec.comparator) == 1:
                baseline_name = spec.comparator[0]
                pairwise_results[f"{condition_name}__vs__{baseline_name}"] = paired_analysis(
                    condition_name,
                    all_primary[condition_name],
                    baseline_name,
                    all_primary[baseline_name],
                )
            else:
                comp_best = {seed: min(all_primary.get(name, {}).get(seed, 1.0) for name in spec.comparator) for seed in SEEDS}
                pairwise_results[f"{condition_name}__vs__best_comparator"] = paired_analysis(
                    condition_name,
                    all_primary[condition_name],
                    "best_comparator",
                    comp_best,
                )
        results_payload["metrics"]["paired"] = pairwise_results
        summary_line = ", ".join(f"{name}={summary[name]:.6f}" for name in registry.keys())
        print(f"SUMMARY: {summary_line}")
        results_payload["metrics"]["summary_line"] = summary_line
        results_payload["metrics"]["runtime_seconds"] = float(time.perf_counter() - start_time)
        results_payload["metrics"]["pilot_success_rate"] = float(pilot_success_rate)
        results_payload["metrics"]["pilot_estimated_total_seconds"] = float(estimated_total)

    finally:
        save_results_json(Path("results.json"), results_payload)
        try:
            harness.finalize()
        except Exception as exc:
            print(f"CONDITION_FAILED: harness_finalize {exc}")
        save_results_json(Path("results.json"), results_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
