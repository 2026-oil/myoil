from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


EXPECTED_TARGETS = ("Com_BrentCrudeOil", "Com_CrudeOil")


@dataclass(frozen=True)
class RunMetric:
    run_dir: Path
    target: str
    h1_h5_mean_mape_pct: float
    h7_h8_mean_mape_pct: float
    h6_mean_mape_pct: float
    last_fold_h7_mape_pct: float
    last_fold_h8_mape_pct: float
    peak_step_hit_rate_h6_h8: float
    source: str

    def selection_key(self) -> tuple[float, float, float, float, str]:
        return (
            self.h7_h8_mean_mape_pct,
            max(self.last_fold_h7_mape_pct, self.last_fold_h8_mape_pct),
            self.h6_mean_mape_pct,
            self.h1_h5_mean_mape_pct,
            self.run_dir.name,
        )


def _mape_pct(actual: float, predicted: float) -> float:
    if actual == 0:
        raise ValueError("MAPE is undefined for zero actual values")
    return abs(actual - predicted) / abs(actual) * 100.0


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    return rows


def _peak_hit_rate(rows: list[dict[str, str]]) -> float:
    by_fold: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_fold.setdefault(int(row["fold_idx"]), []).append(row)
    fold_scores: list[float] = []
    for fold_rows in by_fold.values():
        horizon_rows = [row for row in fold_rows if int(row["horizon_step"]) in {6, 7, 8}]
        if not horizon_rows:
            continue
        actual_peak = max(horizon_rows, key=lambda row: float(row["y"]))["horizon_step"]
        pred_peak = max(horizon_rows, key=lambda row: float(row["y_hat"]))["horizon_step"]
        fold_scores.append(1.0 if actual_peak == pred_peak else 0.0)
    return mean(fold_scores) if fold_scores else 0.0


def _run_metric(run_dir: Path, csv_path: Path, *, source: str) -> RunMetric:
    rows = _load_rows(csv_path)
    targets = {row["unique_id"] for row in rows}
    if len(targets) != 1:
        raise ValueError(f"{csv_path} must contain exactly one target")
    target = next(iter(targets))
    h1_h5: list[float] = []
    h6: list[float] = []
    h7_h8: list[float] = []
    by_fold: dict[int, dict[int, float]] = {}
    for row in rows:
        fold = int(row["fold_idx"])
        step = int(row["horizon_step"])
        metric = _mape_pct(float(row["y"]), float(row["y_hat"]))
        by_fold.setdefault(fold, {})[step] = metric
        if 1 <= step <= 5:
            h1_h5.append(metric)
        elif step == 6:
            h6.append(metric)
        elif step in {7, 8}:
            h7_h8.append(metric)
    if not h1_h5 or not h6 or len(h7_h8) < 2:
        raise ValueError(f"{csv_path} is missing required horizon metrics")
    last_fold = max(by_fold)
    last_fold_metrics = by_fold[last_fold]
    if 7 not in last_fold_metrics or 8 not in last_fold_metrics:
        raise ValueError(f"{csv_path} missing last-fold h7/h8 metrics")
    return RunMetric(
        run_dir=run_dir,
        target=target,
        h1_h5_mean_mape_pct=mean(h1_h5),
        h7_h8_mean_mape_pct=mean(h7_h8),
        h6_mean_mape_pct=mean(h6),
        last_fold_h7_mape_pct=last_fold_metrics[7],
        last_fold_h8_mape_pct=last_fold_metrics[8],
        peak_step_hit_rate_h6_h8=_peak_hit_rate(rows),
        source=source,
    )


def _collect(root: Path, patterns: list[str], *, source: str) -> list[RunMetric]:
    metrics: list[RunMetric] = []
    for pattern in patterns:
        for run_dir in sorted(root.glob(pattern)):
            csv_path = run_dir / "cv" / "NEC_forecasts.csv"
            if not csv_path.exists():
                continue
            try:
                metrics.append(_run_metric(run_dir, csv_path, source=source))
            except ValueError:
                continue
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Mechanical portfolio objective for NEC autoresearch.")
    parser.add_argument(
        "--seed-runs-root",
        type=Path,
        action="append",
        default=[],
        help="Seed runs root(s) from the original checkout.",
    )
    parser.add_argument(
        "--seed-pattern",
        action="append",
        default=["feature_set_nec_*"],
    )
    parser.add_argument(
        "--local-runs-root",
        type=Path,
        default=Path("runs"),
    )
    parser.add_argument(
        "--local-pattern",
        action="append",
        default=["feature_set_nec_neciso_*", "neciso_*"],
    )
    parser.add_argument("--h1_5_max", type=float, default=5.0)
    parser.add_argument("--h7_8_max", type=float, default=8.0)
    parser.add_argument("--last_fold_h7_max", type=float, default=8.0)
    parser.add_argument("--last_fold_h8_max", type=float, default=8.0)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero unless all configured thresholds are satisfied.",
    )
    args = parser.parse_args()

    all_metrics: list[RunMetric] = []
    for root in args.seed_runs_root:
        all_metrics.extend(_collect(root, args.seed_pattern, source="seed"))
    all_metrics.extend(_collect(args.local_runs_root, args.local_pattern, source="local"))
    if not all_metrics:
        raise SystemExit("No NEC forecast runs available for objective computation.")

    best_by_target: dict[str, RunMetric] = {}
    for target in EXPECTED_TARGETS:
        candidates = [metric for metric in all_metrics if metric.target == target]
        if candidates:
            best_by_target[target] = min(candidates, key=lambda metric: metric.selection_key())

    if set(best_by_target) != set(EXPECTED_TARGETS):
        missing = sorted(set(EXPECTED_TARGETS) - set(best_by_target))
        raise SystemExit(f"Missing target portfolio coverage for: {', '.join(missing)}")

    combined_h1_5 = mean(metric.h1_h5_mean_mape_pct for metric in best_by_target.values())
    combined_h7_8 = mean(metric.h7_h8_mean_mape_pct for metric in best_by_target.values())
    combined_h6 = mean(metric.h6_mean_mape_pct for metric in best_by_target.values())
    combined_peak = mean(metric.peak_step_hit_rate_h6_h8 for metric in best_by_target.values())
    last_fold_h7_max = max(metric.last_fold_h7_mape_pct for metric in best_by_target.values())
    last_fold_h8_max = max(metric.last_fold_h8_mape_pct for metric in best_by_target.values())
    constraint_gap = max(
        combined_h1_5 - args.h1_5_max,
        combined_h7_8 - args.h7_8_max,
        last_fold_h7_max - args.last_fold_h7_max,
        last_fold_h8_max - args.last_fold_h8_max,
    )

    payload = {
        "metric_name": "nec_portfolio_constraint_gap_pct",
        "metric_value": constraint_gap,
        "combined_h1_h5_mean_mape_pct": combined_h1_5,
        "combined_h7_h8_mean_mape_pct": combined_h7_8,
        "combined_h6_mean_mape_pct": combined_h6,
        "combined_peak_step_hit_rate_h6_h8": combined_peak,
        "last_fold_h7_max_mape_pct": last_fold_h7_max,
        "last_fold_h8_max_mape_pct": last_fold_h8_max,
        "thresholds": {
            "h1_5_max": args.h1_5_max,
            "h7_8_max": args.h7_8_max,
            "last_fold_h7_max": args.last_fold_h7_max,
            "last_fold_h8_max": args.last_fold_h8_max,
        },
        "best_by_target": {
            target: {
                "run_dir": str(metric.run_dir),
                "source": metric.source,
                "h1_h5_mean_mape_pct": metric.h1_h5_mean_mape_pct,
                "h7_h8_mean_mape_pct": metric.h7_h8_mean_mape_pct,
                "h6_mean_mape_pct": metric.h6_mean_mape_pct,
                "last_fold_h7_mape_pct": metric.last_fold_h7_mape_pct,
                "last_fold_h8_mape_pct": metric.last_fold_h8_mape_pct,
                "peak_step_hit_rate_h6_h8": metric.peak_step_hit_rate_h6_h8,
            }
            for target, metric in best_by_target.items()
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    if args.check and constraint_gap > 0:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
