from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR_PATTERN = re.compile(
    r"^(feature_set(?:_bs)?)_(brentoil|wti)_case(\d)(?:_bs)?_jobs_(\d+)$"
)
TARGET_LABELS = {"brentoil": "BrentCrude", "wti": "WTI"}
VARIANT_LABELS = {"feature_set": "Baseline", "feature_set_bs": "BS added"}
VARIANT_COLORS = {"feature_set": "#9AA5B1", "feature_set_bs": "#2F6FED"}


@dataclass(frozen=True)
class CaseMetric:
    variant: str
    target: str
    case: str
    mean_mape: float
    run_count: int
    row_count: int


def collect_case_metrics(runs_root: Path) -> list[CaseMetric]:
    aggregates: dict[tuple[str, str, str], dict[str, list[float] | set[str]]] = defaultdict(
        lambda: {"values": [], "jobs": set()}
    )

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(run_dir.name)
        if not match:
            continue
        variant, target, case, job_idx = match.groups()
        leaderboard_path = run_dir / "summary" / "leaderboard.csv"
        if not leaderboard_path.exists():
            continue
        with leaderboard_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        key = (variant, target, case)
        aggregates[key]["jobs"].add(job_idx)  # type: ignore[index]
        aggregates[key]["values"].extend(
            float(row["mean_fold_mape"]) for row in rows if row.get("mean_fold_mape")
        )  # type: ignore[index]

    metrics: list[CaseMetric] = []
    for (variant, target, case), payload in sorted(aggregates.items()):
        values = list(payload["values"])  # type: ignore[arg-type]
        jobs = payload["jobs"]  # type: ignore[assignment]
        if not values:
            continue
        metrics.append(
            CaseMetric(
                variant=variant,
                target=target,
                case=case,
                mean_mape=mean(values) * 100,
                run_count=len(jobs),
                row_count=len(values),
            )
        )
    return metrics


def plot_target(metrics: list[CaseMetric], target: str, output_path: Path) -> None:
    target_metrics = sorted(
        (metric for metric in metrics if metric.target == target),
        key=lambda metric: (int(metric.case), metric.variant),
    )
    if not target_metrics:
        raise ValueError(f"No metrics found for target={target!r}")

    cases = sorted({metric.case for metric in target_metrics}, key=int)
    baseline = {
        metric.case: metric.mean_mape
        for metric in target_metrics
        if metric.variant == "feature_set"
    }
    bs_added = {
        metric.case: metric.mean_mape
        for metric in target_metrics
        if metric.variant == "feature_set_bs"
    }
    if set(cases) != set(baseline) or set(cases) != set(bs_added):
        raise ValueError(f"Missing baseline/BS pair for target={target!r}: cases={cases}")

    x_positions = range(len(cases))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5.2))

    baseline_vals = [baseline[case] for case in cases]
    bs_vals = [bs_added[case] for case in cases]
    baseline_bars = axis.bar(
        [x - width / 2 for x in x_positions],
        baseline_vals,
        width=width,
        color=VARIANT_COLORS["feature_set"],
        label=VARIANT_LABELS["feature_set"],
    )
    bs_bars = axis.bar(
        [x + width / 2 for x in x_positions],
        bs_vals,
        width=width,
        color=VARIANT_COLORS["feature_set_bs"],
        label=VARIANT_LABELS["feature_set_bs"],
    )

    axis.set_title(f"{TARGET_LABELS[target]} case-level mean MAPE from runs/*_jobs_*")
    axis.set_xlabel("Case")
    axis.set_ylabel("Mean MAPE (%)")
    axis.set_xticks(list(x_positions), [f"Case {case}" for case in cases])
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False)
    axis.set_axisbelow(True)

    for bars in (baseline_bars, bs_bars):
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for idx, case in enumerate(cases):
        delta = bs_added[case] - baseline[case]
        axis.text(
            idx,
            max(baseline[case], bs_added[case]) + 0.35,
            f"Δ {delta:+.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0F172A",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot case-level mean MAPE baseline vs BS charts from runs/*_jobs_* artifacts."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "jobs_bs_case_mape",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = collect_case_metrics(args.runs_root)
    if not metrics:
        raise SystemExit(f"No matching runs found under {args.runs_root}")

    outputs = {
        "brentoil": args.output_dir / "brentcrude_case_mape.png",
        "wti": args.output_dir / "wti_case_mape.png",
    }
    for target, output_path in outputs.items():
        plot_target(metrics, target=target, output_path=output_path)
        print(f"saved={output_path}")


if __name__ == "__main__":
    main()
