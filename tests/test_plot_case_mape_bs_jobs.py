from __future__ import annotations

import csv
from pathlib import Path

from scripts.plot_case_mape_bs_jobs import collect_case_metrics, plot_target


HEADER = [
    "rank",
    "model",
    "mean_fold_mae",
    "mean_fold_mse",
    "mean_fold_rmse",
    "fold_count",
    "mean_fold_mape",
    "mean_fold_nrmse",
    "mean_fold_r2",
]


def _write_leaderboard(run_dir: Path, mapes: list[float]) -> None:
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "leaderboard.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for rank, mape in enumerate(mapes, start=1):
            writer.writerow([rank, f"Model{rank}", 1.0, 1.0, 1.0, 12, mape, 1.0, -1.0])


def test_collect_case_metrics_aggregates_jobs_only(tmp_path: Path) -> None:
    _write_leaderboard(tmp_path / "feature_set_brentoil_case1_jobs_0", [0.10, 0.12])
    _write_leaderboard(tmp_path / "feature_set_brentoil_case1_jobs_1", [0.08])
    _write_leaderboard(tmp_path / "feature_set_bs_brentoil_case1_bs_jobs_0", [0.07, 0.09])
    _write_leaderboard(tmp_path / "feature_set_bs_brentoil_case1_bs_jobs_1", [0.11])
    _write_leaderboard(tmp_path / "feature_set_legacy" / "feature_set_brentoil_case1", [0.50])

    metrics = collect_case_metrics(tmp_path)

    by_key = {(metric.variant, metric.target, metric.case): metric for metric in metrics}
    baseline = by_key[("feature_set", "brentoil", "1")]
    bs_added = by_key[("feature_set_bs", "brentoil", "1")]

    assert baseline.run_count == 2
    assert baseline.row_count == 3
    assert round(baseline.mean_mape, 2) == 10.00
    assert bs_added.run_count == 2
    assert bs_added.row_count == 3
    assert round(bs_added.mean_mape, 2) == 9.00
    assert all(metric.target == "brentoil" for metric in metrics)


def test_plot_target_creates_png(tmp_path: Path) -> None:
    for case, base_mape, bs_mape in [("1", 8.0, 7.4), ("2", 8.5, 8.1), ("3", 7.9, 7.8), ("4", 7.7, 7.3)]:
        _write_leaderboard(tmp_path / f"feature_set_wti_case{case}_jobs_0", [base_mape / 100])
        _write_leaderboard(tmp_path / f"feature_set_bs_wti_case{case}_bs_jobs_0", [bs_mape / 100])

    metrics = collect_case_metrics(tmp_path)
    output_path = tmp_path / "artifacts" / "wti_case_mape.png"

    plot_target(metrics, target="wti", output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
