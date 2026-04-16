"""Smoke tests for ``scripts/finding.py`` last-fold grid."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from runtime_support.runner import _build_tscv_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
FINDING_MAIN = REPO_ROOT / "tests/fixtures/finding_aaforecast_smoke_main.yaml"
RETRIEVAL_LINKED_MAIN = REPO_ROOT / "tests/fixtures/retrieval_runtime_smoke_linked.yaml"


def _load_finding_script():
    path = REPO_ROOT / "scripts/finding.py"
    spec = importlib.util.spec_from_file_location("_finding_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_iter_combos_skips_upward_not_subset_of_hist_exog() -> None:
    finding = _load_finding_script()
    base = {"A", "B", "C"}
    exog_grid = [["A", "B"], ["A"]]
    upward_grid = [[], ["A"], ["A", "B"]]
    combos = list(
        finding.iter_combos(
            input_sizes=[1],
            exog_grid=exog_grid,
            upward_grid=upward_grid,
            base_dataset_cols=base,
        )
    )
    assert len(combos) == 5
    assert finding.count_incompatible_exog_upward_pairs(
        input_sizes=[1],
        exog_grid=exog_grid,
        upward_grid=upward_grid,
        base_dataset_cols=base,
    ) == 1


def test_build_tscv_splits_last_fold_matches_runner_policy() -> None:
    """Single-window policy: train_end = N - gap - horizon - step*(n_windows-1)."""
    cv = SimpleNamespace(gap=0, horizon=2, step_size=8, n_windows=1, max_train_size=None)
    splits = _build_tscv_splits(100, cv)
    assert len(splits) == 1
    train_idx, test_idx = splits[-1]
    assert train_idx == list(range(0, 98))
    assert test_idx == list(range(98, 100))


def test_finding_cli_writes_csv(tmp_path: Path) -> None:
    out_csv = tmp_path / "finding_out.csv"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/finding.py"),
        "--config",
        str(FINDING_MAIN),
        "--input-sizes",
        "3,4",
        "--metric",
        "align_weighted_last",
        "--output-csv",
        str(out_csv),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_csv.is_file()
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    required = {
        "eval_slice",
        "signature_backend",
        "input_size",
        "hist_exog_cols",
        "upward_tail_cols",
        "rank_score",
        "metric_mode",
        "actual_horizon_cum_ret",
        "bank_size",
    }
    assert required.issubset(rows[0].keys())
    assert all(r["signature_backend"] == "retrieval_plugin_star" for r in rows)


def test_finding_retrieval_only_linked_config_writes_csv(tmp_path: Path) -> None:
    """Standalone retrieval-route YAML: no AAForecast / backbone."""
    out_csv = tmp_path / "retrieval_only.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/finding.py"),
            "--config",
            str(RETRIEVAL_LINKED_MAIN),
            "--input-sizes",
            "3,4",
            "--output-csv",
            str(out_csv),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_csv.is_file()
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["signature_backend"] == "retrieval_plugin_star"


def test_finding_min_input_size_filters_all_returns_error() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/finding.py"),
            "--config",
            str(FINDING_MAIN),
            "--input-sizes",
            "3,4",
            "--min-input-size",
            "99",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "no input_sizes remain" in proc.stderr


def test_finding_max_tail_more_train_rows_than_last_fold_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/finding.py"),
            "--config",
            str(RETRIEVAL_LINKED_MAIN),
            "--eval-slice",
            "max_tail",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "train_rows=9" in proc.stdout


def test_finding_dry_run_zero_exit() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/finding.py"),
            "--config",
            str(FINDING_MAIN),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "combos=" in proc.stdout
