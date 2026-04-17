from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_star_decomposition import (
    load_retrieval_star_config,
    select_tail_anomaly_mask,
    slice_input_window_ending_on_cutoff_calendar_day,
    tail_for_column,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "plot_star_decomposition.py"


def make_weekly_frame(rows: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dt = pd.date_range("2020-01-06", periods=rows, freq="W-MON")
    trend = np.linspace(55.0, 85.0, rows)
    seasonal = 1.0 + 0.08 * np.sin(2 * np.pi * np.arange(rows) / 4.0)
    wti = trend * seasonal + rng.normal(scale=0.8, size=rows)
    brent = (trend + 4.0) * (1.0 + 0.07 * np.cos(2 * np.pi * np.arange(rows) / 4.0)) + rng.normal(
        scale=0.8, size=rows
    )
    wti[[18, 57]] += [8.0, -9.0]
    brent[[21, 60]] += [10.0, -8.0]
    return pd.DataFrame(
        {
            "dt": dt,
            "Com_CrudeOil": wti,
            "Com_BrentCrudeOil": brent,
            "GPRD_THREAT": 120
            + 12 * np.sin(2 * np.pi * np.arange(rows) / 52.0)
            + rng.normal(scale=4.0, size=rows),
        }
    )


def run_script(
    csv_path: Path,
    output_dir: Path,
    *targets: str,
    cutoff_date: str | None = None,
    setting_path: Path | None = None,
    retrieval_config: Path | None = None,
    hist_exog_cols: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    cmd = [sys.executable, str(SCRIPT), "--input", str(csv_path), "--output-dir", str(output_dir)]
    if cutoff_date is not None:
        cmd.extend(["--cutoff-date", cutoff_date])
    if setting_path is not None:
        cmd.extend(["--setting", str(setting_path)])
    if retrieval_config is not None:
        cmd.extend(["--retrieval-config", str(retrieval_config)])
    if hist_exog_cols:
        cmd.append("--hist-exog-cols")
        cmd.extend(hist_exog_cols)
    if targets:
        cmd.extend(["--targets", *targets])
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_plot_star_decomposition_writes_two_expected_pngs(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    output_dir = tmp_path / "run"
    make_weekly_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = run_script(csv_path, output_dir)

    assert result.returncode == 0, result.stderr
    png_names = sorted(path.name for path in output_dir.glob("*.png"))
    assert png_names == [
        "brent_star_decomposition.png",
        "wti_star_decomposition.png",
    ]
    for path in output_dir.glob("*.png"):
        assert path.stat().st_size > 0


def make_dubai_frame(*, rows: int = 80, end: str = "2026-02-23") -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dt = pd.date_range(end=end, periods=rows, freq="D")
    base = 65.0 + 0.02 * np.arange(rows) + rng.normal(scale=0.12, size=rows)
    return pd.DataFrame(
        {
            "dt": dt,
            "Com_DubaiOil": base,
            "Com_CrudeOil": base + 1.2,
            "Com_BrentCrudeOil": base + 2.1,
        }
    )


def test_plot_star_decomposition_dubai_cutoff_writes_png(tmp_path: Path) -> None:
    setting_path = tmp_path / "setting.yaml"
    setting_path.write_text("training:\n  input_size: 20\n", encoding="utf-8")
    csv_path = tmp_path / "df.csv"
    output_dir = tmp_path / "run"
    make_dubai_frame(rows=30, end="2026-02-23").to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = run_script(
        csv_path,
        output_dir,
        "dubai",
        cutoff_date="2026-02-23",
        setting_path=setting_path,
    )

    assert result.returncode == 0, result.stderr
    out = output_dir / "dubai_star_decomposition.png"
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_star_decomposition_cutoff_requires_input_size(tmp_path: Path) -> None:
    setting_path = tmp_path / "setting.yaml"
    setting_path.write_text("runtime:\n  random_seed: 0\n", encoding="utf-8")
    csv_path = tmp_path / "df.csv"
    output_dir = tmp_path / "run"
    make_dubai_frame(rows=30, end="2026-02-23").to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = run_script(
        csv_path,
        output_dir,
        "dubai",
        cutoff_date="2026-02-23",
        setting_path=setting_path,
    )

    assert result.returncode != 0
    assert "input_size" in result.stderr


def test_plot_star_decomposition_can_add_gprd_threat_plot(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    output_dir = tmp_path / "run"
    make_weekly_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = run_script(csv_path, output_dir, "wti", "brent", "gprd_threat")

    assert result.returncode == 0, result.stderr
    png_names = sorted(path.name for path in output_dir.glob("*.png"))
    assert png_names == [
        "brent_star_decomposition.png",
        "gprd_threat_star_decomposition.png",
        "wti_star_decomposition.png",
    ]


def test_plot_star_decomposition_fails_fast_when_required_target_is_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    output_dir = tmp_path / "run"
    frame = make_weekly_frame().drop(columns=["Com_CrudeOil"])
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = run_script(csv_path, output_dir)

    assert result.returncode != 0
    assert "Required wti target column is missing" in result.stderr


def test_select_tail_anomaly_mask_uses_two_sided_threshold() -> None:
    raw_residual = np.array([1.0, 1.1, 0.9, 3.4, -1.2], dtype=float)

    mask, tail_score = select_tail_anomaly_mask(
        raw_residual,
        thresh=3.5,
        tail="two_sided",
    )

    assert mask.tolist() == [False, False, False, True, True]
    assert tail_score.shape == raw_residual.shape


def test_select_tail_anomaly_mask_uses_upward_threshold() -> None:
    raw_residual = np.array([1.0, 1.1, 0.9, 3.4, -1.2], dtype=float)

    mask, tail_score = select_tail_anomaly_mask(
        raw_residual,
        thresh=3.5,
        tail="upward",
    )

    assert mask.tolist() == [False, False, False, True, False]
    assert tail_score.shape == raw_residual.shape


def test_slice_window_ends_on_cutoff_calendar_day_with_exact_input_size() -> None:
    dt = pd.date_range("2026-02-01", periods=10, freq="D")
    frame = pd.DataFrame({"dt": dt, "Com_DubaiOil": np.arange(10, dtype=float)})
    win, marker = slice_input_window_ending_on_cutoff_calendar_day(
        frame,
        cutoff=pd.Timestamp("2026-02-08"),
        input_size=5,
    )
    assert len(win) == 5
    assert pd.Timestamp(win["dt"].iloc[-1]).normalize() == pd.Timestamp("2026-02-08").normalize()
    assert marker.normalize() == pd.Timestamp("2026-02-08").normalize()
    assert win["dt"].iloc[0] == pd.Timestamp("2026-02-04")


def test_plot_dubai_with_retrieval_combined_hist_exog_writes_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    dt = pd.date_range("2025-01-06", periods=40, freq="W-MON")
    dubai = 60.0 + np.cumsum(rng.normal(scale=0.3, size=len(dt)))
    gprd = 100.0 + 10.0 * np.sin(np.arange(len(dt)) / 5.0) + rng.normal(scale=1.0, size=len(dt))
    csv_path = tmp_path / "df.csv"
    pd.DataFrame({"dt": dt, "Com_DubaiOil": dubai, "GPRD_THREAT": gprd}).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )
    output_dir = tmp_path / "run"
    result = run_script(
        csv_path,
        output_dir,
        "dubai",
        hist_exog_cols=["GPRD_THREAT"],
    )
    assert result.returncode == 0, result.stderr
    out = output_dir / "dubai_star_decomposition.png"
    assert out.is_file()
    assert out.stat().st_size > 0


def test_baseline_retrieval_yaml_star_tails_for_columns() -> None:
    star = load_retrieval_star_config(REPO_ROOT / "yaml/plugins/retrieval/baseline_retrieval.yaml")
    assert tail_for_column("GPRD_THREAT", star) == "upward"
    assert tail_for_column("Com_CrudeOil", star) == "two_sided"
    assert tail_for_column("Com_DubaiOil", star) == "two_sided"


def test_select_tail_anomaly_mask_allows_zero_anomaly_window() -> None:
    raw_residual = np.array([0.95, 1.0, 1.05, 1.02, 0.98], dtype=float)

    mask, tail_score = select_tail_anomaly_mask(
        raw_residual,
        thresh=3.5,
        tail="two_sided",
    )

    assert not mask.any()
    assert tail_score.shape == raw_residual.shape
