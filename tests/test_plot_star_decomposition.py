from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.plot_star_decomposition import select_tail_anomaly_mask

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
    csv_path: Path, output_dir: Path, *targets: str, thresh: float | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    cmd = [sys.executable, str(SCRIPT), "--input", str(csv_path), "--output-dir", str(output_dir)]
    if thresh is not None:
        cmd.extend(["--thresh", str(thresh)])
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


def test_plot_star_decomposition_can_add_gprd_threat_plot(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    output_dir = tmp_path / "run"
    make_weekly_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")

    result = run_script(csv_path, output_dir, "wti", "brent", "gprd_threat", thresh=3.5)

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


def test_select_tail_anomaly_mask_allows_zero_anomaly_window() -> None:
    raw_residual = np.array([0.95, 1.0, 1.05, 1.02, 0.98], dtype=float)

    mask, tail_score = select_tail_anomaly_mask(
        raw_residual,
        thresh=3.5,
        tail="two_sided",
    )

    assert not mask.any()
    assert tail_score.shape == raw_residual.shape
