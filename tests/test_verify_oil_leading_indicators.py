from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.test_analyze_oil_leading_indicators import make_weekly_oil_frame, run_analyze

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_oil_leading_indicators.py"


def run_verify(run_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {**dict(), **__import__("os").environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    return subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--run-dir", str(run_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_verify_accepts_complete_run_dir(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    analyze = run_analyze(csv_path, output_dir)
    assert analyze.returncode == 0, analyze.stderr
    verify = run_verify(output_dir)
    assert verify.returncode == 0, verify.stderr
    assert '"status": "ok"' in verify.stdout


def test_verify_rejects_missing_required_family_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    analyze = run_analyze(csv_path, output_dir)
    assert analyze.returncode == 0, analyze.stderr
    (output_dir / "tables" / "family_ccf.csv").unlink()
    verify = run_verify(output_dir)
    assert verify.returncode != 0
    assert "missing required family table" in verify.stderr


def test_verify_rejects_missing_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    analyze = run_analyze(csv_path, output_dir)
    assert analyze.returncode == 0, analyze.stderr
    predictive_path = output_dir / "tables" / "family_predictive.csv"
    predictive = pd.read_csv(predictive_path)
    predictive = predictive.drop(columns=["forward_fdr_pvalue"])
    predictive.to_csv(predictive_path, index=False)
    verify = run_verify(output_dir)
    assert verify.returncode != 0
    assert "missing required columns" in verify.stderr
