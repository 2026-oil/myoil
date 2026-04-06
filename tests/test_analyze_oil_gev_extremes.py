from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze_oil_gev_extremes.py"


def make_weekly_oil_frame(rows: int = 208) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dt = pd.date_range("2018-01-01", periods=rows, freq="W-MON")
    wti_log_returns = rng.normal(loc=0.001, scale=0.035, size=rows)
    brent_log_returns = 0.7 * wti_log_returns + rng.normal(loc=0.0005, scale=0.02, size=rows)
    for idx, shock in {24: 0.20, 63: -0.18, 95: 0.24, 130: -0.21, 168: 0.18}.items():
        if idx < rows:
            wti_log_returns[idx] += shock
    for idx, shock in {28: 0.16, 70: -0.14, 100: 0.22, 145: -0.19, 180: 0.17}.items():
        if idx < rows:
            brent_log_returns[idx] += shock
    wti = 60 * np.exp(np.cumsum(wti_log_returns))
    brent = 65 * np.exp(np.cumsum(brent_log_returns))
    return pd.DataFrame(
        {
            "dt": dt,
            "Com_CrudeOil": wti,
            "Com_BrentCrudeOil": brent,
            "aux_signal": rng.normal(size=rows),
        }
    )


def run_analyze(csv_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ANALYZE_SCRIPT), "--input", str(csv_path), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_smoke_writes_required_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    required = [
        output_dir / "contract" / "analysis_manifest.json",
        output_dir / "contract" / "dataset_audit.json",
        output_dir / "tables" / "com_crudeoil_upper_tail.csv",
        output_dir / "tables" / "com_crudeoil_lower_tail.csv",
        output_dir / "tables" / "com_brentcrudeoil_upper_tail.csv",
        output_dir / "tables" / "com_brentcrudeoil_lower_tail.csv",
        output_dir / "summary.json",
        output_dir / "report.md",
    ]
    for path in required:
        assert path.exists(), path
    stdout_payload = json.loads(result.stdout)
    assert stdout_payload["summary_path"].endswith("summary.json")
    assert stdout_payload["report_path"].endswith("report.md")


def test_manifest_records_targets_tails_representation_and_block_policy(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "contract" / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_columns"] == ["Com_CrudeOil", "Com_BrentCrudeOil"]
    assert manifest["tail_directions"] == ["upper", "lower"]
    assert manifest["block_policy"]["block_size_weeks"] == 8
    assert "lower_tail_sign_convention" in manifest
    for target in manifest["target_columns"]:
        assert manifest["targets"][target]["chosen_representation"]["name"] in {"level", "diff1", "log_return"}
        assert manifest["targets"][target]["representation_reason"]


def test_report_keeps_recommendation_only_boundary(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "## Representation rationale" in report
    assert "## 8-week forecasting strategy recommendations" in report
    assert "recommendation-only" in report
    assert "It does not implement or finalize feature, loss, target, or model configuration changes." in report


def test_tail_tables_are_non_empty_with_planted_extremes(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    for name in [
        "com_crudeoil_upper_tail.csv",
        "com_crudeoil_lower_tail.csv",
        "com_brentcrudeoil_upper_tail.csv",
        "com_brentcrudeoil_lower_tail.csv",
    ]:
        frame = pd.read_csv(output_dir / "tables" / name)
        assert not frame.empty
        assert {"target", "tail", "block_id", "dt_start", "dt_end", "original_value", "transformed_value", "sign_convention"}.issubset(frame.columns)


def test_small_sample_reports_insufficient_evidence_without_crashing(tmp_path: Path) -> None:
    csv_path = tmp_path / "df_short.csv"
    make_weekly_oil_frame(rows=48).to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run_short"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    statuses = {
        summary["targets"][target]["tails"][tail]["status"]
        for target in ["Com_CrudeOil", "Com_BrentCrudeOil"]
        for tail in ["upper", "lower"]
    }
    assert statuses == {"insufficient_evidence"}
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Insufficient tail evidence" in report


def test_missing_target_fails_fast_with_explicit_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "df_missing.csv"
    frame = make_weekly_oil_frame().drop(columns=["Com_BrentCrudeOil"])
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run_missing"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode != 0
    combined = result.stdout + "\n" + result.stderr
    assert "Com_BrentCrudeOil" in combined
    assert "Missing required target columns" in combined
