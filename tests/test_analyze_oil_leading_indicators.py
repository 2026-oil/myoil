from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.oil_leading_indicators_lib import build_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze_oil_leading_indicators.py"


def make_weekly_oil_frame(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dt = pd.date_range("2018-01-01", periods=rows, freq="W-MON")
    base = np.zeros(rows)
    lead_driver = rng.normal(size=rows)
    coincident = rng.normal(size=rows)
    lagging = np.zeros(rows)
    for idx in range(1, rows):
        base[idx] = 0.65 * base[idx - 1] + 0.70 * lead_driver[idx - 1] + 0.05 * rng.normal()
        lagging[idx] = base[idx - 1] + 0.05 * rng.normal()
    brent = 72 + 4.5 * base + 0.2 * coincident + rng.normal(scale=0.2, size=rows)
    crude = 68 + 4.0 * base + 0.15 * coincident + rng.normal(scale=0.2, size=rows)
    return pd.DataFrame(
        {
            "dt": dt,
            "Com_CrudeOil": crude,
            "Com_BrentCrudeOil": brent,
            "lead_signal": np.roll(base, -2) + rng.normal(scale=0.2, size=rows),
            "coincident_signal": base + rng.normal(scale=0.1, size=rows),
            "lag_signal": lagging,
            "noise_signal": rng.normal(size=rows),
            "constant_signal": np.full(rows, 5.0),
        }
    )


def run_analyze(csv_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    env = {**dict(), **__import__("os").environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    return subprocess.run(
        [sys.executable, str(ANALYZE_SCRIPT), "--input", str(csv_path), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_builds_manifest_with_locked_primary_target_and_grids(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    manifest = build_manifest(csv_path, {"predictor_count": 4})
    assert manifest["primary_target"]["kind"] == "zscore_average"
    assert manifest["sensitivity_target"]["kind"] == "pca1_appendix"
    assert manifest["screen_lags"] == list(range(1, 9))
    assert manifest["signed_support_lags"] == list(range(-8, 9))
    assert manifest["reporting_buckets"]["short_weeks"] == [1, 2, 3, 4]
    assert manifest["horizons"] == [1, 2, 4, 8]
    assert manifest["predictive"]["fdr_universe"].startswith("separate_across_all_variables")
    assert manifest["blocked_policy"]["silent_substitution_forbidden"] is True


def test_candidate_universe_excludes_dt_and_oil_components(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    frame = make_weekly_oil_frame()
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    audit = json.loads((output_dir / "contract" / "dataset_audit.json").read_text(encoding="utf-8"))
    assert audit["excluded_columns"] == ["dt", "Com_CrudeOil", "Com_BrentCrudeOil"]
    manifest = json.loads((output_dir / "contract" / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_universe"]["exclude"] == ["dt", "Com_CrudeOil", "Com_BrentCrudeOil"]


def test_analysis_run_writes_required_contract_and_normalized_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    required = [
        output_dir / "contract" / "analysis_manifest.json",
        output_dir / "contract" / "dataset_audit.json",
        output_dir / "tables" / "family_turning_point.csv",
        output_dir / "tables" / "family_ccf.csv",
        output_dir / "tables" / "family_predictive.csv",
        output_dir / "tables" / "family_dfm.csv",
        output_dir / "tables" / "family_oos_dm.csv",
        output_dir / "tables" / "family_frequency.csv",
        output_dir / "tables" / "synthesis.csv",
        output_dir / "summary.json",
        output_dir / "report.md",
    ]
    for path in required:
        assert path.exists(), path


def test_normalized_family_tables_expose_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "df.csv"
    make_weekly_oil_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "run"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    predictive = pd.read_csv(output_dir / "tables" / "family_predictive.csv")
    dfm = pd.read_csv(output_dir / "tables" / "family_dfm.csv")
    synthesis = pd.read_csv(output_dir / "tables" / "synthesis.csv")
    assert {"forward_fdr_pvalue", "reverse_fdr_pvalue", "directional_effect", "generic_pvalue_rule"}.issubset(predictive.columns)
    assert {"factor_id", "factor_count", "abs_loading"}.issubset(dfm.columns)
    assert dfm["eligible"].astype(bool).any()
    assert {"variable", "final_class", "final_reason_code", "leading_votes"}.issubset(synthesis.columns)
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "contradictory cases" in report
    assert "## Blocked Family Rationale" in report


def test_blocked_family_emits_ineligible_row_with_reason_when_gate_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "df_short.csv"
    make_weekly_oil_frame(rows=120).to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_dir = tmp_path / "short_run"
    result = run_analyze(csv_path, output_dir)
    assert result.returncode == 0, result.stderr
    predictive = pd.read_csv(output_dir / "tables" / "family_predictive.csv")
    oos = pd.read_csv(output_dir / "tables" / "family_oos_dm.csv")
    assert ((~predictive["eligible"].astype(bool)) & (predictive["blocked_reason"].fillna("") != "")).any()
    assert ((~oos["eligible"].astype(bool)) & (oos["blocked_reason"].fillna("") != "")).any()
