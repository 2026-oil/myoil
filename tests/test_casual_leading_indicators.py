from __future__ import annotations

import numpy as np
import pandas as pd

from casual.leading_indicators import (
    PipelineConfig,
    compute_lagged_correlations,
    compute_pairwise_granger,
    run_pipeline,
)


def make_synthetic_frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    lead = rng.normal(size=rows)
    aux = rng.normal(size=rows)
    target = np.zeros(rows)
    for idx in range(1, rows):
        target[idx] = (0.65 * target[idx - 1]) + (0.85 * lead[idx - 1]) + (0.05 * rng.normal())
    return pd.DataFrame(
        {
            "dt": pd.date_range("2020-01-01", periods=rows, freq="W"),
            "Lead_X": lead,
            "Noise_Z": aux,
            "Com_CrudeOil": target,
            "Com_BrentCrudeOil": target * 0.9 + (0.1 * rng.normal(size=rows)),
        }
    )


def test_lagged_correlations_identify_lead_signal() -> None:
    frame = make_synthetic_frame()
    predictors = ["Lead_X", "Noise_Z"]
    result = compute_lagged_correlations(
        frame=frame,
        target="Com_CrudeOil",
        predictors=predictors,
        max_lag=4,
        variant="raw",
    )
    assert not result.empty
    top = result.iloc[0]
    assert top["predictor"] == "Lead_X"
    assert int(top["best_lag"]) == 1


def test_pairwise_granger_identifies_lead_signal() -> None:
    frame = make_synthetic_frame(rows=320)
    predictors = ["Lead_X", "Noise_Z"]
    result, metadata = compute_pairwise_granger(
        frame=frame,
        target="Com_CrudeOil",
        predictors=predictors,
        max_lag=3,
    )
    assert "failures" in metadata
    assert not result.empty
    top = result.iloc[0]
    assert top["predictor"] == "Lead_X"
    assert int(top["best_lag"]) >= 1
    assert float(top["best_pvalue"]) < 0.05


def test_run_pipeline_writes_contract_artifacts(tmp_path) -> None:
    frame = make_synthetic_frame()
    csv_path = tmp_path / "df.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    output_root = tmp_path / "out"
    config = PipelineConfig(
        csv_path=csv_path,
        output_root=output_root,
        max_lag=3,
        top_k=5,
        methods=("lagged_correlation", "granger", "tigramite_pcmci"),
    )
    result = run_pipeline(config)
    assert result["output_root"] == output_root
    assert (output_root / "dataset_audit.json").exists()
    assert (output_root / "analysis_manifest.json").exists()
    assert (output_root / "method_status.csv").exists()
    assert (output_root / "summary.json").exists()
    assert (output_root / "report.md").exists()
