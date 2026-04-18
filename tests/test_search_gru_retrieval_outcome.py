from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from runtime_support.outcome_search import (
    audit_sort_key,
    evaluate_run_outcome,
    objective_value,
    winner_sort_key,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "search_gru_retrieval_outcome.py"
    spec = importlib.util.spec_from_file_location("_search_gru_retrieval_outcome_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run_fixture(run_root: Path, *, mismatch_cutoff: str | None = None) -> None:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    retrieval_root = run_root / "aa_forecast" / "retrieval"
    rows: list[dict[str, object]] = []
    base_h1 = 90.0
    for fold_idx in range(12):
        cutoff = pd.Timestamp("2025-12-08") + pd.Timedelta(days=7 * fold_idx)
        cutoff_tag = cutoff.strftime("%Y%m%dT%H%M%S")
        fold_dir = retrieval_root / f"fold_{fold_idx:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        actual = [100.0 + fold_idx, 110.0 + fold_idx]
        base = [base_h1 + fold_idx, base_h1 + 5.0 + fold_idx]
        final = [base[0] + 1.0, base[1] + 2.0]
        if str(cutoff) == "2026-02-23 00:00:00":
            final = [base[0] + 5.0, base[1] + 7.0]
        payload = {
            "cutoff": str(cutoff),
            "base_prediction": base if str(cutoff) != mismatch_cutoff else [base[0]],
            "final_prediction": final,
            "retrieval_applied": True,
        }
        payload_path = fold_dir / f"{cutoff_tag}.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        artifact_rel = payload_path.relative_to(run_root)
        for horizon_step, ds, y, y_hat in (
            (1, cutoff + pd.Timedelta(days=7), actual[0], final[0]),
            (2, cutoff + pd.Timedelta(days=14), actual[1], final[1]),
        ):
            rows.append(
                {
                    "model": "AAForecast",
                    "fold_idx": fold_idx,
                    "cutoff": str(cutoff),
                    "ds": str(ds),
                    "horizon_step": horizon_step,
                    "y": y,
                    "y_hat": y_hat,
                    "aaforecast_retrieval_artifact": str(artifact_rel),
                }
            )
    pd.DataFrame(rows).to_csv(summary_dir / "result.csv", index=False)


def _write_search_fixture(tmp_path: Path, *, model: str = "gru") -> tuple[Path, Path]:
    dataset_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "dt": pd.date_range("2025-09-01", periods=40, freq="7D"),
            "Com_CrudeOil": [70.0 + idx for idx in range(40)],
            "GPRD_THREAT": [1.0 + idx for idx in range(40)],
            "GPRD": [2.0 + idx for idx in range(40)],
            "GPRD_ACT": [3.0 + idx for idx in range(40)],
            "Idx_OVX": [4.0 + idx for idx in range(40)],
            "Com_LMEX": [5.0 + idx for idx in range(40)],
            "Com_BloombergCommodity_BCOM": [6.0 + idx for idx in range(40)],
        }
    ).to_csv(dataset_path, index=False)
    retrieval_path = tmp_path / "baseline_retrieval.yaml"
    retrieval_path.write_text(
        "\n".join(
            [
                "retrieval:",
                "  enabled: true",
                "  mode: posthoc_blend",
                "  top_k: 1",
                "  recency_gap_steps: 16",
                "  trigger_quantile: 0.0126",
                "  min_similarity: 0.362",
                "  similarity: cosine",
                "  temperature: 0.0105",
                "  blend_floor: 0.25",
                "  blend_max: 0.7",
                "  use_uncertainty_gate: false",
                "  insample_y_included: false",
                "  use_event_key: true",
                "  event_score_log_bonus_alpha: 0.522",
                "  event_score_log_bonus_cap: 1.87",
                "",
            ]
        ),
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_forecast_gru-ret.yaml"
    plugin_path.write_text(
        "\n".join(
            [
                "aa_forecast:",
                f"  model: {model}",
                "  lowess_frac: 0.359",
                "  lowess_delta: 0.008",
                "  thresh: 3.08",
                "  retrieval:",
                "    enabled: true",
                f"    config_path: {retrieval_path.name}",
                "  star_anomaly_tails:",
                "    upward:",
                "    - GPRD_THREAT",
                "    - GPRD",
                "    two_sided: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task:",
                "  name: outcome-search-smoke",
                "dataset:",
                f"  path: {dataset_path}",
                "  target_col: Com_CrudeOil",
                "  dt_col: dt",
                "  hist_exog_cols:",
                "  - GPRD_THREAT",
                "  - GPRD",
                "  - GPRD_ACT",
                "  - Idx_OVX",
                "  - Com_LMEX",
                "  - Com_BloombergCommodity_BCOM",
                "aa_forecast:",
                "  enabled: true",
                f"  config_path: {plugin_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    setting_path = tmp_path / "setting.yaml"
    setting_path.write_text(
        "\n".join(
            [
                "training:",
                "  input_size: 64",
                "cv:",
                "  horizon: 2",
                "  n_windows: 16",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path, setting_path


def test_evaluate_run_outcome_recent12_and_gate(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_run_fixture(run_root)
    outcome = evaluate_run_outcome(
        run_root=run_root,
        spike_cutoff=pd.Timestamp("2026-02-23"),
        recent_fold_count=12,
        horizon=2,
    )
    assert outcome.pass_gate is True
    assert outcome.h1_uplift > 0
    assert outcome.h2_uplift > 0
    assert outcome.improved_fold_count == 12
    assert outcome.worst_fold_regression < 0
    assert len(outcome.recent_cutoffs) == 12


def test_evaluate_run_outcome_fails_on_base_prediction_length_mismatch(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_run_fixture(run_root, mismatch_cutoff="2026-02-23 00:00:00")
    with pytest.raises(ValueError, match="base_prediction length mismatch"):
        evaluate_run_outcome(
            run_root=run_root,
            spike_cutoff=pd.Timestamp("2026-02-23"),
            recent_fold_count=12,
            horizon=2,
        )


def test_outcome_search_sort_keys_and_objective_priority() -> None:
    passing = {
        "trial_number": 2,
        "pass_gate": True,
        "min_uplift": 3.0,
        "sum_uplift": 7.0,
        "recent12_mean_mape_on": 0.2,
        "recent12_delta_mape": -0.05,
        "improved_fold_count": 10,
    }
    failing = {
        "trial_number": 1,
        "pass_gate": False,
        "min_uplift": 100.0,
        "sum_uplift": 200.0,
        "recent12_mean_mape_on": 0.01,
        "recent12_delta_mape": -1.0,
        "improved_fold_count": 12,
    }
    assert audit_sort_key(passing) < audit_sort_key(failing)
    assert objective_value(pass_gate=True, min_uplift=1.0, sum_uplift=1.0, recent_mean_mape_on=10.0) > objective_value(
        pass_gate=False,
        min_uplift=999.0,
        sum_uplift=999.0,
        recent_mean_mape_on=0.0,
    )
    better = dict(passing, trial_number=3, min_uplift=4.0)
    assert winner_sort_key(better) < winner_sort_key(passing)


def test_search_cli_rejects_non_gru_config(tmp_path: Path) -> None:
    search = _load_script()
    config_path, setting_path = _write_search_fixture(tmp_path, model="timexer")
    with pytest.raises(ValueError, match="requires GRU"):
        search.main(
            [
                "--config",
                str(config_path),
                "--setting",
                str(setting_path),
                "--output-root",
                str(tmp_path / "out"),
                "--n-trials",
                "1",
            ]
        )


def test_search_cli_resume_hash_mismatch_on_dataset_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    search = _load_script()
    config_path, setting_path = _write_search_fixture(tmp_path)

    def _fake_run(*, output_root: Path, **_: object) -> int:
        _write_run_fixture(output_root)
        return 0

    monkeypatch.setattr(search, "_run_experiment", _fake_run)
    out_root = tmp_path / "bundle"
    rc = search.main(
        [
            "--config",
            str(config_path),
            "--setting",
            str(setting_path),
            "--output-root",
            str(out_root),
            "--n-trials",
            "1",
            "--study-name",
            "resume-test",
        ]
    )
    assert rc == 0
    dataset_path = Path(json.loads((out_root / "search_metadata.json").read_text(encoding="utf-8"))["contract"]["resolved_dataset_path"])
    dataset_frame = pd.read_csv(dataset_path)
    dataset_frame.loc[0, "Com_CrudeOil"] += 123.0
    dataset_frame.to_csv(dataset_path, index=False)
    with pytest.raises(ValueError, match="compatibility hash mismatch"):
        search.main(
            [
                "--config",
                str(config_path),
                "--setting",
                str(setting_path),
                "--output-root",
                str(out_root),
                "--n-trials",
                "1",
                "--study-name",
                "resume-test",
                "--resume",
            ]
        )
