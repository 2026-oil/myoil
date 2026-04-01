from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app_config import load_app_config
from plugin_contracts.stage_registry import get_active_stage_plugin, get_stage_plugin
import runtime_support.runner as runtime


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nec_plugin_is_registered_and_validate_jobs_accepts_nec(tmp_path: Path) -> None:
    assert get_stage_plugin("nec") is not None
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml",
    )
    capability_path = tmp_path / "capability_report.json"

    runtime._validate_jobs(loaded, loaded.config.jobs, capability_path)

    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    assert capability["NEC"]["name"] == "NEC"
    assert capability["NEC"]["validation_error"] is None
    assert capability["nec"]["enabled"] is True
    assert capability["nec"]["handled_jobs"][0]["plugin_owned"] is True
    assert capability["nec"]["shared_training_scaler"] == "robust"


def test_fit_and_predict_fold_dispatches_to_nec_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml",
    )
    source_df = pd.read_csv(REPO_ROOT / "tests/fixtures/nec_runtime_smoke.csv")
    calls: dict[str, object] = {}

    def _fake_predict(loaded_arg, job, train_df, future_df, *, run_root):
        calls["job"] = job.model
        calls["train_rows"] = len(train_df)
        calls["future_rows"] = len(future_df)
        return (
            pd.DataFrame(
                {
                    "unique_id": ["target"] * len(future_df),
                    "ds": pd.to_datetime(future_df["dt"]),
                    "NEC": [42.0] * len(future_df),
                }
            ),
            future_df["target"].reset_index(drop=True),
            pd.Timestamp(train_df["dt"].iloc[-1]),
            train_df,
            None,
        )

    monkeypatch.setattr("plugins.nec.plugin.predict_nec_fold", _fake_predict)

    predictions, actuals, train_end_ds, train_df, nf = runtime._fit_and_predict_fold(
        loaded,
        loaded.config.jobs[0],
        source_df=source_df,
        freq="W",
        train_idx=list(range(8)),
        test_idx=[8, 9],
    )

    assert calls == {"job": "NEC", "train_rows": 8, "future_rows": 2}
    assert predictions["NEC"].tolist() == [42.0, 42.0]
    assert actuals.tolist() == source_df.loc[[8, 9], "target"].tolist()
    assert str(train_end_ds) == "2020-02-19 00:00:00"
    assert train_df["target"].tolist() == source_df.loc[list(range(8)), "target"].tolist()
    assert nf is None


def test_nec_stage_plugin_is_active_after_config_load() -> None:
    loaded = load_app_config(REPO_ROOT, config_path=REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml")
    result = get_active_stage_plugin(loaded.config)

    assert result is not None
    plugin, _config = result
    assert plugin.config_key == "nec"
    assert plugin.owns_top_level_job("NEC") is True
