from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from app_config import load_app_config
import runtime_support.runner as runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECT_MODEL_SELECTORS = {
    "ARIMA": ["order", "include_mean", "include_drift"],
    "ES": ["trend", "damped_trend"],
    "xgboost": ["lags", "n_estimators", "max_depth"],
    "lightgbm": [
        "lags",
        "n_estimators",
        "max_depth",
        "num_leaves",
        "min_child_samples",
        "feature_fraction",
    ],
}


def _payload() -> dict[str, Any]:
    return {
        "task": {"name": "top_level_direct_models"},
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": [],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "runtime": {"random_seed": 1},
        "training": {
            "input_size": 2,
            "batch_size": 8,
            "valid_batch_size": 8,
            "windows_batch_size": 32,
            "inference_windows_batch_size": 32,
            "scaler_type": None,
            "model_step_size": 1,
            "max_steps": 1,
            "val_size": 1,
            "val_check_steps": 1,
            "early_stop_patience_steps": -1,
            "loss": "mse",
            "optimizer": {"name": "adamw", "kwargs": {}},
        },
        "cv": {"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0},
        "scheduler": {
            "gpu_ids": [0],
            "max_concurrent_jobs": 1,
            "worker_devices": 1,
            "parallelize_single_job_tuning": False,
        },
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "jobs": [],
    }


def _write_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_search_space(
    root: Path,
    *,
    models: dict[str, list[str]],
    training: list[str] | None = None,
) -> Path:
    payload = {
        "models": models,
        "training": training or [],
        "residual": {"xgboost": ["n_estimators", "max_depth"]},
    }
    path = root / "yaml/HPO/search_space.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_data(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target\n"
        "2020-01-01,1\n"
        "2020-01-08,2\n"
        "2020-01-15,3\n"
        "2020-01-22,4\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("model_name", "expected_selectors"),
    DIRECT_MODEL_SELECTORS.items(),
    ids=DIRECT_MODEL_SELECTORS.keys(),
)
def test_top_level_direct_models_enter_model_auto_without_training_search(
    tmp_path: Path,
    model_name: str,
    expected_selectors: list[str],
) -> None:
    payload = _payload()
    payload["jobs"] = [{"model": model_name, "params": {}}]
    _write_data(tmp_path)
    config_path = _write_config(tmp_path, payload)
    _write_search_space(tmp_path, models={model_name: expected_selectors}, training=[])

    loaded = load_app_config(tmp_path, config_path=config_path)

    job = loaded.config.jobs[0]
    assert job.requested_mode == "learned_auto_requested"
    assert job.validated_mode == "learned_auto"
    assert list(job.selected_search_params) == expected_selectors
    assert loaded.config.training_search.requested_mode == "training_fixed"
    assert loaded.config.training_search.validated_mode == "training_fixed"
    assert list(loaded.config.training_search.selected_search_params) == []


def test_validate_only_repro_config_accepts_top_level_direct_models(tmp_path: Path) -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / "yaml/experiment/jaeho_bs_forecast_layer1/bs_forecast_uni.yaml",
    )
    capability_path = tmp_path / "capability_report.json"

    runtime._validate_jobs(loaded, loaded.config.jobs, capability_path)

    capability = yaml.safe_load(capability_path.read_text(encoding="utf-8"))
    assert [job.model for job in loaded.config.jobs][:8] == [
        "ARIMA",
        "DLinear",
        "ES",
        "lightgbm",
        "LSTM",
        "NHITS",
        "PatchTST",
        "xgboost",
    ]
    for model_name in ("ARIMA", "ES", "xgboost", "lightgbm"):
        assert capability[model_name]["requested_mode"] == "learned_fixed"
        assert capability[model_name]["validated_mode"] == "learned_fixed"
        assert capability[model_name]["validation_error"] is None
        assert capability[model_name]["supports_auto"] is True
    assert capability["residual"]["validated_mode"] == "residual_disabled"


def test_fit_and_predict_fold_uses_direct_runtime_lane(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = _payload()
    payload["jobs"] = [{"model": "ARIMA", "params": {"order": [1, 1, 0]}}]
    _write_data(tmp_path)
    config_path = _write_config(tmp_path, payload)

    loaded = load_app_config(tmp_path, config_path=config_path)
    source_df = pd.read_csv(tmp_path / "data.csv")

    def _fake_predict(stage_loaded, job, *, target_column, train_df, future_df):
        assert stage_loaded.config.dataset.target_col == "target"
        assert job.model == "ARIMA"
        assert list(train_df["target"]) == [1, 2, 3]
        return [42.0 for _ in range(len(future_df))]

    class _FailNeuralForecast:
        def __init__(self, *args, **kwargs):
            raise AssertionError("direct runtime lane should not instantiate NeuralForecast")

    monkeypatch.setattr(runtime, "predict_univariate_direct", _fake_predict)
    monkeypatch.setattr(runtime, "NeuralForecast", _FailNeuralForecast)

    predictions, actuals, train_end_ds, train_df, nf = runtime._fit_and_predict_fold(
        loaded,
        loaded.config.jobs[0],
        source_df=source_df,
        freq="W",
        train_idx=[0, 1, 2],
        test_idx=[3],
    )

    assert predictions["ARIMA"].tolist() == [42.0]
    assert actuals.tolist() == [4.0]
    assert str(train_end_ds) == "2020-01-15 00:00:00"
    assert train_df["target"].tolist() == [1, 2, 3]
    assert nf is None


def test_validate_only_rejects_residual_enabled_top_level_direct_models(tmp_path: Path) -> None:
    payload = _payload()
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2},
    }
    payload["jobs"] = [{"model": "ARIMA", "params": {"order": [1, 1, 0]}}]
    _write_data(tmp_path)
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(
        ValueError,
        match="Top-level direct models do not yet support residual-enabled runs",
    ):
        runtime.main(["--validate-only", "--config", str(config_path)])
