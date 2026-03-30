from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from app_config import load_app_config
from neuralforecast.models.bs_preforcast_direct import (
    _extract_tree_history_frame,
    _resolve_tree_history_metric,
    _sanitize_structural_warmup_exog_rows,
    predict_univariate_tree,
)
from runtime_support.forecast_models import validate_job
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
    assert [job.model for job in loaded.config.jobs] == [
        "ARIMA",
        "DLinear",
        "ES",
        "lightgbm",
        "xgboost",
    ]
    for model_name in ("ARIMA", "ES", "xgboost", "lightgbm"):
        assert capability[model_name]["requested_mode"] == "learned_auto_requested"
        assert capability[model_name]["validated_mode"] == "learned_auto"
        assert capability[model_name]["validation_error"] is None
        assert capability[model_name]["supports_auto"] is True
    assert capability["ARIMA"]["supports_hist_exog"] is False
    assert capability["ES"]["supports_hist_exog"] is False
    assert capability["xgboost"]["supports_hist_exog"] is True
    assert capability["lightgbm"]["supports_hist_exog"] is True
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


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
def test_direct_tree_models_advertise_hist_exog_support(
    tmp_path: Path,
    model_name: str,
) -> None:
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = ["hist_a"]
    payload["jobs"] = [{"model": model_name, "params": {"lags": [1, 2]}}]
    _write_data(tmp_path)
    config_path = _write_config(tmp_path, payload)

    loaded = load_app_config(tmp_path, config_path=config_path)

    capabilities = validate_job(loaded.config.jobs[0])
    assert capabilities.supports_hist_exog is True
    assert capabilities.supports_futr_exog is False
    assert capabilities.supports_stat_exog is False


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
def test_predict_univariate_tree_uses_hist_exog_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
) -> None:
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = ["hist_a"]
    payload["jobs"] = [
        {
            "model": model_name,
            "params": {
                "lags": [1, 2],
                "n_estimators": 8,
                "max_depth": 2,
            },
        }
    ]
    _write_data(tmp_path)
    config_path = _write_config(tmp_path, payload)
    loaded = load_app_config(tmp_path, config_path=config_path)
    job = loaded.config.jobs[0]

    captured: dict[str, pd.DataFrame | list[float] | int | list[int] | None] = {}

    class _FakeForecasterDirect:
        def __init__(self, *, estimator, steps, lags):
            del estimator
            self.steps = list(range(1, steps + 1))
            captured["lags"] = lags
            captured["constructor_param"] = "estimator"

        def fit(self, y, exog=None, suppress_warnings=True):
            del suppress_warnings
            captured["fit_y"] = list(y)
            captured["fit_exog"] = None if exog is None else exog.copy()
            return self

        def predict(self, *, steps, exog=None):
            captured["predict_steps"] = list(steps)
            captured["predict_exog"] = None if exog is None else exog.copy()
            return pd.Series([101.0 + step for step in range(len(steps))])

    monkeypatch.setattr(
        "skforecast.direct.ForecasterDirect",
        _FakeForecasterDirect,
    )
    monkeypatch.setattr(
        "neuralforecast.models.bs_preforcast_direct._build_tree_curve_frame",
        lambda *args, **kwargs: pd.DataFrame(
            columns=["global_step", "train_loss", "val_loss"]
        ),
    )

    train_df = pd.DataFrame(
        {
            "dt": pd.date_range("2024-01-01", periods=4, freq="W"),
            "target": [1.0, 2.0, 3.0, 4.0],
            "hist_a": [10.0, 20.0, 30.0, 40.0],
        }
    )
    future_df = pd.DataFrame(
        {
            "dt": pd.date_range("2024-02-04", periods=2, freq="W"),
            "target": [5.0, 6.0],
            "hist_a": [50.0, 60.0],
        }
    )

    result = predict_univariate_tree(
        loaded,
        job,
        target_column="target",
        train_df=train_df,
        future_df=future_df,
        model_name=model_name,
    )

    assert result.predictions == [101.0, 102.0]
    fit_exog = captured["fit_exog"]
    predict_exog = captured["predict_exog"]
    assert captured["constructor_param"] == "estimator"
    assert isinstance(fit_exog, pd.DataFrame)
    assert isinstance(predict_exog, pd.DataFrame)
    assert fit_exog.columns.tolist() == ["hist_a_lag_1", "hist_a_lag_2"]
    assert predict_exog.columns.tolist() == ["hist_a_lag_1", "hist_a_lag_2"]
    assert not fit_exog.isna().any().any()
    assert fit_exog.to_dict(orient="records") == [
        {"hist_a_lag_1": 10.0, "hist_a_lag_2": 10.0},
        {"hist_a_lag_1": 10.0, "hist_a_lag_2": 10.0},
        {"hist_a_lag_1": 20.0, "hist_a_lag_2": 10.0},
        {"hist_a_lag_1": 30.0, "hist_a_lag_2": 20.0},
    ]
    assert predict_exog.to_dict(orient="records") == [
        {"hist_a_lag_1": 40.0, "hist_a_lag_2": 30.0},
        {"hist_a_lag_1": 50.0, "hist_a_lag_2": 40.0},
    ]


def test_tree_warmup_sanitization_preserves_effective_supervised_matrix() -> None:
    from skforecast.direct import ForecasterDirect
    from skforecast.exceptions import MissingValuesWarning
    from sklearn.linear_model import LinearRegression

    series = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=float)
    raw_exog = pd.DataFrame(
        {
            "hist_a_lag_1": [float("nan"), 10.0, 20.0, 30.0],
            "hist_a_lag_2": [float("nan"), float("nan"), 10.0, 20.0],
        }
    )
    sanitized_exog = _sanitize_structural_warmup_exog_rows(raw_exog, max_lag=2)
    assert sanitized_exog is not None
    assert not sanitized_exog.isna().any().any()
    forecaster = ForecasterDirect(
        estimator=LinearRegression(),
        steps=1,
        lags=[1, 2],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=MissingValuesWarning)
        raw_X_train, raw_y_train = forecaster.create_train_X_y(y=series, exog=raw_exog)
    sanitized_X_train, sanitized_y_train = forecaster.create_train_X_y(
        y=series,
        exog=sanitized_exog,
    )
    pd.testing.assert_frame_equal(raw_X_train, sanitized_X_train)
    assert raw_y_train.keys() == sanitized_y_train.keys()
    for step_key in raw_y_train:
        pd.testing.assert_series_equal(raw_y_train[step_key], sanitized_y_train[step_key])


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


@pytest.mark.parametrize(
    ("model_name", "training_loss", "expected_metric", "expected_postprocess"),
    [
        ("xgboost", "mse", "rmse", 4.0),
        ("lightgbm", "mse", "l2", 2.0),
        ("xgboost", "mae", "mae", 2.0),
        ("lightgbm", "mae", "l1", 2.0),
    ],
)
def test_tree_history_metric_supports_shared_mse_and_mae_settings(
    model_name: str,
    training_loss: str,
    expected_metric: str,
    expected_postprocess: float,
) -> None:
    metric_name, postprocess = _resolve_tree_history_metric(model_name, training_loss)

    assert metric_name == expected_metric
    assert postprocess(2.0) == expected_postprocess


def test_tree_history_metric_rejects_unsupported_losses() -> None:
    with pytest.raises(
        ValueError,
        match=r"support only training\.loss in \{mse, mae\}",
    ):
        _resolve_tree_history_metric("xgboost", "exloss")


def test_extract_tree_history_frame_reads_xgboost_validation_history() -> None:
    class _Estimator:
        def evals_result(self):
            return {
                "validation_0": {"rmse": [2.0, 1.0]},
                "validation_1": {"rmse": [3.0, 2.0]},
            }

    frame = _extract_tree_history_frame(
        _Estimator(),
        model_name="xgboost",
        training_loss="mse",
    )

    assert frame.to_dict(orient="records") == [
        {"global_step": 1, "train_loss": 4.0, "val_loss": 9.0},
        {"global_step": 2, "train_loss": 1.0, "val_loss": 4.0},
    ]


def test_extract_tree_history_frame_reads_lightgbm_validation_history() -> None:
    class _Estimator:
        evals_result_ = {
            "training": {"l1": [2.5, 1.5]},
            "validation": {"l1": [3.5, 2.5]},
        }

    frame = _extract_tree_history_frame(
        _Estimator(),
        model_name="lightgbm",
        training_loss="mae",
    )

    assert frame.to_dict(orient="records") == [
        {"global_step": 1, "train_loss": 2.5, "val_loss": 3.5},
        {"global_step": 2, "train_loss": 1.5, "val_loss": 2.5},
    ]
