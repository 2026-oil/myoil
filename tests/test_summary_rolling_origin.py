from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import runtime_support.runner as runtime


@pytest.fixture
def rolling_origin_source_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-07", periods=8, freq="W-SUN")
    return pd.DataFrame(
        {
            "dt": dates,
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "event": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )


@pytest.fixture
def rolling_origin_loaded() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            dataset=SimpleNamespace(
                dt_col="dt",
                target_col="target",
                hist_exog_cols=["event"],
                futr_exog_cols=[],
                static_exog_cols=[],
            ),
            cv=SimpleNamespace(
                horizon=2,
                step_size=1,
                n_windows=2,
                gap=0,
                max_train_size=None,
                overlap_eval_policy="by_cutoff_mean",
            ),
        )
    )


def test_build_rolling_origin_windows_uses_post_anchor_observed_origins(
    rolling_origin_source_df: pd.DataFrame,
    rolling_origin_loaded: SimpleNamespace,
) -> None:
    windows = runtime._build_rolling_origin_windows(
        rolling_origin_source_df,
        rolling_origin_loaded,
        freq="W-SUN",
    )

    assert [str(window.anchor_train_end_ds.date()) for window in windows] == [
        "2024-02-11",
        "2024-02-11",
    ]
    assert [str(window.origin_ds.date()) for window in windows] == [
        "2024-02-18",
        "2024-02-25",
    ]
    assert [str(window.forecast_start_ds.date()) for window in windows] == [
        "2024-02-25",
        "2024-03-03",
    ]
    assert [window.observed_steps for window in windows] == [1, 0]
    assert [window.forecast_only for window in windows] == [False, True]


def test_write_per_window_summary_bundles_replays_best_params_and_writes_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rolling_origin_source_df: pd.DataFrame,
) -> None:
    dataset_path = tmp_path / "rolling_origin.csv"
    rolling_origin_source_df.to_csv(dataset_path, index=False)

    config_payload = {
        "task": {"name": "rolling_origin_summary_test"},
        "dataset": {
            "path": str(dataset_path),
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["event"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "runtime": {"random_seed": 7},
        "training": {
            "input_size": 4,
            "batch_size": 1,
            "valid_batch_size": 1,
            "windows_batch_size": 8,
            "inference_windows_batch_size": 8,
            "max_steps": 1,
            "val_size": 2,
            "val_check_steps": 1,
            "early_stop_patience_steps": -1,
            "loss": "mse",
            "accelerator": "cpu",
        },
        "cv": {
            "horizon": 2,
            "step_size": 1,
            "n_windows": 2,
            "gap": 0,
            "overlap_eval_policy": "by_cutoff_mean",
        },
        "scheduler": {"gpu_ids": [0], "max_concurrent_jobs": 1, "worker_devices": 1},
        "aa_forecast": {
            "enabled": True,
            "config_path": str(
                Path("tests/fixtures/aa_forecast_runtime_plugin_auto_model_only.yaml").resolve()
            ),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    run_root = tmp_path / "run"
    (run_root / "manifest").mkdir(parents=True)
    (run_root / "models" / "AAForecast").mkdir(parents=True)
    (run_root / "summary").mkdir(parents=True)
    (run_root / "config").mkdir(parents=True)

    (run_root / "manifest" / "run_manifest.json").write_text(
        json.dumps(
            {
                "config_source_type": "yaml",
                "config_source_path": str(config_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "models" / "AAForecast" / "best_params.json").write_text(
        json.dumps({"best": "param"}, indent=2),
        encoding="utf-8",
    )
    (run_root / "models" / "AAForecast" / "training_best_params.json").write_text(
        json.dumps({"lr": 0.01}, indent=2),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
        run_root=None,
    ):
        del loaded, freq, params_override, run_root
        train_df = source_df.iloc[train_idx].reset_index(drop=True)
        future_df = source_df.iloc[test_idx].reset_index(drop=True)
        calls.append(
            {
                "job_params": dict(job.params),
                "training_override": dict(training_override or {}),
                "train_end_ds": str(pd.Timestamp(train_df["dt"].iloc[-1]).date()),
            }
        )
        predictions = pd.DataFrame(
            {
                "unique_id": ["target"] * len(future_df),
                "ds": pd.to_datetime(future_df["dt"]),
                job.model: [100.0 + idx for idx in range(len(future_df))],
            }
        )
        actuals = future_df["target"].reset_index(drop=True)
        train_end_ds = pd.to_datetime(train_df["dt"].iloc[-1])
        return predictions, actuals, train_end_ds, train_df, None

    monkeypatch.setattr(runtime, "_fit_and_predict_fold", fake_fit_and_predict_fold)
    monkeypatch.setattr(runtime, "_build_summary_plot_bundle", lambda *args, **kwargs: {})

    runtime._write_per_window_summary_bundles(
        run_root,
        pd.DataFrame([{"model": "AAForecast"}]),
        pd.DataFrame([{"model": "AAForecast"}]),
    )

    assert calls == [
        {
            "job_params": {"best": "param"},
            "training_override": {"lr": 0.01},
            "train_end_ds": "2024-02-18",
        },
        {
            "job_params": {"best": "param"},
            "training_override": {"lr": 0.01},
            "train_end_ds": "2024-02-25",
        },
    ]

    manifest_1 = json.loads((run_root / "summary" / "test_1" / "window_manifest.json").read_text())
    manifest_2 = json.loads((run_root / "summary" / "test_2" / "window_manifest.json").read_text())
    assert manifest_1["anchor_train_end_ds"].startswith("2024-02-11")
    assert manifest_1["origin_ds"].startswith("2024-02-18")
    assert manifest_1["forecast_only"] is False
    assert manifest_1["observed_steps"] == 1
    assert manifest_2["origin_ds"].startswith("2024-02-25")
    assert manifest_2["forecast_only"] is True
    assert manifest_2["observed_steps"] == 0

    rolling_metrics = pd.read_csv(run_root / "cv" / "AAForecast_rolling_origin_metrics.csv")
    rolling_forecasts = pd.read_csv(run_root / "cv" / "AAForecast_rolling_origin_forecasts.csv")
    assert rolling_metrics["test_index"].tolist() == [1]
    assert rolling_metrics["used_best_params"].tolist() == [True]
    assert rolling_forecasts["test_index"].tolist() == [1, 1, 2, 2]
    assert rolling_forecasts["used_best_params"].tolist() == [True, True, True, True]

    sample_md = (run_root / "summary" / "test_2" / "sample.md").read_text(encoding="utf-8")
    assert "rolling-origin forecast" in sample_md
    assert "forecast-only" in sample_md
