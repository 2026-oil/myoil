from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import yaml

import runtime_support.runner as runtime


class _FakeAxis:
    def __init__(self) -> None:
        self.plot_calls: list[tuple[object, object]] = []
        self.step_calls: list[tuple[object, object]] = []

    def plot(self, x, y, **_kwargs):
        self.plot_calls.append((x, y))

    def step(self, x, y, **_kwargs):
        self.step_calls.append((x, y))

    def set_title(self, *_args, **_kwargs):
        return None

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, *_args, **_kwargs):
        return None

    def legend(self, *_args, **_kwargs):
        return None

    def set_yticks(self, *_args, **_kwargs):
        return None

    def set_yticklabels(self, *_args, **_kwargs):
        return None

    def set_ylim(self, *_args, **_kwargs):
        return None


class _FakeFigure:
    def autofmt_xdate(self):
        return None

    def tight_layout(self):
        return None

    def savefig(self, path, **_kwargs):
        Path(path).write_text("fake-figure", encoding="utf-8")


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


def test_build_summary_artifacts_do_not_write_per_window_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rolling_origin_source_df: pd.DataFrame,
) -> None:
    dataset_path = tmp_path / "rolling_origin.csv"
    rolling_origin_source_df.to_csv(dataset_path, index=False)

    config_payload = {
        "task": {"name": "cv_summary_test"},
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
        "jobs": [{"model": "Naive", "params": {}}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    run_root = tmp_path / "run"
    (run_root / "manifest").mkdir(parents=True)
    (run_root / "summary").mkdir(parents=True)
    (run_root / "cv").mkdir(parents=True)

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

    monkeypatch.setattr(runtime, "_build_summary_plot_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(runtime, "_write_loss_artifact_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "_load_summary_loaded_config",
        lambda _run_root: SimpleNamespace(
            config=SimpleNamespace(
                dataset=SimpleNamespace(
                    path=str(dataset_path),
                    dt_col="dt",
                    target_col="target",
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
        ),
    )
    metrics_frame = pd.DataFrame(
        [
            {
                "model": "AAForecast",
                "fold_idx": 0,
                "cutoff": "2024-02-04",
                "MAE": 1.0,
                "RMSE": 1.1,
                "MAPE": 1.2,
                "SMAPE": 1.3,
                "MSE": 1.4,
            },
            {
                "model": "AAForecast",
                "fold_idx": 1,
                "cutoff": "2024-02-11",
                "MAE": 2.0,
                "RMSE": 2.1,
                "MAPE": 2.2,
                "SMAPE": 2.3,
                "MSE": 2.4,
            },
        ]
    )
    monkeypatch.setattr(runtime, "_load_metrics_for_summary", lambda _run_root: metrics_frame)

    forecasts_frame = pd.DataFrame(
        [
            {
                "model": "AAForecast",
                "fold_idx": 0,
                "cutoff": "2024-02-04",
                "train_end_ds": "2024-02-04",
                "unique_id": "target",
                "ds": "2024-02-11",
                "horizon_step": 1,
                "y": 15.0,
                "y_hat": 14.5,
                "aaforecast_context_artifact": "aa_forecast/context/test.csv",
            },
            {
                "model": "AAForecast",
                "fold_idx": 0,
                "cutoff": "2024-02-04",
                "train_end_ds": "2024-02-04",
                "unique_id": "target",
                "ds": "2024-02-18",
                "horizon_step": 2,
                "y": 16.0,
                "y_hat": 15.5,
                "aaforecast_context_artifact": "aa_forecast/context/test.csv",
            },
            {
                "model": "AAForecast",
                "fold_idx": 1,
                "cutoff": "2024-02-11",
                "train_end_ds": "2024-02-11",
                "unique_id": "target",
                "ds": "2024-02-18",
                "horizon_step": 1,
                "y": 16.0,
                "y_hat": 15.8,
                "aaforecast_context_artifact": "aa_forecast/context/test.csv",
            },
            {
                "model": "AAForecast",
                "fold_idx": 1,
                "cutoff": "2024-02-11",
                "train_end_ds": "2024-02-11",
                "unique_id": "target",
                "ds": "2024-02-25",
                "horizon_step": 2,
                "y": 17.0,
                "y_hat": 16.8,
                "aaforecast_context_artifact": "aa_forecast/context/test.csv",
            },
        ]
    )
    monkeypatch.setattr(
        runtime,
        "_load_last_fold_forecasts",
        lambda _run_root: forecasts_frame,
    )
    stale_summary_markdown = run_root / "summary" / "sample.md"
    stale_summary_markdown.write_text("stale", encoding="utf-8")
    stale_window_dir = run_root / "summary" / "test_1"
    stale_window_dir.mkdir(parents=True)
    (stale_window_dir / "leaderboard.csv").write_text("stale", encoding="utf-8")

    artifact_paths = runtime._build_summary_artifacts(run_root)
    result_frame = pd.read_csv(run_root / "summary" / "result.csv")

    assert (run_root / "summary" / "leaderboard.csv").exists()
    assert (run_root / "summary" / "result.csv").exists()
    assert not stale_summary_markdown.exists()
    assert artifact_paths["leaderboard"].endswith("summary/leaderboard.csv")
    assert artifact_paths["result"].endswith("summary/result.csv")
    assert not stale_window_dir.exists()
    assert not any((run_root / "summary").glob("test_*"))
    assert result_frame.columns.tolist() == [
        "model",
        "fold_idx",
        "cutoff",
        "train_end_ds",
        "unique_id",
        "ds",
        "horizon_step",
        "y",
        "y_hat",
        "aaforecast_context_artifact",
    ]
    assert result_frame[["model", "fold_idx", "ds", "y", "y_hat"]].to_dict(
        orient="records"
    ) == [
        {
            "model": "AAForecast",
            "fold_idx": 0,
            "ds": "2024-02-11",
            "y": 15.0,
            "y_hat": 14.5,
        },
        {
            "model": "AAForecast",
            "fold_idx": 0,
            "ds": "2024-02-18",
            "y": 16.0,
            "y_hat": 15.5,
        },
        {
            "model": "AAForecast",
            "fold_idx": 1,
            "ds": "2024-02-18",
            "y": 16.0,
            "y_hat": 15.8,
        },
        {
            "model": "AAForecast",
            "fold_idx": 1,
            "ds": "2024-02-25",
            "y": 17.0,
            "y_hat": 16.8,
        },
    ]


def test_plot_last_fold_overlay_uses_single_panel_without_aaforecast_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_subplots(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return _FakeFigure(), _FakeAxis()

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "subplots", fake_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    forecasts = pd.DataFrame(
        {
            "model": ["Naive", "Naive"],
            "ds": ["2024-03-03", "2024-03-10"],
            "y_hat": [100.0, 101.0],
            "fold_idx": [0, 0],
            "cutoff": ["2024-02-25", "2024-02-25"],
        }
    )

    runtime._plot_last_fold_overlay(
        forecasts,
        ["Naive"],
        tmp_path / "plain.png",
        title="plain",
        run_root=tmp_path,
    )

    assert calls["args"] == ()
    assert calls["kwargs"]["figsize"] == (12, 6)
    assert "sharex" not in calls["kwargs"]


def test_summary_overlay_actual_frames_limit_to_input_and_output_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rolling_origin_source_df: pd.DataFrame,
) -> None:
    dataset_path = tmp_path / "overlay.csv"
    rolling_origin_source_df.to_csv(dataset_path, index=False)

    monkeypatch.setattr(
        runtime,
        "_load_summary_loaded_config",
        lambda _run_root: SimpleNamespace(
            config=SimpleNamespace(
                dataset=SimpleNamespace(
                    path=str(dataset_path),
                    dt_col="dt",
                    target_col="target",
                ),
                training=SimpleNamespace(input_size=3),
            )
        ),
    )

    forecasts = pd.DataFrame(
        {
            "model": ["Naive", "Naive"],
            "ds": ["2024-02-18", "2024-02-25"],
            "y": [16.0, 17.0],
            "y_hat": [15.5, 16.5],
            "train_end_ds": ["2024-02-11", "2024-02-11"],
            "fold_idx": [0, 0],
            "cutoff": ["2024-02-11", "2024-02-11"],
        }
    )

    input_actual, output_actual = runtime._summary_overlay_actual_frames(
        tmp_path,
        forecasts,
    )

    assert input_actual["ds"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-28",
        "2024-02-04",
        "2024-02-11",
    ]
    assert output_actual["ds"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-02-18",
        "2024-02-25",
    ]


def test_summary_overlay_actual_frames_supports_history_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2024-01-07", periods=24, freq="W-SUN")
    dataset_path = tmp_path / "overlay_override.csv"
    pd.DataFrame(
        {
            "dt": dates,
            "target": [float(idx) for idx in range(24)],
        }
    ).to_csv(dataset_path, index=False)

    monkeypatch.setattr(
        runtime,
        "_load_summary_loaded_config",
        lambda _run_root: SimpleNamespace(
            config=SimpleNamespace(
                dataset=SimpleNamespace(
                    path=str(dataset_path),
                    dt_col="dt",
                    target_col="target",
                ),
                training=SimpleNamespace(input_size=20),
            )
        ),
    )

    forecasts = pd.DataFrame(
        {
            "model": ["Naive", "Naive"],
            "ds": ["2024-06-23", "2024-06-30"],
            "y": [24.0, 25.0],
            "y_hat": [24.5, 25.5],
            "train_end_ds": ["2024-06-16", "2024-06-16"],
            "fold_idx": [0, 0],
            "cutoff": ["2024-06-16", "2024-06-16"],
        }
    )

    input_actual, output_actual = runtime._summary_overlay_actual_frames(
        tmp_path,
        forecasts,
        history_steps_override=16,
    )

    assert len(input_actual) == 16
    assert input_actual["ds"].dt.strftime("%Y-%m-%d").tolist()[0] == "2024-03-03"
    assert input_actual["ds"].dt.strftime("%Y-%m-%d").tolist()[-1] == "2024-06-16"
    assert output_actual["ds"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-06-23",
        "2024-06-30",
    ]


def test_load_last_fold_forecasts_skips_rolling_origin_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    cv_dir = run_root / "cv"
    cv_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "fold_idx": [0, 1],
            "cutoff": ["2024-01-07", "2024-01-14"],
            "train_end_ds": ["2024-01-07", "2024-01-14"],
            "ds": ["2024-01-14", "2024-01-21"],
            "y": [1.0, 2.0],
            "y_hat": [1.5, 2.5],
        }
    ).to_csv(cv_dir / "AAForecast_forecasts.csv", index=False)
    pd.DataFrame(
        {
            "fold_idx": [1],
            "cutoff": ["2024-01-21"],
            "train_end_ds": ["2024-01-21"],
            "ds": ["2024-01-28"],
            "y": [3.0],
            "y_hat": [3.5],
        }
    ).to_csv(cv_dir / "AAForecast_rolling_origin_forecasts.csv", index=False)

    forecasts = runtime._load_last_fold_forecasts(run_root)

    assert forecasts["fold_idx"].tolist() == [0, 1]
    assert forecasts["train_end_ds"].tolist() == ["2024-01-07", "2024-01-14"]
    assert len(forecasts) == 2


def test_plot_last_fold_overlay_connects_input_actual_to_output_and_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    price_axis = _FakeAxis()

    def fake_subplots(*args, **kwargs):
        return _FakeFigure(), price_axis

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "subplots", fake_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "_summary_overlay_actual_frames",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-02-11", "2024-02-18"]),
                    "y": [10.0, 11.0],
                }
            ),
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-02-25", "2024-03-03"]),
                    "y": [12.0, 13.0],
                }
            ),
        ),
    )

    forecasts = pd.DataFrame(
        {
            "model": ["Naive", "Naive"],
            "ds": ["2024-02-25", "2024-03-03"],
            "y_hat": [12.5, 13.5],
            "fold_idx": [0, 0],
            "cutoff": ["2024-02-18", "2024-02-18"],
        }
    )

    runtime._plot_last_fold_overlay(
        forecasts,
        ["Naive"],
        tmp_path / "connected.png",
        title="connected",
        run_root=tmp_path,
    )

    assert len(price_axis.plot_calls) == 3
    output_x, output_y = price_axis.plot_calls[1]
    pred_x, pred_y = price_axis.plot_calls[2]
    assert [str(value.date()) for value in output_x.tolist()] == [
        "2024-02-18",
        "2024-02-25",
        "2024-03-03",
    ]
    assert output_y.tolist() == [11.0, 12.0, 13.0]
    assert [str(value.date()) for value in pred_x.tolist()] == [
        "2024-02-18",
        "2024-02-25",
        "2024-03-03",
    ]
    assert pred_y.tolist() == [11.0, 12.5, 13.5]


def test_plot_last_fold_overlay_skips_future_only_prediction_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    price_axis = _FakeAxis()

    def fake_subplots(*args, **kwargs):
        return _FakeFigure(), price_axis

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "subplots", fake_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "_summary_overlay_actual_frames",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-02-11", "2024-02-18"]),
                    "y": [10.0, 11.0],
                }
            ),
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-02-25"]),
                    "y": [12.0],
                }
            ),
        ),
    )

    forecasts = pd.DataFrame(
        {
            "model": ["Naive", "Naive"],
            "ds": ["2024-02-25", "2024-03-03"],
            "y": [12.0, None],
            "y_hat": [12.5, 13.5],
            "fold_idx": [0, 0],
            "cutoff": ["2024-02-18", "2024-02-18"],
        }
    )

    runtime._plot_last_fold_overlay(
        forecasts,
        ["Naive"],
        tmp_path / "future-gap.png",
        title="future-gap",
        run_root=tmp_path,
    )

    assert len(price_axis.plot_calls) == 3
    pred_x, pred_y = price_axis.plot_calls[2]
    assert [str(value.date()) for value in pred_x.tolist()] == [
        "2024-02-18",
        "2024-02-25",
    ]
    assert pred_y.tolist() == [11.0, 12.5]


def test_plot_last_fold_overlay_includes_future_only_prediction_segment_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    price_axis = _FakeAxis()

    def fake_subplots(*args, **kwargs):
        return _FakeFigure(), price_axis

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "subplots", fake_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "_summary_overlay_actual_frames",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-02-11", "2024-02-18"]),
                    "y": [10.0, 11.0],
                }
            ),
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-02-25"]),
                    "y": [12.0],
                }
            ),
        ),
    )

    forecasts = pd.DataFrame(
        {
            "model": ["Naive", "Naive"],
            "ds": ["2024-02-25", "2024-03-03"],
            "y": [12.0, None],
            "y_hat": [12.5, 13.5],
            "fold_idx": [0, 0],
            "cutoff": ["2024-02-18", "2024-02-18"],
        }
    )

    runtime._plot_last_fold_overlay(
        forecasts,
        ["Naive"],
        tmp_path / "future-included.png",
        title="future-included",
        run_root=tmp_path,
        include_future_only_predictions=True,
    )

    assert len(price_axis.plot_calls) == 3
    pred_x, pred_y = price_axis.plot_calls[2]
    assert [str(value.date()) for value in pred_x.tolist()] == [
        "2024-02-18",
        "2024-02-25",
        "2024-03-03",
    ]
    assert pred_y.tolist() == [11.0, 12.5, 13.5]


def test_plot_last_fold_overlay_adds_lower_context_subplot_for_aaforecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    price_axis = _FakeAxis()
    context_axis = _FakeAxis()
    calls: dict[str, object] = {}
    context_csv = tmp_path / "aa_forecast" / "context" / "20240301T000000.csv"
    context_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-02-11", "2024-02-18", "2024-02-25"]),
            "context_active": [0, 1, 0],
            "context_label": ["normal_context", "anomaly_context", "normal_context"],
        }
    ).to_csv(context_csv, index=False)

    def fake_subplots(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return _FakeFigure(), (price_axis, context_axis)

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "subplots", fake_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    forecasts = pd.DataFrame(
        {
            "model": ["AAForecast", "AAForecast"],
            "ds": ["2024-03-03", "2024-03-10"],
            "y_hat": [100.0, 101.0],
            "fold_idx": [0, 0],
            "cutoff": ["2024-02-25", "2024-02-25"],
            "aaforecast_context_artifact": [
                "aa_forecast/context/20240301T000000.csv",
                "aa_forecast/context/20240301T000000.csv",
            ],
        }
    )

    runtime._plot_last_fold_overlay(
        forecasts,
        ["AAForecast"],
        tmp_path / "context.png",
        title="context",
        run_root=tmp_path,
    )

    assert calls["args"] == (2, 1)
    assert calls["kwargs"]["sharex"] is True
    assert len(price_axis.plot_calls) == 1
    assert len(price_axis.step_calls) == 0
    assert len(context_axis.step_calls) == 1


def test_plot_last_fold_overlay_clips_context_subplot_to_input_dates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    price_axis = _FakeAxis()
    context_axis = _FakeAxis()
    context_csv = tmp_path / "aa_forecast" / "context" / "20240301T000000.csv"
    context_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2024-02-04",
                    "2024-02-11",
                    "2024-02-18",
                    "2024-02-25",
                    "2024-03-03",
                ]
            ),
            "context_active": [0, 1, 0, 1, 1],
        }
    ).to_csv(context_csv, index=False)

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        plt,
        "subplots",
        lambda *args, **kwargs: (_FakeFigure(), (price_axis, context_axis)),
    )
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "_summary_overlay_actual_frames",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-02-11", "2024-02-18"]),
                    "y": [90.0, 91.0],
                }
            ),
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(["2024-02-25", "2024-03-03"]),
                    "y": [92.0, 93.0],
                }
            ),
        ),
    )

    forecasts = pd.DataFrame(
        {
            "model": ["AAForecast", "AAForecast"],
            "ds": ["2024-02-25", "2024-03-03"],
            "y_hat": [100.0, 101.0],
            "fold_idx": [0, 0],
            "cutoff": ["2024-02-18", "2024-02-18"],
            "train_end_ds": ["2024-02-18", "2024-02-18"],
            "aaforecast_context_artifact": [
                "aa_forecast/context/20240301T000000.csv",
                "aa_forecast/context/20240301T000000.csv",
            ],
        }
    )

    runtime._plot_last_fold_overlay(
        forecasts,
        ["AAForecast"],
        tmp_path / "context-clipped.png",
        title="context-clipped",
        run_root=tmp_path,
    )

    assert len(context_axis.step_calls) == 1
    context_x, context_y = context_axis.step_calls[0]
    assert [str(value.date()) for value in context_x.tolist()] == [
        "2024-02-11",
        "2024-02-18",
    ]
    assert context_y.tolist() == [1, 0]


def test_plot_last_fold_overlay_fails_fast_when_context_artifact_is_missing(
    tmp_path: Path,
) -> None:
    forecasts = pd.DataFrame(
        {
            "model": ["AAForecast"],
            "ds": ["2024-03-03"],
            "y_hat": [100.0],
            "fold_idx": [0],
            "cutoff": ["2024-02-25"],
            "aaforecast_context_artifact": ["aa_forecast/context/missing.csv"],
        }
    )

    with pytest.raises(FileNotFoundError, match="AAForecast context artifact"):
        runtime._plot_last_fold_overlay(
            forecasts,
            ["AAForecast"],
            tmp_path / "missing.png",
            title="missing",
            run_root=tmp_path,
        )


def test_build_summary_plot_bundle_writes_window_16_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_plot(
        forecasts,
        selected_models,
        plot_path,
        *,
        title,
        run_root,
        history_steps_override=None,
        include_future_only_predictions=False,
    ):
        del forecasts, selected_models, title, run_root
        calls.append(
            {
                "path": Path(plot_path).name,
                "history_steps_override": history_steps_override,
                "include_future_only_predictions": include_future_only_predictions,
            }
        )
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        Path(plot_path).write_text("fake-figure", encoding="utf-8")

    monkeypatch.setattr(runtime, "_plot_last_fold_overlay", fake_plot)

    artifact_paths = runtime._build_summary_plot_bundle(
        tmp_path,
        tmp_path / "summary",
        pd.DataFrame([{"rank": 1, "model": "Naive"}]),
        pd.DataFrame([{"model": "Naive"}]),
        title_prefix="Last fold predictions",
    )

    assert (tmp_path / "summary" / "last_fold_all_models.png").exists()
    assert (tmp_path / "summary" / "last_fold_all_models_window_16.png").exists()
    assert artifact_paths["all_models"].endswith("last_fold_all_models.png")
    assert artifact_paths["all_models_window_16"].endswith(
        "last_fold_all_models_window_16.png"
    )
    assert calls == [
        {
            "path": "last_fold_all_models.png",
            "history_steps_override": None,
            "include_future_only_predictions": False,
        },
        {
            "path": "last_fold_all_models_window_16.png",
            "history_steps_override": 16,
            "include_future_only_predictions": False,
        },
    ]


def test_write_summary_bundle_enables_future_tail_only_for_requested_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_build_summary_plot_bundle(
        run_root,
        summary_dir,
        leaderboard,
        scoped_forecasts,
        *,
        title_prefix,
        include_future_only_predictions=False,
    ):
        del run_root, summary_dir, leaderboard, scoped_forecasts, title_prefix
        calls.append(include_future_only_predictions)
        return {"all_models": "fake.png"}

    monkeypatch.setattr(runtime, "_build_summary_plot_bundle", fake_build_summary_plot_bundle)

    metrics = pd.DataFrame(
        [
            {
                "model": "Naive",
                "fold_idx": 0,
                "MAE": 1.0,
                "MSE": 1.0,
                "RMSE": 1.0,
                "MAPE": 0.1,
                "NRMSE": 0.1,
                "R2": 0.9,
            }
        ]
    )
    scoped_forecasts = pd.DataFrame([{"model": "Naive", "fold_idx": 0}])

    runtime._write_summary_bundle(
        tmp_path,
        tmp_path / "summary-default",
        metrics,
        scoped_forecasts=scoped_forecasts,
        title_prefix="default",
    )
    runtime._write_summary_bundle(
        tmp_path,
        tmp_path / "summary-test",
        metrics,
        scoped_forecasts=scoped_forecasts,
        title_prefix="test",
        include_future_only_predictions=True,
    )

    assert calls == [False, True]
