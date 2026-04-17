from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from app_config import load_app_config
import runtime_support.runner as runtime


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_shared_training_defaults_use_robust_scaler_and_mae(tmp_path: Path) -> None:
    config_path = _write_minimal_config(
        tmp_path,
        {
            "task": {"name": "defaults-check"},
            "dataset": {
                "path": "data.csv",
                "target_col": "target",
                "dt_col": "dt",
                "hist_exog_cols": [],
                "futr_exog_cols": [],
                "static_exog_cols": [],
            },
            "runtime": {"random_seed": 1},
            "training": {},
            "cv": {"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0},
            "scheduler": {
                "gpu_ids": [0],
                "max_concurrent_jobs": 1,
                "worker_devices": 1,
                "parallelize_single_job_tuning": False,
            },
            "jobs": [{"model": "Naive", "params": {}}],
        },
    )

    loaded = load_app_config(REPO_ROOT, config_path=config_path)

    assert loaded.config.training.scaler_type == "robust"
    assert loaded.config.training.loss == "mae"


def test_write_loss_curve_artifact_keeps_raw_and_adds_smoothed_views(
    tmp_path: Path,
) -> None:
    curve_frame = pd.DataFrame(
        {
            "global_step": [0, 10, 20, 30],
            "train_loss": [8.0, 4.0, 12.0, 6.0],
            "val_loss": [1.0, None, 0.5, None],
        }
    )

    plot_path = runtime._write_loss_curve_artifact(
        tmp_path,
        "AAForecast",
        3,
        nf=runtime._CurveFrameCarrier(curve_frame=curve_frame),
    )

    assert plot_path is not None
    assert plot_path.exists()

    fold_root = tmp_path / "models" / "AAForecast" / "folds" / "fold_003"
    raw_frame = pd.read_csv(fold_root / runtime.LOSS_CURVE_SAMPLE_FILENAME)
    smoothed_frame = pd.read_csv(
        fold_root / runtime.LOSS_CURVE_SMOOTHED_SAMPLE_FILENAME
    )
    validation_frame = pd.read_csv(fold_root / runtime.LOSS_CURVE_VALIDATION_FILENAME)

    assert raw_frame.columns.tolist() == ["global_step", "train_loss", "val_loss"]
    assert smoothed_frame.columns.tolist() == [
        "global_step",
        "train_loss",
        "train_loss_smoothed",
        "val_loss",
    ]
    assert validation_frame.columns.tolist() == [
        "global_step",
        "train_loss",
        "train_loss_smoothed",
        "val_loss",
    ]
    assert smoothed_frame["train_loss_smoothed"].tolist() == [8.0, 6.0, 8.0, 7.5]
    assert validation_frame.to_dict(orient="records") == [
        {
            "global_step": 0,
            "train_loss": 8.0,
            "train_loss_smoothed": 8.0,
            "val_loss": 1.0,
        },
        {
            "global_step": 20,
            "train_loss": 12.0,
            "train_loss_smoothed": 8.0,
            "val_loss": 0.5,
        },
    ]


def test_loss_artifact_summary_lists_raw_smoothed_and_validation_variants(
    tmp_path: Path,
) -> None:
    (tmp_path / "cv").mkdir(parents=True)
    runtime._write_loss_curve_artifact(
        tmp_path,
        "AAForecast",
        0,
        nf=runtime._CurveFrameCarrier(
            curve_frame=pd.DataFrame(
                {
                    "global_step": [0, 10, 20],
                    "train_loss": [5.0, 7.0, 3.0],
                    "val_loss": [1.5, None, 1.0],
                }
            )
        ),
    )

    summary = runtime._load_loss_artifacts_for_summary(tmp_path)

    assert set(summary["curve_variant"]) == {
        "raw_sampled",
        "smoothed_sampled",
        "validation_aligned",
    }
    raw_row = summary.loc[summary["curve_variant"] == "raw_sampled"].iloc[0]
    smoothed_row = summary.loc[summary["curve_variant"] == "smoothed_sampled"].iloc[0]
    validation_row = summary.loc[
        summary["curve_variant"] == "validation_aligned"
    ].iloc[0]

    assert raw_row["source_granularity"] == "step_sampled"
    assert raw_row["sample_every_n_steps"] == runtime.LOSS_CURVE_SAMPLE_EVERY_N_STEPS
    assert pd.isna(raw_row["smoothing_window"])
    assert smoothed_row["smoothing_window"] == runtime.LOSS_CURVE_TRAIN_SMOOTHING_WINDOW
    assert validation_row["source_granularity"] == "validation"
    assert pd.isna(validation_row["sample_every_n_steps"])
    assert validation_row["sample_count"] == 2
