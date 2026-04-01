from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from app_config import load_app_config
import runtime_support.runner as runtime


def _write_dataset(path: Path, *, include_extremes: bool = True) -> Path:
    rows = [
        ("2020-01-01", 10, 1, 5),
        ("2020-01-08", 10, 1, 5),
        ("2020-01-15", 10, 1, 5),
        ("2020-01-22", 30 if include_extremes else 11, 2, 6),
        ("2020-01-29", 30 if include_extremes else 11, 2, 6),
        ("2020-02-05", 10, 1, 5),
        ("2020-02-12", 10, 1, 5),
        ("2020-02-19", 35 if include_extremes else 12, 3, 7),
        ("2020-02-26", 35 if include_extremes else 12, 3, 7),
        ("2020-03-04", 10, 1, 5),
        ("2020-03-11", 10, 1, 5),
        ("2020-03-18", 40 if include_extremes else 13, 4, 8),
        ("2020-03-25", 40 if include_extremes else 13, 4, 8),
        ("2020-04-01", 10, 1, 5),
        ("2020-04-08", 10, 1, 5),
        ("2020-04-15", 45 if include_extremes else 14, 5, 9),
    ]
    df = pd.DataFrame(rows, columns=["dt", "target", "hist_a", "hist_b"])
    df.to_csv(path, index=False)
    return path


def _write_plugin_yaml(path: Path, *, epsilon: float = 1.2) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "nec": {
                    "history_steps": 4,
                    "hist_columns": ["hist_a", "hist_b"],
                    "preprocessing": {
                        "mode": "diff_std",
                        "probability_feature": True,
                        "gmm_components": 2,
                        "epsilon": epsilon,
                    },
                    "classifier": {
                        "hidden_dim": 16,
                        "layer_dim": 2,
                        "dropout": 0.1,
                        "batch_size": 2,
                        "train_volume": 8,
                        "epochs": 2,
                        "early_stop_patience": 1,
                        "encoder_lr": 0.001,
                        "head_lr": 0.001,
                    },
                    "normal": {
                        "hidden_dim": 16,
                        "layer_dim": 2,
                        "dropout": 0.1,
                        "batch_size": 2,
                        "train_volume": 8,
                        "epochs": 2,
                        "early_stop_patience": 1,
                        "encoder_lr": 0.001,
                        "head_lr": 0.001,
                        "oversampling": False,
                        "normal_ratio": 0.0,
                    },
                    "extreme": {
                        "hidden_dim": 16,
                        "layer_dim": 2,
                        "dropout": 0.1,
                        "batch_size": 2,
                        "train_volume": 8,
                        "epochs": 2,
                        "early_stop_patience": 1,
                        "encoder_lr": 0.001,
                        "head_lr": 0.001,
                        "oversampling": True,
                        "normal_ratio": 0.0,
                    },
                    "validation": {"windows": 2},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_main_config(path: Path, data_path: Path, plugin_path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "nec_runtime"},
                "dataset": {
                    "path": str(data_path),
                    "target_col": "target",
                    "dt_col": "dt",
                    "hist_exog_cols": ["hist_a", "hist_b"],
                    "futr_exog_cols": [],
                    "static_exog_cols": [],
                },
                "runtime": {"random_seed": 1},
                "training": {
                    "input_size": 4,
                    "batch_size": 8,
                    "valid_batch_size": 8,
                    "windows_batch_size": 16,
                    "inference_windows_batch_size": 16,
                    "scaler_type": "robust",
                    "model_step_size": 1,
                    "max_steps": 1,
                    "val_size": 1,
                    "val_check_steps": 1,
                    "early_stop_patience_steps": -1,
                    "loss": "mse",
                    "optimizer": {"name": "adamw", "kwargs": {}},
                },
                "cv": {"horizon": 2, "step_size": 2, "n_windows": 1, "gap": 0, "overlap_eval_policy": "by_cutoff_mean"},
                "scheduler": {"gpu_ids": [0], "max_concurrent_jobs": 1, "worker_devices": 1},
                "residual": {"enabled": False, "model": "xgboost", "params": {}},
                "jobs": [{"model": "NEC", "params": {"variant": "paper"}}],
                "nec": {"enabled": True, "config_path": str(plugin_path)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_nec_runtime_produces_predictions_and_fold_artifacts(tmp_path: Path) -> None:
    data_path = _write_dataset(tmp_path / "data.csv", include_extremes=True)
    plugin_path = _write_plugin_yaml(tmp_path / "nec_plugin.yaml")
    config_path = _write_main_config(tmp_path / "config.yaml", data_path, plugin_path)
    loaded = load_app_config(tmp_path, config_path=config_path)
    source_df = pd.read_csv(data_path)
    run_root = tmp_path / "run"

    predictions, actuals, train_end_ds, train_df, nf = runtime._fit_and_predict_fold(
        loaded,
        loaded.config.jobs[0],
        run_root=run_root,
        source_df=source_df,
        freq="W",
        train_idx=list(range(12)),
        test_idx=[12, 13],
    )

    assert predictions["NEC"].tolist()
    assert len(predictions) == 2
    assert actuals.tolist() == source_df.loc[[12, 13], "target"].tolist()
    assert train_df["target"].tolist() == source_df.loc[list(range(12)), "target"].tolist()
    assert str(train_end_ds) == "2020-03-18 00:00:00"
    assert nf is not None

    artifact = json.loads((run_root / "nec" / "nec_fold_summary.json").read_text())
    assert artifact["history_steps"] == 4
    assert artifact["use_probability_feature"] is True
    assert artifact["hist_columns_used"] == ["hist_a", "hist_b"]


def test_nec_runtime_fails_fast_when_no_extreme_windows_exist(tmp_path: Path) -> None:
    data_path = _write_dataset(tmp_path / "data.csv", include_extremes=False)
    plugin_path = _write_plugin_yaml(tmp_path / "nec_plugin.yaml", epsilon=10.0)
    config_path = _write_main_config(tmp_path / "config.yaml", data_path, plugin_path)
    loaded = load_app_config(tmp_path, config_path=config_path)
    source_df = pd.read_csv(data_path)

    with pytest.raises(ValueError, match="extreme window"):
        runtime._fit_and_predict_fold(
            loaded,
            loaded.config.jobs[0],
            source_df=source_df,
            freq="W",
            train_idx=list(range(12)),
            test_idx=[12, 13],
        )
