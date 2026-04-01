from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app_config import load_app_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nec_config_loads_thin_main_yaml_and_preserves_shared_scaler_metadata() -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml",
    )

    assert loaded.config.stage_plugin_config.enabled is True
    assert loaded.config.stage_plugin_config.history_steps == 4
    assert loaded.config.training.scaler_type == "robust"
    assert loaded.normalized_payload["nec"]["stage1"]["shared_scaler_override_active"] is True
    assert loaded.normalized_payload["nec"]["stage1"]["preprocessing_mode"] == "diff_std"
    assert loaded.normalized_payload["nec"]["stage1"]["hist_columns"] == ["hist_a", "hist_b"]


def test_nec_plugin_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    bad_plugin = tmp_path / "nec_bad.yaml"
    bad_plugin.write_text(
        yaml.safe_dump(
            {
                "nec": {
                    "history_steps": 4,
                    "dataset": {"path": "forbidden.csv"},
                    "preprocessing": {
                        "mode": "diff_std",
                        "probability_feature": True,
                        "gmm_components": 2,
                        "epsilon": 1.2,
                    },
                    "classifier": {
                        "hidden_dim": 8,
                        "layer_dim": 1,
                        "dropout": 0.1,
                        "batch_size": 2,
                        "train_volume": 4,
                        "epochs": 1,
                        "early_stop_patience": 1,
                        "encoder_lr": 0.001,
                        "head_lr": 0.001,
                    },
                    "normal": {
                        "hidden_dim": 8,
                        "layer_dim": 1,
                        "dropout": 0.1,
                        "batch_size": 2,
                        "train_volume": 4,
                        "epochs": 1,
                        "early_stop_patience": 1,
                        "encoder_lr": 0.001,
                        "head_lr": 0.001,
                        "oversampling": False,
                        "normal_ratio": 0.0,
                    },
                    "extreme": {
                        "hidden_dim": 8,
                        "layer_dim": 1,
                        "dropout": 0.1,
                        "batch_size": 2,
                        "train_volume": 4,
                        "epochs": 1,
                        "early_stop_patience": 1,
                        "encoder_lr": 0.001,
                        "head_lr": 0.001,
                        "oversampling": True,
                        "normal_ratio": 0.0,
                    },
                    "validation": {"windows": 1},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_cfg = tmp_path / "config.yaml"
    main_cfg.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "nec_bad"},
                "dataset": {
                    "path": str(REPO_ROOT / "tests/fixtures/nec_runtime_smoke.csv"),
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
                "nec": {"enabled": True, "config_path": str(bad_plugin)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unsupported key\(s\): dataset"):
        load_app_config(tmp_path, config_path=main_cfg)
