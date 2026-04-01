from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app_config import load_app_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nec_config_loads_branch_contract_and_replaces_legacy_metadata() -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml",
    )

    stage1 = loaded.normalized_payload["nec"]["stage1"]
    assert loaded.config.stage_plugin_config.enabled is True
    assert loaded.config.training.scaler_type == "robust"
    assert stage1["shared_scaler_override_active"] is True
    assert stage1["history_steps_source"] == "training.input_size"
    assert stage1["history_steps_value"] == 4
    assert stage1["probability_feature_forced"] is True
    assert stage1["active_hist_columns"] == ["hist_a", "hist_b"]
    assert stage1["branches"]["classifier"]["model"] == "MLP"
    assert stage1["branches"]["classifier"]["variables"] == []
    assert stage1["branches"]["classifier"]["alpha"] == 2.0
    assert stage1["branches"]["classifier"]["beta"] == 0.5
    assert stage1["branches"]["classifier"]["oversample_extreme_windows"] is True
    assert stage1["branches"]["normal"]["variables"] == ["hist_a"]
    assert stage1["branches"]["normal"]["oversample_extreme_windows"] is False
    assert stage1["branches"]["extreme"]["variables"] == ["hist_a", "hist_b"]
    assert stage1["branches"]["extreme"]["oversample_extreme_windows"] is True
    assert "history_steps" not in stage1
    assert "hist_columns" not in stage1
    assert "probability_feature" not in stage1


def test_nec_plugin_yaml_rejects_removed_keys(tmp_path: Path) -> None:
    bad_plugin = tmp_path / "nec_bad.yaml"
    bad_plugin.write_text(
        yaml.safe_dump(
            {
                "nec": {
                    "history_steps": 4,
                    "hist_columns": ["hist_a"],
                    "preprocessing": {
                        "mode": "diff_std",
                        "probability_feature": True,
                        "gmm_components": 2,
                        "epsilon": 1.2,
                    },
                    "classifier": {"model": "MLP", "variables": [], "model_params": {}},
                    "normal": {"model": "MLP", "variables": [], "model_params": {}},
                    "extreme": {"model": "MLP", "variables": [], "model_params": {}},
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
                "jobs": [{"model": "NEC"}],
                "nec": {"enabled": True, "config_path": str(bad_plugin)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"no longer supports top-level key\(s\): (hist_columns, history_steps|history_steps, hist_columns)|probability_feature has been removed"):
        load_app_config(tmp_path, config_path=main_cfg)


def test_nec_plugin_yaml_rejects_unknown_branch_model_params(tmp_path: Path) -> None:
    bad_plugin = tmp_path / "nec_bad_params.yaml"
    bad_plugin.write_text(
        yaml.safe_dump(
            {
                "nec": {
                    "preprocessing": {"mode": "diff_std", "gmm_components": 2, "epsilon": 1.2},
                    "classifier": {"model": "MLP", "variables": [], "model_params": {"bad_param": 1}},
                    "normal": {"model": "MLP", "variables": [], "model_params": {}},
                    "extreme": {"model": "MLP", "variables": [], "model_params": {}},
                    "validation": {"windows": 1},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_cfg = REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml"
    copied = tmp_path / "config.yaml"
    payload = yaml.safe_load(main_cfg.read_text(encoding="utf-8"))
    payload["nec"]["config_path"] = str(bad_plugin)
    copied.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported parameter\(s\).+bad_param"):
        load_app_config(tmp_path, config_path=copied)


def test_nec_plugin_yaml_rejects_centralized_training_branch_params(tmp_path: Path) -> None:
    bad_plugin = tmp_path / "nec_bad_centralized.yaml"
    bad_plugin.write_text(
        yaml.safe_dump(
            {
                "nec": {
                    "preprocessing": {"mode": "diff_std", "gmm_components": 2, "epsilon": 1.2},
                    "classifier": {
                        "model": "MLP",
                        "variables": [],
                        "model_params": {"hidden_size": 16, "max_steps": 1, "batch_size": 2},
                    },
                    "normal": {"model": "MLP", "variables": [], "model_params": {}},
                    "extreme": {"model": "MLP", "variables": [], "model_params": {}},
                    "validation": {"windows": 1},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_cfg = REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml"
    copied = tmp_path / "config.yaml"
    payload = yaml.safe_load(main_cfg.read_text(encoding="utf-8"))
    payload["nec"]["config_path"] = str(bad_plugin)
    copied.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"repeats centralized training key\(s\): batch_size, max_steps",
    ):
        load_app_config(tmp_path, config_path=copied)


def test_nec_plugin_yaml_rejects_non_positive_classifier_alpha(tmp_path: Path) -> None:
    bad_plugin = tmp_path / "nec_bad_alpha.yaml"
    bad_plugin.write_text(
        yaml.safe_dump(
            {
                "nec": {
                    "preprocessing": {"mode": "diff_std", "gmm_components": 2, "epsilon": 1.2},
                    "classifier": {"model": "MLP", "variables": [], "model_params": {}, "alpha": 0},
                    "normal": {"model": "MLP", "variables": [], "model_params": {}},
                    "extreme": {"model": "MLP", "variables": [], "model_params": {}},
                    "validation": {"windows": 1},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_cfg = REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml"
    copied = tmp_path / "config.yaml"
    payload = yaml.safe_load(main_cfg.read_text(encoding="utf-8"))
    payload["nec"]["config_path"] = str(bad_plugin)
    copied.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"nec.classifier.alpha must be > 0"):
        load_app_config(tmp_path, config_path=copied)


def test_nec_plugin_yaml_rejects_incompatible_branch_model(tmp_path: Path) -> None:
    bad_plugin = tmp_path / "nec_bad_model.yaml"
    bad_plugin.write_text(
        yaml.safe_dump(
            {
                "nec": {
                    "preprocessing": {"mode": "diff_std", "gmm_components": 2, "epsilon": 1.2},
                    "classifier": {"model": "TimeXer", "variables": [], "model_params": {}},
                    "normal": {"model": "MLP", "variables": [], "model_params": {}},
                    "extreme": {"model": "MLP", "variables": [], "model_params": {}},
                    "validation": {"windows": 1},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_cfg = REPO_ROOT / "tests/fixtures/nec_runtime_smoke.yaml"
    copied = tmp_path / "config.yaml"
    payload = yaml.safe_load(main_cfg.read_text(encoding="utf-8"))
    payload["nec"]["config_path"] = str(bad_plugin)
    copied.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"multivariate and not supported"):
        load_app_config(tmp_path, config_path=copied)


def test_feature_set_nec_defaults_plugin_route_when_config_path_is_omitted() -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / "yaml/experiment/feature_set_nec/nec.yaml",
    )

    assert len(loaded.config.jobs) == 1
    assert loaded.config.jobs[0].model == "NEC"
    assert loaded.config.jobs[0].params == {}
    assert loaded.config.jobs[0].requested_mode == "learned_fixed"
    assert loaded.config.jobs[0].validated_mode == "learned_fixed"
    assert loaded.config.stage_plugin_config.enabled is True
    assert loaded.config.stage_plugin_config.config_path == "yaml/plugins/nec.yaml"
    assert loaded.normalized_payload["nec"]["stage1"]["source_path"].endswith("yaml/plugins/nec.yaml")
