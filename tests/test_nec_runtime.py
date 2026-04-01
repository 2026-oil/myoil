from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from app_config import load_app_config
import runtime_support.runner as runtime
from runtime_support.forecast_models import build_model
from plugins.nec.runtime import _NecPredictor


REPO_ROOT = Path(__file__).resolve().parents[1]


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
                    "preprocessing": {"mode": "diff_std", "gmm_components": 2, "epsilon": epsilon},
                    "inference": {"mode": "soft_weighted", "threshold": 0.5},
                    "classifier": {
                        "model": "MLP",
                        "variables": [],
                        "model_params": {"hidden_size": 16, "num_layers": 1},
                        "alpha": 2.0,
                        "beta": 0.5,
                        "oversample_extreme_windows": True,
                    },
                    "normal": {
                        "model": "MLP",
                        "variables": ["hist_a"],
                        "model_params": {"hidden_size": 16, "num_layers": 1},
                        "oversample_extreme_windows": False,
                    },
                    "extreme": {
                        "model": "MLP",
                        "variables": ["hist_a", "hist_b"],
                        "model_params": {"hidden_size": 16, "num_layers": 1},
                        "oversample_extreme_windows": True,
                    },
                    "validation": {"windows": 2},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_uniform_backbone_plugin_yaml(
    path: Path,
    *,
    model_name: str,
    model_params: dict,
    epsilon: float = 1.2,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "nec": {
                    "preprocessing": {"mode": "diff_std", "gmm_components": 2, "epsilon": epsilon},
                    "inference": {"mode": "soft_weighted", "threshold": 0.5},
                    "classifier": {
                        "model": model_name,
                        "variables": [],
                        "model_params": dict(model_params),
                        "alpha": 2.0,
                        "beta": 0.5,
                        "oversample_extreme_windows": True,
                    },
                    "normal": {
                        "model": model_name,
                        "variables": ["hist_a"],
                        "model_params": dict(model_params),
                        "oversample_extreme_windows": False,
                    },
                    "extreme": {
                        "model": model_name,
                        "variables": ["hist_a", "hist_b"],
                        "model_params": dict(model_params),
                        "oversample_extreme_windows": True,
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
                "jobs": [{"model": "NEC"}],
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
    loaded = load_app_config(REPO_ROOT, config_path=config_path)
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
    assert nf is None

    artifact = json.loads((run_root / "nec" / "nec_fold_summary.json").read_text())
    assert artifact["history_steps_source"] == "training.input_size"
    assert artifact["history_steps_value"] == 4
    assert artifact["probability_feature_forced"] is True
    assert artifact["active_hist_columns"] == ["hist_a", "hist_b"]
    assert artifact["merge_mode"] == "soft_weighted"
    assert artifact["branches"]["classifier"]["model"] == "MLP"
    assert artifact["branches"]["classifier"]["sampled_series_count"] == 10
    assert artifact["branches"]["classifier"]["oversampled_window_count"] == 5
    assert artifact["branches"]["normal"]["sampled_series_count"] == 5
    assert artifact["branches"]["normal"]["oversampled_window_count"] == 0
    assert artifact["branches"]["extreme"]["sampled_series_count"] == 10
    assert artifact["branches"]["extreme"]["oversampled_window_count"] == 5
    assert "history_steps" not in artifact
    assert "hist_columns_used" not in artifact
    assert (run_root / "summary" / "nec" / "normal" / "fold_000.csv").exists()
    assert (run_root / "summary" / "nec" / "normal" / "fold_000.png").exists()
    assert (run_root / "summary" / "nec" / "extreme" / "fold_000.csv").exists()
    assert (run_root / "summary" / "nec" / "classifier" / "fold_000.csv").exists()


def test_nec_predictor_routes_inputs_by_model_role(tmp_path: Path) -> None:
    data_path = _write_dataset(tmp_path / "data.csv", include_extremes=True)
    plugin_path = _write_plugin_yaml(tmp_path / "nec_plugin.yaml")
    config_path = _write_main_config(tmp_path / "config.yaml", data_path, plugin_path)
    loaded = load_app_config(REPO_ROOT, config_path=config_path)
    source_df = pd.read_csv(data_path)
    predictor = _NecPredictor(
        loaded=loaded,
        train_df=source_df.iloc[:12].reset_index(drop=True),
        future_df=source_df.iloc[12:14].reset_index(drop=True),
    )

    assert predictor.classifier_feature_matrix.shape[1] == 2  # target + probability
    assert predictor.extreme_feature_matrix.shape[1] == 4  # target + hist_a + hist_b + probability
    assert predictor.normal_feature_matrix.shape[1] == 2  # target + hist_a


def test_nec_branch_models_inherit_shared_training_settings(tmp_path: Path) -> None:
    data_path = _write_dataset(tmp_path / "data.csv", include_extremes=True)
    plugin_path = _write_plugin_yaml(tmp_path / "nec_plugin.yaml")
    config_path = _write_main_config(tmp_path / "config.yaml", data_path, plugin_path)
    loaded = load_app_config(REPO_ROOT, config_path=config_path)
    source_df = pd.read_csv(data_path)
    predictor = _NecPredictor(
        loaded=loaded,
        train_df=source_df.iloc[:12].reset_index(drop=True),
        future_df=source_df.iloc[12:14].reset_index(drop=True),
    )

    branch_loaded, branch_job = predictor._branch_config("normal")
    model = build_model(branch_loaded.config, branch_job, params_override=branch_job.params)

    assert "max_steps" not in branch_job.params
    assert "batch_size" not in branch_job.params
    assert model.hparams.max_steps == loaded.config.training.max_steps
    assert model.hparams.batch_size == loaded.config.training.batch_size


@pytest.mark.parametrize(
    ("model_name", "model_params"),
    [
        (
            "TimeXer",
            {
                "patch_len": 1,
                "hidden_size": 8,
                "n_heads": 1,
                "e_layers": 1,
                "d_ff": 16,
                "factor": 1,
                "dropout": 0.1,
                "use_norm": True,
            },
        ),
        (
            "TSMixerx",
            {
                "n_block": 2,
                "ff_dim": 16,
                "dropout": 0.1,
                "revin": True,
            },
        ),
        (
            "iTransformer",
            {
                "hidden_size": 8,
                "n_heads": 1,
                "e_layers": 1,
                "d_ff": 16,
                "dropout": 0.0,
            },
        ),
    ],
)
def test_nec_runtime_supports_variant_backbones(
    tmp_path: Path,
    model_name: str,
    model_params: dict,
) -> None:
    data_path = _write_dataset(tmp_path / "data.csv", include_extremes=True)
    plugin_path = _write_uniform_backbone_plugin_yaml(
        tmp_path / f"{model_name.lower()}_plugin.yaml",
        model_name=model_name,
        model_params=model_params,
    )
    config_path = _write_main_config(tmp_path / "config.yaml", data_path, plugin_path)
    loaded = load_app_config(REPO_ROOT, config_path=config_path)
    source_df = pd.read_csv(data_path)

    predictions, actuals, *_rest = runtime._fit_and_predict_fold(
        loaded,
        loaded.config.jobs[0],
        source_df=source_df,
        freq="W",
        train_idx=list(range(12)),
        test_idx=[12, 13],
    )

    assert len(predictions) == 2
    assert len(actuals) == 2
    assert predictions["NEC"].notna().all()


def test_nec_predictor_uses_selected_losses_and_oversampled_windows(tmp_path: Path) -> None:
    data_path = _write_dataset(tmp_path / "data.csv", include_extremes=True)
    plugin_path = _write_plugin_yaml(tmp_path / "nec_plugin.yaml")
    config_path = _write_main_config(tmp_path / "config.yaml", data_path, plugin_path)
    loaded = load_app_config(REPO_ROOT, config_path=config_path)
    source_df = pd.read_csv(data_path)
    predictor = _NecPredictor(
        loaded=loaded,
        train_df=source_df.iloc[:12].reset_index(drop=True),
        future_df=source_df.iloc[12:14].reset_index(drop=True),
    )

    normal_loss, _ = predictor._branch_losses("normal")
    extreme_loss, _ = predictor._branch_losses("extreme")
    classifier_loss, _ = predictor._branch_losses("classifier")

    import torch

    y = torch.tensor([[[0.0], [2.5]]], dtype=torch.float32)
    y_hat = torch.tensor([[[1.0], [3.5]]], dtype=torch.float32)
    assert torch.isclose(normal_loss(y, y_hat), torch.tensor(1.0))
    assert torch.isclose(extreme_loss(y, y_hat), torch.tensor(1.0))
    assert classifier_loss.alpha == 2.0
    assert classifier_loss.beta == 0.5

    branch_loaded, _branch_job = predictor._branch_config("extreme")
    train_frame = predictor._branch_training_frame("extreme")
    sampled_fit_df, oversampled_window_count = predictor._sampled_branch_fit_df(
        "extreme",
        train_frame,
        branch_loaded,
    )
    assert oversampled_window_count == 5
    assert sampled_fit_df["unique_id"].nunique() == 10


def test_nec_runtime_fails_fast_when_no_extreme_windows_exist(tmp_path: Path) -> None:
    data_path = _write_dataset(tmp_path / "data.csv", include_extremes=False)
    plugin_path = _write_plugin_yaml(tmp_path / "nec_plugin.yaml", epsilon=10.0)
    config_path = _write_main_config(tmp_path / "config.yaml", data_path, plugin_path)
    loaded = load_app_config(REPO_ROOT, config_path=config_path)
    source_df = pd.read_csv(data_path)

    predictions, actuals, *_rest = runtime._fit_and_predict_fold(
        loaded,
        loaded.config.jobs[0],
        source_df=source_df,
        freq="W",
        train_idx=list(range(12)),
        test_idx=[12, 13],
    )
    assert len(predictions) == 2
    assert len(actuals) == 2


@pytest.mark.parametrize(
    ("config_name", "expected_plugin", "expected_model"),
    [
        ("nec_lstm", "yaml/plugins/nec_lstm.yaml", "LSTM"),
        ("nec_timexer", "yaml/plugins/nec_timexer.yaml", "TimeXer"),
        ("nec_tsmixerx", "yaml/plugins/nec_tsmixerx.yaml", "TSMixerx"),
        ("nec_itransformer", "yaml/plugins/nec_itransformer.yaml", "iTransformer"),
    ],
)
def test_feature_set_nec_validate_only_exposes_branch_metadata(
    config_name: str,
    expected_plugin: str,
    expected_model: str,
) -> None:
    code = runtime.main([
        "--config",
        f"yaml/experiment/feature_set_nec/{config_name}.yaml",
        "--validate-only",
    ])
    assert code == 0
    root = Path(f"runs/feature_set_nec_brentoil_case1_{config_name}")
    resolved = json.loads((root / "config" / "config.resolved.json").read_text())
    nec_payload = resolved["nec"]
    assert nec_payload["selected_config_path"].endswith(expected_plugin)
    assert nec_payload["stage1"]["source_path"].endswith(expected_plugin)
    assert nec_payload["history_steps_source"] == "training.input_size"
    assert nec_payload["probability_feature_forced"] is True
    assert "hist_columns" not in nec_payload
    assert "history_steps" not in nec_payload
    assert nec_payload["branches"]["classifier"]["model"] == expected_model
    assert nec_payload["branches"]["normal"]["model"] == expected_model
    assert nec_payload["branches"]["extreme"]["model"] == expected_model
    assert nec_payload["branches"]["classifier"]["alpha"] == 2.0
    assert nec_payload["branches"]["classifier"]["beta"] == 0.5
    assert nec_payload["branches"]["classifier"]["oversample_extreme_windows"] is True
    assert nec_payload["branches"]["normal"]["variables"] == [
        "Idx_OVX",
        "Com_Oil_Spread",
        "BS_Core_Index_A",
        "BS_Core_Index_B",
        "BS_Core_Index_C",
        "Com_LMEX",
        "Com_BloombergCommodity_BCOM",
        "GPRD_THREAT",
        "GPRD",
        "GPRD_ACT",
    ]
    assert nec_payload["active_hist_columns"] == [
        "Idx_OVX",
        "Com_Oil_Spread",
        "BS_Core_Index_A",
        "BS_Core_Index_B",
        "BS_Core_Index_C",
        "Com_LMEX",
        "Com_BloombergCommodity_BCOM",
        "GPRD_THREAT",
        "GPRD",
        "GPRD_ACT",
    ]
