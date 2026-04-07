from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import runtime_support.runner as runtime
import yaml

from app_config import load_app_config
from plugins.aa_forecast.runtime import _aa_params_override
from runtime_support.forecast_models import build_model


FIXED_CONFIG = Path("tests/fixtures/aa_forecast_runtime_smoke.yaml")
AUTO_CONFIG = Path("tests/fixtures/aa_forecast_runtime_auto_smoke.yaml")
AUTO_MODEL_ONLY_CONFIG = Path("tests/fixtures/aa_forecast_runtime_auto_model_only_smoke.yaml")
PLUGIN_AUTO_MODEL_ONLY_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_auto_model_only_main.yaml"
)
PLUGIN_BEST_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_best_main.yaml"
)
PLUGIN_UNCERTAINTY_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_uncertainty_main.yaml"
)
DIRECT_BRNTOIL_CASE1_CONFIG = Path(
    "yaml/experiment/feature_set_aaforecast/brentoil-case1.yaml"
)
LEGACY_ANOMALY_THRESHOLD_PLUGIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_legacy_anomaly_threshold.yaml"
)
FEATURE_SET_AAFORECAST_VARIANTS = {
    "all10": {
        "config_path": Path(
            "yaml/experiment/feature_set_aaforecast/brentoil-case1-best-all10.yaml"
        ),
        "plugin_path": "yaml/plugins/aa_forecast_brentoil_case1_best_all10.yaml",
        "task_name": "brentoil_case1_aaforecast_best_all10",
        "hist_exog_cols": [
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
        ],
        "star_anomaly_tails": {
            "upward": ["GPRD_THREAT", "GPRD", "GPRD_ACT"],
            "two_sided": [],
        },
        "non_star_hist_exog_cols": [
            "Idx_OVX",
            "Com_Oil_Spread",
            "BS_Core_Index_A",
            "BS_Core_Index_B",
            "BS_Core_Index_C",
            "Com_LMEX",
            "Com_BloombergCommodity_BCOM",
        ],
    },
    "no_bs_core": {
        "config_path": Path(
            "yaml/experiment/feature_set_aaforecast/brentoil-case1-best-no_bs_core.yaml"
        ),
        "plugin_path": "yaml/plugins/aa_forecast_brentoil_case1_best_no_bs_core.yaml",
        "task_name": "brentoil_case1_aaforecast_best_no_bs_core",
        "hist_exog_cols": [
            "Idx_OVX",
            "Com_Oil_Spread",
            "Com_LMEX",
            "Com_BloombergCommodity_BCOM",
            "GPRD_THREAT",
            "GPRD",
            "GPRD_ACT",
        ],
        "star_anomaly_tails": {
            "upward": ["GPRD_THREAT", "GPRD", "GPRD_ACT"],
            "two_sided": [],
        },
        "non_star_hist_exog_cols": [
            "Idx_OVX",
            "Com_Oil_Spread",
            "Com_LMEX",
            "Com_BloombergCommodity_BCOM",
        ],
    },
    "no_gprd": {
        "config_path": Path(
            "yaml/experiment/feature_set_aaforecast/brentoil-case1-best-no_gprd.yaml"
        ),
        "plugin_path": "yaml/plugins/aa_forecast_brentoil_case1_best_no_gprd.yaml",
        "task_name": "brentoil_case1_aaforecast_best_no_gprd",
        "hist_exog_cols": [
            "Idx_OVX",
            "Com_Oil_Spread",
            "BS_Core_Index_A",
            "BS_Core_Index_B",
            "BS_Core_Index_C",
            "Com_LMEX",
            "Com_BloombergCommodity_BCOM",
        ],
        "star_anomaly_tails": {
            "upward": [],
            "two_sided": [
                "BS_Core_Index_A",
                "BS_Core_Index_B",
                "BS_Core_Index_C",
            ],
        },
        "non_star_hist_exog_cols": [
            "Idx_OVX",
            "Com_Oil_Spread",
            "Com_LMEX",
            "Com_BloombergCommodity_BCOM",
        ],
    },
}
DIRECT_TARGET_CONFIG = Path("yaml/experiment/feature_set_aaforecast/brentoil-case1.yaml")


def _assert_no_event_column(payload: dict[str, object]) -> None:
    assert "event_column" not in payload


def _assert_grouping_payload(
    payload: dict[str, object],
    *,
    config_path: str | None,
    star_anomaly_tails: dict[str, list[str]],
    non_star_hist_exog_cols: list[str],
) -> None:
    assert payload["config_path"] == config_path
    assert payload["star_anomaly_tails"] == star_anomaly_tails
    assert payload["non_star_hist_exog_cols_resolved"] == non_star_hist_exog_cols
    assert "compatibility_mode" not in payload
    assert "compatibility_source_path" not in payload
    _assert_no_event_column(payload)


def _build_aaforecast_plugin_model(*, training_scaler_type: str | None):
    repo_root = Path.cwd()
    payload = yaml.safe_load((repo_root / FIXED_CONFIG).read_text(encoding="utf-8"))
    if training_scaler_type is None:
        payload.get("training", {}).pop("scaler_type", None)
    else:
        payload.setdefault("training", {})["scaler_type"] = training_scaler_type
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        loaded = load_app_config(repo_root, config_path=config_path)
    job = loaded.config.jobs[0]
    model = build_model(
        loaded.config,
        job,
        n_series=1,
        params_override=_aa_params_override(loaded),
    )
    return loaded, model


def test_aaforecast_plugin_ignores_shared_robust_scaler() -> None:
    loaded, model = _build_aaforecast_plugin_model(training_scaler_type="robust")

    assert loaded.config.training.scaler_type == "robust"
    assert model.hparams.scaler_type is None


def test_aaforecast_plugin_preserves_non_robust_shared_scaler() -> None:
    loaded, model = _build_aaforecast_plugin_model(training_scaler_type="standard")

    assert loaded.config.training.scaler_type == "standard"
    assert model.hparams.scaler_type == "standard"


def test_aaforecast_plugin_uses_shared_setting_adamw_optimizer() -> None:
    repo_root = Path.cwd()
    loaded = load_app_config(repo_root, config_path=str(repo_root / FIXED_CONFIG))
    job = loaded.config.jobs[0]

    model = build_model(
        loaded.config,
        job,
        n_series=1,
        params_override=_aa_params_override(loaded),
    )

    assert loaded.config.training.optimizer.name == "adamw"
    assert model.optimizer is torch.optim.AdamW
    assert model.optimizer_kwargs == {}
    assert isinstance(model.configure_optimizers()["optimizer"], torch.optim.AdamW)


def test_runtime_validate_only_accepts_aaforecast_fixed_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    output_root = tmp_path / "validate-only-aa-forecast-fixed"
    code = runtime.main(
        [
            "--config",
            str(FIXED_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"][0]["model"] == "AAForecast"
    assert manifest["jobs"][0]["requested_mode"] == "learned_fixed"
    assert manifest["jobs"][0]["validated_mode"] == "learned_fixed"
    assert manifest["jobs"][0]["selected_search_params"] == []


def test_runtime_validate_only_accepts_aaforecast_auto_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    output_root = tmp_path / "validate-only-aa-forecast-auto"
    code = runtime.main(
        [
            "--config",
            str(AUTO_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"][0]["model"] == "AAForecast"
    assert manifest["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert manifest["jobs"][0]["validated_mode"] == "learned_auto"
    assert manifest["jobs"][0]["selected_search_params"] == [
        "encoder_hidden_size",
        "encoder_n_layers",
        "encoder_dropout",
        "decoder_hidden_size",
        "decoder_layers",
        "season_length",
        "trend_kernel_size",
        "top_k",
    ]


def test_runtime_validate_only_accepts_aaforecast_auto_model_only_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    output_root = tmp_path / "validate-only-aa-forecast-auto-model-only"
    code = runtime.main(
        [
            "--config",
            str(AUTO_MODEL_ONLY_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"][0]["model"] == "AAForecast"
    assert manifest["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert manifest["jobs"][0]["validated_mode"] == "learned_auto"
    assert manifest["jobs"][0]["selected_search_params"] == [
        "encoder_hidden_size",
        "encoder_n_layers",
        "encoder_dropout",
        "decoder_hidden_size",
        "decoder_layers",
        "season_length",
        "trend_kernel_size",
        "top_k",
    ]
    assert manifest["training_search"] == {
        "requested_mode": "training_fixed",
        "validated_mode": "training_fixed",
        "selected_search_params": [],
    }


def test_runtime_validate_only_accepts_aaforecast_plugin_auto_model_only_path(
    tmp_path: Path,
):
    output_root = tmp_path / "validate-only-aa-forecast-plugin-auto-model-only"
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_AUTO_MODEL_ONLY_MAIN_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"][0]["model"] == "AAForecast"
    assert manifest["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert manifest["jobs"][0]["validated_mode"] == "learned_auto"
    assert manifest["jobs"][0]["selected_search_params"] == [
        "encoder_hidden_size",
        "encoder_n_layers",
        "encoder_dropout",
        "decoder_hidden_size",
        "decoder_layers",
        "season_length",
        "trend_kernel_size",
        "top_k",
    ]
    assert manifest["training_search"] == {
        "requested_mode": "training_fixed",
        "validated_mode": "training_fixed",
        "selected_search_params": [],
    }
    _assert_grouping_payload(
        manifest["aa_forecast"],
        config_path="tests/fixtures/aa_forecast_runtime_plugin_auto_model_only.yaml",
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
    )
    assert manifest["aa_forecast"]["selected_config_path"].endswith(
        "tests/fixtures/aa_forecast_runtime_plugin_auto_model_only.yaml"
    )
    assert manifest["aa_forecast"]["uncertainty"]["enabled"] is False


def test_runtime_validate_only_accepts_aaforecast_plugin_best_path(
    tmp_path: Path,
):
    output_root = tmp_path / "validate-only-aa-forecast-plugin-best"
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_BEST_MAIN_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"][0]["model"] == "AAForecast"
    assert manifest["jobs"][0]["requested_mode"] == "learned_fixed"
    assert manifest["jobs"][0]["validated_mode"] == "learned_fixed"
    assert manifest["jobs"][0]["selected_search_params"] == []
    assert manifest["training_search"] == {
        "requested_mode": "training_fixed",
        "validated_mode": "training_fixed",
        "selected_search_params": [],
    }
    _assert_grouping_payload(
        manifest["aa_forecast"],
        config_path="tests/fixtures/aa_forecast_runtime_plugin_best.yaml",
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
    )
    assert manifest["aa_forecast"]["selected_config_path"].endswith(
        "tests/fixtures/aa_forecast_runtime_plugin_best.yaml"
    )
    assert manifest["aa_forecast"]["uncertainty"]["enabled"] is True


@pytest.mark.parametrize(
    ("variant", "expected"),
    list(FEATURE_SET_AAFORECAST_VARIANTS.items()),
)
def test_feature_set_aaforecast_best_variants_validate_only(
    tmp_path: Path,
    variant: str,
    expected: dict[str, object],
) -> None:
    output_root = tmp_path / f"validate-only-aa-forecast-{variant}"
    code = runtime.main(
        [
            "--config",
            str(expected["config_path"]),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert resolved["task"]["name"] == expected["task_name"]
    assert resolved["dataset"]["hist_exog_cols"] == expected["hist_exog_cols"]
    _assert_grouping_payload(
        resolved["aa_forecast"],
        config_path=expected["plugin_path"],
        star_anomaly_tails=expected["star_anomaly_tails"],
        non_star_hist_exog_cols=expected["non_star_hist_exog_cols"],
    )
    _assert_grouping_payload(
        manifest["aa_forecast"],
        config_path=expected["plugin_path"],
        star_anomaly_tails=expected["star_anomaly_tails"],
        non_star_hist_exog_cols=expected["non_star_hist_exog_cols"],
    )


def test_feature_set_aaforecast_best_validate_only(
    tmp_path: Path,
) -> None:
    config_path = Path("yaml/experiment/feature_set_aaforecast/brentoil-case1-best.yaml")
    output_root = tmp_path / "validate-only-aa-forecast-best"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert resolved["dataset"]["hist_exog_cols"] == [
        "Idx_OVX",
        "Com_Oil_Spread",
        "BS_Core_Index_A",
        "BS_Core_Index_B",
        "BS_Core_Index_C",
        "Com_LMEX",
        "Com_BloombergCommodity_BCOM",
    ]
    for payload in (resolved["aa_forecast"], manifest["aa_forecast"]):
        _assert_grouping_payload(
            payload,
            config_path="yaml/plugins/aa_forecast_brentoil_case1_best.yaml",
            star_anomaly_tails={
                "upward": [],
                "two_sided": [
                    "BS_Core_Index_A",
                    "BS_Core_Index_B",
                    "BS_Core_Index_C",
                ],
            },
            non_star_hist_exog_cols=[
                "Idx_OVX",
                "Com_Oil_Spread",
                "Com_LMEX",
                "Com_BloombergCommodity_BCOM",
            ],
        )


def test_validate_only_brentoil_case1_accepts_plugin_grouped_tails(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "validate-only-aa-forecast-direct-brentoil-case1"
    code = runtime.main(
        [
            "--config",
            str(DIRECT_BRNTOIL_CASE1_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    expected_tails = {
        "upward": ["GPRD_THREAT", "GPRD", "GPRD_ACT"],
        "two_sided": [],
    }
    expected_non_star = [
        "Idx_OVX",
        "Com_Oil_Spread",
        "BS_Core_Index_A",
        "BS_Core_Index_B",
        "BS_Core_Index_C",
        "Com_LMEX",
        "Com_BloombergCommodity_BCOM",
    ]
    for payload in (resolved["aa_forecast"], manifest["aa_forecast"]):
        _assert_grouping_payload(
            payload,
            config_path="yaml/plugins/aa_forecast_brentoil_case1.yaml",
            star_anomaly_tails=expected_tails,
            non_star_hist_exog_cols=expected_non_star,
        )


def test_validate_only_aaforecast_legacy_plugin_anomaly_threshold_fails_fast(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(PLUGIN_BEST_MAIN_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (PLUGIN_BEST_MAIN_CONFIG.parent / payload["dataset"]["path"]).resolve()
    )
    payload["aa_forecast"] = {
        "enabled": True,
        "config_path": str(LEGACY_ANOMALY_THRESHOLD_PLUGIN_CONFIG),
    }
    config_path = tmp_path / "legacy-anomaly-threshold-main.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="anomaly_threshold"):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "legacy-anomaly-threshold-out"),
                "--validate-only",
            ]
        )


def test_validate_only_aaforecast_grouped_tail_missing_dataset_var_fails_fast(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str((AUTO_CONFIG.parent / payload["dataset"]["path"]).resolve())
    payload["dataset"]["hist_exog_cols"] = []
    plugin_payload = yaml.safe_load(
        (AUTO_CONFIG.parent / "aa_forecast_runtime_plugin_auto_model_only.yaml").read_text(
            encoding="utf-8"
        )
    )
    plugin_payload["aa_forecast"]["star_anomaly_tails"] = {
        "upward": ["event"],
        "two_sided": [],
    }
    plugin_path = tmp_path / "missing-grouped-var-plugin.yaml"
    plugin_path.write_text(yaml.safe_dump(plugin_payload, sort_keys=False), encoding="utf-8")
    payload["aa_forecast"]["config_path"] = str(plugin_path)
    config_path = tmp_path / "missing-grouped-var.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"event|hist_exog_cols|star_anomaly_tails"):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "missing-grouped-var-out"),
                "--validate-only",
            ]
        )


def test_validate_only_aaforecast_legacy_inline_without_star_grouping_fails_fast(
    tmp_path: Path,
) -> None:
    payload = {
        "task": {"name": "validate_only_aa_forecast_legacy_inline_fail_fast"},
        "dataset": {
            "path": str((AUTO_CONFIG.parent / "aa_forecast_runtime_smoke.csv").resolve()),
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["event"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "runtime": {"random_seed": 7},
        "training": {
            "input_size": 8,
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
            "n_windows": 1,
            "gap": 0,
            "overlap_eval_policy": "by_cutoff_mean",
        },
        "scheduler": {
            "gpu_ids": [0],
            "max_concurrent_jobs": 1,
            "worker_devices": 1,
        },
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "jobs": [{"model": "AAForecast", "params": {}}],
    }
    config_path = tmp_path / "legacy_inline.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"Direct top-level AAForecast jobs are no longer supported",
    ):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "legacy-inline-out"),
                "--validate-only",
            ]
        )


def test_runtime_smoke_emits_aaforecast_uncertainty_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    payload = yaml.safe_load(PLUGIN_BEST_MAIN_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (PLUGIN_BEST_MAIN_CONFIG.parent / payload["dataset"]["path"]).resolve()
    )
    payload["training"]["input_size"] = 2
    payload["training"]["batch_size"] = 1
    payload["training"]["valid_batch_size"] = 1
    payload["training"]["windows_batch_size"] = 2
    payload["training"]["inference_windows_batch_size"] = 2
    payload["training"]["max_steps"] = 1
    payload["training"]["val_size"] = 2
    payload["cv"]["horizon"] = 2
    payload["cv"]["n_windows"] = 1
    payload["cv"]["step_size"] = 1

    config_path = tmp_path / "aa_forecast_runtime_plugin_best_smoke.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "aa-forecast-runtime-smoke"
    code = runtime.main([
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    ])

    assert code == 0
    uncertainty_dir = output_root / "aa_forecast" / "uncertainty"
    json_files = sorted(uncertainty_dir.glob("*.json"))
    csv_files = sorted(uncertainty_dir.glob("*.csv"))
    assert json_files
    assert csv_files
    summary = json.loads(json_files[0].read_text())
    assert summary["sample_count"] == 3
    assert summary["dropout_candidates"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert summary["star_anomaly_tails"] == {"upward": ["event"], "two_sided": []}
    assert summary["non_star_hist_exog_cols_resolved"] == []
    _assert_no_event_column(summary)
    assert len(summary["selected_dropout_by_horizon"]) == 2
    assert len(summary["selected_std_by_horizon"]) == 2


def test_validate_only_aaforecast_multi_study_catalog_and_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    payload = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (AUTO_CONFIG.parent / payload["dataset"]["path"]).resolve()
    )
    plugin_payload = yaml.safe_load(
        (AUTO_CONFIG.parent / "aa_forecast_runtime_plugin_auto_model_only.yaml").read_text(
            encoding="utf-8"
        )
    )
    plugin_path = tmp_path / "aaforecast_multi_plugin.yaml"
    plugin_path.write_text(yaml.safe_dump(plugin_payload, sort_keys=False), encoding="utf-8")
    payload["jobs"] = []
    payload["aa_forecast"] = {
        "enabled": True,
        "config_path": str(plugin_path),
    }
    payload.setdefault("runtime", {})["opt_study_count"] = 2
    payload["dataset"]["path"] = str((AUTO_CONFIG.parent / payload["dataset"]["path"]).resolve())
    config_path = tmp_path / "aaforecast_multi.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    output_root = tmp_path / "validate-only-aa-forecast-multi"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    study_catalog = json.loads(
        (output_root / "models" / "AAForecast" / "study_catalog.json").read_text()
    )
    assert manifest["optuna"]["study_count"] == 2
    assert manifest["optuna"]["selected_study_index"] is None
    assert manifest["optuna"]["canonical_projection_study_index"] == 1
    assert study_catalog["study_count"] == 2
    assert study_catalog["canonical_projection_study_index"] == 1
    assert (output_root / "models" / "AAForecast" / "visualizations" / "cross_study_dashboard.html").exists()

    selected_output_root = tmp_path / "validate-only-aa-forecast-multi-selected"
    selected_code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(selected_output_root),
            "--validate-only",
            "--optuna-study",
            "2",
        ]
    )

    assert selected_code == 0
    selected_manifest = json.loads(
        (selected_output_root / "manifest" / "run_manifest.json").read_text()
    )
    selected_catalog = json.loads(
        (selected_output_root / "models" / "AAForecast" / "study_catalog.json").read_text()
    )
    assert selected_manifest["optuna"]["selected_study_index"] == 2
    assert selected_manifest["optuna"]["canonical_projection_study_index"] == 2
    assert selected_catalog["selected_study_index"] == 2
    assert selected_catalog["canonical_projection_study_index"] == 2
    _assert_grouping_payload(
        selected_manifest["aa_forecast"],
        config_path=str(plugin_path),
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
    )
    assert (selected_output_root / "models" / "AAForecast" / "visualizations" / "cross_study_dashboard.html").exists()


def test_runtime_aaforecast_plugin_uncertainty_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    output_root = tmp_path / "smoke-aa-forecast-plugin-uncertainty"
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_UNCERTAINTY_MAIN_CONFIG),
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    uncertainty_dir = output_root / "aa_forecast" / "uncertainty"
    distribution_files = sorted(uncertainty_dir.glob("*.json"))
    assert distribution_files
    payload = json.loads(distribution_files[0].read_text())
    assert payload["dropout_candidates"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert payload["star_anomaly_tails"] == {"upward": ["event"], "two_sided": []}
    assert payload["non_star_hist_exog_cols_resolved"] == []
    _assert_no_event_column(payload)
    assert len(payload["selected_dropout_by_horizon"]) == 1
    assert len(payload["selected_std_by_horizon"]) == 1


def test_validate_only_rejects_yaml_managed_aaforecast_dropout_candidates(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str((AUTO_CONFIG.parent / payload["dataset"]["path"]).resolve())
    plugin_payload = yaml.safe_load(
        (AUTO_CONFIG.parent / "aa_forecast_runtime_plugin_auto_model_only.yaml").read_text(
            encoding="utf-8"
        )
    )
    plugin_payload["aa_forecast"]["uncertainty"]["dropout_candidates"] = [0.1, 0.2, 0.3]
    plugin_path = tmp_path / "aaforecast-invalid-dropout-plugin.yaml"
    plugin_path.write_text(yaml.safe_dump(plugin_payload, sort_keys=False), encoding="utf-8")
    payload["aa_forecast"]["config_path"] = str(plugin_path)
    config_path = tmp_path / "aaforecast-invalid-dropout-candidates.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported key\(s\): dropout_candidates"):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "invalid-dropout-candidates"),
                "--validate-only",
            ]
        )


def test_runtime_validate_only_plugin_target_preserves_grouped_tails(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "validate-only-aa-forecast-direct-target"
    code = runtime.main(
        [
            "--config",
            str(DIRECT_TARGET_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    capability = json.loads(
        (output_root / "config" / "capability_report.json").read_text()
    )
    expected_tails = {
        "upward": ["GPRD_THREAT", "GPRD", "GPRD_ACT"],
        "two_sided": [],
    }
    expected_non_star = [
        "Idx_OVX",
        "Com_Oil_Spread",
        "BS_Core_Index_A",
        "BS_Core_Index_B",
        "BS_Core_Index_C",
        "Com_LMEX",
        "Com_BloombergCommodity_BCOM",
    ]
    for payload in (resolved["aa_forecast"], manifest["aa_forecast"], capability["aa_forecast"]):
        _assert_grouping_payload(
            payload,
            config_path="yaml/plugins/aa_forecast_brentoil_case1.yaml",
            star_anomaly_tails=expected_tails,
            non_star_hist_exog_cols=expected_non_star,
        )
