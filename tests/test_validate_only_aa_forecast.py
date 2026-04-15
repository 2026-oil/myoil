from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import runtime_support.runner as runtime
import yaml

from app_config import load_app_config
import plugins.aa_forecast.runtime as aa_runtime
from plugins.aa_forecast.runtime import _aa_params_override
from runtime_support.forecast_models import build_model


FIXED_CONFIG = Path("tests/fixtures/aa_forecast_runtime_smoke.yaml")
AUTO_CONFIG = Path("tests/fixtures/aa_forecast_runtime_auto_smoke.yaml")
AUTO_MODEL_ONLY_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_auto_model_only_smoke.yaml"
)
PLUGIN_AUTO_MODEL_ONLY_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_auto_model_only_main.yaml"
)
PLUGIN_TIMEXER_AUTO_MODEL_ONLY_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_timexer_auto_model_only_main.yaml"
)
PLUGIN_TIMEXER_FIXED_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_timexer_fixed_main.yaml"
)
PLUGIN_ITRANSFORMER_AUTO_MODEL_ONLY_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_itransformer_auto_model_only_main.yaml"
)
PLUGIN_ITRANSFORMER_FIXED_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_itransformer_fixed_main.yaml"
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
PARITY_SHARED_MODEL_SELECTORS = [
    "encoder_hidden_size",
    "encoder_n_layers",
    "encoder_dropout",
    "decoder_hidden_size",
    "decoder_layers",
]
TIMEXER_AA_BACKBONE_SELECTORS = [
    "hidden_size",
    "n_heads",
    "e_layers",
    "dropout",
    "d_ff",
    "patch_len",
    "use_norm",
    "decoder_hidden_size",
    "decoder_layers",
]
ITRANSFORMER_AA_BACKBONE_SELECTORS = [
    "hidden_size",
    "n_heads",
    "e_layers",
    "dropout",
    "d_ff",
    "factor",
    "use_norm",
    "decoder_hidden_size",
    "decoder_layers",
]
TIMEXER_AA_TRAINING_SELECTORS = [
    "input_size",
    "batch_size",
    "scaler_type",
    "model_step_size",
]
ITRANSFORMER_AA_TRAINING_SELECTORS = [
    "input_size",
    "batch_size",
    "scaler_type",
    "model_step_size",
]
FEATURE_SET_AAFORECAST_VARIANTS = {
    "all10": {
        "config_path": Path(
            "yaml/experiment/feature_set_aaforecast/brentoil-case1-best-all10.yaml"
        ),
        "plugin_path": "yaml/plugins/aa_forecsat/aa_forecast_best_all10.yaml",
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
        "plugin_path": "yaml/plugins/aa_forecsat/aa_forecast_best_no_bs_core.yaml",
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
        "plugin_path": "yaml/plugins/aa_forecsat/aa_forecast_best_no_gprd.yaml",
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
EXPECTED_AA_DROPOUT_CANDIDATES = [round(step * 0.05, 2) for step in range(1, 20)]
DIRECT_TARGET_CONFIG = Path(
    "yaml/experiment/feature_set_aaforecast/brentoil-case1.yaml"
)


def _assert_no_event_column(payload: dict[str, object]) -> None:
    assert "event_column" not in payload


def _assert_grouping_payload(
    payload: dict[str, object],
    *,
    config_path: str | None,
    star_anomaly_tails: dict[str, list[str]],
    non_star_hist_exog_cols: list[str],
    model: str = "gru",
) -> None:
    assert payload["config_path"] == config_path
    assert payload["model"] == model
    assert payload["backbone"] == model
    assert payload["star_anomaly_tails"] == star_anomaly_tails
    assert payload["non_star_hist_exog_cols_resolved"] == non_star_hist_exog_cols
    assert "mode" not in payload
    assert "compatibility_mode" not in payload
    assert "compatibility_source_path" not in payload
    _assert_no_event_column(payload)


def _assert_retrieval_payload(
    payload: dict[str, object],
    *,
    enabled: bool,
    top_k: int,
    recency_gap_steps: int,
    event_score_threshold: float,
    trigger_quantile: float | None,
    neighbor_min_event_ratio: float,
    min_similarity: float,
    blend_floor: float,
    blend_max: float,
    use_uncertainty_gate: bool,
) -> None:
    assert payload["enabled"] is enabled
    assert payload["top_k"] == top_k
    assert payload["recency_gap_steps"] == recency_gap_steps
    assert payload["event_score_threshold"] == pytest.approx(event_score_threshold)
    assert payload["trigger_quantile"] == trigger_quantile
    assert payload["neighbor_min_event_ratio"] == pytest.approx(
        neighbor_min_event_ratio
    )
    assert payload["min_similarity"] == pytest.approx(min_similarity)
    assert payload["blend_floor"] == pytest.approx(blend_floor)
    assert payload["blend_max"] == pytest.approx(blend_max)
    assert payload["use_uncertainty_gate"] is use_uncertainty_gate
    assert payload["mode"] == "posthoc_blend"
    assert payload["similarity"] == "cosine"
    assert payload["temperature"] == pytest.approx(0.1)
    assert payload["memory_value_mode"] == "future_return"
    assert payload["use_shape_key"] is True
    assert payload["use_event_key"] is True


def _read_optional_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def test_aaforecast_plugin_forces_null_scaler_for_any_shared_scaler() -> None:
    loaded, model = _build_aaforecast_plugin_model(training_scaler_type="standard")

    assert loaded.config.training.scaler_type == "standard"
    assert model.hparams.scaler_type is None


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


def test_runtime_validate_only_accepts_aaforecast_fixed_path_with_shared_diff_setting(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    shared_settings_path = tmp_path / "setting.yaml"
    shared_settings_path.write_text(
        yaml.safe_dump(
            {"runtime": {"transformations_target": "diff"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_app_config(
        Path.cwd(),
        config_path=str(FIXED_CONFIG),
        shared_settings_path=shared_settings_path,
    )

    assert loaded.config.runtime.transformations_target == "diff"

    output_root = tmp_path / "validate-only-aa-forecast-fixed-shared-diff"
    code = runtime.main(
        [
            "--config",
            str(FIXED_CONFIG),
            "--setting",
            str(shared_settings_path),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert manifest["jobs"][0]["model"] == "AAForecast"
    assert manifest["shared_settings_path"] == str(shared_settings_path.resolve())


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
    assert (
        manifest["jobs"][0]["selected_search_params"] == PARITY_SHARED_MODEL_SELECTORS
    )


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
    assert (
        manifest["jobs"][0]["selected_search_params"] == PARITY_SHARED_MODEL_SELECTORS
    )
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
    assert (
        manifest["jobs"][0]["selected_search_params"] == PARITY_SHARED_MODEL_SELECTORS
    )
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
    stage_config = json.loads(
        (output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )
    assert stage_config["model"] == "gru"
    assert stage_config["backbone"] == "gru"
    assert "mode" not in stage_config


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
    stage_config = json.loads(
        (output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )
    assert stage_config["model"] == "gru"
    assert stage_config["backbone"] == "gru"
    assert "mode" not in stage_config


def test_runtime_validate_only_accepts_aaforecast_plugin_uncertainty_retrieval_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "validate-only-aa-forecast-plugin-uncertainty-retrieval"
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_UNCERTAINTY_MAIN_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    _assert_grouping_payload(
        manifest["aa_forecast"],
        config_path="tests/fixtures/aa_forecast_runtime_plugin_uncertainty.yaml",
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
    )
    assert manifest["aa_forecast"]["uncertainty"]["enabled"] is True
    assert (
        manifest["aa_forecast"]["retrieval"]["config_path"]
        == "aa_forecast_retrieval_detail.yaml"
    )
    _assert_retrieval_payload(
        manifest["aa_forecast"]["retrieval"],
        enabled=True,
        top_k=2,
        recency_gap_steps=1,
        event_score_threshold=100.0,
        trigger_quantile=None,
        neighbor_min_event_ratio=0.0,
        min_similarity=0.55,
        blend_floor=0.0,
        blend_max=0.2,
        use_uncertainty_gate=True,
    )

    stage_config = json.loads(
        (output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )
    assert stage_config["retrieval"]["config_path"] == "aa_forecast_retrieval_detail.yaml"
    _assert_retrieval_payload(
        stage_config["retrieval"],
        enabled=True,
        top_k=2,
        recency_gap_steps=1,
        event_score_threshold=100.0,
        trigger_quantile=None,
        neighbor_min_event_ratio=0.0,
        min_similarity=0.55,
        blend_floor=0.0,
        blend_max=0.2,
        use_uncertainty_gate=True,
    )


def test_runtime_validate_only_accepts_aaforecast_plugin_timexer_auto_model_only_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "validate-only-aa-forecast-plugin-timexer-auto-model-only"
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_TIMEXER_AUTO_MODEL_ONLY_MAIN_CONFIG),
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
    assert (
        manifest["jobs"][0]["selected_search_params"] == TIMEXER_AA_BACKBONE_SELECTORS
    )
    assert manifest["training_search"] == {
        "requested_mode": "training_auto_requested",
        "validated_mode": "training_auto",
        "selected_search_params": TIMEXER_AA_TRAINING_SELECTORS,
    }
    _assert_grouping_payload(
        manifest["aa_forecast"],
        config_path="tests/fixtures/aa_forecast_runtime_plugin_timexer_auto_model_only.yaml",
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
        model="timexer",
    )
    stage_config = json.loads(
        (output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )
    assert stage_config["model"] == "timexer"
    assert stage_config["backbone"] == "timexer"
    assert "mode" not in stage_config


def test_runtime_validate_only_accepts_aaforecast_plugin_timexer_fixed_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "validate-only-aa-forecast-plugin-timexer-fixed"
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_TIMEXER_FIXED_MAIN_CONFIG),
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
        config_path="tests/fixtures/aa_forecast_runtime_plugin_timexer_fixed.yaml",
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
        model="timexer",
    )
    stage_config = json.loads(
        (output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )
    assert stage_config["model"] == "timexer"
    assert stage_config["backbone"] == "timexer"
    assert "mode" not in stage_config


def test_runtime_validate_only_accepts_aaforecast_plugin_itransformer_auto_model_only_path(
    tmp_path: Path,
) -> None:
    output_root = (
        tmp_path / "validate-only-aa-forecast-plugin-itransformer-auto-model-only"
    )
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_ITRANSFORMER_AUTO_MODEL_ONLY_MAIN_CONFIG),
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
    assert (
        manifest["jobs"][0]["selected_search_params"]
        == ITRANSFORMER_AA_BACKBONE_SELECTORS
    )
    assert manifest["training_search"] == {
        "requested_mode": "training_auto_requested",
        "validated_mode": "training_auto",
        "selected_search_params": ITRANSFORMER_AA_TRAINING_SELECTORS,
    }
    _assert_grouping_payload(
        manifest["aa_forecast"],
        config_path="tests/fixtures/aa_forecast_runtime_plugin_itransformer_auto_model_only.yaml",
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
        model="itransformer",
    )
    stage_config = json.loads(
        (output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )
    assert stage_config["model"] == "itransformer"
    assert stage_config["backbone"] == "itransformer"
    assert "mode" not in stage_config


def test_runtime_validate_only_accepts_aaforecast_plugin_itransformer_fixed_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "validate-only-aa-forecast-plugin-itransformer-fixed"
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_ITRANSFORMER_FIXED_MAIN_CONFIG),
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
        config_path="tests/fixtures/aa_forecast_runtime_plugin_itransformer_fixed.yaml",
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
        model="itransformer",
    )
    stage_config = json.loads(
        (output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )
    assert stage_config["model"] == "itransformer"
    assert stage_config["backbone"] == "itransformer"
    assert "mode" not in stage_config


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
    config_path = Path(
        "yaml/experiment/feature_set_aaforecast/brentoil-case1-best.yaml"
    )
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
            config_path="yaml/plugins/aa_forecsat/aa_forecast_best.yaml",
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


def test_validate_only_accepts_plugin_grouped_tails(
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
            config_path="yaml/plugins/aa_forecast.yaml",
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


def test_validate_only_aaforecast_legacy_plugin_top_k_fails_fast(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(PLUGIN_BEST_MAIN_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (PLUGIN_BEST_MAIN_CONFIG.parent / payload["dataset"]["path"]).resolve()
    )
    plugin_payload = yaml.safe_load(
        (
            PLUGIN_BEST_MAIN_CONFIG.parent / "aa_forecast_runtime_plugin_best.yaml"
        ).read_text(encoding="utf-8")
    )
    plugin_payload["aa_forecast"]["top_k"] = 0.1
    plugin_payload["aa_forecast"].pop("thresh", None)
    plugin_path = tmp_path / "legacy-top-k-plugin.yaml"
    plugin_path.write_text(
        yaml.safe_dump(plugin_payload, sort_keys=False), encoding="utf-8"
    )
    payload["aa_forecast"] = {
        "enabled": True,
        "config_path": str(plugin_path),
    }
    config_path = tmp_path / "legacy-top-k-main.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"top_k.*thresh"):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "legacy-top-k-out"),
                "--validate-only",
            ]
        )


def test_validate_only_aaforecast_grouped_tail_missing_dataset_var_fails_fast(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (AUTO_CONFIG.parent / payload["dataset"]["path"]).resolve()
    )
    payload["dataset"]["hist_exog_cols"] = []
    plugin_payload = yaml.safe_load(
        (
            AUTO_CONFIG.parent / "aa_forecast_runtime_plugin_auto_model_only.yaml"
        ).read_text(encoding="utf-8")
    )
    plugin_payload["aa_forecast"]["star_anomaly_tails"] = {
        "upward": ["event"],
        "two_sided": [],
    }
    plugin_path = tmp_path / "missing-grouped-var-plugin.yaml"
    plugin_path.write_text(
        yaml.safe_dump(plugin_payload, sort_keys=False), encoding="utf-8"
    )
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
            "path": str(
                (AUTO_CONFIG.parent / "aa_forecast_runtime_smoke.csv").resolve()
            ),
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
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    uncertainty_dir = output_root / "aa_forecast" / "uncertainty"
    json_files = sorted(uncertainty_dir.glob("*.json"))
    csv_files = sorted(uncertainty_dir.glob("*.csv"))
    png_files = sorted(uncertainty_dir.glob("*.dropout_mae_sd.png"))
    distribution_csv_files = sorted(
        uncertainty_dir.glob("*.prediction_distribution_by_dropout.csv")
    )
    distribution_combined_csv_files = sorted(
        uncertainty_dir.glob("*.prediction_distribution_by_dropout.combined.csv")
    )
    distribution_png_files = sorted(
        uncertainty_dir.glob("*.prediction_distribution_by_dropout.png")
    )
    assert json_files
    assert csv_files
    assert png_files
    assert distribution_csv_files
    assert distribution_combined_csv_files
    assert distribution_png_files
    summary = json.loads(json_files[0].read_text())
    distribution_frame = pd.read_csv(distribution_csv_files[0])
    distribution_combined_frame = pd.read_csv(distribution_combined_csv_files[0])
    assert summary["sample_count"] == 3
    assert summary["dropout_candidates"] == sorted(summary["dropout_candidates"])
    assert summary["dropout_candidates"]
    assert summary["star_anomaly_tails"] == {"upward": ["event"], "two_sided": []}
    assert summary["non_star_hist_exog_cols_resolved"] == []
    _assert_no_event_column(summary)
    assert len(summary["selected_dropout_by_horizon"]) == 2
    assert len(summary["selected_std_by_horizon"]) == 2
    assert {
        "dropout_p",
        "horizon_step",
        "count",
        "mean",
        "std",
        "min",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "max",
    }.issubset(distribution_frame.columns)
    assert {
        "dropout_p",
        "count",
        "mean",
        "std",
        "min",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "max",
    }.issubset(distribution_combined_frame.columns)
    assert distribution_frame["dropout_p"].nunique() == len(
        summary["dropout_candidates"]
    )
    assert distribution_combined_frame["dropout_p"].tolist() == pytest.approx(
        summary["dropout_candidates"]
    )


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
        (
            AUTO_CONFIG.parent / "aa_forecast_runtime_plugin_auto_model_only.yaml"
        ).read_text(encoding="utf-8")
    )
    plugin_path = tmp_path / "aaforecast_multi_plugin.yaml"
    plugin_path.write_text(
        yaml.safe_dump(plugin_payload, sort_keys=False), encoding="utf-8"
    )
    payload["jobs"] = []
    payload["aa_forecast"] = {
        "enabled": True,
        "config_path": str(plugin_path),
    }
    payload.setdefault("runtime", {})["opt_study_count"] = 2
    payload["dataset"]["path"] = str(
        (AUTO_CONFIG.parent / payload["dataset"]["path"]).resolve()
    )
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
    assert (
        output_root
        / "models"
        / "AAForecast"
        / "visualizations"
        / "cross_study_dashboard.html"
    ).exists()
    visual_root = output_root / "models" / "AAForecast" / "visualizations"
    artifact_inventory = json.loads((visual_root / "artifact_inventory.json").read_text())
    artifact_paths = {
        Path(item["path"]).name for item in artifact_inventory["artifacts"] if "path" in item
    }
    assert "study_01_trial_metric_history.html" in artifact_paths
    assert "study_02_trial_metric_history.html" in artifact_paths
    assert (visual_root / "study_01_trial_metric_history.html").exists()
    assert (visual_root / "study_02_trial_metric_history.html").exists()

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
        (
            selected_output_root / "models" / "AAForecast" / "study_catalog.json"
        ).read_text()
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
    assert (
        selected_output_root
        / "models"
        / "AAForecast"
        / "visualizations"
        / "cross_study_dashboard.html"
    ).exists()
    selected_visual_root = selected_output_root / "models" / "AAForecast" / "visualizations"
    selected_inventory = json.loads(
        (selected_visual_root / "artifact_inventory.json").read_text()
    )
    selected_artifact_paths = {
        Path(item["path"]).name for item in selected_inventory["artifacts"] if "path" in item
    }
    assert "study_01_trial_metric_history.html" in selected_artifact_paths
    assert "study_02_trial_metric_history.html" in selected_artifact_paths
    assert (selected_visual_root / "study_01_trial_metric_history.html").exists()
    assert (selected_visual_root / "study_02_trial_metric_history.html").exists()


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
    uncertainty_csv_files = sorted(
        path
        for path in uncertainty_dir.glob("*.csv")
        if ".candidate_" not in path.name
        and ".prediction_distribution_by_dropout" not in path.name
    )
    png_files = sorted(uncertainty_dir.glob("*.dropout_mae_sd.png"))
    distribution_csv_files = sorted(
        uncertainty_dir.glob("*.prediction_distribution_by_dropout.csv")
    )
    distribution_combined_csv_files = sorted(
        uncertainty_dir.glob("*.prediction_distribution_by_dropout.combined.csv")
    )
    distribution_png_files = sorted(
        uncertainty_dir.glob("*.prediction_distribution_by_dropout.png")
    )
    retrieval_dir = output_root / "aa_forecast" / "retrieval"
    retrieval_summary_files = sorted(retrieval_dir.glob("*.json"))
    retrieval_neighbor_files = sorted(retrieval_dir.glob("*.neighbors.csv"))
    assert distribution_files
    assert uncertainty_csv_files
    assert png_files
    assert distribution_csv_files
    assert distribution_combined_csv_files
    assert distribution_png_files
    assert retrieval_summary_files
    assert retrieval_neighbor_files
    payload = json.loads(distribution_files[0].read_text())
    uncertainty_frame = pd.read_csv(uncertainty_csv_files[0])
    distribution_frame = pd.read_csv(distribution_csv_files[0])
    distribution_combined_frame = pd.read_csv(distribution_combined_csv_files[0])
    retrieval_payload = json.loads(retrieval_summary_files[0].read_text())
    retrieval_neighbors = _read_optional_csv(retrieval_neighbor_files[0])
    assert payload["dropout_candidates"] == sorted(payload["dropout_candidates"])
    assert payload["dropout_candidates"]
    assert payload["star_anomaly_tails"] == {"upward": ["event"], "two_sided": []}
    assert payload["non_star_hist_exog_cols_resolved"] == []
    _assert_no_event_column(payload)
    assert len(payload["selected_dropout_by_horizon"]) == 1
    assert len(payload["selected_std_by_horizon"]) == 1
    assert retrieval_payload["retrieval_enabled"] is True
    assert retrieval_payload["retrieval_attempted"] is True
    assert retrieval_payload["retrieval_applied"] is False
    assert retrieval_payload["skip_reason"] == "below_event_threshold"
    assert retrieval_payload["top_k_requested"] == 2
    assert retrieval_payload["top_k_used"] == 0
    assert retrieval_payload["blend_max"] == pytest.approx(0.2)
    assert retrieval_payload["base_prediction"] == pytest.approx(
        uncertainty_frame["prediction_mean"].tolist()
    )
    assert retrieval_payload["final_prediction"] == pytest.approx(
        uncertainty_frame["prediction_mean"].tolist()
    )
    assert distribution_frame["horizon_step"].nunique() == 1
    assert distribution_combined_frame["dropout_p"].tolist() == pytest.approx(
        payload["dropout_candidates"]
    )
    assert retrieval_neighbors.empty


def test_runtime_aaforecast_writes_context_annotation_and_sidecar(
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

    config_path = tmp_path / "aa_forecast_context_main.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output_root = tmp_path / "aa-forecast-context-smoke"

    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    forecast_frame = pd.read_csv(output_root / "cv" / "AAForecast_forecasts.csv")
    assert {
        "aaforecast_context_active",
        "aaforecast_context_label",
        "aaforecast_context_artifact",
    }.issubset(forecast_frame.columns)
    assert forecast_frame["aaforecast_context_active"].isna().all()
    assert forecast_frame["aaforecast_context_label"].isna().all()
    artifact_relpath = forecast_frame["aaforecast_context_artifact"].dropna().iloc[0]
    context_path = output_root / artifact_relpath
    assert context_path.exists()
    context_frame = pd.read_csv(context_path)
    result_frame = pd.read_csv(output_root / "summary" / "result.csv")
    assert {"ds", "context_active", "context_label"}.issubset(context_frame.columns)
    assert (output_root / "summary" / "last_fold_all_models.png").exists()
    assert (output_root / "summary" / "last_fold_all_models_window_16.png").exists()
    assert (output_root / "summary" / "result.csv").exists()
    assert not (output_root / "summary" / "sample.md").exists()
    assert not any((output_root / "summary").glob("test_*"))
    assert {"model", "fold_idx", "ds", "y", "y_hat"}.issubset(result_frame.columns)


def test_runtime_aaforecast_trial_artifacts_include_predictions_and_mc_dropout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(
        runtime, "_should_parallelize_single_job_tuning", lambda *_args: False
    )
    monkeypatch.setattr(runtime, "_should_build_summary_artifacts", lambda: False)

    payload = yaml.safe_load(PLUGIN_UNCERTAINTY_MAIN_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (PLUGIN_UNCERTAINTY_MAIN_CONFIG.parent / payload["dataset"]["path"]).resolve()
    )
    payload["training"]["input_size"] = 2
    payload["training"]["batch_size"] = 1
    payload["training"]["valid_batch_size"] = 1
    payload["training"]["windows_batch_size"] = 4
    payload["training"]["inference_windows_batch_size"] = 4
    payload["training"]["max_steps"] = 1
    payload["training"]["val_size"] = 1
    payload["cv"]["horizon"] = 1
    payload["cv"]["n_windows"] = 1
    payload["cv"]["step_size"] = 1
    payload.setdefault("runtime", {})["opt_n_trial"] = 1

    plugin_payload = yaml.safe_load(
        (
            PLUGIN_UNCERTAINTY_MAIN_CONFIG.parent
            / "aa_forecast_runtime_plugin_uncertainty.yaml"
        ).read_text(encoding="utf-8")
    )
    plugin_payload["aa_forecast"]["model_params"] = {}
    plugin_path = tmp_path / "aa_forecast_trial_uncertainty.yaml"
    plugin_path.write_text(
        yaml.safe_dump(plugin_payload, sort_keys=False), encoding="utf-8"
    )
    payload["aa_forecast"]["config_path"] = str(plugin_path)

    config_path = tmp_path / "aa_forecast_trial_uncertainty_main.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    output_root = tmp_path / "aa-forecast-trial-artifacts"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
    )

    trial_root = (
        output_root
        / "models"
        / "AAForecast"
        / "studies"
        / "study-01"
        / "trials"
        / "trial-0000"
    )
    prediction_frame = pd.read_csv(trial_root / "predictions.csv")
    common_fold_root = trial_root / "folds" / "fold_000"
    fold_root = trial_root / "folds" / "fold_000" / "aa_forecast" / "uncertainty"
    retrieval_fold_root = (
        trial_root / "folds" / "fold_000" / "aa_forecast" / "retrieval"
    )
    candidate_stats_files = sorted(fold_root.glob("*.candidate_stats.csv"))
    candidate_sample_files = sorted(fold_root.glob("*.candidate_samples.csv"))
    candidate_plot_files = sorted(fold_root.glob("*.dropout_mae_sd.png"))
    distribution_csv_files = sorted(
        fold_root.glob("*.prediction_distribution_by_dropout.csv")
    )
    distribution_combined_csv_files = sorted(
        fold_root.glob("*.prediction_distribution_by_dropout.combined.csv")
    )
    distribution_plot_files = sorted(
        fold_root.glob("*.prediction_distribution_by_dropout.png")
    )
    retrieval_summary_files = sorted(retrieval_fold_root.glob("*.json"))
    retrieval_neighbor_files = sorted(retrieval_fold_root.glob("*.neighbors.csv"))
    metrics_payload = json.loads(
        (common_fold_root / "metrics.json").read_text(encoding="utf-8")
    )

    assert code == 0
    assert {"y", "y_hat", "y_hat_uncertainty_std", "y_hat_selected_dropout"}.issubset(
        prediction_frame.columns
    )
    assert (common_fold_root / "predictions.csv").exists()
    assert (common_fold_root / "plot.png").exists()
    assert (common_fold_root / "checkpoint.pt").exists()
    assert {"MAE", "MSE", "RMSE", "MAPE", "NRMSE", "R2"}.issubset(metrics_payload)
    assert candidate_stats_files
    assert candidate_sample_files
    assert candidate_plot_files
    assert distribution_csv_files
    assert distribution_combined_csv_files
    assert distribution_plot_files
    assert retrieval_summary_files
    assert retrieval_neighbor_files

    candidate_stats = pd.read_csv(candidate_stats_files[0])
    candidate_samples = pd.read_csv(candidate_sample_files[0])
    distribution_frame = pd.read_csv(distribution_csv_files[0])
    distribution_combined_frame = pd.read_csv(distribution_combined_csv_files[0])
    retrieval_summary = json.loads(retrieval_summary_files[0].read_text())
    retrieval_neighbors = _read_optional_csv(retrieval_neighbor_files[0])
    assert {
        "horizon_step",
        "dropout_p",
        "prediction_mean",
        "prediction_std",
    }.issubset(candidate_stats.columns)
    assert {
        "horizon_step",
        "dropout_p",
        "sample_idx",
        "prediction",
    }.issubset(candidate_samples.columns)
    assert {
        "dropout_p",
        "horizon_step",
        "count",
        "mean",
        "std",
        "min",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "max",
    }.issubset(distribution_frame.columns)
    assert {
        "dropout_p",
        "count",
        "mean",
        "std",
        "min",
        "q05",
        "q25",
        "median",
        "q75",
        "q95",
        "max",
    }.issubset(distribution_combined_frame.columns)
    assert candidate_stats["prediction_std"].gt(0).any()
    assert candidate_samples["prediction"].nunique() > 1
    assert retrieval_summary["retrieval_enabled"] is True
    assert retrieval_summary["retrieval_attempted"] is True
    assert retrieval_summary["retrieval_applied"] is False
    assert retrieval_summary["skip_reason"] == "below_event_threshold"
    assert retrieval_neighbors.empty


def test_build_uncertainty_error_summary_aggregates_dropout_mae_and_sd() -> None:
    summary = aa_runtime._build_uncertainty_error_summary(
        candidate_samples={
            "0.10": [[1.0, 3.0], [3.0, 5.0]],
            "0.20": [[2.0, 4.0], [2.0, 6.0]],
        },
        target_actuals=pd.Series([2.0, 4.0]),
    )

    assert list(summary["dropout_p"]) == [0.1, 0.2]
    assert summary.loc[0, "mae_mean"] == pytest.approx(1.0)
    assert summary.loc[0, "mae_sd"] == pytest.approx(0.0)
    assert summary.loc[1, "mae_mean"] == pytest.approx(0.5)
    assert summary.loc[1, "mae_sd"] == pytest.approx(0.5)


def test_build_uncertainty_prediction_distribution_summary_aggregates_quantiles() -> (
    None
):
    candidate_sample_frame = pd.DataFrame(
        {
            "horizon_step": [1, 1, 1, 2, 2, 2],
            "dropout_p": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "sample_idx": [0, 1, 2, 0, 1, 2],
            "prediction": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        }
    )

    summary = aa_runtime._build_uncertainty_prediction_distribution_summary(
        candidate_sample_frame=candidate_sample_frame
    )
    combined = aa_runtime._build_uncertainty_prediction_distribution_summary(
        candidate_sample_frame=candidate_sample_frame,
        combine_horizons=True,
    )

    assert summary["horizon_step"].tolist() == [1, 2]
    assert summary["count"].tolist() == [3, 3]
    assert summary["mean"].tolist() == pytest.approx([2.0, 11.0])
    assert summary["median"].tolist() == pytest.approx([2.0, 11.0])
    assert summary["q05"].tolist() == pytest.approx([1.1, 10.1])
    assert summary["q95"].tolist() == pytest.approx([2.9, 11.9])
    assert combined["count"].tolist() == [6]
    assert combined["mean"].tolist() == pytest.approx([6.5])
    assert combined["median"].tolist() == pytest.approx([6.5])


def test_select_uncertainty_predictions_uses_distinct_prediction_seeds(
    monkeypatch,
) -> None:
    seen_seeds: list[int] = []

    def fake_predict_with_adapter(_nf, _adapter_inputs, *, random_seed=None):
        seen_seeds.append(int(random_seed))
        return pd.DataFrame(
            {"unique_id": ["series"], "AAForecast": [float(random_seed)]}
        )

    def fake_extract_target_prediction_frame(
        predictions,
        *,
        target_col,
        model_name,
        diff_context,
        restore_target_predictions,
    ):
        del target_col, model_name, diff_context, restore_target_predictions
        return predictions.rename(columns={"AAForecast": "y_hat"})

    monkeypatch.setattr(aa_runtime, "_predict_with_adapter", fake_predict_with_adapter)
    monkeypatch.setattr(
        aa_runtime,
        "_extract_target_prediction_frame",
        fake_extract_target_prediction_frame,
    )

    model = SimpleNamespace(
        random_seed=7,
        configure_stochastic_inference=lambda **_kwargs: None,
    )
    summary = aa_runtime._select_uncertainty_predictions(
        nf=object(),
        adapter_inputs=object(),
        model=model,
        model_name="AAForecast",
        target_col="series",
        diff_context=None,
        restore_target_predictions=None,
        prediction_column="y_hat",
        dropout_candidates=(0.1, 0.2),
        sample_count=3,
    )

    assert seen_seeds == [7, 8, 9, 11, 12, 13]
    assert summary["candidate_samples"]["0.10"] == [[7.0], [8.0], [9.0]]
    assert summary["candidate_samples"]["0.20"] == [[11.0], [12.0], [13.0]]
    assert summary["candidate_std_grid"][0, 0] > 0
    assert summary["candidate_std_grid"][1, 0] > 0
    assert summary["selection_mode"] == "trajectory_min_dispersion"
    assert summary["selected_path_idx"] in (0, 1)
    assert summary["selected_path_score"] >= 0.0


def test_select_uncertainty_predictions_keeps_one_dropout_for_full_trajectory(
    monkeypatch,
) -> None:
    current_dropout = {"value": None}

    def fake_predict_with_adapter(_nf, _adapter_inputs, *, random_seed=None):
        dropout_p = current_dropout["value"]
        if dropout_p == 0.1:
            lookup = {
                7: [10.0, 11.0],
                8: [10.1, 11.6],
                9: [9.9, 10.4],
            }
        else:
            lookup = {
                11: [9.0, 9.4],
                12: [9.2, 9.5],
                13: [8.8, 9.6],
            }
        values = lookup[int(random_seed)]
        return pd.DataFrame(
            {
                "unique_id": ["series", "series"],
                "AAForecast": values,
            }
        )

    def fake_extract_target_prediction_frame(
        predictions,
        *,
        target_col,
        model_name,
        diff_context,
        restore_target_predictions,
    ):
        del target_col, model_name, diff_context, restore_target_predictions
        return predictions.rename(columns={"AAForecast": "y_hat"})

    def configure_stochastic_inference(*, enabled, dropout_p=None):
        if enabled and dropout_p is not None:
            current_dropout["value"] = float(dropout_p)

    monkeypatch.setattr(aa_runtime, "_predict_with_adapter", fake_predict_with_adapter)
    monkeypatch.setattr(
        aa_runtime,
        "_extract_target_prediction_frame",
        fake_extract_target_prediction_frame,
    )

    model = SimpleNamespace(
        random_seed=7,
        configure_stochastic_inference=configure_stochastic_inference,
    )
    summary = aa_runtime._select_uncertainty_predictions(
        nf=object(),
        adapter_inputs=object(),
        model=model,
        model_name="AAForecast",
        target_col="series",
        diff_context=None,
        restore_target_predictions=None,
        prediction_column="y_hat",
        dropout_candidates=(0.1, 0.2),
        sample_count=3,
    )

    assert summary["selection_mode"] == "trajectory_min_dispersion"
    assert summary["selected_path_idx"] == 1
    assert summary["selected_dropout"].tolist() == [0.2, 0.2]
    assert summary["mean"].tolist() == pytest.approx([9.0, 9.5])


def test_select_uncertainty_predictions_can_use_semantic_tradeoff(
    monkeypatch,
) -> None:
    current_dropout = {"value": None}
    model = SimpleNamespace(
        random_seed=7,
        configure_stochastic_inference=None,
        _latest_decoder_debug={},
    )

    def fake_predict_with_adapter(_nf, _adapter_inputs, *, random_seed=None):
        dropout_p = current_dropout["value"]
        if dropout_p == 0.1:
            lookup = {
                7: [10.0, 10.5],
                8: [10.1, 10.7],
                9: [9.9, 10.3],
            }
            support = 0.2
            direction = 0.56
        else:
            lookup = {
                11: [9.0, 9.7],
                12: [9.05, 9.75],
                13: [8.95, 9.65],
            }
            support = 1.5
            direction = 0.80
        model._latest_decoder_debug = {
            "semantic_spike_component": torch.tensor([support, support]),
            "semantic_baseline_curve": torch.tensor([0.0, 0.0]),
            "semantic_spike_direction": torch.tensor([direction]),
        }
        values = lookup[int(random_seed)]
        return pd.DataFrame(
            {
                "unique_id": ["series", "series"],
                "AAForecast": values,
            }
        )

    def fake_extract_target_prediction_frame(
        predictions,
        *,
        target_col,
        model_name,
        diff_context,
        restore_target_predictions,
    ):
        del target_col, model_name, diff_context, restore_target_predictions
        return predictions.rename(columns={"AAForecast": "y_hat"})

    def configure_stochastic_inference(*, enabled, dropout_p=None):
        if enabled and dropout_p is not None:
            current_dropout["value"] = float(dropout_p)

    model.configure_stochastic_inference = configure_stochastic_inference
    monkeypatch.setattr(aa_runtime, "_predict_with_adapter", fake_predict_with_adapter)
    monkeypatch.setattr(
        aa_runtime,
        "_extract_target_prediction_frame",
        fake_extract_target_prediction_frame,
    )

    summary = aa_runtime._select_uncertainty_predictions(
        nf=SimpleNamespace(models=[model]),
        adapter_inputs=object(),
        model=model,
        model_name="AAForecast",
        target_col="series",
        diff_context=None,
        restore_target_predictions=None,
        prediction_column="y_hat",
        dropout_candidates=(0.1, 0.2),
        sample_count=3,
    )

    assert summary["selection_mode"] == "trajectory_semantic_tradeoff"
    assert summary["selected_path_idx"] == 1
    assert summary["selected_dropout"].tolist() == [0.2, 0.2]


def test_validate_only_rejects_yaml_managed_aaforecast_dropout_candidates(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(AUTO_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(
        (AUTO_CONFIG.parent / payload["dataset"]["path"]).resolve()
    )
    plugin_payload = yaml.safe_load(
        (
            AUTO_CONFIG.parent / "aa_forecast_runtime_plugin_auto_model_only.yaml"
        ).read_text(encoding="utf-8")
    )
    plugin_payload["aa_forecast"]["uncertainty"]["dropout_candidates"] = [0.1, 0.2, 0.3]
    plugin_path = tmp_path / "aaforecast-invalid-dropout-plugin.yaml"
    plugin_path.write_text(
        yaml.safe_dump(plugin_payload, sort_keys=False), encoding="utf-8"
    )
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
    for payload in (
        resolved["aa_forecast"],
        manifest["aa_forecast"],
        capability["aa_forecast"],
    ):
        _assert_grouping_payload(
            payload,
            config_path="yaml/plugins/aa_forecast.yaml",
            star_anomaly_tails=expected_tails,
            non_star_hist_exog_cols=expected_non_star,
        )


PLUGIN_PATCHTST_FIXED_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_patchtst_fixed_main.yaml"
)
PATCHTST_AA_BACKBONE_SELECTORS = [
    "hidden_size",
    "n_heads",
    "encoder_layers",
    "dropout",
    "linear_hidden_size",
    "attn_dropout",
    "patch_len",
    "stride",
    "decoder_hidden_size",
    "decoder_layers",
]


def test_runtime_validate_only_accepts_aaforecast_plugin_patchtst_fixed_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "validate-only-aa-forecast-plugin-patchtst-fixed"
    code = runtime.main(
        [
            "--config",
            str(PLUGIN_PATCHTST_FIXED_MAIN_CONFIG),
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
        config_path="tests/fixtures/aa_forecast_runtime_plugin_patchtst_fixed.yaml",
        star_anomaly_tails={"upward": ["event"], "two_sided": []},
        non_star_hist_exog_cols=[],
        model="patchtst",
    )
    stage_config = json.loads(
        (output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )
    assert stage_config["model"] == "patchtst"
    assert stage_config["backbone"] == "patchtst"
    assert "mode" not in stage_config
