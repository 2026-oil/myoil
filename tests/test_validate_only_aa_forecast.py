from __future__ import annotations

import json
from pathlib import Path

import app_config
import plugin_contracts.stage_registry as stage_registry
import runtime_support.runner as runtime


FIXED_CONFIG = Path("tests/fixtures/aa_forecast_runtime_smoke.yaml")
AUTO_CONFIG = Path("tests/fixtures/aa_forecast_runtime_auto_smoke.yaml")


def test_runtime_validate_only_accepts_aaforecast_fixed_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(app_config, "_ensure_plugins_loaded", lambda: None)
    monkeypatch.setattr(stage_registry, "_ensure_plugins_loaded", lambda: None)

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
    assert manifest["jobs"] == [
        {
            "model": "AAForecast",
            "requested_mode": "learned_fixed",
            "validated_mode": "learned_fixed",
            "selected_search_params": [],
        }
    ]


def test_runtime_validate_only_accepts_aaforecast_auto_path(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(app_config, "_ensure_plugins_loaded", lambda: None)
    monkeypatch.setattr(stage_registry, "_ensure_plugins_loaded", lambda: None)

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
    assert manifest["jobs"] == [
        {
            "model": "AAForecast",
            "requested_mode": "learned_auto_requested",
            "validated_mode": "learned_auto",
            "selected_search_params": [
                "encoder_hidden_size",
                "encoder_n_layers",
                "encoder_dropout",
                "decoder_hidden_size",
                "decoder_layers",
                "season_length",
                "trend_kernel_size",
                "anomaly_threshold",
            ],
        }
    ]
