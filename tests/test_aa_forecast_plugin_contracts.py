"""Contract tests for aa_forecast plugin fail-fast and config helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app_config import load_app_config
from plugins.aa_forecast.config import (
    AAForecastPluginConfig,
    aa_forecast_plugin_state_dict,
    aa_forecast_stage_document_type,
)
from plugins.aa_forecast.plugin import AAForecastStagePlugin


def test_aa_forecast_stage_document_type_accepts_yaml_and_toml() -> None:
    assert aa_forecast_stage_document_type(Path("stage.yaml")) == "yaml"
    assert aa_forecast_stage_document_type(Path("stage.yml")) == "yaml"
    assert aa_forecast_stage_document_type(Path("stage.YML")) == "yaml"
    assert aa_forecast_stage_document_type(Path("stage.toml")) == "toml"


def test_aa_forecast_stage_document_type_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError, match=r"\.json"):
        aa_forecast_stage_document_type(Path("stage.json"))


def test_fanout_filter_payload_requires_dict() -> None:
    plugin = AAForecastStagePlugin()
    with pytest.raises(TypeError, match="dict"):
        plugin.fanout_filter_payload("not-a-dict")  # type: ignore[arg-type]


def test_fanout_stage_payload_none_when_disabled_or_missing() -> None:
    plugin = AAForecastStagePlugin()
    loaded = SimpleNamespace(normalized_payload={})
    assert plugin.fanout_stage_payload(loaded) is None
    loaded2 = SimpleNamespace(
        normalized_payload={"aa_forecast": {"enabled": False}}
    )
    assert plugin.fanout_stage_payload(loaded2) is None


def test_fanout_stage_payload_raises_when_enabled_without_stage1() -> None:
    plugin = AAForecastStagePlugin()
    loaded = SimpleNamespace(
        normalized_payload={"aa_forecast": {"enabled": True}}
    )
    with pytest.raises(ValueError, match="stage1"):
        plugin.fanout_stage_payload(loaded)


def test_fanout_stage_payload_returns_stage1_when_present() -> None:
    plugin = AAForecastStagePlugin()
    stage1 = {"source_path": "/tmp/x.yaml", "source_type": "yaml"}
    loaded = SimpleNamespace(
        normalized_payload={"aa_forecast": {"enabled": True, "stage1": stage1}}
    )
    assert plugin.fanout_stage_payload(loaded) == {"aa_forecast": {"stage1": stage1}}


def test_aa_forecast_plugin_state_dict_omits_compatibility_metadata() -> None:
    payload = aa_forecast_plugin_state_dict(
        AAForecastPluginConfig(enabled=True, config_path="yaml/plugins/example.yaml"),
        selected_config_path="yaml/plugins/example.yaml",
    )
    assert payload["model"] == "gru"
    assert "mode" not in payload
    assert "compatibility_mode" not in payload
    assert "compatibility_source_path" not in payload


def test_load_app_config_rejects_direct_top_level_aaforecast_jobs(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_direct_aaforecast"},
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["event"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "training": {"loss": "mse"},
        "cv": {"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0},
        "scheduler": {"gpu_ids": [0], "max_concurrent_jobs": 1, "worker_devices": 1},
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "jobs": [{"model": "AAForecast", "params": {}}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="Direct top-level AAForecast jobs are no longer supported",
    ):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_mixed_top_level_jobs_when_aa_forecast_route_enabled(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        "aa_forecast:\n"
        "  model: gru\n"
        "  tune_training: false\n"
        "  star_anomaly_tails:\n"
        "    upward:\n"
        "      - event\n"
        "    two_sided: []\n"
        "  model_params: {}\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_mixed_jobs_with_route"},
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["event"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "training": {"loss": "mse"},
        "cv": {"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0},
        "scheduler": {"gpu_ids": [0], "max_concurrent_jobs": 1, "worker_devices": 1},
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "jobs": [{"model": "LSTM", "params": {"encoder_hidden_size": 8}}],
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="aa_forecast plugin route cannot be combined with top-level jobs",
    ):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_jobs_path_when_aa_forecast_route_enabled(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        "aa_forecast:\n"
        "  model: gru\n"
        "  tune_training: false\n"
        "  star_anomaly_tails:\n"
        "    upward:\n"
        "      - event\n"
        "    two_sided: []\n"
        "  model_params: {}\n",
        encoding="utf-8",
    )
    jobs_path = tmp_path / "jobs.yaml"
    jobs_path.write_text(
        yaml.safe_dump(
            [{"model": "LSTM", "params": {"encoder_hidden_size": 8}}],
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_jobs_path_with_route"},
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["event"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "training": {"loss": "mse"},
        "cv": {"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0},
        "scheduler": {"gpu_ids": [0], "max_concurrent_jobs": 1, "worker_devices": 1},
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "jobs": str(jobs_path),
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="aa_forecast plugin route cannot be combined with top-level jobs",
    ):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_unsupported_aa_forecast_model(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        "aa_forecast:\n"
        "  model: lstm\n"
        "  tune_training: false\n"
        "  star_anomaly_tails:\n"
        "    upward:\n"
        "      - event\n"
        "    two_sided: []\n"
        "  model_params: {}\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_unsupported_aaforecast_model"},
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["event"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "training": {"loss": "mse"},
        "cv": {"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0},
        "scheduler": {"gpu_ids": [0], "max_concurrent_jobs": 1, "worker_devices": 1},
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"aa_forecast\.model must be one of: gru"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_legacy_aa_forecast_mode_key(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        "aa_forecast:\n"
        "  mode: learned_auto\n"
        "  tune_training: false\n"
        "  star_anomaly_tails:\n"
        "    upward:\n"
        "      - event\n"
        "    two_sided: []\n"
        "  model_params: {}\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_legacy_aaforecast_mode"},
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["event"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "training": {"loss": "mse"},
        "cv": {"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0},
        "scheduler": {"gpu_ids": [0], "max_concurrent_jobs": 1, "worker_devices": 1},
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"unsupported key\(s\): mode"):
        load_app_config(tmp_path, config_path=config_path)
