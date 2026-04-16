"""Contract tests for aa_forecast plugin fail-fast and config helpers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from app_config import load_app_config
from plugins.aa_forecast.config import (
    AAForecastPluginConfig,
    aa_forecast_plugin_state_dict,
    aa_forecast_stage_document_type,
)
from plugins.aa_forecast.plugin import AAForecastStagePlugin
from plugins.aa_forecast.search_space import AA_FORECAST_STAGE_ONLY_PARAM_REGISTRY
import plugins.aa_forecast.runtime as aa_runtime
import runtime_support.forecast_models as forecast_models
import runtime_support.runner as runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_AUTO_MODEL_ONLY_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_auto_model_only_main.yaml"
)
AA_FORECAST_GRU_RET_EXPERIMENT_CONFIG = Path(
    "yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml"
)
SUPPORTED_AA_BACKBONES = [
    "gru",
    "informer",
    "itransformer",
    "patchtst",
    "timexer",
    "vanillatransformer",
]
FIXED_BACKBONE_PARAMS = {
    "gru": {"encoder_hidden_size": 8, "encoder_n_layers": 2, "encoder_dropout": 0.1},
    "vanillatransformer": {
        "hidden_size": 8,
        "n_head": 2,
        "encoder_layers": 1,
        "dropout": 0.1,
        "linear_hidden_size": 16,
    },
    "informer": {
        "hidden_size": 8,
        "n_head": 2,
        "encoder_layers": 1,
        "dropout": 0.1,
        "linear_hidden_size": 16,
        "factor": 1,
    },
    "itransformer": {
        "hidden_size": 8,
        "n_heads": 2,
        "e_layers": 1,
        "dropout": 0.1,
        "d_ff": 16,
        "factor": 1,
        "use_norm": True,
    },
    "patchtst": {
        "hidden_size": 8,
        "n_heads": 2,
        "encoder_layers": 1,
        "dropout": 0.1,
        "linear_hidden_size": 16,
        "attn_dropout": 0.0,
        "patch_len": 2,
        "stride": 1,
    },
    "timexer": {
        "hidden_size": 8,
        "n_heads": 2,
        "e_layers": 1,
        "dropout": 0.1,
        "d_ff": 16,
        "patch_len": 2,
        "use_norm": True,
    },
}


@pytest.mark.parametrize("backbone", SUPPORTED_AA_BACKBONES)
def test_aaforecast_search_space_registry_includes_lowess_params(backbone: str) -> None:
    registry = AA_FORECAST_STAGE_ONLY_PARAM_REGISTRY[backbone]

    assert "lowess_frac" in registry
    assert "lowess_delta" in registry

    lowess_frac = registry["lowess_frac"]
    lowess_delta = registry["lowess_delta"]
    assert lowess_frac == {"type": "float", "low": 0.0, "high": 1.0}
    assert lowess_delta == {"type": "float", "low": 0.0, "high": 1.0}


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
    loaded2 = SimpleNamespace(normalized_payload={"aa_forecast": {"enabled": False}})
    assert plugin.fanout_stage_payload(loaded2) is None


def test_fanout_stage_payload_raises_when_enabled_without_stage1() -> None:
    plugin = AAForecastStagePlugin()
    loaded = SimpleNamespace(normalized_payload={"aa_forecast": {"enabled": True}})
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
    assert payload["backbone"] == "gru"
    assert "mode" not in payload
    assert "compatibility_mode" not in payload
    assert "compatibility_source_path" not in payload


def test_load_app_config_accepts_retrieval_block_and_projects_state_defaults(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    retrieval_path = tmp_path / "retrieval_detail.yaml"
    retrieval_path.write_text(
        yaml.safe_dump(
            {
                "retrieval": {
                    "enabled": True,
                    "star": {
                        "season_length": 4,
                        "thresh": 3.5,
                        "anomaly_tails": {"upward": ["event"], "two_sided": []},
                    },
                    "top_k": 3,
                    "recency_gap_steps": 2,
                    "event_score_threshold": 10.0,
                    "trigger_quantile": 0.9,
                    "neighbor_min_event_ratio": 0.8,
                    "min_similarity": 0.8,
                    "blend_floor": 0.1,
                    "blend_max": 0.2,
                    "mode": "posthoc_blend",
                    "similarity": "cosine",
                    "temperature": 0.1,
                    "memory_value_mode": "future_level",
                    "use_uncertainty_gate": True,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    plugin_path.write_text(
        yaml.safe_dump(
            {
                "aa_forecast": {
                    "model": "gru",
                    "tune_training": False,
                    "star_anomaly_tails": {"upward": ["event"], "two_sided": []},
                    "uncertainty": {"enabled": True, "sample_count": 2},
                    "retrieval": {
                        "enabled": True,
                        "config_path": retrieval_path.name,
                    },
                    "model_params": {},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "accept_aaforecast_retrieval_route"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(REPO_ROOT, config_path=config_path)

    retrieval = loaded.config.stage_plugin_config.retrieval
    assert retrieval.enabled is True
    assert retrieval.config_path == retrieval_path.name
    assert retrieval.top_k == 3
    assert retrieval.recency_gap_steps == 2
    assert retrieval.event_score_threshold == pytest.approx(10.0)
    assert retrieval.trigger_quantile is None
    assert retrieval.neighbor_min_event_ratio == pytest.approx(0.0)
    assert retrieval.min_similarity == pytest.approx(0.8)
    assert retrieval.blend_floor == pytest.approx(0.1)
    assert retrieval.blend_max == pytest.approx(0.2)
    assert retrieval.mode == "posthoc_blend"
    assert retrieval.similarity == "cosine"
    assert retrieval.temperature == pytest.approx(0.1)
    assert retrieval.memory_value_mode == "future_return"
    assert retrieval.use_uncertainty_gate is True

    payload = aa_forecast_plugin_state_dict(
        loaded.config.stage_plugin_config,
        selected_config_path=str(plugin_path),
    )
    assert payload["retrieval"]["enabled"] is True
    assert payload["retrieval"]["config_path"] == retrieval_path.name
    assert payload["retrieval"]["mode"] == "posthoc_blend"
    assert payload["retrieval"]["similarity"] == "cosine"
    assert payload["retrieval"]["top_k"] == 3
    assert payload["retrieval"]["temperature"] == pytest.approx(0.1)
    assert payload["retrieval"]["trigger_quantile"] is None
    assert payload["retrieval"]["neighbor_min_event_ratio"] == pytest.approx(0.0)
    assert payload["retrieval"]["memory_value_mode"] == "future_return"
    assert payload["retrieval"]["use_shape_key"] is True
    assert payload["retrieval"]["use_event_key"] is True


def test_load_app_config_defaults_to_robust_scaler_when_aa_forecast_not_enabled(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "default_robust_scaler_without_aa_forecast"},
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": [],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "training": {"loss": "mse"},
        "cv": {"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0},
        "scheduler": {"gpu_ids": [0], "max_concurrent_jobs": 1, "worker_devices": 1},
        "jobs": [{"model": "Naive", "params": {}}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.config.training.scaler_type == "robust"


def test_load_app_config_rejects_unknown_retrieval_key(tmp_path: Path) -> None:
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
        "  uncertainty:\n"
        "    enabled: true\n"
        "  retrieval:\n"
        "    enabled: true\n"
        "    unsupported: 1\n"
        "  model_params: {}\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_unknown_retrieval_key"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported key\(s\): unsupported"):
        load_app_config(REPO_ROOT, config_path=config_path)


def test_load_app_config_accepts_encoding_export_for_informer(tmp_path: Path) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        yaml.safe_dump(
            {
                "aa_forecast": {
                    "model": "informer",
                    "tune_training": False,
                    "star_anomaly_tails": {"upward": ["event"], "two_sided": []},
                    "uncertainty": {"enabled": False, "sample_count": 2},
                    "encoding_export": {"enabled": True},
                    "model_params": {
                        "hidden_size": 8,
                        "n_head": 2,
                        "encoder_layers": 1,
                        "dropout": 0.1,
                        "linear_hidden_size": 16,
                        "factor": 1,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "accept_aaforecast_encoding_export"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(REPO_ROOT, config_path=config_path)

    assert loaded.config.stage_plugin_config.encoding_export.enabled is True
    state_payload = aa_forecast_plugin_state_dict(
        loaded.config.stage_plugin_config,
        selected_config_path=str(plugin_path),
    )
    assert state_payload["encoding_export"]["enabled"] is True


def test_load_app_config_rejects_encoding_export_for_non_informer(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        yaml.safe_dump(
            {
                "aa_forecast": {
                    "model": "gru",
                    "tune_training": False,
                    "star_anomaly_tails": {"upward": ["event"], "two_sided": []},
                    "uncertainty": {"enabled": False, "sample_count": 2},
                    "encoding_export": {"enabled": True},
                    "model_params": {},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_aaforecast_encoding_export"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="encoding_export.enabled=true.*informer"):
        load_app_config(REPO_ROOT, config_path=config_path)


def test_load_app_config_rejects_retrieval_without_uncertainty_enabled(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        yaml.safe_dump(
            {
                "aa_forecast": {
                    "model": "gru",
                    "tune_training": False,
                    "star_anomaly_tails": {"upward": ["event"], "two_sided": []},
                    "uncertainty": {"enabled": False},
                    "retrieval": {
                        "enabled": True,
                        "top_k": 3,
                        "recency_gap_steps": 2,
                        "event_score_threshold": 10.0,
                        "min_similarity": 0.8,
                        "blend_max": 0.2,
                        "use_uncertainty_gate": True,
                    },
                    "model_params": {},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_retrieval_without_uncertainty"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"aa_forecast\.retrieval\.enabled=true requires aa_forecast\.uncertainty\.enabled=true",
    ):
        load_app_config(REPO_ROOT, config_path=config_path)


def test_load_app_config_rejects_direct_top_level_aaforecast_jobs(
    tmp_path: Path,
) -> None:
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
        "jobs": [{"model": "AAForecast", "params": {}}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="Direct top-level AAForecast jobs are no longer supported",
    ):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_linked_aa_forecast_yaml_as_top_level_config(
    tmp_path: Path,
) -> None:
    plugin_path = tmp_path / "yaml/plugins/aa_forecast/aa_forecast_parity_informer.yaml"
    plugin_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_path.write_text(
        yaml.safe_dump(
            {
                "aa_forecast": {
                    "model": "informer",
                    "tune_training": False,
                    "star_anomaly_tails": {"upward": ["event"], "two_sided": []},
                    "model_params": {
                        "hidden_size": 8,
                        "n_head": 2,
                        "encoder_layers": 1,
                        "dropout": 0.1,
                        "linear_hidden_size": 16,
                        "factor": 1,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"is an aa_forecast plugin YAML, not a runnable app config",
    ):
        load_app_config(tmp_path, config_path=plugin_path)


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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"aa_forecast\.model must be one of: .*gru.*informer.*itransformer.*patchtst.*timexer.*vanillatransformer",
    ):
        load_app_config(tmp_path, config_path=config_path)


@pytest.mark.parametrize("backbone", SUPPORTED_AA_BACKBONES)
def test_load_app_config_accepts_supported_aa_forecast_backbones(
    tmp_path: Path,
    backbone: str,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        yaml.safe_dump(
            {
                "aa_forecast": {
                    "model": backbone,
                    "tune_training": False,
                    "star_anomaly_tails": {"upward": ["event"], "two_sided": []},
                    "model_params": FIXED_BACKBONE_PARAMS[backbone],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "task": {"name": f"accept_supported_aaforecast_model_{backbone}"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.config.stage_plugin_config.model == backbone
    assert loaded.config.stage_plugin_config.config_path == str(plugin_path)


def test_feature_set_gru_ret_route_resolves_to_learned_auto() -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / AA_FORECAST_GRU_RET_EXPERIMENT_CONFIG,
    )

    assert len(loaded.config.jobs) == 1
    job = loaded.config.jobs[0]
    assert job.model == "AAForecast"
    assert job.requested_mode == "learned_auto_requested"
    assert job.validated_mode == "learned_auto"
    assert job.selected_search_params


def test_load_app_config_rejects_backbone_specific_unknown_model_param(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        "aa_forecast:\n"
        "  model: timexer\n"
        "  tune_training: false\n"
        "  star_anomaly_tails:\n"
        "    upward:\n"
        "      - event\n"
        "    two_sided: []\n"
        "  model_params:\n"
        "    encoder_hidden_size: 8\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_timexer_unknown_model_param"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"unsupported key\(s\) for aa_forecast\.model='timexer': encoder_hidden_size",
    ):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_stale_informer_semantic_negative_scale(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "dt,target,event\n2020-01-01,1,0\n2020-01-08,2,1\n",
        encoding="utf-8",
    )
    plugin_path = tmp_path / "aa_plugin.yaml"
    plugin_path.write_text(
        "aa_forecast:\n"
        "  model: informer\n"
        "  tune_training: false\n"
        "  star_anomaly_tails:\n"
        "    upward:\n"
        "      - event\n"
        "    two_sided: []\n"
        "  model_params:\n"
        "    hidden_size: 8\n"
        "    semantic_negative_scale: 0.95\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_stale_informer_semantic_negative_scale"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"unsupported key\(s\) for aa_forecast\.model='informer': semantic_negative_scale",
    ):
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"unsupported key\(s\): mode"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_legacy_top_k_key(tmp_path: Path) -> None:
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
        "  top_k: 0.1\n"
        "  model_params: {}\n",
        encoding="utf-8",
    )
    payload = {
        "task": {"name": "reject_legacy_aaforecast_top_k"},
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
        "aa_forecast": {"enabled": True, "config_path": str(plugin_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"aa_forecast\.top_k.*aa_forecast\.thresh"):
        load_app_config(tmp_path, config_path=config_path)


def test_fit_and_predict_fold_passes_trial_overrides_to_aa_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = SimpleNamespace(
        config=SimpleNamespace(
            dataset=SimpleNamespace(dt_col="dt", target_col="target")
        )
    )
    job = SimpleNamespace(model="AAForecast")
    source_df = pd.DataFrame({"dt": ["2020-01-01", "2020-01-08"], "target": [1.0, 2.0]})
    captured: dict[str, object] = {}

    class _FakePlugin:
        def predict_fold(
            self,
            loaded_arg,
            job_arg,
            *,
            train_df,
            future_df,
            run_root,
            params_override=None,
            training_override=None,
        ):
            captured["loaded"] = loaded_arg
            captured["job"] = job_arg
            captured["train_rows"] = len(train_df)
            captured["future_rows"] = len(future_df)
            captured["run_root"] = run_root
            captured["params_override"] = params_override
            captured["training_override"] = training_override
            return (
                pd.DataFrame(
                    {
                        "unique_id": ["target"],
                        "ds": pd.to_datetime(["2020-01-08"]),
                        "AAForecast": [1.0],
                    }
                ),
                pd.Series([2.0]),
                pd.Timestamp("2020-01-01"),
                train_df,
                None,
            )

    monkeypatch.setattr(
        runtime,
        "_plugin_owned_top_level_job",
        lambda _loaded, _model_name: _FakePlugin(),
    )

    runtime._fit_and_predict_fold(
        loaded,
        job,
        run_root=Path("/tmp/aa-forecast-overrides"),
        source_df=source_df,
        freq="W",
        train_idx=[0],
        test_idx=[1],
        params_override={"season_length": 4, "thresh": 3.5},
        training_override={"model_step_size": 4, "scaler_type": "standard"},
    )

    assert captured["train_rows"] == 1
    assert captured["future_rows"] == 1
    assert captured["params_override"] == {"season_length": 4, "thresh": 3.5}
    assert captured["training_override"] == {
        "model_step_size": 4,
        "scaler_type": "standard",
    }


def test_predict_aa_forecast_fold_uses_trial_specific_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / PLUGIN_AUTO_MODEL_ONLY_MAIN_CONFIG,
    )
    job = loaded.config.jobs[0]
    source_df = pd.read_csv(loaded.config.dataset.path)
    train_df = source_df.iloc[:4].reset_index(drop=True)
    future_df = source_df.iloc[4:5].reset_index(drop=True)
    captured: dict[str, object] = {}

    class _FakeModel:
        def set_star_precompute_context(self, *, enabled: bool, fold_key: str) -> None:
            captured["star_precompute_enabled"] = enabled
            captured["star_fold_key"] = json.loads(fold_key)

    class _FakeNeuralForecast:
        def __init__(self, *, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, *_args, **_kwargs):
            return None

    def _fake_build_model(
        config,
        job_arg,
        *,
        n_series=None,
        params_override=None,
        loss_override=None,
        valid_loss_override=None,
    ):
        del job_arg, n_series, loss_override, valid_loss_override
        captured["training_model_step_size"] = config.training.model_step_size
        captured["training_scaler_type"] = config.training.scaler_type
        captured["params_override"] = params_override
        return _FakeModel()

    monkeypatch.setattr(forecast_models, "build_model", _fake_build_model)
    monkeypatch.setattr(runtime, "_resolve_freq", lambda _loaded, _source_df: "W-MON")
    monkeypatch.setattr(
        runtime,
        "_build_fold_diff_context",
        lambda _loaded, _train_df: object(),
    )
    monkeypatch.setattr(
        runtime,
        "_transform_training_frame",
        lambda train_df_arg, _diff_context: train_df_arg,
    )
    monkeypatch.setattr(
        runtime,
        "_build_adapter_inputs",
        lambda _loaded, transformed_train_df, futr_df, _job, _dt_col: SimpleNamespace(
            metadata={"n_series": 1},
            fit_df=transformed_train_df,
            static_df=None,
            futr_df=futr_df,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_restore_target_predictions",
        lambda target_predictions, prediction_col, diff_context: target_predictions,
    )
    monkeypatch.setattr(
        aa_runtime,
        "_predict_with_adapter",
        lambda _nf, adapter_inputs: pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.to_datetime(
                    adapter_inputs.futr_df[loaded.config.dataset.dt_col]
                ).reset_index(drop=True),
                job.model: [1.23],
            }
        ),
    )
    monkeypatch.setattr(aa_runtime, "NeuralForecast", _FakeNeuralForecast)

    predictions, actuals, train_end_ds, used_train_df, _nf = (
        aa_runtime.predict_aa_forecast_fold(
            loaded,
            job,
            train_df=train_df,
            future_df=future_df,
            run_root=None,
            params_override={"season_length": 4, "thresh": 3.5},
            training_override={"model_step_size": 4, "scaler_type": "standard"},
        )
    )

    assert captured["training_model_step_size"] == 4
    assert captured["training_scaler_type"] == "standard"
    assert captured["params_override"]["season_length"] == 4
    assert captured["params_override"]["thresh"] == pytest.approx(3.5)
    assert captured["params_override"]["star_hist_exog_list"] == list(
        loaded.config.stage_plugin_config.star_hist_exog_cols_resolved
    )
    assert captured["params_override"]["non_star_hist_exog_list"] == list(
        loaded.config.stage_plugin_config.non_star_hist_exog_cols_resolved
    )
    assert captured["star_precompute_enabled"] is True
    assert captured["star_fold_key"]["params_override"]["season_length"] == 4
    assert captured["star_fold_key"]["training_override"]["model_step_size"] == 4
    assert predictions[job.model].tolist() == [1.23]
    assert actuals.tolist() == future_df[loaded.config.dataset.target_col].tolist()
    assert str(train_end_ds) == str(
        pd.Timestamp(train_df[loaded.config.dataset.dt_col].iloc[-1])
    )
    assert used_train_df.equals(train_df)


def test_run_single_job_replay_routes_plugin_tuned_params_as_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / PLUGIN_AUTO_MODEL_ONLY_MAIN_CONFIG,
    )
    auto_job = replace(
        loaded.config.jobs[0],
        validated_mode="learned_auto",
        selected_search_params=("encoder_hidden_size",),
    )
    loaded = replace(loaded, config=replace(loaded.config, jobs=(auto_job,)))
    selection = runtime.StudySelection(
        study_count=1,
        configured_selected_study_index=None,
        cli_selected_study_index=None,
        selected_study_index=None,
        canonical_projection_study_index=1,
        execute_study_indices=(1,),
    )
    study_root = tmp_path / "run" / "models" / auto_job.model / "studies" / "study-01"

    monkeypatch.setattr(runtime, "resolve_study_selection", lambda _loaded: selection)
    monkeypatch.setattr(
        runtime,
        "build_study_context",
        lambda *_args, **_kwargs: SimpleNamespace(
            study_root=study_root,
            study_index=1,
            selection=selection,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_load_main_tuning_result",
        lambda *_args, **_kwargs: (
            {"encoder_hidden_size": 64, "recency_gap_steps": 4},
            {},
            {"best_value": 0.1, "trial_count": 1, "finished_trial_count": 1},
        ),
    )
    monkeypatch.setattr(runtime, "_resolve_freq", lambda *_args, **_kwargs: "W-MON")
    monkeypatch.setattr(
        runtime, "_build_tscv_splits", lambda *_args, **_kwargs: [([0, 1, 2], [3])]
    )
    monkeypatch.setattr(
        runtime,
        "build_study_visualizations",
        lambda *_args, **_kwargs: {"artifact_inventory_path": "noop"},
    )
    monkeypatch.setattr(
        runtime,
        "build_cross_study_visualizations",
        lambda *_args, **_kwargs: {"artifact_inventory_path": "noop"},
    )
    monkeypatch.setattr(runtime, "_write_stage_study_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "_update_manifest_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime, "_write_loss_curve_artifact", lambda *_args, **_kwargs: None)

    captured: dict[str, Any] = {}

    class _FakePlugin:
        def predict_fold(
            self,
            loaded_arg,
            job_arg,
            *,
            train_df,
            future_df,
            run_root,
            params_override=None,
            training_override=None,
        ):
            del loaded_arg, job_arg, run_root, training_override
            captured["params_override"] = params_override
            return (
                pd.DataFrame(
                    {
                        "unique_id": [loaded.config.dataset.target_col],
                        "ds": pd.to_datetime(
                            future_df[loaded.config.dataset.dt_col]
                        ).reset_index(drop=True),
                        auto_job.model: [1.0],
                    }
                ),
                future_df[loaded.config.dataset.target_col].reset_index(drop=True),
                pd.Timestamp(train_df[loaded.config.dataset.dt_col].iloc[-1]),
                train_df,
                None,
            )

    monkeypatch.setattr(
        runtime,
        "_plugin_owned_top_level_job",
        lambda _loaded, model_name: _FakePlugin() if model_name == "AAForecast" else None,
    )

    run_root = tmp_path / "run"
    runtime._run_single_job(
        loaded,
        auto_job,
        run_root,
        manifest_path=run_root / "manifest" / "run_manifest.json",
        main_stage="replay-only",
    )

    assert captured["params_override"]["encoder_hidden_size"] == 64
    assert captured["params_override"]["recency_gap_steps"] == 4


def test_predict_aa_forecast_fold_writes_informer_encoding_export_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / PLUGIN_AUTO_MODEL_ONLY_MAIN_CONFIG,
    )
    loaded = replace(
        loaded,
        config=replace(
            loaded.config,
            stage_plugin_config=replace(
                loaded.config.stage_plugin_config,
                model="informer",
                encoding_export=aa_runtime._cfg.AAForecastEncodingExportConfig(
                    enabled=True
                ),
            ),
        ),
    )
    job = loaded.config.jobs[0]
    source_df = pd.read_csv(loaded.config.dataset.path)
    train_df = source_df.iloc[:4].reset_index(drop=True)
    future_df = source_df.iloc[4:5].reset_index(drop=True)

    class _FakeModel:
        backbone = "informer"

        def set_star_precompute_context(self, *, enabled: bool, fold_key: str) -> None:
            del enabled, fold_key

    class _FakeNeuralForecast:
        def __init__(self, *, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        forecast_models,
        "build_model",
        lambda *_args, **_kwargs: _FakeModel(),
    )
    monkeypatch.setattr(runtime, "_resolve_freq", lambda _loaded, _source_df: "W-MON")
    monkeypatch.setattr(
        runtime, "_build_fold_diff_context", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        runtime,
        "_transform_training_frame",
        lambda train_df_arg, _diff_context: train_df_arg,
    )
    monkeypatch.setattr(
        runtime,
        "_build_adapter_inputs",
        lambda _loaded, transformed_train_df, futr_df, _job, _dt_col: SimpleNamespace(
            metadata={"n_series": 1},
            fit_df=transformed_train_df,
            static_df=None,
            futr_df=futr_df,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_restore_target_predictions",
        lambda target_predictions, prediction_col, diff_context: target_predictions,
    )

    def _fake_predict_with_adapter(nf, adapter_inputs, *, random_seed=None):
        del adapter_inputs, random_seed
        nf.models[0]._latest_encoding_export = {
            "backbone_states": np.arange(12, dtype=float).reshape(1, 4, 3),
            "hidden_states": np.arange(12, 24, dtype=float).reshape(1, 4, 3),
            "time_axis": 1,
        }
        return pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col] * len(future_df),
                "ds": pd.to_datetime(
                    future_df[loaded.config.dataset.dt_col]
                ).reset_index(drop=True),
                job.model: [1.1],
            }
        )

    monkeypatch.setattr(aa_runtime, "_predict_with_adapter", _fake_predict_with_adapter)
    monkeypatch.setattr(
        aa_runtime,
        "_build_fold_context_frame",
        lambda **_kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(train_df[loaded.config.dataset.dt_col]),
                    "context_active": [0] * len(train_df),
                    "context_label": ["normal_context"] * len(train_df),
                }
            ),
            False,
        ),
    )
    monkeypatch.setattr(aa_runtime, "NeuralForecast", _FakeNeuralForecast)

    predictions, *_rest = aa_runtime.predict_aa_forecast_fold(
        loaded,
        job,
        train_df=train_df,
        future_df=future_df,
        run_root=tmp_path / "run",
    )

    metadata_rel = predictions["aaforecast_encoding_metadata_artifact"].iloc[0]
    backbone_rel = predictions["aaforecast_encoding_backbone_artifact"].iloc[0]
    hidden_rel = predictions["aaforecast_encoding_hidden_artifact"].iloc[0]
    time_axis_rel = predictions["aaforecast_encoding_time_axis_artifact"].iloc[0]
    summary_rel = predictions["aaforecast_encoding_summary_artifact"].iloc[0]

    metadata = json.loads((tmp_path / "run" / metadata_rel).read_text(encoding="utf-8"))
    backbone_frame = pd.read_parquet(tmp_path / "run" / backbone_rel)
    hidden_frame = pd.read_parquet(tmp_path / "run" / hidden_rel)
    time_axis_frame = pd.read_parquet(tmp_path / "run" / time_axis_rel)
    summary_frame = pd.read_parquet(tmp_path / "run" / summary_rel)

    assert metadata["source"] == "base_predict_before_uncertainty"
    assert metadata["time_axis_contract"]["time_axis_dim"] == 1
    assert [entry["tensor_name"] for entry in metadata["tensors"]] == [
        "backbone_states",
        "hidden_states",
    ]
    assert backbone_frame.columns.tolist() == [
        "tensor_name",
        "batch_index",
        "time_index",
        "hidden_index",
        "ds",
        "value",
    ]
    assert hidden_frame["tensor_name"].unique().tolist() == ["hidden_states"]
    assert backbone_frame["time_index"].unique().tolist() == [0, 1, 2, 3]
    assert pd.to_datetime(backbone_frame["ds"]).dt.strftime(
        "%Y-%m-%d"
    ).unique().tolist() == [
        "2024-01-01",
        "2024-01-08",
        "2024-01-15",
        "2024-01-22",
    ]
    assert time_axis_frame["time_axis_dim"].unique().tolist() == [1]
    assert summary_frame["hidden_size"].unique().tolist() == [3]


def test_predict_aa_forecast_fold_applies_retrieval_after_uncertainty_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT
        / "tests/fixtures/aa_forecast_runtime_plugin_uncertainty_main.yaml",
    )
    job = loaded.config.jobs[0]
    source_df = pd.read_csv(loaded.config.dataset.path)
    train_df = source_df.iloc[:4].reset_index(drop=True)
    future_df = source_df.iloc[4:5].reset_index(drop=True)
    captured: dict[str, object] = {}

    class _FakeModel:
        input_size = 2
        h = 1
        hist_exog_list = ("event",)

        def set_star_precompute_context(self, *, enabled: bool, fold_key: str) -> None:
            del enabled, fold_key

    class _FakeNeuralForecast:
        def __init__(self, *, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        forecast_models,
        "build_model",
        lambda *_args, **_kwargs: _FakeModel(),
    )
    monkeypatch.setattr(runtime, "_resolve_freq", lambda _loaded, _source_df: "W-MON")
    monkeypatch.setattr(
        runtime, "_build_fold_diff_context", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        runtime,
        "_transform_training_frame",
        lambda train_df_arg, _diff_context: train_df_arg,
    )
    monkeypatch.setattr(
        runtime,
        "_build_adapter_inputs",
        lambda _loaded, transformed_train_df, futr_df, _job, _dt_col: SimpleNamespace(
            metadata={"n_series": 1},
            fit_df=transformed_train_df,
            static_df=None,
            futr_df=futr_df,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_restore_target_predictions",
        lambda target_predictions, prediction_col, diff_context: target_predictions,
    )
    monkeypatch.setattr(
        aa_runtime,
        "_predict_with_adapter",
        lambda _nf, adapter_inputs: pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.to_datetime(
                    adapter_inputs.futr_df[loaded.config.dataset.dt_col]
                ).reset_index(drop=True),
                job.model: [1.23],
            }
        ),
    )
    monkeypatch.setattr(
        aa_runtime,
        "_build_fold_context_frame",
        lambda **_kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(train_df[loaded.config.dataset.dt_col]),
                    "context_active": [0] * len(train_df),
                    "context_label": ["normal_context"] * len(train_df),
                }
            ),
            False,
        ),
    )
    monkeypatch.setattr(
        aa_runtime,
        "_select_uncertainty_predictions",
        lambda **_kwargs: {
            "mean": np.asarray([10.0]),
            "std": np.asarray([0.5]),
            "selected_dropout": np.asarray([0.1]),
            "candidate_mean_grid": np.asarray([[10.0]]),
            "candidate_std_grid": np.asarray([[0.5]]),
            "candidate_dropout_values": np.asarray([0.1]),
            "candidate_samples": {"0.10": [[10.0]]},
        },
    )
    monkeypatch.setattr(
        aa_runtime,
        "_build_event_memory_bank",
        lambda **_kwargs: (
            [
                {
                    "candidate_end_ds": "2020-01-15 00:00:00",
                    "candidate_future_end_ds": "2020-01-22 00:00:00",
                    "future_returns": np.asarray([0.1]),
                    "event_score": 3.0,
                    "anchor_target_value": 5.0,
                    "shape_vector": np.asarray([1.0]),
                    "event_vector": np.asarray([1.0]),
                }
            ],
            1,
        ),
    )
    monkeypatch.setattr(
        aa_runtime,
        "_build_event_query",
        lambda **_kwargs: {
            "shape_vector": np.asarray([1.0]),
            "event_vector": np.asarray([1.0]),
            "event_score": 5.0,
        },
    )
    monkeypatch.setattr(
        aa_runtime,
        "_retrieve_event_neighbors",
        lambda **_kwargs: {
            "retrieval_attempted": True,
            "retrieval_applied": True,
            "skip_reason": None,
            "top_neighbors": [
                {
                    "candidate_end_ds": "2020-01-15 00:00:00",
                    "candidate_future_end_ds": "2020-01-22 00:00:00",
                    "future_returns": np.asarray([0.1]),
                    "event_score": 3.0,
                    "anchor_target_value": 5.0,
                    "shape_similarity": 0.8,
                    "event_similarity": 0.9,
                    "similarity": 0.86,
                    "softmax_weight": 1.0,
                }
            ],
            "mean_similarity": 0.86,
            "max_similarity": 0.86,
        },
    )

    def _fake_blend(
        *,
        base_prediction,
        memory_prediction,
        uncertainty_std,
        retrieval_cfg,
        mean_similarity,
    ):
        captured["base_prediction"] = base_prediction.tolist()
        captured["memory_prediction"] = memory_prediction.tolist()
        captured["uncertainty_std"] = uncertainty_std.tolist()
        captured["retrieval_cfg_enabled"] = retrieval_cfg.enabled
        captured["mean_similarity"] = mean_similarity
        return np.asarray([12.0]), np.asarray([0.2])

    monkeypatch.setattr(aa_runtime, "_blend_event_memory_prediction", _fake_blend)
    monkeypatch.setattr(aa_runtime, "NeuralForecast", _FakeNeuralForecast)

    predictions, *_rest = aa_runtime.predict_aa_forecast_fold(
        loaded,
        job,
        train_df=train_df,
        future_df=future_df,
        run_root=None,
    )

    assert captured["base_prediction"] == [10.0]
    assert captured["memory_prediction"] == [
        train_df["target"].iloc[-1] + max(abs(train_df["target"].iloc[-1]), 1e-8) * 0.1
    ]
    assert captured["uncertainty_std"] == [0.5]
    assert captured["retrieval_cfg_enabled"] is True
    assert captured["mean_similarity"] == pytest.approx(0.86)
    assert predictions[job.model].tolist() == [12.0]
    assert predictions["aaforecast_retrieval_applied"].tolist() == [True]


def test_retrieve_event_neighbors_event_score_bonus_can_flip_top1() -> None:
    retrieval_cfg = SimpleNamespace(
        top_k=1,
        event_score_threshold=0.1,
        min_similarity=0.0,
        temperature=0.1,
        use_shape_key=False,
        use_event_key=True,
        event_score_log_bonus_alpha=0.15,
        event_score_log_bonus_cap=0.1,
    )
    query = {
        "event_score": 1.0,
        "shape_vector": np.array([1.0, 0.0], dtype=float),
        "event_vector": np.array([1.0, 0.0], dtype=float),
    }
    bank = [
        {
            "candidate_end_ds": "plain-top1",
            "candidate_future_end_ds": "plain-top1+2",
            "shape_vector": np.array([1.0, 0.0], dtype=float),
            "event_vector": np.array([0.885, np.sqrt(1 - 0.885**2)], dtype=float),
            "event_score": 1.0,
            "anchor_target_value": 1.0,
            "future_returns": np.array([0.02, 0.03], dtype=float),
        },
        {
            "candidate_end_ds": "spike-analogue",
            "candidate_future_end_ds": "spike-analogue+2",
            "shape_vector": np.array([1.0, 0.0], dtype=float),
            "event_vector": np.array([0.872, np.sqrt(1 - 0.872**2)], dtype=float),
            "event_score": 1.2,
            "anchor_target_value": 1.0,
            "future_returns": np.array([0.08, 0.18], dtype=float),
        },
    ]

    result = aa_runtime._retrieve_event_neighbors(
        query=query,
        bank=bank,
        retrieval_cfg=retrieval_cfg,
    )

    assert result["retrieval_applied"] is True
    assert result["top_neighbors"][0]["candidate_end_ds"] == "spike-analogue"


def test_retrieve_event_neighbors_ignores_neighbor_min_event_ratio() -> None:
    retrieval_cfg = SimpleNamespace(
        top_k=1,
        event_score_threshold=1.0,
        min_similarity=0.0,
        temperature=0.1,
        use_shape_key=False,
        use_event_key=True,
        event_score_log_bonus_alpha=0.0,
        event_score_log_bonus_cap=0.0,
        neighbor_min_event_ratio=10.0,
    )
    query = {
        "event_score": 5.0,
        "shape_vector": np.array([1.0, 0.0], dtype=float),
        "event_vector": np.array([1.0, 0.0], dtype=float),
    }
    bank = [
        {
            "candidate_end_ds": "anchor-like",
            "candidate_future_end_ds": "anchor-like+1",
            "shape_vector": np.array([1.0, 0.0], dtype=float),
            "event_vector": np.array([1.0, 0.0], dtype=float),
            "event_score": 2.0,
            "anchor_target_value": 1.0,
            "future_returns": np.array([0.05], dtype=float),
        }
    ]

    result = aa_runtime._retrieve_event_neighbors(
        query=query,
        bank=bank,
        retrieval_cfg=retrieval_cfg,
    )

    assert result["retrieval_applied"] is True
    assert result["top_neighbors"][0]["candidate_end_ds"] == "anchor-like"


def test_predict_aa_forecast_fold_records_retrieval_skip_without_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT
        / "tests/fixtures/aa_forecast_runtime_plugin_uncertainty_main.yaml",
    )
    job = loaded.config.jobs[0]
    source_df = pd.read_csv(loaded.config.dataset.path)
    train_df = source_df.iloc[:4].reset_index(drop=True)
    future_df = source_df.iloc[4:5].reset_index(drop=True)

    class _FakeModel:
        input_size = 2
        h = 1
        hist_exog_list = ("event",)

        def set_star_precompute_context(self, *, enabled: bool, fold_key: str) -> None:
            del enabled, fold_key

    class _FakeNeuralForecast:
        def __init__(self, *, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        forecast_models,
        "build_model",
        lambda *_args, **_kwargs: _FakeModel(),
    )
    monkeypatch.setattr(runtime, "_resolve_freq", lambda _loaded, _source_df: "W-MON")
    monkeypatch.setattr(
        runtime, "_build_fold_diff_context", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        runtime,
        "_transform_training_frame",
        lambda train_df_arg, _diff_context: train_df_arg,
    )
    monkeypatch.setattr(
        runtime,
        "_build_adapter_inputs",
        lambda _loaded, transformed_train_df, futr_df, _job, _dt_col: SimpleNamespace(
            metadata={"n_series": 1},
            fit_df=transformed_train_df,
            static_df=None,
            futr_df=futr_df,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_restore_target_predictions",
        lambda target_predictions, prediction_col, diff_context: target_predictions,
    )
    monkeypatch.setattr(
        aa_runtime,
        "_predict_with_adapter",
        lambda _nf, adapter_inputs: pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.to_datetime(
                    adapter_inputs.futr_df[loaded.config.dataset.dt_col]
                ).reset_index(drop=True),
                job.model: [1.23],
            }
        ),
    )
    monkeypatch.setattr(
        aa_runtime,
        "_build_fold_context_frame",
        lambda **_kwargs: (
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(train_df[loaded.config.dataset.dt_col]),
                    "context_active": [0] * len(train_df),
                    "context_label": ["normal_context"] * len(train_df),
                }
            ),
            False,
        ),
    )
    monkeypatch.setattr(
        aa_runtime,
        "_select_uncertainty_predictions",
        lambda **_kwargs: {
            "mean": np.asarray([10.0]),
            "std": np.asarray([0.5]),
            "selected_dropout": np.asarray([0.1]),
            "candidate_mean_grid": np.asarray([[10.0]]),
            "candidate_std_grid": np.asarray([[0.5]]),
            "candidate_dropout_values": np.asarray([0.1]),
            "candidate_samples": {"0.10": [[10.0]]},
        },
    )
    monkeypatch.setattr(
        aa_runtime, "_build_event_memory_bank", lambda **_kwargs: ([], 0)
    )
    monkeypatch.setattr(
        aa_runtime,
        "_build_event_query",
        lambda **_kwargs: {
            "shape_vector": np.asarray([1.0]),
            "event_vector": np.asarray([1.0]),
            "event_score": 5.0,
        },
    )
    monkeypatch.setattr(
        aa_runtime,
        "_retrieve_event_neighbors",
        lambda **_kwargs: {
            "retrieval_attempted": True,
            "retrieval_applied": False,
            "skip_reason": "empty_bank",
            "top_neighbors": [],
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
        },
    )
    monkeypatch.setattr(aa_runtime, "NeuralForecast", _FakeNeuralForecast)

    predictions, *_rest = aa_runtime.predict_aa_forecast_fold(
        loaded,
        job,
        train_df=train_df,
        future_df=future_df,
        run_root=None,
    )

    assert predictions[job.model].tolist() == [10.0]
    assert predictions["aaforecast_retrieval_applied"].tolist() == [False]
    assert predictions["aaforecast_retrieval_skip_reason"].tolist() == ["empty_bank"]


def test_build_retrieval_signature_rejects_missing_star_payload_keys() -> None:
    class _BrokenModel:
        hist_exog_list = ()

        def _compute_star_outputs(self, insample_y, hist_exog):
            del insample_y, hist_exog
            return {"critical_mask": torch.ones(1, 2, 1)}

    with pytest.raises(ValueError, match="requires STAR payload keys"):
        aa_runtime._build_retrieval_signature(
            model=_BrokenModel(),
            transformed_window_df=pd.DataFrame({"target": [1.0, 2.0]}),
            target_col="target",
        )


def test_split_runtime_overrides_moves_retrieval_keys_out_of_model_payload() -> None:
    model_override, retrieval_override = aa_runtime._split_runtime_overrides(
        {
            "season_length": 4,
            "top_k": 3,
            "min_similarity": 0.8,
            "use_shape_key": False,
        }
    )

    assert model_override == {"season_length": 4}
    assert retrieval_override == {
        "top_k": 3,
        "min_similarity": 0.8,
        "use_shape_key": False,
    }


def test_normalize_legacy_best_params_for_replay_maps_aaforecast_top_k() -> None:
    job = SimpleNamespace(model="AAForecast")
    normalized = runtime._normalize_legacy_best_params_for_replay(
        job,
        {"top_k": 0.2, "season_length": 4},
    )

    assert "top_k" not in normalized
    assert normalized["thresh"] == pytest.approx(0.2)
    assert normalized["season_length"] == 4


def test_sanitize_aaforecast_trial_params_enforces_similarity_key() -> None:
    sanitized = runtime._sanitize_aaforecast_trial_params(
        model_name="AAForecast",
        candidate_params={
            "use_shape_key": False,
            "use_event_key": False,
            "season_length": 4,
        },
    )

    assert sanitized["use_shape_key"] is False
    assert sanitized["use_event_key"] is True
    assert sanitized["season_length"] == 4


def test_apply_retrieval_overrides_updates_runtime_retrieval_config() -> None:
    base = aa_runtime._cfg.AAForecastRetrievalConfig(enabled=True)
    patched = aa_runtime._apply_retrieval_overrides(
        base,
        {
            "top_k": 1,
            "event_score_threshold": 400.0,
            "recency_gap_steps": 8,
            "min_similarity": 0.7,
            "temperature": 0.1,
            "blend_floor": 0.0,
            "blend_max": 1.0,
            "use_uncertainty_gate": True,
            "use_shape_key": False,
            "use_event_key": True,
            "event_score_log_bonus_alpha": 0.15,
            "event_score_log_bonus_cap": 0.1,
        },
    )

    assert patched.top_k == 1
    assert patched.event_score_threshold == pytest.approx(400.0)
    assert patched.recency_gap_steps == 8
    assert patched.min_similarity == pytest.approx(0.7)
    assert patched.temperature == pytest.approx(0.1)
    assert patched.blend_floor == pytest.approx(0.0)
    assert patched.blend_max == pytest.approx(1.0)
    assert patched.use_uncertainty_gate is True
    assert patched.use_shape_key is False
    assert patched.use_event_key is True
    assert patched.event_score_log_bonus_alpha == pytest.approx(0.15)
    assert patched.event_score_log_bonus_cap == pytest.approx(0.1)
