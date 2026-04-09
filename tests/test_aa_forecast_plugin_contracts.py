"""Contract tests for aa_forecast plugin fail-fast and config helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from app_config import load_app_config
from plugins.aa_forecast.config import (
    AAForecastPluginConfig,
    aa_forecast_plugin_state_dict,
    aa_forecast_stage_document_type,
)
from plugins.aa_forecast.plugin import AAForecastStagePlugin
import plugins.aa_forecast.runtime as aa_runtime
import runtime_support.forecast_models as forecast_models
import runtime_support.runner as runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_AUTO_MODEL_ONLY_MAIN_CONFIG = Path(
    "tests/fixtures/aa_forecast_runtime_plugin_auto_model_only_main.yaml"
)
SUPPORTED_AA_BACKBONES = [
    "gru",
    "informer",
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
    assert payload["backbone"] == "gru"
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
        match=r"aa_forecast\.model must be one of: .*gru.*informer.*patchtst.*timexer.*vanillatransformer",
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


def test_fit_and_predict_fold_passes_trial_overrides_to_aa_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = SimpleNamespace(
        config=SimpleNamespace(
            dataset=SimpleNamespace(dt_col="dt", target_col="target")
        )
    )
    job = SimpleNamespace(model="AAForecast")
    source_df = pd.DataFrame(
        {"dt": ["2020-01-01", "2020-01-08"], "target": [1.0, 2.0]}
    )
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
        params_override={"season_length": 4, "top_k": 0.01},
        training_override={"model_step_size": 4, "scaler_type": "standard"},
    )

    assert captured["train_rows"] == 1
    assert captured["future_rows"] == 1
    assert captured["params_override"] == {"season_length": 4, "top_k": 0.01}
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
            params_override={"season_length": 4, "top_k": 0.01},
            training_override={"model_step_size": 4, "scaler_type": "standard"},
        )
    )

    assert captured["training_model_step_size"] == 4
    assert captured["training_scaler_type"] == "standard"
    assert captured["params_override"]["season_length"] == 4
    assert captured["params_override"]["top_k"] == pytest.approx(0.01)
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
    assert str(train_end_ds) == str(pd.Timestamp(train_df[loaded.config.dataset.dt_col].iloc[-1]))
    assert used_train_df.equals(train_df)
