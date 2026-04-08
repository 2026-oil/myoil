from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app_config import JobConfig
import plugins.bs_preforcast.runtime as bs_runtime
import runtime_support.runner as residual_runtime
from plugin_contracts.stage_registry import get_stage_plugin


def test_bs_preforcast_plugin_is_registered() -> None:
    import plugins.bs_preforcast.plugin  # noqa: F401
    plugin = get_stage_plugin("bs_preforcast")
    assert plugin is not None
    assert plugin.config_key == "bs_preforcast"


def test_validate_only_bs_preforcast_smoke_fixture_materializes_metadata_shell(
    tmp_path: Path,
) -> None:
    fixture_path = Path("tests/fixtures/bs_preforcast_runtime_smoke.yaml")
    output_root = tmp_path / "bs-preforcast-smoke"

    code = residual_runtime.main(
        [
            "--config",
            str(fixture_path),
            "--validate-only",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    capability = json.loads((output_root / "config" / "capability_report.json").read_text())
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    stage_resolved = output_root / "bs_preforcast" / "config" / "config.resolved.json"
    forecast_artifact = output_root / "bs_preforcast" / "artifacts" / "bs_preforcast_forecasts.csv"

    assert stage_resolved.is_file()
    assert forecast_artifact.is_file()
    assert resolved["bs_preforcast"]["selected_config_path"].endswith(
        "tests/fixtures/bs_preforcast_stage_smoke.yaml"
    )
    assert resolved["bs_preforcast"]["validate_only"] is True
    assert resolved["bs_preforcast"]["stage1_run_roots"] == []
    assert resolved["bs_preforcast"]["stage1_selected_jobs_path"] is None
    assert resolved["bs_preforcast"]["stage1_forecast_artifact_path"] == str(
        forecast_artifact
    )
    assert capability["bs_preforcast"]["enabled"] is True
    assert manifest["bs_preforcast"]["validate_only"] is True
    assert manifest["bs_preforcast"]["stage1_run_roots"] == []
    assert manifest["bs_preforcast"]["stage1_selected_jobs_path"] is None


def test_validate_only_bs_preforcast_uni_catalog_fanout_runs_per_stage_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_repo = tmp_path / "repo"
    shutil.copytree(repo_root / "yaml" / "plugins", tmp_repo / "yaml" / "plugins")
    shutil.copytree(
        repo_root / "yaml" / "jobs" / "bs_preforcast",
        tmp_repo / "yaml" / "jobs" / "bs_preforcast",
    )
    shutil.copytree(repo_root / "yaml" / "HPO", tmp_repo / "yaml" / "HPO")

    data_path = tmp_repo / "data.csv"
    data_path.write_text(
        "dt,target,BS_Core_Index_Integrated\n"
        "2020-01-01,1,10\n"
        "2020-01-08,2,11\n"
        "2020-01-15,3,12\n"
        "2020-01-22,4,13\n",
        encoding="utf-8",
    )
    config_path = tmp_repo / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "task": {"name": "bs_preforcast_uni_catalog_validate"},
                "dataset": {
                    "path": str(data_path),
                    "target_col": "target",
                    "dt_col": "dt",
                    "hist_exog_cols": [],
                    "futr_exog_cols": [],
                    "static_exog_cols": [],
                },
                "bs_preforcast": {
                    "enabled": True,
                    "config_path": "yaml/plugins/bs_preforcast_uni.yaml",
                },
                "training": {
                    "input_size": 2,
                    "val_check_steps": 1,
                    "early_stop_patience_steps": -1,
                    "batch_size": 1,
                    "valid_batch_size": 1,
                    "windows_batch_size": 8,
                    "inference_windows_batch_size": 8,
                    "max_steps": 1,
                    "loss": "mse",
                    "lr_scheduler": {
                        "name": "OneCycleLR",
                        "max_lr": 0.001,
                        "pct_start": 0.3,
                        "div_factor": 25.0,
                        "final_div_factor": 10000.0,
                        "anneal_strategy": "cos",
                        "three_phase": False,
                        "cycle_momentum": False,
                    },
                },
                "cv": {"horizon": 1, "step_size": 1, "n_windows": 1},
                "scheduler": {
                    "gpu_ids": [0],
                    "worker_devices": 1,
                    "parallelize_single_job_tuning": False,
                },
                "jobs": [{"model": "Naive", "params": {}}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        residual_runtime, "__file__", str(tmp_repo / "runtime_support" / "runner.py")
    )

    code = residual_runtime.main(["--config", str(config_path), "--validate-only"])

    assert code == 0
    assert capsys.readouterr().out.strip() == ""


def test_validate_only_bs_preforcast_multi_catalog_fanout_runs_per_stage_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_repo = tmp_path / "repo"
    shutil.copytree(repo_root / "yaml" / "plugins", tmp_repo / "yaml" / "plugins")
    shutil.copytree(
        repo_root / "yaml" / "jobs" / "bs_preforcast",
        tmp_repo / "yaml" / "jobs" / "bs_preforcast",
    )
    shutil.copytree(repo_root / "yaml" / "HPO", tmp_repo / "yaml" / "HPO")

    data_path = tmp_repo / "data.csv"
    data_path.write_text(
        "dt,target,BS_Core_Index_Integrated,Idx_SnPVIX,Bonds_MOVE,Idx_GVZ,Idx_OVX,Idx_DxyUSD,Com_LMEX,Com_BloombergCommodity_BCOM\n"
        "2020-01-01,1,10,1,2,3,4,5,6,7\n"
        "2020-01-08,2,11,2,3,4,5,6,7,8\n"
        "2020-01-15,3,12,3,4,5,6,7,8,9\n"
        "2020-01-22,4,13,4,5,6,7,8,9,10\n",
        encoding="utf-8",
    )
    config_path = tmp_repo / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "task": {"name": "bs_preforcast_multi_catalog_validate"},
                "dataset": {
                    "path": str(data_path),
                    "target_col": "target",
                    "dt_col": "dt",
                    "hist_exog_cols": [],
                    "futr_exog_cols": [],
                    "static_exog_cols": [],
                },
                "bs_preforcast": {
                    "enabled": True,
                    "config_path": "yaml/plugins/bs_preforcast_multi.yaml",
                },
                "training": {
                    "input_size": 2,
                    "val_check_steps": 1,
                    "early_stop_patience_steps": -1,
                    "batch_size": 1,
                    "valid_batch_size": 1,
                    "windows_batch_size": 8,
                    "inference_windows_batch_size": 8,
                    "max_steps": 1,
                    "loss": "mse",
                    "lr_scheduler": {
                        "name": "OneCycleLR",
                        "max_lr": 0.001,
                        "pct_start": 0.3,
                        "div_factor": 25.0,
                        "final_div_factor": 10000.0,
                        "anneal_strategy": "cos",
                        "three_phase": False,
                        "cycle_momentum": False,
                    },
                },
                "cv": {"horizon": 1, "step_size": 1, "n_windows": 1},
                "scheduler": {
                    "gpu_ids": [0],
                    "worker_devices": 1,
                    "parallelize_single_job_tuning": False,
                },
                "jobs": [{"model": "Naive", "params": {}}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        residual_runtime, "__file__", str(tmp_repo / "runtime_support" / "runner.py")
    )

    code = residual_runtime.main(["--config", str(config_path), "--validate-only"])

    assert code == 0
    assert capsys.readouterr().out.strip() == ""


def test_materialize_stage_fails_before_stage_side_effects_for_unsupported_futr_exog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    loaded = SimpleNamespace(
        config=SimpleNamespace(stage_plugin_config=SimpleNamespace(enabled=True)),
        normalized_payload={},
    )
    stage_loaded = SimpleNamespace(
        normalized_payload={
            "bs_preforcast": {
                "config_path": "demo.yaml",
                "target_columns": ["bs_a"],
                "task": {"multivariable": False},
            }
        },
        config=SimpleNamespace(jobs=()),
    )

    monkeypatch.setattr(
        bs_runtime,
        "load_bs_preforcast_stage_config",
        lambda *_args, **_kwargs: stage_loaded,
    )

    def fail_fast(*_args, **_kwargs):
        raise ValueError("unsupported futr_exog main job")

    monkeypatch.setattr(bs_runtime, "_derived_job_injection_results", fail_fast)
    monkeypatch.setattr(bs_runtime, "_write_json", lambda *_args, **_kwargs: calls.append("write_json"))
    monkeypatch.setattr(bs_runtime, "write_manifest", lambda *_args, **_kwargs: calls.append("write_manifest"))
    monkeypatch.setattr(
        bs_runtime,
        "attach_bs_preforcast_stage_metadata",
        lambda *_args, **_kwargs: calls.append("attach_metadata"),
    )

    with pytest.raises(ValueError, match="unsupported futr_exog main job"):
        bs_runtime.materialize_bs_preforcast_stage(
            loaded=loaded,
            selected_jobs=[SimpleNamespace(model="Naive")],
            run_root=tmp_path / "run-root",
            main_resolved_path=tmp_path / "main.resolved.json",
            main_capability_path=tmp_path / "main.capability.json",
            main_manifest_path=tmp_path / "main.manifest.json",
            entrypoint_version="test",
            validate_only=False,
        )

    assert calls == []


def test_predict_stage_univariate_tree_uses_forecaster_direct_and_list_lags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeForecasterDirect:
        def __init__(self, *, regressor, steps, lags):
            captured["regressor"] = regressor
            captured["steps"] = steps
            captured["lags"] = lags

        def fit(self, *, y, **kwargs):
            captured["fit_y"] = list(y)
            captured["fit_kwargs"] = dict(kwargs)

        def predict(self, *, steps):
            captured["predict_steps"] = list(steps)
            return pd.Series([101.0, 102.5])

    fake_direct_module = SimpleNamespace(ForecasterDirect=FakeForecasterDirect)
    fake_skforecast = SimpleNamespace(direct=fake_direct_module)
    monkeypatch.setitem(sys.modules, "skforecast", fake_skforecast)
    monkeypatch.setitem(sys.modules, "skforecast.direct", fake_direct_module)

    stage_loaded = SimpleNamespace(
        config=SimpleNamespace(training=SimpleNamespace(input_size=24))
    )
    job = SimpleNamespace(
        params={
            "lags": [1, 2, 3, 6, 12],
            "n_estimators": 8,
            "max_depth": 3,
            
        }
    )
    train_df = pd.DataFrame({"bs_a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]})
    future_df = pd.DataFrame({"bs_a": [0.0, 0.0]})

    forecasts = bs_runtime._predict_stage_univariate_tree(
        stage_loaded,
        job,
        target_column="bs_a",
        train_df=train_df,
        future_df=future_df,
        model_name="xgboost",
    )

    assert forecasts == [101.0, 102.5]
    assert captured["lags"] == [1, 2, 3, 6, 12]
    assert captured["steps"] == 2
    assert captured["predict_steps"] == [1, 2]


def test_predict_stage_univariate_arima_uses_statsforecast_arima(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeARIMA:
        def __init__(self, **kwargs):
            captured["arima_kwargs"] = dict(kwargs)

    class FakeFitted:
        def predict(self, *, h):
            captured["predict_h"] = h
            return pd.DataFrame(
                {
                    "unique_id": ["bs_a", "bs_a"],
                    "ds": ["2020-01-22", "2020-01-29"],
                    "ARIMA": [11.0, 12.5],
                }
            )

    class FakeStatsForecast:
        def __init__(self, *, models, freq):
            captured["models"] = models
            captured["freq"] = freq

        def fit(self, *, df):
            captured["fit_df_columns"] = list(df.columns)
            captured["fit_df_rows"] = df.to_dict(orient="records")
            return FakeFitted()

    fake_models_module = SimpleNamespace(ARIMA=FakeARIMA)
    fake_statsforecast_module = SimpleNamespace(
        StatsForecast=FakeStatsForecast,
        models=fake_models_module,
    )
    monkeypatch.setitem(sys.modules, "statsforecast", fake_statsforecast_module)
    monkeypatch.setitem(sys.modules, "statsforecast.models", fake_models_module)

    stage_loaded = SimpleNamespace(
        config=SimpleNamespace(
            dataset=SimpleNamespace(dt_col="dt", freq=None),
        )
    )
    job = SimpleNamespace(
        params={
            "order": [1, 1, 0],
            "include_mean": True,
            "include_drift": False,
        }
    )
    train_df = pd.DataFrame(
        {
            "dt": ["2020-01-01", "2020-01-08", "2020-01-15"],
            "bs_a": [10.0, 11.0, 12.0],
        }
    )
    future_df = pd.DataFrame(
        {"dt": ["2020-01-22", "2020-01-29"], "bs_a": [0.0, 0.0]}
    )

    forecasts = bs_runtime._predict_stage_univariate_arima(
        stage_loaded,
        job,
        target_column="bs_a",
        train_df=train_df,
        future_df=future_df,
    )

    assert forecasts == [11.0, 12.5]
    assert captured["arima_kwargs"] == {
        "order": (1, 1, 0),
        "season_length": 1,
        "seasonal_order": (0, 0, 0),
        "include_mean": True,
        "include_drift": False,
        "include_constant": None,
        "blambda": None,
        "biasadj": False,
        "method": "CSS-ML",
    }
    assert captured["predict_h"] == 2


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("ARIMA", {"order": "[1, 1, 0]"}),
        ("xgboost", {"lags": "[1, 2, 3, 6, 12]"}),
    ],
)
def test_normalized_direct_job_params_rejects_stringified_list_literals(
    model_name: str,
    params: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="native YAML list values"):
        bs_runtime._normalized_direct_job_params(model_name, params)


def test_resolved_stage_job_requires_materialized_best_params_for_learned_auto() -> None:
    stage_loaded = SimpleNamespace(
        normalized_payload={"bs_preforcast": {}},
        config=SimpleNamespace(
            jobs=(
                JobConfig(
                    model="TimeXer",
                    params={},
                    requested_mode="learned_auto_requested",
                    validated_mode="learned_auto",
                ),
            ),
            training_search=SimpleNamespace(validated_mode="training_fixed"),
        )
    )

    with pytest.raises(ValueError, match="fixed learned job"):
        bs_runtime._resolved_stage_job(stage_loaded)
