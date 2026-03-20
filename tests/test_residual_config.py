from __future__ import annotations

from pathlib import Path
import json
from typing import Any, TypedDict, cast

import pandas as pd
import pytest
import yaml

from residual.adapters import build_multivariate_inputs, build_univariate_inputs
from residual.config import load_app_config
from residual.models import build_model
from residual.plugins_base import ResidualContext, ResidualPlugin
from residual.registry import build_residual_plugin
from residual.scheduler import build_launch_plan, worker_env


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_search_space(
    root: Path,
    payload: dict[str, Any] | None = None,
    *,
    name: str = "search_space.yaml",
) -> Path:
    if payload is None:
        payload = {
            "models": {
                "TFT": ["hidden_size", "dropout", "n_head"],
                "iTransformer": ["hidden_size", "n_heads", "e_layers", "d_ff"],
            },
            "residual": {
                "xgboost": ["n_estimators", "max_depth", "learning_rate"]
            },
        }
    path = root / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_config(tmp_path: Path, payload: dict, suffix: str) -> Path:
    path = tmp_path / f"config{suffix}"
    if suffix == ".toml":
        task_block = ""
        task_name = payload.get("task", {}).get("name")
        if task_name:
            task_block = f"\n[task]\nname = {task_name!r}\n"
        text = """
[dataset]
path = 'data.csv'
target_col = 'target'
dt_col = 'dt'
hist_exog_cols = ['hist_a']
futr_exog_cols = []
static_exog_cols = []

[runtime]
random_seed = 1

[training]
input_size = 64
season_length = 52
batch_size = 32
valid_batch_size = 32
windows_batch_size = 1024
inference_windows_batch_size = 1024
learning_rate = 0.001
max_steps = 50
loss = 'mse'

[cv]
horizon = 12
step_size = 4
n_windows = 24
gap = 0
overlap_eval_policy = 'by_cutoff_mean'

[scheduler]
gpu_ids = [0, 1]
max_concurrent_jobs = 2
worker_devices = 1

[residual]
enabled = true
model = 'xgboost'

[residual.params]
n_estimators = 8
max_depth = 2
learning_rate = 0.2

[[jobs]]
model = 'TFT'
params = { hidden_size = 32 }

[[jobs]]
model = 'iTransformer'
params = { hidden_size = 32, n_heads = 4, e_layers = 2, d_ff = 64 }
"""
        text = f"{task_block}{text}"
        path.write_text(text, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _payload() -> dict:
    return {
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["hist_a"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "runtime": {"random_seed": 1},
        "training": {
            "input_size": 64,
            "season_length": 52,
            "batch_size": 32,
            "valid_batch_size": 32,
            "windows_batch_size": 1024,
            "inference_windows_batch_size": 1024,
            "learning_rate": 0.001,
            "max_steps": 50,
            "loss": "mse",
        },
        "cv": {
            "horizon": 12,
            "step_size": 4,
            "n_windows": 24,
            "gap": 0,
            "overlap_eval_policy": "by_cutoff_mean",
        },
        "scheduler": {"gpu_ids": [0, 1], "max_concurrent_jobs": 2, "worker_devices": 1},
        "residual": {
            "enabled": True,
            "model": "xgboost",
            "params": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.2},
        },
        "jobs": [
            {"model": "TFT", "params": {"hidden_size": 32}},
            {
                "model": "iTransformer",
                "params": {"hidden_size": 32, "n_heads": 4, "e_layers": 2, "d_ff": 64},
            },
        ],
    }


def test_toml_and_yaml_normalize_to_same_typed_model(tmp_path: Path):
    payload = _payload()
    payload["task"] = {"name": "semi_test"}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    yaml_path = _write_config(tmp_path, payload, ".yaml")
    toml_path = _write_config(tmp_path, payload, ".toml")
    loaded_yaml = load_app_config(tmp_path, config_path=yaml_path)
    loaded_toml = load_app_config(tmp_path, config_toml_path=toml_path)
    assert loaded_yaml.config.to_dict() == loaded_toml.config.to_dict()


def test_task_name_is_loaded_into_normalized_config(tmp_path: Path):
    payload = _payload()
    payload["task"] = {"name": "semi_test"}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    assert loaded.config.task.name == "semi_test"
    assert loaded.normalized_payload["task"] == {"name": "semi_test"}


def test_adapters_materialize_expected_frames(tmp_path: Path):
    payload = _payload()
    yaml_path = _write_config(tmp_path, payload, ".yaml")
    source_path = tmp_path / "data.csv"
    source_path.write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,10,2\n2020-01-08,2,11,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(tmp_path, config_path=yaml_path)
    import pandas as pd

    source_df = pd.read_csv(source_path)
    univariate = build_univariate_inputs(
        source_df, loaded.config.jobs[0], dataset=loaded.config.dataset, dt_col="dt"
    )
    multivariate = build_multivariate_inputs(
        source_df, loaded.config.jobs[1], dataset=loaded.config.dataset, dt_col="dt"
    )
    assert list(univariate.fit_df.columns) == ["unique_id", "ds", "y", "hist_a"]
    assert multivariate.channel_map == {"target": 0, "hist_a": 1}


def test_model_builder_applies_common_loss_and_multivariate_n_series(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, _payload(), ".yaml")
    )
    univariate_model = build_model(loaded.config, loaded.config.jobs[0])
    multivariate_model = build_model(loaded.config, loaded.config.jobs[1], n_series=2)
    assert type(univariate_model.loss).__name__ == "MSE"
    assert type(multivariate_model.loss).__name__ == "MSE"
    assert getattr(multivariate_model, "n_series", 2) == 2


def test_model_builder_propagates_centralized_training_controls(tmp_path: Path):
    payload = _payload()
    payload["training"].update(
        {
            "max_steps": 17,
            "learning_rate": 0.123,
            "val_check_steps": 7,
            "early_stop_patience_steps": 11,
        }
    )
    payload["jobs"] = [
        {"model": "TFT", "params": {"hidden_size": 32}},
        {
            "model": "LSTM",
            "params": {"encoder_hidden_size": 32, "decoder_hidden_size": 32},
        },
        {"model": "NHITS", "params": {"mlp_units": [[32, 32], [32, 32], [32, 32]]}},
        {
            "model": "iTransformer",
            "params": {"hidden_size": 32, "n_heads": 4, "e_layers": 2, "d_ff": 64},
        },
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    tft = build_model(loaded.config, loaded.config.jobs[0])
    lstm = build_model(loaded.config, loaded.config.jobs[1])
    nhits = build_model(loaded.config, loaded.config.jobs[2])
    itransformer = build_model(loaded.config, loaded.config.jobs[3], n_series=2)

    for model in (tft, lstm, nhits, itransformer):
        assert model.hparams.max_steps == 17
        assert model.hparams.learning_rate == pytest.approx(0.123)
        assert model.hparams.val_check_steps == 7
        assert model.hparams.early_stop_patience_steps == 11


def test_scheduler_plan_and_worker_env_use_single_device(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, _payload(), ".yaml")
    )
    launches = build_launch_plan(loaded.config, loaded.config.jobs)
    assert [launch.gpu_id for launch in launches] == [0, 1]
    assert all(launch.devices == 1 for launch in launches)
    env = worker_env(1)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["NEURALFORECAST_WORKER_DEVICES"] == "1"


def test_runtime_executes_single_job_with_dummy_model(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual.runtime import main as runtime_main

    output_root = tmp_path / "run"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    assert (output_root / "cv" / "DummyUnivariate_forecasts.csv").exists()
    assert not (output_root / "holdout").exists()
    capability = json.loads((output_root / "config" / "capability_report.json").read_text())
    assert capability["DummyUnivariate"]["supports_auto"] is False


def test_runtime_uses_task_name_for_default_run_directory(tmp_path: Path):
    payload = _payload()
    payload["task"] = {"name": "pytest_task_default_output"}
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual.runtime import main as runtime_main

    code = runtime_main(["--config", str(config_path), "--jobs", "DummyUnivariate"])

    output_root = REPO_ROOT / "runs" / "pytest_task_default_output"
    try:
        assert code == 0
        assert (output_root / "cv" / "DummyUnivariate_forecasts.csv").exists()
        resolved_config = output_root / "config" / "config.resolved.json"
        assert resolved_config.exists()
        assert "pytest_task_default_output" in resolved_config.read_text(
            encoding="utf-8"
        )
    finally:
        if output_root.exists():
            import shutil

            shutil.rmtree(output_root)


def test_runtime_infers_freq_when_omitted(tmp_path: Path):
    payload = _payload()
    payload["dataset"].pop("freq", None)
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-06,1,9\n2020-01-13,2,8\n2020-01-20,3,7\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    import residual.runtime as runtime

    loaded = load_app_config(tmp_path, config_path=config_path)
    import pandas as pd

    source_df = pd.read_csv(tmp_path / "data.csv")
    assert runtime._resolve_freq(loaded, source_df) == "W-MON"


def test_jobs_use_unique_model_identifiers(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, _payload(), ".yaml")
    )
    assert [job.model for job in loaded.config.jobs] == ["TFT", "iTransformer"]


def test_load_app_config_rejects_duplicate_centralized_job_keys(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {"max_steps": 123}}]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    path = _write_config(tmp_path, payload, ".yaml")
    with pytest.raises(ValueError, match="repeats centralized training key"):
        load_app_config(tmp_path, config_path=path)


def test_load_app_config_rejects_removed_residual_train_source(tmp_path: Path):
    payload = _payload()
    payload["residual"]["train_source"] = "oof_cv"
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    path = _write_config(tmp_path, payload, ".yaml")
    with pytest.raises(ValueError, match="residual.train_source has been removed"):
        load_app_config(tmp_path, config_path=path)


def test_load_app_config_rejects_removed_cv_final_holdout(tmp_path: Path):
    payload = _payload()
    payload["cv"]["final_holdout"] = 12
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    path = _write_config(tmp_path, payload, ".yaml")
    with pytest.raises(ValueError, match="cv.final_holdout has been removed"):
        load_app_config(tmp_path, config_path=path)


def test_residual_registry_builds_xgboost_plugin():
    plugin = build_residual_plugin(
        {
            "model": "xgboost",
            "params": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.2},
        }
    )
    assert plugin.metadata()["plugin"] == "xgboost"
    assert plugin.metadata()["n_estimators"] == 8


def test_runtime_generates_per_fold_residual_artifacts_with_dummy_model(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 4, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.2},
    }
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual.runtime import main as runtime_main

    output_root = tmp_path / "run_resid"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    residual_root = output_root / "residual" / "DummyUnivariate"
    assert not (residual_root / "training_panel.csv").exists()
    assert not (residual_root / "corrected_holdout.csv").exists()
    assert (residual_root / "corrected_folds.csv").exists()
    assert (residual_root / "diagnostics.json").exists()
    for fold_idx in range(4):
        fold_root = residual_root / "folds" / f"fold_{fold_idx:03d}"
        assert (fold_root / "backcast_panel.csv").exists()
        assert (fold_root / "corrected_eval.csv").exists()
        assert (fold_root / "residual_checkpoint" / "model.ubj").exists()
        assert (fold_root / "base_checkpoint" / "fit_summary.json").exists()
    corrected_folds = pd.read_csv(residual_root / "corrected_folds.csv")
    assert corrected_folds["fold_idx"].tolist() == [0, 1, 2, 3]
    assert "panel_split" in corrected_folds.columns
    assert set(corrected_folds["panel_split"]) == {"fold_eval"}
    diagnostics = pd.read_json(residual_root / "diagnostics.json", typ="series")
    assert diagnostics["corrected_eval_mode"] == "per_fold_backcast_runtime"
    assert diagnostics["fold_count"] == 4
    assert diagnostics["tscv_policy"]["gap"] == 0


class _RecordingResidualPlugin(ResidualPlugin):
    name = "recording"

    def __init__(self, plugin_log: list["_RecordingLog"]):
        self._plugin_log = plugin_log
        self._record: _RecordingLog = {
            "fit_lengths": [],
            "predict_panel_splits": [],
        }
        self._plugin_log.append(self._record)

    def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
        self._record["fit_lengths"].append(len(panel_df))

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        self._record["predict_panel_splits"].append(
            panel_df["panel_split"].unique().tolist()
        )
        return panel_df.copy().assign(residual_hat=0.0)

    def metadata(self) -> dict[str, object]:
        return {"plugin": self.name}


class _RecordingLog(TypedDict):
    fit_lengths: list[int]
    predict_panel_splits: list[list[str]]


def test_apply_residual_plugin_uses_fold_local_backcasts_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from residual import runtime

    plugin_log: list[_RecordingLog] = []

    def _build_plugin(_config):
        return _RecordingResidualPlugin(plugin_log)

    monkeypatch.setattr(runtime, "build_residual_plugin", _build_plugin)
    loaded = load_app_config(REPO_ROOT, config_path=REPO_ROOT / "config.yaml")
    job = next(job for job in loaded.config.jobs if job.model == "TFT")
    run_root = tmp_path / "run"
    fold_payloads = []
    for fold_idx, fit_length in enumerate((1, 2, 3)):
        backcast_panel = pd.DataFrame(
            {
                "model_name": [job.model] * fit_length,
                "fold_idx": [fold_idx] * fit_length,
                "panel_split": ["backcast_train"] * fit_length,
                "unique_id": ["target"] * fit_length,
                "cutoff": pd.to_datetime(["2020-01-08"] * fit_length),
                "train_end_ds": pd.to_datetime(["2020-01-08"] * fit_length),
                "ds": pd.to_datetime(["2020-01-15"] * fit_length),
                "horizon_step": list(range(1, fit_length + 1)),
                "y_hat_base": [1.0] * fit_length,
                "y": [1.5] * fit_length,
                "residual_target": [0.5] * fit_length,
            }
        )
        eval_panel = pd.DataFrame(
            {
                "model_name": [job.model],
                "fold_idx": [fold_idx],
                "panel_split": ["fold_eval"],
                "unique_id": ["target"],
                "cutoff": pd.to_datetime(["2020-01-15"]),
                "train_end_ds": pd.to_datetime(["2020-01-15"]),
                "ds": pd.to_datetime(["2020-01-22"]),
                "horizon_step": [1],
                "y_hat_base": [2.0],
                "y": [2.5],
                "residual_target": [0.5],
            }
        )
        fold_payloads.append(
            {
                "fold_idx": fold_idx,
                "backcast_panel": backcast_panel,
                "eval_panel": eval_panel,
                "base_summary": {"fold_idx": fold_idx},
            }
        )

    runtime._apply_residual_plugin(loaded, job, run_root, fold_payloads)

    assert [record["fit_lengths"][0] for record in plugin_log] == [1, 2, 3]
    assert all(
        splits == ["fold_eval"]
        for record in plugin_log
        for splits in record["predict_panel_splits"]
    )
    residual_root = run_root / "residual" / job.model
    assert (residual_root / "corrected_folds.csv").exists()
    assert not (residual_root / "corrected_holdout.csv").exists()


def test_xgboost_plugin_predicts_panel_and_writes_checkpoint(tmp_path: Path):
    plugin = cast(
        Any,
        build_residual_plugin(
            {
                "model": "xgboost",
                "params": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.2},
            }
        ),
    )
    train_df = pd.DataFrame(
        {
            "model_name": ["TFT", "TFT", "TFT"],
            "fold_idx": [0, 0, 0],
            "panel_split": ["backcast_train", "backcast_train", "backcast_train"],
            "unique_id": ["target", "target", "target"],
            "cutoff": pd.to_datetime(["2020-01-08", "2020-01-08", "2020-01-08"]),
            "train_end_ds": pd.to_datetime(["2020-01-08", "2020-01-08", "2020-01-08"]),
            "ds": pd.to_datetime(["2020-01-15", "2020-01-22", "2020-01-29"]),
            "horizon_step": [1, 2, 3],
            "y_hat_base": [1.5, 2.5, 3.5],
            "y": [2.0, 3.0, 4.0],
            "residual_target": [0.5, 0.5, 0.5],
        }
    )
    plugin.fit(
        train_df,
        ResidualContext(
            job_name="TFT",
            model_name="TFT",
            output_dir=tmp_path / "checkpoint",
            config={},
        ),
    )
    feature_frame = plugin._feature_frame(train_df)
    assert "fold_idx" not in feature_frame.columns
    assert list(feature_frame.columns) == [
        "horizon_step",
        "y_hat_base",
        "cutoff_day",
        "ds_day",
    ]
    predicted = plugin.predict(train_df.drop(columns=["residual_target"]))
    assert "residual_hat" in predicted.columns
    assert len(predicted) == 3
    assert (tmp_path / "checkpoint" / "model.ubj").exists()


def test_runtime_skips_residual_artifacts_for_baseline_models(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.2},
    }
    payload["jobs"] = [{"model": "Naive", "params": {}}]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual.runtime import main as runtime_main

    output_root = tmp_path / "run_baseline"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "Naive",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    assert (output_root / "cv" / "Naive_forecasts.csv").exists()
    assert not (output_root / "holdout").exists()
    assert not (output_root / "residual" / "Naive").exists()


def test_repo_config_keeps_only_naive_baseline_job():
    loaded = load_app_config(REPO_ROOT, config_path=REPO_ROOT / "config.yaml")
    params_by_model = {job.model: job.params for job in loaded.config.jobs}
    assert params_by_model["Naive"] == {}
    assert "SeasonalNaive" not in params_by_model
    assert "HistoricAverage" not in params_by_model


def test_repo_config_sets_explicit_itransformer_fairness_target():
    loaded = load_app_config(REPO_ROOT, config_path=REPO_ROOT / "config.yaml")
    params_by_model = {job.model: job.params for job in loaded.config.jobs}
    assert params_by_model["iTransformer"] == {
        "hidden_size": 128,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 256,
    }


def test_repo_config_conservative_fairness_matrix():
    loaded = load_app_config(REPO_ROOT, config_path=REPO_ROOT / "config.yaml")
    params_by_model = {job.model: job.params for job in loaded.config.jobs}

    for model_name in (
        "TFT",
        "VanillaTransformer",
        "Informer",
        "Autoformer",
        "FEDformer",
        "PatchTST",
        "iTransformer",
    ):
        assert params_by_model[model_name]["hidden_size"] == 128

    assert params_by_model["LSTM"]["encoder_hidden_size"] == 128
    assert params_by_model["LSTM"]["decoder_hidden_size"] == 128

    assert params_by_model["PatchTST"]["n_heads"] == 16
    assert params_by_model["PatchTST"]["patch_len"] == 16
    assert params_by_model["FEDformer"]["modes"] == 64
    assert params_by_model["NHITS"] == {
        "mlp_units": [[64, 64], [64, 64], [64, 64]],
        "n_pool_kernel_size": [2, 2, 1],
        "n_freq_downsample": [4, 2, 1],
        "dropout_prob_theta": 0.0,
        "activation": "ReLU",
    }


def test_model_builder_does_not_alias_api_distinct_keys(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [
        {"model": "TFT", "params": {"hidden_size": 32, "encoder_hidden_size": 999}},
        {
            "model": "LSTM",
            "params": {
                "hidden_size": 999,
                "encoder_hidden_size": 32,
                "decoder_hidden_size": 48,
            },
        },
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    tft = build_model(loaded.config, loaded.config.jobs[0])
    lstm = build_model(loaded.config, loaded.config.jobs[1])

    assert tft.hparams.hidden_size == 32
    assert getattr(tft.hparams, "encoder_hidden_size", 999) == 999

    assert lstm.hparams.encoder_hidden_size == 32
    assert lstm.hparams.decoder_hidden_size == 48
    assert getattr(lstm.hparams, "hidden_size", 999) == 999


def test_readme_documents_conservative_fairness_policy():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "baseline (`Naive`, `SeasonalNaive`, `HistoricAverage`)은 fairness normalization 대상이 아닙니다."
        in readme
    )
    assert (
        "`training:`에 있는 공통 key를 `jobs[*].params`에 다시 쓰면 안 됩니다."
        in readme
    )
    assert (
        "API가 다른 key들 사이에는 aliasing이나 canonicalization을 하지 않습니다."
        in readme
    )
    assert "PatchTST.n_heads" in readme
    assert "FEDformer.modes" in readme


def test_load_app_config_marks_auto_requested_and_validated_modes(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": True, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(tmp_path)

    loaded = load_app_config(tmp_path, config_path=config_path)

    job = loaded.config.jobs[0]
    assert job.requested_mode == "learned_auto_requested"
    assert job.validated_mode == "learned_auto"
    assert list(job.selected_search_params) == ["hidden_size", "dropout", "n_head"]
    assert loaded.config.residual.requested_mode == "residual_auto_requested"
    assert loaded.config.residual.validated_mode == "residual_auto"
    assert loaded.normalized_payload["search_space_path"] == str(
        (tmp_path / "search_space.yaml").resolve()
    )
    assert loaded.normalized_payload["search_space_sha256"]


def test_load_app_config_rejects_missing_model_search_space_entry(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {"models": {"iTransformer": ["hidden_size"]}, "residual": {"xgboost": ["n_estimators"]}},
    )

    with pytest.raises(ValueError, match="requires search_space.models.TFT"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_unknown_search_space_param(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {"models": {"TFT": ["encoder_layers"]}, "residual": {"xgboost": ["n_estimators"]}},
    )

    with pytest.raises(ValueError, match="unknown parameter"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_stale_unused_allowlisted_selector_entry(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"], "PatchTST": ["not_a_real_key"]},
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match="PatchTST contains unknown parameter"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_uses_repo_root_search_space_not_config_parent(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_dir = tmp_path / "config_dir"
    repo_root.mkdir()
    config_dir.mkdir()
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (config_dir / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    payload["dataset"]["path"] = str((config_dir / "data.csv").resolve())
    config_path = _write_config(config_dir, payload, ".yaml")
    _write_search_space(repo_root)
    _write_search_space(
        config_dir,
        {"models": {"TFT": ["hidden_size"]}, "residual": {"xgboost": ["max_depth"]}},
    )

    loaded = load_app_config(repo_root, config_path=config_path)

    assert loaded.search_space_path == (repo_root / "search_space.yaml").resolve()
    assert list(loaded.config.jobs[0].selected_search_params) == [
        "hidden_size",
        "dropout",
        "n_head",
    ]


def test_runtime_auto_mode_records_selector_provenance_and_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
        "2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(tmp_path)

    from residual import runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "1")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_auto"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    capability = json.loads(
        (output_root / "config" / "capability_report.json").read_text()
    )
    manifest = json.loads(
        (output_root / "manifest" / "run_manifest.json").read_text()
    )

    assert resolved["search_space_path"] == str((tmp_path / "search_space.yaml").resolve())
    assert resolved["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert resolved["jobs"][0]["validated_mode"] == "learned_auto"
    assert capability["TFT"]["requested_mode"] == "learned_auto_requested"
    assert capability["TFT"]["validated_mode"] == "learned_auto"
    assert manifest["search_space_path"] == str((tmp_path / "search_space.yaml").resolve())
    assert manifest["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert manifest["jobs"][0]["validated_mode"] == "learned_auto"
    assert manifest["jobs"][0]["model_best_params_path"]
    assert manifest["jobs"][0]["model_optuna_study_summary_path"]
    assert (output_root / "models" / "TFT" / "best_params.json").exists()
    assert (output_root / "models" / "TFT" / "optuna_study_summary.json").exists()
