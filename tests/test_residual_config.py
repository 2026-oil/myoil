from __future__ import annotations

import io
from pathlib import Path
import json
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import pandas as pd
import pytest
import yaml

from residual.adapters import build_multivariate_inputs, build_univariate_inputs
from residual.config import load_app_config
from residual.models import (
    MODEL_CLASSES,
    build_model,
    supports_auto_mode,
)
from residual.optuna_spaces import (
    EXCLUDED_AUTO_MODEL_NAMES,
    MODEL_PARAM_REGISTRY,
    SUPPORTED_AUTO_MODEL_NAMES,
    TRAINING_PARAM_REGISTRY,
)
from residual.plugins_base import ResidualContext, ResidualPlugin
from residual.progress import PROGRESS_EVENT_PREFIX
from residual.registry import build_residual_plugin
from residual.scheduler import build_launch_plan, run_parallel_jobs, worker_env


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
            "training": [],
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


def test_model_builder_disables_logger_and_keeps_timemixer_without_future_features(
    tmp_path: Path,
):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["dataset"]["futr_exog_cols"] = []
    payload["jobs"] = [
        {
            "model": "TimeMixer",
            "params": {
                "d_model": 16,
                "d_ff": 32,
                "down_sampling_layers": 1,
                "top_k": 3,
            },
        }
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    timemixer = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert timemixer.trainer_kwargs["logger"] is False
    assert timemixer.trainer_kwargs["enable_progress_bar"] is False
    assert timemixer.use_future_temporal_feature == 0


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
    assert env["NEURALFORECAST_PROGRESS_MODE"] == "structured"
    assert env["NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS"] == "1"


def test_scheduler_respects_max_concurrent_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    payload = _payload()
    payload["jobs"] = [
        {"model": "TFT", "params": {"hidden_size": 32}},
        {
            "model": "LSTM",
            "params": {"encoder_hidden_size": 16, "decoder_hidden_size": 16},
        },
        {"model": "NHITS", "params": {"mlp_units": [[8, 8], [8, 8], [8, 8]]}},
    ]
    payload["scheduler"]["max_concurrent_jobs"] = 2
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )
    launches = build_launch_plan(loaded.config, loaded.config.jobs)
    state = {"active": 0, "max_active": 0}

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            self._completed = False
            self.stdout = io.StringIO("")

        def poll(self):
            return 0 if not self._completed else 0

        def wait(self):
            self._completed = True
            state["active"] -= 1
            return 0

    monkeypatch.setattr("residual.scheduler.subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "residual.scheduler._worker_command",
        lambda *_args, **_kwargs: ["python", "fake_worker.py"],
    )

    results = run_parallel_jobs(tmp_path, loaded, launches, tmp_path / "scheduler")

    assert len(results) == 3
    assert state["max_active"] == 2


def test_scheduler_streams_worker_stdout_to_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, _payload(), ".yaml")
    )
    launch = build_launch_plan(loaded.config, loaded.config.jobs[:1])[0]

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self._completed = False
            self.stdout = io.StringIO(
                f"{PROGRESS_EVENT_PREFIX}"
                + json.dumps(
                    {
                        "job_name": "TFT",
                        "model_index": 1,
                        "total_models": 1,
                        "total_steps": 2,
                        "completed_steps": 1,
                        "total_folds": 2,
                        "current_fold": 0,
                        "phase": "replay",
                        "status": "running",
                        "detail": "mse=1.0000",
                        "event": "fold-done",
                        "progress_pct": 50,
                        "progress_text": "[#########---------] 1/2  50%",
                    }
                )
                + "\nworker note\n"
            )

        def poll(self):
            return 0

        def wait(self):
            self._completed = True
            return 0

    monkeypatch.setattr("residual.scheduler.subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "residual.scheduler._worker_command",
        lambda *_args, **_kwargs: ["python", "fake_worker.py"],
    )

    run_parallel_jobs(tmp_path, loaded, [launch], tmp_path / "scheduler")

    captured = capsys.readouterr()
    assert "[summary]" in captured.out
    assert "[model:TFT]" in captured.out
    assert "completed=1/1" in captured.out
    assert "[worker:TFT] worker note" in captured.out
    assert PROGRESS_EVENT_PREFIX not in captured.out


def test_runtime_executes_single_job_with_dummy_model(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
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


def test_runtime_logs_model_and_fold_progress_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual.runtime import main as runtime_main

    output_root = tmp_path / "run_progress"
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

    captured = capsys.readouterr()
    assert code == 0
    assert "[summary] completed=0/1" in captured.out
    assert "[summary] completed=1/1" in captured.out
    assert "[model:DummyUnivariate]" in captured.out
    assert "phase=replay" in captured.out
    assert "fold=2/2" in captured.out
    assert "100%" in captured.out


def test_runtime_logs_fold_errors_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual import runtime

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _boom)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--jobs",
                "DummyUnivariate",
                "--output-root",
                str(tmp_path / "run_error"),
            ]
        )

    captured = capsys.readouterr()
    assert "[model:DummyUnivariate]" in captured.out
    assert "status=failed" in captured.out
    assert "synthetic failure" in captured.out


def test_summary_builder_writes_leaderboard_and_last_fold_plots(tmp_path: Path):
    from residual import runtime

    run_root = tmp_path / "summary_run"
    cv_dir = run_root / "cv"
    cv_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "fold_idx": 0,
                "cutoff": "2020-01-15",
                "MAE": 1.0,
                "MSE": 1.0,
                "RMSE": 1.0,
                "MAPE": 0.1,
            },
            {
                "fold_idx": 1,
                "cutoff": "2020-01-22",
                "MAE": 0.5,
                "MSE": 0.25,
                "RMSE": 0.5,
                "MAPE": 0.05,
            },
        ]
    ).to_csv(cv_dir / "ModelA_metrics_by_cutoff.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_idx": 0,
                "cutoff": "2020-01-15",
                "MAE": 2.0,
                "MSE": 4.0,
                "RMSE": 2.0,
                "MAPE": 0.2,
            },
            {
                "fold_idx": 1,
                "cutoff": "2020-01-22",
                "MAE": 1.5,
                "MSE": 2.25,
                "RMSE": 1.5,
                "MAPE": 0.15,
            },
        ]
    ).to_csv(cv_dir / "ModelB_metrics_by_cutoff.csv", index=False)
    for model_name, values in {
        "ModelA": [10.0, 11.0],
        "ModelB": [9.5, 10.5],
    }.items():
        pd.DataFrame(
            [
                {
                    "model": model_name,
                    "fold_idx": 1,
                    "cutoff": "2020-01-22",
                    "train_end_ds": "2020-01-22",
                    "unique_id": "target",
                    "ds": "2020-01-29",
                    "horizon_step": 1,
                    "y": 10.0,
                    "y_hat": values[0],
                },
                {
                    "model": model_name,
                    "fold_idx": 1,
                    "cutoff": "2020-01-22",
                    "train_end_ds": "2020-01-22",
                    "unique_id": "target",
                    "ds": "2020-02-05",
                    "horizon_step": 2,
                    "y": 11.0,
                    "y_hat": values[1],
                },
            ]
        ).to_csv(cv_dir / f"{model_name}_forecasts.csv", index=False)

    artifacts = runtime._build_summary_artifacts(run_root)

    leaderboard_path = run_root / "summary" / "leaderboard.csv"
    assert artifacts["leaderboard"] == str(leaderboard_path)
    assert leaderboard_path.exists()
    leaderboard = pd.read_csv(leaderboard_path)
    assert leaderboard.loc[0, "rank"] == 1
    assert leaderboard.loc[0, "model"] == "ModelA"
    assert "mean_fold_mape" in leaderboard.columns
    assert (run_root / "summary" / "last_fold_all_models.png").exists()
    assert (run_root / "summary" / "last_fold_top3.png").exists()
    assert (run_root / "summary" / "last_fold_top5.png").exists()


def test_runtime_skip_summary_env_suppresses_summary_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from residual.runtime import main as runtime_main

    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    output_root = tmp_path / "worker_like_run"

    monkeypatch.setenv("NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS", "1")

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
    assert not (output_root / "summary").exists()


def test_runtime_smoke_writes_summary_artifacts_for_dummy_model(tmp_path: Path):
    from residual.runtime import main as runtime_main

    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")
    output_root = tmp_path / "run_summary"

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

    leaderboard_path = output_root / "summary" / "leaderboard.csv"
    assert code == 0
    assert leaderboard_path.exists()
    assert (output_root / "summary" / "last_fold_all_models.png").exists()
    assert (output_root / "summary" / "last_fold_top3.png").exists()
    assert (output_root / "summary" / "last_fold_top5.png").exists()
    leaderboard = pd.read_csv(leaderboard_path)
    assert "rank" in leaderboard.columns


def test_trajectory_frame_contains_train_and_val_series():
    from residual import runtime

    nf = SimpleNamespace(
        models=[
            SimpleNamespace(
                train_trajectories=[(1, 0.9), (2, 0.7)],
                valid_trajectories=[(1, 1.1), (2, 0.8)],
            )
        ]
    )

    frame = runtime._trajectory_frame(nf)

    assert frame.columns.tolist() == ["global_step", "train_loss", "val_loss"]
    assert frame.to_dict(orient="records") == [
        {"global_step": 1, "train_loss": 0.9, "val_loss": 1.1},
        {"global_step": 2, "train_loss": 0.7, "val_loss": 0.8},
    ]


def test_build_tscv_splits_uses_configured_step_size(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 3, "step_size": 2, "n_windows": 3, "gap": 0})
    payload["dataset"]["hist_exog_cols"] = []
    (tmp_path / "data.csv").write_text(
        "dt,target\n"
        "2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
        "2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n"
        "2020-02-26,9\n2020-03-04,10\n2020-03-11,11\n2020-03-18,12\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    from residual import runtime

    splits = runtime._build_tscv_splits(12, loaded.config.cv)

    assert [test_idx for _, test_idx in splits] == [
        [5, 6, 7],
        [7, 8, 9],
        [9, 10, 11],
    ]


def test_runtime_outer_cv_cutoffs_follow_step_size(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 2, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual.runtime import main as runtime_main

    output_root = tmp_path / "run_step_size_outer_cv"
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

    forecasts = pd.read_csv(output_root / "cv" / "DummyUnivariate_forecasts.csv")

    assert code == 0
    assert forecasts["cutoff"].tolist() == [
        "2020-01-15 00:00:00",
        "2020-01-29 00:00:00",
    ]


def test_runtime_uses_task_name_for_default_run_directory(tmp_path: Path):
    payload = _payload()
    payload["task"] = {"name": "pytest_task_default_output"}
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
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
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
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


def test_runtime_writes_loss_curve_images_for_residual_disabled_learned_folds(
    tmp_path: Path,
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual.runtime import main as runtime_main

    output_root = tmp_path / "run_loss_curves"
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
    for fold_idx in range(2):
        assert (
            output_root
            / "models"
            / "DummyUnivariate"
            / "folds"
            / f"fold_{fold_idx:03d}"
            / "loss_curve.png"
        ).exists()


def test_runtime_writes_loss_curve_images_for_residual_enabled_learned_folds(
    tmp_path: Path,
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.2},
    }
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual.runtime import main as runtime_main

    output_root = tmp_path / "run_loss_curves_residual"
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
    for fold_idx in range(2):
        assert (
            output_root
            / "models"
            / "DummyUnivariate"
            / "folds"
            / f"fold_{fold_idx:03d}"
            / "loss_curve.png"
        ).exists()
        assert (
            output_root
            / "residual"
            / "DummyUnivariate"
            / "folds"
            / f"fold_{fold_idx:03d}"
            / "base_checkpoint"
            / "fit_summary.json"
        ).exists()


def test_baseline_models_do_not_write_loss_curve_images(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["jobs"] = [{"model": "Naive", "params": {}}]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n"
    )
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
    assert not (output_root / "models" / "Naive" / "folds").exists()


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

    manifest_path = run_root / "manifest" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": [{"model": job.model}], "residual": {}}, indent=2),
        encoding="utf-8",
    )

    runtime._apply_residual_plugin(
        loaded, job, run_root, fold_payloads, manifest_path=manifest_path
    )

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
    assert loaded.config.training_search.requested_mode == "training_fixed"
    assert loaded.config.training_search.validated_mode == "training_fixed"
    assert list(loaded.config.training_search.selected_search_params) == []
    assert loaded.normalized_payload["search_space_path"] == str(
        (tmp_path / "search_space.yaml").resolve()
    )
    assert loaded.normalized_payload["search_space_sha256"]


def test_load_app_config_accepts_training_search_space_section(tmp_path: Path):
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
            "models": {"TFT": ["hidden_size", "dropout", "n_head"]},
            "training": ["input_size", "max_steps"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.config.training_search.requested_mode == "training_auto_requested"
    assert loaded.config.training_search.validated_mode == "training_auto"
    assert list(loaded.config.training_search.selected_search_params) == [
        "input_size",
        "max_steps",
    ]


def test_load_app_config_accepts_training_search_space_max_steps(tmp_path: Path):
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
            "models": {"TFT": ["hidden_size"]},
            "training": ["max_steps"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert list(loaded.config.training_search.selected_search_params) == ["max_steps"]


def test_load_app_config_rejects_unknown_training_search_space_param(tmp_path: Path):
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
            "models": {"TFT": ["hidden_size"]},
            "training": ["not_a_real_training_key"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match="search_space.training contains unknown"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_training_model_learning_rate_overlap(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "NLinear", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"NLinear": ["learning_rate"]},
            "training": ["learning_rate"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match="training.learning_rate overlaps"):
        load_app_config(tmp_path, config_path=config_path)


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
    assert resolved["training_search"]["requested_mode"] == "training_fixed"
    assert resolved["training_search"]["validated_mode"] == "training_fixed"
    assert capability["TFT"]["requested_mode"] == "learned_auto_requested"
    assert capability["TFT"]["validated_mode"] == "learned_auto"
    assert capability["training_search"]["validated_mode"] == "training_fixed"
    assert manifest["search_space_path"] == str((tmp_path / "search_space.yaml").resolve())
    assert manifest["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert manifest["jobs"][0]["validated_mode"] == "learned_auto"
    assert manifest["training_search"]["validated_mode"] == "training_fixed"
    assert manifest["jobs"][0]["model_best_params_path"]
    assert manifest["jobs"][0]["model_optuna_study_summary_path"]
    assert (output_root / "models" / "TFT" / "best_params.json").exists()
    assert (output_root / "models" / "TFT" / "optuna_study_summary.json").exists()


def test_runtime_auto_mode_records_training_selector_provenance_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["max_steps"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    from residual import runtime

    calls: list[dict[str, Any]] = []

    def _fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
    ):
        calls.append(training_override or {})
        ds = pd.Series(["2020-01-22"])
        predictions = pd.DataFrame(
            {"unique_id": [loaded.config.dataset.target_col], "ds": ds, job.model: [1.0]}
        )
        actuals = pd.Series([1.0])
        return predictions, actuals, pd.Timestamp("2020-01-15"), source_df.iloc[train_idx], object()

    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "1")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit_and_predict_fold)
    monkeypatch.setattr(
        runtime,
        "suggest_model_params",
        lambda *_args, **_kwargs: {"hidden_size": 32},
    )
    monkeypatch.setattr(
        runtime,
        "suggest_training_params",
        lambda *_args, **_kwargs: {"max_steps": 123},
    )
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_auto_training"
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

    manifest = json.loads(
        (output_root / "manifest" / "run_manifest.json").read_text(encoding="utf-8")
    )
    capability = json.loads(
        (output_root / "config" / "capability_report.json").read_text(encoding="utf-8")
    )
    training_best = json.loads(
        (output_root / "models" / "TFT" / "training_best_params.json").read_text(
            encoding="utf-8"
        )
    )

    assert code == 0
    assert calls[-1]["max_steps"] == 123
    assert manifest["training_search"]["selected_search_params"] == [
        "max_steps",
    ]
    assert manifest["jobs"][0]["training_best_params_path"]
    assert manifest["jobs"][0]["training_optuna_study_summary_path"]
    assert capability["training_search"]["validated_mode"] == "training_auto"
    assert training_best == {"max_steps": 123}


def test_effective_config_pins_val_size_to_horizon_for_training_auto(tmp_path: Path):
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
            "models": {"TFT": ["hidden_size"]},
            "training": ["max_steps"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    from residual import runtime

    loaded = load_app_config(tmp_path, config_path=config_path)
    effective = runtime._effective_config(loaded, {"max_steps": 123})

    assert effective.training.max_steps == 123
    assert effective.training.val_size == loaded.config.cv.horizon


def test_supported_auto_model_matrix_matches_registry_and_yaml():
    search_space = yaml.safe_load((REPO_ROOT / "search_space.yaml").read_text())
    learned_model_classes = {
        model_name
        for model_name in MODEL_CLASSES
        if not model_name.startswith("Dummy")
    }
    learned_registry_models = set(MODEL_PARAM_REGISTRY)
    assert "HINT" not in SUPPORTED_AUTO_MODEL_NAMES
    assert SUPPORTED_AUTO_MODEL_NAMES == learned_model_classes
    assert SUPPORTED_AUTO_MODEL_NAMES == learned_registry_models
    assert SUPPORTED_AUTO_MODEL_NAMES == set(search_space["models"])
    assert tuple(search_space["training"]) == tuple(TRAINING_PARAM_REGISTRY)
    assert set(search_space["residual"]) == {"xgboost"}
    assert all(
        isinstance(search_space["models"][model], list)
        for model in search_space["models"]
    )
    assert "learning_rate" not in search_space["models"]["NLinear"]


def test_supported_auto_model_matrix_includes_v3_expansion():
    for model_name in (
        "RNN",
        "GRU",
        "TCN",
        "DeepAR",
        "DilatedRNN",
        "BiTCN",
        "xLSTM",
        "MLP",
        "NBEATS",
        "NBEATSx",
        "DLinear",
        "NLinear",
        "TiDE",
        "DeepNPTS",
        "KAN",
        "TimeLLM",
        "TimeXer",
        "TimesNet",
        "NonstationaryTransformer",
        "StemGNN",
        "TSMixer",
        "TSMixerx",
        "MLPMultivariate",
        "SOFTS",
        "TimeMixer",
        "Mamba",
        "SMamba",
        "CMamba",
        "xLSTMMixer",
        "RMoK",
        "XLinear",
    ):
        assert model_name in SUPPORTED_AUTO_MODEL_NAMES
    assert EXCLUDED_AUTO_MODEL_NAMES == {"HINT"}


def test_supports_auto_mode_expands_to_newly_added_models():
    for model_name in (
        "RNN",
        "DeepAR",
        "TimeMixer",
        "XLinear",
        "StemGNN",
        "TimeLLM",
        "NonstationaryTransformer",
        "Mamba",
        "SMamba",
        "CMamba",
        "xLSTMMixer",
    ):
        assert supports_auto_mode(model_name) is True
    assert supports_auto_mode("Naive") is False


def test_should_use_multivariate_for_no_exog_multivariate_model(tmp_path: Path):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyMultivariate", "params": {"start_padding_enabled": True}}
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import residual.runtime as runtime

    assert runtime._should_use_multivariate(loaded, loaded.config.jobs[0]) is True


def test_should_use_univariate_adapter_for_multivariate_model_with_native_exog(
    tmp_path: Path,
):
    payload = _payload()
    payload["jobs"] = [
        {
            "model": "TimeXer",
            "params": {
                "patch_len": 1,
                "hidden_size": 8,
                "n_heads": 1,
                "e_layers": 1,
                "d_ff": 16,
            },
        }
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,10\n2020-01-08,2,11\n2020-01-15,3,12\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import residual.runtime as runtime

    assert runtime._should_use_multivariate(loaded, loaded.config.jobs[0]) is False


def test_build_model_supports_representative_expanded_models(tmp_path: Path):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {
            "model": "RNN",
            "params": {
                "encoder_hidden_size": 16,
                "encoder_n_layers": 1,
                "context_size": 5,
                "decoder_hidden_size": 16,
            },
        },
        {
            "model": "MLP",
            "params": {"hidden_size": 32, "num_layers": 2},
        },
        {
            "model": "TimeMixer",
            "params": {
                "d_model": 16,
                "d_ff": 32,
                "down_sampling_layers": 1,
                "top_k": 3,
            },
        },
        {
            "model": "MLPMultivariate",
            "params": {"hidden_size": 32, "num_layers": 2},
        },
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    rnn = build_model(loaded.config, loaded.config.jobs[0])
    mlp = build_model(loaded.config, loaded.config.jobs[1])
    timemixer = build_model(loaded.config, loaded.config.jobs[2], n_series=1)
    mlpmulti = build_model(loaded.config, loaded.config.jobs[3], n_series=1)

    assert rnn.hparams.encoder_hidden_size == 16
    assert mlp.hparams.hidden_size == 32
    assert timemixer.hparams.d_model == 16
    assert getattr(mlpmulti.hparams, "n_series", 1) == 1


def test_runtime_executes_multivariate_model_without_hist_exog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["dataset"]["futr_exog_cols"] = []
    payload["jobs"] = [{"model": "iTransformer", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    _write_search_space(tmp_path)
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
        "2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from residual import runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "1")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_multivariate"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "iTransformer",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    assert (output_root / "cv" / "iTransformer_forecasts.csv").exists()


@pytest.mark.parametrize(
    ("source_name", "derived_name"),
    [
        ("baseline-wti.yaml", "baseline-wti_uni.yaml"),
        ("baseline-brentoil.yaml", "baseline-brentoil_uni.yaml"),
    ],
)
def test_generate_baseline_uni_yaml_files_exist_and_parse(
    source_name: str, derived_name: str
):
    source = yaml.safe_load((REPO_ROOT / source_name).read_text(encoding="utf-8"))
    derived_path = REPO_ROOT / derived_name
    derived = yaml.safe_load(derived_path.read_text(encoding="utf-8"))

    assert derived_path.exists()
    assert isinstance(derived, dict)
    assert derived["task"] == source["task"]
    assert derived["dataset"]["target_col"] == source["dataset"]["target_col"]


@pytest.mark.parametrize(
    "derived_name",
    ["baseline-wti_uni.yaml", "baseline-brentoil_uni.yaml"],
)
def test_uni_yaml_exogenous_lists_are_empty(derived_name: str):
    derived = yaml.safe_load((REPO_ROOT / derived_name).read_text(encoding="utf-8"))

    assert derived["dataset"]["hist_exog_cols"] == []
    assert derived["dataset"]["futr_exog_cols"] == []
    assert derived["dataset"]["static_exog_cols"] == []


@pytest.mark.parametrize(
    ("source_name", "derived_name"),
    [
        ("baseline-wti.yaml", "baseline-wti_uni.yaml"),
        ("baseline-brentoil.yaml", "baseline-brentoil_uni.yaml"),
    ],
)
def test_uni_yaml_jobs_follow_capability_rule(
    source_name: str, derived_name: str
):
    from residual.models import capabilities_for as resolve_capabilities

    source = yaml.safe_load((REPO_ROOT / source_name).read_text(encoding="utf-8"))
    derived = yaml.safe_load((REPO_ROOT / derived_name).read_text(encoding="utf-8"))

    source_jobs = [job["model"] for job in source["jobs"]]
    derived_jobs = [job["model"] for job in derived["jobs"]]

    expected_jobs = []
    dropped_jobs = []
    for model_name in source_jobs:
        caps = resolve_capabilities(model_name)
        needs_exog = (
            caps.supports_hist_exog
            or caps.supports_futr_exog
            or caps.supports_stat_exog
        )
        if needs_exog:
            dropped_jobs.append(model_name)
        else:
            expected_jobs.append(model_name)

    assert derived_jobs == expected_jobs
    assert set(derived_jobs).issubset(source_jobs)
    assert set(derived_jobs).isdisjoint(dropped_jobs)


@pytest.mark.parametrize(
    ("source_name", "derived_name"),
    [
        ("baseline-wti.yaml", "baseline-wti_uni.yaml"),
        ("baseline-brentoil.yaml", "baseline-brentoil_uni.yaml"),
    ],
)
def test_uni_yaml_preserves_non_exog_fields(
    source_name: str, derived_name: str
):
    source = yaml.safe_load((REPO_ROOT / source_name).read_text(encoding="utf-8"))
    derived = yaml.safe_load((REPO_ROOT / derived_name).read_text(encoding="utf-8"))

    source_dataset = dict(source["dataset"])
    derived_dataset = dict(derived["dataset"])
    for key in ("hist_exog_cols", "futr_exog_cols", "static_exog_cols"):
        source_dataset.pop(key, None)
        derived_dataset.pop(key, None)

    source_without_jobs = dict(source)
    derived_without_jobs = dict(derived)
    source_without_jobs["dataset"] = source_dataset
    derived_without_jobs["dataset"] = derived_dataset
    source_without_jobs.pop("jobs", None)
    derived_without_jobs.pop("jobs", None)

    assert derived_without_jobs == source_without_jobs


def test_source_baseline_yaml_files_unchanged():
    wti = yaml.safe_load((REPO_ROOT / "baseline-wti.yaml").read_text(encoding="utf-8"))
    brent = yaml.safe_load(
        (REPO_ROOT / "baseline-brentoil.yaml").read_text(encoding="utf-8")
    )

    assert len(wti["dataset"]["hist_exog_cols"]) > 0
    assert len(brent["dataset"]["hist_exog_cols"]) > 0
    assert any(job["model"] == "LSTM" for job in wti["jobs"])
    assert any(job["model"] == "LSTM" for job in brent["jobs"])


CASE_YAML_FILES = [
    REPO_ROOT / 'brentoil-case1.yaml',
    REPO_ROOT / 'brentoil-case2.yaml',
    REPO_ROOT / 'brentoil-case3.yaml',
    REPO_ROOT / 'brentoil-case4.yaml',
    REPO_ROOT / 'wti-case1.yaml',
    REPO_ROOT / 'wti-case2.yaml',
    REPO_ROOT / 'wti-case3.yaml',
    REPO_ROOT / 'wti-case4.yaml',
]

EXPECTED_CASE_TRAINING = {
    'input_size': 64,
    'season_length': 52,
    'batch_size': 32,
    'valid_batch_size': 32,
    'windows_batch_size': 1024,
    'inference_windows_batch_size': 1024,
    'learning_rate': 0.001,
    'max_steps': 1000,
    'val_size': 8,
    'val_check_steps': 100,
    'train_protocol': 'expanding_window_tscv',
    'early_stop_patience_steps': -1,
    'loss': 'mse',
}

EXPECTED_CASE_MODEL_PARAMS = {
    'LSTM': {
        'encoder_hidden_size': 64,
        'decoder_hidden_size': 64,
        'encoder_n_layers': 2,
        'context_size': 10,
    },
    'NHITS': {
        'n_pool_kernel_size': [2, 2, 1],
        'n_freq_downsample': [24, 12, 1],
        'dropout_prob_theta': 0.0,
    },
    'DLinear': {'moving_avg_window': 7},
    'Autoformer': {
        'hidden_size': 64,
        'dropout': 0.1,
        'factor': 3,
        'n_head': 4,
    },
    'FEDformer': {
        'hidden_size': 64,
        'modes': 32,
        'dropout': 0.1,
        'n_head': 8,
    },
    'PatchTST': {
        'hidden_size': 64,
        'n_heads': 4,
        'encoder_layers': 2,
        'patch_len': 16,
        'dropout': 0.1,
    },
    'iTransformer': {
        'hidden_size': 64,
        'n_heads': 4,
        'e_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
    },
    'TimesNet': {
        'hidden_size': 64,
        'conv_hidden_size': 64,
        'top_k': 5,
        'encoder_layers': 2,
    },
    'Naive': {},
}

EXPECTED_CASE_METADATA = {
    'brentoil-case1.yaml': {
        'task_name': 'brentoil_case1',
        'target_col': 'Com_BrentCrudeOil',
        'hist_exog_cols': [
            'Com_Gasoline', 'Com_Steel', 'Bonds_US_Spread_10Y_1Y',
            'Bonds_CHN_Spread_30Y_5Y', 'EX_USD_BRL', 'Com_Cheese',
            'Bonds_BRZ_Spread_10Y_1Y', 'Com_Cu_Gold_Ratio', 'Idx_OVX',
            'Com_Oil_Spread', 'Com_LME_Zn_Spread', 'Idx_CSI300',
            'Bonds_CHN_Spread_5Y_1Y', 'Com_LME_Cu_Spread',
            'Com_LME_Pb_Spread', 'Com_LME_Al_Spread',
        ],
    },
    'brentoil-case2.yaml': {
        'task_name': 'brentoil_case2',
        'target_col': 'Com_BrentCrudeOil',
        'hist_exog_cols': [
            'Com_Gasoline', 'Com_BloombergCommodity_BCOM', 'Com_LME_Ni_Cash',
            'Com_Coal', 'Com_Cotton', 'Com_LME_Al_Cash', 'Bonds_KOR_10Y',
            'Com_Barley', 'Com_Canola', 'Com_LMEX', 'Com_LME_Ni_Inv',
            'Com_Corn', 'Com_Wheat',
        ],
    },
    'brentoil-case3.yaml': {
        'task_name': 'brentoil_case3',
        'target_col': 'Com_BrentCrudeOil',
        'hist_exog_cols': [
            'Com_Gasoline', 'Com_BloombergCommodity_BCOM', 'Com_LME_Ni_Cash',
            'Com_Coal', 'Com_LME_Al_Cash', 'Bonds_KOR_10Y', 'Com_LMEX',
            'Com_LME_Ni_Inv',
        ],
    },
    'brentoil-case4.yaml': {
        'task_name': 'brentoil_case4',
        'target_col': 'Com_BrentCrudeOil',
        'hist_exog_cols': [
            'Com_Gasoline', 'Com_BloombergCommodity_BCOM', 'Com_LME_Ni_Cash',
            'Com_Coal', 'Com_Cotton', 'Com_LME_Al_Cash', 'Bonds_KOR_10Y',
            'Com_Barley', 'Com_Canola', 'Com_LMEX', 'Com_LME_Ni_Inv',
            'Com_Corn', 'Com_Wheat', 'Com_NaturalGas', 'Idx_OVX', 'Com_Gold',
        ],
    },
    'wti-case1.yaml': {
        'task_name': 'wti_case1',
        'target_col': 'Com_CrudeOil',
        'hist_exog_cols': [
            'Com_Gasoline', 'Com_LME_Zn_Inv', 'Com_OrangeJuice', 'Com_Cheese',
            'Bonds_BRZ_1Y', 'Idx_OVX', 'Com_Cu_Gold_Ratio', 'Com_LME_Sn_Inv',
            'Idx_CSI300', 'Com_LME_Zn_Spread', 'Bonds_CHN_Spread_5Y_2Y',
            'Com_LME_Al_Spread', 'Bonds_CHN_Spread_2Y_1Y', 'Com_Oil_Spread',
            'Bonds_CHN_Spread_10Y_5Y',
        ],
    },
    'wti-case2.yaml': {
        'task_name': 'wti_case2',
        'target_col': 'Com_CrudeOil',
        'hist_exog_cols': [
            'Com_Gasoline', 'Com_BloombergCommodity_BCOM', 'Com_LME_Ni_Cash',
            'Com_Coal', 'Com_Canola', 'Com_Cotton', 'Com_LME_Al_Cash',
            'Com_LMEX', 'Bonds_KOR_10Y', 'Com_PalmOil', 'Com_Barley',
            'Com_Corn', 'Com_Oat', 'Com_Wheat', 'Com_Soybeans',
            'Com_LME_Ni_Inv',
        ],
    },
    'wti-case3.yaml': {
        'task_name': 'wti_case3',
        'target_col': 'Com_CrudeOil',
        'hist_exog_cols': [
            'Com_Gasoline', 'Com_BloombergCommodity_BCOM', 'Com_LME_Ni_Cash',
            'Com_Coal', 'Com_LME_Al_Cash', 'Com_LMEX', 'Bonds_KOR_10Y',
            'Com_LME_Ni_Inv',
        ],
    },
    'wti-case4.yaml': {
        'task_name': 'wti_case4',
        'target_col': 'Com_CrudeOil',
        'hist_exog_cols': [
            'Com_Gasoline', 'Com_BloombergCommodity_BCOM', 'Com_LME_Ni_Cash',
            'Com_Coal', 'Com_Canola', 'Com_Cotton', 'Com_LME_Al_Cash',
            'Com_LMEX', 'Bonds_KOR_10Y', 'Com_PalmOil', 'Com_Barley',
            'Com_Corn', 'Com_Oat', 'Com_Wheat', 'Com_Soybeans',
            'Com_LME_Ni_Inv', 'Com_NaturalGas', 'Idx_OVX', 'Com_Gold',
        ],
    },
}


def _load_case_yaml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding='utf-8')))


def _case_jobs_by_model(path: Path) -> dict[str, dict[str, Any]]:
    return {
        job['model']: job['params']
        for job in _load_case_yaml(path)['jobs']
    }


def test_case_yaml_training_mapping_matches_expected_across_all_files():
    for path in CASE_YAML_FILES:
        payload = _load_case_yaml(path)
        assert payload['training'] == EXPECTED_CASE_TRAINING


@pytest.mark.parametrize('path', CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_learned_model_params_match_expected(path: Path):
    jobs = _case_jobs_by_model(path)

    for model_name, expected_params in EXPECTED_CASE_MODEL_PARAMS.items():
        if model_name == 'Naive':
            continue
        assert jobs[model_name] == expected_params
        assert jobs[model_name]


@pytest.mark.parametrize('path', CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_naive_params_remain_empty(path: Path):
    jobs = _case_jobs_by_model(path)
    assert jobs['Naive'] == {}


@pytest.mark.parametrize('path', CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_feature_lists_and_targets_do_not_drift(path: Path):
    payload = _load_case_yaml(path)
    expected = EXPECTED_CASE_METADATA[path.name]

    assert payload['task']['name'] == expected['task_name']
    assert payload['dataset']['target_col'] == expected['target_col']
    assert payload['dataset']['hist_exog_cols'] == expected['hist_exog_cols']


@pytest.mark.parametrize('path', CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_normalizes_to_fixed_modes_without_auto(path: Path):
    loaded = load_app_config(REPO_ROOT, config_path=path)

    for job in loaded.config.jobs:
        if job.model == 'Naive':
            assert job.requested_mode == 'baseline_fixed'
            assert job.validated_mode == 'baseline_fixed'
        else:
            assert job.params == EXPECTED_CASE_MODEL_PARAMS[job.model]
            assert job.requested_mode == 'learned_fixed'
            assert job.validated_mode == 'learned_fixed'

    assert all(job.validated_mode != 'learned_auto' for job in loaded.config.jobs)


def test_case_yaml_build_model_preserves_expected_fixed_params():
    loaded = load_app_config(REPO_ROOT, config_path=REPO_ROOT / 'brentoil-case1.yaml')

    for job in loaded.config.jobs:
        if job.model == 'Naive':
            continue
        model = build_model(
            loaded.config,
            job,
            n_series=1 if job.model == 'iTransformer' else None,
        )
        for key, expected_value in EXPECTED_CASE_MODEL_PARAMS[job.model].items():
            actual_value = getattr(model, key, None)
            if actual_value is None and hasattr(model, 'hparams'):
                actual_value = getattr(model.hparams, key, None)
            assert actual_value == expected_value
