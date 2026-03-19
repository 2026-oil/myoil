from __future__ import annotations

from pathlib import Path
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


def _write_config(tmp_path: Path, payload: dict, suffix: str) -> Path:
    path = tmp_path / f"config{suffix}"
    if suffix == ".toml":
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
final_holdout = 12
overlap_eval_policy = 'by_cutoff_mean'

[scheduler]
gpu_ids = [0, 1]
max_concurrent_jobs = 2
worker_devices = 1

[residual]
enabled = true
train_source = 'oof_cv'
model = 'lstm'

[residual.params]
lookback = 2
epochs = 1

[[jobs]]
model = 'TFT'
params = {}

[[jobs]]
model = 'iTransformer'
params = {}
"""
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
            "final_holdout": 12,
            "overlap_eval_policy": "by_cutoff_mean",
        },
        "scheduler": {"gpu_ids": [0, 1], "max_concurrent_jobs": 2, "worker_devices": 1},
        "residual": {
            "enabled": True,
            "train_source": "oof_cv",
            "model": "lstm",
            "params": {"lookback": 2, "epochs": 1},
        },
        "jobs": [
            {"model": "TFT", "params": {}},
            {"model": "iTransformer", "params": {}},
        ],
    }


def test_toml_and_yaml_normalize_to_same_typed_model(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    yaml_path = _write_config(tmp_path, _payload(), ".yaml")
    toml_path = _write_config(tmp_path, _payload(), ".toml")
    loaded_yaml = load_app_config(tmp_path, config_path=yaml_path)
    loaded_toml = load_app_config(tmp_path, config_toml_path=toml_path)
    assert loaded_yaml.config.to_dict() == loaded_toml.config.to_dict()


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
    payload["cv"].update(
        {"horizon": 1, "step_size": 1, "n_windows": 1, "final_holdout": 1}
    )
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
    assert (output_root / "holdout" / "DummyUnivariate_metrics.csv").exists()


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


def test_residual_registry_builds_lstm_plugin():
    plugin = build_residual_plugin(
        {"model": "lstm", "params": {"lookback": 2, "epochs": 1}}
    )
    assert plugin.metadata()["plugin"] == "lstm"
    assert plugin.metadata()["lookback"] == 2


def test_runtime_generates_residual_artifacts_with_dummy_model(tmp_path: Path):
    payload = _payload()
    payload["cv"].update(
        {"horizon": 1, "step_size": 1, "n_windows": 4, "final_holdout": 1}
    )
    payload["training"].update({"input_size": 1, "max_steps": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "train_source": "oof_cv",
        "model": "lstm",
        "params": {"lookback": 2, "epochs": 1, "hidden_size": 4},
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
    assert (residual_root / "source_oof_cv_resolved.csv").exists()
    assert (residual_root / "corrected_cv.csv").exists()
    assert (residual_root / "corrected_holdout.csv").exists()
    assert (residual_root / "diagnostics.json").exists()
    diagnostics = pd.read_json(residual_root / "diagnostics.json", typ="series")
    assert diagnostics["corrected_cv_mode"] == "strict_oof_runtime"
    assert diagnostics["holdout_truth_included"] is False


class _RecordingResidualPlugin(ResidualPlugin):
    name = "recording"

    def __init__(self, plugin_log: list["_RecordingLog"]):
        self._plugin_log = plugin_log
        self._record: _RecordingLog = {
            "fit_lengths": [],
            "future_inputs": [],
            "predict_train_calls": 0,
        }
        self._plugin_log.append(self._record)

    def fit(self, train_df: pd.DataFrame, context: ResidualContext) -> None:
        self._record["fit_lengths"].append(len(train_df))

    def predict_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self._record["predict_train_calls"] += 1
        raise AssertionError(
            "runtime should not call predict_train for corrected_cv generation"
        )

    def predict_future(self, future_df: pd.DataFrame) -> pd.DataFrame:
        self._record["future_inputs"].append(list(future_df.columns))
        residual_hat = [float(self._record["fit_lengths"][-1])] * len(future_df)
        return future_df.copy().assign(residual_hat=residual_hat)

    def metadata(self) -> dict[str, object]:
        return {"plugin": self.name}


class _RecordingLog(TypedDict):
    fit_lengths: list[int]
    future_inputs: list[list[str]]
    predict_train_calls: int


def test_apply_residual_plugin_builds_corrected_cv_in_memory_and_keeps_holdout_truth_out(
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
    cv_rows = [
        {
            "model": job.model,
            "fold_idx": 0,
            "cutoff": "2020-01-08",
            "unique_id": "target",
            "ds": "2020-01-08",
            "y": 2.0,
            "y_hat": 1.5,
        },
        {
            "model": job.model,
            "fold_idx": 1,
            "cutoff": "2020-01-15",
            "unique_id": "target",
            "ds": "2020-01-15",
            "y": 3.0,
            "y_hat": 2.5,
        },
        {
            "model": job.model,
            "fold_idx": 2,
            "cutoff": "2020-01-22",
            "unique_id": "target",
            "ds": "2020-01-22",
            "y": 4.0,
            "y_hat": 3.5,
        },
    ]
    holdout_df = pd.DataFrame({"dt": ["2020-01-29"], "target": [5.0]})
    target_holdout = pd.DataFrame(
        {"unique_id": ["target"], "ds": ["2020-01-29"], job.model: [4.5]}
    )

    runtime._apply_residual_plugin(
        loaded, job, run_root, cv_rows, holdout_df, target_holdout, job.model, nf=None
    )

    residual_root = run_root / "residual" / job.model
    corrected_cv = pd.read_csv(residual_root / "corrected_cv.csv")
    assert corrected_cv["residual_hat"].tolist() == [0.0, 1.0, 2.0]
    assert corrected_cv["y_hat_corrected"].tolist() == [1.5, 3.5, 5.5]
    assert all(record["predict_train_calls"] == 0 for record in plugin_log)
    assert [record["fit_lengths"][0] for record in plugin_log[:-1]] == [0, 1, 2]
    assert plugin_log[-1]["future_inputs"] == [
        ["model", "unique_id", "ds", "y_hat_base"]
    ]


def test_lstm_plugin_uses_fallback_only_for_short_history(
    monkeypatch: pytest.MonkeyPatch,
):
    plugin = cast(
        Any,
        build_residual_plugin(
            {"model": "lstm", "params": {"lookback": 2, "epochs": 1}}
        ),
    )
    train_df = pd.DataFrame(
        {
            "model": ["TFT", "TFT"],
            "unique_id": ["target", "target"],
            "ds": pd.to_datetime(["2020-01-08", "2020-01-15"]),
            "y": [2.0, 3.0],
            "y_hat_base": [1.5, 2.5],
            "fold_count": [1, 1],
            "source_type": ["oof_cv", "oof_cv"],
            "residual_target": [0.5, 0.5],
        }
    )
    plugin.fit(
        train_df,
        ResidualContext(
            job_name="TFT", model_name="TFT", output_dir=Path("."), config={}
        ),
    )

    def _raise_on_forward(*args, **kwargs):
        raise AssertionError(
            "network forward should not be used for short-history fallback"
        )

    monkeypatch.setattr(plugin.model, "forward", _raise_on_forward)
    future_df = pd.DataFrame(
        {
            "model": ["TFT"],
            "unique_id": ["target"],
            "ds": pd.to_datetime(["2020-01-22"]),
            "y_hat_base": [3.5],
        }
    )
    predicted = plugin.predict_future(future_df)
    assert predicted["residual_hat"].tolist() == [0.5]


def test_runtime_skips_residual_artifacts_for_baseline_models(tmp_path: Path):
    payload = _payload()
    payload["cv"].update(
        {"horizon": 1, "step_size": 1, "n_windows": 2, "final_holdout": 1}
    )
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "train_source": "oof_cv",
        "model": "lstm",
        "params": {"lookback": 2, "epochs": 1},
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
    assert (output_root / "holdout" / "Naive_metrics.csv").exists()
    assert not (output_root / "residual" / "Naive").exists()


def test_repo_config_keeps_baseline_jobs_empty():
    loaded = load_app_config(REPO_ROOT, config_path=REPO_ROOT / "config.yaml")
    params_by_model = {job.model: job.params for job in loaded.config.jobs}
    assert params_by_model["Naive"] == {}
    assert params_by_model["SeasonalNaive"] == {}
    assert params_by_model["HistoricAverage"] == {}


def test_repo_config_sets_explicit_itransformer_fairness_target():
    loaded = load_app_config(REPO_ROOT, config_path=REPO_ROOT / "config.yaml")
    params_by_model = {job.model: job.params for job in loaded.config.jobs}
    assert params_by_model["iTransformer"] == {
        "hidden_size": 128,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 256,
    }
