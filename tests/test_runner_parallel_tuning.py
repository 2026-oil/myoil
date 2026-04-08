from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pandas as pd
import pytest

import runtime_support.runner as runtime


def test_sync_study_roots_replaces_stale_projection_inputs(tmp_path: Path) -> None:
    source_models_dir = tmp_path / "scheduler" / "models" / "AAForecast"
    target_models_dir = tmp_path / "run" / "models" / "AAForecast"

    source_study_1 = source_models_dir / "studies" / "study-01"
    source_study_2 = source_models_dir / "studies" / "study-02"
    source_study_1.mkdir(parents=True, exist_ok=True)
    source_study_2.mkdir(parents=True, exist_ok=True)
    (source_study_1 / "best_params.json").write_text('{"trial": 1}', encoding="utf-8")
    (source_study_2 / "best_params.json").write_text('{"trial": 2}', encoding="utf-8")

    stale_root = target_models_dir / "studies" / "study-01"
    stale_root.mkdir(parents=True, exist_ok=True)
    (stale_root / "stale.txt").write_text("old", encoding="utf-8")

    runtime._sync_study_roots(
        source_models_dir,
        target_models_dir,
        study_indices=(1, 2),
    )

    assert (target_models_dir / "studies" / "study-01" / "best_params.json").read_text(
        encoding="utf-8"
    ) == '{"trial": 1}'
    assert (target_models_dir / "studies" / "study-02" / "best_params.json").read_text(
        encoding="utf-8"
    ) == '{"trial": 2}'
    assert not (target_models_dir / "studies" / "study-01" / "stale.txt").exists()


def test_parallel_tuning_syncs_scheduler_studies_before_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    manifest_path = run_root / "manifest" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    loaded = SimpleNamespace(
        config=SimpleNamespace(
            scheduler=SimpleNamespace(max_concurrent_jobs=2),
        )
    )
    job = SimpleNamespace(model="AAForecast", validated_mode="learned_auto")
    selection = SimpleNamespace(
        execute_study_indices=(1, 2),
        canonical_projection_study_index=1,
    )
    launches = [
        SimpleNamespace(
            job_name="AAForecast",
            devices=1,
            phase="tune-main-only",
            worker_index=0,
            selected_study=1,
        )
    ]

    monkeypatch.setattr(
        runtime,
        "_should_prune_model_run_artifacts",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(runtime, "resolve_study_selection", lambda _loaded: selection)
    monkeypatch.setattr(
        runtime,
        "loaded_with_study_selection_override",
        lambda current_loaded, _study_index: current_loaded,
    )
    monkeypatch.setattr(runtime, "build_device_groups", lambda _config: [(0,), (1,)])
    monkeypatch.setattr(
        runtime,
        "build_tuning_launch_plan",
        lambda *_args, **_kwargs: launches,
    )

    def _fake_run_parallel_jobs(_repo_root, _study_loaded, _launches, study_scheduler_dir):
        scheduler_models_dir = study_scheduler_dir.parent / "models" / "AAForecast"
        source_study_root = scheduler_models_dir / "studies" / study_scheduler_dir.name
        source_study_root.mkdir(parents=True, exist_ok=True)
        (source_study_root / "best_params.json").write_text(
            f'{{"source":"{study_scheduler_dir.name}"}}',
            encoding="utf-8",
        )
        return [{"returncode": 0}]

    monkeypatch.setattr(runtime, "run_parallel_jobs", _fake_run_parallel_jobs)

    replay_calls: list[str] = []

    def _fake_run_single_job(
        _projection_loaded,
        _job,
        replay_run_root,
        *,
        manifest_path: Path,
        main_stage: str,
    ) -> None:
        assert manifest_path.name == "run_manifest.json"
        assert main_stage == "replay-only"
        synced_file = (
            replay_run_root
            / "models"
            / "AAForecast"
            / "studies"
            / "study-01"
            / "best_params.json"
        )
        replay_calls.append(synced_file.read_text(encoding="utf-8"))

    monkeypatch.setattr(runtime, "_run_single_job", _fake_run_single_job)

    worker_results = runtime._run_single_job_with_parallel_tuning(
        Path("/repo"),
        loaded,
        job,
        run_root,
        manifest_path=manifest_path,
    )

    assert worker_results == [{"returncode": 0}, {"returncode": 0}]
    assert replay_calls == ['{"source":"study-01"}']


class _FakeSaveableModel:
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"checkpoint")


class _FakeTrial:
    def __init__(self, *, should_prune_steps: set[int] | None = None) -> None:
        self.number = 0
        self._should_prune_steps = should_prune_steps or set()
        self._last_step: int | None = None
        self.user_attrs: dict[str, object] = {}

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value

    def report(self, _value: float, step: int) -> None:
        self._last_step = step

    def should_prune(self) -> bool:
        return self._last_step in self._should_prune_steps


def _build_fake_loaded() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(opt_n_trial=2),
            stage_plugin_config=SimpleNamespace(enabled=False),
            training_search=SimpleNamespace(selected_search_params=()),
        ),
        search_space_payload=None,
    )


def _build_fake_job() -> SimpleNamespace:
    return SimpleNamespace(
        model="DummyUnivariate",
        selected_search_params=("input_size",),
        params={},
        validated_mode="learned_auto",
    )


def _build_fake_study_context(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        study_root=tmp_path / "studies" / "study-01",
        study_index=1,
        study_name="study-01",
        proposal_flow_id="proposal-001",
        sampler_seed=7,
    )


def _build_fold_return() -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, object]:
    predictions = pd.DataFrame(
        {
            "unique_id": ["target", "target"],
            "ds": pd.to_datetime(["2024-01-05", "2024-01-06"]),
            "DummyUnivariate": [10.0, 11.0],
            "DummyUnivariate__uncertainty_std": [0.1, 0.2],
            "dummy_context_flag": [True, False],
        }
    )
    actuals = pd.Series([9.5, 10.5], dtype=float)
    train_df = pd.DataFrame(
        {
            "unique_id": ["target"] * 4,
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "y": [7.0, 8.0, 8.5, 9.0],
        }
    )
    return predictions, actuals, pd.Timestamp("2024-01-04"), train_df, _FakeSaveableModel()


def test_main_job_objective_pruned_trial_keeps_completed_fold_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _build_fake_loaded()
    job = _build_fake_job()
    study_context = _build_fake_study_context(tmp_path)
    source_df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=6, freq="D"), "y": range(6)})
    splits = [([0, 1, 2, 3], [4, 5]), ([0, 1, 2, 3, 4], [5])]

    monkeypatch.setattr(runtime, "suggest_model_params", lambda *_a, **_k: {"width": 8})
    monkeypatch.setattr(runtime, "suggest_training_params", lambda *_a, **_k: {"batch_size": 4})
    monkeypatch.setattr(runtime, "training_param_registry_for_model", lambda *_a, **_k: None)
    monkeypatch.setattr(runtime, "optuna_num_trials", lambda *_a, **_k: 2)
    monkeypatch.setattr(runtime, "_fit_and_predict_fold", lambda *_a, **_k: _build_fold_return())

    objective = runtime._main_job_objective(
        loaded,
        job,
        study_context=study_context,
        source_df=source_df,
        freq="D",
        splits=splits,
    )
    trial = _FakeTrial(should_prune_steps={0})

    with pytest.raises(optuna.TrialPruned):
        objective(trial)

    trial_dir = study_context.study_root / "trials" / "trial-0000"
    fold_root = trial_dir / "folds" / "fold_000"
    result_payload = json.loads((trial_dir / "trial_result.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((fold_root / "metrics.json").read_text(encoding="utf-8"))
    prediction_frame = pd.read_csv(fold_root / "predictions.csv")

    assert result_payload["status"] == "pruned"
    assert (fold_root / "plot.png").exists()
    assert (fold_root / "checkpoint.pt").exists()
    assert {"MAE", "MSE", "RMSE", "MAPE", "NRMSE", "R2"}.issubset(metrics_payload)
    assert {"y", "y_hat", "y_hat_uncertainty_std", "dummy_context_flag"}.issubset(
        prediction_frame.columns
    )
    assert (trial_dir / "predictions.csv").exists()


def test_main_job_objective_failed_trial_keeps_prior_fold_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _build_fake_loaded()
    job = _build_fake_job()
    study_context = _build_fake_study_context(tmp_path)
    source_df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=6, freq="D"), "y": range(6)})
    splits = [([0, 1, 2, 3], [4, 5]), ([0, 1, 2, 3, 4], [5])]

    monkeypatch.setattr(runtime, "suggest_model_params", lambda *_a, **_k: {"width": 8})
    monkeypatch.setattr(runtime, "suggest_training_params", lambda *_a, **_k: {"batch_size": 4})
    monkeypatch.setattr(runtime, "training_param_registry_for_model", lambda *_a, **_k: None)
    monkeypatch.setattr(runtime, "optuna_num_trials", lambda *_a, **_k: 2)

    call_count = {"value": 0}

    def _fake_fit(*_args, **_kwargs):
        if call_count["value"] == 0:
            call_count["value"] += 1
            return _build_fold_return()
        raise ValueError("boom on fold 1")

    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit)

    objective = runtime._main_job_objective(
        loaded,
        job,
        study_context=study_context,
        source_df=source_df,
        freq="D",
        splits=splits,
    )
    trial = _FakeTrial()

    with pytest.raises(runtime._OptunaTrialFailure, match="boom on fold 1"):
        objective(trial)

    trial_dir = study_context.study_root / "trials" / "trial-0000"
    fold_root = trial_dir / "folds" / "fold_000"
    result_payload = json.loads((trial_dir / "trial_result.json").read_text(encoding="utf-8"))

    assert result_payload["status"] == "failed"
    assert "boom on fold 1" in result_payload["failure_reason"]
    assert (fold_root / "predictions.csv").exists()
    assert (fold_root / "plot.png").exists()
    assert (fold_root / "metrics.json").exists()
    assert (fold_root / "checkpoint.pt").exists()
