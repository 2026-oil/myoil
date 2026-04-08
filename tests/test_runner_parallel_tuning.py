from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
