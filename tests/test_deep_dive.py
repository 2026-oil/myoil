from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "deep_dive.py"
SPEC = importlib.util.spec_from_file_location("deep_dive", MODULE_PATH)
assert SPEC and SPEC.loader
DEEP_DIVE = importlib.util.module_from_spec(SPEC)
sys.modules["deep_dive"] = DEEP_DIVE
SPEC.loader.exec_module(DEEP_DIVE)


def _search_space_payload() -> dict[str, Any]:
    return {
        "models": {
            "TimeXer": {
                "patch_len": {"type": "categorical", "choices": [8, 16]},
                "hidden_size": {"type": "categorical", "choices": [256, 512]},
                "n_heads": {"type": "categorical", "choices": [16, 32]},
                "e_layers": {"type": "categorical", "choices": [4, 8]},
                "dropout": {"type": "categorical", "choices": [0.1, 0.2]},
            },
            "TSMixerx": {
                "n_block": {"type": "categorical", "choices": [2, 4]},
                "ff_dim": {"type": "categorical", "choices": [64, 256]},
                "dropout": {"type": "categorical", "choices": [0.1, 0.3]},
                "revin": {"type": "categorical", "choices": [True, False]},
            },
            "iTransformer": {
                "hidden_size": {"type": "categorical", "choices": [256, 512]},
                "n_heads": {"type": "categorical", "choices": [16, 32]},
                "e_layers": {"type": "categorical", "choices": [4, 8]},
                "d_ff": {"type": "categorical", "choices": [256, 512]},
                "dropout": {"type": "categorical", "choices": [0.1, 0.3]},
            },
            "LSTM": {
                "encoder_hidden_size": {"type": "categorical", "choices": [64, 128]},
                "decoder_hidden_size": {"type": "categorical", "choices": [64, 128]},
                "encoder_n_layers": {"type": "categorical", "choices": [2, 3]},
                "context_size": {"type": "categorical", "choices": [10, 20]},
            },
        },
        "training": {
            "global": {
                "input_size": {"type": "categorical", "choices": [64]},
                "batch_size": {"type": "categorical", "choices": [32]},
                "learning_rate": {"type": "categorical", "choices": [0.001]},
                "scaler_type": {"type": "categorical", "choices": [None]},
            },
            "per_model": {},
        },
        "residual": {
            "xgboost": {"n_estimators": {"type": "categorical", "choices": [8]}}
        },
    }


def _base_payload(task_name: str, target: str, *, bs: bool) -> dict[str, Any]:
    hist = ["Base_A", "Base_B"]
    if bs:
        hist += ["BS_Core_Index_A", "BS_Core_Index_B"]
    return {
        "task": {"name": task_name},
        "dataset": {
            "path": "data/df.csv",
            "target_col": target,
            "dt_col": "dt",
            "hist_exog_cols": hist,
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "runtime": {"random_seed": 1},
        "training": dict(DEEP_DIVE.FIXED_TRAINING),
        "cv": {
            "horizon": 8,
            "step_size": 8,
            "n_windows": 12,
            "gap": 0,
            "max_train_size": None,
            "overlap_eval_policy": "by_cutoff_mean",
        },
        "scheduler": {"gpu_ids": [0, 1], "max_concurrent_jobs": 2, "worker_devices": 1},
        "residual": {"enabled": False},
        "jobs": [
            {"model": "TimeXer", "params": {"patch_len": 16, "hidden_size": 64, "n_heads": 4, "e_layers": 2, "dropout": 0.1}},
            {"model": "TSMixerx", "params": {"n_block": 2, "ff_dim": 64, "dropout": 0.1, "revin": True}},
            {"model": "Naive", "params": {}},
            {"model": "iTransformer", "params": {"hidden_size": 64, "n_heads": 4, "e_layers": 2, "d_ff": 256, "dropout": 0.1}},
            {"model": "LSTM", "params": {"encoder_hidden_size": 64, "decoder_hidden_size": 64, "encoder_n_layers": 2, "context_size": 10}},
        ],
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_leaderboard(run_root: Path, rows: list[dict[str, Any]]) -> None:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_dir / "leaderboard.csv", index=False)


def _write_resolved_config(run_root: Path, payload: dict[str, Any], *, valid_batch_size: int = 64, jobs: list[str] | None = None) -> None:
    config_dir = run_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    selected_models = jobs or [job["model"] for job in payload["jobs"]]
    selected_jobs = [
        {"model": job["model"], "params": job.get("params", {})}
        for job in payload["jobs"]
        if job["model"] in selected_models
    ]
    resolved = {
        "task": payload["task"],
        "dataset": payload["dataset"],
        "training": {**DEEP_DIVE.FIXED_TRAINING, "valid_batch_size": valid_batch_size},
        "jobs": selected_jobs,
        "cv": payload["cv"],
    }
    (config_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")


def _write_manifest(
    run_root: Path,
    config_source_path: Path,
    jobs: list[str],
    *,
    include_hash: bool = True,
) -> None:
    manifest_dir = run_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_source_path": str(config_source_path.resolve()),
        "jobs": [{"model": name} for name in jobs],
    }
    if include_hash:
        payload["config_input_sha256"] = DEEP_DIVE._sha256_text(
            config_source_path.read_text(encoding="utf-8")
        )
    (manifest_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture()
def temp_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "df.csv").write_text("dt,target\n2020-01-01,1\n", encoding="utf-8")
    _write_yaml(root / "search_space.yaml", _search_space_payload())
    for stem, target in (("brentoil-case1", "Com_BrentCrudeOil"), ("wti-case1", "Com_CrudeOil")):
        baseline_payload = _base_payload(stem.replace("-", "_"), target, bs=False)
        incumbent_payload = _base_payload(stem.replace("-", "_") + "_bs", target, bs=True)
        _write_yaml(root / "yaml" / "feature_set" / f"{stem}.yaml", baseline_payload)
        _write_yaml(root / "yaml" / "feature_set_bs" / f"{stem}.yaml", incumbent_payload)
        baseline_root = root / "runs" / "feature_set" / f"feature_set_{baseline_payload['task']['name']}"
        incumbent_root = root / "runs" / "feature_set_bs" / f"feature_set_bs_{incumbent_payload['task']['name']}"
        for run_root, payload, cfg_path in (
            (baseline_root, baseline_payload, root / "yaml" / "feature_set" / f"{stem}.yaml"),
            (incumbent_root, incumbent_payload, root / "yaml" / "feature_set_bs" / f"{stem}.yaml"),
        ):
            _write_manifest(run_root, cfg_path, [job["model"] for job in payload["jobs"]])
            _write_resolved_config(run_root, payload, valid_batch_size=32)
            rows = []
            for idx, job in enumerate(payload["jobs"], start=1):
                rows.append({"rank": idx, "model": job["model"], "mean_fold_mape": 0.05 + idx * 0.01})
            _write_leaderboard(run_root, rows)
    return root


def test_fixed_training_temp_yaml_generation(temp_repo: Path) -> None:
    cases = DEEP_DIVE.discover_cases(temp_repo)
    payload = DEEP_DIVE.build_payload(cases[0].incumbent_payload, task_name="tmp", jobs=[{"model": "TimeXer", "params": {"patch_len": 8}}])
    assert payload["training"] == DEEP_DIVE.FIXED_TRAINING
    assert payload["jobs"] == [{"model": "TimeXer", "params": {"patch_len": 8}}]


def test_rejects_stale_reference_runs(temp_repo: Path) -> None:
    cases = DEEP_DIVE.discover_cases(temp_repo)
    decision = DEEP_DIVE.evaluate_historical_provenance(
        DEEP_DIVE.historical_run_root_for_config(temp_repo, cases[0].baseline_config_path, cases[0].baseline_payload),
        expected_source_path=cases[0].baseline_config_path,
        expected_payload=cases[0].baseline_payload,
        role="baseline",
    )
    assert not decision.comparable
    assert any("training.valid_batch_size" in reason for reason in decision.reasons)


def test_rejects_missing_manifest_hash_and_param_mismatch(temp_repo: Path) -> None:
    cases = DEEP_DIVE.discover_cases(temp_repo)
    case = cases[0]
    run_root = temp_repo / "runs" / "feature_set" / "feature_set_param_mismatch"
    _write_manifest(
        run_root,
        case.baseline_config_path,
        [job["model"] for job in case.baseline_payload["jobs"]],
        include_hash=False,
    )
    payload = json.loads(json.dumps(case.baseline_payload))
    payload["task"]["name"] = "param_mismatch"
    payload["jobs"][0]["params"]["patch_len"] = 999
    _write_resolved_config(run_root, payload)

    decision = DEEP_DIVE.evaluate_historical_provenance(
        run_root,
        expected_source_path=case.baseline_config_path,
        expected_payload=case.baseline_payload,
        role="baseline",
    )
    assert not decision.comparable
    assert "missing manifest config_input_sha256" in decision.reasons
    assert any("job.params mismatch for TimeXer" == reason for reason in decision.reasons)


def test_stage1_uses_jobs_flag_and_output_root(monkeypatch: pytest.MonkeyPatch, temp_repo: Path, tmp_path: Path) -> None:
    cases = DEEP_DIVE.discover_cases(temp_repo)
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        output_root = Path(cmd[cmd.index("--output-root") + 1])
        config_path = Path(cmd[cmd.index("--config") + 1])
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = cmd[cmd.index("--jobs") + 1]
        _write_resolved_config(output_root, payload, jobs=[model])
        _write_manifest(output_root, config_path, [model])
        _write_leaderboard(output_root, [{"rank": 1, "model": model, "mean_fold_mape": 0.02}])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(DEEP_DIVE.subprocess, "run", fake_run)
    baseline_root = tmp_path / "baseline"
    incumbent_root = tmp_path / "incumbent"
    payload = cases[0].incumbent_payload
    _write_resolved_config(baseline_root, payload, jobs=["TimeXer"])
    _write_resolved_config(incumbent_root, payload, jobs=["TimeXer"])
    _write_leaderboard(baseline_root, [{"rank": 1, "model": "TimeXer", "mean_fold_mape": 0.05}])
    _write_leaderboard(incumbent_root, [{"rank": 1, "model": "TimeXer", "mean_fold_mape": 0.04}])

    outcome = DEEP_DIVE.execute_stage1_candidate(
        temp_repo,
        case_spec=cases[0],
        model="TimeXer",
        params={"patch_len": 8, "hidden_size": 256, "n_heads": 16, "e_layers": 4, "dropout": 0.1},
        candidate_id="trial-001",
        source="sampler",
        baseline_run_root=baseline_root,
        incumbent_run_root=incumbent_root,
        work_root=tmp_path / "work",
        resume=False,
    )
    assert "--jobs" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--jobs") + 1] == "TimeXer"
    assert "--output-root" in captured["cmd"]
    assert outcome.delta_vs_baseline_pp == pytest.approx(-3.0)


def test_candidate_generation_uses_canonical_search_space_contract(temp_repo: Path) -> None:
    contract = DEEP_DIVE.load_search_space_contract(temp_repo)
    specs = contract.payload["models"]["TimeXer"]
    assert set(specs) == {"patch_len", "hidden_size", "n_heads", "e_layers", "dropout"}


def test_search_models_exclude_tsmixerx() -> None:
    assert "TSMixerx" not in DEEP_DIVE.SEARCH_MODELS
    assert set(DEEP_DIVE.EXPECTED_MODELS) == {"TimeXer", "iTransformer", "LSTM", "Naive"}
    assert DEEP_DIVE.SUCCESS_TARGET_CASES == 4


def test_stage1_seed_params_injects_off_grid_anchor(temp_repo: Path) -> None:
    cases = DEEP_DIVE.discover_cases(temp_repo)
    contract = DEEP_DIVE.load_search_space_contract(temp_repo)
    seeds = DEEP_DIVE.stage1_seed_params(
        cases[0],
        "TimeXer",
        model_specs=contract.payload["models"]["TimeXer"],
    )
    sources = [source for source, _ in seeds]
    assert sources[0] == "incumbent_yaml_seed"
    assert any(source.startswith("off_grid_anchor_") for source in sources[1:])
    incumbent_params = seeds[0][1]
    assert any(params != incumbent_params for _, params in seeds[1:])


def test_bundle_scoring_and_cross_case_selection(tmp_path: Path) -> None:
    bundle_a = DEEP_DIVE.BundleOutcome(
        bundle_id="a",
        case="brentoil-case1",
        target="Com_BrentCrudeOil",
        selected_model_candidate_ids={m: "x" for m in DEEP_DIVE.SEARCH_MODELS},
        bundle_run_root=tmp_path / "a",
        baseline_run_root=tmp_path / "base",
        incumbent_run_root=tmp_path / "inc",
        bundle_mean_mape=0.05,
        baseline_bundle_mean_mape=0.07,
        incumbent_bundle_mean_mape=0.06,
        delta_case_mean_mape_pp=-2.0,
        delta_case_mean_mape_vs_incumbent_pp=-1.0,
        delta_case_mean_mape_learned_pp=-2.5,
        in_target_band=True,
    )
    bundle_b = DEEP_DIVE.BundleOutcome(
        bundle_id="b",
        case="brentoil-case1",
        target="Com_BrentCrudeOil",
        selected_model_candidate_ids={m: "y" for m in DEEP_DIVE.SEARCH_MODELS},
        bundle_run_root=tmp_path / "b",
        baseline_run_root=tmp_path / "base",
        incumbent_run_root=tmp_path / "inc",
        bundle_mean_mape=0.06,
        baseline_bundle_mean_mape=0.07,
        incumbent_bundle_mean_mape=0.065,
        delta_case_mean_mape_pp=-1.0,
        delta_case_mean_mape_vs_incumbent_pp=-0.5,
        delta_case_mean_mape_learned_pp=-1.5,
        in_target_band=True,
    )
    bundle_c = DEEP_DIVE.BundleOutcome(
        bundle_id="c",
        case="wti-case1",
        target="Com_CrudeOil",
        selected_model_candidate_ids={m: "z" for m in DEEP_DIVE.SEARCH_MODELS},
        bundle_run_root=tmp_path / "c",
        baseline_run_root=tmp_path / "base2",
        incumbent_run_root=tmp_path / "inc2",
        bundle_mean_mape=0.09,
        baseline_bundle_mean_mape=0.10,
        incumbent_bundle_mean_mape=0.095,
        delta_case_mean_mape_pp=-1.0,
        delta_case_mean_mape_vs_incumbent_pp=-0.5,
        delta_case_mean_mape_learned_pp=-1.1,
        in_target_band=True,
    )
    rows = DEEP_DIVE.cross_case_summary_rows([bundle_b, bundle_a, bundle_c])
    assert rows[0]["bundle_id"] == "a"
    assert rows[0]["success_case_count"] == 2
    assert rows[0]["delta_case_mean_mape_vs_incumbent_pp"] == pytest.approx(-1.0)
    assert rows[1]["global_goal_met"] is False


def test_rebuilds_noncomparable_reference(monkeypatch: pytest.MonkeyPatch, temp_repo: Path, tmp_path: Path) -> None:
    cases = DEEP_DIVE.discover_cases(temp_repo)
    called: dict[str, Any] = {"count": 0}

    def fake_run_main(repo_root: Path, *, config_path: Path, output_root: Path, jobs: list[str] | None = None, expected_jobs: list[str] | None = None) -> Path:
        called["count"] += 1
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        _write_resolved_config(output_root, payload, jobs=[job["model"] for job in payload["jobs"]])
        _write_manifest(output_root, config_path, [job["model"] for job in payload["jobs"]])
        rows = []
        for idx, job in enumerate(payload["jobs"], start=1):
            rows.append({"rank": idx, "model": job["model"], "mean_fold_mape": 0.04 + idx * 0.01})
        _write_leaderboard(output_root, rows)
        return output_root

    monkeypatch.setattr(DEEP_DIVE, "run_main", fake_run_main)
    run_root, decision = DEEP_DIVE.ensure_reference_run(
        temp_repo,
        case_spec=cases[0],
        role="baseline",
        work_root=tmp_path / "work",
        resume=False,
    )
    assert called["count"] == 1
    assert run_root.exists()
    assert decision.comparable
    assert not decision.reused
    resolved = json.loads((run_root / "config" / "config.resolved.json").read_text(encoding="utf-8"))
    assert {job["model"] for job in resolved["jobs"]} == set(DEEP_DIVE.EXPECTED_MODELS)
    assert "TSMixerx" not in {job["model"] for job in resolved["jobs"]}


def test_resolved_config_training_gate(temp_repo: Path, tmp_path: Path) -> None:
    payload = _base_payload("case_x", "Com_CrudeOil", bs=True)
    run_root = tmp_path / "run"
    _write_resolved_config(run_root, payload)
    ok, reasons = DEEP_DIVE.resolved_config_training_gate(run_root)
    assert ok, reasons
    resolved = json.loads((run_root / "config" / "config.resolved.json").read_text(encoding="utf-8"))
    assert resolved["training"]["valid_batch_size"] == 64


def test_resume_reuse_requires_matching_generated_run_guard(temp_repo: Path, tmp_path: Path) -> None:
    cases = DEEP_DIVE.discover_cases(temp_repo)
    run_root = tmp_path / "stage1" / "case" / "TimeXer" / "trial-001"
    config_path = tmp_path / "temp_configs" / "model_trials" / "case" / "TimeXer" / "trial-001.yaml"
    payload = DEEP_DIVE.build_payload(
        cases[0].incumbent_payload,
        task_name="deep_dive_case_TimeXer_trial-001",
        jobs=[{"model": "TimeXer", "params": {"patch_len": 8}}],
    )
    DEEP_DIVE.dump_yaml(payload, config_path)
    _write_resolved_config(run_root, payload, jobs=["TimeXer"])
    _write_leaderboard(run_root, [{"rank": 1, "model": "TimeXer", "mean_fold_mape": 0.02}])
    DEEP_DIVE.write_generated_run_guard(run_root, config_path=config_path, expected_jobs=["TimeXer"])

    assert DEEP_DIVE.maybe_reuse_generated_run(
        run_root,
        config_path=config_path,
        resume=True,
        expected_jobs=["TimeXer"],
    )

    mutated_payload = DEEP_DIVE.build_payload(
        cases[0].incumbent_payload,
        task_name="deep_dive_case_TimeXer_trial-001",
        jobs=[{"model": "TimeXer", "params": {"patch_len": 16}}],
    )
    DEEP_DIVE.dump_yaml(mutated_payload, config_path)

    assert not DEEP_DIVE.maybe_reuse_generated_run(
        run_root,
        config_path=config_path,
        resume=True,
        expected_jobs=["TimeXer"],
    )


def test_continue_picks_latest_run_root(monkeypatch: pytest.MonkeyPatch, temp_repo: Path) -> None:
    older = temp_repo / "runs" / "deep_dive_20260325T010000Z"
    newer = temp_repo / "runs" / "deep_dive_20260325T020000Z"
    older.mkdir(parents=True, exist_ok=True)
    newer.mkdir(parents=True, exist_ok=True)
    older.touch()
    newer.touch()
    captured: dict[str, Any] = {}

    def fake_run_deep_dive(repo_root: Path, *, output_root: Path, controls: Any) -> dict[str, Any]:
        captured["repo_root"] = repo_root
        captured["output_root"] = output_root
        captured["controls"] = controls
        return {}

    monkeypatch.setattr(DEEP_DIVE, "run_deep_dive", fake_run_deep_dive)
    rc = DEEP_DIVE.main(["--repo-root", str(temp_repo), "--continue"])
    assert rc == 0
    assert captured["output_root"] == newer.resolve()
    assert captured["controls"].resume is True


def test_continue_without_existing_run_errors(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    with pytest.raises(SystemExit):
        DEEP_DIVE.main(["--repo-root", str(repo), "--continue"])


def test_run_deep_dive_emits_artifacts(monkeypatch: pytest.MonkeyPatch, temp_repo: Path, tmp_path: Path) -> None:
    def fake_run(repo_root: Path, *, config_path: Path, output_root: Path, jobs: list[str] | None = None, expected_jobs: list[str] | None = None) -> Path:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        selected_jobs = jobs or [job["model"] for job in payload["jobs"]]
        _write_resolved_config(output_root, payload, jobs=selected_jobs)
        _write_manifest(output_root, config_path, selected_jobs)
        rows = []
        for idx, model in enumerate(selected_jobs, start=1):
            metric = 0.03 + idx * 0.005
            if model == "Naive":
                metric = 0.06
            rows.append({"rank": idx, "model": model, "mean_fold_mape": metric})
        _write_leaderboard(output_root, rows)
        return output_root

    monkeypatch.setattr(DEEP_DIVE, "run_main", fake_run)
    summary = DEEP_DIVE.run_deep_dive(
        temp_repo,
        output_root=tmp_path / "deep_dive_run",
        controls=DEEP_DIVE.Controls(stage1_trials_per_model=1, stage2_top_k=1, seed=1, resume=False),
    )
    results_dir = tmp_path / "deep_dive_run" / "results"
    assert (results_dir / "provenance.csv").exists()
    assert (results_dir / "per_model_deltas.csv").exists()
    assert (results_dir / "per_case_bundle_deltas.csv").exists()
    assert (results_dir / "cross_case_summary.csv").exists()
    assert (results_dir / "summary.json").exists()
    bundle_frame = pd.read_csv(results_dir / "per_case_bundle_deltas.csv")
    cross_frame = pd.read_csv(results_dir / "cross_case_summary.csv")
    assert "delta_case_mean_mape_vs_incumbent_pp" in bundle_frame.columns
    assert "delta_case_mean_mape_vs_incumbent_pp" in cross_frame.columns
    assert "success_case_count" in summary
