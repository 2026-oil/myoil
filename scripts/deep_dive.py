from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import optuna
import pandas as pd
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


SEARCH_MODELS = ("TSMixerx", "iTransformer", "LSTM")
CONTROL_MODEL = "Naive"
EXPECTED_MODELS = SEARCH_MODELS + (CONTROL_MODEL,)
FIXED_TRAINING: dict[str, Any] = {
    "input_size": 64,
    "batch_size": 32,
    "valid_batch_size": 64,
    "windows_batch_size": 1024,
    "inference_windows_batch_size": 1024,
    "learning_rate": 0.001,
    "model_step_size": 8,
    "max_steps": 1000,
    "val_size": 8,
    "val_check_steps": 50,
    "early_stop_patience_steps": 5,
    "loss": "mse",
}
TRAINING_GATE_KEYS = tuple(FIXED_TRAINING)
SUCCESS_TARGET_CASES = 4
ALL_SEARCH_MODELS = ("TimeXer", "TSMixerx", "iTransformer", "LSTM")
GENERATED_RUN_GUARD_NAME = "deep_dive_resume_guard.json"


@dataclass(frozen=True)
class SearchSpaceContract:
    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class Controls:
    stage1_trials_per_model: int = 8
    stage2_top_k: int = 2
    seed: int = 1
    resume: bool = False


@dataclass(frozen=True)
class CaseSpec:
    key: str
    case_name: str
    target_col: str
    baseline_config_path: Path
    incumbent_config_path: Path
    baseline_payload: dict[str, Any]
    incumbent_payload: dict[str, Any]


@dataclass(frozen=True)
class ProvenanceDecision:
    role: str
    run_root: Path
    comparable: bool
    reasons: tuple[str, ...]
    reused: bool = False

    def to_row(self, case_key: str, target_col: str) -> dict[str, Any]:
        return {
            "case": case_key,
            "target": target_col,
            "role": self.role,
            "run_root": str(self.run_root),
            "comparable": self.comparable,
            "reused": self.reused,
            "reasons": " | ".join(self.reasons),
        }


@dataclass(frozen=True)
class CandidateOutcome:
    candidate_id: str
    case: str
    target: str
    model: str
    model_params: dict[str, Any]
    run_root: Path
    baseline_run_root: Path
    incumbent_run_root: Path
    candidate_mean_mape: float
    baseline_mean_mape: float
    incumbent_mean_mape: float
    delta_vs_baseline_pp: float
    delta_vs_incumbent_pp: float
    source: str

    def sort_key(self) -> tuple[float, float, float, str]:
        return (
            self.delta_vs_baseline_pp,
            self.delta_vs_incumbent_pp,
            self.candidate_mean_mape,
            self.candidate_id,
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "case": self.case,
            "target": self.target,
            "model": self.model,
            "model_params": json.dumps(self.model_params, sort_keys=True),
            "run_root": str(self.run_root),
            "baseline_run_root": str(self.baseline_run_root),
            "incumbent_run_root": str(self.incumbent_run_root),
            "candidate_mean_mape": self.candidate_mean_mape,
            "baseline_mean_mape": self.baseline_mean_mape,
            "incumbent_mean_mape": self.incumbent_mean_mape,
            "delta_vs_baseline_pp": self.delta_vs_baseline_pp,
            "delta_vs_incumbent_pp": self.delta_vs_incumbent_pp,
            "source": self.source,
        }


@dataclass(frozen=True)
class BundleOutcome:
    bundle_id: str
    case: str
    target: str
    selected_model_candidate_ids: dict[str, str]
    bundle_run_root: Path
    baseline_run_root: Path
    incumbent_run_root: Path
    bundle_mean_mape: float
    baseline_bundle_mean_mape: float
    incumbent_bundle_mean_mape: float
    delta_case_mean_mape_pp: float
    delta_case_mean_mape_vs_incumbent_pp: float
    delta_case_mean_mape_learned_pp: float
    in_target_band: bool

    def winner_sort_key(self) -> tuple[int, float, float, str]:
        band_center_distance = abs(self.delta_case_mean_mape_pp - (-2.0))
        return (
            0 if self.in_target_band else 1,
            band_center_distance,
            self.delta_case_mean_mape_learned_pp,
            self.bundle_id,
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "case": self.case,
            "target": self.target,
            "selected_model_candidate_ids": json.dumps(
                self.selected_model_candidate_ids, sort_keys=True
            ),
            "bundle_run_root": str(self.bundle_run_root),
            "baseline_run_root": str(self.baseline_run_root),
            "incumbent_run_root": str(self.incumbent_run_root),
            "bundle_mean_mape": self.bundle_mean_mape,
            "baseline_bundle_mean_mape": self.baseline_bundle_mean_mape,
            "incumbent_bundle_mean_mape": self.incumbent_bundle_mean_mape,
            "delta_case_mean_mape_pp": self.delta_case_mean_mape_pp,
            "delta_case_mean_mape_vs_incumbent_pp": self.delta_case_mean_mape_vs_incumbent_pp,
            "delta_case_mean_mape_learned_pp": self.delta_case_mean_mape_learned_pp,
            "in_target_band": self.in_target_band,
        }




def _normalize_model_search_specs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("search_space.yaml must contain a models mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for model_name, specs in models.items():
        if not isinstance(specs, dict):
            raise ValueError(f"search_space.models.{model_name} must be a mapping")
        normalized[str(model_name)] = {str(name): dict(spec) for name, spec in specs.items()}
    return normalized


def load_search_space_contract(repo_root: Path) -> SearchSpaceContract:
    path = (repo_root / "search_space.yaml").resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SearchSpaceContract(path=path, payload={"models": _normalize_model_search_specs(payload)})


def _suggest_from_spec(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    spec_type = str(spec.get("type", "")).strip().lower()
    if spec_type == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if spec_type == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if spec_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    raise ValueError(f"Unsupported search-space spec type: {spec_type}")


def suggest_model_params(
    model_name: str,
    selected_names: tuple[str, ...],
    trial: optuna.Trial,
    *,
    param_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        name: _suggest_from_spec(trial, name, param_specs[name])
        for name in selected_names
    }


def repo_root_from_script(script_path: str | Path = __file__) -> Path:
    return Path(script_path).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if payload is None else payload


def dump_yaml(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def discover_cases(repo_root: Path) -> list[CaseSpec]:
    out: list[CaseSpec] = []
    baseline_dir = repo_root / "yaml" / "feature_set"
    incumbent_dir = repo_root / "yaml" / "feature_set_bs"
    for incumbent_path in sorted(incumbent_dir.glob("*.yaml")):
        baseline_path = baseline_dir / incumbent_path.name
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline config for {incumbent_path.name}")
        incumbent_payload = load_yaml(incumbent_path)
        baseline_payload = load_yaml(baseline_path)
        case_name = incumbent_path.stem
        out.append(
            CaseSpec(
                key=case_name,
                case_name=case_name,
                target_col=str(incumbent_payload["dataset"]["target_col"]),
                baseline_config_path=baseline_path,
                incumbent_config_path=incumbent_path,
                baseline_payload=baseline_payload,
                incumbent_payload=incumbent_payload,
            )
        )
    return out


def fixed_training_matches(training_payload: dict[str, Any] | None) -> tuple[bool, list[str]]:
    training_payload = training_payload or {}
    reasons: list[str] = []
    for key in TRAINING_GATE_KEYS:
        if training_payload.get(key) != FIXED_TRAINING[key]:
            reasons.append(
                f"training.{key}={training_payload.get(key)!r} != {FIXED_TRAINING[key]!r}"
            )
    return (not reasons, reasons)


def resolved_config_training_gate(
    run_root: Path,
    *,
    expected_jobs: Iterable[str] | None = None,
) -> tuple[bool, list[str]]:
    resolved_path = run_root / "config" / "config.resolved.json"
    if not resolved_path.exists():
        return False, [f"missing {resolved_path}"]
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    ok, reasons = fixed_training_matches(resolved.get("training"))
    if expected_jobs is not None:
        expected_job_set = set(expected_jobs)
        resolved_job_set = {job.get("model") for job in resolved.get("jobs", [])}
        if resolved_job_set != expected_job_set:
            reasons.append(
                f"resolved jobs={sorted(resolved_job_set)} != expected jobs={sorted(expected_job_set)}"
            )
    return (not reasons, reasons)


def historical_run_root_for_config(repo_root: Path, config_path: Path, payload: dict[str, Any]) -> Path:
    task_name = str(payload["task"]["name"])
    parent = config_path.parent.name
    if parent == "feature_set":
        return repo_root / "runs" / "feature_set" / f"feature_set_{task_name}"
    if parent == "feature_set_bs":
        return repo_root / "runs" / "feature_set_bs" / f"feature_set_bs_{task_name}"
    raise ValueError(f"Unsupported config parent: {parent}")


def load_manifest(run_root: Path) -> dict[str, Any] | None:
    manifest_path = run_root / "manifest" / "run_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_text(path.read_text(encoding="utf-8"))


def generated_run_guard_path(run_root: Path) -> Path:
    return run_root / "manifest" / GENERATED_RUN_GUARD_NAME


def write_generated_run_guard(
    run_root: Path,
    *,
    config_path: Path,
    expected_jobs: Iterable[str] | None,
) -> Path:
    guard_path = generated_run_guard_path(run_root)
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_path": str(config_path.resolve(strict=False)),
        "config_sha256": _sha256_file(config_path),
        "expected_jobs": list(expected_jobs) if expected_jobs is not None else None,
    }
    guard_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return guard_path


def generated_run_guard_matches(
    run_root: Path,
    *,
    config_path: Path,
    expected_jobs: Iterable[str] | None,
) -> bool:
    guard_path = generated_run_guard_path(run_root)
    if not guard_path.exists() or not config_path.exists():
        return False
    payload = json.loads(guard_path.read_text(encoding="utf-8"))
    if payload.get("config_path") != str(config_path.resolve(strict=False)):
        return False
    if payload.get("config_sha256") != _sha256_file(config_path):
        return False
    if expected_jobs is not None and payload.get("expected_jobs") != list(expected_jobs):
        return False
    return True


def evaluate_historical_provenance(
    run_root: Path,
    *,
    expected_source_path: Path,
    expected_payload: dict[str, Any],
    role: str,
) -> ProvenanceDecision:
    reasons: list[str] = []
    manifest = load_manifest(run_root)
    if manifest is None:
        reasons.append("missing manifest/run_manifest.json")
        return ProvenanceDecision(role=role, run_root=run_root, comparable=False, reasons=tuple(reasons))
    resolved_path = run_root / "config" / "config.resolved.json"
    if not resolved_path.exists():
        reasons.append("missing config/config.resolved.json")
        return ProvenanceDecision(role=role, run_root=run_root, comparable=False, reasons=tuple(reasons))
    source_path = Path(str(manifest.get("config_source_path", ""))).resolve(strict=False)
    if source_path != expected_source_path.resolve(strict=False):
        reasons.append(f"config_source_path={source_path} != {expected_source_path.resolve(strict=False)}")
    expected_input_hash = _sha256_text(expected_source_path.read_text(encoding="utf-8"))
    manifest_input_hash = manifest.get("config_input_sha256")
    if manifest_input_hash is None:
        reasons.append("missing manifest config_input_sha256")
    elif manifest_input_hash != expected_input_hash:
        reasons.append(
            f"config_input_sha256={manifest_input_hash} != current source hash={expected_input_hash}"
        )
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    dataset = resolved.get("dataset", {})
    expected_dataset = expected_payload["dataset"]
    if dataset.get("target_col") != expected_dataset["target_col"]:
        reasons.append("dataset.target_col mismatch")
    if tuple(dataset.get("hist_exog_cols", [])) != tuple(expected_dataset.get("hist_exog_cols", [])):
        reasons.append("dataset.hist_exog_cols mismatch")
    resolved_cv = resolved.get("cv", {})
    expected_cv = expected_payload.get("cv", {})
    for key in ("horizon", "step_size", "n_windows", "gap", "overlap_eval_policy"):
        if resolved_cv.get(key) != expected_cv.get(key):
            reasons.append(f"cv.{key} mismatch")
    resolved_jobs = resolved.get("jobs", [])
    resolved_job_map = {job.get("model"): job.get("params", {}) for job in resolved_jobs}
    expected_job_map = {
        job.get("model"): job.get("params", {})
        for job in expected_payload.get("jobs", [])
    }
    if set(resolved_job_map) != set(EXPECTED_MODELS):
        reasons.append("job roster mismatch")
    for model, expected_params in expected_job_map.items():
        if resolved_job_map.get(model, {}) != expected_params:
            reasons.append(f"job.params mismatch for {model}")
    ok_training, training_reasons = fixed_training_matches(resolved.get("training"))
    if not ok_training:
        reasons.extend(training_reasons)
    return ProvenanceDecision(
        role=role,
        run_root=run_root,
        comparable=not reasons,
        reasons=tuple(reasons) or ("historical run is comparable",),
        reused=not reasons,
    )


def leaderboard_frame(run_root: Path) -> pd.DataFrame:
    path = run_root / "summary" / "leaderboard.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing leaderboard at {path}")
    return pd.read_csv(path)


def mean_fold_mape_for_models(run_root: Path, models: Iterable[str]) -> float:
    frame = leaderboard_frame(run_root)
    wanted = list(models)
    selected = frame[frame["model"].isin(wanted)]
    if len(selected) != len(wanted):
        missing = sorted(set(wanted).difference(selected["model"].tolist()))
        raise RuntimeError(f"Missing leaderboard row(s) for: {', '.join(missing)}")
    return float(selected["mean_fold_mape"].mean())


def model_mean_mape(run_root: Path, model: str) -> float:
    frame = leaderboard_frame(run_root)
    selected = frame[frame["model"] == model]
    if selected.empty:
        raise RuntimeError(f"Missing leaderboard row for {model}")
    return float(selected.iloc[0]["mean_fold_mape"])


def build_payload(source_payload: dict[str, Any], *, task_name: str, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.loads(json.dumps(source_payload))
    payload.setdefault("task", {})["name"] = task_name
    payload["training"] = dict(FIXED_TRAINING)
    payload["jobs"] = jobs
    return payload


def job_from_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
    for job in payload.get("jobs", []):
        if job.get("model") == model:
            return json.loads(json.dumps(job))
    raise KeyError(f"Missing job {model}")


def run_main(
    repo_root: Path,
    *,
    config_path: Path,
    output_root: Path,
    jobs: list[str] | None = None,
) -> Path:
    cmd = [sys.executable, str(repo_root / "main.py"), "--config", str(config_path), "--output-root", str(output_root)]
    if jobs:
        cmd.extend(["--jobs", *jobs])
    prior = os.environ.get("NEURALFORECAST_ALLOW_INTERNAL_OUTPUT_ROOT")
    os.environ["NEURALFORECAST_ALLOW_INTERNAL_OUTPUT_ROOT"] = "1"
    try:
        subprocess.run(cmd, cwd=repo_root, check=True)
    finally:
        if prior is None:
            os.environ.pop("NEURALFORECAST_ALLOW_INTERNAL_OUTPUT_ROOT", None)
        else:
            os.environ["NEURALFORECAST_ALLOW_INTERNAL_OUTPUT_ROOT"] = prior
    return output_root


def run_main_with_gate(
    repo_root: Path,
    *,
    config_path: Path,
    output_root: Path,
    jobs: list[str] | None = None,
    expected_jobs: Iterable[str] | None = None,
) -> Path:
    run_main(
        repo_root,
        config_path=config_path,
        output_root=output_root,
        jobs=jobs,
    )
    gate_ok, gate_reasons = resolved_config_training_gate(
        output_root,
        expected_jobs=expected_jobs,
    )
    if not gate_ok:
        raise RuntimeError(
            f"resolved-config training gate failed for {output_root}: {'; '.join(gate_reasons)}"
        )
    return output_root


def maybe_reuse_generated_run(
    run_root: Path,
    *,
    config_path: Path,
    resume: bool,
    expected_jobs: Iterable[str] | None = None,
) -> bool:
    if not resume:
        return False
    leaderboard_path = run_root / "summary" / "leaderboard.csv"
    if not leaderboard_path.exists():
        return False
    if not generated_run_guard_matches(
        run_root,
        config_path=config_path,
        expected_jobs=expected_jobs,
    ):
        return False
    ok, _ = resolved_config_training_gate(run_root, expected_jobs=expected_jobs)
    return ok


def ensure_reference_run(
    repo_root: Path,
    *,
    case_spec: CaseSpec,
    role: str,
    work_root: Path,
    resume: bool,
) -> tuple[Path, ProvenanceDecision]:
    if role == "baseline":
        payload = case_spec.baseline_payload
        source_path = case_spec.baseline_config_path
    elif role == "incumbent":
        payload = case_spec.incumbent_payload
        source_path = case_spec.incumbent_config_path
    else:
        raise ValueError(f"Unsupported role: {role}")

    historical_root = historical_run_root_for_config(repo_root, source_path, payload)
    historical_decision = evaluate_historical_provenance(
        historical_root,
        expected_source_path=source_path,
        expected_payload=payload,
        role=role,
    )
    if historical_decision.comparable:
        return historical_root, historical_decision

    task_name = f"deep_dive_{case_spec.key}_{role}"
    payload_out = build_payload(
        payload,
        task_name=task_name,
        jobs=[json.loads(json.dumps(job)) for job in payload["jobs"] if job.get("model") in EXPECTED_MODELS],
    )
    config_path = dump_yaml(
        payload_out,
        work_root / "temp_configs" / "reference" / case_spec.key / f"{role}.yaml",
    )
    generated_root = work_root / "reference" / case_spec.key / role
    if maybe_reuse_generated_run(
        generated_root,
        config_path=config_path,
        resume=resume,
        expected_jobs=EXPECTED_MODELS,
    ):
        return generated_root, ProvenanceDecision(
            role=role,
            run_root=generated_root,
            comparable=True,
            reasons=("reused generated reference",),
            reused=True,
        )
    run_main_with_gate(
        repo_root,
        config_path=config_path,
        output_root=generated_root,
        expected_jobs=EXPECTED_MODELS,
    )
    write_generated_run_guard(
        generated_root,
        config_path=config_path,
        expected_jobs=EXPECTED_MODELS,
    )
    return generated_root, ProvenanceDecision(
        role=role,
        run_root=generated_root,
        comparable=True,
        reasons=("rebuilt reference with fixed training",),
        reused=False,
    )


def _neighbor_choice(choices: list[Any], current: Any) -> Any | None:
    if not choices:
        return None
    if current not in choices:
        return choices[0]
    if len(choices) == 1:
        return None
    current_index = choices.index(current)
    if current_index + 1 < len(choices):
        return choices[current_index + 1]
    return choices[current_index - 1]


def stage1_seed_params(
    case_spec: CaseSpec,
    model: str,
    *,
    model_specs: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    seeds: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    incumbent_job = job_from_payload(case_spec.incumbent_payload, model)
    incumbent_params = dict(incumbent_job.get("params", {}))
    incumbent_key = json.dumps(incumbent_params, sort_keys=True)
    seeds.append(("incumbent_yaml_seed", incumbent_params))
    seen.add(incumbent_key)

    for param_name, param_spec in model_specs.items():
        choices = list(param_spec.get("choices", []))
        neighbor = _neighbor_choice(choices, incumbent_params.get(param_name))
        if neighbor is None or neighbor == incumbent_params.get(param_name):
            continue
        candidate_params = dict(incumbent_params)
        candidate_params[param_name] = neighbor
        candidate_key = json.dumps(candidate_params, sort_keys=True)
        if candidate_key in seen:
            continue
        seeds.append((f"off_grid_anchor_{param_name}", candidate_params))
        seen.add(candidate_key)
    return seeds


def execute_stage1_candidate(
    repo_root: Path,
    *,
    case_spec: CaseSpec,
    model: str,
    params: dict[str, Any],
    candidate_id: str,
    source: str,
    baseline_run_root: Path,
    incumbent_run_root: Path,
    work_root: Path,
    resume: bool,
) -> CandidateOutcome:
    candidate_root = work_root / "stage1" / case_spec.key / model / candidate_id
    task_name = f"deep_dive_{case_spec.key}_{model}_{candidate_id}"
    payload = build_payload(
        case_spec.incumbent_payload,
        task_name=task_name,
        jobs=[{"model": model, "params": params}],
    )
    config_path = dump_yaml(
        payload,
        work_root / "temp_configs" / "model_trials" / case_spec.key / model / f"{candidate_id}.yaml",
    )
    if not maybe_reuse_generated_run(
        candidate_root,
        config_path=config_path,
        resume=resume,
        expected_jobs=[model],
    ):
        run_main_with_gate(
            repo_root,
            config_path=config_path,
            output_root=candidate_root,
            jobs=[model],
            expected_jobs=[model],
        )
        write_generated_run_guard(
            candidate_root,
            config_path=config_path,
            expected_jobs=[model],
        )
    candidate_mape = model_mean_mape(candidate_root, model)
    baseline_mape = model_mean_mape(baseline_run_root, model)
    incumbent_mape = model_mean_mape(incumbent_run_root, model)
    return CandidateOutcome(
        candidate_id=candidate_id,
        case=case_spec.key,
        target=case_spec.target_col,
        model=model,
        model_params=dict(params),
        run_root=candidate_root,
        baseline_run_root=baseline_run_root,
        incumbent_run_root=incumbent_run_root,
        candidate_mean_mape=candidate_mape,
        baseline_mean_mape=baseline_mape,
        incumbent_mean_mape=incumbent_mape,
        delta_vs_baseline_pp=100.0 * (candidate_mape - baseline_mape),
        delta_vs_incumbent_pp=100.0 * (candidate_mape - incumbent_mape),
        source=source,
    )


def generate_stage1_candidates(
    repo_root: Path,
    *,
    case_spec: CaseSpec,
    model: str,
    baseline_run_root: Path,
    incumbent_run_root: Path,
    work_root: Path,
    controls: Controls,
    model_specs: dict[str, Any],
) -> list[CandidateOutcome]:
    selected_names = tuple(model_specs)
    outcomes: list[CandidateOutcome] = []
    seen: set[str] = set()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=controls.seed),
    )

    for idx, (source, params) in enumerate(
        stage1_seed_params(case_spec, model, model_specs=model_specs),
        start=1,
    ):
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        outcomes.append(
            execute_stage1_candidate(
                repo_root,
                case_spec=case_spec,
                model=model,
                params=params,
                candidate_id=f"seed-{idx:03d}",
                source=source,
                baseline_run_root=baseline_run_root,
                incumbent_run_root=incumbent_run_root,
                work_root=work_root,
                resume=controls.resume,
            )
        )

    trial_index = 0
    max_attempts = max(controls.stage1_trials_per_model * 20, controls.stage1_trials_per_model + 5)
    attempts = 0
    while trial_index < controls.stage1_trials_per_model:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Unable to generate {controls.stage1_trials_per_model} unique candidates for {case_spec.key}/{model}; exhausted {max_attempts} attempts"
            )
        trial = study.ask()
        params = suggest_model_params(
            model,
            selected_names,
            trial,
            param_specs=model_specs,
        )
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            study.tell(trial, 0.0)
            continue
        seen.add(key)
        trial_index += 1
        outcome = execute_stage1_candidate(
            repo_root,
            case_spec=case_spec,
            model=model,
            params=params,
            candidate_id=f"trial-{trial_index:03d}",
            source="sampler",
            baseline_run_root=baseline_run_root,
            incumbent_run_root=incumbent_run_root,
            work_root=work_root,
            resume=controls.resume,
        )
        study.tell(trial, outcome.delta_vs_baseline_pp)
        outcomes.append(outcome)

    outcomes.sort(key=lambda item: item.sort_key())
    return outcomes


def choose_top_k(outcomes: list[CandidateOutcome], *, top_k: int) -> list[CandidateOutcome]:
    return list(sorted(outcomes, key=lambda item: item.sort_key())[:top_k])


def execute_bundle(
    repo_root: Path,
    *,
    case_spec: CaseSpec,
    bundle_id: str,
    chosen_candidates: dict[str, CandidateOutcome],
    baseline_run_root: Path,
    incumbent_run_root: Path,
    work_root: Path,
    resume: bool,
) -> BundleOutcome:
    bundle_root = work_root / "bundles" / case_spec.key / bundle_id
    jobs = []
    for model in SEARCH_MODELS:
        jobs.append({"model": model, "params": dict(chosen_candidates[model].model_params)})
    jobs.append(json.loads(json.dumps(job_from_payload(case_spec.incumbent_payload, CONTROL_MODEL))))
    payload = build_payload(
        case_spec.incumbent_payload,
        task_name=f"deep_dive_{case_spec.key}_{bundle_id}",
        jobs=jobs,
    )
    config_path = dump_yaml(
        payload,
        work_root / "temp_configs" / "bundles" / case_spec.key / f"{bundle_id}.yaml",
    )
    if not maybe_reuse_generated_run(
        bundle_root,
        config_path=config_path,
        resume=resume,
        expected_jobs=EXPECTED_MODELS,
    ):
        run_main_with_gate(
            repo_root,
            config_path=config_path,
            output_root=bundle_root,
            expected_jobs=EXPECTED_MODELS,
        )
        write_generated_run_guard(
            bundle_root,
            config_path=config_path,
            expected_jobs=EXPECTED_MODELS,
        )
    bundle_mean = mean_fold_mape_for_models(bundle_root, EXPECTED_MODELS)
    baseline_mean = mean_fold_mape_for_models(baseline_run_root, EXPECTED_MODELS)
    incumbent_mean = mean_fold_mape_for_models(incumbent_run_root, EXPECTED_MODELS)
    bundle_learned = mean_fold_mape_for_models(bundle_root, SEARCH_MODELS)
    baseline_learned = mean_fold_mape_for_models(baseline_run_root, SEARCH_MODELS)
    delta_pp = 100.0 * (bundle_mean - baseline_mean)
    delta_vs_incumbent_pp = 100.0 * (bundle_mean - incumbent_mean)
    learned_pp = 100.0 * (bundle_learned - baseline_learned)
    return BundleOutcome(
        bundle_id=bundle_id,
        case=case_spec.key,
        target=case_spec.target_col,
        selected_model_candidate_ids={model: chosen_candidates[model].candidate_id for model in SEARCH_MODELS},
        bundle_run_root=bundle_root,
        baseline_run_root=baseline_run_root,
        incumbent_run_root=incumbent_run_root,
        bundle_mean_mape=bundle_mean,
        baseline_bundle_mean_mape=baseline_mean,
        incumbent_bundle_mean_mape=incumbent_mean,
        delta_case_mean_mape_pp=delta_pp,
        delta_case_mean_mape_vs_incumbent_pp=delta_vs_incumbent_pp,
        delta_case_mean_mape_learned_pp=learned_pp,
        in_target_band=(-3.0 <= delta_pp <= -1.0),
    )


def cross_case_summary_rows(bundle_outcomes: list[BundleOutcome]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    grouped: dict[str, list[BundleOutcome]] = {}
    for item in bundle_outcomes:
        grouped.setdefault(item.case, []).append(item)
    for case_key, items in sorted(grouped.items()):
        winner = sorted(items, key=lambda item: item.winner_sort_key())[0]
        winners.append(
            {
                "case": case_key,
                "target": winner.target,
                "bundle_id": winner.bundle_id,
                "bundle_run_root": str(winner.bundle_run_root),
                "delta_case_mean_mape_pp": winner.delta_case_mean_mape_pp,
                "delta_case_mean_mape_vs_incumbent_pp": winner.delta_case_mean_mape_vs_incumbent_pp,
                "delta_case_mean_mape_learned_pp": winner.delta_case_mean_mape_learned_pp,
                "in_target_band": winner.in_target_band,
            }
        )
    success_count = sum(1 for row in winners if row["in_target_band"])
    for row in winners:
        row["success_case_count"] = success_count
        row["global_goal_met"] = success_count >= SUCCESS_TARGET_CASES
    return winners


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    ensure_dir(path.parent)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_report(
    *,
    work_root: Path,
    provenance_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    bundle_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
) -> Path:
    path = work_root / "results" / "report.md"
    lines = [
        "# Deep Dive Report",
        "",
        f"- provenance rows: {len(provenance_rows)}",
        f"- model trial rows: {len(model_rows)}",
        f"- bundle rows: {len(bundle_rows)}",
        f"- cross-case winners: {len(cross_rows)}",
    ]
    if cross_rows:
        success_count = int(cross_rows[0]["success_case_count"])
        lines.append(f"- success_case_count: {success_count}")
        lines.append(f"- global_goal_met: {bool(cross_rows[0]['global_goal_met'])}")
    lines.append("")
    lines.append("## Cross-case winners")
    lines.append("")
    for row in cross_rows:
        lines.append(
            f"- {row['case']}: bundle={row['bundle_id']} "
            f"delta_case_mean_mape_pp={row['delta_case_mean_mape_pp']:.4f} "
            f"delta_case_mean_mape_vs_incumbent_pp={row['delta_case_mean_mape_vs_incumbent_pp']:.4f} "
            f"delta_case_mean_mape_learned_pp={row['delta_case_mean_mape_learned_pp']:.4f} "
            f"in_target_band={row['in_target_band']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_deep_dive(repo_root: Path, *, output_root: Path, controls: Controls) -> dict[str, Any]:
    search_contract = load_search_space_contract(repo_root)
    cases = discover_cases(repo_root)
    ensure_dir(output_root)

    provenance_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    all_bundles: list[BundleOutcome] = []

    for case_spec in cases:
        baseline_run_root, baseline_decision = ensure_reference_run(
            repo_root,
            case_spec=case_spec,
            role="baseline",
            work_root=output_root,
            resume=controls.resume,
        )
        provenance_rows.append(baseline_decision.to_row(case_spec.key, case_spec.target_col))
        incumbent_run_root, incumbent_decision = ensure_reference_run(
            repo_root,
            case_spec=case_spec,
            role="incumbent",
            work_root=output_root,
            resume=controls.resume,
        )
        provenance_rows.append(incumbent_decision.to_row(case_spec.key, case_spec.target_col))

        stage1_top: dict[str, list[CandidateOutcome]] = {}
        for model in SEARCH_MODELS:
            model_specs = search_contract.payload["models"][model]
            outcomes = generate_stage1_candidates(
                repo_root,
                case_spec=case_spec,
                model=model,
                baseline_run_root=baseline_run_root,
                incumbent_run_root=incumbent_run_root,
                work_root=output_root,
                controls=controls,
                model_specs=model_specs,
            )
            candidate_rows.extend(item.to_row() for item in outcomes)
            stage1_top[model] = choose_top_k(outcomes, top_k=controls.stage2_top_k)

        bundle_counter = 0
        for combo in itertools.product(*(stage1_top[model] for model in SEARCH_MODELS)):
            bundle_counter += 1
            chosen = {candidate.model: candidate for candidate in combo}
            bundle = execute_bundle(
                repo_root,
                case_spec=case_spec,
                bundle_id=f"bundle-{bundle_counter:03d}",
                chosen_candidates=chosen,
                baseline_run_root=baseline_run_root,
                incumbent_run_root=incumbent_run_root,
                work_root=output_root,
                resume=controls.resume,
            )
            all_bundles.append(bundle)
            bundle_rows.append(bundle.to_row())

    cross_rows = cross_case_summary_rows(all_bundles)
    write_csv(provenance_rows, output_root / "results" / "provenance.csv")
    write_csv(candidate_rows, output_root / "results" / "per_model_deltas.csv")
    write_csv(bundle_rows, output_root / "results" / "per_case_bundle_deltas.csv")
    write_csv(cross_rows, output_root / "results" / "cross_case_summary.csv")
    summary = {
        "stage1_trials_per_model": controls.stage1_trials_per_model,
        "stage2_top_k": controls.stage2_top_k,
        "seed": controls.seed,
        "resume": controls.resume,
        "cases": [case.key for case in cases],
        "success_case_count": int(cross_rows[0]["success_case_count"]) if cross_rows else 0,
        "global_goal_met": bool(cross_rows[0]["global_goal_met"]) if cross_rows else False,
    }
    write_json(summary, output_root / "results" / "summary.json")
    write_report(
        work_root=output_root,
        provenance_rows=provenance_rows,
        model_rows=candidate_rows,
        bundle_rows=bundle_rows,
        cross_rows=cross_rows,
    )
    return summary





def configure_search_models(requested_models: Iterable[str] | None = None) -> tuple[str, ...]:
    global SEARCH_MODELS, EXPECTED_MODELS
    if requested_models is None:
        requested = SEARCH_MODELS
    else:
        requested = tuple(dict.fromkeys(str(item) for item in requested_models))
        invalid = sorted(set(requested).difference(ALL_SEARCH_MODELS))
        if invalid:
            raise ValueError(
                "Unsupported --models value(s): " + ", ".join(invalid)
            )
        if not requested:
            raise ValueError("--models must include at least one learned model")
    SEARCH_MODELS = tuple(requested)
    EXPECTED_MODELS = SEARCH_MODELS + (CONTROL_MODEL,)
    return SEARCH_MODELS

def latest_deep_dive_run(repo_root: Path) -> Path | None:
    runs_root = repo_root / "runs"
    candidates = sorted(
        runs_root.glob("deep_dive_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep-dive executor for feature_set vs feature_set_bs.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stage1-trials-per-model", type=int, default=8)
    parser.add_argument("--stage2-top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue", dest="continue_run", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    configure_search_models(args.models)
    continue_run = bool(getattr(args, "continue_run", False))
    if continue_run and args.output_root is None:
        latest_run = latest_deep_dive_run(repo_root)
        if latest_run is None:
            parser.error("--continue requires an existing runs/deep_dive_* directory or an explicit --output-root")
        output_root = latest_run
    else:
        output_root = args.output_root or (repo_root / "runs" / f"deep_dive_{ts}")
    controls = Controls(
        stage1_trials_per_model=args.stage1_trials_per_model,
        stage2_top_k=args.stage2_top_k,
        seed=args.seed,
        resume=(args.resume or continue_run),
    )
    run_deep_dive(repo_root, output_root=output_root.resolve(), controls=controls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
