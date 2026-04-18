from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_support.outcome_search import (
    AAFORECAST_MODEL_NAME,
    OUTCOME_OBJECTIVE_VERSION,
    OUTCOME_SEARCH_SCHEMA_VERSION,
    OutcomeEvaluation,
    audit_sort_key,
    compatibility_hash,
    evaluate_run_outcome,
    sha256_file,
    winner_sort_key,
)


DEFAULT_CONFIG = REPO_ROOT / "yaml/experiment/feature_set_aaforecast_wti/aaforecast-gru-ret.yaml"
DEFAULT_SETTING = REPO_ROOT / "yaml/setting/setting.yaml"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
DEFAULT_STUDY_NAME = "gru-retrieval-outcome-search"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "retrieval_outcome_search"
ALLOW_INTERNAL_OUTPUT_ROOT_ENV = "NEURALFORECAST_ALLOW_INTERNAL_OUTPUT_ROOT"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping at {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(payload, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_name_list(raw: str | None, *, label: str) -> tuple[str, ...] | None:
    if raw is None:
        return None
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError(f"{label} must contain at least one name")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicates: {names}")
    return names


def _read_int_list(raw: str | None, *, label: str) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"{label} must contain at least one integer")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates: {values}")
    return values


def _sanitize_param_fragment(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _feature_param_name(prefix: str, column: str) -> str:
    return f"{prefix}__{_sanitize_param_fragment(column)}"


def _load_experiment_bundle(config_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path]:
    experiment_doc = _load_yaml_mapping(config_path)
    aa_block = experiment_doc.get("aa_forecast")
    if not isinstance(aa_block, dict):
        raise ValueError(f"{config_path}: aa_forecast must be a mapping")
    aa_config_rel = aa_block.get("config_path")
    if not isinstance(aa_config_rel, str) or not aa_config_rel.strip():
        raise ValueError(f"{config_path}: aa_forecast.config_path must be set")
    aa_config_candidate = Path(aa_config_rel)
    if aa_config_candidate.is_absolute():
        aa_config_path = aa_config_candidate.resolve()
    else:
        repo_root_candidate = (REPO_ROOT / aa_config_candidate).resolve()
        config_parent_candidate = (config_path.parent / aa_config_candidate).resolve()
        aa_config_path = (
            repo_root_candidate if repo_root_candidate.exists() else config_parent_candidate
        )
    aa_doc = _load_yaml_mapping(aa_config_path)
    aa_payload = aa_doc.get("aa_forecast")
    if not isinstance(aa_payload, dict):
        raise ValueError(f"{aa_config_path}: aa_forecast must be a mapping")
    retrieval_block = aa_payload.get("retrieval")
    if not isinstance(retrieval_block, dict):
        raise ValueError(f"{aa_config_path}: aa_forecast.retrieval must be a mapping")
    retrieval_rel = retrieval_block.get("config_path")
    if not isinstance(retrieval_rel, str) or not retrieval_rel.strip():
        raise ValueError(f"{aa_config_path}: aa_forecast.retrieval.config_path must be set")
    retrieval_path = (aa_config_path.parent / retrieval_rel).resolve()
    retrieval_doc = _load_yaml_mapping(retrieval_path)
    return experiment_doc, aa_config_path, aa_doc, retrieval_path, retrieval_doc, config_path


def _resolve_dataset_path(experiment_doc: dict[str, Any], *, override: Path | None) -> Path:
    dataset = experiment_doc.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("experiment dataset must be a mapping")
    if override is not None:
        return override
    dataset_path = dataset.get("path")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError("experiment dataset.path must be set")
    return (REPO_ROOT / dataset_path).resolve()


def _default_upward_candidates(aa_doc: dict[str, Any]) -> tuple[str, ...]:
    aa_payload = aa_doc.get("aa_forecast")
    if not isinstance(aa_payload, dict):
        raise ValueError("aa_forecast plugin doc must contain aa_forecast mapping")
    tails = aa_payload.get("star_anomaly_tails")
    if not isinstance(tails, dict):
        raise ValueError("aa_forecast plugin doc missing star_anomaly_tails mapping")
    upward = tails.get("upward")
    if not isinstance(upward, list) or not upward or not all(isinstance(item, str) and item.strip() for item in upward):
        raise ValueError("aa_forecast.star_anomaly_tails.upward must be a non-empty string list")
    return tuple(item.strip() for item in upward)


def _default_hist_candidates(experiment_doc: dict[str, Any]) -> tuple[str, ...]:
    dataset = experiment_doc.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("experiment dataset must be a mapping")
    hist_exog_cols = dataset.get("hist_exog_cols")
    if not isinstance(hist_exog_cols, list) or not hist_exog_cols or not all(
        isinstance(item, str) and item.strip() for item in hist_exog_cols
    ):
        raise ValueError("dataset.hist_exog_cols must be a non-empty string list")
    return tuple(item.strip() for item in hist_exog_cols)


def _input_size_values(*, min_input_size: int, max_input_size: int, step: int) -> tuple[int, ...]:
    if min_input_size <= 0 or max_input_size <= 0 or step <= 0:
        raise ValueError("input size range values must be positive")
    if min_input_size > max_input_size:
        raise ValueError("min_input_size must be <= max_input_size")
    return tuple(range(min_input_size, max_input_size + 1, step))


def _resolve_bundle_root(output_root: str | None, *, resume: bool) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    if resume:
        raise ValueError("--resume requires --output-root")
    return (DEFAULT_OUTPUT_ROOT / _now_tag()).resolve()


def _prepare_bundle_root(path: Path, *, resume: bool) -> None:
    if path.exists() and not resume:
        raise FileExistsError(f"output root already exists: {path}; pass --resume or choose another --output-root")
    if not path.exists() and resume:
        raise FileNotFoundError(f"resume requested but output root does not exist: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _build_retrieval_doc(base_doc: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    retrieval_doc = json.loads(json.dumps(base_doc))
    retrieval = retrieval_doc.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("baseline retrieval doc must contain top-level retrieval mapping")
    for key in (
        "top_k",
        "recency_gap_steps",
        "trigger_quantile",
        "min_similarity",
        "temperature",
        "blend_floor",
        "blend_max",
        "use_uncertainty_gate",
        "event_score_log_bonus_alpha",
        "event_score_log_bonus_cap",
    ):
        if key not in params:
            continue
        retrieval[key] = params[key]
    return retrieval_doc


def _build_aa_forecast_doc(
    base_doc: dict[str, Any],
    retrieval_config_path: Path,
    *,
    source_dir: Path,
    upward_cols: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    aa_doc = json.loads(json.dumps(base_doc))
    aa = aa_doc.get("aa_forecast")
    if not isinstance(aa, dict):
        raise ValueError("aa_forecast doc must contain top-level aa_forecast mapping")
    aa["lowess_frac"] = float(params["star_lowess_frac"])
    aa["lowess_delta"] = float(params["star_lowess_delta"])
    aa["thresh"] = float(params["star_thresh"])
    tails = aa.setdefault("star_anomaly_tails", {})
    if not isinstance(tails, dict):
        raise ValueError("aa_forecast.star_anomaly_tails must be a mapping")
    tails["upward"] = list(upward_cols)
    tails.setdefault("two_sided", [])
    retrieval = aa.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("aa_forecast.retrieval must be a mapping")
    retrieval["enabled"] = True
    retrieval["config_path"] = os.path.relpath(retrieval_config_path, start=source_dir)
    return aa_doc


def _build_experiment_doc(
    base_doc: dict[str, Any],
    aa_config_path: Path,
    *,
    dataset_path: Path,
    hist_exog_cols: list[str],
    task_name: str,
) -> dict[str, Any]:
    experiment_doc = json.loads(json.dumps(base_doc))
    task = experiment_doc.setdefault("task", {})
    if not isinstance(task, dict):
        raise ValueError("experiment task must be a mapping")
    task["name"] = task_name
    dataset = experiment_doc.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("experiment dataset must be a mapping")
    dataset["path"] = str(dataset_path)
    dataset["hist_exog_cols"] = list(hist_exog_cols)
    aa = experiment_doc.get("aa_forecast")
    if not isinstance(aa, dict):
        raise ValueError("experiment aa_forecast must be a mapping")
    aa["config_path"] = str(aa_config_path)
    return experiment_doc


def _build_setting_doc(base_doc: dict[str, Any], *, input_size: int) -> dict[str, Any]:
    setting_doc = json.loads(json.dumps(base_doc))
    training = setting_doc.get("training")
    if not isinstance(training, dict):
        raise ValueError("setting.training must be a mapping")
    training["input_size"] = int(input_size)
    return setting_doc


def _resolve_feature_selection(
    *,
    params: dict[str, Any],
    hist_candidates: tuple[str, ...],
    upward_candidates: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    selected_hist = [
        column
        for column in hist_candidates
        if bool(params.get(_feature_param_name("use_hist", column)))
    ]
    if not selected_hist:
        raise optuna.TrialPruned("no hist_exog columns selected")
    selected_upward = [
        column
        for column in upward_candidates
        if column in selected_hist and bool(params.get(_feature_param_name("use_upward", column)))
    ]
    if not selected_upward:
        raise optuna.TrialPruned("no upward columns selected")
    return selected_hist, selected_upward


def _run_experiment(
    *,
    config_path: Path,
    setting_path: Path,
    output_root: Path,
    log_path: Path,
    timeout_seconds: int,
) -> int:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env[ALLOW_INTERNAL_OUTPUT_ROOT_ENV] = "1"
    cmd = [
        "uv",
        "run",
        "python",
        "main.py",
        "--config",
        str(config_path),
        "--setting",
        str(setting_path),
        "--output-root",
        str(output_root),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    return int(completed.returncode)


def _tail_text(path: Path, *, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _validate_candidate_columns(frame: pd.DataFrame, *, hist_candidates: tuple[str, ...], upward_candidates: tuple[str, ...], target_col: str) -> None:
    columns = set(frame.columns)
    missing_target = {target_col}.difference(columns)
    if missing_target:
        raise ValueError(f"dataset missing target column(s): {sorted(missing_target)}")
    missing_hist = sorted(set(hist_candidates).difference(columns))
    if missing_hist:
        raise ValueError(f"dataset missing hist_exog candidate column(s): {missing_hist}")
    missing_upward = sorted(set(upward_candidates).difference(columns))
    if missing_upward:
        raise ValueError(f"dataset missing upward candidate column(s): {missing_upward}")
    if not set(upward_candidates).issubset(set(hist_candidates)):
        raise ValueError("upward candidate universe must be a subset of hist_exog candidate universe")


def _build_contract_metadata(
    *,
    config_path: Path,
    aa_config_path: Path,
    retrieval_path: Path,
    setting_path: Path,
    dataset_path: Path,
    spike_cutoff: str,
    recent_fold_count: int,
    hist_candidates: tuple[str, ...],
    upward_candidates: tuple[str, ...],
    top_k_values: tuple[int, ...],
    input_sizes: tuple[int, ...],
) -> dict[str, Any]:
    contract = {
        "schema_version": OUTCOME_SEARCH_SCHEMA_VERSION,
        "objective_version": OUTCOME_OBJECTIVE_VERSION,
        "config_sha256": sha256_file(config_path),
        "aa_forecast_sha256": sha256_file(aa_config_path),
        "retrieval_sha256": sha256_file(retrieval_path),
        "setting_sha256": sha256_file(setting_path),
        "dataset_sha256": sha256_file(dataset_path),
        "resolved_dataset_path": str(dataset_path.resolve()),
        "spike_cutoff": spike_cutoff,
        "recent_fold_count": int(recent_fold_count),
        "hist_candidates": list(hist_candidates),
        "upward_candidates": list(upward_candidates),
        "top_k_values": list(top_k_values),
        "input_sizes": list(input_sizes),
        "searchable_params": [
            "input_size",
            "top_k",
            "recency_gap_steps",
            "trigger_quantile",
            "min_similarity",
            "temperature",
            "blend_floor",
            "blend_max",
            "use_uncertainty_gate",
            "event_score_log_bonus_alpha",
            "event_score_log_bonus_cap",
            "star_lowess_frac",
            "star_lowess_delta",
            "star_thresh",
            "hist_exog_subset",
            "upward_subset",
        ],
    }
    return {
        "schema_version": OUTCOME_SEARCH_SCHEMA_VERSION,
        "objective_version": OUTCOME_OBJECTIVE_VERSION,
        "contract": contract,
        "compatibility_hash": compatibility_hash(contract),
    }


def _write_search_metadata(path: Path, metadata: dict[str, Any], *, study_name: str) -> None:
    payload = dict(metadata)
    payload["study_name"] = study_name
    _write_json(path, payload)


def _check_resume_metadata(path: Path, metadata: dict[str, Any], *, study_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"resume requested but metadata file is missing: {path}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("study_name") != study_name:
        raise ValueError(
            f"resume study_name mismatch: expected {existing.get('study_name')!r}, got {study_name!r}"
        )
    if existing.get("compatibility_hash") != metadata.get("compatibility_hash"):
        raise ValueError("resume compatibility hash mismatch")


def _suggest_trial_params(
    trial: optuna.trial.Trial,
    *,
    hist_candidates: tuple[str, ...],
    upward_candidates: tuple[str, ...],
    input_sizes: tuple[int, ...],
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    blend_floor = trial.suggest_float("blend_floor", 0.0, 0.25)
    blend_max = trial.suggest_float("blend_max", max(0.2, blend_floor), 1.0)
    params: dict[str, Any] = {
        "input_size": int(trial.suggest_categorical("input_size", list(input_sizes))),
        "top_k": int(trial.suggest_categorical("top_k", list(top_k_values))),
        "recency_gap_steps": int(trial.suggest_int("recency_gap_steps", 0, 24)),
        "trigger_quantile": float(trial.suggest_float("trigger_quantile", 0.005, 0.25, log=True)),
        "min_similarity": float(trial.suggest_float("min_similarity", 0.0, 0.95)),
        "temperature": float(trial.suggest_float("temperature", 0.005, 0.5, log=True)),
        "blend_floor": float(blend_floor),
        "blend_max": float(blend_max),
        "use_uncertainty_gate": bool(trial.suggest_categorical("use_uncertainty_gate", [True, False])),
        "event_score_log_bonus_alpha": float(trial.suggest_float("event_score_log_bonus_alpha", 0.0, 0.6)),
        "event_score_log_bonus_cap": float(trial.suggest_float("event_score_log_bonus_cap", 0.0, 3.0)),
        "star_lowess_frac": float(trial.suggest_float("star_lowess_frac", 0.1, 0.5)),
        "star_lowess_delta": float(trial.suggest_float("star_lowess_delta", 0.0, 0.05)),
        "star_thresh": float(trial.suggest_float("star_thresh", 1.5, 5.0)),
    }
    for column in hist_candidates:
        params[_feature_param_name("use_hist", column)] = trial.suggest_categorical(
            _feature_param_name("use_hist", column), [True, False]
        )
    for column in upward_candidates:
        params[_feature_param_name("use_upward", column)] = trial.suggest_categorical(
            _feature_param_name("use_upward", column), [True, False]
        )
    return params


def _trial_output_root(root: Path, trial_number: int) -> Path:
    return root / "trial_runs" / f"trial_{trial_number:04d}"


def _trial_payload_base(trial: optuna.trial.Trial, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_number": trial.number,
        "status": "running",
        "params": params,
    }


def _collect_trial_rows(bundle_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(bundle_root.glob("trial_runs/trial_*/trial_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "trial_number": payload.get("trial_number"),
            "status": payload.get("status"),
            "objective": payload.get("objective"),
            "pass_gate": payload.get("pass_gate"),
            "prune_reason": payload.get("prune_reason"),
        }
        params = payload.get("params") or {}
        for key, value in params.items():
            row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        outcome = payload.get("outcome") or {}
        row.update(outcome)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_best_trial_payload(*, gate_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]], study_name: str, bundle_root: Path) -> dict[str, Any]:
    base = {
        "study_name": study_name,
        "bundle_root": str(bundle_root),
    }
    if gate_rows:
        best = gate_rows[0]
        return {**base, "status": "ok", "winner_source": "leaderboard_gate_pass", "trial": best}
    if all_rows:
        best = all_rows[0]
        return {**base, "status": "ok", "winner_source": "leaderboard_all", "no_gate_pass_trials": True, "trial": best}
    return {**base, "status": "no_complete_trials"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search GRU retrieval settings against spike uplift + recent-fold MAPE")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--setting", default=str(DEFAULT_SETTING))
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--candidate-hist-exog-cols", default=None)
    parser.add_argument("--candidate-upward-cols", default=None)
    parser.add_argument("--top-k-values", default="1,2,3,4")
    parser.add_argument("--min-input-size", type=int, default=16)
    parser.add_argument("--max-input-size", type=int, default=96)
    parser.add_argument("--input-size-step", type=int, default=8)
    parser.add_argument("--spike-cutoff", default="2026-02-23")
    parser.add_argument("--recent-fold-count", type=int, default=12)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--run-timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    setting_path = Path(args.setting).expanduser().resolve()
    dataset_override = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else None
    experiment_doc, aa_config_path, aa_doc, retrieval_path, retrieval_doc, _ = _load_experiment_bundle(config_path)
    aa_payload = aa_doc["aa_forecast"]
    if str(aa_payload.get("model", "")).strip().lower() != "gru":
        raise ValueError(f"outcome search requires GRU aa_forecast model, got {aa_payload.get('model')!r}")
    retrieval_block = aa_payload.get("retrieval")
    if not isinstance(retrieval_block, dict) or not bool(retrieval_block.get("enabled", False)):
        raise ValueError("outcome search requires retrieval enabled in aa_forecast plugin config")
    setting_doc = _load_yaml_mapping(setting_path)
    cv_block = setting_doc.get("cv")
    if not isinstance(cv_block, dict):
        raise ValueError("setting.cv must be a mapping")
    horizon = int(cv_block.get("horizon", 0))
    if horizon < 2:
        raise ValueError("outcome search requires cv.horizon >= 2")
    n_windows = int(cv_block.get("n_windows", 0))
    if n_windows < int(args.recent_fold_count):
        raise ValueError(
            f"setting.cv.n_windows={n_windows} cannot satisfy recent_fold_count={args.recent_fold_count}"
        )

    dataset_path = _resolve_dataset_path(experiment_doc, override=dataset_override)
    dataset_frame = pd.read_csv(dataset_path)
    hist_candidates = _read_name_list(args.candidate_hist_exog_cols, label="candidate hist_exog cols") or _default_hist_candidates(experiment_doc)
    upward_candidates = _read_name_list(args.candidate_upward_cols, label="candidate upward cols") or _default_upward_candidates(aa_doc)
    _validate_candidate_columns(
        dataset_frame,
        hist_candidates=hist_candidates,
        upward_candidates=upward_candidates,
        target_col=str(experiment_doc["dataset"]["target_col"]),
    )
    top_k_values = _read_int_list(args.top_k_values, label="top_k_values")
    if top_k_values is None:
        raise ValueError("--top-k-values must be set")
    input_sizes = _input_size_values(
        min_input_size=int(args.min_input_size),
        max_input_size=int(args.max_input_size),
        step=int(args.input_size_step),
    )
    spike_cutoff = str(pd.Timestamp(args.spike_cutoff))
    bundle_root = _resolve_bundle_root(args.output_root, resume=bool(args.resume))
    _prepare_bundle_root(bundle_root, resume=bool(args.resume))

    metadata = _build_contract_metadata(
        config_path=config_path,
        aa_config_path=aa_config_path,
        retrieval_path=retrieval_path,
        setting_path=setting_path,
        dataset_path=dataset_path,
        spike_cutoff=spike_cutoff,
        recent_fold_count=int(args.recent_fold_count),
        hist_candidates=hist_candidates,
        upward_candidates=upward_candidates,
        top_k_values=top_k_values,
        input_sizes=input_sizes,
    )
    metadata_path = bundle_root / "search_metadata.json"
    if args.resume:
        _check_resume_metadata(metadata_path, metadata, study_name=args.study_name)
    else:
        _write_search_metadata(metadata_path, metadata, study_name=args.study_name)

    storage_path = bundle_root / "study.db"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=bool(args.resume),
        sampler=sampler,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_trial_params(
            trial,
            hist_candidates=hist_candidates,
            upward_candidates=upward_candidates,
            input_sizes=input_sizes,
            top_k_values=top_k_values,
        )
        trial_root = _trial_output_root(bundle_root, trial.number)
        if trial_root.exists():
            shutil.rmtree(trial_root)
        trial_root.mkdir(parents=True, exist_ok=True)
        payload = _trial_payload_base(trial, params)
        try:
            hist_exog_cols, upward_cols = _resolve_feature_selection(
                params=params,
                hist_candidates=hist_candidates,
                upward_candidates=upward_candidates,
            )
            config_root = trial_root / "configs"
            retrieval_out = config_root / "baseline_retrieval.yaml"
            aa_out = config_root / "aa_forecast_gru-ret.yaml"
            experiment_out = config_root / "experiment.yaml"
            setting_out = config_root / "setting.yaml"
            _write_yaml(retrieval_out, _build_retrieval_doc(retrieval_doc, params))
            _write_yaml(
                aa_out,
                _build_aa_forecast_doc(
                    aa_doc,
                    retrieval_out,
                    source_dir=aa_out.parent,
                    upward_cols=upward_cols,
                    params=params,
                ),
            )
            _write_yaml(
                experiment_out,
                _build_experiment_doc(
                    experiment_doc,
                    aa_out,
                    dataset_path=dataset_path,
                    hist_exog_cols=hist_exog_cols,
                    task_name=f"retrieval-outcome-trial-{trial.number:04d}",
                ),
            )
            _write_yaml(setting_out, _build_setting_doc(setting_doc, input_size=int(params["input_size"])))
            params["selected_hist_exog_cols"] = list(hist_exog_cols)
            params["selected_upward_cols"] = list(upward_cols)
            run_root = trial_root / "run"
            log_path = trial_root / "logs" / "run.log"
            rc = _run_experiment(
                config_path=experiment_out,
                setting_path=setting_out,
                output_root=run_root,
                log_path=log_path,
                timeout_seconds=int(args.run_timeout_seconds),
            )
            if rc != 0:
                raise RuntimeError(f"run failed with rc={rc}; log tail:\n{_tail_text(log_path)}")
            outcome = evaluate_run_outcome(
                run_root=run_root,
                spike_cutoff=pd.Timestamp(spike_cutoff),
                recent_fold_count=int(args.recent_fold_count),
                horizon=horizon,
                model_name=AAFORECAST_MODEL_NAME,
            )
            payload.update(
                {
                    "status": "complete",
                    "objective": float(outcome.objective),
                    "pass_gate": bool(outcome.pass_gate),
                    "run_root": str(run_root),
                    "outcome": outcome.to_row(),
                }
            )
            _write_json(trial_root / "trial_result.json", payload)
            trial.set_user_attr("trial_result_path", str(trial_root / "trial_result.json"))
            trial.set_user_attr("pass_gate", bool(outcome.pass_gate))
            return float(outcome.objective)
        except optuna.TrialPruned as exc:
            payload.update({"status": "pruned", "prune_reason": str(exc)})
            _write_json(trial_root / "trial_result.json", payload)
            trial.set_user_attr("trial_result_path", str(trial_root / "trial_result.json"))
            raise
        except Exception as exc:
            payload.update({"status": "failed", "prune_reason": f"{type(exc).__name__}: {exc}"})
            _write_json(trial_root / "trial_result.json", payload)
            trial.set_user_attr("trial_result_path", str(trial_root / "trial_result.json"))
            raise optuna.TrialPruned(str(exc)) from exc

    study.optimize(objective, n_trials=int(args.n_trials))
    rows = _collect_trial_rows(bundle_root)
    _write_csv(bundle_root / "trials.csv", rows)
    complete_rows = [row for row in rows if str(row.get("status")) == "complete"]
    all_rows = sorted(complete_rows, key=audit_sort_key)
    gate_rows = sorted([row for row in complete_rows if bool(row.get("pass_gate"))], key=winner_sort_key)
    _write_csv(bundle_root / "leaderboard_all.csv", all_rows)
    _write_csv(bundle_root / "leaderboard_gate_pass.csv", gate_rows)
    best_payload = _build_best_trial_payload(
        gate_rows=gate_rows,
        all_rows=all_rows,
        study_name=args.study_name,
        bundle_root=bundle_root,
    )
    _write_json(bundle_root / "best_trial.json", best_payload)

    print(f"bundle_root: {bundle_root}")
    print(f"study_db: {storage_path}")
    print(f"trials_csv: {bundle_root / 'trials.csv'}")
    print(f"leaderboard_all: {bundle_root / 'leaderboard_all.csv'}")
    print(f"leaderboard_gate_pass: {bundle_root / 'leaderboard_gate_pass.csv'}")
    print(f"best_trial_json: {bundle_root / 'best_trial.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
