"""Search retrieval settings that make both WTI and Brent AAForecast GRU runs spike upward.

This script runs Optuna over the shared retrieval config used by:
- yaml/experiment/feature_set_aaforecast_wti/aaforecast-gru-ret.yaml
- yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru-ret.yaml

It writes per-trial configs, logs, run artifacts, and recommended YAML outputs under a
search bundle directory so the tracked repo YAMLs remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import optuna
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_config import load_app_config
from runtime_support.runner import _default_output_root

DEFAULT_WTI_CONFIG = REPO_ROOT / "yaml/experiment/feature_set_aaforecast_wti/aaforecast-gru-ret.yaml"
DEFAULT_BRENT_CONFIG = REPO_ROOT / "yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru-ret.yaml"
DEFAULT_SETTING_PATH = REPO_ROOT / "yaml/setting/setting.yaml"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
DEFAULT_SPIKE_THRESHOLD = 0.07
DEFAULT_TRIAL_COUNT = 50
DEFAULT_SEARCH_ROOT = REPO_ROOT / "runs" / "retrieval_search"
ALLOW_INTERNAL_OUTPUT_ROOT_ENV = "NEURALFORECAST_ALLOW_INTERNAL_OUTPUT_ROOT"
DEFAULT_STUDY_NAME = "aaforecast-gru-ret-spike-search"
AAFORECAST_MODEL_NAME = "AAForecast"

STAR_SEARCH_COLUMNS: tuple[str, ...] = (
    "GPRD_THREAT",
    "GPRD",
    "GPRD_ACT",
    "Idx_OVX",
)


def _build_tail_choice_map() -> dict[str, tuple[str, ...]]:
    choices: dict[str, tuple[str, ...]] = {}
    for subset_size in range(1, len(STAR_SEARCH_COLUMNS) + 1):
        for subset in itertools.combinations(STAR_SEARCH_COLUMNS, subset_size):
            key = "__".join(column.lower() for column in subset)
            choices[key] = subset
    return choices


TAIL_CHOICE_MAP: dict[str, tuple[str, ...]] = _build_tail_choice_map()


@dataclass(frozen=True)
class SearchCase:
    key: str
    config_path: Path
    target_col: str
    dataset_path: Path
    hist_exog_cols: tuple[str, ...]


@dataclass(frozen=True)
class CaseEvaluation:
    case_key: str
    run_root: str
    log_path: str
    cutoff: str
    last_observed: float
    h1_prediction: float
    h2_prediction: float
    h1_growth: float
    h2_growth: float
    monotonic_up: bool
    spike_pass: bool
    mean_abs_pct_error: float
    retrieval_artifact: str | None
    retrieval_applied: bool | None


@dataclass(frozen=True)
class TrialEvaluation:
    objective: float
    pass_both: bool
    case_results: dict[str, CaseEvaluation]


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping at {path}")
    return payload


def _yaml_dump(payload: dict[str, Any]) -> str:
    return yaml.dump(
        payload,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_yaml_dump(payload), encoding="utf-8")


def _read_experiment_case(config_path: Path) -> SearchCase:
    payload = _load_yaml_mapping(config_path)
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"{config_path}: dataset must be a mapping")
    target_col = dataset.get("target_col")
    dataset_path = dataset.get("path")
    hist_exog_cols = dataset.get("hist_exog_cols")
    if not isinstance(target_col, str) or not target_col.strip():
        raise ValueError(f"{config_path}: dataset.target_col must be set")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError(f"{config_path}: dataset.path must be set")
    if not isinstance(hist_exog_cols, list) or not all(
        isinstance(item, str) and item.strip() for item in hist_exog_cols
    ):
        raise ValueError(f"{config_path}: dataset.hist_exog_cols must be a non-empty string list")
    return SearchCase(
        key=config_path.parent.name,
        config_path=config_path,
        target_col=target_col,
        dataset_path=(REPO_ROOT / dataset_path).resolve(),
        hist_exog_cols=tuple(item.strip() for item in hist_exog_cols),
    )


def _resolve_shared_plugin_paths(cases: list[SearchCase]) -> tuple[Path, Path]:
    aa_plugin_paths: set[Path] = set()
    for case in cases:
        payload = _load_yaml_mapping(case.config_path)
        aa = payload.get("aa_forecast")
        if not isinstance(aa, dict):
            raise ValueError(f"{case.config_path}: aa_forecast must be a mapping")
        config_path = aa.get("config_path")
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError(f"{case.config_path}: aa_forecast.config_path must be set")
        aa_plugin_paths.add((REPO_ROOT / config_path).resolve())
    if len(aa_plugin_paths) != 1:
        raise ValueError(
            "expected WTI and Brent configs to share one aa_forecast plugin path, got "
            f"{sorted(str(path) for path in aa_plugin_paths)}"
        )
    aa_plugin_path = next(iter(aa_plugin_paths))
    aa_doc = _load_yaml_mapping(aa_plugin_path)
    aa_block = aa_doc.get("aa_forecast")
    if not isinstance(aa_block, dict):
        raise ValueError(
            f"{aa_plugin_path}: expected top-level aa_forecast mapping"
        )
    retrieval = aa_block.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError(f"{aa_plugin_path}: aa_forecast.retrieval must be a mapping")
    retrieval_path_text = retrieval.get("config_path")
    if not isinstance(retrieval_path_text, str) or not retrieval_path_text.strip():
        raise ValueError(
            f"{aa_plugin_path}: aa_forecast.retrieval.config_path must be set"
        )
    retrieval_path = (aa_plugin_path.parent / retrieval_path_text).resolve()
    return aa_plugin_path, retrieval_path


def _trial_output_root(root: Path, trial_number: int) -> Path:
    return root / "trials" / f"trial-{trial_number:04d}"


def _sanitize_param_fragment(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _optional_name_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("candidate column list must contain at least one name")
    if len(set(names)) != len(names):
        raise ValueError(f"candidate column list must not contain duplicates: {names}")
    return names


def _feature_param_name(prefix: str, column: str) -> str:
    return f"{prefix}__{_sanitize_param_fragment(column)}"


def _suggest_trial_params(
    trial: optuna.trial.Trial,
    *,
    hist_exog_candidates: tuple[str, ...] | None,
    star_candidates: tuple[str, ...] | None,
) -> dict[str, Any]:
    blend_floor = trial.suggest_float("blend_floor", 0.0, 0.25)
    blend_max = trial.suggest_float("blend_max", max(0.2, blend_floor), 1.0)
    params = {
        "top_k": trial.suggest_int("top_k", 1, 8),
        "recency_gap_steps": trial.suggest_int("recency_gap_steps", 0, 16),
        "trigger_quantile": trial.suggest_float("trigger_quantile", 0.005, 0.25, log=True),
        "min_similarity": trial.suggest_float("min_similarity", 0.0, 0.95),
        "temperature": trial.suggest_float("temperature", 0.005, 0.5, log=True),
        "blend_floor": blend_floor,
        "blend_max": blend_max,
        "use_uncertainty_gate": trial.suggest_categorical(
            "use_uncertainty_gate", [True, False]
        ),
        "event_score_log_bonus_alpha": trial.suggest_float(
            "event_score_log_bonus_alpha", 0.0, 0.6
        ),
        "event_score_log_bonus_cap": trial.suggest_float(
            "event_score_log_bonus_cap", 0.0, 3.0
        ),
        "star_lowess_frac": trial.suggest_float("star_lowess_frac", 0.1, 0.5),
        "star_lowess_delta": trial.suggest_float("star_lowess_delta", 0.0, 0.05),
        "star_thresh": trial.suggest_float("star_thresh", 1.5, 5.0),
    }
    if hist_exog_candidates is None and star_candidates is None:
        params["star_anomaly_tails_upward_key"] = trial.suggest_categorical(
            "star_anomaly_tails_upward_key", list(TAIL_CHOICE_MAP)
        )
        params["star_anomaly_tails_upward"] = list(
            TAIL_CHOICE_MAP[params["star_anomaly_tails_upward_key"]]
        )
        return params

    active_hist_candidates = hist_exog_candidates or ()
    active_star_candidates = star_candidates or active_hist_candidates
    for column in active_hist_candidates:
        params[_feature_param_name("use_hist", column)] = trial.suggest_categorical(
            _feature_param_name("use_hist", column),
            [True, False],
        )
    for column in active_star_candidates:
        params[_feature_param_name("use_star", column)] = trial.suggest_categorical(
            _feature_param_name("use_star", column),
            [True, False],
        )
    return params


def _resolve_feature_selection(
    *,
    params: dict[str, Any],
    default_hist_exog_cols: tuple[str, ...],
    hist_exog_candidates: tuple[str, ...] | None,
    star_candidates: tuple[str, ...] | None,
) -> tuple[list[str], list[str]]:
    if hist_exog_candidates is None and star_candidates is None:
        return list(default_hist_exog_cols), list(params["star_anomaly_tails_upward"])

    candidate_hist = hist_exog_candidates or default_hist_exog_cols
    candidate_star = star_candidates or candidate_hist
    hist_selected = [
        column
        for column in candidate_hist
        if params.get(_feature_param_name("use_hist", column), False)
        or params.get(_feature_param_name("use_star", column), False)
    ]
    if not hist_selected:
        hist_selected = [candidate_hist[0]]
    star_selected = [
        column
        for column in candidate_star
        if column in hist_selected
        and params.get(_feature_param_name("use_star", column), False)
    ]
    if not star_selected:
        fallback_star = next(
            (column for column in hist_selected if column in candidate_star),
            None,
        )
        if fallback_star is None:
            fallback_star = candidate_star[0]
            if fallback_star not in hist_selected:
                hist_selected = [*hist_selected, fallback_star]
        star_selected = [fallback_star]
    return hist_selected, star_selected


def _build_retrieval_doc(
    base_doc: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    retrieval_doc = json.loads(json.dumps(base_doc))
    retrieval = retrieval_doc.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("baseline retrieval doc must contain top-level retrieval mapping")
    retrieval["top_k"] = int(params["top_k"])
    retrieval["recency_gap_steps"] = int(params["recency_gap_steps"])
    retrieval["trigger_quantile"] = float(params["trigger_quantile"])
    retrieval["min_similarity"] = float(params["min_similarity"])
    retrieval["temperature"] = float(params["temperature"])
    retrieval["blend_floor"] = float(params["blend_floor"])
    retrieval["blend_max"] = float(params["blend_max"])
    retrieval["use_uncertainty_gate"] = bool(params["use_uncertainty_gate"])
    retrieval["event_score_log_bonus_alpha"] = float(
        params["event_score_log_bonus_alpha"]
    )
    retrieval["event_score_log_bonus_cap"] = float(
        params["event_score_log_bonus_cap"]
    )
    return retrieval_doc


def _build_aa_forecast_doc(
    base_doc: dict[str, Any],
    retrieval_config_path: Path,
    *,
    source_dir: Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    aa_doc = json.loads(json.dumps(base_doc))
    aa = aa_doc.get("aa_forecast")
    if not isinstance(aa, dict):
        raise ValueError("aa_forecast doc must contain top-level aa_forecast mapping")
    aa["lowess_frac"] = float(params["star_lowess_frac"])
    aa["lowess_delta"] = float(params["star_lowess_delta"])
    aa["thresh"] = float(params["star_thresh"])
    star_tails = aa.setdefault("star_anomaly_tails", {})
    if not isinstance(star_tails, dict):
        raise ValueError("aa_forecast.star_anomaly_tails must be a mapping")
    # Optuna persists only suggested params in trial.params. The upward tails list is a
    # derived value, so rebuild it from the persisted selector key when needed.
    upward = params.get("star_anomaly_tails_upward")
    if upward is None:
        choice_key = params.get("star_anomaly_tails_upward_key")
        if isinstance(choice_key, str) and choice_key in TAIL_CHOICE_MAP:
            upward = list(TAIL_CHOICE_MAP[choice_key])
        else:
            raise KeyError(
                "star_anomaly_tails_upward (or star_anomaly_tails_upward_key) is required "
                "to build the aa_forecast STAR tails recommendation"
            )
    star_tails["upward"] = list(upward)
    star_tails.setdefault("two_sided", [])
    retrieval = aa.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("aa_forecast.retrieval must be a mapping")
    retrieval["config_path"] = os.path.relpath(
        retrieval_config_path,
        start=source_dir,
    )
    return aa_doc


def _build_experiment_doc(
    base_doc: dict[str, Any],
    aa_config_path: Path,
    *,
    task_name: str,
    dataset_path: Path | None = None,
    hist_exog_cols: list[str] | None = None,
) -> dict[str, Any]:
    experiment_doc = json.loads(json.dumps(base_doc))
    task = experiment_doc.setdefault("task", {})
    if not isinstance(task, dict):
        raise ValueError("experiment task must be a mapping")
    task["name"] = task_name
    aa = experiment_doc.get("aa_forecast")
    if not isinstance(aa, dict):
        raise ValueError("experiment aa_forecast must be a mapping")
    aa["config_path"] = str(aa_config_path)
    dataset = experiment_doc.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("experiment dataset must be a mapping")
    if dataset_path is not None:
        dataset["path"] = str(dataset_path)
    if hist_exog_cols is not None:
        dataset["hist_exog_cols"] = list(hist_exog_cols)
    return experiment_doc


def _load_dataset_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "dt" not in frame.columns:
        raise ValueError(f"dataset missing dt column: {path}")
    frame = frame.copy()
    frame["dt"] = pd.to_datetime(frame["dt"])
    return frame


def _validate_candidate_columns(
    *,
    cases: list[SearchCase],
    dataset_frames: dict[str, pd.DataFrame],
    hist_exog_candidates: tuple[str, ...] | None,
    star_candidates: tuple[str, ...] | None,
) -> None:
    for case in cases:
        frame_columns = set(dataset_frames[case.key].columns)
        missing_targets = {case.target_col}.difference(frame_columns)
        if missing_targets:
            raise ValueError(
                f"{case.key}: dataset is missing target column(s): {sorted(missing_targets)}"
            )
        for label, candidates in (
            ("hist_exog", hist_exog_candidates),
            ("star", star_candidates),
        ):
            if candidates is None:
                continue
            missing = sorted(set(candidates).difference(frame_columns))
            if missing:
                raise ValueError(
                    f"{case.key}: dataset is missing {label} candidate column(s): {missing}"
                )


def _latest_result_rows(run_root: Path) -> pd.DataFrame:
    result_path = run_root / "summary" / "result.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"missing summary/result.csv: {result_path}")
    frame = pd.read_csv(result_path)
    if frame.empty:
        raise ValueError(f"empty summary/result.csv: {result_path}")
    if "model" in frame.columns:
        frame = frame.loc[frame["model"] == AAFORECAST_MODEL_NAME].copy()
    if frame.empty:
        raise ValueError(f"no {AAFORECAST_MODEL_NAME} rows in {result_path}")
    if "fold_idx" in frame.columns:
        frame = frame.loc[frame["fold_idx"] == frame["fold_idx"].max()].copy()
    frame = frame.sort_values("horizon_step").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(f"expected at least 2 horizon rows in {result_path}")
    return frame.iloc[:2].copy()


def _latest_retrieval_artifact(run_root: Path) -> tuple[str | None, bool | None]:
    forecast_path = run_root / "cv" / f"{AAFORECAST_MODEL_NAME}_forecasts.csv"
    if not forecast_path.exists():
        return None, None
    frame = pd.read_csv(forecast_path)
    if frame.empty:
        return None, None
    if "fold_idx" in frame.columns:
        frame = frame.loc[frame["fold_idx"] == frame["fold_idx"].max()].copy()
    if "horizon_step" in frame.columns:
        frame = frame.sort_values("horizon_step")
    row = frame.iloc[-1]
    artifact = row.get("aaforecast_retrieval_artifact")
    applied = row.get("aaforecast_retrieval_applied")
    artifact_text = None if pd.isna(artifact) else str(artifact)
    applied_value = None if pd.isna(applied) else bool(applied)
    return artifact_text, applied_value


def _evaluate_case_run(
    *,
    case: SearchCase,
    dataset_frame: pd.DataFrame,
    run_root: Path,
    log_path: Path,
    spike_threshold: float,
) -> CaseEvaluation:
    rows = _latest_result_rows(run_root)
    cutoff = pd.to_datetime(rows.iloc[0]["cutoff"])
    matches = dataset_frame.loc[dataset_frame["dt"] == cutoff, case.target_col]
    if matches.empty:
        raise ValueError(
            f"{case.key}: unable to find dataset row for cutoff {cutoff} in {case.dataset_path}"
        )
    last_observed = float(matches.iloc[-1])
    h1_prediction = float(rows.iloc[0]["y_hat"])
    h2_prediction = float(rows.iloc[1]["y_hat"])
    h1_growth = _relative_growth(h1_prediction, last_observed)
    h2_growth = _relative_growth(h2_prediction, last_observed)
    mean_abs_pct_error = float(
        ((rows["y_hat"] - rows["y"]).abs() / rows["y"].abs()).mean()
    )
    retrieval_artifact, retrieval_applied = _latest_retrieval_artifact(run_root)
    monotonic_up = h2_prediction > h1_prediction
    spike_pass = monotonic_up and h2_growth >= spike_threshold
    return CaseEvaluation(
        case_key=case.key,
        run_root=str(run_root),
        log_path=str(log_path),
        cutoff=str(cutoff),
        last_observed=last_observed,
        h1_prediction=h1_prediction,
        h2_prediction=h2_prediction,
        h1_growth=h1_growth,
        h2_growth=h2_growth,
        monotonic_up=monotonic_up,
        spike_pass=spike_pass,
        mean_abs_pct_error=mean_abs_pct_error,
        retrieval_artifact=retrieval_artifact,
        retrieval_applied=retrieval_applied,
    )


def _relative_growth(prediction: float, anchor: float) -> float:
    if anchor == 0:
        raise ValueError("cannot compute growth relative to zero anchor")
    return (prediction - anchor) / abs(anchor)


def _case_objective_score(case: CaseEvaluation) -> float:
    monotonic_bonus = 0.025 if case.monotonic_up else -0.025
    return (
        case.h2_growth
        + 0.35 * case.h1_growth
        + monotonic_bonus
        - 0.15 * case.mean_abs_pct_error
    )


def _score_trial_result(
    case_results: dict[str, CaseEvaluation],
    *,
    spike_threshold: float,
) -> TrialEvaluation:
    case_scores = {key: _case_objective_score(value) for key, value in case_results.items()}
    weakest_score = min(case_scores.values())
    average_score = mean(case_scores.values())
    weakest_gap = min(value.h2_growth - spike_threshold for value in case_results.values())
    pass_both = all(value.spike_pass for value in case_results.values())
    objective = weakest_score + 0.25 * average_score + 0.50 * weakest_gap
    if pass_both:
        objective += 1.0
    return TrialEvaluation(
        objective=objective,
        pass_both=pass_both,
        case_results=case_results,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _collect_trial_rows(bundle_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(bundle_root.glob("trials/trial-*/trial_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "trial_number": payload.get("trial_number"),
            "objective": payload.get("objective"),
            "pass_both": payload.get("pass_both"),
            "status": payload.get("status"),
        }
        params = payload.get("params") or {}
        for key, value in params.items():
            row[key] = json.dumps(value) if isinstance(value, list) else value
        for case_key, case_data in (payload.get("case_results") or {}).items():
            row[f"{case_key}_h1_growth"] = case_data.get("h1_growth")
            row[f"{case_key}_h2_growth"] = case_data.get("h2_growth")
            row[f"{case_key}_monotonic_up"] = case_data.get("monotonic_up")
            row[f"{case_key}_spike_pass"] = case_data.get("spike_pass")
            row[f"{case_key}_run_root"] = case_data.get("run_root")
        rows.append(row)
    return rows


def _write_trials_csv(bundle_root: Path) -> Path:
    rows = _collect_trial_rows(bundle_root)
    out_path = bundle_root / "trials.csv"
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return out_path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _study_storage_path(bundle_root: Path) -> Path:
    return bundle_root / "study.db"


def _resolve_bundle_root(output_root: str | None, *, resume: bool) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    if resume:
        raise ValueError("--resume requires --output-root")
    return (DEFAULT_SEARCH_ROOT / _now_tag()).resolve()


def _prepare_bundle_root(path: Path, *, resume: bool) -> None:
    if path.exists() and not resume:
        raise FileExistsError(
            f"output root already exists: {path}; pass --resume or choose another --output-root"
        )
    path.mkdir(parents=True, exist_ok=True)


def _resolve_run_root(config_path: Path) -> Path:
    loaded = load_app_config(REPO_ROOT, config_path=str(config_path))
    return _default_output_root(REPO_ROOT, loaded)


def _build_recommendations(
    *,
    bundle_root: Path,
    base_retrieval_doc: dict[str, Any],
    base_aa_doc: dict[str, Any],
    best_params: dict[str, Any],
) -> dict[str, str]:
    recommended_retrieval_path = bundle_root / "recommended_baseline_retrieval.yaml"
    recommended_aa_path = bundle_root / "recommended_aa_forecast_gru-ret.yaml"
    _write_yaml(
        recommended_retrieval_path,
        _build_retrieval_doc(base_retrieval_doc, best_params),
    )
    aa_doc = _build_aa_forecast_doc(
        base_aa_doc,
        retrieval_config_path=recommended_retrieval_path,
        source_dir=recommended_aa_path.parent,
        params=best_params,
    )
    _write_yaml(recommended_aa_path, aa_doc)
    return {
        "recommended_baseline_retrieval": str(recommended_retrieval_path),
        "recommended_aa_forecast": str(recommended_aa_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search retrieval settings for WTI/Brent AAForecast GRU spike behavior"
    )
    parser.add_argument("--wti-config", default=str(DEFAULT_WTI_CONFIG), help="WTI experiment config path")
    parser.add_argument("--brent-config", default=str(DEFAULT_BRENT_CONFIG), help="Brent experiment config path")
    parser.add_argument("--setting", default=str(DEFAULT_SETTING_PATH), help="Shared setting YAML path")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override dataset.path for both WTI and Brent experiment configs",
    )
    parser.add_argument(
        "--candidate-hist-exog-cols",
        default=None,
        help="Comma-separated hist_exog candidate universe; when provided, search subsets instead of keeping hist_exog_cols fixed",
    )
    parser.add_argument(
        "--candidate-star-cols",
        default=None,
        help="Comma-separated STAR candidate universe; defaults to candidate-hist-exog-cols when subset search is enabled",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIAL_COUNT, help="Optuna trial count")
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=None,
        help="Stop the whole study after this many minutes",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-case runtime timeout in seconds",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Bundle directory for configs, logs, study DB, and recommended YAMLs",
    )
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--resume", action="store_true", help="Resume an existing study bundle")
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=DEFAULT_SPIKE_THRESHOLD,
        help="Required H2 growth vs last observed value for each asset",
    )
    parser.add_argument("--seed", type=int, default=7, help="Optuna sampler seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    bundle_root = _resolve_bundle_root(args.output_root, resume=args.resume)
    _prepare_bundle_root(bundle_root, resume=args.resume)

    wti_case = _read_experiment_case(Path(args.wti_config).expanduser().resolve())
    brent_case = _read_experiment_case(Path(args.brent_config).expanduser().resolve())
    cases = [wti_case, brent_case]
    hist_exog_candidates = _optional_name_list(args.candidate_hist_exog_cols)
    star_candidates = _optional_name_list(args.candidate_star_cols)
    if star_candidates is not None and hist_exog_candidates is None:
        raise ValueError("--candidate-star-cols requires --candidate-hist-exog-cols")
    aa_plugin_path, retrieval_path = _resolve_shared_plugin_paths(cases)

    base_retrieval_doc = _load_yaml_mapping(retrieval_path)
    base_aa_doc = _load_yaml_mapping(aa_plugin_path)
    base_experiment_docs = {
        case.key: _load_yaml_mapping(case.config_path) for case in cases
    }
    dataset_path_override = (
        Path(args.dataset_path).expanduser().resolve()
        if args.dataset_path is not None
        else None
    )
    dataset_frames = {
        case.key: _load_dataset_frame(dataset_path_override or case.dataset_path) for case in cases
    }
    _validate_candidate_columns(
        cases=cases,
        dataset_frames=dataset_frames,
        hist_exog_candidates=hist_exog_candidates,
        star_candidates=star_candidates,
    )
    setting_path = Path(args.setting).expanduser().resolve()
    timeout_seconds = int(args.run_timeout_seconds)
    time_budget_seconds = None
    if args.timeout_minutes is not None:
        time_budget_seconds = max(1.0, float(args.timeout_minutes) * 60.0)

    storage_path = _study_storage_path(bundle_root)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=args.resume,
        sampler=sampler,
    )

    metadata = {
        "wti_config": str(wti_case.config_path),
        "brent_config": str(brent_case.config_path),
        "aa_forecast_plugin": str(aa_plugin_path),
        "baseline_retrieval": str(retrieval_path),
        "setting_path": str(setting_path),
        "dataset_path_override": None if dataset_path_override is None else str(dataset_path_override),
        "candidate_hist_exog_cols": None if hist_exog_candidates is None else list(hist_exog_candidates),
        "candidate_star_cols": None if star_candidates is None else list(star_candidates),
        "spike_threshold": float(args.spike_threshold),
        "scope_note": (
            "AAForecast retrieval core lives in baseline_retrieval.yaml, but STAR tails/"
            "lowess/thresh still live in aa_forecast_gru-ret.yaml. The search bundle writes"
            " recommendations for both files so the winning trial is reproducible."
        ),
    }
    _write_json(bundle_root / "search_metadata.json", metadata)

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_trial_params(
            trial,
            hist_exog_candidates=hist_exog_candidates,
            star_candidates=star_candidates,
        )
        trial_root = _trial_output_root(bundle_root, trial.number)
        config_root = trial_root / "configs"
        logs_root = trial_root / "logs"
        runs_root = trial_root / "runs"
        if trial_root.exists():
            shutil.rmtree(trial_root)
        trial_root.mkdir(parents=True, exist_ok=True)

        selected_hist_exog_cols, selected_star_cols = _resolve_feature_selection(
            params=params,
            default_hist_exog_cols=wti_case.hist_exog_cols,
            hist_exog_candidates=hist_exog_candidates,
            star_candidates=star_candidates,
        )
        params["selected_hist_exog_cols"] = list(selected_hist_exog_cols)
        params["star_anomaly_tails_upward"] = list(selected_star_cols)
        retrieval_out = config_root / "baseline_retrieval.yaml"
        aa_out = config_root / "aa_forecast_gru-ret.yaml"
        _write_yaml(retrieval_out, _build_retrieval_doc(base_retrieval_doc, params))
        _write_yaml(
            aa_out,
            _build_aa_forecast_doc(
                base_aa_doc,
                retrieval_config_path=retrieval_out,
                source_dir=aa_out.parent,
                params=params,
            ),
        )

        case_results: dict[str, CaseEvaluation] = {}
        trial_payload: dict[str, Any] = {
            "trial_number": trial.number,
            "params": params,
            "status": "running",
            "config_paths": {
                "baseline_retrieval": str(retrieval_out),
                "aa_forecast": str(aa_out),
            },
        }

        try:
            for case in cases:
                experiment_out = config_root / f"{case.key}.yaml"
                explicit_run_root = runs_root / case.key
                task_name = f"search-{case.key}-trial-{trial.number:04d}"
                experiment_doc = _build_experiment_doc(
                    base_experiment_docs[case.key],
                    aa_out,
                    task_name=task_name,
                    dataset_path=dataset_path_override,
                    hist_exog_cols=selected_hist_exog_cols,
                )
                _write_yaml(experiment_out, experiment_doc)
                rc = _run_experiment(
                    config_path=experiment_out,
                    setting_path=setting_path,
                    output_root=explicit_run_root,
                    log_path=logs_root / f"{case.key}.log",
                    timeout_seconds=timeout_seconds,
                )
                if rc != 0:
                    raise RuntimeError(
                        f"{case.key} run failed with rc={rc}; log tail:\n{_tail_text(logs_root / f'{case.key}.log')}"
                    )
                case_results[case.key] = _evaluate_case_run(
                    case=case,
                    dataset_frame=dataset_frames[case.key],
                    run_root=explicit_run_root,
                    log_path=logs_root / f"{case.key}.log",
                    spike_threshold=float(args.spike_threshold),
                )
                trial_payload.setdefault("expected_default_run_root", {})[case.key] = str(
                    _resolve_run_root(experiment_out)
                )
            trial_result = _score_trial_result(
                case_results,
                spike_threshold=float(args.spike_threshold),
            )
            trial_payload.update(
                {
                    "status": "ok",
                    "objective": trial_result.objective,
                    "pass_both": trial_result.pass_both,
                    "case_results": {
                        key: asdict(value) for key, value in trial_result.case_results.items()
                    },
                }
            )
            _write_json(trial_root / "trial_result.json", trial_payload)
            trial.set_user_attr("trial_result_path", str(trial_root / "trial_result.json"))
            trial.set_user_attr("pass_both", trial_result.pass_both)
            for key, value in trial_result.case_results.items():
                trial.set_user_attr(f"{key}_h2_growth", value.h2_growth)
                trial.set_user_attr(f"{key}_spike_pass", value.spike_pass)
            return float(trial_result.objective)
        except Exception as exc:
            trial_payload.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "case_results": {key: asdict(value) for key, value in case_results.items()},
                }
            )
            _write_json(trial_root / "trial_result.json", trial_payload)
            trial.set_user_attr("trial_result_path", str(trial_root / "trial_result.json"))
            trial.set_user_attr("failed", True)
            return -1_000_000.0

    study.optimize(objective, n_trials=args.n_trials, timeout=time_budget_seconds)

    trials_csv = _write_trials_csv(bundle_root)
    best_trial = study.best_trial
    recommendations = _build_recommendations(
        bundle_root=bundle_root,
        base_retrieval_doc=base_retrieval_doc,
        base_aa_doc=base_aa_doc,
        best_params=dict(best_trial.params),
    )
    summary = {
        "study_name": study.study_name,
        "storage": str(storage_path),
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": dict(best_trial.params),
        "trial_count": len(study.trials),
        "recommendations": recommendations,
        "trials_csv": str(trials_csv),
    }
    best_result_path = best_trial.user_attrs.get("trial_result_path")
    if isinstance(best_result_path, str):
        summary["best_trial_result"] = best_result_path
    _write_json(bundle_root / "best_trial.json", summary)

    print(f"bundle_root: {bundle_root}")
    print(f"study_db: {storage_path}")
    print(f"trials_csv: {trials_csv}")
    print(f"best_trial: {best_trial.number}")
    print(f"best_value: {best_trial.value}")
    print(f"recommended_baseline_retrieval: {recommendations['recommended_baseline_retrieval']}")
    print(f"recommended_aa_forecast: {recommendations['recommended_aa_forecast']}")
    if best_result_path:
        print(f"best_trial_result: {best_result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
