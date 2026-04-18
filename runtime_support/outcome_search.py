from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from runtime_support.runner import compute_metrics


AAFORECAST_MODEL_NAME = "AAForecast"
OUTCOME_SEARCH_SCHEMA_VERSION = "retrieval-outcome-search-v1"
OUTCOME_OBJECTIVE_VERSION = "retrieval-outcome-lexicographic-v1"


@dataclass(frozen=True)
class CutoffEvaluation:
    cutoff: str
    on_mape: float
    off_mape: float
    delta_mape: float
    on_predictions: tuple[float, ...]
    off_predictions: tuple[float, ...]


@dataclass(frozen=True)
class OutcomeEvaluation:
    run_root: str
    model: str
    spike_cutoff: str
    recent_cutoffs: tuple[str, ...]
    h1_uplift: float
    h2_uplift: float
    min_uplift: float
    sum_uplift: float
    pass_gate: bool
    recent_mean_mape_on: float
    recent_mean_mape_off: float
    recent_delta_mape: float
    improved_fold_count: int
    worst_fold_regression: float
    objective: float
    provenance_mode: str = "derived_off_from_same_run_payload"

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["recent12_cutoffs_json"] = json.dumps(list(self.recent_cutoffs))
        row["recent12_mean_mape_on"] = row.pop("recent_mean_mape_on")
        row["recent12_mean_mape_off"] = row.pop("recent_mean_mape_off")
        row["recent12_delta_mape"] = row.pop("recent_delta_mape")
        row["run_root"] = row.pop("run_root")
        return row


def normalize_score(value: float, *, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    clamped = max(lower, min(upper, float(value)))
    return (clamped - lower) / (upper - lower)


def objective_value(
    *,
    pass_gate: bool,
    min_uplift: float,
    sum_uplift: float,
    recent_mean_mape_on: float,
) -> float:
    gate_score = 1.0 if pass_gate else 0.0
    min_score = normalize_score(min_uplift, lower=-1000.0, upper=1000.0)
    sum_score = normalize_score(sum_uplift, lower=-1000.0, upper=1000.0)
    mape_score = 1.0 - normalize_score(recent_mean_mape_on, lower=0.0, upper=1000.0)
    return gate_score * 1e9 + min_score * 1e6 + sum_score * 1e3 + mape_score


def winner_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["min_uplift"]),
        -float(row["sum_uplift"]),
        float(row["recent12_mean_mape_on"]),
        float(row["recent12_delta_mape"]),
        -int(row["improved_fold_count"]),
        int(row["trial_number"]),
    )


def audit_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(row["pass_gate"])),
        *winner_sort_key(row),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compatibility_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_result_frame(run_root: Path, *, model_name: str = AAFORECAST_MODEL_NAME) -> pd.DataFrame:
    result_path = run_root / "summary" / "result.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"missing summary/result.csv: {result_path}")
    frame = pd.read_csv(result_path)
    if frame.empty:
        raise ValueError(f"empty summary/result.csv: {result_path}")
    if "model" not in frame.columns:
        raise ValueError(f"summary/result.csv missing model column: {result_path}")
    frame = frame.loc[frame["model"] == model_name].copy()
    if frame.empty:
        raise ValueError(f"no {model_name} rows in {result_path}")
    required_columns = {"cutoff", "horizon_step", "y", "y_hat"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"summary/result.csv missing columns {missing_columns}: {result_path}")
    frame["cutoff_ts"] = pd.to_datetime(frame["cutoff"], errors="coerce")
    if frame["cutoff_ts"].isna().any():
        raise ValueError(f"invalid cutoff values in {result_path}")
    frame["ds_ts"] = pd.to_datetime(frame["ds"], errors="coerce") if "ds" in frame.columns else pd.NaT
    frame["horizon_step"] = pd.to_numeric(frame["horizon_step"], errors="raise").astype(int)
    return frame


def select_recent_cutoffs(frame: pd.DataFrame, *, recent_fold_count: int) -> tuple[pd.Timestamp, ...]:
    if recent_fold_count <= 0:
        raise ValueError("recent_fold_count must be positive")
    unique_cutoffs = sorted(frame["cutoff_ts"].drop_duplicates().tolist())
    if len(unique_cutoffs) < recent_fold_count:
        raise ValueError(
            f"result frame contains only {len(unique_cutoffs)} unique cutoffs; need {recent_fold_count}"
        )
    return tuple(unique_cutoffs[-recent_fold_count:])


def _artifact_column(frame: pd.DataFrame) -> str:
    for column in ("aaforecast_retrieval_artifact", "retrieval_artifact"):
        if column in frame.columns:
            return column
    raise ValueError("result frame missing retrieval artifact column")


def _load_payload(run_root: Path, artifact_relpath: str) -> dict[str, Any]:
    payload_path = (run_root / artifact_relpath).resolve()
    if not payload_path.exists():
        raise FileNotFoundError(f"missing retrieval payload: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"retrieval payload must be a JSON object: {payload_path}")
    return payload


def _ensure_finite_metric(value: float, *, label: str, cutoff: pd.Timestamp) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite {label} for cutoff={cutoff}")
    return float(value)


def _evaluate_cutoff_group(
    *,
    run_root: Path,
    group: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
) -> CutoffEvaluation:
    artifact_column = _artifact_column(group)
    ordered = group.sort_values(["horizon_step", "ds_ts"]).reset_index(drop=True)
    if ordered["horizon_step"].duplicated().any():
        raise ValueError(f"duplicate horizon_step rows for cutoff={cutoff}")
    if len(ordered) != horizon:
        raise ValueError(f"row count mismatch for cutoff={cutoff}: expected {horizon}, got {len(ordered)}")
    artifacts = [
        str(value).strip()
        for value in ordered[artifact_column].tolist()
        if str(value).strip() and str(value).strip().lower() != "nan"
    ]
    if not artifacts:
        raise ValueError(f"missing retrieval artifact path for cutoff={cutoff}")
    artifact_relpath = artifacts[0]
    payload = _load_payload(run_root, artifact_relpath)
    base_prediction = payload.get("base_prediction")
    if not isinstance(base_prediction, list):
        raise ValueError(f"retrieval payload missing base_prediction list for cutoff={cutoff}")
    if len(base_prediction) != horizon:
        raise ValueError(
            f"base_prediction length mismatch for cutoff={cutoff}: "
            f"expected {horizon}, got {len(base_prediction)}"
        )
    actual = pd.to_numeric(ordered["y"], errors="raise")
    on_prediction = pd.to_numeric(ordered["y_hat"], errors="raise")
    off_prediction = pd.Series([float(value) for value in base_prediction], dtype=float)
    on_metrics = compute_metrics(actual, on_prediction)
    off_metrics = compute_metrics(actual, off_prediction)
    on_mape = _ensure_finite_metric(float(on_metrics["MAPE"]), label="on MAPE", cutoff=cutoff)
    off_mape = _ensure_finite_metric(float(off_metrics["MAPE"]), label="off MAPE", cutoff=cutoff)
    return CutoffEvaluation(
        cutoff=str(cutoff),
        on_mape=on_mape,
        off_mape=off_mape,
        delta_mape=on_mape - off_mape,
        on_predictions=tuple(float(value) for value in on_prediction.tolist()),
        off_predictions=tuple(float(value) for value in off_prediction.tolist()),
    )


def evaluate_run_outcome(
    *,
    run_root: Path,
    spike_cutoff: pd.Timestamp,
    recent_fold_count: int,
    horizon: int,
    model_name: str = AAFORECAST_MODEL_NAME,
) -> OutcomeEvaluation:
    frame = load_result_frame(run_root, model_name=model_name)
    recent_cutoffs = select_recent_cutoffs(frame, recent_fold_count=recent_fold_count)
    grouped: dict[pd.Timestamp, CutoffEvaluation] = {}
    for cutoff, group in frame.groupby("cutoff_ts", sort=True):
        grouped[pd.Timestamp(cutoff)] = _evaluate_cutoff_group(
            run_root=run_root,
            group=group,
            cutoff=pd.Timestamp(cutoff),
            horizon=horizon,
        )
    spike_cutoff = pd.Timestamp(spike_cutoff)
    if spike_cutoff not in grouped:
        raise ValueError(f"spike cutoff missing from result.csv: {spike_cutoff}")
    spike_eval = grouped[spike_cutoff]
    if len(spike_eval.on_predictions) < 2 or len(spike_eval.off_predictions) < 2:
        raise ValueError(f"spike cutoff requires at least 2 horizon rows: {spike_cutoff}")
    recent_evals = [grouped[pd.Timestamp(cutoff)] for cutoff in recent_cutoffs]
    recent_mean_on = sum(item.on_mape for item in recent_evals) / len(recent_evals)
    recent_mean_off = sum(item.off_mape for item in recent_evals) / len(recent_evals)
    h1_uplift = spike_eval.on_predictions[0] - spike_eval.off_predictions[0]
    h2_uplift = spike_eval.on_predictions[1] - spike_eval.off_predictions[1]
    pass_gate = h1_uplift > 0.0 and h2_uplift > 0.0
    improved_fold_count = sum(1 for item in recent_evals if item.on_mape < item.off_mape)
    worst_fold_regression = max(item.delta_mape for item in recent_evals)
    return OutcomeEvaluation(
        run_root=str(run_root),
        model=model_name,
        spike_cutoff=str(spike_cutoff),
        recent_cutoffs=tuple(str(pd.Timestamp(cutoff)) for cutoff in recent_cutoffs),
        h1_uplift=float(h1_uplift),
        h2_uplift=float(h2_uplift),
        min_uplift=float(min(h1_uplift, h2_uplift)),
        sum_uplift=float(h1_uplift + h2_uplift),
        pass_gate=pass_gate,
        recent_mean_mape_on=float(recent_mean_on),
        recent_mean_mape_off=float(recent_mean_off),
        recent_delta_mape=float(recent_mean_on - recent_mean_off),
        improved_fold_count=int(improved_fold_count),
        worst_fold_regression=float(worst_fold_regression),
        objective=float(
            objective_value(
                pass_gate=pass_gate,
                min_uplift=min(h1_uplift, h2_uplift),
                sum_uplift=h1_uplift + h2_uplift,
                recent_mean_mape_on=recent_mean_on,
            )
        ),
    )
