"""Find AAForecast retrieval settings that pull target windows into top-k neighbors.

This script is intentionally narrower than ``scripts/finding.py``:

- Uses the **AAForecast exact retrieval signature path** (``model._compute_star_outputs``),
  not the standalone retrieval-plugin STAR proxy.
- Keeps ``insample_y_included=false`` as a hard contract.
- Scores combos by whether specified target windows appear inside retrieval top-k.

Example:
  uv run python scripts/final_find.py \
    --config yaml/experiment/feature_set_aaforecast_wti/aaforecast-timexer-ret.yaml \
    --data-path data/test.csv \
    --targets-json runs/targets_wti.json \
    --input-sizes 32,48,64 \
    --output-csv runs/final_find/results.csv \
    --summary-json runs/final_find/summary.json

  # Inline pools: generate all non-empty subsets from the provided columns
  uv run python scripts/final_find.py \
    --config yaml/experiment/feature_set_aaforecast_wti/aaforecast-timexer-ret.yaml \
    --targets-json runs/targets_wti.json \
    --min-input-size 12 \
    --max-input-size 96 \
    --top-k-values 1,2,3,4 \
    --exog-grid GPRD_THREAT,BS_Core_Index_A,BS_Core_Index_C,GPRD,GPRD_ACT,BS_Core_Index_B,Idx_OVX,Com_LMEX,Com_BloombergCommodity_BCOM,Idx_DxyUSD \
    --upward-grid GPRD_THREAT,BS_Core_Index_A,BS_Core_Index_C,GPRD,GPRD_ACT,BS_Core_Index_B,Idx_OVX \
    --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import optuna
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_config import LoadedConfig, load_app_config
from plugins.aa_forecast.config import (
    AAForecastPluginConfig,
    resolve_aa_forecast_hist_exog,
)
from plugins.aa_forecast.runtime import (
    _aa_params_override,
    _build_event_memory_bank,
    _build_event_query,
    _retrieve_event_neighbors,
)
from plugins.retrieval.runtime import _effective_event_threshold
from runtime_support.forecast_models import build_model
from runtime_support.runner import _build_fold_diff_context, _transform_training_frame
from tuning.search_space import build_optuna_sampler, optuna_seed


@dataclass(frozen=True)
class TargetWindow:
    label: str
    candidate_end_ds: str
    normalized_candidate_end_ds: str
    notes: str | None = None


@dataclass(frozen=True)
class ComboSpec:
    input_size: int
    hist_exog_cols: tuple[str, ...]
    upward_cols: tuple[str, ...]
    top_k: int


_WORKER_LOADED: LoadedConfig | None = None
_WORKER_TRAIN_DF: pd.DataFrame | None = None
_WORKER_FUTURE_DF: pd.DataFrame | None = None


def _parse_int_list(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_name_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _expand_non_empty_subsets(columns: Sequence[str]) -> list[list[str]]:
    unique = tuple(dict.fromkeys(columns))
    out: list[list[str]] = []
    for size in range(1, len(unique) + 1):
        for combo in itertools.combinations(unique, size):
            out.append(list(combo))
    return out


def _load_json_grid(path: Path | str) -> list[list[str]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"grid file not found: {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    out: list[list[str]] = []
    for row in data:
        if not isinstance(row, list):
            raise ValueError(f"{path}: each entry must be a list of column names")
        out.append([str(x) for x in row])
    return out


def _load_grid_or_inline_pool(raw: Path | str | None) -> list[list[str]] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate_path = Path(text)
    if candidate_path.exists():
        return _load_json_grid(candidate_path)
    names = _parse_name_list(text)
    if not names:
        raise ValueError("inline grid column pool must not be empty")
    return _expand_non_empty_subsets(names)


def _normalize_candidate_end_ds(value: str) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize().strftime("%Y-%m-%d")


def _load_targets_json(path: Path) -> list[TargetWindow]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"targets json not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("targets json must be a non-empty array")
    out: list[TargetWindow] = []
    seen: set[str] = set()
    for idx, item in enumerate(payload):
        if isinstance(item, str):
            candidate_end_ds = item
            label = item
            notes = None
        elif isinstance(item, dict):
            if "candidate_end_ds" not in item:
                raise ValueError(
                    f"targets[{idx}] must contain candidate_end_ds when using object form"
                )
            candidate_end_ds = str(item["candidate_end_ds"])
            label = str(item.get("label") or candidate_end_ds)
            notes = None if item.get("notes") is None else str(item["notes"])
        else:
            raise ValueError("each target must be either a string or an object")
        normalized = _normalize_candidate_end_ds(candidate_end_ds)
        if normalized in seen:
            raise ValueError(f"duplicate target candidate_end_ds after normalization: {normalized}")
        seen.add(normalized)
        out.append(
            TargetWindow(
                label=label,
                candidate_end_ds=candidate_end_ds,
                normalized_candidate_end_ds=normalized,
                notes=notes,
            )
        )
    return out


def resolve_train_future_frames(
    loaded: LoadedConfig,
    source_df: pd.DataFrame,
    *,
    eval_slice: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from runtime_support.runner import _build_tscv_splits

    horizon = int(loaded.config.cv.horizon)
    if eval_slice == "last_cv_fold":
        splits = _build_tscv_splits(len(source_df), loaded.config.cv)
        train_idx, test_idx = splits[-1]
        train_df = source_df.iloc[train_idx].reset_index(drop=True)
        future_df = source_df.iloc[test_idx].reset_index(drop=True)
        return train_df, future_df
    if eval_slice == "max_tail":
        if len(source_df) <= horizon:
            raise ValueError(
                "max_tail eval_slice needs len(source_df) > cv.horizon "
                f"(rows={len(source_df)}, horizon={horizon})"
            )
        train_df = source_df.iloc[:-horizon].reset_index(drop=True)
        future_df = source_df.iloc[-horizon:].reset_index(drop=True)
        return train_df, future_df
    raise ValueError(f"unknown eval_slice: {eval_slice!r}")


def iter_combos(
    *,
    input_sizes: Sequence[int],
    exog_grid: list[list[str]],
    upward_grid: list[list[str]],
    base_dataset_cols: set[str],
) -> Iterable[tuple[int, tuple[str, ...], tuple[str, ...]]]:
    for input_size in input_sizes:
        for exog_cols in exog_grid:
            exog_t = tuple(exog_cols)
            missing = sorted(set(exog_t).difference(base_dataset_cols))
            if missing:
                raise ValueError(
                    f"hist_exog_cols {exog_t!r} must exist in dataset columns (missing: {missing})"
                )
            for upward_cols in upward_grid:
                upward_t = tuple(upward_cols)
                if not set(upward_t).issubset(set(exog_t)):
                    continue
                yield input_size, exog_t, upward_t


def count_incompatible_exog_upward_pairs(
    *,
    input_sizes: Sequence[int],
    exog_grid: list[list[str]],
    upward_grid: list[list[str]],
    base_dataset_cols: set[str],
) -> int:
    n = 0
    for _input_size in input_sizes:
        for exog_cols in exog_grid:
            exog_t = tuple(exog_cols)
            if not set(exog_t).issubset(base_dataset_cols):
                continue
            for upward_cols in upward_grid:
                if not set(upward_cols).issubset(set(exog_t)):
                    n += 1
    return n


def _validate_grid_columns_exist(
    *,
    grid_name: str,
    grid: Sequence[Sequence[str]],
    dataset_columns: set[str],
) -> None:
    missing = sorted({column for row in grid for column in row if column not in dataset_columns})
    if missing:
        raise ValueError(
            f"{grid_name} contains column(s) missing from dataset: {missing}"
        )


def _build_combo_loaded(
    loaded: LoadedConfig,
    *,
    input_size: int,
    hist_exog_cols: tuple[str, ...],
    upward_cols: tuple[str, ...],
    top_k: int,
) -> LoadedConfig:
    stage_cfg = loaded.config.stage_plugin_config
    if not isinstance(stage_cfg, AAForecastPluginConfig):
        raise ValueError("final_find.py requires an aa_forecast experiment config")
    if not stage_cfg.retrieval.enabled:
        raise ValueError("final_find.py requires aa_forecast.retrieval.enabled=true")
    if stage_cfg.retrieval.insample_y_included:
        raise ValueError(
            "final_find.py requires aa_forecast.retrieval.insample_y_included=false"
        )
    combo_stage = replace(
        stage_cfg,
        star_anomaly_tails={
            "upward": tuple(upward_cols),
            "two_sided": tuple(stage_cfg.star_anomaly_tails.get("two_sided", ())),
        },
        retrieval=replace(stage_cfg.retrieval, top_k=top_k),
    )
    combo_stage = resolve_aa_forecast_hist_exog(
        combo_stage, hist_exog_cols=hist_exog_cols
    )
    combo_config = replace(
        loaded.config,
        dataset=replace(loaded.config.dataset, hist_exog_cols=hist_exog_cols),
        training=replace(
            loaded.config.training,
            input_size=input_size,
            accelerator="cpu",
        ),
        stage_plugin_config=combo_stage,
    )
    return replace(loaded, config=combo_config)


def _target_match_payload(
    target: TargetWindow,
    neighbors: list[dict[str, Any]],
) -> dict[str, Any]:
    matched_rank: int | None = None
    matched_neighbor: dict[str, Any] | None = None
    normalized_neighbors: list[str] = []
    for idx, neighbor in enumerate(neighbors, start=1):
        normalized = _normalize_candidate_end_ds(str(neighbor["candidate_end_ds"]))
        normalized_neighbors.append(normalized)
        if normalized == target.normalized_candidate_end_ds and matched_rank is None:
            matched_rank = idx
            matched_neighbor = neighbor
    top1 = neighbors[0] if neighbors else None
    return {
        "match_found": matched_rank is not None,
        "matched_rank": matched_rank if matched_rank is not None else 0,
        "matched_candidate_end_ds": (
            str(matched_neighbor["candidate_end_ds"]) if matched_neighbor is not None else ""
        ),
        "matched_similarity": (
            float(matched_neighbor["similarity"]) if matched_neighbor is not None else float("nan")
        ),
        "best_neighbor_candidate_end_ds": (
            str(top1["candidate_end_ds"]) if top1 is not None else ""
        ),
        "best_neighbor_similarity": (
            float(top1["similarity"]) if top1 is not None else float("nan")
        ),
        "top_neighbors_candidate_end_ds": json.dumps(
            [str(n["candidate_end_ds"]) for n in neighbors], ensure_ascii=False
        ),
        "top_neighbors_normalized_end_ds": json.dumps(normalized_neighbors, ensure_ascii=False),
    }


def evaluate_exact_combo(
    *,
    loaded: LoadedConfig,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    input_size: int,
    hist_exog_cols: tuple[str, ...],
    upward_cols: tuple[str, ...],
    top_k: int,
) -> dict[str, Any]:
    combo_loaded = _build_combo_loaded(
        loaded,
        input_size=input_size,
        hist_exog_cols=hist_exog_cols,
        upward_cols=upward_cols,
        top_k=top_k,
    )
    job = combo_loaded.config.jobs[0]
    dt_col = combo_loaded.config.dataset.dt_col
    target_col = combo_loaded.config.dataset.target_col
    horizon = int(combo_loaded.config.cv.horizon)
    diff_context = _build_fold_diff_context(combo_loaded, train_df)
    transformed_train_df = _transform_training_frame(train_df, diff_context)
    if len(transformed_train_df) < input_size:
        return {
            "error": "short_transformed_train",
            "input_size": input_size,
            "transformed_rows": len(transformed_train_df),
        }
    model = build_model(
        combo_loaded.config,
        job,
        params_override=_aa_params_override(combo_loaded),
    )
    bank, candidate_count = _build_event_memory_bank(
        model=model,
        transformed_train_df=transformed_train_df,
        raw_train_df=train_df.reset_index(drop=True),
        dt_col=dt_col,
        target_col=target_col,
        retrieval_cfg=combo_loaded.config.stage_plugin_config.retrieval,
        input_size=input_size,
        horizon=horizon,
    )
    query = _build_event_query(
        model=model,
        transformed_train_df=transformed_train_df,
        target_col=target_col,
        retrieval_cfg=combo_loaded.config.stage_plugin_config.retrieval,
        input_size=input_size,
    )
    effective = _effective_event_threshold(
        bank=bank,
        retrieval_cfg=combo_loaded.config.stage_plugin_config.retrieval,
    )
    eligible = [entry for entry in bank if float(entry["event_score"]) >= effective]
    retrieval_result = _retrieve_event_neighbors(
        query=query,
        bank=eligible,
        retrieval_cfg=combo_loaded.config.stage_plugin_config.retrieval,
        effective_event_threshold=effective,
    )
    return {
        "error": "",
        "bank_size": len(bank),
        "candidate_count": candidate_count,
        "eligible_count": len(eligible),
        "query_event_score": float(query["event_score"]),
        "effective_event_threshold": float(effective),
        "skip_reason": retrieval_result["skip_reason"] or "",
        "retrieval_applied": bool(retrieval_result["retrieval_applied"]),
        "top_k_used": len(retrieval_result["top_neighbors"]),
        "mean_similarity": float(retrieval_result["mean_similarity"]),
        "max_similarity": float(retrieval_result["max_similarity"]),
        "neighbors": retrieval_result["top_neighbors"],
    }


def _init_combo_worker(
    loaded: LoadedConfig,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> None:
    global _WORKER_LOADED, _WORKER_TRAIN_DF, _WORKER_FUTURE_DF
    _WORKER_LOADED = loaded
    _WORKER_TRAIN_DF = train_df
    _WORKER_FUTURE_DF = future_df


def _evaluate_combo_worker(spec: ComboSpec) -> tuple[ComboSpec, dict[str, Any]]:
    if _WORKER_LOADED is None or _WORKER_TRAIN_DF is None or _WORKER_FUTURE_DF is None:
        raise RuntimeError("final_find worker state was not initialized")
    stats = evaluate_exact_combo(
        loaded=_WORKER_LOADED,
        train_df=_WORKER_TRAIN_DF,
        future_df=_WORKER_FUTURE_DF,
        input_size=spec.input_size,
        hist_exog_cols=spec.hist_exog_cols,
        upward_cols=spec.upward_cols,
        top_k=spec.top_k,
    )
    return spec, stats


def _evaluate_combo_batch_worker(
    specs: Sequence[ComboSpec],
) -> list[tuple[ComboSpec, dict[str, Any]]]:
    return [_evaluate_combo_worker(spec) for spec in specs]


def _combo_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool(row["match_found"])),
        -int(row["matched_rank"]) if row["matched_rank"] else -10**9,
        int(row["top_k_requested"]),
        float(row["best_neighbor_similarity"])
        if not pd.isna(row["best_neighbor_similarity"])
        else float("-inf"),
        float(row["query_event_score"])
        if not pd.isna(row["query_event_score"])
        else float("-inf"),
    )


def _trial_loss_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    top_k_requested: int,
) -> float:
    matched = [row for row in rows if row["match_found"]]
    missed_count = len(rows) - len(matched)
    mean_rank = (
        float(np.mean([int(row["matched_rank"]) for row in matched]))
        if matched
        else float(top_k_requested + 1)
    )
    mean_similarity = (
        float(np.mean([float(row["matched_similarity"]) for row in matched]))
        if matched
        else 0.0
    )
    # Minimize: prioritize matching more targets, then lower rank, then higher similarity.
    return (1000.0 * missed_count) + mean_rank - (0.01 * mean_similarity)


def _trial_summary_row(rows: Sequence[dict[str, Any]], *, trial_number: int) -> dict[str, Any]:
    matched = [row for row in rows if row["match_found"]]
    sample = rows[0]
    matched_rank_values = [int(row["matched_rank"]) for row in matched]
    return {
        "trial_number": trial_number,
        "top_k_requested": sample["top_k_requested"],
        "input_size": sample["input_size"],
        "hist_exog_cols": sample["hist_exog_cols"],
        "upward_tail_cols": sample["upward_tail_cols"],
        "matched_target_count": len(matched),
        "total_targets": len(rows),
        "all_targets_matched": len(matched) == len(rows),
        "best_matched_rank": min(matched_rank_values) if matched_rank_values else 0,
        "mean_matched_rank": float(np.mean(matched_rank_values)) if matched_rank_values else float("nan"),
        "mean_best_neighbor_similarity": float(
            np.nanmean([row["best_neighbor_similarity"] for row in rows])
        ),
        "mean_query_event_score": float(np.nanmean([row["query_event_score"] for row in rows])),
        "objective_loss": _trial_loss_from_rows(rows, top_k_requested=int(sample["top_k_requested"])),
    }


def _rows_for_combo(
    *,
    spec: ComboSpec,
    stats: dict[str, Any],
    targets: Sequence[TargetWindow],
    eval_slice: str,
    trial_number: int | None = None,
) -> list[dict[str, Any]]:
    common: dict[str, Any] = {
        "eval_slice": eval_slice,
        "input_size": spec.input_size,
        "hist_exog_cols": json.dumps(list(spec.hist_exog_cols)),
        "upward_tail_cols": json.dumps(list(spec.upward_cols)),
        "error": stats["error"],
        "bank_size": stats.get("bank_size", float("nan")),
        "candidate_count": stats.get("candidate_count", float("nan")),
        "eligible_count": stats.get("eligible_count", float("nan")),
        "query_event_score": stats.get("query_event_score", float("nan")),
        "effective_event_threshold": stats.get("effective_event_threshold", float("nan")),
        "skip_reason": stats.get("skip_reason", ""),
        "retrieval_applied": stats.get("retrieval_applied", False),
        "top_k_requested": spec.top_k,
        "top_k_used": stats.get("top_k_used", 0),
        "mean_similarity": stats.get("mean_similarity", float("nan")),
        "max_similarity": stats.get("max_similarity", float("nan")),
    }
    if trial_number is not None:
        common["trial_number"] = trial_number
    neighbors = stats.get("neighbors", [])
    rows: list[dict[str, Any]] = []
    for target in targets:
        row = {
            **common,
            "target_label": target.label,
            "target_candidate_end_ds": target.candidate_end_ds,
            "target_notes": target.notes or "",
        }
        row.update(_target_match_payload(target, neighbors))
        rows.append(row)
    return rows


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


def _resolve_workers(raw: str) -> int:
    text = str(raw).strip().lower()
    if text in {"", "auto"}:
        return max(1, os.cpu_count() or 1)
    workers = int(text)
    if workers <= 0:
        raise ValueError("--workers must be a positive integer or 'auto'")
    return workers


def _iter_combo_specs(
    *,
    input_sizes: Sequence[int],
    exog_grid: list[list[str]],
    upward_grid: list[list[str]],
    top_k_values: Sequence[int],
    base_dataset_cols: set[str],
) -> Iterator[ComboSpec]:
    for top_k in top_k_values:
        for input_size, exog_t, upward_t in iter_combos(
            input_sizes=input_sizes,
            exog_grid=exog_grid,
            upward_grid=upward_grid,
            base_dataset_cols=base_dataset_cols,
        ):
            yield ComboSpec(
                input_size=input_size,
                hist_exog_cols=exog_t,
                upward_cols=upward_t,
                top_k=top_k,
            )


def _iter_chunks(items: Iterable[ComboSpec], *, size: int) -> Iterator[list[ComboSpec]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    batch: list[ComboSpec] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _progress_line(*, label: str, completed: int, total: int, started_at: float) -> str:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta_seconds = (remaining / rate) if rate > 0 else float("inf")
    eta_text = "inf" if not math.isfinite(eta_seconds) else f"{eta_seconds:.1f}s"
    return (
        f"final_find[{label}] completed={completed}/{total} "
        f"rate={rate:.2f}/s eta={eta_text}"
    )


def _pool_from_grid(grid: Sequence[Sequence[str]]) -> list[str]:
    return list(dict.fromkeys(column for row in grid for column in row))


def _select_required_column(
    trial: optuna.Trial,
    *,
    prefix: str,
    allowed_columns: Sequence[str],
) -> str | None:
    allowed = [str(column) for column in allowed_columns]
    if not allowed:
        return None
    if len(allowed) == 1:
        return allowed[0]
    selector = float(trial.suggest_float(f"{prefix}__required_selector", 0.0, 1.0))
    index = min(int(selector * len(allowed)), len(allowed) - 1)
    return allowed[index]


def _suggest_non_empty_subset(
    trial: optuna.Trial,
    *,
    columns: Sequence[str],
    prefix: str,
    allowed_columns: Sequence[str] | None = None,
) -> list[str]:
    universe = [str(column) for column in columns]
    allowed = set(universe if allowed_columns is None else map(str, allowed_columns))
    if not allowed:
        return []
    if len(universe) == 1:
        column = universe[0]
        return [column] if column in allowed else []
    include_flags = {
        column: bool(trial.suggest_categorical(f"{prefix}__{column}", [False, True]))
        for column in universe
    }
    selected = [column for column in universe if column in allowed and include_flags[column]]
    if selected:
        return selected
    required = _select_required_column(
        trial,
        prefix=prefix,
        allowed_columns=[column for column in universe if column in allowed],
    )
    return [required] if required is not None else []


def _filter_input_sizes_for_backbone(
    input_sizes: Sequence[int],
    *,
    stage_cfg: AAForecastPluginConfig,
) -> list[int]:
    filtered = [int(size) for size in input_sizes]
    if str(stage_cfg.model).strip().lower() != "timexer":
        return filtered
    patch_len = int(stage_cfg.model_params.get("patch_len", 1))
    if patch_len <= 0:
        raise ValueError(f"aa_forecast timexer patch_len must be positive, got {patch_len}")
    return [size for size in filtered if size % patch_len == 0]


def run_final_find_optuna(
    *,
    loaded: LoadedConfig,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    targets: Sequence[TargetWindow],
    input_sizes: Sequence[int],
    exog_grid: list[list[str]],
    upward_grid: list[list[str]],
    top_k_values: Sequence[int],
    eval_slice: str,
    n_trials: int,
    seed: int,
    workers: int,
    storage_path: Path | None = None,
    study_name: str | None = None,
    resume: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    exog_pool = _pool_from_grid(exog_grid)
    upward_pool = _pool_from_grid(upward_grid)
    upward_pool_set = set(upward_pool)
    if not upward_pool:
        raise ValueError("optuna mode requires at least one upward-grid candidate column")
    if not set(upward_pool).issubset(set(exog_pool)):
        raise ValueError("optuna mode requires upward-grid pool to be a subset of exog-grid pool")

    sampler = build_optuna_sampler(seed)
    storage = (
        f"sqlite:///{storage_path.resolve()}"
        if storage_path is not None
        else None
    )
    effective_study_name = study_name or "final-find-optuna"
    if resume and storage is None:
        raise ValueError("--resume requires --optuna-storage-path")
    if workers > 1 and storage is None:
        raise ValueError("optuna mode with --workers > 1 requires --optuna-storage-path")
    if storage_path is not None:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=effective_study_name,
        storage=storage,
        load_if_exists=bool(resume or storage is not None),
    )
    started_at = time.monotonic()
    launched = 0
    completed = 0
    progress_interval = max(1, workers)

    def _suggest_spec(trial: optuna.Trial) -> ComboSpec:
        input_size = int(trial.suggest_categorical("input_size", list(input_sizes)))
        top_k = int(trial.suggest_categorical("top_k", list(top_k_values)))
        selected_exog = _suggest_non_empty_subset(
            trial, columns=exog_pool, prefix="use_exog"
        )
        if not selected_exog:
            raise optuna.TrialPruned("no exog columns selected")
        available_upward = [column for column in upward_pool if column in selected_exog]
        if not available_upward:
            raise optuna.TrialPruned("no upward candidates remain after exog selection")
        selected_upward = _suggest_non_empty_subset(
            trial,
            columns=upward_pool,
            allowed_columns=available_upward,
            prefix="use_upward",
        )
        if not selected_upward:
            raise optuna.TrialPruned("no upward columns selected")
        return ComboSpec(
            input_size=input_size,
            hist_exog_cols=tuple(selected_exog),
            upward_cols=tuple(selected_upward),
            top_k=top_k,
        )

    if workers == 1:
        for _ in range(n_trials):
            trial = study.ask()
            try:
                spec = _suggest_spec(trial)
                stats = evaluate_exact_combo(
                    loaded=loaded,
                    train_df=train_df,
                    future_df=future_df,
                    input_size=spec.input_size,
                    hist_exog_cols=spec.hist_exog_cols,
                    upward_cols=spec.upward_cols,
                    top_k=spec.top_k,
                )
            except optuna.TrialPruned as exc:
                trial.set_user_attr("prune_reason", str(exc))
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                launched += 1
                completed += 1
                if completed == n_trials or completed % progress_interval == 0:
                    print(
                        _progress_line(
                            label="optuna",
                            completed=completed,
                            total=n_trials,
                            started_at=started_at,
                        ),
                        file=sys.stderr,
                    )
                continue
            except ValueError as exc:
                trial.set_user_attr("prune_reason", str(exc))
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                launched += 1
                completed += 1
                if completed == n_trials or completed % progress_interval == 0:
                    print(
                        _progress_line(
                            label="optuna",
                            completed=completed,
                            total=n_trials,
                            started_at=started_at,
                        ),
                        file=sys.stderr,
                    )
                continue
            rows = _rows_for_combo(
                spec=spec,
                stats=stats,
                targets=targets,
                eval_slice=eval_slice,
                trial_number=trial.number,
            )
            if stats["error"]:
                trial.set_user_attr("prune_reason", stats["error"])
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                loss = _trial_loss_from_rows(rows, top_k_requested=spec.top_k)
                trial.set_user_attr(
                    "matched_target_count",
                    sum(1 for row in rows if row["match_found"]),
                )
                trial.set_user_attr("hist_exog_cols", list(spec.hist_exog_cols))
                trial.set_user_attr("upward_tail_cols", list(spec.upward_cols))
                trial.set_user_attr("target_rows", _json_ready(rows))
                trial.set_user_attr(
                    "summary_row",
                    _json_ready(_trial_summary_row(rows, trial_number=trial.number)),
                )
                study.tell(trial, loss)
            launched += 1
            completed += 1
            if completed == n_trials or completed % progress_interval == 0:
                print(
                    _progress_line(
                        label="optuna",
                        completed=completed,
                        total=n_trials,
                        started_at=started_at,
                    ),
                    file=sys.stderr,
                )
    else:
        pending: dict[Future[tuple[ComboSpec, dict[str, Any]]], tuple[optuna.Trial, ComboSpec]] = {}
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_combo_worker,
            initargs=(loaded, train_df, future_df),
        ) as executor:
            while launched < n_trials or pending:
                while launched < n_trials and len(pending) < workers:
                    trial = study.ask()
                    try:
                        spec = _suggest_spec(trial)
                    except optuna.TrialPruned as exc:
                        trial.set_user_attr("prune_reason", str(exc))
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        launched += 1
                        completed += 1
                        if completed == n_trials or completed % progress_interval == 0:
                            print(
                                _progress_line(
                                    label="optuna",
                                    completed=completed,
                                    total=n_trials,
                                    started_at=started_at,
                                ),
                                file=sys.stderr,
                            )
                        continue
                    pending[executor.submit(_evaluate_combo_worker, spec)] = (trial, spec)
                    launched += 1
                if not pending:
                    continue
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    trial, spec = pending.pop(future)
                    try:
                        _completed_spec, stats = future.result()
                    except ValueError as exc:
                        trial.set_user_attr("prune_reason", str(exc))
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    else:
                        rows = _rows_for_combo(
                            spec=spec,
                            stats=stats,
                            targets=targets,
                            eval_slice=eval_slice,
                            trial_number=trial.number,
                        )
                        if stats["error"]:
                            trial.set_user_attr("prune_reason", stats["error"])
                            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        else:
                            loss = _trial_loss_from_rows(rows, top_k_requested=spec.top_k)
                            trial.set_user_attr(
                                "matched_target_count",
                                sum(1 for row in rows if row["match_found"]),
                            )
                            trial.set_user_attr("hist_exog_cols", list(spec.hist_exog_cols))
                            trial.set_user_attr("upward_tail_cols", list(spec.upward_cols))
                            trial.set_user_attr("target_rows", _json_ready(rows))
                            trial.set_user_attr(
                                "summary_row",
                                _json_ready(_trial_summary_row(rows, trial_number=trial.number)),
                            )
                            study.tell(trial, loss)
                    completed += 1
                    if completed == n_trials or completed % progress_interval == 0:
                        print(
                            _progress_line(
                                label="optuna",
                                completed=completed,
                                total=n_trials,
                                started_at=started_at,
                            ),
                            file=sys.stderr,
                        )
    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    trial_target_rows: list[dict[str, Any]] = []
    trial_summary_rows: list[dict[str, Any]] = []
    for trial in completed_trials:
        target_rows = trial.user_attrs.get("target_rows")
        if isinstance(target_rows, list):
            trial_target_rows.extend(target_rows)
        summary_row = trial.user_attrs.get("summary_row")
        if isinstance(summary_row, dict):
            trial_summary_rows.append(summary_row)
    trial_target_rows.sort(key=_combo_sort_key, reverse=True)
    trial_summary_rows.sort(
        key=lambda row: (
            int(row["matched_target_count"]),
            int(row["top_k_requested"]),
            -int(row["best_matched_rank"]) if row["best_matched_rank"] else -10**9,
            float(row["mean_best_neighbor_similarity"]),
            -float(row["objective_loss"]),
        ),
        reverse=True,
    )
    study_summary = {
        "study_name": effective_study_name,
        "storage": storage,
        "best_trial_number": study.best_trial.number if completed_trials else None,
        "best_value": float(study.best_value) if completed_trials else None,
        "n_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": sum(
            1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED
        ),
    }
    return trial_target_rows, trial_summary_rows, study_summary


def run_final_find(
    *,
    loaded: LoadedConfig,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    targets: Sequence[TargetWindow],
    input_sizes: Sequence[int],
    exog_grid: list[list[str]],
    upward_grid: list[list[str]],
    top_k_values: Sequence[int],
    eval_slice: str,
    workers: int,
    chunksize: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_cols = set(train_df.columns)
    combo_rows: list[dict[str, Any]] = []
    combo_specs = _iter_combo_specs(
        input_sizes=input_sizes,
        exog_grid=exog_grid,
        upward_grid=upward_grid,
        top_k_values=top_k_values,
        base_dataset_cols=base_cols,
    )
    total_specs = sum(
        1
        for _ in _iter_combo_specs(
            input_sizes=input_sizes,
            exog_grid=exog_grid,
            upward_grid=upward_grid,
            top_k_values=top_k_values,
            base_dataset_cols=base_cols,
        )
    )
    started_at = time.monotonic()
    progress_interval = max(1, workers)
    completed = 0

    if workers == 1:
        for spec in combo_specs:
            stats = evaluate_exact_combo(
                loaded=loaded,
                train_df=train_df,
                future_df=future_df,
                input_size=spec.input_size,
                hist_exog_cols=spec.hist_exog_cols,
                upward_cols=spec.upward_cols,
                top_k=spec.top_k,
            )
            combo_rows.extend(
                _rows_for_combo(
                    spec=spec,
                    stats=stats,
                    targets=targets,
                    eval_slice=eval_slice,
                )
            )
            completed += 1
            if completed == total_specs or completed % progress_interval == 0:
                print(
                    _progress_line(
                        label="exhaustive",
                        completed=completed,
                        total=total_specs,
                        started_at=started_at,
                    ),
                    file=sys.stderr,
                )
    else:
        chunked_specs = _iter_chunks(combo_specs, size=chunksize)
        pending: dict[
            Future[list[tuple[ComboSpec, dict[str, Any]]]],
            int,
        ] = {}
        max_pending = max(workers * 2, 1)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_combo_worker,
            initargs=(loaded, train_df, future_df),
        ) as executor:
            while True:
                while len(pending) < max_pending:
                    try:
                        chunk = next(chunked_specs)
                    except StopIteration:
                        break
                    pending[executor.submit(_evaluate_combo_batch_worker, chunk)] = len(chunk)
                if not pending:
                    break
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    chunk_size = pending.pop(future)
                    for spec, stats in future.result():
                        combo_rows.extend(
                            _rows_for_combo(
                                spec=spec,
                                stats=stats,
                                targets=targets,
                                eval_slice=eval_slice,
                            )
                        )
                    completed += chunk_size
                    if completed == total_specs or completed % progress_interval == 0:
                        print(
                            _progress_line(
                                label="exhaustive",
                                completed=completed,
                                total=total_specs,
                                started_at=started_at,
                            ),
                            file=sys.stderr,
                        )

    combo_rows.sort(key=_combo_sort_key, reverse=True)

    global_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in combo_rows:
        key = (
            row["top_k_requested"],
            row["input_size"],
            row["hist_exog_cols"],
            row["upward_tail_cols"],
        )
        grouped.setdefault(key, []).append(row)
    for (_top_k, _input_size, _hist_exog, _upward), rows in grouped.items():
        matches = [row for row in rows if row["match_found"]]
        matched_rank_values = [int(row["matched_rank"]) for row in matches]
        global_rows.append(
            {
                "top_k_requested": rows[0]["top_k_requested"],
                "input_size": rows[0]["input_size"],
                "hist_exog_cols": rows[0]["hist_exog_cols"],
                "upward_tail_cols": rows[0]["upward_tail_cols"],
                "matched_target_count": len(matches),
                "total_targets": len(rows),
                "all_targets_matched": len(matches) == len(rows),
                "best_matched_rank": min(matched_rank_values) if matched_rank_values else 0,
                "mean_best_neighbor_similarity": float(
                    np.nanmean([row["best_neighbor_similarity"] for row in rows])
                ),
                "mean_query_event_score": float(
                    np.nanmean([row["query_event_score"] for row in rows])
                ),
            }
        )
    global_rows.sort(
        key=lambda row: (
            int(row["matched_target_count"]),
            int(row["top_k_requested"]),
            -int(row["best_matched_rank"]) if row["best_matched_rank"] else -10**9,
            float(row["mean_best_neighbor_similarity"]),
            float(row["mean_query_event_score"]),
        ),
        reverse=True,
    )
    return combo_rows, global_rows


def build_summary_payload(
    *,
    combo_rows: Sequence[dict[str, Any]],
    global_rows: Sequence[dict[str, Any]],
    targets: Sequence[TargetWindow],
    config_path: Path,
    dataset_path: Path,
    top_k_values: Sequence[int],
    eval_slice: str,
    search_mode: str,
    study_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_targets: list[dict[str, Any]] = []
    for target in targets:
        target_rows = [row for row in combo_rows if row["target_label"] == target.label]
        successful = [row for row in target_rows if row["match_found"]]
        best = successful[0] if successful else None
        summary_targets.append(
            {
                "label": target.label,
                "candidate_end_ds": target.candidate_end_ds,
                "normalized_candidate_end_ds": target.normalized_candidate_end_ds,
                "notes": target.notes,
                "match_count": len(successful),
                "best_match": None
                if best is None
                else {
                    "input_size": best["input_size"],
                    "hist_exog_cols": json.loads(best["hist_exog_cols"]),
                    "upward_tail_cols": json.loads(best["upward_tail_cols"]),
                    "matched_rank": best["matched_rank"],
                    "matched_candidate_end_ds": best["matched_candidate_end_ds"],
                    "matched_similarity": best["matched_similarity"],
                },
            }
        )
    return {
        "config": str(config_path),
        "data_path": str(dataset_path),
        "eval_slice": eval_slice,
        "search_mode": search_mode,
        "top_k_values": list(top_k_values),
        "targets": summary_targets,
        "global_best_configs": global_rows[:10],
        "study_summary": study_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override dataset.path from --config for this run only.",
    )
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument(
        "--search-mode",
        choices=("exhaustive", "optuna"),
        default="exhaustive",
        help="exhaustive: evaluate every valid combo, optuna: use TPESampler over the same search knobs.",
    )
    parser.add_argument("--input-sizes", type=str, default="")
    parser.add_argument("--min-input-size", type=int, default=1)
    parser.add_argument(
        "--max-input-size",
        type=int,
        default=None,
        help="If set and --input-sizes is omitted, expand the inclusive range min..max.",
    )
    parser.add_argument(
        "--eval-slice",
        choices=("last_cv_fold", "max_tail"),
        default="last_cv_fold",
    )
    parser.add_argument(
        "--exog-grid",
        type=str,
        default=None,
        help="JSON grid path or inline comma-separated column pool (expanded to all non-empty subsets).",
    )
    parser.add_argument(
        "--upward-grid",
        type=str,
        default=None,
        help="JSON grid path or inline comma-separated column pool (expanded to all non-empty subsets).",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="1",
        help="Comma-separated top_k values to evaluate (default: 1).",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=250000,
        help="Fail fast if total evaluated combinations would exceed this value (0 disables the guard).",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--workers",
        type=str,
        default="auto",
        help="Worker count for combo evaluation ('auto' uses all logical CPUs).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Number of combos per worker task in exhaustive mode.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Force single-process execution for debugging/repro comparisons.",
    )
    parser.add_argument(
        "--optuna-n-trials",
        type=int,
        default=200,
        help="Number of Optuna trials when --search-mode optuna.",
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=None,
        help="Override Optuna seed (defaults to runtime.random_seed via repo helper).",
    )
    parser.add_argument(
        "--optuna-storage-path",
        type=Path,
        default=None,
        help="Persist Optuna study to this sqlite file for resume/continue.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="final-find-optuna",
        help="Optuna study name used with --optuna-storage-path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing Optuna study from --optuna-storage-path.",
    )
    args = parser.parse_args(argv)

    loaded = load_app_config(REPO_ROOT, config_path=args.config.resolve())
    stage_cfg = loaded.config.stage_plugin_config
    if not isinstance(stage_cfg, AAForecastPluginConfig):
        raise ValueError("final_find.py requires an aa_forecast experiment config")
    if stage_cfg.retrieval.insample_y_included:
        raise ValueError(
            "final_find.py requires aa_forecast.retrieval.insample_y_included=false"
        )
    targets = _load_targets_json(args.targets_json)

    dataset_path = (
        args.data_path.resolve()
        if args.data_path is not None
        else Path(str(loaded.config.dataset.path)).resolve()
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset path not found: {dataset_path}")
    source_df = pd.read_csv(dataset_path)
    dt_col = loaded.config.dataset.dt_col
    source_df[dt_col] = pd.to_datetime(source_df[dt_col])
    source_df = source_df.sort_values(dt_col).reset_index(drop=True)
    train_df, future_df = resolve_train_future_frames(
        loaded, source_df, eval_slice=args.eval_slice
    )

    if args.input_sizes.strip():
        input_sizes = _parse_int_list(args.input_sizes)
    else:
        if args.max_input_size is not None:
            if args.max_input_size < args.min_input_size:
                raise ValueError("--max-input-size must be >= --min-input-size")
            input_sizes = list(range(args.min_input_size, args.max_input_size + 1))
        else:
            input_sizes = [int(loaded.config.training.input_size)]
    before_filter = list(input_sizes)
    input_sizes = [size for size in input_sizes if size >= args.min_input_size]
    filtered_for_backbone = _filter_input_sizes_for_backbone(
        input_sizes, stage_cfg=stage_cfg
    )
    if filtered_for_backbone != input_sizes:
        print(
            "final_find: filtered input_sizes for backbone constraints "
            f"({stage_cfg.model}): {filtered_for_backbone}",
            file=sys.stderr,
        )
    input_sizes = filtered_for_backbone
    if not input_sizes:
        print(
            f"error: no input_sizes remain after filtering "
            f"(min-input-size/backbone constraints; candidates were {before_filter})",
            file=sys.stderr,
        )
        return 1

    exog_grid = _load_grid_or_inline_pool(args.exog_grid)
    if exog_grid is None:
        exog_grid = [list(loaded.config.dataset.hist_exog_cols)]
    upward_grid = _load_grid_or_inline_pool(args.upward_grid)
    if upward_grid is None:
        upward_grid = [list(stage_cfg.star_anomaly_tails.get("upward", ()))]
    _validate_grid_columns_exist(
        grid_name="--exog-grid",
        grid=exog_grid,
        dataset_columns=set(source_df.columns),
    )
    _validate_grid_columns_exist(
        grid_name="--upward-grid",
        grid=upward_grid,
        dataset_columns=set(source_df.columns),
    )
    top_k_values = _parse_int_list(args.top_k_values)
    if not top_k_values:
        raise ValueError("--top-k-values must contain at least one integer")
    if any(value <= 0 for value in top_k_values):
        raise ValueError("--top-k-values must contain only positive integers")
    workers = 1 if args.serial else _resolve_workers(args.workers)
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    combo_count_per_topk = sum(
        1
        for _ in iter_combos(
            input_sizes=input_sizes,
            exog_grid=exog_grid,
            upward_grid=upward_grid,
            base_dataset_cols=set(source_df.columns),
        )
    )
    combo_count = combo_count_per_topk * len(top_k_values)
    skipped_incompatible = count_incompatible_exog_upward_pairs(
        input_sizes=input_sizes,
        exog_grid=exog_grid,
        upward_grid=upward_grid,
        base_dataset_cols=set(source_df.columns),
    )
    print(
        f"eval_slice={args.eval_slice} train_rows={len(train_df)} test_rows={len(future_df)} "
        f"targets={len(targets)} input_sizes={input_sizes} top_k_values={top_k_values} "
        f"combos={combo_count} workers={workers} chunksize={args.chunksize}"
    )
    if skipped_incompatible:
        print(
            f"final_find: skipped {skipped_incompatible} incompatible "
            "(input_size, hist_exog, upward) combo(s)",
            file=sys.stderr,
        )
    if args.search_mode == "exhaustive" and args.max_combos and combo_count > args.max_combos:
        raise ValueError(
            f"total combos {combo_count} exceed --max-combos {args.max_combos}. "
            "Narrow the grids/ranges or pass --max-combos 0 to disable the guard."
        )
    if args.dry_run:
        return 0

    study_summary = None
    if args.search_mode == "optuna":
        combo_rows, global_rows, study_summary = run_final_find_optuna(
            loaded=loaded,
            train_df=train_df,
            future_df=future_df,
            targets=targets,
            input_sizes=input_sizes,
            exog_grid=exog_grid,
            upward_grid=upward_grid,
            top_k_values=top_k_values,
            eval_slice=args.eval_slice,
            n_trials=int(args.optuna_n_trials),
            seed=(
                int(args.optuna_seed)
                if args.optuna_seed is not None
                else optuna_seed(loaded.config.runtime.random_seed)
            ),
            workers=workers,
            storage_path=args.optuna_storage_path,
            study_name=args.study_name,
            resume=bool(args.resume),
        )
    else:
        combo_rows, global_rows = run_final_find(
            loaded=loaded,
            train_df=train_df,
            future_df=future_df,
            targets=targets,
            input_sizes=input_sizes,
            exog_grid=exog_grid,
            upward_grid=upward_grid,
            top_k_values=top_k_values,
            eval_slice=args.eval_slice,
            workers=workers,
            chunksize=int(args.chunksize),
        )

    if args.output_csv:
        out = args.output_csv.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if not combo_rows:
            raise RuntimeError("no rows to write")
        fieldnames = list(combo_rows[0].keys())
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combo_rows)
        print(str(out))

    if args.summary_json:
        summary = build_summary_payload(
            combo_rows=combo_rows,
            global_rows=global_rows,
            targets=targets,
            config_path=args.config.resolve(),
            dataset_path=dataset_path,
            top_k_values=top_k_values,
            eval_slice=args.eval_slice,
            search_mode=args.search_mode,
            study_summary=study_summary,
        )
        out_json = args.summary_json.resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(str(out_json))

    if not args.output_csv and not args.summary_json:
        preview = combo_rows[:10]
        print(json.dumps(preview, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
