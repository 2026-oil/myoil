#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from app_config import JobConfig, LoadedConfig, load_app_config
from runtime_support.runner import (
    _build_tscv_splits,
    _compute_metrics,
    _fit_and_predict_fold,
    _resolve_freq,
)


KEYWORD_FILTERS = ("precompute", "cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one AAForecast fold for the STAR precompute refactor using "
            "the approved command shape."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config path relative to repo root or absolute path.",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help="Optional shared settings path relative to repo root or absolute path.",
    )
    parser.add_argument(
        "--phase",
        default="fold",
        choices=("fold",),
        help="Benchmark phase. Only 'fold' is supported for this entrypoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. The benchmark JSON is always printed to stdout.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Zero-based fold index to benchmark (default: 0).",
    )
    parser.add_argument(
        "--job-model",
        default=None,
        help="Optional explicit model name when the config contains multiple jobs.",
    )
    parser.add_argument(
        "--disable-precompute",
        action="store_true",
        help="Disable the STAR precompute path for baseline benchmarking.",
    )
    return parser.parse_args()


def _resolve_repo_path(raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((ROOT / candidate).resolve())


def _resolve_dataset_path(loaded: LoadedConfig) -> Path:
    dataset_path = Path(loaded.config.dataset.path)
    if dataset_path.is_absolute():
        return dataset_path.resolve()
    return (ROOT / dataset_path).resolve()


def _maxrss_kb() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value // 1024
    return value


def _json_number(value: Any) -> float | int | None:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    return None


def _json_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    return {key: _json_number(value) for key, value in metrics.items()}


def _select_job(loaded: LoadedConfig, explicit_model: str | None) -> JobConfig:
    jobs = list(loaded.config.jobs)
    if explicit_model is not None:
        matches = [job for job in jobs if job.model == explicit_model]
        if not matches:
            available = ", ".join(job.model for job in jobs)
            raise ValueError(
                f"Requested --job-model={explicit_model!r} was not found. Available jobs: {available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Requested --job-model={explicit_model!r} resolved to multiple jobs; benchmark requires one job"
            )
        return matches[0]

    aa_jobs = [job for job in jobs if job.model == "AAForecast"]
    if len(aa_jobs) == 1:
        return aa_jobs[0]
    if len(jobs) == 1 and jobs[0].model == "AAForecast":
        return jobs[0]
    if not aa_jobs:
        available = ", ".join(job.model for job in jobs)
        raise ValueError(
            "Benchmark requires an AAForecast job, but none were found. "
            f"Available jobs: {available}"
        )
    raise ValueError(
        "Benchmark config contains multiple AAForecast jobs; pass --job-model to disambiguate"
    )


def _load_source_frame(loaded: LoadedConfig) -> pd.DataFrame:
    dataset_path = _resolve_dataset_path(loaded)
    frame = pd.read_csv(dataset_path)
    dt_col = loaded.config.dataset.dt_col
    if dt_col not in frame.columns:
        raise ValueError(
            f"Configured dt column {dt_col!r} is missing from dataset {dataset_path}"
        )
    return frame.sort_values(dt_col).reset_index(drop=True)


def _summarize_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        summary: dict[str, Any] = {"type": "dict", "len": len(value)}
        scalar_items = {}
        for key in sorted(value.keys(), key=str):
            item = value[key]
            scalar = _json_number(item)
            if scalar is not None:
                scalar_items[str(key)] = scalar
            if len(scalar_items) >= 5:
                break
        if scalar_items:
            summary["scalar_items"] = scalar_items
        return summary
    if isinstance(value, (list, tuple, set)):
        return {"type": type(value).__name__, "len": len(value)}
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "size": int(value.size),
            "dtype": str(value.dtype),
        }
    scalar = _json_number(value)
    if scalar is not None:
        return {"type": type(value).__name__, "value": scalar}
    return None


def _collect_cache_stats(fitted_model: Any) -> dict[str, Any] | None:
    model = getattr(fitted_model, "models", [None])[0]
    if model is None:
        return None

    stats: dict[str, Any] = {}
    star = getattr(model, "star", None)
    if star is not None and hasattr(star, "_trend_cache"):
        trend_cache = getattr(star, "_trend_cache")
        stats["star_trend_cache_entries"] = len(trend_cache)
        stats["star_trend_cache_limit"] = int(
            getattr(star, "_trend_cache_limit", len(trend_cache))
        )

    candidate_attrs: dict[str, dict[str, Any]] = {}
    for owner_name, owner in (("model", model), ("star", star)):
        if owner is None:
            continue
        for attr_name, attr_value in vars(owner).items():
            lowered = attr_name.lower()
            if not any(keyword in lowered for keyword in KEYWORD_FILTERS):
                continue
            summary = _summarize_value(attr_value)
            if summary is None:
                continue
            candidate_attrs[f"{owner_name}.{attr_name}"] = summary
    if candidate_attrs:
        stats["candidate_attributes"] = candidate_attrs

    return stats or None


def _artifact_stats(run_root: Path) -> dict[str, Any] | None:
    if not run_root.exists():
        return None
    files = sorted(
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file()
    )
    if not files:
        return None
    return {
        "file_count": len(files),
        "sample_files": files[:10],
    }


def run_fold_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["NEURALFORECAST_AA_STAR_PRECOMPUTE"] = (
        "0" if args.disable_precompute else "1"
    )
    loaded = load_app_config(
        ROOT,
        config_path=args.config,
        shared_settings_path=args.settings,
    )
    job = _select_job(loaded, args.job_model)
    source_df = _load_source_frame(loaded)
    freq = _resolve_freq(loaded, source_df)
    splits = _build_tscv_splits(len(source_df), loaded.config.cv)
    if args.fold_index < 0 or args.fold_index >= len(splits):
        raise ValueError(
            f"--fold-index must be between 0 and {len(splits) - 1}, got {args.fold_index}"
        )
    train_idx, test_idx = splits[args.fold_index]

    gc.collect()
    rss_before_kb = _maxrss_kb()
    started_at = datetime.now(UTC).isoformat()
    with TemporaryDirectory(prefix="aaforecast-star-bench-") as tmp_dir:
        run_root = Path(tmp_dir)
        started = time.perf_counter()
        target_predictions, target_actuals, train_end_ds, train_df, fitted = _fit_and_predict_fold(
            loaded,
            job,
            run_root=run_root,
            source_df=source_df,
            freq=freq,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        fold_wall_seconds = time.perf_counter() - started
        rss_after_kb = _maxrss_kb()
        cache_stats = _collect_cache_stats(fitted)
        artifact_stats = _artifact_stats(run_root)

    metrics = _compute_metrics(target_actuals, target_predictions[job.model])
    payload: dict[str, Any] = {
        "benchmark": "aaforecast_star_precompute",
        "phase": args.phase,
        "config_path": _resolve_repo_path(args.config),
        "settings_path": _resolve_repo_path(args.settings),
        "job_model": job.model,
        "job_requested_mode": job.requested_mode,
        "job_validated_mode": job.validated_mode,
        "job_param_keys": sorted(job.params.keys()),
        "dataset_path": str(_resolve_dataset_path(loaded)),
        "fold_index": args.fold_index,
        "fold_count": len(splits),
        "fold_wall_seconds": fold_wall_seconds,
        "star_precompute_enabled": not args.disable_precompute,
        "rss_kb_before": rss_before_kb,
        "rss_kb_after": rss_after_kb,
        "rss_kb_delta": max(0, rss_after_kb - rss_before_kb),
        "train_rows": len(train_idx),
        "fit_rows": len(train_df),
        "test_rows": len(test_idx),
        "prediction_rows": int(len(target_predictions)),
        "train_end_ds": pd.Timestamp(train_end_ds).isoformat(),
        "target_column": loaded.config.dataset.target_col,
        "metrics": _json_metrics(metrics),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    if cache_stats is not None:
        payload["cache_stats"] = cache_stats
    if artifact_stats is not None:
        payload["artifact_stats"] = artifact_stats
    return payload


def main() -> int:
    args = parse_args()
    if args.phase != "fold":
        raise ValueError("Only --phase fold is supported")
    payload = run_fold_benchmark(args)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
