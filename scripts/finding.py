"""Grid-search window size and exog / STAR-tail combos for last-fold retrieval diagnostics.

Uses the same TSCV slice as ``runtime_support.runner._build_tscv_splits`` and the same
diff transforms as ``_build_fold_diff_context`` / ``_transform_training_frame``.

**Signature backend:** ``plugins.retrieval`` STAR + ``compute_star_signature`` (same path as
standalone retrieval memory bank). This is **not** identical to AAForecast Informer
``_compute_star_outputs`` signatures; treat results as a fast proxy for ranking combos.

Examples:
  # Standalone retrieval route (no AAForecast / no backbone YAML)
  uv run python scripts/finding.py \\
    --config tests/fixtures/retrieval_runtime_smoke_linked.yaml \\
    --input-sizes 3,4 \\
    --min-input-size 1 \\
    --output-csv runs/_finding_retrieval_only.csv

  # Long windows (e.g. input_size>=12): use max train tail, not TSCV last fold
  uv run python scripts/finding.py \\
    --config yaml/experiment/feature_set_aaforecast/baseline-ret.yaml \\
    --eval-slice max_tail \\
    --min-input-size 12 \\
    --input-sizes 12,16,24,32,48,64 \\
    --exog-grid tests/fixtures/finding_exog_grid_baseline_ret_example.json \\
    --upward-grid tests/fixtures/finding_upward_grid_baseline_ret_example.json \\

  GPRD_THREAT / BS_Core_Index_A / BS_Core_Index_C / GPRD / GPRD_ACT 탐색
  (BS 있는 행 vs 없는 행을 같은 ``finding_exog_grid_gprd_core5.json``에 넣을 수 있음.
  ``upward`` 항목이 어떤 ``hist_exog`` 행의 부분집합이 아니면 해당 (exog, upward) 쌍만
  건너뛰고, 건너뛴 횟수는 stderr에 한 줄로 표시됨):

  uv run python scripts/finding.py \\
    --config yaml/experiment/feature_set_aaforecast/baseline-ret.yaml \\
    --eval-slice max_tail \\
    --min-input-size 12 \\
    --input-sizes 12,16,24,32 \\
    --exog-grid tests/fixtures/finding_exog_grid_gprd_core5.json \\
    --upward-grid tests/fixtures/finding_upward_grid_gprd_core5.json \\
    --output-csv runs/_finding_gprd_core5.csv

  # Optional: AAForecast experiment instead
  uv run python scripts/finding.py \\
    --config yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret.yaml \\
    --input-sizes 32,48,64 \\
    --output-csv runs/_finding_last_fold.csv

  uv run python scripts/finding.py --config path/to/exp.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_config import (
    LoadedConfig,
    _coerce_bool,
    _load_document,
    _unknown_keys,
    load_app_config,
)
from plugins.aa_forecast.config import (
    AAForecastPluginConfig,
    AAForecastStageLoadedConfig,
)
from plugins.retrieval.config import RetrievalPluginConfig
from plugins.retrieval import config as retrieval_config_mod
from plugins.retrieval.plugin import RetrievalStagePlugin
from plugins.retrieval.runtime import (
    _build_memory_bank,
    _build_query,
    _build_star_extractor,
    _effective_event_threshold,
    _resolve_tail_modes,
    _retrieve_neighbors,
)
from runtime_support.runner import (
    _build_fold_diff_context,
    _build_tscv_splits,
    _transform_training_frame,
)

SIGNATURE_BACKEND = "retrieval_plugin_star"


def resolve_train_future_frames(
    loaded: LoadedConfig,
    source_df: pd.DataFrame,
    *,
    eval_slice: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``last_cv_fold``: runner TSCV last window. ``max_tail``: all but last ``horizon`` rows as train."""
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


def probe_transformed_row_count(
    loaded: LoadedConfig,
    train_df: pd.DataFrame,
    hist_exog_cols: tuple[str, ...],
) -> int:
    """Rows available after diff transform for a given hist_exog selection."""
    patched = replace(
        loaded,
        config=replace(
            loaded.config,
            dataset=replace(loaded.config.dataset, hist_exog_cols=hist_exog_cols),
        ),
    )
    diff_context = _build_fold_diff_context(patched, train_df)
    transformed = _transform_training_frame(train_df, diff_context)
    return len(transformed)


def load_retrieval_plugin_config_from_path(path: Path) -> Any:
    suffix = path.suffix.lower()
    doc_type = "yaml" if suffix in {".yaml", ".yml"} else "toml"
    detail_payload = _load_document(path.resolve(), doc_type)
    return retrieval_config_mod.normalize_retrieval_detail_payload(
        detail_payload,
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
    )


def load_retrieval_plugin_config(loaded: LoadedConfig) -> Any:
    """Return merged ``RetrievalPluginConfig`` from standalone retrieval or AAForecast link."""
    stage = loaded.config.stage_plugin_config
    if isinstance(stage, RetrievalPluginConfig):
        if not stage.enabled:
            raise ValueError(
                "finding.py requires retrieval.enabled=true when using a retrieval-only --config"
            )
        return stage
    if isinstance(stage, AAForecastPluginConfig):
        sl = loaded.stage_plugin_loaded
        if not isinstance(sl, AAForecastStageLoadedConfig):
            raise ValueError("loaded.stage_plugin_loaded must be AAForecastStageLoadedConfig")
        if not stage.retrieval.enabled or not stage.retrieval.config_path:
            raise ValueError("AAForecast retrieval must be enabled with config_path")
        cfg = RetrievalStagePlugin().load_stage(
            REPO_ROOT,
            source_path=sl.source_path,
            source_type=sl.source_type,
            config=stage.retrieval,
            search_space_contract=None,
        )
        if cfg is None:
            raise ValueError("failed to load retrieval detail YAML")
        return cfg
    raise ValueError(
        "--config must be either (1) a retrieval-route YAML "
        "(top-level ``retrieval:`` + dataset/cv/...) or (2) an AAForecast experiment with retrieval."
    )


def _parse_int_list(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_json_grid(path: Path) -> list[list[str]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"grid file not found: {resolved}\n"
            "Use an existing JSON path (array of string arrays). Example copies live under "
            "tests/fixtures/finding_exog_grid_baseline_ret_example.json and "
            "tests/fixtures/finding_upward_grid_baseline_ret_example.json"
        )
    text = resolved.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    out: list[list[str]] = []
    for row in data:
        if not isinstance(row, list):
            raise ValueError(f"{path}: each entry must be a list of column names")
        out.append([str(x) for x in row])
    return out


def _default_exog_grid(base_cols: tuple[str, ...]) -> list[list[str]]:
    return [list(base_cols)]


def _default_upward_grid(base_upward: tuple[str, ...]) -> list[list[str]]:
    if not base_upward:
        return [[]]
    return [list(base_upward)]


def _actual_horizon_cum_ret(
    *,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target_col: str,
    horizon: int,
) -> float:
    anchor = float(train_df[target_col].iloc[-1])
    last_y = float(future_df[target_col].iloc[horizon - 1])
    scale = max(abs(anchor), 1e-8)
    return (last_y - anchor) / scale


def _neighbor_stats(
    neighbors: list[dict[str, Any]],
) -> tuple[float, float, float, int]:
    """Returns weighted_last_ret, mean_last_ret, mean_l2, count."""
    if not neighbors:
        return float("nan"), float("nan"), float("nan"), 0
    last_vals: list[float] = []
    l2s: list[float] = []
    w_last = 0.0
    w_sum = 0.0
    for n in neighbors:
        w = float(n.get("softmax_weight", 0.0))
        fr = np.asarray(n["future_returns"], dtype=float).reshape(-1)
        if fr.size == 0:
            continue
        last = float(fr[-1])
        last_vals.append(last)
        l2s.append(float(np.linalg.norm(fr)))
        w_last += w * last
        w_sum += w
    mean_last = float(np.mean(last_vals)) if last_vals else float("nan")
    mean_l2 = float(np.mean(l2s)) if l2s else float("nan")
    weighted_last = w_last / w_sum if w_sum > 1e-12 else float("nan")
    return weighted_last, mean_last, mean_l2, len(neighbors)


def _rank_score(
    metric: str,
    *,
    actual_cum_ret: float,
    weighted_last: float,
    mean_last: float,
    max_abs_last: float,
) -> float:
    if metric == "align_weighted_last":
        sgn = np.sign(actual_cum_ret) if abs(actual_cum_ret) > 1e-12 else 1.0
        return float(weighted_last * sgn) if not np.isnan(weighted_last) else float("nan")
    if metric == "align_mean_last":
        sgn = np.sign(actual_cum_ret) if abs(actual_cum_ret) > 1e-12 else 1.0
        return float(mean_last * sgn) if not np.isnan(mean_last) else float("nan")
    if metric == "max_abs_neighbor_last":
        return float(max_abs_last)
    raise ValueError(f"unknown metric: {metric}")


def evaluate_combo(
    *,
    loaded: LoadedConfig,
    plugin_cfg: Any,
    input_size: int,
    hist_exog_cols: tuple[str, ...],
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> dict[str, Any]:
    patched = replace(
        loaded,
        config=replace(
            loaded.config,
            dataset=replace(loaded.config.dataset, hist_exog_cols=hist_exog_cols),
        ),
    )
    target_col = patched.config.dataset.target_col
    dt_col = patched.config.dataset.dt_col
    horizon = patched.config.cv.horizon
    diff_context = _build_fold_diff_context(patched, train_df)
    transformed_train_df = _transform_training_frame(train_df, diff_context)

    retrieval_cfg = plugin_cfg.retrieval
    star = _build_star_extractor(plugin_cfg.star)
    available_hist_exog = tuple(
        col for col in hist_exog_cols if col in transformed_train_df.columns
    )
    hist_exog_tail_modes = _resolve_tail_modes(
        available_hist_exog, plugin_cfg.star.anomaly_tails
    )

    if len(transformed_train_df) < input_size:
        return {
            "error": "short_transformed_train",
            "transformed_rows": len(transformed_train_df),
            "input_size": input_size,
        }

    bank, candidate_count = _build_memory_bank(
        star=star,
        transformed_train_df=transformed_train_df,
        raw_train_df=train_df.reset_index(drop=True),
        dt_col=dt_col,
        target_col=target_col,
        hist_exog_cols=available_hist_exog,
        hist_exog_tail_modes=hist_exog_tail_modes,
        retrieval_cfg=retrieval_cfg,
        input_size=input_size,
        horizon=horizon,
    )
    query = _build_query(
        star=star,
        transformed_train_df=transformed_train_df,
        target_col=target_col,
        hist_exog_cols=available_hist_exog,
        hist_exog_tail_modes=hist_exog_tail_modes,
        input_size=input_size,
    )
    effective = _effective_event_threshold(bank=bank, retrieval_cfg=retrieval_cfg)
    eligible = [e for e in bank if float(e["event_score"]) >= effective]
    result = _retrieve_neighbors(
        query=query,
        bank=eligible,
        retrieval_cfg=retrieval_cfg,
        effective_event_threshold=effective,
    )
    neighbors = result["top_neighbors"]
    w_last, mean_last, mean_l2, n_neigh = _neighbor_stats(neighbors)
    last_abs = [
        abs(float(np.asarray(n["future_returns"], dtype=float).reshape(-1)[-1]))
        for n in neighbors
    ]
    max_abs_last = float(max(last_abs)) if last_abs else float("nan")
    actual_cum = _actual_horizon_cum_ret(
        train_df=train_df,
        future_df=future_df,
        target_col=target_col,
        horizon=horizon,
    )
    return {
        "bank_size": len(bank),
        "candidate_count": candidate_count,
        "eligible_count": len(eligible),
        "query_event_score": float(query["event_score"]),
        "effective_event_threshold": float(effective),
        "skip_reason": result["skip_reason"],
        "retrieval_applied": bool(result["retrieval_applied"]),
        "top_k_used": len(neighbors),
        "mean_similarity": float(result["mean_similarity"]),
        "max_similarity": float(result["max_similarity"]),
        "weighted_neighbor_last_ret": w_last,
        "mean_neighbor_last_ret": mean_last,
        "neighbor_future_l2_mean": mean_l2,
        "max_abs_neighbor_last_ret": max_abs_last,
        "neighbor_count": n_neigh,
        "actual_horizon_cum_ret": float(actual_cum),
    }


def iter_combos(
    *,
    input_sizes: Sequence[int],
    exog_grid: list[list[str]],
    upward_grid: list[list[str]],
    base_dataset_cols: set[str],
) -> Iterable[tuple[int, tuple[str, ...], tuple[str, ...]]]:
    """Yield (input_size, hist_exog_cols, upward_tail_cols).

    If ``upward_t`` is not a subset of ``exog_t`` for a given row, that pair is skipped
    so one ``upward_grid`` can include BS columns while some ``exog_grid`` rows omit them.
    Missing dataset columns for ``exog_t`` still raise ``ValueError``.
    """
    for input_size in input_sizes:
        for exog_cols in exog_grid:
            exog_t = tuple(exog_cols)
            if not all(c in base_dataset_cols for c in exog_t):
                raise ValueError(
                    f"hist_exog_cols {exog_t!r} must exist in dataset columns "
                    f"(missing: {set(exog_t) - base_dataset_cols})"
                )
            for upward in upward_grid:
                upward_t = tuple(upward)
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
    """Count (input_size, exog, upward) tuples skipped because upward is not subset of exog."""
    n = 0
    for _input_size in input_sizes:
        for exog_cols in exog_grid:
            exog_t = tuple(exog_cols)
            if not all(c in base_dataset_cols for c in exog_t):
                continue
            for upward in upward_grid:
                upward_t = tuple(upward)
                if not set(upward_t).issubset(set(exog_t)):
                    n += 1
    return n


def run_finding(
    *,
    loaded: LoadedConfig,
    plugin_cfg_base: Any,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    input_sizes: Sequence[int],
    exog_grid: list[list[str]],
    upward_grid: list[list[str]],
    metric: str,
    eval_slice: str,
) -> list[dict[str, Any]]:
    base_cols = set(train_df.columns)
    base_row_nan = {
        "bank_size": float("nan"),
        "candidate_count": float("nan"),
        "eligible_count": float("nan"),
        "query_event_score": float("nan"),
        "effective_event_threshold": float("nan"),
        "skip_reason": "",
        "retrieval_applied": False,
        "top_k_used": 0,
        "mean_similarity": float("nan"),
        "max_similarity": float("nan"),
        "weighted_neighbor_last_ret": float("nan"),
        "mean_neighbor_last_ret": float("nan"),
        "neighbor_future_l2_mean": float("nan"),
        "max_abs_neighbor_last_ret": float("nan"),
        "neighbor_count": 0,
        "actual_horizon_cum_ret": float("nan"),
    }
    rows: list[dict[str, Any]] = []
    for input_size, exog_t, upward_t in iter_combos(
        input_sizes=input_sizes,
        exog_grid=exog_grid,
        upward_grid=upward_grid,
        base_dataset_cols=base_cols,
    ):
        plugin_cfg = replace(
            plugin_cfg_base,
            star=replace(
                plugin_cfg_base.star,
                anomaly_tails={
                    "upward": upward_t,
                    "two_sided": plugin_cfg_base.star.anomaly_tails.get("two_sided", ()),
                },
            ),
        )
        stats = evaluate_combo(
            loaded=loaded,
            plugin_cfg=plugin_cfg,
            input_size=input_size,
            hist_exog_cols=exog_t,
            train_df=train_df,
            future_df=future_df,
        )
        if "error" in stats:
            row = {
                "eval_slice": eval_slice,
                "signature_backend": SIGNATURE_BACKEND,
                "input_size": input_size,
                "hist_exog_cols": json.dumps(list(exog_t)),
                "upward_tail_cols": json.dumps(list(upward_t)),
                "error": stats["error"],
                "metric_mode": metric,
                "rank_score": float("nan"),
                **base_row_nan,
            }
            rows.append(row)
            continue
        rank = _rank_score(
            metric,
            actual_cum_ret=stats["actual_horizon_cum_ret"],
            weighted_last=stats["weighted_neighbor_last_ret"],
            mean_last=stats["mean_neighbor_last_ret"],
            max_abs_last=stats["max_abs_neighbor_last_ret"],
        )
        row = {
            "eval_slice": eval_slice,
            "signature_backend": SIGNATURE_BACKEND,
            "input_size": input_size,
            "hist_exog_cols": json.dumps(list(exog_t)),
            "upward_tail_cols": json.dumps(list(upward_t)),
            "error": "",
            "metric_mode": metric,
            "rank_score": rank,
            **stats,
        }
        rows.append(row)

    def _rank_sort_key(r: dict[str, Any]) -> float:
        v = r.get("rank_score")
        if v is None or (isinstance(v, float) and v != v):
            return float("-inf")
        return float(v)

    rows.sort(key=_rank_sort_key, reverse=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "YAML: standalone ``retrieval`` route (dataset + cv + training; no backbone) "
            "or an AAForecast experiment with retrieval."
        ),
    )
    parser.add_argument(
        "--input-sizes",
        type=str,
        default="",
        help="Comma-separated input sizes (default: training.input_size from config only).",
    )
    parser.add_argument(
        "--min-input-size",
        type=int,
        default=1,
        help="Drop input sizes below this (use 12 when searching long STAR windows).",
    )
    parser.add_argument(
        "--eval-slice",
        choices=("last_cv_fold", "max_tail"),
        default="last_cv_fold",
        help=(
            "last_cv_fold: same train/future as runner TSCV last window. "
            "max_tail: train=all rows except last horizon (longer history for large input_size)."
        ),
    )
    parser.add_argument(
        "--exog-grid",
        type=Path,
        default=None,
        help="JSON file: array of hist_exog column-name arrays (default: dataset.hist_exog_cols once).",
    )
    parser.add_argument(
        "--upward-grid",
        type=Path,
        default=None,
        help="JSON file: array of upward-tail column-name arrays (default: retrieval.star anomaly upward once).",
    )
    parser.add_argument(
        "--retrieval-yaml",
        type=Path,
        default=None,
        help=(
            "Use this retrieval detail YAML instead of the one implied by --config "
            "(standalone: replaces merged retrieval; AAForecast: same override)."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("align_weighted_last", "align_mean_last", "max_abs_neighbor_last"),
        default="align_weighted_last",
        help="Ranking key (default: weighted neighbor last-step return aligned with actual cum sign).",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Write ranked rows CSV.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write top-N rows as JSON (requires --limit if set).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0 with --summary-json, only keep top N rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combo count and exit without evaluating.",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    loaded = load_app_config(REPO_ROOT, config_path=config_path)
    if args.retrieval_yaml is not None:
        plugin_cfg = load_retrieval_plugin_config_from_path(args.retrieval_yaml.resolve())
    else:
        plugin_cfg = load_retrieval_plugin_config(loaded)

    dataset_path = Path(str(loaded.config.dataset.path))
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
        input_sizes = [int(loaded.config.training.input_size)]

    before_filter = list(input_sizes)
    input_sizes = [s for s in input_sizes if s >= args.min_input_size]
    if not input_sizes:
        print(
            f"error: no input_sizes remain after --min-input-size {args.min_input_size} "
            f"(candidates were {before_filter})",
            file=sys.stderr,
        )
        return 1

    exog_grid = (
        _load_json_grid(args.exog_grid.resolve())
        if args.exog_grid
        else _default_exog_grid(loaded.config.dataset.hist_exog_cols)
    )
    upward_base = tuple(plugin_cfg.star.anomaly_tails.get("upward", ()))
    upward_grid = (
        _load_json_grid(args.upward_grid.resolve())
        if args.upward_grid
        else _default_upward_grid(upward_base)
    )

    probe_hist = tuple(exog_grid[0]) if exog_grid else loaded.config.dataset.hist_exog_cols
    probe_rows = probe_transformed_row_count(loaded, train_df, probe_hist)
    max_input = max(input_sizes)
    base_dataset_cols = set(source_df.columns)
    combo_count = sum(
        1
        for _ in iter_combos(
            input_sizes=input_sizes,
            exog_grid=exog_grid,
            upward_grid=upward_grid,
            base_dataset_cols=base_dataset_cols,
        )
    )
    skipped_incompatible = count_incompatible_exog_upward_pairs(
        input_sizes=input_sizes,
        exog_grid=exog_grid,
        upward_grid=upward_grid,
        base_dataset_cols=base_dataset_cols,
    )
    print(
        f"eval_slice={args.eval_slice} train_rows={len(train_df)} test_rows={len(future_df)} "
        f"transformed_probe_rows={probe_rows} (first exog grid) "
        f"input_sizes={input_sizes} combos={combo_count}"
    )
    if skipped_incompatible:
        print(
            f"finding: skipped {skipped_incompatible} (input_size, hist_exog, upward) "
            "combo(s): upward_tail_cols not a subset of hist_exog_cols for that row",
            file=sys.stderr,
        )
    if max_input > probe_rows:
        print(
            f"warning: max input_size {max_input} > transformed_probe_rows={probe_rows}; "
            "expect short_transformed_train rows unless other exog grids shorten diff less "
            "(unlikely). Use --eval-slice max_tail and/or a longer dataset.",
            file=sys.stderr,
        )
    if args.dry_run:
        return 0

    rows = run_finding(
        loaded=loaded,
        plugin_cfg_base=plugin_cfg,
        train_df=train_df,
        future_df=future_df,
        input_sizes=input_sizes,
        exog_grid=exog_grid,
        upward_grid=upward_grid,
        metric=args.metric,
        eval_slice=args.eval_slice,
    )

    if args.output_csv:
        out = args.output_csv.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            raise RuntimeError("no rows to write")
        fieldnames = list(rows[0].keys())
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(str(out))

    if args.summary_json:
        lim = args.limit if args.limit > 0 else len(rows)
        payload = rows[:lim]
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(str(args.summary_json.resolve()))

    if not args.output_csv and not args.summary_json:
        print(json.dumps(rows[:10], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
