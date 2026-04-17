"""Runtime logic for standalone retrieval: memory bank, neighbour search, blending.

Mirrors the retrieval pipeline from ``plugins.aa_forecast.runtime`` but is
decoupled from the AAForecast model.  All STAR signature work is delegated to
``plugins.retrieval.signatures``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as _cfg
from .event_score_distribution_plot import write_event_score_distribution_plot
from .similarity_window_plot import write_similarity_plot_set
from .signatures import (
    _build_star_extractor,
    _resolve_tail_modes,
    compute_star_signature,
)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("retrieval cosine similarity requires matching vector shapes")
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


# ---------------------------------------------------------------------------
# Memory bank
# ---------------------------------------------------------------------------


def _build_memory_bank(
    *,
    star: Any,
    transformed_train_df: pd.DataFrame,
    raw_train_df: pd.DataFrame,
    dt_col: str,
    target_col: str,
    hist_exog_cols: tuple[str, ...],
    hist_exog_tail_modes: tuple[str, ...],
    retrieval_cfg: _cfg.RetrievalConfig,
    input_size: int,
    horizon: int,
) -> tuple[list[dict[str, Any]], int]:
    last_idx = len(raw_train_df) - 1
    max_end_idx = last_idx - horizon - retrieval_cfg.recency_gap_steps
    if max_end_idx < input_size - 1:
        return [], 0
    candidate_count = max_end_idx - (input_size - 1) + 1
    bank: list[dict[str, Any]] = []
    for end_idx in range(input_size - 1, max_end_idx + 1):
        start_idx = end_idx - input_size + 1
        transformed_window = transformed_train_df.iloc[
            start_idx : end_idx + 1
        ].reset_index(drop=True)
        signature = compute_star_signature(
            star=star,
            window_df=transformed_window,
            target_col=target_col,
            hist_exog_cols=hist_exog_cols,
            hist_exog_tail_modes=hist_exog_tail_modes,
            insample_y_included=retrieval_cfg.insample_y_included,
        )
        anchor_value = float(raw_train_df[target_col].iloc[end_idx])
        future_values = raw_train_df[target_col].iloc[
            end_idx + 1 : end_idx + 1 + horizon
        ]
        if len(future_values) != horizon:
            continue
        scale = max(abs(anchor_value), 1e-8)
        future_returns = (future_values.to_numpy(dtype=float) - anchor_value) / scale
        bank.append(
            {
                "candidate_end_ds": str(
                    pd.Timestamp(raw_train_df[dt_col].iloc[end_idx])
                ),
                "candidate_future_end_ds": str(
                    pd.Timestamp(raw_train_df[dt_col].iloc[end_idx + horizon])
                ),
                "event_vector": signature["event_vector"],
                "event_score": signature["event_score"],
                "anchor_target_value": anchor_value,
                "future_returns": future_returns,
            }
        )
    return bank, candidate_count


def _build_query(
    *,
    star: Any,
    transformed_train_df: pd.DataFrame,
    target_col: str,
    hist_exog_cols: tuple[str, ...],
    hist_exog_tail_modes: tuple[str, ...],
    retrieval_cfg: _cfg.RetrievalConfig,
    input_size: int,
) -> dict[str, Any]:
    window = transformed_train_df.iloc[-input_size:].reset_index(drop=True)
    return compute_star_signature(
        star=star,
        window_df=window,
        target_col=target_col,
        hist_exog_cols=hist_exog_cols,
        hist_exog_tail_modes=hist_exog_tail_modes,
        insample_y_included=retrieval_cfg.insample_y_included,
    )


# ---------------------------------------------------------------------------
# Neighbour retrieval
# ---------------------------------------------------------------------------


def _retrieve_neighbors(
    *,
    query: dict[str, Any],
    bank: list[dict[str, Any]],
    retrieval_cfg: _cfg.RetrievalConfig,
    effective_event_threshold: float,
) -> dict[str, Any]:
    threshold = effective_event_threshold
    if query["event_score"] < threshold:
        return {
            "retrieval_attempted": True,
            "retrieval_applied": False,
            "skip_reason": "below_event_threshold",
            "top_neighbors": [],
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
        }
    if not bank:
        return {
            "retrieval_attempted": True,
            "retrieval_applied": False,
            "skip_reason": "empty_bank",
            "top_neighbors": [],
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
        }
    if not retrieval_cfg.use_event_key:
        raise ValueError("retrieval requires use_event_key=true")
    scored_neighbors: list[dict[str, Any]] = []
    for entry in bank:
        if float(entry["event_score"]) < threshold:
            continue
        event_similarity = _cosine_similarity(
            query["event_vector"], entry["event_vector"]
        )
        event_component = event_similarity
        if retrieval_cfg.event_score_log_bonus_alpha > 0.0:
            query_event_score = max(float(query["event_score"]), 1e-8)
            candidate_event_score = max(float(entry["event_score"]), 1e-8)
            event_score_log_bonus = min(
                max(math.log(candidate_event_score / query_event_score), 0.0),
                retrieval_cfg.event_score_log_bonus_cap,
            )
            event_component = event_component + (
                retrieval_cfg.event_score_log_bonus_alpha * event_score_log_bonus
            )
        similarity = event_component
        if similarity < retrieval_cfg.min_similarity:
            continue
        scored_neighbors.append(
            {
                **entry,
                "event_similarity": event_similarity,
                "similarity": similarity,
            }
        )
    if not scored_neighbors:
        return {
            "retrieval_attempted": True,
            "retrieval_applied": False,
            "skip_reason": "min_similarity",
            "top_neighbors": [],
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
        }
    scored_neighbors.sort(key=lambda item: item["similarity"], reverse=True)
    top_neighbors = scored_neighbors[: retrieval_cfg.top_k]
    logits = (
        np.asarray(
            [neighbor["similarity"] for neighbor in top_neighbors],
            dtype=float,
        )
        / retrieval_cfg.temperature
    )
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights = weights / weights.sum()
    for neighbor, weight in zip(top_neighbors, weights, strict=True):
        neighbor["softmax_weight"] = float(weight)
    similarities = np.asarray(
        [neighbor["similarity"] for neighbor in top_neighbors],
        dtype=float,
    )
    return {
        "retrieval_attempted": True,
        "retrieval_applied": True,
        "skip_reason": None,
        "top_neighbors": top_neighbors,
        "mean_similarity": float(similarities.mean()),
        "max_similarity": float(similarities.max()),
    }


def _effective_event_threshold(
    *,
    bank: list[dict[str, Any]],
    retrieval_cfg: _cfg.RetrievalConfig,
) -> float:
    if not bank:
        # Prefer 0.0 so query gating doesn't hide the more-informative
        # "empty_bank" skip_reason downstream.
        return 0.0
    q = retrieval_cfg.trigger_quantile
    if q is None:
        raise ValueError("retrieval.trigger_quantile is required")
    scores = np.asarray([float(item["event_score"]) for item in bank], dtype=float)
    return float(np.quantile(scores, float(q)))


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


def _blend_prediction(
    *,
    base_prediction: np.ndarray,
    memory_prediction: np.ndarray,
    uncertainty_std: np.ndarray | None,
    retrieval_cfg: _cfg.RetrievalConfig,
    mean_similarity: float,
) -> tuple[np.ndarray, np.ndarray]:
    similarity_scale = float(np.clip(mean_similarity, 0.0, 1.0))
    if retrieval_cfg.use_uncertainty_gate and uncertainty_std is not None:
        std_values = np.asarray(uncertainty_std, dtype=float)
        max_std = float(np.max(std_values))
        if max_std > 1e-12:
            uncertainty_scale = std_values / max_std
        else:
            uncertainty_scale = np.ones_like(std_values)
    else:
        uncertainty_scale = np.ones_like(base_prediction, dtype=float)
    blend_weight = (
        retrieval_cfg.blend_floor
        + (retrieval_cfg.blend_max - retrieval_cfg.blend_floor)
        * similarity_scale
        * uncertainty_scale
    )
    blend_weight = np.clip(
        blend_weight, retrieval_cfg.blend_floor, retrieval_cfg.blend_max
    )
    final_prediction = (1.0 - blend_weight) * np.asarray(
        base_prediction, dtype=float
    ) + blend_weight * np.asarray(memory_prediction, dtype=float)
    return final_prediction, np.asarray(blend_weight, dtype=float)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _train_end_index_for_candidate(
    train_df: pd.DataFrame,
    dt_col: str,
    candidate_end_ds: str,
) -> int | None:
    cand = pd.Timestamp(candidate_end_ds)
    series = pd.to_datetime(train_df[dt_col])
    for i in range(len(series)):
        if pd.Timestamp(series.iloc[i]) == cand:
            return i
    cand_norm = cand.normalize()
    for i in range(len(series)):
        if pd.Timestamp(series.iloc[i]).normalize() == cand_norm:
            return i
    return None


def _window_series_slice(
    train_df: pd.DataFrame,
    transformed_train_df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    target_col: str,
    dt_col: str,
) -> dict[str, Any]:
    sl_train = train_df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    sl_tf = transformed_train_df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    y_raw = sl_train[target_col].astype(float).tolist()
    ds = pd.to_datetime(sl_train[dt_col]).astype(str).tolist()
    if target_col in sl_tf.columns:
        y_tf = sl_tf[target_col].astype(float).tolist()
    else:
        y_tf = [float("nan")] * len(ds)
    return {"ds": ds, "y_raw": y_raw, "y_transformed": y_tf}


def _align_retrieval_window_lists(window: dict[str, Any]) -> dict[str, Any]:
    ds = list(window["ds"])
    y_raw = list(window["y_raw"])
    y_tf = list(window["y_transformed"])
    L = max(len(ds), len(y_raw), len(y_tf))
    nan = float("nan")
    if len(ds) < L:
        ds = [""] * (L - len(ds)) + ds
    elif len(ds) > L:
        ds = ds[-L:]
    if len(y_raw) < L:
        y_raw = [nan] * (L - len(y_raw)) + y_raw
    elif len(y_raw) > L:
        y_raw = y_raw[-L:]
    if len(y_tf) < L:
        y_tf = [nan] * (L - len(y_tf)) + y_tf
    elif len(y_tf) > L:
        y_tf = y_tf[-L:]
    return {"ds": ds, "y_raw": y_raw, "y_transformed": y_tf}


def _normalize_retrieval_series_window(
    window: dict[str, Any], *, input_size: int
) -> dict[str, Any]:
    window = _align_retrieval_window_lists(window)
    ds = list(window["ds"])
    y_raw = list(window["y_raw"])
    y_tf = list(window["y_transformed"])
    n_w = len(ds)
    if n_w == input_size:
        return {"ds": ds, "y_raw": y_raw, "y_transformed": y_tf}
    if n_w > input_size:
        drop = n_w - input_size
        return {
            "ds": ds[drop:],
            "y_raw": y_raw[drop:],
            "y_transformed": y_tf[drop:],
        }
    pad = input_size - n_w
    nan = float("nan")
    return {
        "ds": [""] * pad + ds,
        "y_raw": [nan] * pad + y_raw,
        "y_transformed": [nan] * pad + y_tf,
    }


def _build_retrieval_window_artifacts(
    *,
    train_df: pd.DataFrame,
    transformed_train_df: pd.DataFrame,
    dt_col: str,
    target_col: str,
    input_size: int,
    horizon: int,
    neighbors: list[dict[str, Any]],
    train_end_ds: pd.Timestamp,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = len(train_df)
    q_start = max(0, n - input_size)
    query_w = _window_series_slice(
        train_df,
        transformed_train_df,
        q_start,
        n - 1,
        target_col,
        dt_col,
    )
    query_w = _normalize_retrieval_series_window(query_w, input_size=input_size)
    long_rows: list[dict[str, Any]] = []
    for step_ix in range(input_size):
        long_rows.append(
            {
                "role": "query",
                "rank": "",
                "step_ix": step_ix,
                "ds": query_w["ds"][step_ix],
                "y_raw": query_w["y_raw"][step_ix],
                "y_transformed": query_w["y_transformed"][step_ix],
                "similarity": float("nan"),
                "softmax_weight": float("nan"),
                "candidate_end_ds": "",
            }
        )
    windows_payload: dict[str, Any] = {
        "train_end_ds": str(pd.Timestamp(train_end_ds)),
        "input_size": input_size,
        "horizon": horizon,
        "target_col": target_col,
        "query": query_w,
        "neighbors": [],
    }
    for rank, neighbor in enumerate(neighbors, start=1):
        end_idx = _train_end_index_for_candidate(
            train_df, dt_col, str(neighbor["candidate_end_ds"])
        )
        if end_idx is None:
            windows_payload["neighbors"].append(
                {
                    "rank": rank,
                    "candidate_end_ds": neighbor["candidate_end_ds"],
                    "similarity": neighbor.get("similarity"),
                    "softmax_weight": neighbor.get("softmax_weight"),
                    "ds": [],
                    "y_raw": [],
                    "y_transformed": [],
                    "future_ds": [],
                    "future_y_raw": [],
                    "match_error": "anchor_not_found",
                }
            )
            continue
        start_idx = end_idx - input_size + 1
        if start_idx < 0:
            windows_payload["neighbors"].append(
                {
                    "rank": rank,
                    "candidate_end_ds": neighbor["candidate_end_ds"],
                    "similarity": neighbor.get("similarity"),
                    "softmax_weight": neighbor.get("softmax_weight"),
                    "ds": [],
                    "y_raw": [],
                    "y_transformed": [],
                    "future_ds": [],
                    "future_y_raw": [],
                    "match_error": "window_underflow",
                }
            )
            continue
        win = _align_retrieval_window_lists(
            _window_series_slice(
                train_df,
                transformed_train_df,
                start_idx,
                end_idx,
                target_col,
                dt_col,
            )
        )
        fut_slice = train_df.iloc[end_idx + 1 : end_idx + 1 + horizon]
        fut_ds = pd.to_datetime(fut_slice[dt_col]).astype(str).tolist()
        fut_y = fut_slice[target_col].astype(float).tolist()
        sim_v = float(neighbor["similarity"])
        w_v = float(neighbor["softmax_weight"])
        for step_ix in range(input_size):
            long_rows.append(
                {
                    "role": "neighbor",
                    "rank": rank,
                    "step_ix": step_ix,
                    "ds": win["ds"][step_ix],
                    "y_raw": win["y_raw"][step_ix],
                    "y_transformed": win["y_transformed"][step_ix],
                    "similarity": sim_v,
                    "softmax_weight": w_v,
                    "candidate_end_ds": str(neighbor["candidate_end_ds"]),
                }
            )
        for j, (fds, fy) in enumerate(zip(fut_ds, fut_y)):
            long_rows.append(
                {
                    "role": "neighbor_future",
                    "rank": rank,
                    "step_ix": input_size + j,
                    "ds": fds,
                    "y_raw": fy,
                    "y_transformed": float("nan"),
                    "similarity": sim_v,
                    "softmax_weight": w_v,
                    "candidate_end_ds": str(neighbor["candidate_end_ds"]),
                }
            )
        windows_payload["neighbors"].append(
            {
                "rank": rank,
                "candidate_end_ds": neighbor["candidate_end_ds"],
                "similarity": neighbor.get("similarity"),
                "softmax_weight": neighbor.get("softmax_weight"),
                "ds": win["ds"],
                "y_raw": win["y_raw"],
                "y_transformed": win["y_transformed"],
                "future_ds": fut_ds,
                "future_y_raw": fut_y,
            }
        )
    return windows_payload, long_rows


def _write_retrieval_artifacts(
    *,
    run_root: Path,
    train_end_ds: pd.Timestamp,
    retrieval_summary: dict[str, Any],
    train_df: pd.DataFrame,
    transformed_train_df: pd.DataFrame,
    dt_col: str,
    target_col: str,
    input_size: int,
    horizon: int,
) -> str:
    stage_root = run_root / "retrieval"
    stage_root.mkdir(parents=True, exist_ok=True)
    cutoff_tag = str(train_end_ds).replace(" ", "_").replace(":", "")
    summary_path = stage_root / f"retrieval_summary_{cutoff_tag}.json"
    windows_path = stage_root / f"retrieval_summary_{cutoff_tag}_windows.json"
    windows_long_csv = stage_root / f"retrieval_summary_{cutoff_tag}_windows_long.csv"

    serializable_summary = {}
    for key, value in retrieval_summary.items():
        if isinstance(value, np.ndarray):
            serializable_summary[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            serializable_neighbors = []
            for neighbor in value:
                clean = {}
                for nk, nv in neighbor.items():
                    if isinstance(nv, np.ndarray):
                        clean[nk] = nv.tolist()
                    else:
                        clean[nk] = nv
                serializable_neighbors.append(clean)
            serializable_summary[key] = serializable_neighbors
        else:
            serializable_summary[key] = value

    _write_json(summary_path, serializable_summary)
    plot_path = summary_path.with_name(
        f"{summary_path.stem}_event_score_dist.png"
    )
    write_event_score_distribution_plot(
        serializable_summary, out_path=plot_path
    )
    windows_payload, long_rows = _build_retrieval_window_artifacts(
        train_df=train_df.reset_index(drop=True),
        transformed_train_df=transformed_train_df.reset_index(drop=True),
        dt_col=dt_col,
        target_col=target_col,
        input_size=input_size,
        horizon=horizon,
        neighbors=list(retrieval_summary["neighbors"]),
        train_end_ds=train_end_ds,
    )
    _write_json(windows_path, windows_payload)
    pd.DataFrame(long_rows).to_csv(windows_long_csv, index=False)
    write_similarity_plot_set(
        serializable_summary,
        windows_payload,
        out_dir=stage_root,
        stem=summary_path.stem,
    )
    return str(summary_path)


# ---------------------------------------------------------------------------
# Post-predict entry point
# ---------------------------------------------------------------------------


def post_predict_retrieval(
    *,
    plugin_cfg: _cfg.RetrievalPluginConfig,
    target_predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    transformed_train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target_col: str,
    dt_col: str,
    hist_exog_cols: tuple[str, ...],
    prediction_col: str,
    input_size: int,
    horizon: int,
    run_root: Path | None,
) -> pd.DataFrame:
    """Apply standalone retrieval to ``target_predictions`` and return the
    modified frame.
    """
    retrieval_cfg = plugin_cfg.retrieval
    star_cfg = plugin_cfg.star
    star = _build_star_extractor(star_cfg)

    available_hist_exog = tuple(
        col for col in hist_exog_cols
        if col in transformed_train_df.columns
    )
    hist_exog_tail_modes = _resolve_tail_modes(available_hist_exog, star_cfg.anomaly_tails)

    if len(transformed_train_df) < input_size:
        raise ValueError(
            "retrieval requires transformed_train_df rows >= input_size"
        )

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
        retrieval_cfg=retrieval_cfg,
        input_size=input_size,
    )

    effective_event_threshold = _effective_event_threshold(
        bank=bank,
        retrieval_cfg=retrieval_cfg,
    )

    eligible_bank = [
        entry
        for entry in bank
        if float(entry["event_score"]) >= effective_event_threshold
    ]

    retrieval_result = _retrieve_neighbors(
        query=query,
        bank=eligible_bank,
        retrieval_cfg=retrieval_cfg,
        effective_event_threshold=effective_event_threshold,
    )

    base_prediction = np.asarray(target_predictions[prediction_col], dtype=float)
    current_last_y = float(train_df[target_col].iloc[-1])
    bank_event_scores = [float(entry["event_score"]) for entry in bank]

    retrieval_summary: dict[str, Any] = {
        "cutoff": str(pd.Timestamp(train_df[dt_col].iloc[-1])),
        "train_end_ds": str(pd.Timestamp(train_df[dt_col].iloc[-1])),
        "retrieval_enabled": True,
        "retrieval_attempted": retrieval_result["retrieval_attempted"],
        "retrieval_applied": False,
        "skip_reason": retrieval_result["skip_reason"],
        "top_k_requested": retrieval_cfg.top_k,
        "top_k_used": len(retrieval_result["top_neighbors"]),
        "candidate_count": candidate_count,
        "eligible_candidate_count": len(eligible_bank),
        "bank_event_scores": bank_event_scores,
        "effective_event_threshold": effective_event_threshold,
        "trigger_quantile": retrieval_cfg.trigger_quantile,
        "recency_gap_steps": retrieval_cfg.recency_gap_steps,
        "min_similarity": retrieval_cfg.min_similarity,
        "mean_similarity": retrieval_result["mean_similarity"],
        "max_similarity": retrieval_result["max_similarity"],
        "query_event_score": query["event_score"],
        "blend_max": retrieval_cfg.blend_max,
        "blend_weight_by_horizon": [0.0] * len(base_prediction),
        "used_uncertainty_gate": retrieval_cfg.use_uncertainty_gate,
        "base_prediction": base_prediction.tolist(),
        "memory_prediction": base_prediction.tolist(),
        "final_prediction": base_prediction.tolist(),
        "neighbors": retrieval_result["top_neighbors"],
    }

    if retrieval_result["retrieval_applied"]:
        weighted_returns = np.zeros_like(base_prediction, dtype=float)
        for neighbor in retrieval_result["top_neighbors"]:
            weighted_returns = weighted_returns + (
                float(neighbor["softmax_weight"])
                * np.asarray(neighbor["future_returns"], dtype=float)
            )
        scale = max(abs(current_last_y), 1e-8)
        memory_prediction = current_last_y + scale * weighted_returns
        final_prediction, blend_weight = _blend_prediction(
            base_prediction=base_prediction,
            memory_prediction=memory_prediction,
            uncertainty_std=None,
            retrieval_cfg=retrieval_cfg,
            mean_similarity=retrieval_result["mean_similarity"],
        )
        target_predictions[prediction_col] = final_prediction
        retrieval_summary["retrieval_applied"] = True
        retrieval_summary["skip_reason"] = None
        retrieval_summary["blend_weight_by_horizon"] = blend_weight.tolist()
        retrieval_summary["memory_prediction"] = memory_prediction.tolist()
        retrieval_summary["final_prediction"] = final_prediction.tolist()

    retrieval_artifact = None
    if run_root is not None:
        retrieval_artifact = _write_retrieval_artifacts(
            run_root=run_root,
            train_end_ds=pd.to_datetime(train_df[dt_col].iloc[-1]),
            retrieval_summary=retrieval_summary,
            train_df=train_df,
            transformed_train_df=transformed_train_df,
            dt_col=dt_col,
            target_col=target_col,
            input_size=input_size,
            horizon=horizon,
        )

    target_predictions["retrieval_enabled"] = pd.Series(
        [True] * len(target_predictions),
        dtype="boolean",
    )
    target_predictions["retrieval_applied"] = pd.Series(
        [retrieval_summary["retrieval_applied"]] * len(target_predictions),
        dtype="boolean",
    )
    target_predictions["retrieval_skip_reason"] = pd.Series(
        [retrieval_summary["skip_reason"]] * len(target_predictions),
        dtype="object",
    )
    if retrieval_artifact is not None:
        target_predictions["retrieval_artifact"] = retrieval_artifact

    return target_predictions
