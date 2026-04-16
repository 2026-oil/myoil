"""Standalone STAR-based signature builder for retrieval.

Replicates the signature logic from ``plugins.aa_forecast.runtime`` but
operates without the AAForecast model.  A fresh ``STARFeatureExtractor``
instance is created from the plugin config and applied directly to the
target + hist_exog columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from plugins.aa_forecast.modules import STARFeatureExtractor

from . import config as _cfg


def _normalize_signature(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError("retrieval signature must not be empty")
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _build_star_extractor(star_cfg: _cfg.RetrievalStarConfig) -> STARFeatureExtractor:
    return STARFeatureExtractor(
        season_length=star_cfg.season_length,
        lowess_frac=star_cfg.lowess_frac,
        lowess_delta=star_cfg.lowess_delta,
        thresh=star_cfg.thresh,
    )


def _resolve_tail_modes(
    hist_exog_cols: tuple[str, ...],
    anomaly_tails: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Assign a tail mode to each hist_exog column based on anomaly_tails config."""
    upward_set = set(anomaly_tails.get("upward", ()))
    modes: list[str] = []
    for col in hist_exog_cols:
        modes.append("upward" if col in upward_set else "two_sided")
    return tuple(modes)


def compute_star_signature(
    *,
    star: STARFeatureExtractor,
    window_df: pd.DataFrame,
    target_col: str,
    hist_exog_cols: tuple[str, ...],
    hist_exog_tail_modes: tuple[str, ...],
) -> dict[str, Any]:
    """Build event_vector and event_score from STAR decomposition.

    This mirrors ``_build_retrieval_signature`` in the aa_forecast runtime but
    uses a standalone ``STARFeatureExtractor`` instead of the model's internal
    STAR outputs.
    """
    target_values = window_df[target_col].to_numpy(dtype=np.float32)
    insample_y = torch.as_tensor(target_values, dtype=torch.float32).reshape(1, -1, 1)

    with torch.no_grad():
        target_star = star(insample_y, tail_modes=("two_sided",))

    target_mask = target_star["critical_mask"].numpy().astype(float)
    target_ranking = target_star["ranking_score"].numpy().astype(float)
    target_activity = target_ranking * target_mask

    # target_count: per-timestep count of active channels (target has 1 channel)
    target_count = target_mask.astype(float).sum(axis=2, keepdims=True)

    hist_activities: list[np.ndarray] = []
    hist_count = np.zeros_like(target_count)

    if hist_exog_cols:
        hist_values = window_df[list(hist_exog_cols)].to_numpy(dtype=np.float32)
        hist_tensor = torch.as_tensor(hist_values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            hist_star = star(hist_tensor, tail_modes=hist_exog_tail_modes)
        hist_mask = hist_star["critical_mask"].numpy().astype(float)
        hist_ranking = hist_star["ranking_score"].numpy().astype(float)
        hist_act = hist_ranking * hist_mask
        hist_activities.append(hist_act.reshape(hist_act.shape[1], hist_act.shape[2]))
        hist_count = hist_mask.astype(float).sum(axis=2, keepdims=True)

    combined_count = target_count + hist_count

    # Flatten to 2D (time, channels) for concatenation
    target_activity_2d = target_activity.reshape(
        target_activity.shape[1], target_activity.shape[2]
    )
    all_activity = [target_activity_2d] + hist_activities
    channel_activity = np.concatenate(all_activity, axis=1)

    critical_mask = (combined_count > 0).reshape(-1).astype(float)
    count_active = combined_count.reshape(-1).astype(float)

    activity_sums = channel_activity.sum(axis=0)
    activity_max = channel_activity.max(axis=0)
    event_vector = np.concatenate(
        [
            critical_mask,
            count_active,
            channel_activity.reshape(-1),
            activity_sums,
            activity_max,
        ]
    )
    event_score = float(count_active.sum() + np.abs(channel_activity).sum())

    return {
        "event_vector": _normalize_signature(event_vector),
        "event_score": event_score,
    }
