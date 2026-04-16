"""Histogram of memory-bank ``event_score`` with query and threshold markers.

Used by ``scripts/plot_retrieval_event_score_distribution.py`` and written
automatically when retrieval summary JSON is persisted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

MISSING_BANK_SCORES = (
    "JSON has no 'bank_event_scores' field. Re-run the fold with a runtime that "
    "serializes bank scores (current plugins/aa_forecast and plugins/retrieval), "
    "or regenerate the retrieval artifact."
)


def load_payload(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def _query_rank_percentile(scores: np.ndarray, query_score: float) -> float | None:
    if scores.size == 0:
        return None
    ordered = np.sort(scores)
    # Fraction of bank scores strictly below query (position in distribution).
    below = float(np.searchsorted(ordered, query_score, side="left"))
    return 100.0 * below / float(scores.size)


def write_event_score_distribution_plot(
    payload: dict[str, Any],
    *,
    out_path: Path,
    dpi: int = 150,
) -> None:
    if "bank_event_scores" not in payload:
        raise ValueError(MISSING_BANK_SCORES)
    scores = np.asarray(payload["bank_event_scores"], dtype=float)
    query_score = float(payload["query_event_score"])
    threshold = float(payload["effective_event_threshold"])
    trigger_q = payload.get("trigger_quantile")
    applied = payload.get("retrieval_applied")
    skip = payload.get("skip_reason")
    cutoff = payload.get("cutoff") or payload.get("train_end_ds", "")

    rank_pct = _query_rank_percentile(scores, query_score)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if scores.size > 0:
        bins = min(50, max(10, int(np.sqrt(scores.size))))
        ax.hist(
            scores,
            bins=bins,
            density=True,
            alpha=0.72,
            color="tab:blue",
            edgecolor="white",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "bank_event_scores is empty (no bank candidates).",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.axvline(
        query_score,
        color="tab:orange",
        linewidth=2.0,
        label=f"query_event_score = {query_score:.4g}",
    )
    ax.axvline(
        threshold,
        color="tab:red",
        linewidth=2.0,
        linestyle="--",
        label=f"effective_event_threshold = {threshold:.4g}",
    )
    ax.set_xlabel("event_score (memory bank candidates)")
    ax.set_ylabel("density")
    ax.grid(axis="y", alpha=0.25)

    title_parts = [f"Retrieval event_score distribution — cutoff {cutoff}"]
    if trigger_q is not None:
        title_parts.append(f"trigger_quantile={trigger_q}")
    title_parts.append(f"retrieval_applied={applied}")
    if skip is not None:
        title_parts.append(f"skip_reason={skip!r}")
    if rank_pct is not None:
        title_parts.append(
            f"query below-pct in bank = {rank_pct:.1f}% "
            f"(share of candidates with score < query)"
        )
    ax.set_title("\n".join(title_parts), fontsize=10)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
