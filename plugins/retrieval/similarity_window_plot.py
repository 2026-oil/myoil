"""PNG plot helpers for retrieval query/neighbor similarity inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

MISSING_QUERY = "windows payload has no 'query' section."
MISSING_NEIGHBORS = "windows payload has invalid 'neighbors' data."
MISSING_TRANSFORMED = "windows payload is missing transformed values required for plotting."


def load_payload(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def _require_query(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query")
    if not isinstance(query, dict):
        raise ValueError(MISSING_QUERY)
    return query


def _require_neighbors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    neighbors = payload.get("neighbors")
    if not isinstance(neighbors, list):
        raise ValueError(MISSING_NEIGHBORS)
    return [n for n in neighbors if isinstance(n, dict)]


def _require_series(section: dict[str, Any], key: str) -> np.ndarray:
    values = section.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(MISSING_TRANSFORMED if key == "y_transformed" else MISSING_QUERY)
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        raise ValueError(MISSING_TRANSFORMED if key == "y_transformed" else MISSING_QUERY)
    return arr


def _summary_text(summary_payload: dict[str, Any]) -> str:
    neighbors = summary_payload.get("neighbors") or []
    lines = [
        f"cutoff: {summary_payload.get('cutoff') or summary_payload.get('train_end_ds', 'n/a')}",
        f"top_k_used: {summary_payload.get('top_k_used', 'n/a')}",
        f"recency_gap_steps: {summary_payload.get('recency_gap_steps', 'n/a')}",
        f"trigger_quantile: {summary_payload.get('trigger_quantile', 'n/a')}",
        f"min_similarity: {summary_payload.get('min_similarity', 'n/a')}",
        f"query_event_score: {summary_payload.get('query_event_score', 'n/a')}",
        f"effective_event_threshold: {summary_payload.get('effective_event_threshold', 'n/a')}",
        f"retrieval_applied: {summary_payload.get('retrieval_applied', 'n/a')}",
    ]
    for idx, neighbor in enumerate(neighbors[:3], start=1):
        lines.append(
            " | ".join(
                [
                    f"rank {idx}",
                    f"end={neighbor.get('candidate_end_ds', 'n/a')}",
                    f"sim={float(neighbor.get('similarity', float('nan'))):.4f}",
                    f"event_sim={float(neighbor.get('event_similarity', float('nan'))):.4f}",
                    f"event_score={float(neighbor.get('event_score', float('nan'))):.4g}",
                ]
            )
        )
    return "\n".join(lines)


def write_similarity_raw_overlay_plot(
    windows_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    *,
    out_path: Path,
    dpi: int = 150,
) -> None:
    query = _require_query(windows_payload)
    neighbors = _require_neighbors(windows_payload)
    query_raw = _require_series(query, "y_raw")
    input_size = int(windows_payload.get("input_size", len(query_raw)))
    valid_neighbors = [
        n
        for n in neighbors
        if isinstance(n.get("y_raw"), list) and len(n["y_raw"]) == input_size
    ]

    fig, axes = plt.subplots(
        1 + len(valid_neighbors),
        1,
        figsize=(11, 3.0 * (1 + len(valid_neighbors))),
        squeeze=False,
    )
    ax_list = axes.ravel()
    x_in = np.arange(input_size)
    ax_list[0].plot(x_in, query_raw, color="C0", linewidth=1.8, label="query raw")
    ax_list[0].scatter([input_size - 1], [query_raw[-1]], color="C0", s=28, zorder=3)
    ax_list[0].set_title(
        f"Query raw window — end={windows_payload.get('train_end_ds', 'n/a')}"
    )
    ax_list[0].set_ylabel(windows_payload.get("target_col", "target"))
    ax_list[0].grid(True, alpha=0.3)
    ax_list[0].legend(loc="upper left")
    if not valid_neighbors:
        ax_list[0].text(
            0.99,
            0.02,
            "No selected neighbors.\nSee skip_reason in summary panel.",
            ha="right",
            va="bottom",
            transform=ax_list[0].transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    for idx, neighbor in enumerate(valid_neighbors, start=1):
        ax = ax_list[idx]
        neighbor_raw = _require_series(neighbor, "y_raw")
        ax.plot(x_in, query_raw, color="C0", linewidth=1.4, alpha=0.8, label="query raw")
        ax.plot(
            x_in,
            neighbor_raw,
            color="C1",
            linewidth=1.8,
            label="neighbor raw",
        )
        future_y = np.asarray(neighbor.get("future_y_raw") or [], dtype=float)
        if future_y.size > 0:
            x_future = np.arange(input_size - 1, input_size + future_y.size)
            y_future = np.concatenate(([neighbor_raw[-1]], future_y))
            ax.plot(
                x_future,
                y_future,
                color="C1",
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label="neighbor realized future",
            )
        title = (
            f"Neighbor rank={neighbor.get('rank', idx)} "
            f"end={neighbor.get('candidate_end_ds', 'n/a')} "
            f"sim={float(neighbor.get('similarity', float('nan'))):.4f} "
            f"w={float(neighbor.get('softmax_weight', float('nan'))):.4f}"
        )
        ax.set_title(title)
        ax.set_ylabel(windows_payload.get("target_col", "target"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    ax_list[-1].set_xlabel("aligned step index (0..L-1, dashed = neighbor future)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_similarity_transformed_overlay_plot(
    windows_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    *,
    out_path: Path,
    dpi: int = 150,
) -> None:
    query = _require_query(windows_payload)
    neighbors = _require_neighbors(windows_payload)
    query_transformed = _require_series(query, "y_transformed")
    input_size = int(windows_payload.get("input_size", len(query_transformed)))
    valid_neighbors = [
        n
        for n in neighbors
        if isinstance(n.get("y_transformed"), list)
        and len(n["y_transformed"]) == input_size
    ]

    fig, axes = plt.subplots(
        max(1, len(valid_neighbors)),
        1,
        figsize=(11, 3.0 * max(1, len(valid_neighbors))),
        squeeze=False,
        sharex=True,
    )
    ax_list = axes.ravel()
    x_in = np.arange(input_size)
    if not valid_neighbors:
        ax = ax_list[0]
        ax.plot(
            x_in,
            query_transformed,
            color="C0",
            linewidth=1.6,
            label="query transformed",
        )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title("Similarity inputs — no selected neighbors")
        ax.set_ylabel("transformed")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    for idx, neighbor in enumerate(valid_neighbors):
        ax = ax_list[idx]
        neighbor_transformed = _require_series(neighbor, "y_transformed")
        ax.plot(
            x_in,
            query_transformed,
            color="C0",
            linewidth=1.6,
            label="query transformed",
        )
        ax.plot(
            x_in,
            neighbor_transformed,
            color="C3",
            linewidth=1.6,
            label="neighbor transformed",
        )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        title = (
            f"Similarity inputs — rank={neighbor.get('rank', idx + 1)} "
            f"end={neighbor.get('candidate_end_ds', 'n/a')} "
            f"sim={float(neighbor.get('similarity', float('nan'))):.4f} "
            f"event_sim={float(_neighbor_stat(summary_payload, idx, 'event_similarity')):.4f}"
        )
        ax.set_title(title)
        ax.set_ylabel("transformed")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    ax_list[-1].set_xlabel("aligned step index used for similarity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _neighbor_stat(summary_payload: dict[str, Any], idx: int, key: str) -> float:
    neighbors = summary_payload.get("neighbors")
    if not isinstance(neighbors, list) or idx >= len(neighbors):
        return float("nan")
    return float(neighbors[idx].get(key, float("nan")))


def write_similarity_summary_plot(
    summary_payload: dict[str, Any],
    windows_payload: dict[str, Any],
    *,
    out_path: Path,
    dpi: int = 150,
) -> None:
    bank_scores = np.asarray(summary_payload.get("bank_event_scores") or [], dtype=float)
    query_score = float(summary_payload.get("query_event_score", float("nan")))
    threshold = float(summary_payload.get("effective_event_threshold", float("nan")))
    neighbors = summary_payload.get("neighbors") or []

    fig, (ax_hist, ax_text) = plt.subplots(
        1,
        2,
        figsize=(13, 4.8),
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )
    if bank_scores.size > 0:
        bins = min(50, max(10, int(np.sqrt(bank_scores.size))))
        ax_hist.hist(
            bank_scores,
            bins=bins,
            density=True,
            alpha=0.72,
            color="tab:blue",
            edgecolor="white",
        )
    else:
        ax_hist.text(
            0.5,
            0.5,
            "bank_event_scores is empty",
            ha="center",
            va="center",
            transform=ax_hist.transAxes,
        )
    if not np.isnan(query_score):
        ax_hist.axvline(
            query_score,
            color="tab:orange",
            linewidth=2.0,
            label=f"query={query_score:.4g}",
        )
    if not np.isnan(threshold):
        ax_hist.axvline(
            threshold,
            color="tab:red",
            linewidth=1.8,
            linestyle="--",
            label=f"threshold={threshold:.4g}",
        )
    palette = ["tab:green", "tab:purple", "tab:brown"]
    for idx, neighbor in enumerate(neighbors[:3]):
        score = float(neighbor.get("event_score", float("nan")))
        if np.isnan(score):
            continue
        ax_hist.axvline(
            score,
            color=palette[idx % len(palette)],
            linewidth=1.6,
            linestyle="-.",
            label=f"rank{idx + 1} event={score:.4g}",
        )
    ax_hist.set_xlabel("event_score")
    ax_hist.set_ylabel("density")
    ax_hist.set_title(
        "Retrieval score summary — "
        f"cutoff {summary_payload.get('cutoff') or summary_payload.get('train_end_ds', 'n/a')}"
    )
    ax_hist.grid(axis="y", alpha=0.25)
    ax_hist.legend(loc="upper right", fontsize=8)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        _summary_text(summary_payload),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_similarity_plot_set(
    summary_payload: dict[str, Any],
    windows_payload: dict[str, Any],
    *,
    out_dir: Path,
    stem: str,
    dpi: int = 150,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{stem}_similarity_raw_overlay.png"
    transformed_path = out_dir / f"{stem}_similarity_transformed_overlay.png"
    summary_path = out_dir / f"{stem}_similarity_summary.png"
    write_similarity_raw_overlay_plot(
        windows_payload,
        summary_payload,
        out_path=raw_path,
        dpi=dpi,
    )
    write_similarity_transformed_overlay_plot(
        windows_payload,
        summary_payload,
        out_path=transformed_path,
        dpi=dpi,
    )
    write_similarity_summary_plot(
        summary_payload,
        windows_payload,
        out_path=summary_path,
        dpi=dpi,
    )
    return {
        "raw_overlay": raw_path,
        "transformed_overlay": transformed_path,
        "summary": summary_path,
    }
