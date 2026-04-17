"""Tests for retrieval similarity window plotting helpers and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.retrieval import similarity_window_plot as mod
from scripts import plot_retrieval_similarity_windows as cli


def _summary_payload() -> dict:
    return {
        "bank_event_scores": [1.0, 2.0, 3.0, 5.0, 8.0],
        "query_event_score": 2.5,
        "effective_event_threshold": 1.8,
        "trigger_quantile": 0.6,
        "recency_gap_steps": 16,
        "min_similarity": 0.3,
        "retrieval_applied": True,
        "cutoff": "2026-02-23 00:00:00",
        "top_k_used": 1,
        "neighbors": [
            {
                "candidate_end_ds": "2020-05-11 00:00:00",
                "candidate_future_end_ds": "2020-05-25 00:00:00",
                "similarity": 1.83,
                "event_similarity": 0.86,
                "softmax_weight": 1.0,
                "event_score": 8.0,
                "anchor_target_value": 26.44,
                "future_returns": [0.25, 0.29],
            }
        ],
    }


def _windows_payload() -> dict:
    return {
        "train_end_ds": "2026-02-23 00:00:00",
        "input_size": 4,
        "horizon": 2,
        "target_col": "Com_CrudeOil",
        "query": {
            "ds": ["2026-02-02", "2026-02-09", "2026-02-16", "2026-02-23"],
            "y_raw": [60.0, 61.5, 63.0, 65.0],
            "y_transformed": [0.0, 1.5, 1.5, 2.0],
        },
        "neighbors": [
            {
                "rank": 1,
                "candidate_end_ds": "2020-05-11 00:00:00",
                "similarity": 1.83,
                "softmax_weight": 1.0,
                "ds": ["2020-04-20", "2020-04-27", "2020-05-04", "2020-05-11"],
                "y_raw": [20.0, 22.0, 25.0, 26.44],
                "y_transformed": [0.0, 2.0, 3.0, 1.44],
                "future_ds": ["2020-05-18", "2020-05-25"],
                "future_y_raw": [30.0, 34.0],
            }
        ],
    }


def test_write_similarity_plot_set_writes_pngs(tmp_path: Path) -> None:
    outputs = mod.write_similarity_plot_set(
        _summary_payload(),
        _windows_payload(),
        out_dir=tmp_path,
        stem="20260223T000000",
    )
    assert set(outputs) == {"raw_overlay", "transformed_overlay", "summary"}
    for path in outputs.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_write_similarity_plot_set_requires_transformed_values(tmp_path: Path) -> None:
    windows_payload = _windows_payload()
    windows_payload["query"]["y_transformed"] = [float("nan")] * 4
    with pytest.raises(ValueError, match="transformed"):
        mod.write_similarity_plot_set(
            _summary_payload(),
            windows_payload,
            out_dir=tmp_path,
            stem="bad",
        )


def test_cli_main_writes_default_pngs(tmp_path: Path) -> None:
    summary_path = tmp_path / "20260223T000000.json"
    windows_path = tmp_path / "20260223T000000_windows.json"
    summary_path.write_text(json.dumps(_summary_payload()), encoding="utf-8")
    windows_path.write_text(json.dumps(_windows_payload()), encoding="utf-8")

    rc = cli.main(["--json", str(summary_path)])

    assert rc == 0
    assert (tmp_path / "20260223T000000_similarity_raw_overlay.png").is_file()
    assert (tmp_path / "20260223T000000_similarity_transformed_overlay.png").is_file()
    assert (tmp_path / "20260223T000000_similarity_summary.png").is_file()
