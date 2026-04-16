"""Tests for ``plugins.retrieval.event_score_distribution_plot``."""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.retrieval import event_score_distribution_plot as mod


def test_write_event_score_distribution_plot_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "dist.png"
    mod.write_event_score_distribution_plot(
        {
            "bank_event_scores": [1.0, 2.0, 3.0, 10.0, 11.0],
            "query_event_score": 2.5,
            "effective_event_threshold": 5.0,
            "trigger_quantile": 0.6,
            "retrieval_applied": False,
            "skip_reason": "below_event_threshold",
            "cutoff": "2020-01-01",
        },
        out_path=out,
    )
    assert out.is_file()
    assert out.stat().st_size > 0


def test_write_event_score_distribution_plot_empty_bank(tmp_path: Path) -> None:
    out = tmp_path / "empty.png"
    mod.write_event_score_distribution_plot(
        {
            "bank_event_scores": [],
            "query_event_score": 1.0,
            "effective_event_threshold": 0.0,
            "retrieval_applied": False,
            "skip_reason": "empty_bank",
        },
        out_path=out,
    )
    assert out.is_file()


def test_write_event_score_distribution_plot_requires_bank_scores(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="bank_event_scores"):
        mod.write_event_score_distribution_plot(
            {"query_event_score": 1.0, "effective_event_threshold": 0.5},
            out_path=tmp_path / "missing.png",
        )
