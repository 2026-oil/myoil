from __future__ import annotations

from runtime_support.aaforecast_last_window_doc import _build_completeness_manifest, _render_markdown


def test_build_completeness_manifest_groups_tensor_names_by_stage() -> None:
    records = [
        {"stage": "raw_rows", "tensor_name": "raw.a", "ordering": 1},
        {"stage": "raw_rows", "tensor_name": "raw.b", "ordering": 2},
        {"stage": "forecast", "tensor_name": "forecast.final", "ordering": 3},
    ]

    manifest = _build_completeness_manifest(records)

    assert manifest == {
        "raw_rows": ["raw.a", "raw.b"],
        "forecast": ["forecast.final"],
    }


def test_render_markdown_includes_provenance_and_inline_payloads() -> None:
    payload = {
        "provenance": {
            "identity_lock_passed": False,
            "retrieval_enabled": False,
            "fitted_state_source": "runs/example.ckpt",
            "archived_horizon": 6,
            "current_replay_horizon": 2,
            "archived_final_prediction_level": [1.0, 2.0],
            "current_replay_deterministic_prediction_level": [1.1, 2.1],
            "current_replay_selected_mean_level": [1.2, 2.2],
            "selected_mean_abs_diff_vs_archived": None,
            "blocker": "example blocker",
        },
        "archival": {
            "archived_actual": [3.0, 4.0],
            "archived_final": [1.0, 2.0],
            "archived_std": [0.1, 0.2],
            "archived_selected_dropout": [0.03, 0.03],
            "context_json": {"context_active": True},
        },
        "records": [
            {
                "stage": "forecast",
                "tensor_name": "forecast.final_prediction_level",
                "shape": [2],
                "dtype": "float64",
                "meaning": "Example forecast",
                "payload": [1.2, 2.2],
                "ordering": 1,
            }
        ],
        "completeness_manifest": {"forecast": ["forecast.final_prediction_level"]},
    }

    markdown = _render_markdown(payload)

    assert "identity_lock_passed: **False**" in markdown
    assert "archived_horizon: `6`" in markdown
    assert "current_replay_horizon: `2`" in markdown
    assert "forecast.final_prediction_level" in markdown
    assert "example blocker" in markdown
    assert '"forecast": [' in markdown
    assert "```json" in markdown
