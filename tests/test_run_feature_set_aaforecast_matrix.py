from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_feature_set_aaforecast_matrix_script_targets_expected_configs() -> None:
    script_path = REPO_ROOT / "run_feature_set_aaforecast.sh"

    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111

    script = script_path.read_text(encoding="utf-8")
    assert "yaml/experiment/feature_set_aaforecast/baseline.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-patchtst-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-timexer-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-patchtst.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-timexer.yaml" in script
    assert (
        'NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/feature_set_aaforecast_matrix}"'
        in script
    )
    assert 'NF_FEATURE_SET_RAW_ROOT="${NF_FEATURE_SET_RAW_ROOT:-runs/raw_feature_set_aaforecast}"' in script
    assert 'NF_FEATURE_SET_GRAPH_X_START="${NF_FEATURE_SET_GRAPH_X_START:-2025-08-15}"' in script
    assert 'NF_FEATURE_SET_GRAPH_X_END="${NF_FEATURE_SET_GRAPH_X_END:-2026-03-09}"' in script
    assert 'bash "$repo_root/run.sh" "$@" || run_status=$?' in script
    assert 'scripts/feature_set_aaforecast_postprocess.py' in script
