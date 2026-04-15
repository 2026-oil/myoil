from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_feature_set_aaforecast_matrix_script_targets_expected_configs() -> None:
    script_path = REPO_ROOT / "run_feature_set_aaforecast_matrix.sh"

    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111

    script = script_path.read_text(encoding="utf-8")
    assert "yaml/experiment/feature_set_aaforecast/baseline.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml" in script
    assert (
        'NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/feature_set_aaforecast_matrix}"'
        in script
    )
    assert 'exec bash "$repo_root/run.sh" "$@"' in script
