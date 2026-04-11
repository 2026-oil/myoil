from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_brentoil_case1_aaforecast_ordering_script_targets_expected_configs() -> None:
    script_path = REPO_ROOT / "run_brentoil_case1_aaforecast_ordering.sh"

    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111

    script = script_path.read_text(encoding="utf-8")
    assert (
        "yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-informer.yaml"
        in script
    )
    assert "yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-gru.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/brentoil-case1-baseline.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-gru.yaml" not in script
    assert "yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml" not in script
    assert (
        'NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/brentoil_case1_aaforecast_ordering}"'
        in script
    )
    assert 'exec bash "$repo_root/run.sh" "$@"' in script
