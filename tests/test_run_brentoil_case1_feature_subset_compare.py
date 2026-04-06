from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_brentoil_case1_feature_subset_compare_script_targets_expected_configs() -> None:
    script_path = REPO_ROOT / "run_brentoil_case1_feature_subset_compare.sh"

    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111

    script = script_path.read_text(encoding="utf-8")
    assert "yaml/experiment/feature_set_aaforecast/brentoil-case1-best-all10.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/brentoil-case1-best-no_bs_core.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast/brentoil-case1-best-no_gprd.yaml" in script
    assert (
        "yaml/experiment/feature_set_nec/neciso_brent_hybrid_tsmixerx_lstm_inverse_all10.yaml"
        in script
    )
    assert (
        "yaml/experiment/feature_set_nec/neciso_brent_hybrid_tsmixerx_lstm_inverse_no_bs_core.yaml"
        in script
    )
    assert (
        "yaml/experiment/feature_set_nec/neciso_brent_hybrid_tsmixerx_lstm_inverse_no_gprd.yaml"
        in script
    )
    assert 'NF_CASE_LOG_ROOT="${NF_CASE_LOG_ROOT:-runs/_batch_logs/brentoil_case1_feature_subset_compare}"' in script
    assert 'exec bash "$repo_root/run.sh" "$@"' in script
