from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_feature_set_aaforecast_bs_compare_matrix_script_exists() -> None:
    script_path = REPO_ROOT / "run_feature_set_aaforecast_bs_compare.sh"

    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111


def test_run_feature_set_aaforecast_bs_compare_matrix_script_targets_expected_configs() -> None:
    script_path = REPO_ROOT / "run_feature_set_aaforecast_bs_compare.sh"
    script = script_path.read_text(encoding="utf-8")

    assert "yaml/experiment/feature_set_aaforecast_YES_BS/baseline.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_YES_BS/baseline-ret.yaml" in script
    assert (
        "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-informer-ret.yaml"
        in script
    )
    assert "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-gru-ret.yaml" in script
    assert (
        "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-patchtst-ret.yaml"
        in script
    )
    assert "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-timexer-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-informer.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-gru.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-patchtst.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_YES_BS/aaforecast-timexer.yaml" in script

    assert "yaml/experiment/feature_set_aaforecast_NO_BS/baseline.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_NO_BS/baseline-ret.yaml" in script
    assert (
        "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-informer-ret.yaml"
        in script
    )
    assert "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-gru-ret.yaml" in script
    assert (
        "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-patchtst-ret.yaml"
        in script
    )
    assert "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-timexer-ret.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-informer.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-gru.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-patchtst.yaml" in script
    assert "yaml/experiment/feature_set_aaforecast_NO_BS/aaforecast-timexer.yaml" in script

    assert "runs/_batch_logs/feature_set_aaforecast_bs_compare_matrix" in script
    assert "runs/raw_feature_set_aaforecast_bs_compare" in script
    assert 'bash "$repo_root/run.sh" "$@" || run_status=$?' in script
    assert "scripts/feature_set_aaforecast_postprocess.py" in script

