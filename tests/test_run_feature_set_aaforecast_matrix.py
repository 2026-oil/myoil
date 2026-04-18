from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_feature_set_aaforecast_matrix_script_targets_expected_configs() -> None:
    script_path = REPO_ROOT / "run_feature_set_aaforecast.sh"

    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111

    script = script_path.read_text(encoding="utf-8")
    assert 'case_selector="${TARGET_CASE:-}"' in script
    assert 'all|dubai|wti|brent' in script
    assert 'target_cases=(wti brent dubai)' in script
    assert 'local config_dir="yaml/experiment/feature_set_aaforecast_${target_case}"' in script
    assert 'local run_label="feature_set_aaforecast_${target_case}_matrix"' in script
    assert '"${config_dir}/baseline.yaml"' not in script
    assert '"${config_dir}/baseline-ret.yaml"' in script
    assert '"${config_dir}/aaforecast-informer-ret.yaml"' in script
    assert '"${config_dir}/aaforecast-gru-ret.yaml"' in script
    assert '"${config_dir}/aaforecast-timexer-ret.yaml"' in script
    assert 'case_log_root_default="runs/_batch_logs/${run_label}"' in script
    assert 'case_raw_root_default="runs/raw_feature_set_aaforecast_${target_case}"' in script
    assert 'case_log_root="${NF_CASE_LOG_ROOT}/${target_case}"' in script
    assert 'case_raw_root="${NF_FEATURE_SET_RAW_ROOT}/${target_case}"' in script
    assert 'graph_x_start="${NF_FEATURE_SET_GRAPH_X_START:-2025-08-15}"' in script
    assert 'graph_x_end="${NF_FEATURE_SET_GRAPH_X_END:-2026-03-09}"' in script
    assert 'bash "$repo_root/run.sh" "$@" || run_status=$?' in script
    assert 'scripts/feature_set_aaforecast_postprocess.py' in script
    assert 'for target_case in "${target_cases[@]}"; do' in script
