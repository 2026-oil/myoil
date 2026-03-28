from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "run.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_run_brent_wti_cases_script_writes_logs_and_summary(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    ok_cfg = configs_dir / "ok.yaml"
    fail_cfg = configs_dir / "fail.yaml"
    ok_cfg.write_text("task:\n  name: ok\n", encoding="utf-8")
    fail_cfg.write_text("task:\n  name: fail\n", encoding="utf-8")

    runner = tmp_path / "fake_python.sh"
    _write_executable(
        runner,
        """#!/usr/bin/env bash
set -euo pipefail
main_py="$1"
shift
config=""
job=""
output_root=""
while (($#)); do
  if [[ "$1" == "--config" ]]; then
    config="$2"
    shift 2
    continue
  fi
  if [[ "$1" == "--jobs" ]]; then
    job="$2"
    shift 2
    continue
  fi
  if [[ "$1" == "--output-root" ]]; then
    output_root="$2"
    shift 2
    continue
  fi
  shift
done
echo "runner main=${main_py##*/} config=${config##*/} job=${job} output_root=${output_root}"
if [[ "${config##*/}" == "fail.yaml" ]]; then
  exit 7
fi
""",
    )

    env = os.environ.copy()
    missing_cfg = configs_dir / "missing.yaml"
    env["NF_CASE_CONFIGS"] = f"{ok_cfg}\n{fail_cfg}\n{missing_cfg}\n"
    env["NF_CASE_PYTHON_BIN"] = str(runner)
    env["NF_CASE_LOG_ROOT"] = str(tmp_path / "logs")
    env["NF_CASE_GLOBAL_GPU_QUEUE"] = "0"

    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--validate-only"],
        cwd=configs_dir,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert f"[batch] ok: {ok_cfg}" in completed.stdout
    assert f"[batch] fail: {fail_cfg} (exit=7" in completed.stdout
    assert f"[batch] missing config: {missing_cfg}" in completed.stdout

    log_roots = sorted((tmp_path / "logs").iterdir())
    assert len(log_roots) == 1
    log_dir = log_roots[0]

    summary_json = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["passed"] == [str(ok_cfg)]
    assert summary_json["failed"] == [f"{fail_cfg} (exit=7)"]
    assert summary_json["missing"] == [str(missing_cfg)]
    assert summary_json["extra_args"] == ["--validate-only"]

    results = {row["config"]: row for row in summary_json["results"]}
    assert results[str(ok_cfg)]["status"] == "passed"
    assert results[str(fail_cfg)]["status"] == "failed"
    assert results[str(fail_cfg)]["exit_code"] == "7"
    assert results[str(missing_cfg)]["status"] == "missing"

    assert (log_dir / "ok.log").exists()
    assert (log_dir / "fail.log").exists()
    assert "runner main=main.py config=ok.yaml job= output_root=" in (log_dir / "ok.log").read_text(
        encoding="utf-8"
    )
    assert "runner main=main.py config=fail.yaml job= output_root=" in (log_dir / "fail.log").read_text(
        encoding="utf-8"
    )

    summary_txt = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "[batch] passed=1 failed=1 missing=1" in summary_txt


def test_run_brent_wti_cases_script_uses_duration_aware_global_gpu_queue(
    tmp_path: Path,
) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    fast_cfg = configs_dir / "fast.yaml"
    medium_cfg = configs_dir / "medium.yaml"
    slow_cfg = configs_dir / "slow.yaml"
    for path in (fast_cfg, medium_cfg, slow_cfg):
        path.write_text("task:\n  name: demo\n", encoding="utf-8")

    runner = tmp_path / "fake_python.sh"
    _write_executable(
        runner,
        """#!/usr/bin/env bash
set -euo pipefail
config=""
while (($#)); do
  if [[ "$1" == "--config" ]]; then
    config="$2"
    shift 2
    continue
  fi
  shift
done
echo "cfg=${config##*/} cuda=${CUDA_VISIBLE_DEVICES:-} assigned=${NEURALFORECAST_ASSIGNED_GPU_IDS:-}"
case "${config##*/}" in
  slow.yaml) sleep 0.4 ;;
  medium.yaml) sleep 0.2 ;;
  fast.yaml) sleep 0.05 ;;
esac
""",
    )

    log_root = tmp_path / "logs"
    history_dir = log_root / "20260323T000000Z"
    history_dir.mkdir(parents=True)
    history_payload = {
        "results": [
            {
                "config": str(fast_cfg),
                "status": "passed",
                "exit_code": "0",
                "started_at": "2026-03-23T00:00:00Z",
                "finished_at": "2026-03-23T00:00:05Z",
            },
            {
                "config": str(medium_cfg),
                "status": "passed",
                "exit_code": "0",
                "started_at": "2026-03-23T00:00:00Z",
                "finished_at": "2026-03-23T00:00:30Z",
            },
            {
                "config": str(slow_cfg),
                "status": "passed",
                "exit_code": "0",
                "started_at": "2026-03-23T00:00:00Z",
                "finished_at": "2026-03-23T00:01:00Z",
            },
        ]
    }
    (history_dir / "summary.json").write_text(
        json.dumps(history_payload), encoding="utf-8"
    )

    env = os.environ.copy()
    env["NF_CASE_CONFIGS"] = f"{fast_cfg}\n{medium_cfg}\n{slow_cfg}\n"
    env["NF_CASE_PYTHON_BIN"] = str(runner)
    env["NF_CASE_LOG_ROOT"] = str(log_root)
    env["NF_CASE_GPU_IDS"] = "0,1"
    env["NF_CASE_USE_PTY"] = "0"

    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--validate-only"],
        cwd=configs_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert completed.returncode == 0
    slow_start = f"[batch] start[gpu0] worker_devices=1 estimate_s=60.0: {slow_cfg}"
    medium_start = f"[batch] start[gpu1] worker_devices=1 estimate_s=30.0: {medium_cfg}"
    fast_start = f"[batch] start[gpu1] worker_devices=1 estimate_s=5.0: {fast_cfg}"
    assert slow_start in completed.stdout
    assert medium_start in completed.stdout
    assert fast_start in completed.stdout
    assert completed.stdout.index(slow_start) < completed.stdout.index(medium_start)
    assert completed.stdout.index(medium_start) < completed.stdout.index(fast_start)

    log_dir = sorted(log_root.iterdir())[-1]
    assert "cfg=slow.yaml cuda=0 assigned=0" in (log_dir / "slow.log").read_text(
        encoding="utf-8"
    )
    assert "cfg=medium.yaml cuda=1 assigned=1" in (log_dir / "medium.log").read_text(
        encoding="utf-8"
    )
    assert "cfg=fast.yaml cuda=1 assigned=1" in (log_dir / "fast.log").read_text(
        encoding="utf-8"
    )


def test_run_brent_wti_cases_script_assigns_multi_device_groups(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    dual_a_cfg = configs_dir / "dual-a.yaml"
    dual_b_cfg = configs_dir / "dual-b.yaml"
    dual_payload = "task:\n  name: demo\nscheduler:\n  worker_devices: 2\n"
    dual_a_cfg.write_text(dual_payload, encoding="utf-8")
    dual_b_cfg.write_text(dual_payload, encoding="utf-8")

    runner = tmp_path / "fake_python.sh"
    _write_executable(
        runner,
        """#!/usr/bin/env bash
set -euo pipefail
config=""
while (($#)); do
  if [[ "$1" == "--config" ]]; then
    config="$2"
    shift 2
    continue
  fi
  shift
done
echo "cfg=${config##*/} cuda=${CUDA_VISIBLE_DEVICES:-} assigned=${NEURALFORECAST_ASSIGNED_GPU_IDS:-}"
sleep 0.05
""",
    )

    env = os.environ.copy()
    env["NF_CASE_CONFIGS"] = f"{dual_a_cfg}\n{dual_b_cfg}\n"
    env["NF_CASE_PYTHON_BIN"] = str(runner)
    env["NF_CASE_LOG_ROOT"] = str(tmp_path / "logs")
    env["NF_CASE_GPU_IDS"] = "0,1,2,3"
    env["NF_CASE_USE_PTY"] = "0"

    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--validate-only"],
        cwd=configs_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert completed.returncode == 0
    assert (
        f"[batch] start[gpu0,1] worker_devices=2 estimate_s=0.0: {dual_a_cfg}"
        in completed.stdout
    )
    assert (
        f"[batch] start[gpu2,3] worker_devices=2 estimate_s=0.0: {dual_b_cfg}"
        in completed.stdout
    )

    log_dir = sorted(log_root for log_root in (tmp_path / "logs").iterdir())[-1]
    assert "cfg=dual-a.yaml cuda=0,1 assigned=0,1" in (
        log_dir / "dual-a.log"
    ).read_text(encoding="utf-8")
    assert "cfg=dual-b.yaml cuda=2,3 assigned=2,3" in (
        log_dir / "dual-b.log"
    ).read_text(encoding="utf-8")


def test_run_brent_wti_cases_script_default_config_scope_uses_yaml_list_registration():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "yaml_list=(" in script
    assert script.count('"yaml/') == 20
    assert '"yaml/experiment/feature_set_HPT_n100_bs/brentoil-case1.yaml"' not in script
    assert '"yaml/experiment/feature_set_HPT_n100_bs/wti-case1.yaml"' not in script
    assert '"yaml/experiment/feature_set_HPT_n100_bs/wti-case4.yaml"' in script
    assert '"yaml/experiment/feature_set_residual_bs_HPT/brentoil-case1.yaml"' not in script
    assert '"yaml/experiment/feature_set_residual_bs_HPT/wti-case1.yaml"' not in script
    assert '"yaml/experiment/feature_set_residual_bs_HPT/wti-case4.yaml"' in script
    assert '"yaml/experiment/feature_set_residual/brentoil-case3.yaml"' in script
    assert '"yaml/experiment/feature_set_residual/wti-case3.yaml"' in script
    assert '"yaml/experiment/feature_set_residual_bs/brentoil-case1.yaml"' not in script
    assert '"yaml/experiment/feature_set_residual_bs/wti-case1.yaml"' not in script
    assert '"yaml/experiment/feature_set_residual_bs/wti-case4.yaml"' in script
    assert script.index('"yaml/experiment/feature_set_HPT_n100_bs/brentoil-case2.yaml"') < script.index(
        '"yaml/experiment/feature_set_residual_bs_HPT/brentoil-case2.yaml"'
    )
    assert script.index('"yaml/experiment/feature_set_residual_bs_HPT/brentoil-case2.yaml"') < script.index(
        '"yaml/experiment/feature_set_residual/brentoil-case3.yaml"'
    )
    assert script.index('"yaml/experiment/feature_set_residual/brentoil-case3.yaml"') < script.index(
        '"yaml/experiment/feature_set_residual_bs/brentoil-case2.yaml"'
    )
    assert 'configs=("${yaml_list[@]}")' in script
    assert '--config" "$cfg" "$@' in script
    assert "NF_CASE_PATCHTST_JOB" not in script


def test_run_brent_wti_cases_script_does_not_auto_commit_or_push() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "NF_CASE_AUTO_GIT" not in script
    assert "NF_CASE_GIT_REMOTE" not in script
    assert "_commit_and_push_all_changes" not in script
    assert "git commit" not in script
    assert "git push" not in script


def test_run_brent_wti_cases_script_allows_jobs_override(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    ok_cfg = configs_dir / "ok.yaml"
    ok_cfg.write_text("task:\n  name: ok\n", encoding="utf-8")

    runner = tmp_path / "fake_python.sh"
    _write_executable(
        runner,
        """#!/usr/bin/env bash
set -euo pipefail
job=""
while (($#)); do
  if [[ "$1" == "--jobs" ]]; then
    job="$2"
    shift 2
    continue
  fi
  shift
done
echo "job=${job}"
""",
    )

    env = os.environ.copy()
    env["NF_CASE_CONFIGS"] = str(ok_cfg)
    env["NF_CASE_PYTHON_BIN"] = str(runner)
    env["NF_CASE_LOG_ROOT"] = str(tmp_path / "logs")

    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--jobs", "NHITS"],
        cwd=configs_dir,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    log_dir = next((tmp_path / "logs").iterdir())
    assert "job=NHITS" in (log_dir / "ok.log").read_text(encoding="utf-8")


def test_run_brent_wti_cases_script_rejects_output_root_override(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--output-root", "runs/custom"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "--output-root is managed by run.sh; do not pass it explicitly" in completed.stderr


def test_run_brent_wti_cases_script_rejects_invalid_global_gpu_queue_flag(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["NF_CASE_GLOBAL_GPU_QUEUE"] = "maybe"

    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2

    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--output-root=runs/custom"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "--output-root is managed by run.sh; do not pass it explicitly" in completed.stderr
