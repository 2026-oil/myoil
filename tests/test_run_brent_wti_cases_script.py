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
    assert "runner main=main.py config=ok.yaml job=PatchTST output_root=" in (log_dir / "ok.log").read_text(
        encoding="utf-8"
    )
    assert "runner main=main.py config=fail.yaml job=PatchTST output_root=" in (log_dir / "fail.log").read_text(
        encoding="utf-8"
    )

    summary_txt = (log_dir / "summary.txt").read_text(encoding="utf-8")
    assert "[batch] passed=1 failed=1 missing=1" in summary_txt


def test_run_brent_wti_cases_script_default_config_scope_uses_yaml_list_registration():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "yaml_list=(" in script
    assert '"yaml/feature_set/brentoil-case1.yaml"' in script
    assert '"yaml/feature_set/brentoil-case4.yaml"' in script
    assert '"yaml/feature_set/wti-case1.yaml"' in script
    assert '"yaml/feature_set/wti-case4.yaml"' in script
    assert '"yaml/feature_set_HPT_c3/brentoil-case3.yaml"' in script
    assert '"yaml/feature_set_HPT_c3/wti-case3.yaml"' in script
    assert 'configs=("${yaml_list[@]}")' in script
    assert "--jobs" in script
    assert "PatchTST" in script


def test_run_brent_wti_cases_script_does_not_auto_commit_or_push() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "NF_CASE_AUTO_GIT" not in script
    assert "NF_CASE_GIT_REMOTE" not in script
    assert "_commit_and_push_all_changes" not in script
    assert "git commit" not in script
    assert "git push" not in script


def test_run_brent_wti_cases_script_rejects_jobs_and_output_root_overrides(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--jobs", "NHITS"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "--jobs is managed by run.sh PatchTST mode" in completed.stderr

    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--output-root", "runs/custom"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "--output-root is managed by run.sh PatchTST mode" in completed.stderr
