from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "run_txer.sh"
CONFIG_SOURCE_PATH = SCRIPT_PATH


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_run_txer_script_forces_timexer_job(tmp_path: Path) -> None:
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
  shift
done
echo "runner main=${main_py##*/} config=${config##*/} job=${job}"
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
    assert summary_json["extra_args"] == ["--jobs", "TimeXer", "--validate-only"]

    assert "runner main=main.py config=ok.yaml job=TimeXer" in (log_dir / "ok.log").read_text(
        encoding="utf-8"
    )
    assert "runner main=main.py config=fail.yaml job=TimeXer" in (log_dir / "fail.log").read_text(
        encoding="utf-8"
    )


def test_run_txer_script_sources_timexer_job_from_wrapper() -> None:
    script = CONFIG_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'run.sh" --jobs TimeXer "$@"' in script
