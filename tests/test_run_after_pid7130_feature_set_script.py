from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "run_after_pid7130_feature_set.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("task:\n  name: demo\n", encoding="utf-8")


def test_run_after_pid7130_feature_set_waits_then_runs_expected_sequence(tmp_path: Path) -> None:
    feature_set_dir = tmp_path / "feature_set"
    hpt_dir = tmp_path / "feature_set_HPT_c3"
    alpha_cfg = feature_set_dir / "alpha.yaml"
    beta_cfg = feature_set_dir / "beta.yaml"
    brent_cfg = hpt_dir / "brentoil-case3.yaml"
    wti_cfg = hpt_dir / "wti-case3.yaml"

    for cfg in (alpha_cfg, beta_cfg, brent_cfg, wti_cfg):
        _make_config(cfg)

    fake_bin_dir = tmp_path / "bin"
    fake_bin_dir.mkdir()
    uv_path = fake_bin_dir / "uv"
    record_path = tmp_path / "uv_calls.txt"
    _write_executable(
        uv_path,
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> {record_path}
""",
    )

    watcher = subprocess.Popen(
        ["bash", "-lc", "sleep 1 & echo $!; wait"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    runner = None
    try:
        waited_pid = watcher.stdout.readline().strip()
        assert waited_pid

        env = os.environ.copy()
        env["PATH"] = f"{fake_bin_dir}:{env['PATH']}"
        env["NF_WAIT_CHAIN_WAIT_PID"] = waited_pid
        env["NF_WAIT_CHAIN_POLL_SECONDS"] = "0.05"
        env["NF_WAIT_CHAIN_FEATURE_SET_DIR"] = str(feature_set_dir)
        env["NF_WAIT_CHAIN_HPT_C3_DIR"] = str(hpt_dir)

        runner = subprocess.Popen(
            ["bash", str(SCRIPT_PATH), "--validate-only"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        assert runner.poll() is None
        assert not record_path.exists()

        stdout, stderr = runner.communicate(timeout=5)
    finally:
        if runner is not None and runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        watcher.wait(timeout=5)

    assert runner.returncode == 0, stderr
    assert f"[wait-chain] waiting for PID {waited_pid} to exit naturally" in stdout
    assert f"[wait-chain] PID {waited_pid} is gone; starting queued runs" in stdout

    commands = record_path.read_text(encoding="utf-8").splitlines()
    assert commands == [
        f"run main.py --config {wti_cfg} --validate-only",
        f"run main.py --config {brent_cfg} --jobs TSMixerx --validate-only",
        f"run main.py --config {alpha_cfg} --validate-only",
        f"run main.py --config {beta_cfg} --validate-only",
    ]


def test_run_after_pid7130_feature_set_script_defaults_are_scoped_correctly() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'NF_WAIT_CHAIN_WAIT_PID:-7130' in script
    assert 'yaml/feature_set' in script
    assert 'yaml/feature_set_HPT_c3' in script
    assert '--jobs TSMixerx' in script
    assert 'wti-case3.yaml' in script
    assert 'brentoil-case3.yaml' in script
    assert 'kill -' not in script
