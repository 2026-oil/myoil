from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "watch_wti_case3_then_stop_run.sh"


def test_watch_wti_case3_then_stop_run_stops_target_process_after_completion(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "runs" / "_batch_logs" / "20260324T072251Z"
    log_dir.mkdir(parents=True)
    results_tsv = log_dir / "results.tsv"
    results_tsv.write_text(
        "config\tstatus\texit_code\tlog_path\tstarted_at\tfinished_at\n",
        encoding="utf-8",
    )

    marker_path = tmp_path / "signal_marker.txt"
    sleeper = subprocess.Popen(
        [
            "bash",
            "-lc",
            (
                f"trap 'echo INT >> {marker_path}; exit 0' INT; "
                f"trap 'echo TERM >> {marker_path}; exit 0' TERM; "
                "while true; do sleep 1; done"
            ),
        ],
        text=True,
        preexec_fn=os.setsid,
    )

    watcher = None
    try:
        env = os.environ.copy()
        env["NF_WTI_STOP_LOG_DIR"] = str(log_dir)
        env["NF_WTI_STOP_RUN_PID"] = str(sleeper.pid)
        env["NF_WTI_STOP_POLL_SECONDS"] = "0.05"
        env["NF_WTI_STOP_GRACE_SECONDS"] = "0.5"

        watcher = subprocess.Popen(
            ["bash", str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(0.2)
        assert watcher.poll() is None
        assert sleeper.poll() is None

        with results_tsv.open("a", encoding="utf-8") as handle:
            handle.write(
                "yaml/experiment/feature_set_HPT_n100_bs/wti-case3.yaml\tpassed\t0\t"
                "runs/_batch_logs/20260324T072251Z/wti-case3.log\t"
                "2026-03-24T07:22:51Z\t2026-03-24T10:22:51Z\n"
            )

        stdout, stderr = watcher.communicate(timeout=5)
        sleeper.wait(timeout=5)
    finally:
        if watcher is not None and watcher.poll() is None:
            watcher.kill()
            watcher.wait(timeout=5)
        if sleeper.poll() is None:
            os.killpg(sleeper.pid, signal.SIGKILL)
            sleeper.wait(timeout=5)

    assert watcher.returncode == 0, stderr
    assert "[watch-stop] detected yaml/experiment/feature_set_HPT_n100_bs/wti-case3.yaml completion with status=passed" in stdout
    assert f"[watch-stop] run_pid={sleeper.pid}" in stdout
    assert marker_path.read_text(encoding="utf-8").splitlines() == ["INT"]


def test_watch_wti_case3_then_stop_run_script_defaults_are_scoped_correctly() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "yaml/experiment/feature_set_HPT_n100_bs/wti-case3.yaml" in script
    assert 'runs/_batch_logs' in script
    assert 'NF_WTI_STOP_SIGNAL:-INT' in script
    assert 'NF_WTI_STOP_GRACE_SECONDS:-20' in script
    assert 'multiple run.sh PIDs found' in script
