from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import json
import os
import subprocess
import sys
import threading
import time

from app_config import AppConfig, JobConfig, LoadedConfig
from runtime_support.progress import ConsoleProgressRenderer, ModelProgressState, parse_progress_event

_ALLOW_INTERNAL_OUTPUT_ROOT_ENV = "NEURALFORECAST_ALLOW_INTERNAL_OUTPUT_ROOT"


@dataclass(frozen=True)
class WorkerLaunch:
    job_name: str
    devices: int = 1
    phase: str = "full"
    worker_index: int = 0
    selected_study: int | None = None


def build_device_groups(config: AppConfig) -> list[tuple[int, ...]]:
    assigned_gpu_ids = os.environ.get("NEURALFORECAST_ASSIGNED_GPU_IDS", "").strip()
    if assigned_gpu_ids:
        gpu_ids = [
            int(part.strip())
            for part in assigned_gpu_ids.split(",")
            if part.strip()
        ]
    else:
        gpu_ids = list(config.scheduler.gpu_ids)
    group_size = max(1, int(config.scheduler.worker_devices))
    groups: list[tuple[int, ...]] = []
    for start in range(0, len(gpu_ids), group_size):
        group = tuple(gpu_ids[start : start + group_size])
        if len(group) == group_size:
            groups.append(group)
    if not groups:
        raise ValueError("scheduler.gpu_ids must define at least one full worker device group")
    return groups


def build_launch_plan(
    config: AppConfig,
    jobs: Iterable[JobConfig],
    *,
    selected_study: int | None = None,
) -> list[WorkerLaunch]:
    return [
        WorkerLaunch(
            job_name=job.model,
            devices=int(config.scheduler.worker_devices),
            selected_study=selected_study,
        )
        for job in jobs
    ]


def build_tuning_launch_plan(
    config: AppConfig,
    *,
    job_name: str,
    worker_count: int | None = None,
    selected_study: int | None = None,
) -> list[WorkerLaunch]:
    max_workers = len(build_device_groups(config))
    requested = max_workers if worker_count is None else max(1, int(worker_count))
    return [
        WorkerLaunch(
            job_name=job_name,
            devices=int(config.scheduler.worker_devices),
            phase="tune-main-only",
            worker_index=index,
            selected_study=selected_study,
        )
        for index in range(min(max_workers, requested))
    ]


def worker_env(gpu_ids: int | Sequence[int]) -> dict[str, str]:
    if isinstance(gpu_ids, int):
        assigned_gpu_ids = (gpu_ids,)
    else:
        assigned_gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
    env = os.environ.copy()
    if assigned_gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in assigned_gpu_ids)
        env["NEURALFORECAST_ASSIGNED_GPU_IDS"] = ",".join(
            str(gpu_id) for gpu_id in assigned_gpu_ids
        )
    env["NEURALFORECAST_WORKER_DEVICES"] = str(len(assigned_gpu_ids))
    env["NEURALFORECAST_PROGRESS_MODE"] = "structured"
    env["NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS"] = "1"
    env[_ALLOW_INTERNAL_OUTPUT_ROOT_ENV] = "1"
    return env


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_command(
    entrypoint: Path,
    loaded: LoadedConfig,
    launch: WorkerLaunch,
    worker_output_root: Path,
) -> list[str]:
    command_output_root = (
        worker_output_root.parents[2]
        if launch.phase == "tune-main-only"
        else worker_output_root
    )
    command = [
        sys.executable,
        str(entrypoint),
        "--jobs",
        launch.job_name,
        "--output-root",
        str(command_output_root),
    ]
    if loaded.active_jobs_route_slug:
        command.extend(["--internal-jobs-route", loaded.active_jobs_route_slug])
    if launch.phase != "full":
        command.extend(["--internal-stage", launch.phase])
    if launch.selected_study is not None:
        command.extend(["--optuna-study", str(launch.selected_study)])
    if loaded.source_type == "toml":
        command.extend(["--config-toml", str(loaded.source_path)])
    else:
        command.extend(["--config", str(loaded.source_path)])
    if loaded.shared_settings_path is not None:
        command.extend(["--setting", str(loaded.shared_settings_path)])
    return command


def run_parallel_jobs(
    repo_root: Path,
    loaded: LoadedConfig,
    launches: list[WorkerLaunch],
    scheduler_root: Path,
) -> list[dict[str, object]]:
    scheduler_root.mkdir(parents=True, exist_ok=True)
    workers_root = scheduler_root / "workers"
    workers_root.mkdir(parents=True, exist_ok=True)
    entrypoint = repo_root / "main.py"
    events_path = scheduler_root / "events.jsonl"
    results: list[dict[str, object]] = []
    progress_states = {
        f"{launch.job_name}#{launch.worker_index}" if launch.phase != "full" else launch.job_name: ModelProgressState(
            job_name=f"{launch.job_name}#{launch.worker_index}" if launch.phase != "full" else launch.job_name,
            model_index=index + 1,
            total_models=len(launches),
            total_steps=1,
        )
        for index, launch in enumerate(launches)
    }
    progress_renderer = ConsoleProgressRenderer()
    render_lock = threading.Lock()
    available_groups = build_device_groups(loaded.config)

    def _render_progress() -> None:
        with render_lock:
            progress_renderer.render(list(progress_states.values()))

    active: list[
        tuple[
            WorkerLaunch,
            tuple[int, ...],
            subprocess.Popen[str],
            Path,
            Path,
            object,
            object,
            threading.Thread | None,
        ]
    ] = []
    max_concurrent = min(
        max(1, loaded.config.scheduler.max_concurrent_jobs), len(available_groups)
    )

    def _progress_key(launch: WorkerLaunch) -> str:
        if launch.phase == "full":
            return launch.job_name
        return f"{launch.job_name}#{launch.worker_index}"

    def _finalize_worker(
        launch: WorkerLaunch,
        assigned_gpu_ids: tuple[int, ...],
        process: subprocess.Popen[str],
        stdout_path: Path,
        stderr_path: Path,
        stdout_handle,
        stderr_handle,
        stdout_thread: threading.Thread | None,
    ) -> dict[str, object]:
        returncode = process.wait()
        if stdout_thread is not None:
            stdout_thread.join()
        stdout_handle.close()
        stderr_handle.close()
        summary = {
            "job_name": launch.job_name,
            "phase": launch.phase,
            "worker_index": launch.worker_index,
            "selected_study": launch.selected_study,
            "assigned_gpu_ids": list(assigned_gpu_ids),
            "devices": launch.devices,
            "cuda_visible_devices": ",".join(str(gpu_id) for gpu_id in assigned_gpu_ids),
            "returncode": returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "completed_at": _now_iso(),
        }
        worker_root = workers_root / _progress_key(launch)
        worker_root.mkdir(parents=True, exist_ok=True)
        (worker_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "worker_completed", **summary}) + "\n")
        state = progress_states[_progress_key(launch)]
        state.status = "completed" if returncode == 0 else "failed"
        state.detail = f"returncode={returncode} gpu_ids={summary['assigned_gpu_ids']}"
        _render_progress()
        results.append(summary)
        available_groups.append(assigned_gpu_ids)
        return summary

    launch_queue = list(launches)
    while launch_queue or active:
        while launch_queue and len(active) < max_concurrent and available_groups:
            launch = launch_queue.pop(0)
            assigned_gpu_ids = available_groups.pop(0)
            worker_root = workers_root / _progress_key(launch)
            worker_root.mkdir(parents=True, exist_ok=True)
            stdout_path = worker_root / "stdout.log"
            stderr_path = worker_root / "stderr.log"
            env = worker_env(assigned_gpu_ids)
            env["NEURALFORECAST_OPTUNA_WORKER_INDEX"] = str(launch.worker_index)
            command = _worker_command(entrypoint, loaded, launch, worker_root)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "worker_started",
                            "job_name": launch.job_name,
                            "phase": launch.phase,
                            "worker_index": launch.worker_index,
                            "selected_study": launch.selected_study,
                            "assigned_gpu_ids": list(assigned_gpu_ids),
                            "devices": launch.devices,
                            "started_at": _now_iso(),
                            "command": command,
                        }
                    )
                    + "\n"
                )
            progress_states[_progress_key(launch)].status = "running"
            progress_states[_progress_key(launch)].detail = (
                f"gpu_ids={','.join(str(gpu_id) for gpu_id in assigned_gpu_ids)}"
            )
            _render_progress()
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                text=True,
                bufsize=1,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
            )
            stdout_thread = None
            if process.stdout is not None:

                def _pump_stdout(
                    stream,
                    file_handle,
                    *,
                    progress_key: str,
                ) -> None:
                    for line in stream:
                        file_handle.write(line)
                        file_handle.flush()
                        payload = parse_progress_event(line)
                        if payload is not None:
                            state = progress_states[progress_key]
                            state.total_steps = max(int(payload["total_steps"]), 1)
                            state.completed_steps = int(payload["completed_steps"])
                            state.total_folds = payload.get("total_folds")
                            state.current_fold = payload.get("current_fold")
                            state.phase = payload.get("phase")
                            state.status = payload.get("status", state.status)
                            state.detail = payload.get("detail")
                            state.event = payload.get("event", state.event)
                            _render_progress()
                            continue
                        print(f"[worker:{progress_key}] {line}", end="", flush=True)
                    stream.close()

                stdout_thread = threading.Thread(
                    target=_pump_stdout,
                    args=(process.stdout, stdout_handle),
                    kwargs={"progress_key": _progress_key(launch)},
                    daemon=True,
                )
                stdout_thread.start()
            active.append(
                (
                    launch,
                    assigned_gpu_ids,
                    process,
                    stdout_path,
                    stderr_path,
                    stdout_handle,
                    stderr_handle,
                    stdout_thread,
                )
            )

        completed_idx = next(
            (
                index
                for index, (_, _gpu_ids, process, *_rest) in enumerate(active)
                if process.poll() is not None
            ),
            None,
        )
        if completed_idx is not None:
            _finalize_worker(*active.pop(completed_idx))
            continue
        if active:
            time.sleep(1)
    with render_lock:
        progress_renderer.close()
    return results
