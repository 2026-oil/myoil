from __future__ import annotations

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

from .config import AppConfig, JobConfig, LoadedConfig


@dataclass(frozen=True)
class WorkerLaunch:
    job_name: str
    gpu_id: int
    devices: int = 1


def build_launch_plan(config: AppConfig, jobs: Iterable[JobConfig]) -> list[WorkerLaunch]:
    gpu_ids = list(config.scheduler.gpu_ids)
    launches: list[WorkerLaunch] = []
    for index, job in enumerate(jobs):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        launches.append(WorkerLaunch(job_name=job.model, gpu_id=gpu_id, devices=1))
    return launches


def worker_env(gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['NEURALFORECAST_WORKER_DEVICES'] = '1'
    return env


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_command(
    entrypoint: Path,
    loaded: LoadedConfig,
    launch: WorkerLaunch,
    worker_output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(entrypoint),
        '--jobs',
        launch.job_name,
        '--output-root',
        str(worker_output_root),
    ]
    if loaded.source_type == 'toml':
        command.extend(['--config-toml', str(loaded.source_path)])
    else:
        command.extend(['--config', str(loaded.source_path)])
    return command


def run_parallel_jobs(
    repo_root: Path,
    loaded: LoadedConfig,
    launches: list[WorkerLaunch],
    scheduler_root: Path,
) -> list[dict[str, object]]:
    scheduler_root.mkdir(parents=True, exist_ok=True)
    workers_root = scheduler_root / 'workers'
    workers_root.mkdir(parents=True, exist_ok=True)
    entrypoint = repo_root / 'main.py'
    events_path = scheduler_root / 'events.jsonl'
    results: list[dict[str, object]] = []
    active: list[
        tuple[
            WorkerLaunch,
            subprocess.Popen[str],
            Path,
            Path,
            object,
            object,
            threading.Thread | None,
        ]
    ] = []
    max_concurrent = max(1, loaded.config.scheduler.max_concurrent_jobs)

    def _finalize_worker(
        launch: WorkerLaunch,
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
            'job_name': launch.job_name,
            'gpu_id': launch.gpu_id,
            'devices': launch.devices,
            'cuda_visible_devices': str(launch.gpu_id),
            'returncode': returncode,
            'stdout_path': str(stdout_path),
            'stderr_path': str(stderr_path),
            'completed_at': _now_iso(),
        }
        worker_root = workers_root / launch.job_name
        (worker_root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with events_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'event': 'worker_completed', **summary}) + '\n')
        results.append(summary)
        return summary

    launch_queue = list(launches)
    while launch_queue or active:
        while launch_queue and len(active) < max_concurrent:
            launch = launch_queue.pop(0)
            worker_root = workers_root / launch.job_name
            worker_root.mkdir(parents=True, exist_ok=True)
            stdout_path = worker_root / 'stdout.log'
            stderr_path = worker_root / 'stderr.log'
            env = worker_env(launch.gpu_id)
            command = _worker_command(entrypoint, loaded, launch, worker_root)
            with events_path.open('a', encoding='utf-8') as handle:
                handle.write(
                    json.dumps(
                        {
                            'event': 'worker_started',
                            'job_name': launch.job_name,
                            'gpu_id': launch.gpu_id,
                            'devices': launch.devices,
                            'started_at': _now_iso(),
                            'command': command,
                        }
                    )
                    + '\n'
                )
            stdout_handle = stdout_path.open('w', encoding='utf-8')
            stderr_handle = stderr_path.open('w', encoding='utf-8')
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
                    job_name: str,
                ) -> None:
                    for line in stream:
                        file_handle.write(line)
                        file_handle.flush()
                        print(f"[worker:{job_name}] {line}", end='', flush=True)
                    stream.close()

                stdout_thread = threading.Thread(
                    target=_pump_stdout,
                    args=(process.stdout, stdout_handle),
                    kwargs={"job_name": launch.job_name},
                    daemon=True,
                )
                stdout_thread.start()
            active.append(
                (
                    launch,
                    process,
                    stdout_path,
                    stderr_path,
                    stdout_handle,
                    stderr_handle,
                    stdout_thread,
                )
            )

        completed_idx = next(
            (index for index, (_, process, *_rest) in enumerate(active) if process.poll() is not None),
            None,
        )
        if completed_idx is not None:
            _finalize_worker(*active.pop(completed_idx))
            continue
        if active:
            time.sleep(1)
    return results
