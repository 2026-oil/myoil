from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import json
import os
import subprocess
import sys

from .config import AppConfig, JobConfig, LoadedConfig


@dataclass(frozen=True)
class WorkerLaunch:
    job_name: str
    gpu_id: int
    devices: int = 1


def build_launch_plan(config: AppConfig, jobs: Iterable[JobConfig]) -> list[WorkerLaunch]:
    gpu_ids = list(config.scheduler.gpu_ids)
    launches: list[WorkerLaunch] = []
    for index, job in enumerate(job for job in jobs if job.enabled):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        launches.append(WorkerLaunch(job_name=job.name, gpu_id=gpu_id, devices=1))
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
    active: list[tuple[WorkerLaunch, subprocess.Popen[str], Path, Path]] = []

    for launch in launches:
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
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        active.append((launch, process, stdout_path, stderr_path))

    for launch, process, stdout_path, stderr_path in active:
        returncode = process.wait()
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
    return results
