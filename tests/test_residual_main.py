from __future__ import annotations

import os
from pathlib import Path

import main as residual_main


def test_build_env_preserves_workspace_on_pythonpath(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/tmp/example')
    env = residual_main._build_env()
    parts = env['PYTHONPATH'].split(os.pathsep)
    assert str(residual_main.WORKSPACE_ROOT) == parts[0]
    assert '/tmp/example' in parts


def test_exec_args_points_back_to_main_file():
    args = residual_main._exec_args(['--validate-only'])
    assert args[1] == str(Path(residual_main.__file__).resolve())
    assert args[-1] == '--validate-only'


def test_scheduler_worker_env_sets_single_visible_gpu():
    from residual.scheduler import worker_env

    env = worker_env(0)
    assert env['CUDA_VISIBLE_DEVICES'] == '0'
    assert env['NEURALFORECAST_WORKER_DEVICES'] == '1'
