from __future__ import annotations

from residual.scheduler import worker_env


def test_worker_env_supports_scalar_gpu_id():
    env = worker_env(0)

    assert env['CUDA_VISIBLE_DEVICES'] == '0'
    assert env['NEURALFORECAST_ASSIGNED_GPU_IDS'] == '0'
    assert env['NEURALFORECAST_WORKER_DEVICES'] == '1'
    assert env['NEURALFORECAST_PROGRESS_MODE'] == 'structured'
    assert env['NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS'] == '1'
