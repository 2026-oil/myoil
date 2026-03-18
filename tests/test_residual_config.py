from __future__ import annotations

from pathlib import Path

import yaml

from residual.adapters import build_multivariate_inputs, build_univariate_inputs
from residual.config import load_app_config
from residual.models import build_model
from residual.scheduler import build_launch_plan, worker_env


def _write_config(tmp_path: Path, payload: dict, suffix: str) -> Path:
    path = tmp_path / f'config{suffix}'
    if suffix == '.toml':
        text = """
[dataset]
path = 'data.csv'
dt_col = 'dt'
freq = 'W-MON'

[runtime]
random_seed = 1

[training]
input_size = 64
season_length = 52
batch_size = 32
valid_batch_size = 32
windows_batch_size = 1024
inference_windows_batch_size = 1024
learning_rate = 0.001
max_steps = 50
loss = 'mse'

[cv]
horizon = 12
step_size = 4
n_windows = 24
final_holdout = 12
overlap_eval_policy = 'by_cutoff_mean'

[scheduler]
gpu_ids = [0, 1]
max_concurrent_jobs = 2
worker_devices = 1

[residual]
enabled = true
train_source = 'oof_cv'

[[jobs]]
name = 'u1'
model = 'TFT'
job_type = 'univariate_with_exog'
target_col = 'target'
hist_exog_cols = ['hist_a']
futr_exog_cols = []
static_exog_cols = []

[[jobs]]
name = 'm1'
model = 'iTransformer'
job_type = 'multivariate_channels'
target_col = 'target'
channel_cols = ['chan_b']
"""
        path.write_text(text, encoding='utf-8')
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return path


def _payload() -> dict:
    return {
        'dataset': {'path': 'data.csv', 'dt_col': 'dt', 'freq': 'W-MON'},
        'runtime': {'random_seed': 1},
        'training': {
            'input_size': 64,
            'season_length': 52,
            'batch_size': 32,
            'valid_batch_size': 32,
            'windows_batch_size': 1024,
            'inference_windows_batch_size': 1024,
            'learning_rate': 0.001,
            'max_steps': 50,
            'loss': 'mse',
        },
        'cv': {
            'horizon': 12,
            'step_size': 4,
            'n_windows': 24,
            'final_holdout': 12,
            'overlap_eval_policy': 'by_cutoff_mean',
        },
        'scheduler': {'gpu_ids': [0, 1], 'max_concurrent_jobs': 2, 'worker_devices': 1},
        'residual': {'enabled': True, 'train_source': 'oof_cv'},
        'jobs': [
            {
                'name': 'u1',
                'model': 'TFT',
                'job_type': 'univariate_with_exog',
                'target_col': 'target',
                'hist_exog_cols': ['hist_a'],
                'futr_exog_cols': [],
                'static_exog_cols': [],
            },
            {
                'name': 'm1',
                'model': 'iTransformer',
                'job_type': 'multivariate_channels',
                'target_col': 'target',
                'channel_cols': ['chan_b'],
            },
        ],
    }


def test_toml_and_yaml_normalize_to_same_typed_model(tmp_path: Path):
    (tmp_path / 'data.csv').write_text("dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding='utf-8')
    yaml_path = _write_config(tmp_path, _payload(), '.yaml')
    toml_path = _write_config(tmp_path, _payload(), '.toml')
    loaded_yaml = load_app_config(tmp_path, config_path=yaml_path)
    loaded_toml = load_app_config(tmp_path, config_toml_path=toml_path)
    assert loaded_yaml.config.to_dict() == loaded_toml.config.to_dict()
    assert loaded_yaml.config.training.loss == 'mse'
    assert loaded_toml.config.training.loss == 'mse'


def test_adapters_materialize_expected_frames(tmp_path: Path):
    payload = _payload()
    yaml_path = _write_config(tmp_path, payload, '.yaml')
    source_path = tmp_path / 'data.csv'
    source_path.write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,10,2\n2020-01-08,2,11,3\n",
        encoding='utf-8',
    )
    loaded = load_app_config(tmp_path, config_path=yaml_path)
    import pandas as pd

    source_df = pd.read_csv(source_path)
    univariate = build_univariate_inputs(source_df, loaded.config.jobs[0], dt_col='dt')
    multivariate = build_multivariate_inputs(source_df, loaded.config.jobs[1], dt_col='dt')
    assert list(univariate.fit_df.columns) == ['unique_id', 'ds', 'y', 'hist_a']
    assert multivariate.channel_map == {'target': 0, 'chan_b': 1}
    assert set(multivariate.fit_df['unique_id']) == {'target', 'chan_b'}


def test_model_builder_applies_common_loss_and_multivariate_n_series(tmp_path: Path):
    (tmp_path / 'data.csv').write_text("dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding='utf-8')
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, _payload(), '.yaml'))
    univariate_model = build_model(loaded.config, loaded.config.jobs[0])
    multivariate_model = build_model(loaded.config, loaded.config.jobs[1], n_series=2)
    assert type(univariate_model.loss).__name__ == 'MSE'
    assert type(multivariate_model.loss).__name__ == 'MSE'
    assert getattr(multivariate_model, 'n_series', 2) == 2


def test_scheduler_plan_and_worker_env_use_single_device(tmp_path: Path):
    (tmp_path / 'data.csv').write_text("dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding='utf-8')
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, _payload(), '.yaml'))
    launches = build_launch_plan(loaded.config, loaded.config.jobs)
    assert [launch.gpu_id for launch in launches] == [0, 1]
    assert all(launch.devices == 1 for launch in launches)
    env = worker_env(1)
    assert env['CUDA_VISIBLE_DEVICES'] == '1'
    assert env['NEURALFORECAST_WORKER_DEVICES'] == '1'


def test_runtime_executes_single_job_with_dummy_model(tmp_path: Path):
    payload = _payload()
    payload['cv'].update({'horizon': 1, 'step_size': 1, 'n_windows': 1, 'final_holdout': 1})
    payload['training'].update({'input_size': 1, 'max_steps': 1})
    payload['jobs'] = [
        {
            'name': 'dummy_uni',
            'model': 'DummyUnivariate',
            'job_type': 'univariate_with_exog',
            'target_col': 'target',
            'hist_exog_cols': [],
            'futr_exog_cols': [],
            'static_exog_cols': [],
            'params': {'start_padding_enabled': True},
        }
    ]
    data = (
        'dt,target\n'
        '2020-01-01,1\n'
        '2020-01-08,2\n'
        '2020-01-15,3\n'
        '2020-01-22,4\n'
        '2020-01-29,5\n'
        '2020-02-05,6\n'
        '2020-02-12,7\n'
    )
    (tmp_path / 'data.csv').write_text(data, encoding='utf-8')
    config_path = _write_config(tmp_path, payload, '.yaml')

    from residual.runtime import main as runtime_main

    output_root = tmp_path / 'run'
    code = runtime_main([
        '--config',
        str(config_path),
        '--jobs',
        'dummy_uni',
        '--output-root',
        str(output_root),
    ])
    assert code == 0
    assert (output_root / 'cv' / 'dummy_uni_forecasts.csv').exists()
    assert (output_root / 'holdout' / 'dummy_uni_metrics.csv').exists()
    fit_summary = (output_root / 'models' / 'dummy_uni' / 'fit_summary.json').read_text()
    assert '"loss": "mse"' in fit_summary
    assert '"devices": 1' in fit_summary

