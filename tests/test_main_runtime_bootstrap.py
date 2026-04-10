from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from app_config import load_app_config
import main as bootstrap_main


def _minimal_app_config_payload() -> dict[str, object]:
    return {
        'dataset': {
            'path': str(
                (
                    bootstrap_main.WORKSPACE_ROOT
                    / 'tests/fixtures/optuna_smoke_data.csv'
                ).resolve()
            ),
            'target_col': 'target',
            'dt_col': 'dt',
        },
        'jobs': [{'model': 'Naive', 'params': {}}],
    }


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return path


@pytest.fixture(autouse=True)
def runtime_runner_stub(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    import runtime_support

    stub = ModuleType('runtime_support.runner')
    stub.load_app_config = lambda *args, **kwargs: SimpleNamespace(
        jobs_fanout_specs=[]
    )
    stub.run_loaded_config = lambda *args, **kwargs: {'ok': True}
    stub.loaded_config_for_jobs_fanout = lambda repo_root, loaded, spec: loaded
    stub.main = lambda argv=None: bootstrap_main._run_cli(
        argv, repo_root=Path(__file__).resolve().parents[1]
    )
    monkeypatch.setitem(sys.modules, 'runtime_support.runner', stub)
    monkeypatch.setattr(runtime_support, 'runner', stub, raising=False)
    return stub


def test_main_loads_config_and_dispatches_without_reexec_and_preserves_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
    runtime_runner_stub: ModuleType,
):
    calls: dict[str, object] = {}
    workspace_root = str(bootstrap_main.WORKSPACE_ROOT)
    monkeypatch.setenv(bootstrap_main._BOOTSTRAP_ENV, '1')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([workspace_root, '/tmp/example']))

    loaded = SimpleNamespace(jobs_fanout_specs=[])

    def fake_load_app_config(
        repo_root: Path,
        *,
        config_path=None,
        config_toml_path=None,
        shared_settings_path=None,
    ):
        calls['load_repo_root'] = repo_root
        calls['config_path'] = config_path
        calls['config_toml_path'] = config_toml_path
        calls['shared_settings_path'] = shared_settings_path
        return loaded

    def fake_run_loaded_config(repo_root: Path, loaded_config, args):
        calls['run_repo_root'] = repo_root
        calls['loaded'] = loaded_config
        calls['args'] = args
        return {'ok': True}

    runtime_runner_stub.load_app_config = fake_load_app_config
    runtime_runner_stub.run_loaded_config = fake_run_loaded_config

    assert (
        bootstrap_main.main(
            ['--config', 'yaml/experiment/feature_set/brentoil-case1.yaml', '--validate-only']
        )
        == 0
    )
    assert calls['load_repo_root'] == bootstrap_main.WORKSPACE_ROOT
    assert calls['run_repo_root'] == bootstrap_main.WORKSPACE_ROOT
    assert calls['loaded'] is loaded
    assert getattr(calls['args'], 'validate_only') is True
    assert getattr(calls['args'], 'jobs') is None
    assert calls['config_path'] == 'yaml/experiment/feature_set/brentoil-case1.yaml'
    assert calls['shared_settings_path'] is None
    assert getattr(calls['args'], 'optuna_study') is None

    parts = os.environ['PYTHONPATH'].split(os.pathsep)
    assert parts[0] == workspace_root
    assert parts.count(workspace_root) == 1
    assert '/tmp/example' in parts


def test_main_requires_explicit_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(bootstrap_main._BOOTSTRAP_ENV, '1')

    with pytest.raises(SystemExit) as exc_info:
        bootstrap_main.main(['--validate-only'])

    assert exc_info.value.code == 2
    assert (
        'config path is required; pass --config/--config-path or --config-toml'
        in capsys.readouterr().err
    )


def test_main_passes_setting_override_to_load_app_config(
    monkeypatch: pytest.MonkeyPatch,
    runtime_runner_stub: ModuleType,
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setenv(bootstrap_main._BOOTSTRAP_ENV, '1')

    loaded = SimpleNamespace(jobs_fanout_specs=[])

    def fake_load_app_config(
        repo_root: Path,
        *,
        config_path=None,
        config_toml_path=None,
        shared_settings_path=None,
    ):
        calls['repo_root'] = repo_root
        calls['config_path'] = config_path
        calls['config_toml_path'] = config_toml_path
        calls['shared_settings_path'] = shared_settings_path
        return loaded

    runtime_runner_stub.load_app_config = fake_load_app_config
    runtime_runner_stub.run_loaded_config = lambda *args, **kwargs: {'ok': True}

    assert (
        bootstrap_main.main(
            [
                '--config',
                'demo.yaml',
                '--setting',
                'yaml/setting/setting.yaml',
                '--validate-only',
            ]
        )
        == 0
    )
    assert calls['repo_root'] == bootstrap_main.WORKSPACE_ROOT
    assert calls['config_path'] == 'demo.yaml'
    assert calls['config_toml_path'] is None
    assert calls['shared_settings_path'] == 'yaml/setting/setting.yaml'


def test_main_forwards_optuna_study_to_runtime_args(
    monkeypatch: pytest.MonkeyPatch,
    runtime_runner_stub: ModuleType,
) -> None:
    import runtime_support

    calls: dict[str, object] = {}
    monkeypatch.setenv(bootstrap_main._BOOTSTRAP_ENV, '1')

    loaded = SimpleNamespace(jobs_fanout_specs=[])

    runtime_runner_stub.load_app_config = lambda *args, **kwargs: loaded
    monkeypatch.setitem(sys.modules, 'runtime_support.runner', runtime_runner_stub)
    monkeypatch.setattr(runtime_support, 'runner', runtime_runner_stub, raising=False)

    def fake_run_loaded_config(repo_root: Path, loaded_config, args):
        calls['repo_root'] = repo_root
        calls['loaded'] = loaded_config
        calls['args'] = args
        return {'ok': True}

    runtime_runner_stub.run_loaded_config = fake_run_loaded_config

    assert (
        bootstrap_main.main(
            ['--config', 'demo.yaml', '--optuna-study', '3', '--validate-only']
        )
        == 0
    )
    assert calls['repo_root'] == bootstrap_main.WORKSPACE_ROOT
    assert calls['loaded'] is loaded
    assert getattr(calls['args'], 'optuna_study') == 3


def test_runtime_main_delegates_back_to_bootstrap_main(
    monkeypatch: pytest.MonkeyPatch,
    runtime_runner_stub: ModuleType,
) -> None:
    calls: dict[str, object] = {}

    def fake_run_cli(args, **kwargs):
        calls['args'] = list(args)
        calls['kwargs'] = dict(kwargs)
        return 23

    monkeypatch.setattr(bootstrap_main, '_run_cli', fake_run_cli)

    assert runtime_runner_stub.main(['--validate-only']) == 23
    assert calls['args'] == ['--validate-only']
    assert calls['kwargs']['repo_root'] == Path(__file__).resolve().parents[1]


def test_bootstrap_owned_contracts_expose_direct_runtime_modules() -> None:
    import app_config
    import app_config as config_module
    from plugin_contracts.stage_plugin import StagePlugin
    from plugin_contracts.stage_registry import get_active_stage_plugin
    import plugin_contracts.stage_plugin as stage_plugin_module
    import plugin_contracts.stage_registry as stage_registry_module

    assert config_module.load_app_config is app_config.load_app_config
    assert stage_plugin_module.StagePlugin is StagePlugin
    assert stage_registry_module.get_active_stage_plugin is get_active_stage_plugin


def test_main_reexecs_with_expected_args_and_bootstrap_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake_venv_python = tmp_path / 'venv-python'
    fake_venv_python.write_text('', encoding='utf-8')
    other_python = tmp_path / 'system-python'
    other_python.write_text('', encoding='utf-8')

    monkeypatch.setattr(bootstrap_main, 'VENV_PYTHON', fake_venv_python)
    monkeypatch.setattr(sys, 'executable', str(other_python))
    monkeypatch.delenv(bootstrap_main._BOOTSTRAP_ENV, raising=False)
    monkeypatch.setenv('PYTHONPATH', '/tmp/example')

    captured: dict[str, object] = {}

    def fake_execvpe(executable: str, argv: list[str], env: dict[str, str]) -> None:
        captured['executable'] = executable
        captured['argv'] = list(argv)
        captured['env'] = dict(env)
        raise RuntimeError('execvpe intercepted')

    monkeypatch.setattr(os, 'execvpe', fake_execvpe)

    with pytest.raises(RuntimeError, match='execvpe intercepted'):
        bootstrap_main.main(['--validate-only'])

    assert captured['executable'] == str(fake_venv_python)
    assert captured['argv'] == [
        str(fake_venv_python),
        str(Path(bootstrap_main.__file__).resolve()),
        '--validate-only',
    ]
    env = captured['env']
    assert isinstance(env, dict)
    assert env[bootstrap_main._BOOTSTRAP_ENV] == '1'
    parts = env['PYTHONPATH'].split(os.pathsep)
    assert parts[0] == str(bootstrap_main.WORKSPACE_ROOT)
    assert parts.count(str(bootstrap_main.WORKSPACE_ROOT)) == 1
    assert '/tmp/example' in parts


def test_needs_reexec_falls_back_to_true_when_sys_executable_resolution_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake_venv_python = tmp_path / 'venv-python'
    fake_venv_python.write_text('', encoding='utf-8')

    class _MissingExecutablePath:
        def resolve(self) -> Path:
            raise FileNotFoundError

    real_path = bootstrap_main.Path
    monkeypatch.setattr(bootstrap_main, 'VENV_PYTHON', fake_venv_python)
    monkeypatch.delenv(bootstrap_main._BOOTSTRAP_ENV, raising=False)

    def fake_path(value: str | os.PathLike[str]) -> Path | _MissingExecutablePath:
        if os.fspath(value) == sys.executable:
            return _MissingExecutablePath()
        return real_path(value)

    monkeypatch.setattr(bootstrap_main, 'Path', fake_path)

    assert bootstrap_main._needs_reexec() is True


@pytest.mark.parametrize(
    'argv',
    [
        ['--output-root', 'runs/custom'],
        ['--output-root=runs/custom'],
        ['--optuna-study', '2', '--output-root', 'runs/custom'],
        ['--output-root=runs/custom', '--optuna-study', '2'],
    ],
)
def test_main_rejects_removed_output_root_flag(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match='--output-root is no longer supported'):
        bootstrap_main.main(argv)


def test_build_parser_rejects_non_positive_optuna_study(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = bootstrap_main.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(['--optuna-study', '0'])

    assert exc_info.value.code == 2
    assert 'argument --optuna-study: must be a positive integer' in capsys.readouterr().err


def test_main_allows_internal_output_root_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(bootstrap_main._ALLOW_INTERNAL_OUTPUT_ROOT_ENV, '1')

    bootstrap_main._reject_removed_args(['--output-root', 'runs/internal'])


def test_load_app_config_shared_settings_round_trip_preserves_optuna_study_selection(
    tmp_path: Path,
) -> None:
    config_path = _write_yaml(tmp_path / 'config.yaml', _minimal_app_config_payload())
    shared_settings_path = _write_yaml(
        tmp_path / 'setting.yaml',
        {
            'runtime': {
                'random_seed': 11,
                'opt_n_trial': 7,
                'opt_study_count': 3,
                'opt_selected_study': 2,
                'transformations_target': 'diff',
            }
        },
    )

    loaded = load_app_config(
        tmp_path,
        config_path=config_path,
        shared_settings_path=shared_settings_path.name,
    )

    assert loaded.config.runtime.random_seed == 11
    assert loaded.config.runtime.opt_n_trial == 7
    assert loaded.config.runtime.opt_study_count == 3
    assert loaded.config.runtime.opt_selected_study == 2
    assert loaded.config.runtime.transformations_target == 'diff'
    assert loaded.config.to_dict()['runtime'] == {
        'random_seed': 11,
        'opt_n_trial': 7,
        'opt_study_count': 3,
        'opt_selected_study': 2,
        'transformations_target': 'diff',
    }
    assert loaded.normalized_payload['runtime'] == {
        'random_seed': 11,
        'opt_n_trial': 7,
        'opt_study_count': 3,
        'opt_selected_study': 2,
        'transformations_target': 'diff',
    }


def test_load_app_config_rejects_duplicate_shared_runtime_transformation(
    tmp_path: Path,
) -> None:
    payload = _minimal_app_config_payload()
    payload['runtime'] = {'transformations_target': 'diff'}
    config_path = _write_yaml(tmp_path / 'config.yaml', payload)
    shared_settings_path = _write_yaml(
        tmp_path / 'setting.yaml',
        {'runtime': {'transformations_target': 'diff'}},
    )

    with pytest.raises(
        ValueError,
        match='config repeats shared setting path\\(s\\): runtime.transformations_target',
    ):
        load_app_config(
            tmp_path,
            config_path=config_path,
            shared_settings_path=shared_settings_path.name,
        )


def test_load_app_config_defaults_opt_study_count_and_omits_unset_selected_study(
    tmp_path: Path,
) -> None:
    config_path = _write_yaml(tmp_path / 'config.yaml', _minimal_app_config_payload())

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.config.runtime.opt_study_count == 1
    assert loaded.config.runtime.opt_selected_study is None
    assert loaded.config.to_dict()['runtime'] == {
        'random_seed': 1,
        'opt_n_trial': None,
        'opt_study_count': 1,
    }
    assert loaded.normalized_payload['runtime'] == {
        'random_seed': 1,
        'opt_n_trial': None,
        'opt_study_count': 1,
    }


def test_load_app_config_rejects_opt_selected_study_above_count(tmp_path: Path) -> None:
    payload = _minimal_app_config_payload()
    payload['runtime'] = {'opt_study_count': 2, 'opt_selected_study': 3}
    config_path = _write_yaml(tmp_path / 'config.yaml', payload)

    with pytest.raises(
        ValueError, match='runtime.opt_selected_study cannot exceed runtime.opt_study_count'
    ):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_preserves_loss_params_for_supported_parametric_losses(
    tmp_path: Path,
) -> None:
    payload = _minimal_app_config_payload()
    payload['training'] = {
        'loss': 'latehorizonweightedmape',
        'loss_params': {
        'horizon': 4,
        'late_start': 3,
        'late_multiplier': 2.5,
    }
    }
    config_path = _write_yaml(tmp_path / 'config.yaml', payload)

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.config.training.loss == 'latehorizonweightedmape'
    assert loaded.config.to_dict()['training']['loss_params'] == {
        'up_th': 0.9,
        'down_th': 0.1,
        'lamda_underestimate': 1.2,
        'lamda_overestimate': 1.0,
        'lamda': 1.0,
        'horizon': 4,
        'ramp_power': 1.5,
        'base_weight': 1.0,
        'late_multiplier': 2.5,
        'late_start': 3,
        'delta': 1.0,
        'late_weight': 3.0,
        'q_under': 0.7,
        'q_over': 0.3,
        'late_factor': 2.0,
    }
    assert 'loss_params' in loaded.normalized_payload['training']
