from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import main as bootstrap_main


def test_main_delegates_without_reexec_and_preserves_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
):
    import residual.runtime as runtime

    calls: dict[str, object] = {}
    workspace_root = str(bootstrap_main.WORKSPACE_ROOT)
    monkeypatch.setenv(bootstrap_main._BOOTSTRAP_ENV, '1')
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([workspace_root, '/tmp/example']))

    def fake_runtime_main(args: list[str]) -> int:
        calls['args'] = list(args)
        return 17

    monkeypatch.setattr(runtime, 'main', fake_runtime_main)

    assert bootstrap_main.main(['--validate-only']) == 17
    assert calls['args'] == ['--validate-only']

    parts = os.environ['PYTHONPATH'].split(os.pathsep)
    assert parts[0] == workspace_root
    assert parts.count(workspace_root) == 1
    assert '/tmp/example' in parts


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
