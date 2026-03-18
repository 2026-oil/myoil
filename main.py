from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = WORKSPACE_ROOT / '.venv' / 'bin' / 'python'
_BOOTSTRAP_ENV = 'NEURALFORECAST_RESIDUAL_BOOTSTRAPPED'


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', str(WORKSPACE_ROOT))
    parts = [part for part in env['PYTHONPATH'].split(os.pathsep) if part]
    if str(WORKSPACE_ROOT) not in parts:
        env['PYTHONPATH'] = os.pathsep.join([str(WORKSPACE_ROOT), *parts])
    return env


def _needs_reexec() -> bool:
    if os.environ.get(_BOOTSTRAP_ENV) == '1':
        return False
    if not VENV_PYTHON.exists():
        return False
    try:
        return Path(sys.executable).resolve() != VENV_PYTHON.resolve()
    except FileNotFoundError:
        return True


def _exec_args(argv: Sequence[str]) -> list[str]:
    return [str(VENV_PYTHON), str(Path(__file__).resolve()), *argv]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if _needs_reexec():
        env = _build_env()
        env[_BOOTSTRAP_ENV] = '1'
        os.execvpe(str(VENV_PYTHON), _exec_args(args), env)

    os.environ.update(_build_env())
    from residual.runtime import main as residual_main

    return residual_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
