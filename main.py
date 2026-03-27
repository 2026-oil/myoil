from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = WORKSPACE_ROOT / '.venv' / 'bin' / 'python'
_BOOTSTRAP_ENV = 'NEURALFORECAST_RESIDUAL_BOOTSTRAPPED'
_ALLOW_INTERNAL_OUTPUT_ROOT_ENV = 'NEURALFORECAST_ALLOW_INTERNAL_OUTPUT_ROOT'


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


def _reject_removed_args(args: Sequence[str]) -> None:
    if os.environ.get(_ALLOW_INTERNAL_OUTPUT_ROOT_ENV) == '1':
        return
    for idx, arg in enumerate(args):
        if arg == '--output-root':
            raise SystemExit(
                '--output-root is no longer supported; main.py now derives run roots automatically from task.name and jobs route suffixes.'
            )
        if arg.startswith('--output-root='):
            raise SystemExit(
                '--output-root is no longer supported; main.py now derives run roots automatically from task.name and jobs route suffixes.'
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _reject_removed_args(args)
    if _needs_reexec():
        env = _build_env()
        env[_BOOTSTRAP_ENV] = '1'
        os.execvpe(
            str(VENV_PYTHON),
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *args],
            env,
        )

    os.environ.update(_build_env())
    from residual.runtime import main as residual_main

    return residual_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
