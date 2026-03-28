from __future__ import annotations

import argparse
import json
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
    for arg in args:
        if arg == '--output-root' or arg.startswith('--output-root='):
            raise SystemExit(
                '--output-root is no longer supported; main.py now derives run roots automatically from task.name and jobs route suffixes.'
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Residual wrapper runtime for neuralforecast.'
    )
    parser.add_argument('--config', default=None)
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--config-toml', default=None)
    parser.add_argument('--setting', default=None)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--jobs', nargs='+', default=None)
    parser.add_argument('--output-root', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--internal-jobs-route', default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        '--internal-stage',
        choices=('full', 'tune-main-only', 'replay-only'),
        default='full',
        help=argparse.SUPPRESS,
    )
    return parser


def _run_cli(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
) -> int:
    from residual import runtime as runtime_module

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = WORKSPACE_ROOT if repo_root is None else repo_root
    config_path = args.config or args.config_path
    loaded = runtime_module.load_app_config(
        repo_root,
        config_path=config_path,
        config_toml_path=args.config_toml,
        shared_settings_path=args.setting,
    )
    if loaded.jobs_fanout_specs and args.internal_jobs_route is not None:
        selected_spec = next(
            (
                spec
                for spec in loaded.jobs_fanout_specs
                if spec.route_slug == args.internal_jobs_route
            ),
            None,
        )
        if selected_spec is None:
            parser.error(
                f'--internal-jobs-route={args.internal_jobs_route} did not match any jobs route slug'
            )
        loaded = runtime_module.loaded_config_for_jobs_fanout(
            repo_root, loaded, selected_spec
        )
    if loaded.jobs_fanout_specs:
        if args.output_root is not None:
            parser.error(
                '--output-root is not supported when jobs is a list of job-file paths'
            )
        fanout_results = []
        for spec in loaded.jobs_fanout_specs:
            variant = runtime_module.loaded_config_for_jobs_fanout(
                repo_root, loaded, spec
            )
            fanout_results.append(runtime_module.run_loaded_config(repo_root, variant, args))
        if args.validate_only:
            print(json.dumps({'ok': True, 'fanout_runs': fanout_results}))
        return 0
    runtime_module.run_loaded_config(repo_root, loaded, args)
    return 0


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
    return _run_cli(args, repo_root=WORKSPACE_ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
