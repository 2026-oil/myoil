<!-- Generated: 2026-03-23 | Updated: 2026-04-03 -->

# neuralforecast

## Purpose
This checkout is not just the upstream `neuralforecast` package. It is a hybrid workspace that combines the library source under `neuralforecast/` with a local experiment runner built around `main.py`, `runtime_support/`, `app_config.py`, `yaml/HPO/search_space.yaml`, and the curated `yaml/` case matrix used for Brent/WTI research runs.

## Key Files
| File | Description |
|------|-------------|
| `main.py` | Bootstrap entrypoint that re-execs into `.venv` and hands control to `runtime_support.runner.main()`. |
| `yaml/HPO/search_space.yaml` | Central Optuna/search-space contract shared by runtime config validation. |
| `run.sh` | Batch runner that executes config sweeps, records summaries, and can auto-commit/push outputs. |
| `run_case3_bomb.sh` | Preloads the case3 bomb config set into `run.sh`. |
| `run_bomb_trans.sh` | Preloads the case3 transformation bomb config set into `run.sh`. |
| `run_after_pid7130_feature_set.sh` | Wait-chain runner that resumes queued feature-set jobs after another PID exits. |
| `README.md` | Wrapper-focused operator README for `uv` setup, config semantics, and runtime usage. |
| `pyproject.toml` | Python package metadata and test/lint configuration. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `neuralforecast/` | Upstream-style forecasting package sources plus added model surfaces (see `neuralforecast/AGENTS.md`). |
| `runtime_support/` | Runtime orchestration, adapters, manifests, scheduling, and common forecasting helpers for the local experiment harness. |
| `plugins/bs_preforcast/` | `bs_preforcast` stage plugin — registers as a `StagePlugin` via `plugins/bs_preforcast/plugin.py`; shared runtime modules never import it directly. |
| `yaml/` | Experiment matrix for Brent/WTI, bomb, HPT, univariate, and ad-hoc case families (see `yaml/AGENTS.md`). |
| `tests/` | Regression coverage for package code, runtime contracts, YAML contracts, and shell helpers (see `tests/AGENTS.md`). |
| `scripts/` | Analysis and helper utilities used around the runtime and research workflows (see `scripts/AGENTS.md`). |
| `docs/` | Generated API docs and local docs notes for this checkout (see `docs/AGENTS.md`). |
| `data/` | Local datasets and small research artifacts used by validate-only and live runs (see `data/AGENTS.md`). |
| `experiments/` | Upstream benchmark/reference experiments kept alongside the wrapper code (see `experiments/AGENTS.md`). |
| `runs/`, `lightning_logs/`, `htmlcov/`, `.omx/` | Generated runtime/test artifacts; do not mass-edit or clean them unless the task explicitly asks for it. |

## For AI Agents

### Working In This Repository
- Default to `uv run ...` for Python execution, tests, and validation.
- Treat this repo as a config-driven experiment harness layered on top of the library package; changes often need package code, runtime code, and YAML/test parity together.
- Preserve the current wrapper contract: `main.py` is the operator entrypoint, while the real scheduler/runtime lives under `runtime_support/`.
- When writing code in this repository, do not design or add fallback paths; if execution reaches a would-be fallback case, fail fast with an explicit error instead of silently degrading behavior.
- When adding or retiring model support, check shared surfaces together: `neuralforecast/models/__init__.py`, `neuralforecast/auto.py`, `neuralforecast/core.py`, `runtime_support/forecast_models.py`, `tuning/search_space.py`, `yaml/HPO/search_space.yaml`, and the relevant tests.
- **Stage plugins** (e.g. `bs_preforcast`) are managed exclusively in their own packages. The runtime dispatches to them via the `StagePlugin` Protocol in `plugin_contracts/stage_plugin.py` and the registry in `plugin_contracts/stage_registry.py`. Do NOT add direct `bs_preforcast` imports to shared runtime modules.
- Avoid editing generated outputs under `runs/`, `lightning_logs/`, `htmlcov/`, or `.omx/` unless the user explicitly asks for artifact manipulation.

### Testing Requirements
- Package/model changes: start with targeted `uv run pytest --no-cov` selectors, then finish with the broader affected suite.
- Legacy residual-retirement changes: run `uv run pytest --no-cov tests/test_legacy_residual_rejection.py tests/test_main_runtime_bootstrap.py` plus a validate-only smoke using a representative supported config.
- YAML/run-script changes: validate with `uv run python main.py --validate-only --config <path>` and the matching shell-script tests under `tests/test_run_*.py` or `tests/test_bomb_yaml_configs.py`.
- Repo-wide confidence pass when needed: `pre-commit run --all-files` or `uv run pytest`.

### Common Patterns
- `task.name` and `--output-root` drive run-folder naming under `runs/`.
- Brent/WTI case configs live under `yaml/` and are typically grouped by case family (`feature_set`, `feature_set_HPT`, `bomb`, `bomb_trans`, `univar`, etc.).
- Runtime transformations are normalized in config/runtime code via `transformations_target` / `transformations_exog` rather than ad-hoc YAML fields.

## Dependencies

### Internal
- `main.py` depends on `runtime_support/runner.py`.
- `runtime_support/` depends on `neuralforecast/` model surfaces and `yaml/HPO/search_space.yaml`.
- `plugins/bs_preforcast/` depends on `app_config.py` (for `AppConfig`/`LoadedConfig` types), `runtime_support/`, and `plugin_contracts/stage_registry.py`.
- `tests/` includes both package-style tests and wrapper/runtime contract tests.

### External
- `uv` for environment and command execution.
- `pytest`, Ruff, and mypy/pre-commit tooling for verification.
- `optuna`, `pandas`, `numpy`, and PyTorch/Lightning-adjacent stack for tuning and training.
- Residual backends including `xgboost`, `lightgbm`, and scikit-learn random forest support.

<!-- MANUAL: Add repository-specific notes below this line. -->

## Manual Notes

- New dependencies are allowed in this repository when they materially help complete the task. This overrides higher-level "no new dependencies without explicit request" guidance for files under this workspace.
- When adding a dependency, keep the change scoped, update the relevant manifest/lockfiles, and include verification that the new dependency is wired correctly.
- When writing or modifying code, do not implement fallback behavior. If execution reaches a former fallback path or unsupported branch, raise an explicit error immediately instead of silently degrading or switching behavior.
