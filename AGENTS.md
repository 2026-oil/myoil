<!-- Generated: 2026-03-23 | Updated: 2026-04-17 -->

# neuralforecast

## Purpose
This checkout is a hybrid workspace: the upstream-style `neuralforecast/` package lives alongside a local experiment harness driven by `main.py`, `runtime_support/`, `app_config.py`, plugin packages, and the curated `yaml/` Brent/WTI case matrix.

## Key Files
| File | Description |
|------|-------------|
| `main.py` | Bootstrap entrypoint that re-execs into `.venv` and hands control to `runtime_support.runner.main()`. |
| `app_config.py` | Central config loading, normalization, validation, and typed dataclass definitions. |
| `yaml/HPO/search_space.yaml` | Canonical Optuna/search-space contract used by runtime validation and tuning. |
| `run.sh` | Batch sweep runner for config sets and summary capture. |
| `README.md` | Operator-facing setup and runtime usage guide. |
| `pyproject.toml` | Project metadata and tooling configuration. |
| `uv.lock` | Locked dependency graph for the `uv` environment. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `neuralforecast/` | Forecasting library code, including custom model surfaces and exports (see `neuralforecast/AGENTS.md`). |
| `runtime_support/` | Runtime orchestration, scheduling, manifests, adapters, and Optuna helpers (see `runtime_support/AGENTS.md`). |
| `plugins/` | Stage/runtime plugin packages such as `aa_forecast`, `retrieval`, and optimizer integration (see `plugins/AGENTS.md`). |
| `plugin_contracts/` | Shared plugin protocol and registry used by the runtime (see `plugin_contracts/AGENTS.md`). |
| `tuning/` | Search-space loading, parameter registries, and Optuna helper surfaces (see `tuning/AGENTS.md`). |
| `yaml/` | Experiment matrix and shared YAML contracts (see `yaml/AGENTS.md`). |
| `tests/` | Regression coverage for runtime, plugins, configs, and wrapper integrations (see `tests/AGENTS.md`). |
| `scripts/` | Helper utilities used around research workflows and analysis (see `scripts/AGENTS.md`). |
| `docs/` | Generated/reference docs and local notes (see `docs/AGENTS.md`). |
| `data/` | Local datasets and small research artifacts used by validate-only and live runs (see `data/AGENTS.md`). |
| `experiments/` | Self-contained reference experiment projects kept beside the main harness (see `experiments/AGENTS.md`). |
| `nbs/`, `reference/`, `wiki/` | Supplemental notebooks, references, and notes; inspect narrowly before editing. |
| `runs/`, `lightning_logs/`, `htmlcov/`, `.omx/` | Generated runtime/test artifacts; do not mass-edit or clean them unless explicitly requested. |

## For AI Agents

### Working In This Repository
- Default to `uv run ...` for Python execution, tests, and validation.
- Treat this repo as a config-driven experiment harness layered on top of the library package; changes often need package code, runtime code, YAML, and tests updated together.
- Preserve the current wrapper contract: `main.py` is the operator entrypoint, while the real runtime lives under `runtime_support/`.
- Do not add fallback behavior. If execution reaches an unsupported path, raise an explicit error instead of silently degrading behavior.
- When adding or retiring model support, check the shared surfaces together: `neuralforecast/models/__init__.py`, `neuralforecast/auto.py`, `neuralforecast/core.py`, `runtime_support/forecast_models.py`, `tuning/search_space.py`, `yaml/HPO/search_space.yaml`, and the relevant tests.
- Stage plugins are managed inside `plugins/` and dispatched through `plugin_contracts/`; do not add plugin-specific imports directly into unrelated shared runtime modules.
- Avoid editing generated outputs under `runs/`, `lightning_logs/`, `htmlcov/`, `.omx/`, or nested OMX state unless the task explicitly asks for artifact manipulation.

### Testing Requirements
- Package/model changes: start with targeted `uv run pytest --no-cov` selectors, then finish with the broader affected suite.
- Runtime/plugin changes: run the smallest relevant `tests/test_*.py` selectors plus a representative `uv run python main.py --validate-only --config <path>` smoke when config flow is affected.
- YAML/run-script changes: validate touched configs and run the matching shell-script or YAML contract tests.
- Repo-wide confidence pass when needed: `pre-commit run --all-files` or `uv run pytest`.

### Common Patterns
- `task.name` and `--output-root` drive run-folder naming under `runs/`.
- Brent/WTI case configs live under `yaml/` and are grouped by family (`experiment/`, plugin-linked YAML, settings, HPO, sweep configs).
- Runtime transformations are normalized via config/runtime code (`transformations_target`, `transformations_exog`) rather than ad-hoc YAML keys.
- Plugin-enabled runs typically flow through `app_config.py` normalization → `plugin_contracts.stage_registry` dispatch → `runtime_support.runner` orchestration.

## Dependencies

### Internal
- `main.py` depends on `runtime_support/runner.py`.
- `runtime_support/` depends on `app_config.py`, `neuralforecast/`, `plugin_contracts/`, `plugins/`, `tuning/`, and `yaml/HPO/search_space.yaml`.
- `plugins/` depends on `app_config.py`, `runtime_support/`, `plugin_contracts/`, and selected package/runtime surfaces.
- `tests/` exercises wrapper/runtime contracts rather than just upstream package behavior.

### External
- `uv` for environment and command execution.
- `pytest`, Ruff, and pre-commit for verification.
- `optuna`, `pandas`, `numpy`, and the PyTorch/Lightning stack for tuning and training.
- Optional/runtime-integrated dependencies such as `pytorch-optimizer`, `xgboost`, `lightgbm`, and scikit-learn backends.

<!-- MANUAL: Add repository-specific notes below this line. -->

## Manual Notes

- New dependencies are allowed in this repository when they materially help complete the task. Keep dependency changes scoped and update the relevant manifests/locks.
- When writing or modifying code, do not implement fallback behavior. If execution reaches a former fallback path or unsupported branch, raise an explicit error immediately.
