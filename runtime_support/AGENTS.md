<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# runtime_support

## Purpose
This directory contains the local experiment harness implementation: config-to-run orchestration, dataset adapters, model dispatch, scheduling, progress reporting, manifests, and Optuna study helpers.

## Key Files
| File | Description |
|------|-------------|
| `runner.py` | Main runtime entrypoint used by `main.py`; drives validation, training, tuning, artifact writing, and summaries. |
| `forecast_models.py` | Model capability registry, runtime model construction, and custom loss/optimizer wiring. |
| `scheduler.py` | Multi-worker/GPU launch planning and worker subprocess orchestration. |
| `adapters.py` | Univariate/multivariate dataframe adapters that translate source data into NeuralForecast inputs. |
| `manifest.py` | Manifest assembly and atomic manifest writing helpers. |
| `optuna_studies.py` | Study catalog selection, metadata, and persistent study bookkeeping. |
| `optuna_visuals.py` | Optuna visualization generation and cross-study report helpers. |
| `progress.py` | Structured progress event rendering and parsing helpers. |
| `shared_training_contract.py` | Shared training-surface helpers used across runtime flows. |

## For AI Agents

### Working In This Directory
- Keep runtime behavior explicit; fail fast rather than silently skipping unsupported branches.
- Changes here often require parity updates in `app_config.py`, plugin packages, YAML contracts, and tests.
- Preserve the distinction between config validation, study/tuning selection, and actual fit/predict execution.
- Do not bypass `plugin_contracts.stage_registry` when stage plugins are involved.

### Testing Requirements
- Start with the narrowest runtime selector, e.g. `uv run pytest --no-cov tests/test_main_runtime_bootstrap.py tests/test_runner_parallel_tuning.py` plus the relevant plugin/runtime tests.
- If dataset adaptation or job validation changes, run the matching validate-only smoke config through `uv run python main.py --validate-only --config <path>`.

### Common Patterns
- Runtime outputs are written atomically where possible and keyed off the selected run root.
- Scheduling logic computes worker launch plans first, then delegates execution to subprocess workers with structured progress events.
- Model resolution flows through `forecast_models.py` rather than ad-hoc imports in the runner.

## Dependencies

### Internal
- Depends on `app_config.py`, `neuralforecast/`, `plugin_contracts/`, `plugins/`, and `tuning/`.
- Feeds manifests and outputs consumed by tests, shell wrappers, and downstream analysis scripts.

### External
- `optuna`, `numpy`, `pandas`, `torch`, and subprocess/process-management stdlib modules.
