<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-04-17 -->

# tests

## Purpose
This directory contains regression coverage for the wrapper runtime, plugins, YAML/config contracts, runtime helpers, analysis scripts, and selected custom model integrations in this workspace.

## Key Areas
| Path | Description |
|------|-------------|
| `test_main_runtime_bootstrap.py` | Runtime entrypoint/bootstrap and validate-only behavior coverage. |
| `test_runner_parallel_tuning.py` | Parallel tuning and scheduler/runtime orchestration coverage. |
| `test_aa_forecast_plugin_contracts.py` | AA-Forecast plugin config and contract tests. |
| `test_retrieval_plugin_contracts.py` | Retrieval plugin contract tests. |
| `test_optimizer_plugins.py` | Optimizer registry coverage. |
| `test_validate_only_*.py` | Validate-only smoke tests for representative models and plugin routes. |
| `test_run_*.py` | Shell-wrapper and experiment-matrix regression coverage. |
| `dummy/` | Lightweight dummy model implementations used through `conftest.py` (see `dummy/AGENTS.md`). |
| `fixtures/` | Small YAML/CSV/JSON fixtures for runtime and validate-only tests (see `fixtures/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Add or update regression coverage in the smallest relevant area whenever behavior changes.
- Prefer targeted `--no-cov` selectors while iterating; save broad runs for final confidence.
- Reuse fixtures and dummy models instead of building large new harnesses.
- Keep fixture configs representative but small enough for fast validate-only and smoke tests.
