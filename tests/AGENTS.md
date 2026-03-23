<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-23 -->

# tests

## Purpose
This directory mirrors both the upstream package test layout and the local wrapper/runtime contract tests for configs, shell runners, analysis utilities, and residual behavior.

## Key Areas
| Path | Description |
|------|-------------|
| `test_models/` | Per-model regression tests for forecasting model implementations. |
| `test_common/` | Shared package abstractions, registry, and module tests. |
| `test_losses/` | Loss-function coverage. |
| `test_residual_config.py` | Residual config/search-space validation coverage. |
| `test_residual_main.py` | Runtime entrypoint and validate-only behavior coverage. |
| `test_bomb_yaml_configs.py` | Bomb-family YAML contract checks. |
| `test_run_*.py` | Shell wrapper regression coverage. |
| `fixtures/` | Runtime smoke configs and small data fixtures. |
| `helpers/` | Test-only helpers. |
| `dummy/` | Dummy models used for lightweight runtime tests. |

## For AI Agents

### Working In This Directory
- Add or update regression coverage in the smallest relevant area whenever behavior changes.
- Prefer targeted `--no-cov` selectors while iterating; save full coverage runs for broader confidence passes.
- Reuse fixtures/dummy models instead of introducing large new test harnesses.

### Common Patterns
- Wrapper-runtime tests live alongside package tests in the same tree; choose names that make the affected contract obvious.
- Lightweight smoke configs belong in `fixtures/` and should stay fast enough for validate-only runs.
