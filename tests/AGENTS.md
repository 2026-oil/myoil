<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-29 -->

# tests

## Purpose
Runtime wrapper/contract tests for configs, residual behavior, stage plugins, and YAML matrix validation. Upstream package tests (per-model, losses, scalers, backward compat) have been removed — those belong to the neuralforecast package itself.

## Key Areas
| Path | Description |
|------|-------------|
| `test_residual_config.py` | Residual config/search-space validation coverage (large integration). |
| `test_residual_main.py` | Runtime entrypoint and validate-only behavior coverage. |
| `test_bs_preforcast_*.py` | Stage plugin config, runtime, registry, search space, and contract tests. |
| `test_optimizer_plugins.py` | Optimizer plugin registry coverage. |
| `test_top_level_direct_models.py` | Direct stage model support. |
| `test_validate_only_*.py` | Validate-only smoke tests for specific model/exog combinations. |
| `test_bomb_yaml_configs.py` | Bomb-family YAML contract checks. |
| `test_yaml_timexer_contract.py` | YAML matrix-wide TimeXer/PatchTST constraints. |
| `fixtures/` | Runtime smoke configs and small data fixtures. |
| `dummy/` | Dummy models registered via conftest.py for lightweight runtime tests. |

## For AI Agents

### Working In This Directory
- Add or update regression coverage in the smallest relevant area whenever behavior changes.
- Prefer targeted `--no-cov` selectors while iterating; save full coverage runs for broader confidence passes.
- Reuse fixtures/dummy models instead of introducing large new test harnesses.
- DummyUnivariate/DummyMultivariate are registered into MODEL_CLASSES via `conftest.py`, not via runtime code.

### Common Patterns
- Wrapper-runtime tests only — no upstream package model/loss/core tests in this tree.
- Lightweight smoke configs belong in `fixtures/` and should stay fast enough for validate-only runs.
