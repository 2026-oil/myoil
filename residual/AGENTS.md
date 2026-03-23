<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-23 -->

# residual runtime

## Purpose
This directory contains the local experiment runtime layered on top of `neuralforecast`: config loading, scheduler flow, Optuna/search-space logic, manifests, feature engineering, progress events, and residual model plugins.

## Key Files
| File | Description |
|------|-------------|
| `runtime.py` | Main CLI/runtime orchestration, CV execution, reporting, and scheduler entrypoint. |
| `config.py` | YAML/TOML config loader, normalization, and validation logic. |
| `models.py` | Forecast-model registry, capability checks, and common loss resolution. |
| `optuna_spaces.py` | Search-space contract, tuning defaults, residual selector logic, and supported model lists. |
| `registry.py` | Residual plugin factory routing. |
| `features.py` | Residual feature-frame construction. |
| `manifest.py` | Run-manifest and metadata writers. |
| `scheduler.py` | Parallel launch planning and worker execution. |
| `progress.py` | Console/structured progress emission. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `plugins/` | Backend-specific residual adapters for `xgboost`, `randomforest`, and `lightgbm`. |

## For AI Agents

### Working In This Directory
- Preserve config/runtime/reporting compatibility; many bugs here are contract drift rather than isolated code defects.
- Keep `config.py`, `optuna_spaces.py`, `models.py`, `registry.py`, and plugin implementations aligned whenever residual/model support changes.
- Runtime transformation support is normalized through config fields (`transformations_target`, `transformations_exog`); do not reintroduce raw ad-hoc runtime keys.
- Keep the residual checkpoint filename/path contract stable (`residual_checkpoint/model.ubj`) unless the task explicitly changes it.

### Testing Requirements
- Primary regression pass: `uv run pytest --no-cov tests/test_residual_config.py tests/test_residual_main.py`.
- Residual backend changes should also cover representative validate-only fixture runs in `tests/fixtures/residual_runtime_smoke_*.yaml`.
- If you touch search-space wiring, check `search_space.yaml` parity and run the relevant targeted tests before broader runs.

### Common Patterns
- Config dataclasses live in `config.py`; runtime consumes normalized dataclass objects rather than raw YAML dicts.
- Auto-tuning and residual selectors are centralized in `optuna_spaces.py`.
- Residual backends are plugin-based and route through lowercase backend names.
