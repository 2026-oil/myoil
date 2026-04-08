<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-23 -->

# neuralforecast package

## Purpose
This directory contains the forecasting library code: model classes, auto wrappers, shared modules, losses, dataset helpers, and core orchestration consumed by the local runtime.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package exports. |
| `core.py` | Main `NeuralForecast` orchestration surface. |
| `auto.py` | Auto-model wrappers and tuning-facing exports. |
| `compat.py` | Compatibility helpers. |
| `tsdataset.py` | Dataset utilities used by training/runtime code. |
| `utils.py` | Shared package utilities. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `common/` | Base model/auto abstractions and shared modules. |
| `losses/` | Loss definitions (`numpy.py`, `pytorch.py`). |
| `models/` | Concrete forecasting model implementations. |

## For AI Agents

### Working In This Directory
- Keep package exports, auto wrappers, and core orchestration in sync when model support changes.
- Prefer matching existing model file patterns instead of introducing new abstractions.
- Check capability flags and constructor signatures before wiring new models into runtime code.

### Testing Requirements
- Model registry/export changes: `uv run pytest --no-cov tests/test_common/test_model_registry_new_models.py tests/test_common/test_base_auto.py tests/test_core.py`.
- Model-specific edits: run the matching `tests/test_models/test_<model>.py` selector when it exists.
- Loss changes: run the relevant `tests/test_losses/` selectors.

### Common Patterns
- Model files are one-class-per-file under `models/` with exports re-collected in `models/__init__.py`.
- Shared building blocks belong under `common/`; public wrapper surfaces belong in `core.py` / `auto.py`.
