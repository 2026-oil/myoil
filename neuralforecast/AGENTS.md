<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-04-17 -->

# neuralforecast package

## Purpose
This directory contains the forecasting library code: public package exports, orchestration surfaces, shared modules, losses, preprocessing hooks, and concrete model implementations consumed by the local runtime.

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
| `common/` | Base model/auto abstractions and shared modules (see `common/AGENTS.md`). |
| `losses/` | Loss definitions including wrapper-added research losses (see `losses/AGENTS.md`). |
| `models/` | Concrete forecasting model implementations and export registry (see `models/AGENTS.md`). |
| `preprocessing/` | Package preprocessing hooks and namespace scaffolding (see `preprocessing/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Keep package exports, auto wrappers, and core orchestration in sync when model support changes.
- Prefer matching existing model-file patterns instead of introducing new abstractions.
- Check capability flags and constructor signatures before wiring new models into runtime code.

### Testing Requirements
- Model registry/export changes: run targeted package/runtime selectors that cover the touched surface.
- Loss changes: run the relevant selectors plus any wrapper tests that depend on the modified loss behavior.
