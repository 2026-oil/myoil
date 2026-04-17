<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# neuralforecast models

## Purpose
This directory contains concrete forecasting model implementations plus the package export registry and wrapper-specific direct/stage-model helpers.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Central model export registry and stage-model helper re-exports. |
| `bs_preforcast_catalog.py` | Direct-stage model catalog and stage-only parameter registry exports. |
| `bs_preforcast_direct.py` | Direct prediction helpers for stage-only models. |
| `timexer.py`, `patchtst.py`, `informer.py`, `gru.py`, etc. | Concrete model implementations exposed through the package. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `aaforecast/` | AA-Forecast model package, backbones, and internal submodules (see `aaforecast/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Keep `__init__.py`, runtime model registration, search-space support, and docs/tests aligned when adding or removing models.
- Most files are one-model-per-file; preserve that organization unless the existing subpackage already groups internals.
- Wrapper-specific direct-stage helpers live here too, so confirm whether a change affects both top-level runtime and package import surfaces.
