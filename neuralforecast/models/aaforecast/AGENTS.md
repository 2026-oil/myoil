<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# aaforecast model package

## Purpose
This subpackage contains the AA-Forecast model implementation, backbone selection logic, and AA-specific model internals used by the plugin/runtime layer.

## Key Files
| File | Description |
|------|-------------|
| `model.py` | Main `AAForecast` model implementation and custom decoding/head logic. |
| `backbones.py` | Backbone registry/build helpers for supported AA-Forecast backbones. |
| `gru.py` | AA-Forecast GRU helpers and shared runtime utilities. |
| `__init__.py` | Package export surface for AA-Forecast. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `models/` | Internal AA-Forecast support modules used by the main model implementation. |

## For AI Agents

### Working In This Directory
- Keep plugin-side config assumptions and model-side constructor parameters synchronized.
- AA-Forecast changes often require updates in `plugins/aa_forecast/`, runtime fixtures, and AA-specific tests.
