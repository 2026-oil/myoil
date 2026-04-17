<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tuning

## Purpose
This directory owns the runtime search-space contract: supported model sets, training parameter registries, Optuna defaults, and YAML-backed search-space loading.

## Key Files
| File | Description |
|------|-------------|
| `search_space.py` | Canonical search-space loader and registry for model/training parameters, execution modes, and Optuna defaults. |

## For AI Agents

### Working In This Directory
- Keep this module synchronized with `yaml/HPO/search_space.yaml`, `runtime_support/forecast_models.py`, and the exported model list under `neuralforecast/models/__init__.py`.
- Do not introduce silent fallbacks for missing search-space keys; validation should fail explicitly.

### Testing Requirements
- Run the relevant tuning/runtime selectors plus a validate-only smoke through `main.py` when the search-space contract changes.
