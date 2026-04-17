<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# aa_forecast plugin

## Purpose
This package implements the AA-Forecast stage plugin: config normalization, stage-1 route loading, selected-model/search-space integration, runtime materialization, and AA-specific helper modules.

## Key Files
| File | Description |
|------|-------------|
| `plugin.py` | `StagePlugin` implementation that validates routes, loads stage configs, and merges AA-Forecast state into `AppConfig`. |
| `config.py` | AA-Forecast config dataclasses, normalization, linked-config loading, and payload serialization helpers. |
| `runtime.py` | Runtime-side AA-Forecast artifact materialization, retrieval blending hooks, and override assembly. |
| `search_space.py` | AA-Forecast-specific search-space helpers and selected-parameter registries. |
| `modules.py` | Shared AA-Forecast neural modules imported by model code. |

## For AI Agents

### Working In This Directory
- Keep stage-1 loading semantics synchronized with `app_config.py`, `plugin_contracts/stage_plugin.py`, and `runtime_support/runner.py`.
- The plugin owns AA-Forecast route semantics; do not duplicate that logic elsewhere.
- If you change selected-model or training-search behavior, update both plugin tests and representative fixture YAMLs.

### Testing Requirements
- Run `uv run pytest --no-cov tests/test_aa_forecast_plugin_contracts.py tests/test_aaforecast_adapter_contract.py tests/test_feature_set_aaforecast_postprocess.py` and any AA-Forecast runtime selectors touched by the change.

## Dependencies

### Internal
- Depends on `app_config.py`, `plugin_contracts/`, `runtime_support/`, `plugins/retrieval/`, `tuning/`, and AA-related model surfaces under `neuralforecast/models/aaforecast/`.
