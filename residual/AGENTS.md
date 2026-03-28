<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-28 -->

# residual runtime

## Purpose
This directory contains the local experiment runtime layered on top of `neuralforecast`: config loading, scheduler flow, Optuna/search-space logic, manifests, feature engineering, progress events, residual model plugins, and the generic **Stage Plugin** system.

## Key Files
| File | Description |
|------|-------------|
| `runtime.py` | Main CLI/runtime orchestration, CV execution, reporting, and scheduler entrypoint. |
| `config.py` | YAML/TOML config loader, normalization, and validation logic. `AppConfig.stage_plugin_config` / `LoadedConfig.stage_plugin_loaded` hold opaque stage data. |
| `models.py` | Forecast-model registry, capability checks, and common loss resolution. |
| `optuna_spaces.py` | Search-space contract, tuning defaults, residual selector logic, and supported model lists. |
| `stage_plugin.py` | **StagePlugin Protocol** — generic interface for pre-main-stage pipelines. |
| `stage_registry.py` | **Stage Plugin Registry** — register/lookup `StagePlugin` instances by config key. |
| `registry.py` | Residual plugin factory routing. |
| `features.py` | Residual feature-frame construction. |
| `manifest.py` | Run-manifest and metadata writers. |
| `scheduler.py` | Parallel launch planning and worker execution. |
| `progress.py` | Console/structured progress emission. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `plugins/` | Backend-specific residual adapters for `xgboost`, `randomforest`, and `lightgbm`. |

## Stage Plugin System
`residual/` has **zero direct references** to any concrete stage plugin (e.g. `bs_preforcast`). All stage-specific logic is provided through the `StagePlugin` Protocol defined in `stage_plugin.py` and dispatched via `stage_registry.py`.

- **Protocol**: `StagePlugin` in `stage_plugin.py` — config lifecycle, route validation, stage loading, search-space integration, runtime hooks, manifest/validation, and fanout helpers.
- **Registry**: `stage_registry.py` — plugins register at import time; lazy discovery triggers `plugins.bs_preforcast.plugin` import on first use.
- **Config fields**: `AppConfig.stage_plugin_config` (typed config from plugin), `LoadedConfig.stage_plugin_loaded` (loaded stage-1 data from plugin).

Concrete implementations live in their own packages (e.g. `plugins/bs_preforcast/plugin.py`).

## For AI Agents

### Working In This Directory
- **Do NOT add direct imports of `bs_preforcast` (or any concrete stage plugin) into `residual/`**. Use the Stage Plugin Protocol and Registry for all stage interactions.
- Preserve config/runtime/reporting compatibility; many bugs here are contract drift rather than isolated code defects.
- Keep `config.py`, `optuna_spaces.py`, `models.py`, `registry.py`, and plugin implementations aligned whenever residual/model support changes.
- Runtime transformation support is normalized through config fields (`transformations_target`, `transformations_exog`); do not reintroduce raw ad-hoc runtime keys.
- Keep the residual checkpoint filename/path contract stable (`residual_checkpoint/model.ubj`) unless the task explicitly changes it.

### Testing Requirements
- Primary regression pass: `uv run pytest --no-cov tests/test_residual_config.py tests/test_residual_main.py`.
- Residual backend changes should also cover representative validate-only fixture runs in `tests/fixtures/residual_runtime_smoke_*.yaml`.
- If you touch search-space wiring, check `yaml/HPO/search_space.yaml` parity and run the relevant targeted tests before broader runs.
- Stage plugin changes: `uv run pytest --no-cov tests/test_bs_preforcast_config.py tests/test_bs_preforcast_runtime.py tests/test_bs_preforcast_plugin_only_contract.py`.

### Common Patterns
- Config dataclasses live in `config.py`; runtime consumes normalized dataclass objects rather than raw YAML dicts.
- Auto-tuning and residual selectors are centralized in `optuna_spaces.py`.
- Residual backends are plugin-based and route through lowercase backend names.
- Stage plugins register via `stage_registry.register_stage_plugin()` at import time. The registry uses lazy discovery for known packages.
