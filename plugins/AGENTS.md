<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# plugins

## Purpose
This directory holds plugin packages that extend the wrapper runtime without hard-coding feature-specific logic into shared orchestration modules.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `aa_forecast/` | AA-Forecast stage plugin, runtime materialization, and plugin-specific search/config helpers (see `aa_forecast/AGENTS.md`). |
| `optimizer/` | Optimizer registry and optional third-party optimizer integration (see `optimizer/AGENTS.md`). |
| `retrieval/` | Standalone retrieval plugin and post-prediction retrieval runtime helpers (see `retrieval/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Keep plugin ownership local to the relevant package; shared runtime code should talk to plugins through `plugin_contracts/`.
- Avoid cross-plugin coupling unless the integration is already explicit and tested.
- If a plugin changes public config semantics, update YAML examples/fixtures and the associated tests.

### Testing Requirements
- Run the plugin’s focused test selectors, e.g. `tests/test_aa_forecast_plugin_contracts.py`, `tests/test_retrieval_plugin_contracts.py`, or `tests/test_optimizer_plugins.py`.
