<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# plugin_contracts

## Purpose
This directory defines the shared protocol and registry that let the runtime discover and invoke stage plugins without importing plugin-specific logic into generic runtime code.

## Key Files
| File | Description |
|------|-------------|
| `stage_plugin.py` | `StagePlugin` protocol describing config lifecycle, stage loading, search-space hooks, and runtime callbacks. |
| `stage_registry.py` | Plugin registration and active-plugin resolution used by config loading and runtime orchestration. |

## For AI Agents

### Working In This Directory
- Treat this directory as the abstraction boundary between `runtime_support/` and `plugins/`.
- Interface changes here are high-impact: update plugin implementations, runtime callers, and tests together.

### Testing Requirements
- Run `uv run pytest --no-cov tests/test_stage_registry.py tests/test_aa_forecast_plugin_contracts.py tests/test_retrieval_plugin_contracts.py` when protocol or registry behavior changes.
