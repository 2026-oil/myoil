<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# optimizer plugin helpers

## Purpose
This package provides optimizer resolution for runtime training configs, including built-in Torch optimizers and optional `pytorch-optimizer` integrations.

## Key Files
| File | Description |
|------|-------------|
| `registry.py` | Registry, compatibility validation, and normalized optimizer resolution payloads. |

## For AI Agents

### Working In This Directory
- Keep optimizer names, kwargs validation, and compatibility probing deterministic.
- If new optimizers are added, update tests and ensure the runtime contract remains explicit rather than permissive.

### Testing Requirements
- Run `uv run pytest --no-cov tests/test_optimizer_plugins.py`.
