<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# retrieval plugin

## Purpose
This package implements the standalone retrieval plugin used for post-prediction retrieval, event scoring, detail-config loading, and retrieval-specific analysis outputs.

## Key Files
| File | Description |
|------|-------------|
| `plugin.py` | `StagePlugin`-style retrieval integration and config-path validation. |
| `config.py` | Retrieval config normalization and serialization helpers. |
| `runtime.py` | Retrieval neighbor selection, blending, and runtime post-processing helpers. |
| `event_score_distribution_plot.py` | Event-score visualization helper used by retrieval outputs. |
| `signatures.py` | Shared retrieval signature helpers/types. |

## For AI Agents

### Working In This Directory
- Keep retrieval detail-config validation aligned with `app_config.py` and YAML fixtures.
- Retrieval is a plugin-layer augmentation; do not fold its post-processing logic into unrelated model code.

### Testing Requirements
- Run `uv run pytest --no-cov tests/test_retrieval_plugin_contracts.py tests/test_validate_only_retrieval.py tests/test_plot_retrieval_event_score_distribution.py` as relevant.
