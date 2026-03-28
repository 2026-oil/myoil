<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-23 -->

# yaml experiment matrix

## Purpose
This directory holds the curated experiment configurations used by the local wrapper runtime. The layout is case-family oriented rather than package-oriented.

## Key Subdirectories
| Directory | Purpose |
|-----------|---------|
| `experiment/feature_set/` | Base Brent/WTI feature-set case configs. |
| `experiment/feature_set_HPT*/` | Hyperparameter-tuning feature-set case families. |
| `jobs/` | Shared jobs route files used by case configs. |
| `setting/` | Shared setting authority files for centralized YAML controls. |
| `HPO/` | Central Optuna/search-space contracts. |
| `plugins/` | Plugin-owned linked YAMLs such as `bs_preforcast.yaml`. |
| `bomb/` | Case3 bomb sweep configs for exloss + diff/residual variations. |
| `bomb_trans/` | Transformation-focused case3 bomb sweep configs used by `run_bomb_trans.sh`. |
| `univar/` | Univariate baseline configs. |
| `blackswan/` | Black-swan case configs and exploratory variants. |
| `jaeho_feature_set/` | Researcher-specific exploratory configs retained for local experimentation. |

## For AI Agents

### Working In This Directory
- Preserve family naming and Brent/WTI pairing conventions; downstream scripts depend on predictable file paths.
- Prefer editing the narrowest family affected by the request instead of copying settings across unrelated trees.
- Keep config semantics aligned with `residual/config.py` and `yaml/HPO/search_space.yaml`.
- Do not silently add new top-level keys; runtime validation is strict.

### Testing Requirements
- Always run `uv run python main.py --validate-only --config <changed-config>` for touched configs.
- For bomb-family edits, also run `uv run pytest --no-cov tests/test_bomb_yaml_configs.py`.
- If a shell wrapper hardcodes this directory, run the matching `tests/test_run_*.py` regression.

### Common Patterns
- `jobs[*].params` carries model-specific knobs; shared training/runtime knobs belong in top-level sections.
- Case families often encode target, horizon/input-size, residual mode, and transformation mode directly in the filename.
