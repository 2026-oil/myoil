<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-04-17 -->

# yaml experiment matrix

## Purpose
This directory holds the curated YAML configuration set used by the local wrapper runtime. The layout is organized by shared contracts, plugin-linked YAML, experiment families, and reusable settings/jobs.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `HPO/` | Central Optuna/search-space contracts (see `HPO/AGENTS.md`). |
| `experiment/` | Family-oriented Brent/Dubai/WTI experiment configs (see `experiment/AGENTS.md`). |
| `jobs/` | Shared job-list YAML fragments and routes (see `jobs/AGENTS.md`). |
| `plugins/` | Plugin-linked YAML payloads and parity configs (see `plugins/AGENTS.md`). |
| `setting/` | Shared setting authority YAML (see `setting/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Preserve family naming and Brent/WTI pairing conventions; downstream scripts depend on predictable file paths.
- Prefer editing the narrowest family affected by the request instead of copying settings across unrelated trees.
- Keep config semantics aligned with `app_config.py`, plugin config loaders, and `yaml/HPO/search_space.yaml`.
- Do not silently add new top-level keys; runtime validation is strict.

### Testing Requirements
- Always run `uv run python main.py --validate-only --config <changed-config>` for touched configs.
- Run any matching YAML or shell-wrapper regression selector when family structure or linked configs change.
