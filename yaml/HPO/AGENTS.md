<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# yaml HPO

## Purpose
This directory contains the authoritative YAML search-space contracts used by Optuna-driven tuning and validate-only config normalization.

## Key Files
| File | Description |
|------|-------------|
| `search_space.yaml` | Canonical runtime search-space contract. |
| `search_space_research.yaml` | Alternate/research search-space variant. |

## For AI Agents

### Working In This Directory
- Keep these files synchronized with `tuning/search_space.py` and any model/runtime support changes.
- Search-space edits are high-impact; validate at least one representative config after changing them.
