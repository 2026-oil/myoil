<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests fixtures

## Purpose
This directory stores small runtime/config/data fixtures used by validate-only tests, plugin contract tests, Optuna contract tests, and smoke regressions.

## Key Files
| File | Description |
|------|-------------|
| `aa_forecast_runtime_*.yaml` | AA-Forecast runtime smoke and route variants. |
| `retrieval_*.yaml` | Retrieval plugin smoke and linked-detail fixtures. |
| `bs_preforcast_*.yaml` | Direct-stage legacy compatibility smoke configs. |
| `*_smoke.csv` / `nec_runtime_data.csv` | Small input datasets for smoke tests. |
| `optuna_*.yaml` | Optuna search-space/selection fixtures. |
| `finding_*.json` | Retrieval/finding analysis fixture payloads. |

## For AI Agents

### Working In This Directory
- Prefer extending existing fixture families instead of creating one-off config naming schemes.
- Keep fixtures minimal, readable, and stable because many tests depend on exact paths and payload shapes.
