<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-04-17 -->

# docs

## Purpose
This directory holds generated/reference documentation for the package, Mintlify site assets, and a small number of local wrapper-specific notes.

## Key Files
| File | Description |
|------|-------------|
| `*.html.md` | Generated API reference pages for package modules and models. |
| `runtime-transformations-diff-review.md` | Local review note for runtime transformation behavior. |
| `aa_forecast_brentoil_runtime_flow.md` | Narrative note describing an AA-Forecast runtime flow. |
| `to_mdx.py` | Helper for docs conversion/generation. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `mintlify/` | Mintlify docs site assets, config, and image bundles (see `mintlify/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Treat most `*.html.md` files as generated artifacts; only patch them directly when the task explicitly calls for docs regeneration or a generated-doc fix.
- If package/runtime surfaces change, update the relevant docs and mention whether regeneration was performed.
