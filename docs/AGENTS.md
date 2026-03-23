<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-23 -->

# docs

## Purpose
This directory holds generated/reference documentation for the package plus a small number of wrapper-specific notes.

## Key Areas
| Path | Description |
|------|-------------|
| `*.html.md` | Generated API reference pages for package modules/models. |
| `mintlify/` | Mintlify site assets and config. |
| `runtime-transformations-diff-review.md` | Local review note for runtime transformation behavior. |
| `to_mdx.py` | Helper for docs conversion/generation. |

## For AI Agents

### Working In This Directory
- Treat most `*.html.md` files as generated artifacts; only hand-edit them when the task explicitly calls for docs regeneration or patching generated output.
- If public model/runtime surfaces change, update the relevant docs and mention whether regeneration was performed.
