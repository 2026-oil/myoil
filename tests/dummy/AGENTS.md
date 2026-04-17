<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# tests dummy models

## Purpose
This directory contains lightweight dummy model implementations used to exercise registry/runtime behavior without incurring the cost of real training stacks.

## Key Files
| File | Description |
|------|-------------|
| `dummy_models.py` | Dummy univariate/multivariate model implementations registered for tests via `tests/conftest.py`. |

## For AI Agents

### Working In This Directory
- Keep dummy models simple and deterministic; they are test scaffolding, not production models.
- If runtime contracts change, prefer updating these dummies over introducing heavier fixtures.
