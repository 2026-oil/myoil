<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-23 -->

# experiments

## Purpose
This directory contains upstream/reference experiment projects kept alongside the local wrapper workspace.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `kan_benchmark/` | KAN benchmark environment and runner. |
| `long_horizon/` | Long-horizon experiment environment and NHITS runner. |
| `nbeats_basis/` | N-BEATS basis notebook experiment. |

## For AI Agents

### Working In This Directory
- Treat these as self-contained reference experiments rather than the main operator runtime.
- Prefer narrow edits inside the specific experiment requested; do not propagate wrapper-runtime assumptions here unless the task clearly asks for it.
