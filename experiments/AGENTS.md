<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-04-17 -->

# experiments

## Purpose
This directory contains self-contained reference experiment projects kept alongside the main wrapper workspace for focused research or benchmarking.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `brent_weekly_gate/` | Custom Brent weekly admissibility-gate experiment harness and results (see `brent_weekly_gate/AGENTS.md`). |
| `kan_benchmark/` | KAN benchmark environment and runner (see `kan_benchmark/AGENTS.md`). |
| `long_horizon/` | Long-horizon experiment environment and NHITS runner (see `long_horizon/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Treat these as self-contained experiment sandboxes rather than the main operator runtime.
- Prefer narrow edits inside the specific experiment requested; do not propagate wrapper-runtime assumptions here unless explicitly needed.
