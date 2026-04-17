<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# brent_weekly_gate

## Purpose
This experiment directory contains a focused Brent weekly forecasting benchmark centered on an admissibility/order gate for a 2-step horizon claim.

## Key Files
| File | Description |
|------|-------------|
| `main.py` | Experiment driver, metric definitions, hyperparameters, and evaluation protocol. |
| `data_utils.py` | Dataset loading, fold construction, metrics, and feature engineering helpers. |
| `models.py` | Compact forecasting baselines and ensemble/router model implementations. |
| `experiment_harness.py` | Training/evaluation harness used by the experiment driver. |
| `results.json` | Stored experiment output snapshot. |

## For AI Agents

### Working In This Directory
- Keep the experiment self-contained; avoid pulling in main-runtime assumptions unless intentionally refactoring the experiment.
- If modifying the claim/metric protocol, update both the code comments and stored result expectations.
