<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# neuralforecast common

## Purpose
This directory contains shared package infrastructure such as base model classes, auto-model utilities, model checks, common modules, scalers, and enums.

## Key Files
| File | Description |
|------|-------------|
| `_base_model.py` | Shared Lightning-backed model base class and training/prediction lifecycle helpers. |
| `_base_auto.py` | Auto-model base support for search/tuning flows. |
| `_model_checks.py` | Validation helpers for model compatibility and runtime assumptions. |
| `_modules.py` | Reusable neural modules used by multiple model implementations. |
| `_scalers.py` | Temporal scaling helpers. |
| `enums.py` | Shared enums/constants used across package code. |

## For AI Agents

### Working In This Directory
- Changes here fan out broadly across many models; keep edits minimal and well-tested.
- Preserve package-wide conventions for tensor shapes, training hooks, and scaler behavior.
