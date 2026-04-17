<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-17 | Updated: 2026-04-17 -->

# neuralforecast losses

## Purpose
This directory contains loss-function implementations used by package models and wrapper-specific research flows.

## Key Files
| File | Description |
|------|-------------|
| `pytorch.py` | Main PyTorch loss implementations and distribution losses. |
| `numpy.py` | NumPy-based metric/loss helpers. |
| `research_losses.py` | Additional research-oriented losses used by the local runtime. |

## For AI Agents

### Working In This Directory
- Preserve the existing NeuralForecast loss contract (`domain_map`, weighting, output metadata) when adding or editing losses.
- Loss changes can alter training semantics across many models; update downstream runtime expectations and tests as needed.
