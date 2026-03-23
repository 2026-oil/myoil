<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-23 | Updated: 2026-03-23 -->

# scripts

## Purpose
This directory contains helper utilities used around the wrapper runtime, especially local dataframe analysis and small CLI helpers.

## Key Files
| File | Description |
|------|-------------|
| `analyze_df_correlations.py` | Correlation report generator for the local `df.csv`-style research dataset. |
| `analyze_df_multicollinearity.py` | Multicollinearity/VIF analysis helper. |
| `cli.py` | Local CLI helper script. |
| `cvt.py` | Small conversion utility. |
| `extract_test.sh` | Shell helper used for extraction/testing tasks. |
| `filter_licenses.py` | License filtering helper. |

## For AI Agents

### Working In This Directory
- Keep scripts runnable from the repo root and avoid assuming a global Python outside `uv run`.
- Favor deterministic file outputs and explicit paths because these scripts are often used to produce research artifacts.

### Testing Requirements
- Run the narrowest relevant pytest selector when one exists (for example `tests/test_analyze_df_correlations.py`).
- For standalone script edits without direct tests, run the script against a small fixture or `--help` style smoke check and report the exact command.
