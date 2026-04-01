# casual leading-indicator workspace

This directory holds the reusable causal / lag-aware oil leading-indicator workflow requested for `data/df.csv`.

## Entry point

```bash
uv run python casual/analyze_leading_indicators.py --help
```

## Default behavior

- reads `data/df.csv`
- analyzes both `Com_CrudeOil` and `Com_BrentCrudeOil`
- attempts:
  - practical implemented methods:
    - `lagged_correlation`
    - `granger`
    - `tigramite_pcmci`
    - `nonlincausality`
    - `neural_gc`
    - `tcdf`
  - optional probed-only methods:
    - `gvar`
    - `jrngc`
    - `causalformer`
    - `cuts_plus`
    - `sru_gci`
    - `gc_xlstm`
- writes a run root under `runs/casual-leading-indicators-<timestamp>/`

## Optional method policy

- Heavy methods use a bounded predictor subset selected from baseline rankings.
- Installed / vendored methods are executed and surfaced in `method_status.csv`.
- Remaining cited families without a local adapter stay visible as `skipped-not-feasible` or `blocked` with reasons.

## Artifact contract

Each run emits:

- `dataset_audit.json`
- `analysis_manifest.json`
- `method_status.csv`
- `summary.json`
- `report.md`
- target-specific ranking tables under `tables/`
- target/method metadata under `metadata/`

Optional/heavier methods are never dropped silently. If they are unavailable in the local environment, they are surfaced as `skipped-not-feasible` or `blocked` with a reason.
