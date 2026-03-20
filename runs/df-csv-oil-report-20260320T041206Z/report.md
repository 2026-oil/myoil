# df.csv Oil Analysis Artifact Bundle

- Root: `runs/df-csv-oil-report-20260320T041206Z`
- Scope override: Notion-related work skipped per latest user instruction; local artifacts only.

## Correlation targets
- `Com_CrudeOil`: rows=89, n_obs=584..584
- `Com_BrentCrudeOil`: rows=89, n_obs=584..584
- `diff(Com_CrudeOil)`: rows=89, n_obs=583..583
- `diff(Com_BrentCrudeOil)`: rows=89, n_obs=583..583

## Raw-target multicollinearity only
- `Com_CrudeOil`: predictors=89, high_vif=87, severe_vif=82, condition_number=2.25181e+15
- `Com_BrentCrudeOil`: predictors=89, high_vif=87, severe_vif=82, condition_number=2.25181e+15

## Artifact entrypoints
- Correlation report: `runs/df-csv-oil-report-20260320T041206Z/correlations/report.md`
- Multicollinearity report: `runs/df-csv-oil-report-20260320T041206Z/multicollinearity/report.md`
- Manifest: `runs/df-csv-oil-report-20260320T041206Z/artifact_manifest.json`
