# runtime.transformations diff review notes

This note captures the current review/doc status for the planned `runtime.transformations: diff` work.
It is intentionally evidence-first so lane integration can compare the merged runtime against the agreed contract in `.omx/plans/ralplan-runtime-transformations-diff.md`.

## Current code evidence

### Config surface is not wired yet
- `app_config.py:87-90` — `RuntimeConfig` currently exposes only `random_seed` and `opt_n_trial`.
- `app_config.py:737-769` — `_normalize_payload()` reads `runtime` but only validates `opt_n_trial`; there is no `runtime.transformations` validation or unset-omission path.
- `app_config.py:994-1008` — `load_app_config()` hashes `config.to_dict()` directly, so omission stability for `runtime.transformations` is not yet implemented.

### Runtime still operates on raw targets end-to-end
- `runtime_support/runner.py:442-477` — `_fit_and_predict_fold()` trains and predicts from raw `train_df` / `future_df` with no fold-local diff transform or inverse reconstruction.
- `runtime_support/runner.py:1324-1361` — `_baseline_cross_validation()` predicts directly from the raw history (`history.iloc[-1]`, seasonal tail, or mean), so baseline diff-space forecasting is not present.
- `runtime_support/runner.py:595-714` — `_tune_main_job()` scores Optuna trials directly on raw predictions from `_fit_and_predict_fold()`.
- `runtime_support/runner.py:841-852` and `runtime_support/adapters.py:71-117` — multivariate input building still forwards raw target and exogenous channels together; target-only differencing is not wired.

## Existing review evidence already present

### Summary artifact coverage exists today

### Residual non-interference coverage exists today

## Missing integration pieces to check when lane 1/2 land
1. `RuntimeConfig` / config normalization must accept only `runtime.transformations: diff` and omit the key entirely when unset so legacy `config.resolved.json` and `resolved_hash` stay stable.
2. Baseline, learned, and Optuna flows must share one fold-local diff contract: forward diff on the target only, first diff row dropped, inverse cumulative sum anchored on the last raw training target.
3. Multivariate mode must transform only `dataset.target_col`; exogenous channels must remain raw while the first timestamp is dropped across all channels to preserve a rectangular panel.
4. Residual artifacts (`backcast_panel.csv`, `corrected_eval.csv`, `corrected_folds.csv`, diagnostics, summary outputs) must remain original-scale only.
5. Named regression tests from the plan are still absent and should be added before closing the feature lane.

## Integration risks to review during merge
- **Hash stability risk:** adding a defaulted `transformations` field directly to `RuntimeConfig.to_dict()` will change `config.resolved.json` and `resolved_hash` even when unset.
- **Baseline fairness risk:** if baseline models continue using raw history while learned jobs use diff history, leaderboard comparisons will mix objective spaces.
- **Multivariate shape risk:** target-only differencing without synchronized timestamp dropping will break rectangular multivariate inputs.

## Review-ready verification bundle after implementation lands
Run these checks after lane 1/2/3 changes are merged:

```bash
```

Then add the planned `runtime.transformations`-specific selectors from the ralplan diff and confirm:
- unset `runtime.transformations` leaves `config/config.resolved.json` unchanged
- baseline / learned / Optuna forecasts are written on original scale only
- summary outputs (`summary/leaderboard.csv`, `summary/sample.md`, last-fold plots) still build successfully


## Verification snapshot (2026-03-22)

The coverage failure is a harness issue for targeted single-test invocations, not a functional regression in the reviewed runtime surfaces. The `--no-cov` reruns provide the task-scoped verification evidence requested by the plan.
