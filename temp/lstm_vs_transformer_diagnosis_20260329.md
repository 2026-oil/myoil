# LSTM / TSMixerx vs Transformer-family diagnosis (2026-03-29)

## Scope
Existing run artifacts only. No reruns.

- `runs/feature_set_*_jobs_*/summary/leaderboard.csv` (32 base jobs)
- `runs/feature_set_bs_*_bs_jobs_1/summary/leaderboard.csv` (2 extra current jobs)
- `runs/feature_set_legacy/**/summary/leaderboard.csv` (8 legacy runs)

## Current base jobs (`feature_set_*_jobs_*`, n=32)
- LSTM: avg rank 1.7812, top1 16, avg MAPE 0.0606, avg nRMSE 0.8490
- iTransformer: avg rank 2.9062, top1 4, avg MAPE 0.0791, avg nRMSE 1.0177
- TimeXer: avg rank 4.1875, top1 0, avg MAPE 0.0909, avg nRMSE 1.3436
- TSMixerx: avg rank 4.4375, top1 0, avg MAPE 0.1149, avg nRMSE 1.7517
- Transformer family (iTransformer + TimeXer): avg rank 3.5469, top1 4, avg MAPE 0.0850, avg nRMSE 1.1806

## Legacy vs current shift
Legacy (`feature_set_legacy`, n=8) favored transformer-family models over LSTM:
- Legacy avg MAPE: iTransformer 0.0717, TimeXer 0.0728, TSMixerx 0.0857, LSTM 0.0865

Current jobs flipped that ordering:
- Current avg MAPE: LSTM 0.0606, iTransformer 0.0791, TimeXer 0.0909, TSMixerx 0.1149

## Key findings
1. LSTM dominance is real in current jobs.
2. TSMixerx dominance is not supported by current artifacts; it is usually the weakest learned model.
3. The reversal from legacy to current jobs points to a setup/configuration effect, not an intrinsic “transformers are always worse” conclusion.

## Evidence-backed causes
### A. Problem structure is very LSTM-friendly
- `data/df.csv`: 584 rows, Brent lag-1 autocorr 0.983, WTI lag-1 autocorr 0.982.
- Current CV/training: horizon 8, input_size 64.
- Short horizon + very strong autocorrelation favors local autoregressive bias.

### B. Multivariate-vs-univariate formulation mismatch
- `runtime_support/adapters.py` builds multivariate inputs from `[target, *hist_exog, *futr_exog]`.
- Current configs use target + 16 historic exogenous columns, so transformer-family multivariate models run with `n_series=17`.
- `LSTM` is univariate (`MULTIVARIATE = False`) while `TSMixerx`, `iTransformer`, and `TimeXer` are multivariate (`MULTIVARIATE = True`).
- `runtime_support/runner.py` fits the multivariate model, then filters predictions back to `target_col`, so the multivariate models spend capacity on the 17-channel problem before target extraction.

### C. Current runs are fixed-parameter, not actively tuned
- `config/capability_report.json` across checked runs reports `search_space_entry_found: false` and `selected_search_params: []`.
- So the current performance gap mainly reflects fixed config choices, not each family’s best achievable result.

### D. TimeXer is over-sized for this dataset/setup
Representative current config (`feature_set_brentoil_case1_jobs_1`) parameter counts:
- LSTM: 125,601
- TSMixerx: 85,723
- iTransformer: 88,264
- TimeXer: 25,341,960

### E. Current shared training contract likely favors smaller/simple models
Current `yaml/setting/setting.yaml`:
- max_steps 2000
- val_size 24
- val_check_steps 20
- min_steps_before_early_stop 500
- early_stop_patience_steps 3
- optimizer AdamW
- scheduler OneCycleLR max_lr 0.001

This is a single shared schedule for all families; small/simple models can stabilize quickly, while larger attention models may need a different schedule.

### F. TimeXer shows runtime instability on some runs
`mse=nan` appears in 5 current TimeXer worker logs:
- `runs/feature_set_brentoil_case2_jobs_1/.../TimeXer/stdout.log`
- `runs/feature_set_brentoil_case4_jobs_3/.../TimeXer/stdout.log`
- `runs/feature_set_wti_case1_jobs_3/.../TimeXer/stdout.log`
- `runs/feature_set_wti_case3_jobs_1/.../TimeXer/stdout.log`
- `runs/feature_set_wti_case4_jobs_3/.../TimeXer/stdout.log`

### G. Some TimeXer configs waste lookback due to patch mismatch
Example current jobs_3 configs use `patch_len=24` with `input_size=64`.
Because `timexer.py` patches with `unfold(..., size=patch_len, step=patch_len)`, 64 is not evenly divisible by 24, so part of the lookback is effectively unused.

## Fold stability signal
Mean fold-MSE std across current base jobs:
- LSTM: 52.625
- TimeXer: 63.284
- iTransformer: 94.832
- TSMixerx: 87.991
- Transformer family mean (TimeXer + iTransformer): 79.058

## Practical takeaway
Current results are best explained as:
- LSTM benefiting from target-centric autoregressive bias on a short-horizon, high-autocorrelation dataset,
- transformer-family models being penalized by multivariate formulation + fixed shared training settings,
- and TimeXer additionally suffering from over-capacity and some unstable / patch-mismatch configurations.

## Suggested next checks
1. Compare LSTM vs iTransformer vs TimeXer at matched capacity.
2. Restrict TimeXer patch_len to divisors of input_size (8/16/32 for input_size 64).
3. Give transformer-family models their own LR/patience schedule.
4. Audit whether multivariate formulation is appropriate for these target-plus-exog runs.
