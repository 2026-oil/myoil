# colab_batch strategy summary

## Evidence base
- Source: `/home/sonet/gdrive2/runs/colab_batch` via `rclone`
- Aggregate sheet: `batch_leaderboard_summary.csv` (120 model rows / 12 experiment yamls)
- Last-3 detailed probe: `winner_last3_analysis.csv` for rank-1 winner of each yaml

## Core findings
1. **WTI best overall is stable:** `bomb/wti-case3-family-h8-diff-exloss-i128*` with `LSTM`/`LSTM_res` dominates (nRMSE about 0.1983, R2 about 0.637). `LSTM_res` is effectively a tie/fallback, not a meaningfully separate lane.
2. **Brent best overall is weaker and model-dependent:**
   - `i48` family: `TimeMixer_res` wins (nRMSE about 0.3240 to 0.3301)
   - `i128` family: `NHITS_res` wins (nRMSE about 0.3336 to 0.3422)
3. **Residual correction usually helps on Brent and shorter-window cases**, but gives only marginal gain on the already-strong WTI i128 LSTM lane.
4. **Last-3 failure mode is universal underprediction on upward spikes.** Every winner has negative `last3_bias`, especially horizons 7-8. This supports late-horizon/spike-aware optimization as the next experiment direction, but does not by itself prove causality.
5. **Longer history helps WTI, shorter history helps Brent.** WTI i128 winner last3 RMSE is about 10.18, while WTI i48 winners are 15.63 to 18.07. Brent i48 winners are 17.65 to 18.08, slightly better than Brent i128 winners 18.36 to 19.60.

## Recommended strategy
### Priority 1: Split target strategy
- **WTI main lane:** keep `h8 + diff + exloss + i128`, base model `LSTM`, with `LSTM_res` treated as a near-equal fallback/tie-breaker only.
- **Brent main lane:** move default shortlist to `h8 + diff + exloss + i48 + residual-level`, base model `TimeMixer_res`; keep `NHITS_res i128` as secondary challenger.

### Priority 2: Optimize for last-3 explicitly
- Add selection metric that blends overall nRMSE with `last3_RMSE` or `last3_MAE` instead of ranking only by mean-fold nRMSE.
- Increase late-horizon loss weight for steps 6-8.
- Track `last3_bias` separately and penalize large negative bias.

### Priority 3: Improve spike capture
- Preserve/expand spike-oriented exogenous features because the main miss pattern is lagged response into the final spike.
- Prefer **delta residual** or **level residual** only when it reduces negative last3 bias on validation; for WTI LSTM this gain is tiny, for Brent it is meaningful.
- Add horizon-aware residual model inputs (recent momentum / slope / rolling change features) to push steps 7-8 upward faster.

### Priority 4: Narrow search, not broad search
- For WTI, spend trials on `LSTM` family training knobs and late-horizon weighting, not on switching architecture.
- For Brent, spend trials on `TimeMixer_res` i48 and `NHITS_res` i128 with residual config and spike-feature variants.
- De-prioritize DLinear / iTransformer for this regime because both overall and last-3 behavior are materially worse.

## Concrete next experiment order
1. `WTI`: LSTM i128 base vs residual-level with late-horizon weighted objective
2. `WTI`: same lane + stronger spike/momentum exogs, judge by blended score `(0.7 * nRMSE + 0.3 * normalized last3_RMSE)`
3. `Brent`: TimeMixer_res i48 level-residual with spike-feature variants
4. `Brent`: NHITS_res i128 delta vs level residual head-to-head under the same blended score
5. Final shortlist: only keep candidates whose `last3_bias` improves toward zero without losing more than ~3% overall nRMSE
