### L-1: Run summary: Reduce the key spike-case max h6-h8 MAPE across Brent and WTI to below 3%.
- **Strategy:** Runtime completion summary
- **Outcome:** summary
- **Insight:** Best retained metric 18.746852770317194 at iteration 0; the run ended with status baseline.
- **Context:** goal=Reduce the key spike-case max h6-h8 MAPE across Brent and WTI to below 3%.; scope=**; metric=keycase_max_mape_h6_h8_pct; direction=lower
- **Iteration:** h6h8-mape3-spike#0
- **Timestamp:** 2026-04-01T13:57:36Z

### L-2: [labels: event_mask_sparsity] Use a robust upward-spike event mask for the first historical exogenous channel so AAForec
- **Strategy:** [labels: event_mask_sparsity] Use a robust upward-spike event mask for the first historical exogenous channel so AAForecast attention stays sparse instead of always-on.
- **Outcome:** keep
- **Insight:** [labels: event_mask_sparsity] Use a robust upward-spike event mask for the first historical exogenous channel so AAForecast attention stays sparse instead of always-on.
- **Context:** goal=Reduce Brent fold1 h6-h8 mean MAPE to below 3% using AAForecast-based methodology on the fixed 8-week forecast setup.; scope=**; metric=brent_fold1_mean_mape_h6_h8_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-mape3#2
- **Timestamp:** 2026-04-01T14:17:27Z

### L-3: [labels: reactive_decomposition] Shorten the AAForecast seasonal/trend windows and lower the anomaly threshold so STAR-s
- **Strategy:** [labels: reactive_decomposition] Shorten the AAForecast seasonal/trend windows and lower the anomaly threshold so STAR-style decomposition reacts faster to short Brent spike regimes.
- **Outcome:** keep
- **Insight:** [labels: reactive_decomposition] Shorten the AAForecast seasonal/trend windows and lower the anomaly threshold so STAR-style decomposition reacts faster to short Brent spike regimes.
- **Context:** goal=Reduce Brent fold1 h6-h8 mean MAPE to below 3% using AAForecast-based methodology on the fixed 8-week forecast setup.; scope=**; metric=brent_fold1_mean_mape_h6_h8_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-mape3#5
- **Timestamp:** 2026-04-01T14:20:34Z

### L-4: [labels: lower_dropout] Remove encoder dropout on the retained reactive setup so the point forecast preserves more rare-
- **Strategy:** [labels: lower_dropout] Remove encoder dropout on the retained reactive setup so the point forecast preserves more rare-spike signal instead of smoothing it away during training.
- **Outcome:** keep
- **Insight:** [labels: lower_dropout] Remove encoder dropout on the retained reactive setup so the point forecast preserves more rare-spike signal instead of smoothing it away during training.
- **Context:** goal=Reduce Brent fold1 h6-h8 mean MAPE to below 3% using AAForecast-based methodology on the fixed 8-week forecast setup.; scope=**; metric=brent_fold1_mean_mape_h6_h8_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-mape3#10
- **Timestamp:** 2026-04-01T14:25:13Z

### L-5: [labels: lower_anomaly_threshold] Lower the retained anomaly threshold from 2.5 to 2.0 so more pre-spike Brent timesteps
- **Strategy:** [labels: lower_anomaly_threshold] Lower the retained anomaly threshold from 2.5 to 2.0 so more pre-spike Brent timesteps enter the anomaly-aware path without changing the broader retained recipe.
- **Outcome:** keep
- **Insight:** [labels: lower_anomaly_threshold] Lower the retained anomaly threshold from 2.5 to 2.0 so more pre-spike Brent timesteps enter the anomaly-aware path without changing the broader retained recipe.
- **Context:** goal=Reduce Brent fold1 h6-h8 mean MAPE to below 3% using AAForecast-based methodology on the fixed 8-week forecast setup.; scope=**; metric=brent_fold1_mean_mape_h6_h8_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-mape3#13
- **Timestamp:** 2026-04-01T14:28:19Z

### L-6: [PIVOT] Micro-tuning decomposition memory, depth, and event-channel ordering did not beat the retained OVX-led recipe; s
- **Strategy:** [PIVOT] Micro-tuning decomposition memory, depth, and event-channel ordering did not beat the retained OVX-led recipe; switch to attention-selectivity tuning next.
- **Outcome:** pivot
- **Insight:** [PIVOT] Micro-tuning decomposition memory, depth, and event-channel ordering did not beat the retained OVX-led recipe; switch to attention-selectivity tuning next.
- **Context:** goal=Reduce Brent fold1 h6-h8 mean MAPE to below 3% using AAForecast-based methodology on the fixed 8-week forecast setup.; scope=**; metric=brent_fold1_mean_mape_h6_h8_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-mape3#23
- **Timestamp:** 2026-04-01T14:36:03Z

### L-7: [labels: narrower_attention] Reduce attention_hidden_size to 32 so AAForecast sparse attention scores critical steps thr
- **Strategy:** [labels: narrower_attention] Reduce attention_hidden_size to 32 so AAForecast sparse attention scores critical steps through a tighter bottleneck and less diffuse weighting.
- **Outcome:** keep
- **Insight:** [labels: narrower_attention] Reduce attention_hidden_size to 32 so AAForecast sparse attention scores critical steps through a tighter bottleneck and less diffuse weighting.
- **Context:** goal=Reduce Brent fold1 h6-h8 mean MAPE to below 3% using AAForecast-based methodology on the fixed 8-week forecast setup.; scope=**; metric=brent_fold1_mean_mape_h6_h8_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-mape3#24
- **Timestamp:** 2026-04-01T14:37:39Z

### L-8: [PIVOT] Attention bottleneck tuning found a retained winner at 32 but over-tightening regressed; switch next to multi-si
- **Strategy:** [PIVOT] Attention bottleneck tuning found a retained winner at 32 but over-tightening regressed; switch next to multi-signal event-mask fusion across all historical exogenous channels.
- **Outcome:** pivot
- **Insight:** [PIVOT] Attention bottleneck tuning found a retained winner at 32 but over-tightening regressed; switch next to multi-signal event-mask fusion across all historical exogenous channels.
- **Context:** goal=Reduce Brent fold1 h6-h8 mean MAPE to below 3% using AAForecast-based methodology on the fixed 8-week forecast setup.; scope=**; metric=brent_fold1_mean_mape_h6_h8_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-mape3#27
- **Timestamp:** 2026-04-01T14:39:20Z

### L-9: [labels: narrower_attention] Restore the Brent AAForecast attention bottleneck to 32 so sparse critical-step scoring is
- **Strategy:** [labels: narrower_attention] Restore the Brent AAForecast attention bottleneck to 32 so sparse critical-step scoring is less diffuse than the current 64-width workspace state.
- **Outcome:** keep
- **Insight:** [labels: narrower_attention] Restore the Brent AAForecast attention bottleneck to 32 so sparse critical-step scoring is less diffuse than the current 64-width workspace state.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#1
- **Timestamp:** 2026-04-01T14:50:02Z

### L-10: [labels: late_horizon_context] Blend the latest attended AAForecast context progressively into later horizons so the far
- **Strategy:** [labels: late_horizon_context] Blend the latest attended AAForecast context progressively into later horizons so the farthest Brent steps can react more strongly to recent critical signals.
- **Outcome:** keep
- **Insight:** [labels: late_horizon_context] Blend the latest attended AAForecast context progressively into later horizons so the farthest Brent steps can react more strongly to recent critical signals.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#6
- **Timestamp:** 2026-04-01T15:02:27Z

### L-11: [labels: late_horizon_context] Square the AAForecast late-context ramp so h7/h8 still receive extra recent-event context
- **Strategy:** [labels: late_horizon_context] Square the AAForecast late-context ramp so h7/h8 still receive extra recent-event context while h6 is less over-boosted than with the linear blend.
- **Outcome:** keep
- **Insight:** [labels: late_horizon_context] Square the AAForecast late-context ramp so h7/h8 still receive extra recent-event context while h6 is less over-boosted than with the linear blend.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#8
- **Timestamp:** 2026-04-01T15:07:52Z

### L-12: [PIVOT] Late-context micro-tuning and event-mask rewrites failed to beat the retained squared late-context blend; switch
- **Strategy:** [PIVOT] Late-context micro-tuning and event-mask rewrites failed to beat the retained squared late-context blend; switch next to a fundamentally different AAForecast approach, likely decoder-side horizon conditioning or a residual-style correction path if it can stay within scope.
- **Outcome:** pivot
- **Insight:** [PIVOT] Late-context micro-tuning and event-mask rewrites failed to beat the retained squared late-context blend; switch next to a fundamentally different AAForecast approach, likely decoder-side horizon conditioning or a residual-style correction path if it can stay within scope.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#14
- **Timestamp:** 2026-04-01T15:20:54Z

### L-13: [labels: late_trend_correction] Add a small squared-horizon trend correction from the latest insample delta so AAForecas
- **Strategy:** [labels: late_trend_correction] Add a small squared-horizon trend correction from the latest insample delta so AAForecast can push farther horizons upward when the recent Brent trajectory is still rising.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Add a small squared-horizon trend correction from the latest insample delta so AAForecast can push farther horizons upward when the recent Brent trajectory is still rising.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#16
- **Timestamp:** 2026-04-01T15:26:03Z

### L-14: [labels: late_trend_correction] Concentrate the late-trend correction more heavily on the far end of the horizon by swit
- **Strategy:** [labels: late_trend_correction] Concentrate the late-trend correction more heavily on the far end of the horizon by switching to a fourth-power profile, so h8 receives much more upward bias than h6.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Concentrate the late-trend correction more heavily on the far end of the horizon by switching to a fourth-power profile, so h8 receives much more upward bias than h6.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#17
- **Timestamp:** 2026-04-01T15:29:01Z

### L-15: [labels: late_trend_correction] Push the late-trend correction farther toward the final horizon with a sixth-power profi
- **Strategy:** [labels: late_trend_correction] Push the late-trend correction farther toward the final horizon with a sixth-power profile so h8 gets more uplift while h6 receives less of the rising-trend bias.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Push the late-trend correction farther toward the final horizon with a sixth-power profile so h8 gets more uplift while h6 receives less of the rising-trend bias.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#18
- **Timestamp:** 2026-04-01T15:31:24Z

### L-16: [labels: late_trend_correction] Use a stronger, more back-loaded late-trend correction with a 1.5x eighth-power horizon
- **Strategy:** [labels: late_trend_correction] Use a stronger, more back-loaded late-trend correction with a 1.5x eighth-power horizon profile so the recent upward Brent delta boosts h8 far more than h6.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Use a stronger, more back-loaded late-trend correction with a 1.5x eighth-power horizon profile so the recent upward Brent delta boosts h8 far more than h6.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#19
- **Timestamp:** 2026-04-01T15:34:14Z

### L-17: [labels: late_trend_correction] Use an even more back-loaded late-trend correction with a 2.0x tenth-power horizon profi
- **Strategy:** [labels: late_trend_correction] Use an even more back-loaded late-trend correction with a 2.0x tenth-power horizon profile so the recent upward Brent delta concentrates its lift on h8 while h6 stays comparatively restrained.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Use an even more back-loaded late-trend correction with a 2.0x tenth-power horizon profile so the recent upward Brent delta concentrates its lift on h8 while h6 stays comparatively restrained.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#20
- **Timestamp:** 2026-04-01T15:37:12Z

### L-18: [labels: late_trend_correction] Delay the late-trend correction even further while strengthening the final-horizon lift
- **Strategy:** [labels: late_trend_correction] Delay the late-trend correction even further while strengthening the final-horizon lift with a 2.5x twelfth-power profile, reducing h6 spillover while improving h8.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay the late-trend correction even further while strengthening the final-horizon lift with a 2.5x twelfth-power profile, reducing h6 spillover while improving h8.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#21
- **Timestamp:** 2026-04-01T15:39:42Z

### L-19: [labels: late_trend_correction] Delay the late-trend correction further with a 3.0x fourteenth-power horizon profile so
- **Strategy:** [labels: late_trend_correction] Delay the late-trend correction further with a 3.0x fourteenth-power horizon profile so the recent upward Brent delta concentrates even more of its lift at the final horizon.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay the late-trend correction further with a 3.0x fourteenth-power horizon profile so the recent upward Brent delta concentrates even more of its lift at the final horizon.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#22
- **Timestamp:** 2026-04-01T15:42:53Z

### L-20: [labels: late_trend_correction] Delay the late-context blend to a cubic ramp while keeping the stronger late-trend corre
- **Strategy:** [labels: late_trend_correction] Delay the late-context blend to a cubic ramp while keeping the stronger late-trend correction, reducing h6 spillover and improving the overall Brent threshold metric.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay the late-context blend to a cubic ramp while keeping the stronger late-trend correction, reducing h6 spillover and improving the overall Brent threshold metric.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#23
- **Timestamp:** 2026-04-01T15:45:20Z

### L-21: [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 3.5x sixteenth-power hori
- **Strategy:** [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 3.5x sixteenth-power horizon profile, sharply reducing h6 spillover while continuing to lift h8.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 3.5x sixteenth-power horizon profile, sharply reducing h6 spillover while continuing to lift h8.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#24
- **Timestamp:** 2026-04-01T15:48:01Z

### L-22: [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 4.0x eighteenth-powe
- **Strategy:** [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 4.0x eighteenth-power horizon profile, continuing to reduce h6 spillover while maintaining h8 uplift.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 4.0x eighteenth-power horizon profile, continuing to reduce h6 spillover while maintaining h8 uplift.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#25
- **Timestamp:** 2026-04-01T15:50:22Z

### L-23: [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 4.5x twentieth-power
- **Strategy:** [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 4.5x twentieth-power horizon profile, shaving more h6 spillover while preserving the h8 lift.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 4.5x twentieth-power horizon profile, shaving more h6 spillover while preserving the h8 lift.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#26
- **Timestamp:** 2026-04-01T15:52:11Z

### L-24: [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 5.0x twenty-second-power
- **Strategy:** [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 5.0x twenty-second-power horizon profile, continuing to trim h6 spillover while keeping h8 inside its threshold band.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 5.0x twenty-second-power horizon profile, continuing to trim h6 spillover while keeping h8 inside its threshold band.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#27
- **Timestamp:** 2026-04-01T15:54:26Z

### L-25: [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 5.5x twenty-fourth-p
- **Strategy:** [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 5.5x twenty-fourth-power horizon profile, continuing to reduce h6 while preserving the later-horizon fit.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay and concentrate the late-trend correction even further with a 5.5x twenty-fourth-power horizon profile, continuing to reduce h6 while preserving the later-horizon fit.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#28
- **Timestamp:** 2026-04-01T15:56:47Z

### L-26: [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 6.0x twenty-sixth-power h
- **Strategy:** [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 6.0x twenty-sixth-power horizon profile, continuing to reduce the remaining h6 excess while keeping later horizons within target.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay and concentrate the late-trend correction further with a 6.0x twenty-sixth-power horizon profile, continuing to reduce the remaining h6 excess while keeping later horizons within target.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#29
- **Timestamp:** 2026-04-01T15:59:02Z

### L-27: [labels: late_trend_correction] Delay the late-context blend to a quartic ramp while keeping the strongly back-loaded la
- **Strategy:** [labels: late_trend_correction] Delay the late-context blend to a quartic ramp while keeping the strongly back-loaded late-trend correction, sharply reducing the remaining Brent h6 excess.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay the late-context blend to a quartic ramp while keeping the strongly back-loaded late-trend correction, sharply reducing the remaining Brent h6 excess.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#30
- **Timestamp:** 2026-04-01T16:02:33Z

### L-28: [labels: late_trend_correction] Delay the late-context blend to a quintic ramp while keeping the strongly back-loaded la
- **Strategy:** [labels: late_trend_correction] Delay the late-context blend to a quintic ramp while keeping the strongly back-loaded late-trend correction, pushing Brent fold1 h6 below the 3 percent target while retaining h7 and h8 under 5 percent.
- **Outcome:** keep
- **Insight:** [labels: late_trend_correction] Delay the late-context blend to a quintic ramp while keeping the strongly back-loaded late-trend correction, pushing Brent fold1 h6 below the 3 percent target while retaining h7 and h8 under 5 percent.
- **Context:** goal=Reduce Brent fold1 h6 MAPE below 3% and h7/h8 MAPE below 5% using AAForecast-based methodology.; scope=**; metric=brent_fold1_h6_3_h7h8_5_threshold_gap_pct; direction=lower
- **Iteration:** brent-aaf-h6h8-thresholds#31
- **Timestamp:** 2026-04-01T16:06:03Z
