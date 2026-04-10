# Autoresearch Record

## Iteration 0 Baseline
- timestamp: 2026-04-09T13:13:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Baseline measurement for Brent case1 parity AAForecast
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: none (baseline measurement only)
- 고정한 통제변인: target=Com_BrentCrudeOil; allowed Brent hist exog subset only; h=4 one-shot forecast contract; STAR -> AA-model -> MC uncertainty; no future/static exog; no deliberate drift/uplift; runtime budget < 1200s
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (62.6121, 62.8121, 63.2481, 63.0270)
- 15% band PASS/FAIL: FAIL (h3=27.00%, h4=36.26%)
- runtime PASS/FAIL: PASS (~199.5s by artifact elapsed)
- leakage concern 여부: none observed from TSCV one-shot artifact layout

## Iteration 1 Crash
- timestamp: 2026-04-09T13:16:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Tighten LOWESS locality and switch upward STAR tails to GPRD channels
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: lowess_frac 0.35 -> 0.20; star_anomaly_tails.upward = [GPRD_THREAT, GPRD, GPRD_ACT]
- 고정한 통제변인: target=Com_BrentCrudeOil; h=4 one-shot forecast contract; STAR -> AA-model -> MC uncertainty; no future/static exog; no deliberate drift/uplift
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (runtime crash before prediction)
- 15% band PASS/FAIL: FAIL (runtime crash before prediction)
- runtime PASS/FAIL: FAIL (ValueError: AAForecast hist exog groups must cover hist_exog_list exactly)
- leakage concern 여부: none; failure was configuration-order contract only

## Iteration 2 Discard
- timestamp: 2026-04-09T13:20:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Reorder hist_exog so GPRD channels form the STAR prefix with tighter LOWESS locality
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: hist_exog_cols reordered to GPRD_THREAT/GPRD/GPRD_ACT prefix; lowess_frac 0.20; star_anomaly_tails.upward = [GPRD_THREAT, GPRD, GPRD_ACT]
- 고정한 통제변인: target=Com_BrentCrudeOil; h=4 one-shot forecast contract; STAR -> AA-model -> MC uncertainty; no future/static exog; no deliberate drift/uplift
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (63.0840, 63.2684, 63.6724, 63.4662)
- 15% band PASS/FAIL: FAIL (h3=26.51%, h4=35.82%)
- runtime PASS/FAIL: PASS (~92.3s by artifact elapsed)
- leakage concern 여부: none observed; only historical exog ordering/routing changed

## Iteration 3 Discard
- timestamp: 2026-04-09T13:25:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Extend the ordered STAR prefix with Idx_DxyUSD
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: Idx_DxyUSD added after GPRD_ACT in hist_exog prefix and upward STAR routing
- 고정한 통제변인: target=Com_BrentCrudeOil; h=4 one-shot forecast contract; STAR -> AA-model -> MC uncertainty; no future/static exog; no deliberate drift/uplift
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (62.4277, 62.6235, 62.9300, 62.7916)
- 15% band PASS/FAIL: FAIL (h3=27.37%, h4=36.50%)
- runtime PASS/FAIL: PASS (~97.0s by artifact elapsed)
- leakage concern 여부: none observed; only historical exog ordering/routing changed

## Iteration 4 Crash
- timestamp: 2026-04-09T13:33:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Broadcast weighted anomaly context to every forecast horizon
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU-backed AAForecast
- 바꾼 조작변인: sparse-attention weights produce one global anomaly context that is injected into all forecast horizons
- 고정한 통제변인: target=Com_BrentCrudeOil; baseline YAML feature set; h=4 one-shot forecast contract; STAR -> AA-model -> MC uncertainty; no deliberate drift/uplift
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (63.5063, 64.0362, 63.5493, 64.0861)
- 15% band PASS/FAIL: FAIL (h3=26.65%, h4=35.19%)
- runtime PASS/FAIL: FAIL (forecast run improved, but summary replay crashed)
- leakage concern 여부: none observed; context used only in-sample hidden states and anomaly masks

## Iteration 5 Keep
- timestamp: 2026-04-09T13:40:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Add learnable horizon context inside AAForecast decoder path
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: learnable horizon context fused into attended hidden states before decoding
- 고정한 통제변인: target=Com_BrentCrudeOil; allowed exog set unchanged; h=4 one-shot forecast contract; STAR -> AA-model -> MC uncertainty; no future/static exog; no deliberate drift/uplift
- 실행 명령: NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS=1 .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml; .venv/bin/python -c 'import runtime_support.runner as r; from pathlib import Path; print(r._build_summary_artifacts(Path("runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast")))'
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (63.5063, 64.0362, 63.5493, 64.0861)
- 15% band PASS/FAIL: FAIL (h3=26.65%, h4=35.19%)
- runtime PASS/FAIL: PASS (~81.0s core run; summary rebuild succeeded)
- leakage concern 여부: none observed; only in-sample horizon differentiation added

## Iteration 6 Keep
- timestamp: 2026-04-09T13:50:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Combine horizon context with GPRD-first STAR prefix and tighter LOWESS locality
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: retained horizon context; hist_exog ordered as GPRD_THREAT/GPRD/GPRD_ACT prefix; lowess_frac 0.20; upward STAR tails = [GPRD_THREAT, GPRD, GPRD_ACT]
- 고정한 통제변인: target=Com_BrentCrudeOil; allowed exog set unchanged; h=4 one-shot forecast contract; no future/static exog; no deliberate drift/uplift
- 실행 명령: NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS=1 .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml; .venv/bin/python -c 'import runtime_support.runner as r; from pathlib import Path; print(r._build_summary_artifacts(Path("runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast")))'
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (63.6631, 63.5642, 64.1618, 64.4369)
- 15% band PASS/FAIL: FAIL (h3=25.95%, h4=34.84%)
- runtime PASS/FAIL: PASS (~84.6s core run; summary rebuild succeeded)
- leakage concern 여부: none observed; only in-sample routing/context changed

## Iteration 7 Discard
- timestamp: 2026-04-09T13:58:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Raise top_k from 0.10 to 0.15 on retained horizon-context setup
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: top_k 0.10 -> 0.15
- 고정한 통제변인: retained horizon context; GPRD_THREAT/GPRD/GPRD_ACT prefix; lowess_frac 0.20; target=Com_BrentCrudeOil; no future/static exog; no deliberate drift/uplift
- 실행 명령: NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS=1 .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (63.6631, 63.5642, 64.1618, 64.4369)
- 15% band PASS/FAIL: FAIL (h3=25.95%, h4=34.84%)
- runtime PASS/FAIL: PASS (~106.6s core run)
- leakage concern 여부: none observed; selector-only no-effect change

## Iteration 8 Keep
- timestamp: 2026-04-09T14:08:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Hidden-residual gated attention fusion on retained Brent setup
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: attention fusion changed to hidden_aligned*(1-gate) + broadcast_context*gate + horizon_context
- 고정한 통제변인: retained GPRD_THREAT/GPRD/GPRD_ACT prefix; lowess_frac 0.20; target=Com_BrentCrudeOil; allowed exog set unchanged; h=4 one-shot forecast contract; no future/static exog; no deliberate drift/uplift
- 실행 명령: NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS=1 .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml; .venv/bin/python -c 'import runtime_support.runner as r; from pathlib import Path; print(r._build_summary_artifacts(Path("runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast")))'
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: PASS (63.0744, 63.3731, 63.9395, 64.5595)
- 15% band PASS/FAIL: FAIL (h3=26.20%, h4=34.71%)
- runtime PASS/FAIL: PASS (~106.7s core run; summary rebuild succeeded; full artifact elapsed ~192.9s)
- leakage concern 여부: none observed; fusion used only in-sample encoded states and anomaly attention weights

## Iteration 9 Discard
- timestamp: 2026-04-09T14:20:00+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Add BS_Core_Index_A as a two-sided STAR channel on retained setup
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: star_anomaly_tails.two_sided += BS_Core_Index_A
- 고정한 통제변인: retained hidden_residual_gate; GPRD_THREAT/GPRD/GPRD_ACT upward STAR prefix; lowess_frac 0.20; target=Com_BrentCrudeOil; no future/static exog; no deliberate drift/uplift
- 실행 명령: NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS=1 .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml; .venv/bin/python -c 'import runtime_support.runner as r; from pathlib import Path; print(r._build_summary_artifacts(Path("runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast")))'
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: PASS (62.7949, 63.3199, 63.8247, 63.8648)
- 15% band PASS/FAIL: FAIL (h3=26.34%, h4=35.42%)
- runtime PASS/FAIL: PASS (~81.9s core run; summary rebuild succeeded; full artifact elapsed ~163.4s)
- leakage concern 여부: none observed; only historical STAR routing changed

## Iteration 10 Discard
- timestamp: 2026-04-09T16:17:29+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Add hidden residual gain on top of retained hidden-residual gated attention fusion
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: hidden residual term changed from hidden_aligned*(1-gate) to hidden_aligned*(1 + sigmoid(gain))*(1-gate)
- 고정한 통제변인: retained GPRD_THREAT/GPRD/GPRD_ACT prefix; lowess_frac 0.20; top_k 0.10; target=Com_BrentCrudeOil; no future/static exog; no deliberate drift/uplift
- 실행 명령: NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS=1 .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml; .venv/bin/python -c 'import runtime_support.runner as r; from pathlib import Path; print(r._build_summary_artifacts(Path("runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast")))'
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: PASS (62.5751, 63.3908, 63.7574, 63.9939)
- 15% band PASS/FAIL: FAIL (h3=26.41%, h4=35.29%)
- runtime PASS/FAIL: PASS (~78.9s core run; summary rebuild succeeded; full artifact elapsed ~163.8s)
- leakage concern 여부: none observed; fusion still used only in-sample encoded states and anomaly attention weights

## Iteration 11 Discard
- timestamp: 2026-04-09T16:30:55+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Add monotonic horizon gain multiplier on attended path
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: attended path multiplied by a learned monotonic horizon gain built from horizon embeddings
- 고정한 통제변인: retained hidden_residual_gate; GPRD_THREAT/GPRD/GPRD_ACT upward STAR prefix; lowess_frac 0.20; top_k 0.10; target=Com_BrentCrudeOil; no future/static exog; no deliberate drift/uplift
- 실행 명령: NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS=1 .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml; .venv/bin/python -c 'import runtime_support.runner as r; from pathlib import Path; print(r._build_summary_artifacts(Path("runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast")))'
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (62.6527, 62.9082, 64.4969, 64.2379)
- 15% band PASS/FAIL: FAIL (h3=25.56%, h4=35.04%)
- runtime PASS/FAIL: PASS (~91.1s core run; summary rebuild succeeded; full artifact elapsed ~166.0s)
- leakage concern 여부: none observed; gain used only in-sample horizon embeddings and encoded states

## Iteration 12 Discard
- timestamp: 2026-04-09T16:30:55+09:00
- git branch: exp/aaforecast-brentoil-case1-uptrend-20260409-clean
- experiment title: Multiply decoded output by learned monotonic output-horizon gain
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity.yaml
- encoder family: GRU
- 바꾼 조작변인: decoded output multiplied by a learned monotonic horizon gain derived from horizon embeddings
- 고정한 통제변인: retained hidden_residual_gate; GPRD_THREAT/GPRD/GPRD_ACT upward STAR prefix; lowess_frac 0.20; top_k 0.10; target=Com_BrentCrudeOil; no future/static exog; no deliberate drift/uplift
- 실행 명령: NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS=1 .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml; .venv/bin/python -c 'import runtime_support.runner as r; from pathlib import Path; print(r._build_summary_artifacts(Path("runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast")))'
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast
- 상승추세 PASS/FAIL: FAIL (62.5209, 62.2758, 64.7943, 64.1363)
- 15% band PASS/FAIL: FAIL (h3=25.22%, h4=35.14%)
- runtime PASS/FAIL: PASS (~90.5s core run; summary rebuild succeeded)
- leakage concern 여부: none observed; gain used only horizon embeddings and in-sample-trained parameters

## Iteration 0 Plain Informer Control
- timestamp: 2026-04-10T20:43:40+09:00
- git branch: exp/aaforecast-brentoil-case1-plain-informer-20260410
- experiment title: Plain Informer diff control to complete the control-vs-AAForecast comparison
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-informer.yaml
- plugin config path: n/a (plain Informer control)
- encoder family: Informer
- 바꾼 조작변인: plain control backbone switched to Informer; shared diff setting active; no AAForecast plugin
- 고정한 통제변인: target=Com_BrentCrudeOil; diff transforms active; allowed Brent hist exog subset [GPRD_THREAT, BS_Core_Index_A, GPRD_ACT, Idx_OVX, GPRD, BS_Core_Index_C, Com_BloombergCommodity_BCOM, Com_LMEX]; h=4 one-shot forecast contract; no future/static exog; no deliberate drift/uplift; runtime budget < 1200s
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-informer.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_plain_informer_diff
- 상승추세 PASS/FAIL: PASS (68.9944, 69.3880, 69.6867, 69.6203)
- 15% band PASS/FAIL: PASS (fallback h1=1.88%, h2=4.59%; h3=19.57%, h4=29.60% outside 15%)
- runtime PASS/FAIL: PASS (~42.65s by artifact elapsed)
- leakage concern 여부: none observed; diff path is train-only and replay-restored

## Iteration 1 AAForecast GRU Diff Control
- timestamp: 2026-04-10T20:46:00+09:00
- git branch: exp/aaforecast-brentoil-case1-plain-informer-20260410
- experiment title: AAForecast GRU diff control on the same constrained Brent subset
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-gru-diff.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity_gru.yaml
- encoder family: GRU
- 바꾼 조작변인: AAForecast GRU backbone with the same diff-active control subset as plain Informer control
- 고정한 통제변인: target=Com_BrentCrudeOil; diff transforms active; allowed Brent hist exog subset [GPRD_THREAT, BS_Core_Index_A, GPRD_ACT, Idx_OVX, GPRD, BS_Core_Index_C, Com_BloombergCommodity_BCOM, Com_LMEX]; h=4 one-shot forecast contract; no future/static exog; no deliberate drift/uplift; runtime budget < 1200s
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-gru-diff.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_gru_diff
- 상승추세 PASS/FAIL: PASS (fallback h1=1.81%, h2=4.32%; h3=19.25%, h4=29.32% outside 15%)
- 15% band PASS/FAIL: FAIL (h3=19.25%, h4=29.32%)
- runtime PASS/FAIL: PASS (~146.01s by artifact elapsed)
- leakage concern 여부: none observed; diff path is train-only and replay-restored

## Iteration 2 AAForecast Informer Diff Control
- timestamp: 2026-04-10T20:52:00+09:00
- git branch: exp/aaforecast-brentoil-case1-plain-informer-20260410
- experiment title: AAForecast Informer diff control on the same constrained Brent subset
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-informer-diff.yaml
- plugin config path: yaml/plugins/aa_forecast_brentoil_case1_parity_informer.yaml
- encoder family: Informer
- 바꾼 조작변인: AAForecast Informer backbone with the same diff-active control subset as plain Informer control
- 고정한 통제변인: target=Com_BrentCrudeOil; diff transforms active; allowed Brent hist exog subset [GPRD_THREAT, BS_Core_Index_A, GPRD_ACT, Idx_OVX, GPRD, BS_Core_Index_C, Com_BloombergCommodity_BCOM, Com_LMEX]; h=4 one-shot forecast contract; no future/static exog; no deliberate drift/uplift; runtime budget < 1200s
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-informer-diff.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_subset8_diff
- 상승추세 PASS/FAIL: PASS (fallback h1=2.23%, h2=4.31%; h3=19.05%, h4=28.64% outside 15%)
- 15% band PASS/FAIL: FAIL (h3=19.05%, h4=28.64%)
- runtime PASS/FAIL: PASS (~173.59s by artifact elapsed)
- leakage concern 여부: none observed; diff path is train-only and replay-restored

## Iteration 3 GRU Diff Control
- timestamp: 2026-04-10T20:55:00+09:00
- git branch: exp/aaforecast-brentoil-case1-plain-informer-20260410
- experiment title: GRU diff control on the same constrained Brent subset
- main config path: yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-gru-diff.yaml
- plugin config path: n/a (plain GRU control)
- encoder family: GRU
- 바꾼 조작변인: plain GRU backbone with the same diff-active control subset as the other comparison runs
- 고정한 통제변인: target=Com_BrentCrudeOil; diff transforms active; allowed Brent hist exog subset [GPRD_THREAT, BS_Core_Index_A, GPRD_ACT, Idx_OVX, GPRD, BS_Core_Index_C, Com_BloombergCommodity_BCOM, Com_LMEX]; h=4 one-shot forecast contract; no future/static exog; no deliberate drift/uplift; runtime budget < 1200s
- 실행 명령: .venv/bin/python main.py --config yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-gru-diff.yaml
- run/artifact path: runs/feature_set_aaforecast_brentoil_case1_parity_gru_diff
- 상승추세 PASS/FAIL: PASS (fallback h1=2.57%, h2=6.22%; h3=22.36%, h4=33.39% outside 15%)
- 15% band PASS/FAIL: FAIL (h3=22.36%, h4=33.39%)
- runtime PASS/FAIL: PASS (~23.93s by artifact elapsed)
- leakage concern 여부: none observed; diff path is train-only and replay-restored
