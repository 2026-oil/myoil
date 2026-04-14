# AAForecast no-retrieval spike capture 진행 노트

## 1. 현재 목표
사용자 목표는 다음이다.

- `retrieval=false`를 유지하면서도
- AA-Forecast 내부 아키텍처만으로
- 마지막 fold에서 Brent 예측이 retrieval-enabled 때처럼 더 높은 amplitude를 내도록 개선
- 이상적 목표:
  - `h2 > h1`
  - 최신 fold 기준 `h1`은 80대, `h2`는 90대 수준
  - 가능하면 실제값 대비 ±15% 이내

## 2. 절대 유지한 제약

- retrieval 자체 사용 금지 (`retrieval=false`)
- loss weighting 변경 금지
- horizon-specific bonus / uplift / drift 금지
- leakage 금지
- 허용 변수만 사용
- 공통 철학: **retrieval 철학은 internalize하되 retrieval plugin 자체는 쓰지 않음**

## 3. 지금까지 얻은 가장 중요한 결론

### 3.1 이미 해결된 부분
초기의 고질 문제였던
- `h1`, `h2`가 거의 평평하게 나오는 현상
- `h2 > h1`이 자주 깨지는 현상
은 상당 부분 완화되었다.

현재는 대부분 실험에서:
- `h2 > h1`는 유지되고
- 문제는 **방향성이 아니라 amplitude compression** 이다.

즉,
- 모델이 “오를 가능성”은 안다.
- spike 구조도 어느 정도 안다.
- 하지만 최신 fold에서 **충분히 크게 못 올린다.**

### 3.2 retrieval 철학 중 이미 내부화된 부분
fresh diagnostic 결과,
**retrieval의 핵심인 eventful point 선택 자체는 이미 내부에서 상당히 sharp하게 구현되어 있다.**

artifact:
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid7/aa_forecast/uncertainty/20260223T000000.decoder_debug_report.md`

핵심 관찰:
- `weight_mass_top1 ~= 1.0`
- `selected_top_index = 17`

즉,
- internal top1 memory selection은 이미 매우 강함
- 문제는 **무엇을 볼지 선택**이 아니라
- **선택한 정보를 decoder가 amplitude로 transport 못하는 것**이다.

### 3.3 현재 가장 유력한 병목
가장 중요한 decoder diagnostic artifact:
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid2debug/aa_forecast/uncertainty/20260223T000000.decoder_debug_report.md`

여기서 확인된 핵심:
- `level` + `level_shift` 가 공통적으로 음수
- `local_path`, `global_path`, `delta_path` 도 최신 fold에서 음수 기여가 큼
- `event_delta × gate` 만이 주요 양의 branch
- `path_amplitude`는 이미 1보다 큼

결론:
- amplification scalar가 부족한 문제가 아니다.
- **amplification 전에 residual 합이 작게 남는 구조**가 본질 병목이다.

## 4. 현재까지 best run 정리

### 4.1 최신 fold 기준 best no-retrieval
가장 좋은 최신 fold 결과는 아직 아래 run이다.

- run:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid2/summary/result.csv`
- latest fold:
  - `h1 = 75.3761`
  - `h2 = 78.9167`
- 평가:
  - `h2 > h1 = PASS`
  - 하지만 여전히 retrieval-like 80s/90s에는 못 미침

### 4.2 역사적 spike replay 기준 best
history replay는 오히려 후속 구조에서 더 좋아진 경우가 있다.

- hybrid2 replay:
  - `mean_ape_h1 = 0.0644`
  - `mean_ape_h2 = 0.0782`
  - `h2_gt_h1_rate = 0.8750`
- hybrid15 replay:
  - `mean_ape_h1 = 0.0612`
  - `mean_ape_h2 = 0.0649`
  - `h2_gt_h1_rate = 1.0000`

즉,
- historical spike understanding은 좋아졌지만
- 최신 fold amplitude는 여전히 막힌다.

## 5. retrieval-enabled reference
사용자가 특히 중요하게 본 retrieval 설정:

```yaml
retrieval:
  enabled: true
  top_k: 1
  event_score_threshold: 400.0
  min_similarity: 0.35
  blend_max: 1.0
  use_uncertainty_gate: false
  use_shape_key: false
  use_event_key: true
  event_score_log_bonus_alpha: 0.15
  event_score_log_bonus_cap: 0.1
```

이 설정이 켜진 경우 사용자가 체감한 것은:
- 예측 amplitude가 더 높아지고
- 대체로 80s/90s 방향의 결과가 나온다는 점

현재까지의 결론은:
- retrieval의 “selection”은 내부화 가능/상당 부분 달성
- retrieval의 “transported weighted future pattern”은 아직 decoder 내부에서 제대로 재현 못 함

## 6. 구조 변경 실험 로그 요약

### 6.1 효과가 있었던 핵심 변경
#### Hybrid2 계열
핵심 아이디어:
- event/path latent를 `tanh` 대신 `GELU`로 unsaturate
- pooled memory를 decoder input 전체에 직접 broadcast하지 않음
- non-star exogenous STAR activity를 `event_trajectory` 경로로 복원

왜 의미 있었나:
- LMEX / BCOM / OVX / BS-core 쪽 burst가 path decoder로 실제 전달되기 시작함
- historical spike replay가 크게 개선됨

---

### 6.2 효과 없거나 폐기한 시도들
아래는 모두 fresh runtime + replay + tests까지 확인 후 폐기한 시도들이다.

#### Hybrid4
- 내용: dynamic internal top-k memory
- 결과: latest fold 악화

#### Hybrid5
- 내용: regime tail summary를 event_trajectory에 추가
- 결과: latest fold 악화

#### Hybrid6
- 내용: anchor-scaled internal return branch
- 결과: retrieval-like return transport 아이디어였지만 악화

#### Hybrid7
- 내용: retrieval-like memory transport gate
- 결과: selection은 sharp하지만 decoder drag를 뒤집지 못함

#### Hybrid8
- 내용: learned negative-drag suppression gate
- 결과: latest fold 크게 악화

#### Hybrid9
- 내용: deterministic retrieval-strength gate
- 결과: hybrid2보다 악화

#### Hybrid10
- 내용: selected token 이후 hidden-state continuation 주입
- 결과: retrieval continuation template 주입도 악화

#### Hybrid11
- 내용: pooled memory를 baseline head에서 제거하고 continuation/shock branch에만 사용
- 결과: hybrid10보다는 회복, 그래도 hybrid2보다 낮음

#### Hybrid12
- 내용: retrieval-strength로 level/level_shift 포함 음수항 attenuation
- 결과: 악화

#### Hybrid13
- 내용: attenuation을 더 좁게 level/level_shift에만 적용
- 결과: 악화

#### Hybrid14
- 내용: baseline head와 spike head 분리 시작점
  - `level_head`는 baseline context
  - `global_path`, `delta_path`, `spike_expert`는 pooled-memory가 있는 spike context
- latest fold:
  - `h1 = 75.3291`
  - `h2 = 77.7950`
- 최신 fold는 hybrid2보다 약간 낮지만,
  history replay는 매우 강함:
  - `mean_ape_h1 = 0.0658`
  - `mean_ape_h2 = 0.1010`
  - `h2_gt_h1_rate = 1.0`

#### Hybrid15
- 내용: pooled-memory를 path branch(global/delta/spike_expert)에만 더 직접 주는 구조
- latest fold:
  - `h1 = 74.4411`
  - `h2 = 76.7112`
- replay는 매우 좋았음:
  - `mean_ape_h1 = 0.0612`
  - `mean_ape_h2 = 0.0649`
  - `h2_gt_h1_rate = 1.0`
- 하지만 최신 fold amplitude는 오히려 더 낮아짐

## 7. 최신까지의 핵심 해석

### 7.1 지금까지 falsified 된 가설
다음은 “그럴듯했지만 지금은 핵심 병목이 아닌 것”으로 봄.

1. **retrieval처럼 과거 event point를 못 찾는 게 문제다**
   - 아님
   - 내부 top1 selection은 이미 sharp함

2. **amplitude가 낮은 건 path_amplitude 부족 때문이다**
   - 아님
   - path_amplitude는 보통 1보다 큼

3. **transport gate만 넣으면 retrieval-like amplitude가 나올 것이다**
   - 아님
   - 여러 gate류가 오히려 decoder를 destabilize 함

4. **selected token 뒤 continuation hidden state를 넣으면 해결된다**
   - 아님
   - direct continuation injection은 latest fold를 개선하지 못함

### 7.2 현재 가장 유력한 해석
현 시점에서 가장 유력한 설명은 다음과 같다.

- retrieval는 외부 memory에서 **이미 고른 패턴 자체를 blend**해준다.
- 반면 no-retrieval internal 모델은
  - 선택은 잘해도
  - 최종적으로는 learned decoder heads가 값을 만들어야 한다.
- 그런데 이 decoder heads가 현재
  - baseline을 음수로 끌고
  - local/global/delta path도 음수 기여를 많이 내기 때문에
  - spike branch의 positive contribution이 상쇄된다.

즉,
**retrieval-like transport 실패의 본질은 “memory lookup 실패”가 아니라 “decoder head parameterization failure”** 다.

## 8. 지금 코드 상태
현재 working tree는 여러 실험/diagnostic이 함께 들어있는 dirty state다.
다만 검증은 통과했다.

검증 evidence:
- `uv run pytest --no-cov tests/test_aaforecast_star_precompute.py tests/test_aaforecast_adapter_contract.py tests/test_aaforecast_backbone_faithfulness.py -q`
  - `46 passed`
- `uv run pytest --no-cov tests/test_aa_forecast_plugin_contracts.py -q -k 'predict_aa_forecast_fold or uncertainty'`
  - `4 passed`
- validate-only smoke 다수 PASS

관련 핵심 수정 파일:
- `neuralforecast/models/aaforecast/model.py`
- `plugins/aa_forecast/runtime.py`
- `tests/test_aaforecast_adapter_contract.py`
- `tests/test_aaforecast_star_precompute.py`

관련 분석 스크립트:
- `scripts/analyze_aaforecast_spike_diagnostics.py`
- `scripts/replay_trained_aaforecast_spike_windows.py`

## 9. 중요한 artifact 목록

### 최신 fold / best run
- best latest no-retrieval:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid2/summary/result.csv`

### 핵심 diagnostic
- decoder ceiling diagnostic:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid2debug/aa_forecast/uncertainty/20260223T000000.decoder_debug_report.md`
- retrieval-behavior internalization diagnostic:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid7/aa_forecast/uncertainty/20260223T000000.decoder_debug_report.md`

### spike replay 참고
- hybrid2 replay:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid2/aa_forecast/diagnostics/trained_model_spike_window_replay.md`
- hybrid14 replay:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid14/aa_forecast/diagnostics/trained_model_spike_window_replay.md`
- hybrid15 replay:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid15/aa_forecast/diagnostics/trained_model_spike_window_replay.md`

## 10. 지금 시점의 next best move
현재까지의 결과를 바탕으로, 다음 단계는 다음 중 하나가 가장 합리적이다.

1. **decoder head parameter distribution / learned bias 직접 진단**
   - level_head
   - level_shift_head
   - global_head
   - delta_head
   - local_head
   - spike_expert_head

2. **head 구조를 더 좁게 분리**
   - baseline head는 anchor-neutral / low-variance 역할만
   - spike head는 memory-conditioned shock transport만 담당
   - 단, gate로 posthoc attenuation 하지 말고 구조적으로 분리

3. **shock branch를 더 강하게 하는 것이 아니라**
   - baseline negative offset이 왜 반복적으로 학습되는지
   - 그 inductive bias를 깨는 설계로 가야 함

한 줄 결론:

> retrieval 철학은 이미 “선택” 단계에서는 꽤 잘 내부화되었다.  
> 남은 문제는 **decoder가 그 선택을 retrieval처럼 큰 amplitude로 번역하지 못하는 구조**다.

#### Hybrid16
- 내용: baseline head와 spike head를 더 명확히 분리
  - `level`, `level_shift`, `normal_expert`는 baseline context 중심
  - `spike_expert`는 pooled-memory가 있는 spike context 사용
  - latest fold 기준 최신 성능이 크게 좋아짐
- latest fold:
  - `h1 = 76.8504`
  - `h2 = 79.6834`
- replay:
  - `mean_ape_h1 = 0.0766`
  - `mean_ape_h2 = 0.0877`
  - `h2_gt_h1_rate = 1.0`
- 해석:
  - 최신 fold amplitude 측면에서는 지금까지 no-retrieval 중 가장 retrieval-like 방향으로 이동한 구조
  - 다만 h2는 아직 80 미만이고, replay는 hybrid2/15보다 약간 불리함

#### Hybrid17
- 내용: event_bias를 non-negative (`softplus`)로 강제하여 spike branch uplift만 남기기
- latest fold:
  - `h1 = 74.0538`
  - `h2 = 76.7421`
- 결과: 악화
- 해석:
  - event_bias를 무조건 양수화하는 것은 decoder 균형을 깨뜨려 오히려 성능 저하를 유발


#### Hybrid18
- 내용: `spike_expert`를 unconstrained residual이 아니라 **positive cumulative uplift expert**로 변경
- latest fold:
  - `h1 = 75.3314`
  - `h2 = 78.9124`
- replay:
  - `mean_ape_h1 = 0.0779`
  - `mean_ape_h2 = 0.1037`
  - `h2_gt_h1_rate = 1.0000`
- 해석:
  - spike expert를 양의 shock transport expert로 바꾸는 방향 자체는 맞다는 fresh evidence를 제공
  - latest fold h2는 hybrid2와 거의 비슷한 수준까지 도달
  - 하지만 h1과 replay 전반은 hybrid2보다 약간 불리하여, 아직 완전한 best replacement는 아님


#### Hybrid19
- 내용: split-head 구조 위에서 `memory_signal`로 `expert_gate`를 직접 bias하여 positive spike expert transport를 강화
- latest fold:
  - `h1 = 75.9038`
  - `h2 = 79.5477`
- replay:
  - `mean_ape_h1 = 0.0851`
  - `mean_ape_h2 = 0.1254`
  - `h2_gt_h1_rate = 0.8750`
- 해석:
  - spike expert를 실제 양의 transport branch로 더 강하게 쓰는 방향은 맞고, h1은 올라갔다.
  - 하지만 replay generalization이 악화되어 현재 세팅은 overfit 성향이 강함.


#### Hybrid35
- 내용: 기존 additive head family 대신, **trajectory-GRU 기반 shock generator**를 도입
- latest fold:
  - `h1 = 74.9785`
  - `h2 = 77.6657`
- replay:
  - `mean_ape_h1 ≈ 0.0612`
  - `mean_ape_h2 ≈ 0.0902`
  - `h2_gt_h1_rate = 1.0000`
- 해석:
  - 최신 fold best는 아니지만, current informer-wrapper local tweak들보다 **다른 generator family** 쪽이 더 의미 있는 신호를 보였다.
  - 즉 다음 단계는 같은 head-mixing family 미세조정이 아니라, hybrid35류의 새 stage-2 generator lane을 더 파고드는 것이 합리적이다.


#### Hybrid49
- 내용: trajectory-GRU family 위에 **learned positive trajectory template bank**를 추가
- latest fold:
  - `h1 = 75.5900`
  - `h2 = 78.4709`
- 해석:
  - latest fold amplitude는 hybrid35보다 다소 좋아질 수 있지만, replay generalization이 매우 불안정했다.
  - 즉 template-bank family는 흥미롭지만 현재 형태로는 brittle해서 즉시 채택하기 어렵다.


#### Hybrid50
- 내용: learned positive trajectory template bank의 스케일을 0.25x로 줄여 regularize
- latest fold:
  - `h1 = 75.1948`
  - `h2 = 77.6701`
- 해석:
  - template-bank family의 불안정성은 일부 줄어들지만, 여전히 hybrid16/19의 latest-fold 수준은 넘지 못한다.
  - 즉 template-bank family는 research direction으로는 흥미롭지만, 현재 최적 해법은 아니다.

## 11. 최신까지의 업데이트된 결론
현재 시점의 가장 중요한 업데이트는 다음이다.

1. **best latest-fold no-retrieval run은 hybrid16**
   - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_hybrid16/summary/result.csv`
   - `h1 = 76.8504`
   - `h2 = 79.6834`

2. retrieval 철학을 모델링할 때,
   - selection을 더 sharpen 하는 것보다
   - **baseline head와 spike head를 구조적으로 분리하는 것**이 더 효과적이었다.

3. 하지만 아직도
   - 최신 fold 기준 80대/90대에는 도달하지 못했고
   - 특히 h2 amplitude가 부족하다.

4. 따라서 다음 최우선 방향은:
   - baseline / spike head 분리를 더 정교하게 하되
   - 단순 gate, attenuation, continuation injection이 아니라
   - **head별 역할 분리를 더 명시적**으로 만드는 것

즉,
> retrieval-like behavior를 더 잘 internalize하려면  
> “무엇을 볼지”보다  
> **“선택된 spike memory를 어떤 head가 얼마나 책임지고 번역할지”**를 더 강하게 분리해야 한다.

## 12. 2026-04-14 새 실패 증거 업데이트

이번 턴에서 새로 확인한 것은 아래 3가지다.

1. **learned prototype-bank analogue path**
   - artifact:
     - `runs/iter_20260414_053412_aa_informer_proto_bank_v1/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
   - result:
     - `h1 = 74.8102`
     - `h2 = 77.1377`
   - 해석:
     - retrieval-like “analogue bank”를 모델 안에 직접 두는 발상 자체는 가능하지만,
     - static prototype bank는 current-window spike transport를 frontier 수준까지 끌어올리지 못했다.

2. **anchor-scaled prototype return path**
   - artifact:
     - `runs/iter_20260414_053904_aa_informer_proto_bank_v2/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
   - result:
     - `h1 = 73.3998`
     - `h2 = 75.1361`
   - 해석:
     - anchor scale을 analogue path에 직접 곱하는 방식은
     - “return-space로 바꾸면 해결될 것”이라는 가설을 지지하지 못했다.
     - 즉 **anchor-scale alone is not the missing ingredient**.

3. **top-k internal memory-bank cross-attention transport**
   - artifact:
     - `runs/iter_20260414_054102_aa_informer_memory_transport_v1/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
   - result:
     - `h1 = 74.7108`
     - `h2 = 77.3587`
   - 해석:
     - pooled top1 collapse를 줄이기 위해 top-k internal token bank를 horizon-wise cross-attention으로 transport했지만,
     - 결과는 prototype-bank보다는 낫고 hybrid16보다는 못하다.
     - 즉 **collapse 문제가 일부 있더라도, 단순 token-bank transport만으로는 frontier 갱신이 되지 않는다.**

### 업데이트된 결론
- 현재 frontier는 여전히
  - latest-fold: `hybrid16`
  - replay/generalization: `hybrid35`
- 이번 새 시도들로 인해 더 강해진 결론은:
  1. **selection 자체는 이미 충분히 sharp하다**
  2. **transport는 중요하지만, “bank를 하나 더 붙이는 것”만으로는 해결되지 않는다**
  3. 다음은 bank/prototype 추가보다,
     - baseline family와 spike family의 **학습 의미 자체를 더 다르게 만드는 generator semantics**
     - 즉 **normal path는 level/continuation, spike path는 signed cumulative shock** 처럼 더 명시적인 역할 분리가 필요하다.

## 13. semantic generator family 업데이트

이번 턴에서 가장 의미 있었던 새 결과는 아래다.

### Semantic Spike Generator v1
- artifact:
  - `runs/iter_20260414_054812_aa_informer_semantic_spike_v1/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 74.9696`
  - `h2 = 78.5254`
- 의미:
  - bank/prototype를 추가하는 것보다
  - **normal continuation path** 와 **signed cumulative spike path** 를 의미적으로 나누는 것이 더 유망했다.
  - 즉 다음 단계는 “memory bank를 더 정교하게 붙이는 것”보다
  - **spike generator family를 더 순수하게 다듬는 방향**이 맞다.

### Semantic Spike Generator v2
- artifact:
  - `runs/iter_20260414_055111_aa_informer_semantic_spike_v2/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 73.9857`
  - `h2 = 76.9992`
- 의미:
  - semantic spike family 위에 다시 analogue transport family를 섞자 성능이 후퇴했다.
  - 해석상 중요한 점은:
    - **semantic split path와 old analogue transport path는 서로 잘 안 섞인다**
    - 즉 다음 iteration은 v1처럼 semantic family를 유지하고,
    - 내부 세부(예: baseline drag 억제, spike seed/context 구조)를 다듬는 편이 낫다.

## 14. 지금 기준 다음 방향

현재 가장 합리적인 다음 방향은 다음이다.

1. **semantic spike family 유지**
   - v1을 current basis로 둔다.

2. **baseline path를 더 local continuation 의미로 제한**
   - baseline이 spike를 잡으려고 하지 않게 하고,
   - spike path가 shock amplitude를 더 책임지게 한다.

3. **spike seed/input 쪽에서 STAR target/non-star regime의 signed contrast를 더 직접 반영**
   - selection sharpen이 아니라
   - spike generator 내부에서 signed shock semantics를 더 잘 보이게 해야 한다.

## 15. semantic generator family 최신 업데이트

### Semantic Spike Generator v3
- artifact:
  - `runs/iter_20260414_060302_aa_informer_semantic_spike_v3/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 74.7537`
  - `h2 = 76.6351`
- 해석:
  - raw signed STAR direction을 별도 projector로 semantic gate에 주입하는 방식은,
  - direction을 더 잘 알게 만들 것 같았지만 실제로는 dispersion과 amplitude를 같이 낮췄다.
  - 즉 **direction 문제는 맞지만, raw scalar bias 추가는 해법이 아니다.**

### Semantic Spike Generator v1 refresh
- artifact:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 76.5263`
  - `h2 = 80.5961`
- 해석:
  - pure semantic family를 다시 basis로 두고 재실행하자,
  - 이전 semantic v1 archive보다 훨씬 좋은 결과가 나왔다.
  - 특히 `h2`가 처음으로 **80선 위**까지 올라갔다.
  - 현재 기준으로 semantic family는:
    - `h1`는 아직 hybrid16보다 약간 낮지만
    - `h2`와 fold MSE는 더 좋아졌다.

## 16. 현재 활성 기준선

현재 active basis는 다음이다.
- code basis: **pure semantic spike family**
- active run root:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer`
- latest fold:
  - `h1 = 76.5263`
  - `h2 = 80.5961`

즉 지금은
- `h2 > h1` 유지
- `h1`는 ±15% 이내 진입
- `h2`는 아직 ±15% 바깥이지만, 이전 frontier보다 상당히 가까워짐

### 지금 남은 가장 중요한 병목
semantic family에서도 standalone debug 기준으로는 아직
- `semantic_spike_direction`이 강하게 positive로 치우치지 않고
- positive/negative cumulative branch가 서로 상쇄되는 문제가 남아있다.

그래서 다음은
1. **semantic family 유지**
2. **negative spike branch의 불필요한 상쇄를 줄이고**
3. **positive spike branch가 late-step(h2)까지 cumulative하게 유지되도록**
정교화하는 방향이 가장 타당하다.

## 17. cancellation 완화 실험 업데이트

이번 턴에서 확인한 핵심은 다음이다.

### Negative-gate variant
- last fold:
  - `h1 = 76.4147`
  - `h2 = 80.5214`
- 해석:
  - negative correction을 semantic context + memory signal로 억제하는 방향은 완전히 틀리진 않았다.
  - 하지만 pure semantic family를 뚜렷하게 넘지는 못했다.

### Split pos/neg branch variant
- last fold:
  - `h1 = 75.6897`
  - `h2 = 78.6161`
- 해석:
  - positive/negative branch를 step path부터 분리하면 더 좋아질 것 같았지만,
  - 현재 구현에서는 positive transport 자체가 약해졌다.
  - 즉 **branch split 자체보다, pure semantic branch 안에서 cancellation을 다루는 편이 낫다.**

### Pure semantic restore rerun
- artifact:
  - `runs/iter_20260414_062106_aa_informer_semantic_spike_restore/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 76.2853`
  - `h2 = 80.0916`
- 해석:
  - pure semantic family는 여전히 가장 안정적인 상위 계열이다.
  - run variance는 있지만, 현재 repo-local no-retrieval 조건에서 가장 일관되게 h2를 80 부근까지 올린다.

## 18. 현재 최우선 방향 재정리

다음은 split branch 확대가 아니라,

1. **pure semantic family 유지**
2. **negative cancellation을 완전히 별도 branch로 떼기보다는, same family 안에서 억제**
3. **run variance를 줄이면서 h2 80+를 더 안정화**

쪽이 맞다.

## 19. positive event-seed variant 결과

- artifact:
  - `runs/iter_20260414_063152_aa_informer_positive_seed_variant/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 75.0063`
  - `h2 = 77.1648`
- 해석:
  - positive event-path seed를 semantic spike hidden에 직접 넣는 방식은 예상과 달리 성능을 낮췄다.
  - 따라서 다음 방향은 extra seed injection이 아니라 pure semantic family 내부의 안정화 쪽이다.

## 20. semantic baseline-regime gate 결과

- last fold:
  - `h1 = 74.7342`
  - `h2 = 76.7089`
- 해석:
  - baseline path를 anomaly regime에서 positive continuation으로 바꾸는 접근은 오히려 pure semantic family보다 크게 나빠졌다.
  - 따라서 baseline 쪽을 크게 바꾸기보다 spike family 안정화에 더 집중하는 편이 맞다.

## 21. semantic cleanup v1 결과

- artifact:
  - `runs/iter_20260414_065245_aa_informer_semantic_cleanup_v1/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 76.3334`
  - `h2 = 80.1706`
- 해석:
  - split-branch residue를 걷어내고 pure semantic family를 단순화하자 성능이 회복되었다.
  - 현재는 architecture를 더 늘리는 것보다 semantic spike family를 lean하게 유지하는 편이 낫다.

## 22. semantic uncertainty selector v1 결과

- artifact:
  - `runs/iter_20260414_071526_aa_informer_semantic_selector_v1/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 76.2570`
  - `h2 = 80.4283`
- 해석:
  - uncertainty selector가 semantic spike support와 direction을 읽도록 바꾸자 h2가 다시 80선 위로 복원되었다.
  - 즉 현재는 decoder를 더 복잡하게 만들기보다 clean semantic family + semantic-aware selector 조합이 더 효율적이다.

## 23. semantic uncertainty selector v2 결과

- artifact:
  - `runs/iter_20260414_072845_aa_informer_semantic_selector_v2/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 76.7550`
  - `h2 = 80.9615`
- 해석:
  - lean semantic spike family를 유지한 채 selector만 semantic-aware하게 다듬는 것이 현재까지 가장 효율적인 개선 경로였다.
  - 지금은 decoder 구조 확장보다 selector stability tuning 쪽이 계속 유리하다.

## 24. negative-weight 0.9 결과

- artifact:
  - `runs/iter_20260414_073642_aa_informer_negative_weight_0p9/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/summary/result.csv`
- last fold:
  - `h1 = 77.4647`
  - `h2 = 82.7683`
- 해석:
  - decoder를 더 복잡하게 만들지 않고 negative spike cancellation만 소폭 줄이는 것이 현재까지 가장 효과적이었다.
  - 이제 다음은 같은 계열에서 작은 cancellation 계수 탐색(예: 0.85~0.95)이나 selector repeatability 확인이 맞다.

## 25. negative-weight bracket 결과

- 0.85: `h1 = 75.3040`, `h2 = 77.7913` -> 악화
- 0.95: `h1 = 76.2975`, `h2 = 80.0363` -> 0.9보다 약함
- 0.9 rerun: `h1 = 76.7348`, `h2 = 81.5516` -> current keep

결론적으로 0.9가 현재 local optimum에 가장 가깝다.
또한 exp 시트 append도 완료했다 (exp!A221:Z221).

## 26. repeatability check 결과

- stability_a: `h1 = 76.3739`, `h2 = 80.1664`
- stability_b: `h1 = 76.3360`, `h2 = 80.1704`
- reference frontier rerun: `h1 = 76.7348`, `h2 = 81.5516`

해석: 같은 설정을 다시 돌리면 h2>h1과 80선은 유지되지만, amplitude variance가 여전히 존재한다.
즉 다음은 구조 확장보다 variance reduction / selector robustness가 핵심이다.

## 27. adaptive memory-gated negative cancellation 결과

- last fold: `h1 = 75.2336`, `h2 = 77.6748`
- 해석: memory signal로 negative cancellation을 adaptive하게 줄이는 방식은 기대와 달리 크게 악화되었다.
- 결론: constant 0.9 negative weight가 여전히 더 낫다.

## 28. repeatability batch c/d/e 결과

- stability_c: `76.6463 / 81.1915`
- stability_d: `76.0032 / 79.2821`
- stability_e: `75.4444 / 78.0413`
- 해석: 같은 설정에서도 amplitude variance가 매우 크며, 남은 메인 병목은 구조보다 repeatability다.

## 29. selector semantic tolerance 0.20 결과

- current: `75.7179 / 79.0633`
- stability_c: `75.9516 / 79.2304`
- 해석: tolerance를 0.20으로 넓히면 variance는 줄지 않고 오히려 selected path quality가 낮아졌다. 0.15가 더 안전하다.

## 30. repeatability batch f/g/h 결과

- stability_f: `76.3790 / 80.2145`
- stability_g: `74.8054 / 76.6875`
- stability_h: `75.8280 / 78.9240`
- aggregate a~h: mean h1=`75.9770`, std h1=`0.5674`, mean h2=`79.3347`, std h2=`1.3446`
- 해석: 현재 핵심 병목은 구조보다 stochastic variance이며, 0.9+semantic selector family 안에서 반복성 확보가 제일 중요하다.

## 31. repeatability batch i/l 결과

- stability_i: `75.9114 / 79.1353`
- stability_j: `75.2790 / 77.7656`
- stability_k: `76.1822 / 79.9134`
- stability_l: `76.0635 / 79.4522`
- aggregate a~l: mean h1=`75.9377`, std h1=`0.5081`, mean h2=`79.2453`, std h2=`1.1979`
- 해석: variance는 약간 줄어들었지만 여전히 크고, 새로운 run이 archived frontier 82.77을 넘지는 못했다.

## 32. repeatability batch m/p 결과

- stability_m: `75.5097 / 78.3046`
- stability_n: `75.6927 / 78.6461`
- stability_o: `75.8922 / 79.0425`
- stability_p: `75.8693 / 79.4173`
- aggregate a~p: mean h1=`75.8885`, std h1=`0.4548`, mean h2=`79.1472`, std h2=`1.0718`
- 해석: variance는 조금 줄어들었지만 분포 중심은 여전히 archived best보다 낮다. 현재 문제는 구조보다 확률/분산 문제다.

## 33. repeatability batch q/t 결과

- stability_q: `76.7909 / 81.6651`
- stability_r: `76.2442 / 79.8496`
- stability_s: `76.5109 / 80.5211`
- stability_t: `76.2824 / 80.3206`
- aggregate a~t: mean h1=`76.0022`, std h1=`0.4761`, mean h2=`79.4356`, std h2=`1.1579`
- 해석: repeat harvesting을 더 하자 평균과 분산이 약간 좋아졌고, q는 81.67로 강한 재현 증거가 됐다. 하지만 archived best 82.77과 15% h2 band는 여전히 못 넘었다.

## 34. repeatability batch u/x 결과

- stability_u: `76.9828 / 81.7159`
- stability_v: `75.8526 / 78.9989`
- stability_w: `75.7577 / 78.7321`
- stability_x: `76.0277 / 79.5711`
- aggregate a~x: mean h1=`76.0277`, std h1=`0.4814`, mean h2=`79.4887`, std h2=`1.1664`
- 해석: stability_u가 81.72까지 올라와 repeat ceiling은 조금 높아졌지만, archived best 82.77은 여전히 upper-tail 수준이다.

## 35. repeatability batch y/ab 결과

- stability_y: `76.4868 / 80.4107`
- stability_z: `76.1170 / 79.5613`
- stability_aa: `77.0258 / 81.9021`
- stability_ab: `74.6283 / 76.2972`
- aggregate a~ab: mean h1=`76.0330`, std h1=`0.5585`, mean h2=`79.4964`, std h2=`1.3298`
- 해석: repeat ceiling은 81.90까지 올랐지만 archived best 82.77에는 못 닿았다. 동시에 low tail도 여전해서 variance 문제가 더 분명해졌다.

## 36. repeatability batch ac/af 결과

- stability_ac: `75.5757 / 78.4296`
- stability_ad: `76.2470 / 80.4001`
- stability_ae: `76.9014 / 81.5439`
- stability_af: `75.8591 / 78.9918`
- aggregate a~af: mean h1=`76.0471`, std h1=`0.5525`, mean h2=`79.5396`, std h2=`1.3212`
- 해석: ae가 다시 81.54를 보여 repeat upper band는 유지되지만, archived best 82.77은 계속 upper-tail로 남아 있다.

## 37. repeatability batch ag/aj 결과

- stability_ag: `74.9881 / 77.0456`
- stability_ah: `75.0092 / 77.1999`
- stability_ai: `76.1953 / 80.0535`
- stability_aj: `74.8306 / 76.7033`
- aggregate a~aj: mean h1=`75.9592`, std h1=`0.6053`, mean h2=`79.3408`, std h2=`1.4379`
- 해석: lower tail이 다시 길어지며, 지금 남은 병목이 variance라는 점이 더 강해졌다.

## 38. repeatability batch ak/an 결과

- stability_ak: `75.5957 / 78.5918`
- stability_al: `76.2939 / 79.9685`
- stability_am: `74.7828 / 76.6321`
- stability_an: `76.8644 / 81.4215`
- aggregate a~an: mean h1=`75.9517`, std h1=`0.6252`, mean h2=`79.3220`, std h2=`1.4752`
- 해석: an이 다시 81대에 올라오지만 am이 lower tail을 열어 variance가 오히려 넓어졌다. 이 lane의 harvest-only 반복은 한계가 분명하다.

## 39. repeatability batch ao/ar 결과

- stability_ao: `75.0195 / 77.1326`
- stability_ap: `75.2291 / 77.5939`
- stability_aq: `76.7890 / 81.1224`
- stability_ar: `76.7029 / 81.2732`
- aggregate a~ar: mean h1=`75.9502`, std h1=`0.6448`, mean h2=`79.3183`, std h2=`1.5216`
- 해석: high tail와 low tail이 동시에 강화되어 variance가 오히려 더 넓어졌다. harvest-only continuation의 한계가 더 분명하다.

## 40. repeatability batch as/av 결과

- stability_as: `75.6318 / 78.6021`
- stability_at: `75.2055 / 77.6308`
- stability_au: `74.9725 / 77.0401`
- stability_av: `76.4855 / 80.6181`
- aggregate a~av: mean h1=`75.9188`, std h1=`0.6478`, mean h2=`79.2478`, std h2=`1.5267`
- 해석: variance 추정은 더 넓어졌고, harvest-only continuation의 한계가 이제 충분히 입증되었다.

## 41. repeatability batch aw/az 결과

- stability_aw: `76.4175 / 80.2314`
- stability_ax: `75.5694 / 78.3571`
- stability_ay: `75.5812 / 78.3747`
- stability_az: `76.2010 / 79.7506`
- aggregate a~az: mean h1=`75.9206`, std h1=`0.6310`, mean h2=`79.2425`, std h2=`1.4848`
- 해석: harvest-only continuation이 더는 결론을 바꾸지 않고, variance evidence만 누적시키는 상태다.

## 42. repeatability batch ba/bd 결과

- stability_ba: `75.4916 / 78.2030`
- stability_bb: `75.2032 / 77.6074`
- stability_bc: `74.6368 / 76.4149`
- stability_bd: `76.3073 / 80.3119`
- aggregate a~bd: mean h1=`75.8841`, std h1=`0.6427`, mean h2=`79.1633`, std h2=`1.5071`
- 해석: harvest-only continuation은 사실상 포화이며, 이제는 variance 진단만 더 강화하고 있다.

## 43. repeatability batch be/bh 결과

- stability_be: `76.2664 / 79.9528`
- stability_bf: `76.2032 / 79.7842`
- stability_bg: `76.1030 / 79.6981`
- stability_bh: `76.2199 / 79.9415`
- aggregate a~bh: mean h1=`75.9050`, std h1=`0.6260`, mean h2=`79.2087`, std h2=`1.4661`
- 해석: 이번 배치는 분포 중심 근처를 다시 확인해줬고, 새로운 상단 frontier는 만들지 못했다.

## 44. repeatability batch bi/bl 결과

- stability_bi: `75.8213 / 78.8791`
- stability_bj: `75.9623 / 79.2169`
- stability_bk: `76.4915 / 80.5793`
- stability_bl: `76.1853 / 79.8508`
- aggregate a~bl: mean h1=`75.9182`, std h1=`0.6115`, mean h2=`79.2351`, std h2=`1.4324`
- 해석: 추가 수확도 분포 중심만 재확인했고, frontier 자체는 여전히 재현되지 않는다.

## 45. repeatability batch bm/bp 결과

- stability_bm: `74.8080 / 76.6683`
- stability_bn: `74.6698 / 76.3462`
- stability_bo: `76.2926 / 79.9250`
- stability_bp: `75.2721 / 77.8015`
- aggregate a~bp: mean h1=`75.8795`, std h1=`0.6322`, mean h2=`79.1440`, std h2=`1.4764`
- 해석: 추가 수확은 더 이상 새 정보를 만들지 않고, high-variance라는 동일 결론만 더 강하게 만든다.

## 46. repeatability batch bq/bt 결과

- stability_bq: `76.6672 / 81.5498`
- stability_br: `75.9456 / 79.3646`
- stability_bs: `75.7438 / 78.7832`
- stability_bt: `76.0837 / 79.8650`
- aggregate a~bt: mean h1=`75.8923`, std h1=`0.6220`, mean h2=`79.1855`, std h2=`1.4653`
- 해석: 반복 batch가 low-81 repeat는 다시 보여주지만, overall conclusion은 더 이상 바뀌지 않는다.

## 47. repeatability batch bu/bx 결과

- stability_bu: `76.3635 / 80.2659`
- stability_bv: `76.5782 / 80.8722`
- stability_bw: `75.8280 / 79.0298`
- stability_bx: `74.8164 / 76.6564`
- aggregate a~bx: mean h1=`75.8925`, std h1=`0.6252`, mean h2=`79.1865`, std h2=`1.4735`
- 해석: 중심도 ceiling도 사실상 고정됐고, 추가 수확은 같은 패턴만 재현하고 있다.

## 48. repeatability batch ci/cl 결과

- stability_ci: `75.1396 / 77.3811`
- stability_cj: `76.5632 / 80.8992`
- stability_ck: `75.3962 / 77.9817`
- stability_cl: `76.2215 / 80.5182`
- aggregate a~cl: mean h1=`75.8894`, std h1=`0.6233`, mean h2=`79.1870`, std h2=`1.4766`
- 해석: 분포 중심과 ceiling이 더는 움직이지 않는다. 이 lane은 사실상 종료 조건을 충족하는 수준으로 포화되었다.

## 49. repeatability batch cm/cp 결과

- stability_cm: `75.9912 / 79.2826`
- stability_cn: `76.3583 / 80.1302`
- stability_co: `76.2133 / 79.8811`
- stability_cp: `75.9452 / 79.4342`
- aggregate a~cp: mean h1=`75.9007`, std h1=`0.6114`, mean h2=`79.2105`, std h2=`1.4468`
- 해석: 중심값 재확인만 일어나고 있고, frontier는 더 이상 갱신되지 않는다.

## 50. repeatability batch cq/ct 결과

- stability_cq: `74.8219 / 76.6854`
- stability_cr: `75.9775 / 79.2118`
- stability_cs: `76.5107 / 80.4487`
- stability_ct: `75.1292 / 77.3288`
- aggregate a~ct: mean h1=`75.8875`, std h1=`0.6172`, mean h2=`79.1745`, std h2=`1.4584`
- 해석: 추가 수확이 같은 분포를 다시 보여줄 뿐, 새로운 frontier나 새로운 통찰은 더 이상 만들지 못한다.

## 51. repeatability batch cu/cx 결과

- stability_cu: `75.7942 / 79.2657`
- stability_cv: `76.0391 / 79.7895`
- stability_cw: `76.7770 / 81.4530`
- stability_cx: `76.3732 / 80.6690`
- aggregate a~cx: mean h1=`75.9031`, std h1=`0.6129`, mean h2=`79.2232`, std h2=`1.4550`
- 해석: high tail 재확인은 가능하지만 archived best는 여전히 upper-tail로 남아 있다.

## 52. repeatability batch cy/db 결과

- stability_cy: `75.4802 / 78.1711`
- stability_cz: `75.3843 / 77.9599`
- stability_da: `76.1373 / 79.6437`
- stability_db: `76.7356 / 81.3225`
- aggregate a~db: mean h1=`75.8788`, std h1=`0.6048`, mean h2=`79.1626`, std h2=`1.4378`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 는 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 네 개 모두 ±15% 밖, 최고값 db도 `81.3225 < 84.0537`
- 해석: db가 low-81 재현에는 성공했지만, frontier도 15% gate도 넘지 못했다. 추가 수확이 lane outcome을 바꾸지 못하고 variance만 다시 샘플링하고 있다는 결론이 더 강해졌다.

## 53. repeatability batch dc/df 결과

- stability_dc: `75.9335 / 79.2017`
- stability_dd: `75.9752 / 79.2982`
- stability_de: `75.0096 / 77.1249`
- stability_df: `76.3315 / 80.1808`
- aggregate a~df: mean h1=`75.8761`, std h1=`0.6007`, mean h2=`79.1541`, std h2=`1.4271`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, batch best df도 `80.1808 < 84.0537`
- 해석: dc/dd는 중심값 재현, de는 lower tail, df도 low-81 ceiling 아래였다. 100회 반복 기준으로도 분포 중심과 ceiling이 거의 안 움직여서, 이제는 frontier 개선이 아니라 variance 수확만 반복되는 상태로 봐야 한다.

## 54. repeatability batch dg/dj 결과

- stability_dg: `76.0394 / 79.4310`
- stability_dh: `77.5370 / 82.8792`
- stability_di: `76.9053 / 81.5219`
- stability_dj: `76.4811 / 80.5591`
- aggregate a~dj: mean h1=`75.9094`, std h1=`0.6216`, mean h2=`79.2289`, std h2=`1.4696`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖이지만, dh는 `82.8792`로 하한 `84.0537`에 가장 근접
- 해석: dh가 기존 archived frontier `77.4647 / 82.7683`를 넘어서면서 no-retrieval 반복 수확의 새 frontier가 됐다. 아직 h2 15% gate는 못 넘었지만, upper tail이 완전히 닫힌 건 아니고 아주 드물게 더 강한 spike capture가 나올 수 있다는 증거가 생겼다.

## 55. repeatability batch dk/dn 결과

- stability_dk: `75.3499 / 77.8644`
- stability_dl: `75.4075 / 78.0613`
- stability_dm: `76.8142 / 81.1934`
- stability_dn: `76.6068 / 80.7117`
- aggregate a~dn: mean h1=`75.9144`, std h1=`0.6240`, mean h2=`79.2374`, std h2=`1.4716`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: dk/dl은 lower tail로 돌아갔고 dm/dn은 low-81 band만 재현했다. 즉 dh는 의미 있는 breakthrough였지만, 아직 “새로운 안정 밴드”라고 보기엔 부족하고 upper-tail sample 하나로 보는 해석이 더 맞다.

## 56. repeatability batch do/dr 결과

- stability_do: `76.0658 / 79.4812`
- stability_dp: `76.7378 / 81.0388`
- stability_dq: `76.1269 / 79.6135`
- stability_dr: `75.2991 / 77.8468`
- aggregate a~dr: mean h1=`75.9195`, std h1=`0.6209`, mean h2=`79.2466`, std h2=`1.4616`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: dp는 low-81 band를 다시 보여줬지만, dh-class upper tail은 재현되지 않았다. 즉 현재까지는 dh가 “활성 frontier”이긴 하지만, 여전히 rare sample 성격이 강하다.

## 57. repeatability batch ds/dv 결과

- stability_ds: `76.3636 / 80.7572`
- stability_dt: `76.7849 / 81.1618`
- stability_du: `76.4900 / 80.8769`
- stability_dv: `76.3971 / 80.6285`
- aggregate a~dv: mean h1=`75.9398`, std h1=`0.6202`, mean h2=`79.3021`, std h2=`1.4663`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: 이번 batch는 전부 high-80 / low-81 band에 모여서 lower tail도 dh-class upper tail도 없이 “중상단 repeat band”만 재확인했다. 중심값은 약간 올라갔지만, dh breakthrough를 두 번째로 확인하진 못했다.

## 58. repeatability batch dw/dz 결과

- stability_dw: `75.8746 / 79.0292`
- stability_dx: `74.8074 / 76.7431`
- stability_dy: `75.8280 / 78.9356`
- stability_dz: `75.8553 / 78.9920`
- aggregate a~dz: mean h1=`75.9282`, std h1=`0.6186`, mean h2=`79.2728`, std h2=`1.4610`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: 이번 batch는 다시 center/lower-tail 쪽으로 내려왔다. 즉 high-80 / low-81 band조차 안정적으로 유지되지 않고, dh는 계속 희귀 upper-tail sample로 남아 있다.

## 59. repeatability batch ea/ed 결과

- stability_ea: `75.2757 / 77.9281`
- stability_eb: `75.1813 / 77.4631`
- stability_ec: `76.7320 / 81.0741`
- stability_ed: `75.8906 / 79.0519`
- aggregate a~ed: mean h1=`75.9231`, std h1=`0.6192`, mean h2=`79.2601`, std h2=`1.4605`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: ea/eb는 lower tail, ec는 low-81 band, ed는 center에 위치했다. 즉 이번 batch도 dh-class upper tail을 재현하지 못했고, dh는 계속 희귀 sample로 남아 있다.

## 60. repeatability batch ee/eh 결과

- stability_ee: `76.1094 / 79.5682`
- stability_ef: `75.7497 / 79.0772`
- stability_eg: `76.4335 / 80.2562`
- stability_eh: `75.5774 / 78.3690`
- aggregate a~eh: mean h1=`75.9245`, std h1=`0.6123`, mean h2=`79.2619`, std h2=`1.4427`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: eg가 low-80 정도는 다시 보여줬지만, 나머지는 center~lower-tail에 머물렀다. 즉 이번 batch도 frontier 도전이라기보다는 기존 분포의 variance 재샘플링에 가깝다.

## 61. repeatability batch ei/el 결과

- stability_ei: `76.1896 / 79.7370`
- stability_ej: `75.9793 / 79.2473`
- stability_ek: `75.5790 / 78.4424`
- stability_el: `75.7252 / 78.9464`
- aggregate a~el: mean h1=`75.9228`, std h1=`0.6044`, mean h2=`79.2568`, std h2=`1.4234`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: 이번 batch는 center~high-70/low-80 구간 안에서만 움직였고, lower-tail 붕괴도 dh-class upper-tail도 없었다. 누적 분포가 거의 고정되었다는 해석이 더 강해졌다.

## 62. repeatability batch em/ep 결과

- stability_em: `75.8603 / 79.0059`
- stability_en: `75.9585 / 79.5723`
- stability_eo: `76.5470 / 80.6414`
- stability_ep: `75.1531 / 77.5278`
- aggregate a~ep: mean h1=`75.9215`, std h1=`0.6015`, mean h2=`79.2548`, std h2=`1.4155`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: eo는 low-80 band, em/en은 center, ep는 lower tail이었다. 이번 batch도 기존 variance profile만 재확인했고, dh-class upper-tail은 끝내 다시 안 나왔다.

## 63. repeatability batch eq/et 결과

- stability_eq: `75.8983 / 79.2343`
- stability_er: `76.1185 / 79.6431`
- stability_es: `76.1834 / 79.7062`
- stability_et: `76.0474 / 79.6814`
- aggregate a~et: mean h1=`75.9255`, std h1=`0.5936`, mean h2=`79.2637`, std h2=`1.3965`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: 이번 batch는 center~high-70 구간에 더 촘촘하게 모였다. 즉 dh를 제외한 반복 분포가 점점 더 안정적으로 굳어지고 있다는 쪽의 증거가 강해졌다.

## 64. repeatability batch eu/ex 결과

- stability_eu: `76.4300 / 80.7390`
- stability_ev: `76.3755 / 80.1751`
- stability_ew: `76.3836 / 80.3855`
- stability_ex: `75.2923 / 77.9861`
- aggregate a~ex: mean h1=`75.9309`, std h1=`0.5916`, mean h2=`79.2792`, std h2=`1.3916`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: eu/ev/ew는 다시 low-80 repeat band를 보여줬고, ex는 weaker tail로 내려갔다. 즉 분포 구조는 여전히 `low-80 repeat band + weaker tail + single frontier(dh)` 로 유지되고 있다.

## 65. repeatability batch ey/fb 결과

- stability_ey: `76.2326 / 79.8489`
- stability_ez: `76.0319 / 79.4029`
- stability_fa: `75.1885 / 77.5219`
- stability_fb: `76.2158 / 80.2767`
- aggregate a~fb: mean h1=`75.9306`, std h1=`0.5877`, mean h2=`79.2787`, std h2=`1.3836`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: ey/ez는 center-high, fb는 low-80 band, fa는 weaker tail에 해당한다. 즉 이번 batch도 같은 3층 분포 구조만 다시 확인했고, frontier는 전혀 바뀌지 않았다.

## 66. repeatability batch fc/ff 결과

- stability_fc: `76.0188 / 79.3439`
- stability_fd: `76.1005 / 79.5448`
- stability_fe: `76.1249 / 79.6269`
- stability_ff: `74.6679 / 76.3300`
- aggregate a~ff: mean h1=`75.9252`, std h1=`0.5893`, mean h2=`79.2638`, std h2=`1.3864`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: fc/fd/fe는 center-high band, ff는 weaker tail로 내려갔다. 이번 batch도 결국 같은 3층 구조를 재현했고, 분포는 더 타이트해졌지만 상단 frontier는 전혀 안 올라갔다.

## 67. repeatability batch fg/fj 결과

- stability_fg: `75.4976 / 78.2548`
- stability_fh: `75.9612 / 79.4298`
- stability_fi: `75.8301 / 78.9416`
- stability_fj: `76.3339 / 80.5582`
- aggregate a~fj: mean h1=`75.9247`, std h1=`0.5837`, mean h2=`79.2646`, std h2=`1.3751`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: fj는 low-80 band, fg는 weaker tail, fh/fi는 center-high band였다. 이번 batch도 동일한 3층 분포 구조만 재현했고, frontier 변화는 없었다.

## 68. repeatability batch fk/fn 결과

- stability_fk: `76.4724 / 80.7070`
- stability_fl: `74.8493 / 76.7657`
- stability_fm: `76.1625 / 79.6548`
- stability_fn: `75.2362 / 77.7130`
- aggregate a~fn: mean h1=`75.9186`, std h1=`0.5870`, mean h2=`79.2508`, std h2=`1.3826`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: fk는 low-80 band, fl/fn은 weaker tail, fm은 center-high band였다. 이번 batch도 같은 3층 구조를 재현했고, 분포는 계속 타이트하지만 frontier는 그대로다.

## 69. repeatability batch fo/fr 결과

- stability_fo: `76.4733 / 80.3790`
- stability_fp: `74.9242 / 76.9204`
- stability_fq: `77.0017 / 81.6986`
- stability_fr: `76.3639 / 80.6792`
- aggregate a~fr: mean h1=`75.9253`, std h1=`0.5936`, mean h2=`79.2671`, std h2=`1.3980`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: fq가 low-81 band 최상단을 다시 찍으면서 dh 아래의 ‘강한 secondary tier’가 살아있음을 재확인했다. 하지만 dh 자체는 여전히 못 넘었고, active frontier는 변함없다.

## 70. repeatability batch fs/fv 결과

- stability_fs: `76.6733 / 81.2086`
- stability_ft: `74.8415 / 76.7545`
- stability_fu: `75.5863 / 78.3725`
- stability_fv: `75.1338 / 77.5853`
- aggregate a~fv: mean h1=`75.9165`, std h1=`0.5989`, mean h2=`79.2483`, std h2=`1.4104`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: fs가 low-81 band를 다시 보여줬고, ft/fv는 weaker tail, fu는 upper-center였다. 즉 이번 batch도 똑같은 3층 구조만 유지됐고, frontier 변화는 없었다.

## 71. repeatability batch fw/fz 결과

- stability_fw: `75.9150 / 79.2608`
- stability_fx: `74.6801 / 76.3619`
- stability_fy: `76.2105 / 79.7590`
- stability_fz: `75.5985 / 78.6361`
- aggregate a~fz: mean h1=`75.9092`, std h1=`0.6002`, mean h2=`79.2310`, std h2=`1.4124`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: fy는 center-high, fw는 center, fz는 upper-70, fx는 weaker tail이었다. 이번 batch도 같은 3층 구조만 재확인했고, 전체 분포는 거의 변하지 않았다.

## 72. repeatability batch gaa/gad 결과

- stability_gaa: `76.4478 / 80.5269`
- stability_gab: `75.9893 / 79.2635`
- stability_gac: `76.2925 / 79.9404`
- stability_gad: `76.1964 / 79.7275`
- aggregate a~gad: mean h1=`75.9165`, std h1=`0.5958`, mean h2=`79.2454`, std h2=`1.4011`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: 이번 batch는 center~low-80 구간 안에서만 움직였고 weaker tail도 심하지 않았다. 하지만 dh-class upper tail이나 low-81 secondary-tier 최상단 재현도 없어서, 분포 안정성만 더 강화됐다.

## 73. repeatability batch gae/gah 결과

- stability_gae: `76.6468 / 81.1687`
- stability_gaf: `76.2207 / 79.8049`
- stability_gag: `76.2122 / 80.2153`
- stability_gah: `74.8310 / 76.6863`
- aggregate a~gah: mean h1=`75.9179`, std h1=`0.5980`, mean h2=`79.2504`, std h2=`1.4083`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gae는 다시 low-81 band, gaf/gag는 center~low-80, gah는 weaker tail이었다. 이번 batch도 같은 3층 구조만 재확인했고, frontier는 그대로였다.

## 74. repeatability batch gai/gal 결과

- stability_gai: `76.4328 / 80.6383`
- stability_gaj: `75.2240 / 77.5703`
- stability_gak: `76.9542 / 82.1254`
- stability_gal: `76.0055 / 79.3035`
- aggregate a~gal: mean h1=`75.9230`, std h1=`0.5998`, mean h2=`79.2647`, std h2=`1.4180`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gak가 `82.1254`까지 올라오면서 dh 바로 아래의 strong secondary tier가 다시 강하게 살아있음을 보여줬다. 하지만 dh는 여전히 못 넘었고, active frontier는 그대로다.

## 75. repeatability batch gam/gap 결과

- stability_gam: `76.4031 / 80.2030`
- stability_gan: `76.1167 / 79.6198`
- stability_gao: `75.9898 / 79.2833`
- stability_gap: `76.1217 / 79.8168`
- aggregate a~gap: mean h1=`75.9280`, std h1=`0.5947`, mean h2=`79.2746`, std h2=`1.4053`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: 이번 batch는 거의 전부 center~low-80 안에서만 움직였고, secondary tier 상단도 weaker tail도 강하게 나타나지 않았다. 즉 분포가 사실상 고정 수준으로 평탄해지고 있다.

## 76. repeatability batch gaq/gat 결과

- stability_gaq: `74.9205 / 76.9131`
- stability_gar: `76.3401 / 80.1014`
- stability_gas: `74.8267 / 76.7525`
- stability_gat: `75.8362 / 79.0836`
- aggregate a~gat: mean h1=`75.9187`, std h1=`0.5990`, mean h2=`79.2525`, std h2=`1.4139`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gaq/gas는 weaker tail, gar는 low-80 band, gat는 center-high였다. 이번 batch도 정확히 같은 3층 구조만 재현했고, 분포 평균도 거의 변하지 않았다.

## 77. repeatability batch gau/gax 결과

- stability_gau: `76.3807 / 80.1863`
- stability_gav: `75.3805 / 78.0396`
- stability_gaw: `75.5897 / 78.4598`
- stability_gax: `76.1076 / 79.8120`
- aggregate a~gax: mean h1=`75.9176`, std h1=`0.5956`, mean h2=`79.2499`, std h2=`1.4054`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gau는 low-80 band, gav/gaw는 upper-70, gax는 center-high에 위치했다. 이번 batch도 결국 같은 성숙한 3층 분포 구조만 다시 확인했고, frontier는 변하지 않았다.

## 78. repeatability batch gay/gbb 결과

- stability_gay: `76.3104 / 80.3225`
- stability_gaz: `76.0116 / 79.3082`
- stability_gba: `75.9296 / 79.3388`
- stability_gbb: `74.8418 / 76.7340`
- aggregate a~gbb: mean h1=`75.9147`, std h1=`0.5952`, mean h2=`79.2434`, std h2=`1.4046`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gay는 low-80 band, gaz/gba는 center-high, gbb는 weaker tail이었다. 200회 반복 기준으로도 분포 평균과 분산이 거의 안 움직여서, 현 lane은 사실상 saturation 상태라고 봐야 한다.

## 79. repeatability batch gbc/gbf 결과

- stability_gbc: `75.0984 / 77.2905`
- stability_gbd: `76.3554 / 80.4632`
- stability_gbe: `75.8953 / 79.0589`
- stability_gbf: `75.3351 / 77.9420`
- aggregate a~gbf: mean h1=`75.9099`, std h1=`0.5943`, mean h2=`79.2325`, std h2=`1.4031`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gbd는 low-80 band, gbc/gbf는 weaker tail, gbe는 center-high였다. 204회 반복 시점에도 분포 중심과 분산이 거의 안 움직여서 saturation 해석이 더 강해졌다.

## 80. repeatability batch gbg/gbj 결과

- stability_gbg: `76.2299 / 79.8183`
- stability_gbh: `75.9664 / 79.2289`
- stability_gbi: `76.8785 / 81.6115`
- stability_gbj: `76.3974 / 80.4940`
- aggregate a~gbj: mean h1=`75.9187`, std h1=`0.5937`, mean h2=`79.2528`, std h2=`1.4024`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gbi가 `81.6115`로 upper secondary tier를 다시 보여줬지만, dh는 여전히 못 넘었다. 208회 반복 기준 aggregate도 거의 고정이라 saturation 해석은 더 강해졌다.

## 81. repeatability batch gbk/gbn 결과

- stability_gbk: `75.9246 / 79.3852`
- stability_gbl: `76.0006 / 79.4982`
- stability_gbm: `76.3561 / 80.1253`
- stability_gbn: `75.7855 / 78.8337`
- aggregate a~gbn: mean h1=`75.9206`, std h1=`0.5889`, mean h2=`79.2568`, std h2=`1.3909`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gbm은 low-80 band, gbk/gbl은 center-high, gbn은 upper-70였다. 212회 반복 기준에서도 분포는 거의 고정되어 있고 saturation 결론이 유지된다.

## 82. repeatability batch gbo/gbr 결과

- stability_gbo: `76.3222 / 80.3002`
- stability_gbp: `74.6427 / 76.2840`
- stability_gbq: `76.8401 / 81.6271`
- stability_gbr: `76.6024 / 81.0201`
- aggregate a~gbr: mean h1=`75.9239`, std h1=`0.5956`, mean h2=`79.2670`, std h2=`1.4089`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gbq/gbr가 다시 81대 secondary tier를 보여줬지만 dh는 여전히 못 넘었다. 216회 반복 기준 aggregate도 거의 그대로라서, upper secondary tier 재현과 saturation이 동시에 확인된다.

## 83. repeatability batch gbs/gbv 결과

- stability_gbs: `75.9772 / 79.2843`
- stability_gbt: `75.2480 / 77.6235`
- stability_gbu: `76.6404 / 80.8447`
- stability_gbv: `75.9816 / 79.4637`
- aggregate a~gbv: mean h1=`75.9246`, std h1=`0.5939`, mean h2=`79.2676`, std h2=`1.4045`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gbu는 low-80 band, gbt는 weaker tail, gbs/gbv는 center-high였다. 220회 반복 기준에서도 aggregate가 사실상 고정이라 saturation 해석이 유지된다.

## 84. repeatability batch gbw/gbz 결과

- stability_gbw: `76.7147 / 81.1300`
- stability_gbx: `74.5107 / 76.0211`
- stability_gby: `76.5019 / 80.6307`
- stability_gbz: `75.6061 / 78.4738`
- aggregate a~gbz: mean h1=`75.9230`, std h1=`0.6001`, mean h2=`79.2640`, std h2=`1.4181`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gbw는 low-81 band, gby는 low-80 band, gbz는 upper-70, gbx는 weaker tail이었다. 224회 반복 기준에서도 frontier는 그대로고, saturation 해석이 더 강해졌다.

## 85. repeatability batch gca/gcd 결과

- stability_gca: `75.7606 / 78.8586`
- stability_gcb: `76.6534 / 80.9183`
- stability_gcc: `77.0738 / 82.1532`
- stability_gcd: `76.2754 / 79.9897`
- aggregate a~gcd: mean h1=`75.9321`, std h1=`0.6021`, mean h2=`79.2853`, std h2=`1.4237`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gcc가 `82.1532`까지 올라와 late search 기준 strongest upper secondary tier를 다시 보여줬다. 하지만 dh는 여전히 못 넘었고, aggregate도 거의 그대로라서 saturation 결론은 유지된다.

## 86. repeatability batch gce/gch 결과

- stability_gce: `76.2176 / 79.8389`
- stability_gcf: `76.1824 / 79.7183`
- stability_gcg: `75.3044 / 77.7779`
- stability_gch: `75.9146 / 79.2554`
- aggregate a~gch: mean h1=`75.9316`, std h1=`0.5988`, mean h2=`79.2830`, std h2=`1.4156`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: 이번 batch는 mostly center-high 구간에 모였고 gcg만 weaker tail로 내려갔다. upper secondary tier 재현조차 없어서 saturation 해석이 더 강해졌다.

## 87. repeatability batch gci/gcl 결과

- stability_gci: `75.9550 / 79.2715`
- stability_gcj: `74.9753 / 77.0678`
- stability_gck: `77.1332 / 82.0834`
- stability_gcl: `74.6033 / 76.1513`
- aggregate a~gcl: mean h1=`75.9271`, std h1=`0.6082`, mean h2=`79.2721`, std h2=`1.4372`
- 목표 체크:
  - 네 개 모두 `h2 > h1` 유지
  - 네 개 모두 h1은 실제값 대비 ±15% 안
  - h2는 전부 ±15% 밖, active frontier는 여전히 dh의 `82.8792`
- 해석: gck가 `82.0834`로 upper secondary tier를 다시 강하게 보여줬지만 dh는 못 넘었다. gcj/gcl은 weaker tail이었고, 결국 이번 batch도 frontier 고정 + saturated profile 결론만 강화했다.
