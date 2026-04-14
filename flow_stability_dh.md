# AA-Forecast + Informer 온보딩: `stability_dh` 시계열 샘플 1개가 실제로 어떻게 계산되는가

이 문서는 `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_stability_dh/config/config.resolved.json` 기준으로,
**"코드가 어디를 타는가"보다 "Brent 시계열 샘플 1개가 어떤 수치/텐서 흐름을 거쳐 예측이 되는가"**에 집중한다.

이 설정에서 핵심은 아래 5단계다.

1. 최근 **64개 시점의 diff window** 를 자른다.
2. target과 star exogenous에 대해 **LOWESS + STAR 분해** 를 수행한다.
3. 그 결과를 포함한 feature tensor를 **AA-Forecast Informer backbone** 에 넣어 trajectory/event/regime 표현을 만든다.
4. anomaly-aware decoding이 spike 방향을 증폭하려는 경로를 만든다.
5. 여러 dropout 후보로 예측을 반복해서 **선택 규칙에 따라 최종 horizon 예측** 을 고른다.

---

## 1. 이 문서가 설명하는 정확한 설정

### 실험 식별자
- task name: `brentoil_case1_parity_aaforecast_informer_stability_dh`
- artifact root:
  - `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_stability_dh`

### dataset
- target: `Com_BrentCrudeOil`
- hist exog 10개:
  - `GPRD_THREAT`
  - `BS_Core_Index_A`
  - `GPRD`
  - `GPRD_ACT`
  - `BS_Core_Index_B`
  - `BS_Core_Index_C`
  - `Idx_OVX`
  - `Com_LMEX`
  - `Com_BloombergCommodity_BCOM`
  - `Idx_DxyUSD`

### runtime / CV
- `transformations_target = diff`
- `transformations_exog = diff`
- `input_size = 64`
- `horizon = 2`
- `n_windows = 1`
- `step_size = 4`

### training
- `max_steps = 800`
- `val_size = 16`
- `batch_size = 32`
- `valid_batch_size = 8`
- `loss = mse`
- optimizer: `adamw`
- scheduler: `ReduceLROnPlateau`

### AA stage config
- backbone: `informer`
- `lowess_frac = 0.35`
- `lowess_delta = 0.01`
- `thresh = 3.5`
- `retrieval.enabled = false`
- uncertainty:
  - `enabled = true`
  - dropout candidates:
    - `0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3`
  - `sample_count = 30`

### Informer backbone params
- `hidden_size = 128`
- `n_head = 4`
- `encoder_layers = 2`
- `dropout = 0.0`
- `linear_hidden_size = 96`
- `factor = 3`
- `decoder_hidden_size = 128`
- `decoder_layers = 4`
- `season_length = 4`

### STAR grouping
- star hist exog:
  - `GPRD_THREAT`
- non-star hist exog:
  - `BS_Core_Index_A`
  - `GPRD`
  - `GPRD_ACT`
  - `BS_Core_Index_B`
  - `BS_Core_Index_C`
  - `Idx_OVX`
  - `Com_LMEX`
  - `Com_BloombergCommodity_BCOM`
  - `Idx_DxyUSD`
- star anomaly tails:
  - upward: `GPRD_THREAT`
  - two_sided: 없음

---

## 2. 이 설정에서 모델이 실제로 보는 것은 raw level이 아니라 diff-transformed window

이 run은 runtime transform이 들어가 있다.

즉 개념적으로 모델이 보는 입력은:
- target 입력: 최근 64개 시점의 `Brent diff`
- exogenous 입력: 같은 64개 시점의 hist exogenous diff

최종 산출물은 artifact 상에서 level-scale `y_hat` 로 정리되지만,
모델 내부 인코딩은 **diff 변환된 시계열 기반** 으로 이해하는 것이 맞다.

---

## 3. 이 run의 실제 마지막 fold 결과

artifact `summary/result.csv` 기준 마지막 fold horizon 2개는 아래다.

| horizon | ds | actual y | predicted y_hat |
|---:|---|---:|---:|
| 1 | 2026-03-02 | 86.64365714 | 77.5370032270 |
| 2 | 2026-03-09 | 98.88672857 | 82.8791667262 |

즉 이 `stability_dh` 는
- `h1 = 77.54`
- `h2 = 82.88`
- `gap = 5.34`
를 만들었다.

이건 retrieval off 기준에서 좋은 케이스지만,
아직 목표인 `h1 >= 78`, `h2 >= 85` 는 못 넘는다.

---

## 4. Step 1 — window construction과 tensor shape

이 run의 핵심 길이는 다음과 같다.

- `input_size = 64`
- `horizon = 2`
- hist exog 수 = `10`
- star hist exog 수 = `1`
- non-star hist exog 수 = `9`

forward 직전 개념적 텐서는 다음처럼 볼 수 있다.

- `insample_y`: `[B, 64, 1]`
- `hist_exog`: `[B, 64, 10]`

이후 hist exog는 둘로 나뉜다.

- star hist exog: `GPRD_THREAT` → `[B, 64, 1]`
- non-star hist exog 9개 → `[B, 64, 9]`

---

## 5. Step 2 — LOWESS + STAR 분해

AA runtime은 target과 star exogenous에 대해 STAR 분해를 수행한다.

각 시계열 `s_t` 에 대해 대략 다음을 만든다.

1. **trend**
   - `T_t = LOWESS(s_t)`
2. **detrended / seasonal decomposition용 신호**
3. **seasonal**
   - `season_length = 4` 기준
4. **anomaly / residual**
   - threshold `3.5`
   - target은 two-sided
   - `GPRD_THREAT` 는 upward only

즉 target과 star exog는 각각
- trend
- seasonal
- anomalies
- residual
의 4개 조각으로 분해된다.

non-star exog는 raw diff sequence 쪽으로 남는다.

---

## 6. Step 3 — encoder 입력 feature 구성

이 run에서 encoder 입력은 개념적으로 아래 조각들의 concat이다.

1. raw/diff target: `1`
2. non-star hist exog: `9`
3. target STAR outputs: `4`
4. star hist exog STAR outputs: `4 × 1 = 4`

합치면:
- `1 + 9 + 4 + 4 = 18`

즉 개념적으로 Informer backbone 쪽 입력은:
- `encoder_input`: `[B, 64, 18]`

처럼 볼 수 있다.

---

## 7. Step 4 — AA-Informer 내부 표현 흐름

이 run의 Informer AA head는 단순 dense 하나가 아니라,
이벤트/경로/레짐 정보를 분리해서 decoder 입력을 조건화한다.

개념적으로 흐름은 이렇다.

1. **event_summary**
   - star anomaly / event 정보를 요약한 벡터
2. **event_path**
   - anomaly-aware path representation
3. **raw_regime**
   - 현재 시점 레짐을 설명하는 representation
4. **pooled_context / memory_signal**
   - 전체 window에서 압축된 문맥
5. **conditioned decoder features**
   - decoder_input을 event/path/regime gate로 modulation
6. **shared trunk + attention + mixer**
   - conditioned path를 한 번 더 섞음
7. **trajectory / baseline / expert heads**
   - baseline level
   - trajectory shock
   - gate / amplitude
   - event / local / regime 관련 보정

즉 핵심은
**"과거 64-step diff window에서 spike 관련 event/path/regime를 뽑고, 그걸 horizon 2-step decoder에 주입해 마지막 level forecast를 만든다"** 로 이해하면 된다.

---

## 8. Step 5 — uncertainty selection

이 run은 uncertainty가 켜져 있다.

- dropout 후보 12개
- 각 후보마다 `sample_count = 30`

즉 단일 forward 한 번으로 끝나는 게 아니라,
여러 dropout 후보와 sample 반복을 통해 candidate prediction distribution을 만들고,
그 안에서 selection rule에 따라 최종 경로를 고른다.

이 run artifact를 보면 `aaforecast_context_artifact` 가 남아 있어서,
선택된 문맥/경로가 최종 예측에 연결된 것을 추적할 수 있다.

중요한 점은,
이 run은 **retrieval off** 이므로,
최종 예측 uplift는 retrieval blending이 아니라
**AA-Informer 내부의 event/path/regime + uncertainty selection 구조** 에서 나온 것이다.

---

## 9. 이 케이스가 상대적으로 잘된 이유

`stability_dh` 가 retrieval off 기준에서 좋은 이유는 대략 이렇게 읽을 수 있다.

1. **우상향을 만든다**
   - `77.54 -> 82.88`
2. **gap이 충분하다**
   - `+5.34`
3. **retrieval 없이도 상승 path를 유지한다**
   - 즉 외부 retrieval uplift가 아니라 내부 anomaly-aware 구조에서 나온 결과다.
4. **STAR를 과도하게 넓히지 않았다**
   - star exog를 `GPRD_THREAT` 하나로 유지해 구조 불안정을 줄였다.
5. **작은 Informer head**
   - `128 / 96 / 128` 수준이라 오히려 과적합/불안정성을 덜 만든다.

---

## 10. 그런데 왜 아직 목표에는 못 미쳤는가

이 케이스는 방향성은 좋지만 절대 amplitude가 부족하다.

실제값 대비 보면:
- h1 actual `86.64` vs pred `77.54`
- h2 actual `98.89` vs pred `82.88`

즉,
- spike direction은 잡지만
- **spike level까지 충분히 transport하지는 못한다**

그래서 현재 병목은
**direction 문제가 아니라 amplitude transport 부족** 으로 보는 것이 맞다.

---

## 11. 한 줄 요약

`stability_dh` 는
**"retrieval 없이, 64-step diff window + 1개 star exog(`GPRD_THREAT`) + Informer AA decoder + uncertainty selection"** 조합으로,
마지막 fold에서 `77.54 -> 82.88` 을 만든 케이스다.

즉 이 실험은
- gap 4 이상을 만족하고,
- no-retrieval 경로에서 좋은 기준점이지만,
- 아직 절대 레벨 `78 / 85` 를 넘기기엔 amplitude가 부족한 **근접 성공 케이스** 라고 볼 수 있다.
