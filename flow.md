# AA-Forecast + GRU 온보딩: 시계열 샘플 1개가 실제로 어떻게 계산되는가

이 문서는 `yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-gru.yaml` 기준으로,
**"코드가 어디를 타는가"보다 "Brent 시계열 샘플 1개가 어떤 수치/텐서 흐름을 거쳐 예측이 되는가"**에 집중한다.

이 설정에서 핵심은 아래 5단계다.

1. 최근 **24개 시점의 raw level window** 를 자른다.
2. target과 두 개의 star exogenous에 대해 **LOWESS + STAR 분해** 를 수행한다.
3. 그 결과를 포함한 feature tensor를 **GRU backbone** 에 넣어 hidden state를 만든다.
4. anomaly-aware sparse attention이 이벤트성 시점에 더 무게를 준다.
5. 여러 dropout 후보로 예측을 반복해서 **표준편차가 가장 작은 dropout** 을 horizon별로 선택한다.

---

## 1. 이 문서가 설명하는 정확한 설정

### 메인 config
- 파일: `yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast-gru.yaml`
- target: `Com_BrentCrudeOil`
- hist exog 10개:
  - `GPRD_THREAT`
  - `BS_Core_Index_A`
  - `GPRD`
  - `GPRD_ACT`
  - `BS_Core_Index_B`
  - `BS_Core_Index_C`
  - `Idx_OVX`
  - `Com_Oil_Spread`
  - `Com_LMEX`
  - `Com_BloombergCommodity_BCOM`
- stage route: `aa_forecast.enabled=true`
- stage config: `yaml/plugins/aa_forecast_brentoil_case1_parity_gru.yaml`

### 공통 setting (`yaml/setting/setting.yaml`)
- `training.input_size = 24`
- `cv.horizon = 4`
- `cv.n_windows = 4`
- `training.max_steps = 800`
- `training.val_size = 16`
- `training.scaler_type = robust`

### AA stage config
- `model = gru`
- `lowess_frac = 0.35`
- `lowess_delta = 0.01`
- `thresh = 3.5`
- `uncertainty.enabled = true`
- `uncertainty.sample_count = 50`
- star-upward exogenous:
  - `GPRD_THREAT`
  - `BS_Core_Index_A`
- non-star exogenous 8개:
  - `GPRD`, `GPRD_ACT`, `BS_Core_Index_B`, `BS_Core_Index_C`,
    `Idx_OVX`, `Com_Oil_Spread`, `Com_LMEX`, `Com_BloombergCommodity_BCOM`
- GRU backbone params:
  - `encoder_hidden_size = 128`
  - `encoder_n_layers = 4`
  - `encoder_dropout = 0.1`
  - `decoder_hidden_size = 128`
  - `decoder_layers = 4`
  - `season_length = 4`

---

## 2. 이 설정에서 모델이 실제로 보는 것은 "diff window"가 아니라 "raw level window"

현재 검증 대상 설정에는 runtime diff 변환이 없다.

즉 이 run의 핵심 입력은 다음처럼 이해하는 것이 맞다.

- target 입력: 최근 24주 `Brent level`
- exogenous 입력: 같은 24주 동안의 hist exogenous level들
- scaler: `robust` 이므로, 이 설정 설명에서는 raw level을 입력으로 보되 window scaling이 함께 적용된다고 읽어야 한다

즉 이 config의 AAForecast+GRU는
**"최근 24주 raw level 시계열 + exogenous level" 을 직접 받아 STAR feature를 만들고 예측하는 경로** 다.

---

## 3. 실데이터 예시: 2025-11-17 cutoff fold 하나를 따라가 보기

이 예시는 실제 run artifact와 맞춰서 본다.

- resolved config run root: `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_gru`
- full example artifact root: `runs/iter_comparison_20260411_1/feature_set_aaforecast_brentoil_case1_parity_aaforecast_gru`
- 실제 forecast cutoff: `2025-11-17`
- 실제 horizon output:
  - `2025-11-24`
  - `2025-12-01`
  - `2025-12-08`
  - `2025-12-15`

### 3.1 입력 구간

`input_size=24` 이므로,
이 fold에서 모델이 보는 target/exogenous window는 다음 24개 주간 row다.

- **input window**: `2025-06-09` ~ `2025-11-17`
- **예측 horizon**: `2025-11-24` ~ `2025-12-15`

### 3.2 실제 raw level window 일부

| row | dt | Brent | GPRD_THREAT | BS_Core_Index_A |
|---:|---|---:|---:|---:|
| 544 | 2025-06-09 | 70.9054 | 262.8814 | 0.5100 |
| 545 | 2025-06-16 | 76.5399 | 360.4386 | 1.3993 |
| 546 | 2025-06-23 | 67.5205 | 331.2443 | 0.1867 |
| 547 | 2025-06-30 | 67.9452 | 130.1629 | -0.1422 |
| ... | ... | ... | ... | ... |
| 562 | 2025-10-13 | 61.7680 | 188.2843 | 1.5630 |
| 563 | 2025-10-20 | 63.9050 | 175.0157 | 1.2128 |
| 564 | 2025-10-27 | 64.5329 | 135.5129 | 0.6648 |
| 565 | 2025-11-03 | 63.9403 | 105.3471 | 0.3168 |
| 566 | 2025-11-10 | 63.9245 | 96.1971 | 0.7213 |
| 567 | 2025-11-17 | 63.3826 | 114.6814 | 1.1038 |

중요한 점은,
이 입력은 diff가 아니라 **그대로 raw level 24-step sequence** 라는 것이다.

### 3.3 실제 horizon truth 와 실제 run 예측

실제 artifact `runs/iter_comparison_20260411_1/feature_set_aaforecast_brentoil_case1_parity_aaforecast_gru/cv/AAForecast_forecasts.csv` 에서 같은 fold의 horizon은 아래다.

| horizon | ds | actual y | predicted y_hat |
|---:|---|---:|---:|
| 1 | 2025-11-24 | 62.65865714 | 62.0243890381 |
| 2 | 2025-12-01 | 63.27702857 | 62.5134156799 |
| 3 | 2025-12-08 | 61.66008571 | 63.0292874146 |
| 4 | 2025-12-15 | 60.13292857 | 63.2790955353 |

이 설정에서는 runtime diff 복원이 없으므로,
위 `y_hat`은 이미 **최종 level-scale 예측** 으로 이해하면 된다.

---

## 4. toy 예시: 아주 작은 수열로 같은 흐름 보기

아래는 **이해용 toy 예시**다. 실제 runtime 숫자와 정확히 일치하는 예시는 아니고,
"무슨 계산이 일어나는가"를 작게 축소한 것이다.

### 4.1 원본 level

- Brent level: `[100, 103, 102, 108, 109]`
- star exog `GPRD_THREAT`: `[10, 12, 11, 30, 35]`
- star exog `BS_Core_Index_A`: `[0.1, 0.0, 0.2, 1.4, 1.6]`

### 4.2 toy 관점의 직관

이 config에서는 raw level이 직접 STAR 분해로 들어간다.
따라서 toy에서는 이렇게 읽으면 된다.

- Brent sequence는 최근 5개 level 패턴을 보여 준다.
- `GPRD_THREAT` 는 30, 35처럼 위로 크게 치솟는 지점이 있다.
- `BS_Core_Index_A` 도 1.4, 1.6처럼 upward burst가 있다.

이런 구간은 STAR의 robust score에서 큰 값을 만들 가능성이 높다.
즉 toy 관점에서는 대충 이렇게 읽으면 된다.

- 평범한 level 변화: normal residual
- 위로 튀는 exogenous level: upward anomaly
- anomaly가 발생한 시점: `critical_mask = 1`

target도 비슷하지만 차이가 하나 있다.

- target `Brent` 는 **two-sided** 로 anomaly를 본다.
- `GPRD_THREAT`, `BS_Core_Index_A` 는 **upward only** 로 anomaly를 본다.

즉 Brent는 큰 상승/하락 둘 다 중요하고,
star exog 2개는 "위로 튀는 것"만 문맥적으로 중요하다고 취급한다.

---

## 5. Step 1 — window construction과 tensor shape

이 run의 핵심 길이는 다음과 같다.

- `input_size = 24`
- `horizon = 4`
- hist exog 수 = `10`
- star hist exog 수 = `2`
- non-star hist exog 수 = `8`

AAForecast의 forward 직전 주요 텐서는 개념적으로 이렇게 볼 수 있다.

- `insample_y`: `[B, 24, 1]`
- `hist_exog`: `[B, 24, 10]`

여기서 `B` 는 mini-batch 안에 들어온 window 개수다.

이후 hist exog는 둘로 나뉜다.

- star hist exog: `GPRD_THREAT`, `BS_Core_Index_A` → `[B, 24, 2]`
- non-star hist exog 나머지 8개 → `[B, 24, 8]`

### 실제 encoder input feature 수

AAForecast는 encoder input을 아래 조각들을 concat 해서 만든다.

1. raw insample target: `1`
2. non-star hist exog: `8`
3. target STAR outputs: `4`
   - trend
   - seasonal
   - anomalies
   - residual
4. star hist exog STAR outputs: `4 × 2 = 8`
   - 각 star exog마다 trend/seasonal/anomalies/residual

합치면:

- `1 + 8 + 4 + 8 = 21`

즉 GRU encoder로 들어가는 최종 입력은 개념적으로:

- `encoder_input`: `[B, 24, 21]`

이다.

---

## 6. Step 2 — LOWESS + STAR 분해는 정확히 무엇을 만드는가

AA runtime은 `STARFeatureExtractor` 에서 각 시계열을 아래처럼 다룬다.

### 6.1 trend

먼저 LOWESS로 trend를 만든다.

- 입력 시계열: `s_t`
- LOWESS trend: `T_t = LOWESS(s_t)`

코드상 파라미터는:

- `frac = 0.35`
- `delta = lowess_delta * seq_len = 0.01 * 24` (실제 내부 LOWESS 호출시 sequence 길이에 비례)

### 6.2 detrended

trend를 나눠서 제거한다.

- `D_t = s_t / T_t`

코드에서는 0 division을 피하려고 작은 `eps` 로 안정화한 safe divide를 쓴다.

### 6.3 seasonal

`season_length = 4` 이므로,
동일 phase(예: 4주 주기에서 같은 위치)의 평균으로 seasonal baseline을 만든다.

- `S_t = phase_mean(D_t)`

### 6.4 residual

- `R_t = D_t / S_t`

즉 STAR는 시계열을 대략 다음 구조로 나눈다.

- trend: 천천히 움직이는 배경
- seasonal: 주기 패턴
- residual: 위 둘로 설명되지 않는 나머지
- anomalies: residual 중에서도 robust threshold를 넘는 부분

### 6.5 robust anomaly score

코드는 residual에 대해 median/MAD 기반 robust score를 만든다.

- `center = median(R)`
- `MAD = median(|R - center|)`
- `signed_score = 0.6745 * (R - center) / MAD`

그 다음 tail mode에 따라 anomaly를 고른다.

- two-sided: `|signed_score| > 3.5`
- upward: `signed_score > 3.5`

이 run에서는:

- target Brent → `two_sided`
- `GPRD_THREAT`, `BS_Core_Index_A` → `upward`

즉 같은 spike라도,
- Brent는 급등/급락 모두 context가 될 수 있고,
- star exog 2개는 **위쪽 spike만** context가 된다.

---



좋아.
네가 지정한 구간은 사실상 이거 3개야.

* **Step 3:** anomaly를 어떻게 모아서 `critical mask`를 만드는가
* **Step 4:** 그 mask를 가지고 attention이 hidden state를 어떻게 다시 보정하는가
* **Step 5:** 24개 hidden 중에서 왜 마지막 4개만 써서 horizon 4개 예측으로 맞추는가

아래처럼 보면 된다.

---

# 7. Step 3 — critical mask와 context activation

핵심은:

**“24개 시점 중 어느 시점이 이벤트성인가?”**
이걸 target과 star exog에서 같이 보고 하나의 문맥 신호로 만드는 단계다.

---

## 7-1. 먼저 각 채널별 anomaly mask가 나온다

네 설정에서 STAR를 적용하는 건:

* target: `Brent`
* star exog 1: `GPRD_THREAT`
* star exog 2: `BS_Core_Index_A`

즉 총 **3개 채널**이다.

각 채널마다 24개 시점에 대해:

* anomaly 아님 → `0`
* anomaly 맞음 → `1`

이런 mask가 생긴다.

---

## 7-2. toy 예시로 보자

24개는 길어서, 일단 8개 시점만 있다고 치자.

#### (1) target Brent anomaly mask

Brent는 two-sided라서 급등/급락 둘 다 잡는다.

`target_mask = [0, 0, 1, 0, 0, 1, 0, 0]`

의미:

* t=3 에 anomaly
* t=6 에 anomaly

---

#### (2) GPRD_THREAT anomaly mask

이건 upward only.

`gprd_threat_mask = [0, 1, 1, 0, 0, 0, 0, 0]`

의미:

* t=2, t=3 에 위로 튄 이벤트성 값

---

#### (3) BS_Core_Index_A anomaly mask

`bs_a_mask = [0, 0, 0, 0, 1, 1, 0, 0]`

의미:

* t=5, t=6 에 이벤트

---

## 7-3. 이걸 합치면 channel-wise mask 텐서는 이렇게 생김

시점별로 3채널을 모으면:

| t | Brent | GPRD_THREAT | BS_A |
| - | ----: | ----------: | ---: |
| 1 |     0 |           0 |    0 |
| 2 |     0 |           1 |    0 |
| 3 |     1 |           1 |    0 |
| 4 |     0 |           0 |    0 |
| 5 |     0 |           0 |    1 |
| 6 |     1 |           0 |    1 |
| 7 |     0 |           0 |    0 |
| 8 |     0 |           0 |    0 |

shape으로 보면 개념적으로:

* `[B, T, C] = [1, 8, 3]`

실제 runtime에서는 T=24다.

---

## 7-4. critical mask는 “그 시점에 하나라도 anomaly가 있냐”다

각 시점에서 채널 3개 중 하나라도 1이면 그 시점은 중요 시점이다.

즉:

`critical_mask[t] = OR over channels`

위 표에서 하면:

* t1: 0,0,0 → 0
* t2: 0,1,0 → 1
* t3: 1,1,0 → 1
* t4: 0,0,0 → 0
* t5: 0,0,1 → 1
* t6: 1,0,1 → 1
* t7: 0,0,0 → 0
* t8: 0,0,0 → 0

그래서

`critical_mask = [0, 1, 1, 0, 1, 1, 0, 0]`

이게 된다.

즉 **“이 시점 자체가 문맥적으로 중요한가”**를 나타내는 1비트 신호다.

---

## 7-5. count_active_channels는 “얼마나 강한 이벤트인가”를 단순 개수로 본다

아까 표에서 각 시점의 anomaly 채널 개수를 세면:

* t1: 0개
* t2: 1개
* t3: 2개
* t4: 0개
* t5: 1개
* t6: 2개
* t7: 0개
* t8: 0개

즉

`count_active_channels = [0,1,2,0,1,2,0,0]`

이게 왜 중요하냐면,

* 단순히 “이상치 있음/없음”만 보면 t2와 t3가 둘 다 1이지만,
* 실제로는 t3가 **더 많은 채널이 동시에 터진 복합 이벤트 시점**이다.

그래서 모델이 t3, t6 같은 시점을 더 강하게 보도록 만들 수 있다.

---

## 7-6. channel_activity는 단순 0/1보다 더 센 정보다

문서에 적은 것처럼, 구현은 보통

* `critical_mask`
* robust anomaly score

를 같이 쓴다.

예를 들어 t3에서:

* Brent score = 4.2
* GPRD_THREAT score = 6.8
* BS_A score = 0

이라고 해보자.

그러면 단순 mask는 `[1,1,0]` 이지만,
강도까지 보면:

`channel_activity[t3] = [4.2, 6.8, 0.0]`

처럼 볼 수 있다.

즉:

* **mask** = 이벤트 유무
* **count_active_channels** = 동시에 몇 채널이 터졌는지
* **channel_activity** = 각 채널이 얼마나 세게 터졌는지

이 3개가 같이 attention 쪽으로 넘어간다.

---

# 8. Step 4 — GRU hidden state 위에 anomaly-aware attention이 어떻게 붙나

여기서 핵심은:

GRU가 먼저 전체 24-step sequence를 읽어서 hidden state를 만들고,
그 다음 attention이 **critical한 시점의 hidden state를 더 중요하게 재조합**한다는 것.

---

## 8-1. 먼저 GRU가 hidden state를 만든다

입력:

* `encoder_input`: `[B, 24, 21]`

출력:

* `hidden_states`: `[B, 24, 128]`

즉 시점마다 128차원 표현이 생긴다.

toy로 8-step만 보면:

* `h1, h2, h3, h4, h5, h6, h7, h8`
* 각 `h_t` 는 길이 128 벡터

예를 들어 단순화해서 3차원으로만 쓰면:

* `h1 = [0.2, 0.1, 0.0]`
* `h2 = [0.5, 0.2, 0.1]`
* `h3 = [1.4, 1.1, 0.7]`
* `h4 = [0.3, 0.1, 0.2]`
* `h5 = [0.9, 0.8, 0.4]`
* `h6 = [1.6, 1.2, 0.9]`
* `h7 = [0.2, 0.1, 0.1]`
* `h8 = [0.1, 0.1, 0.0]`

여기서 t3, t5, t6이 이벤트 정보가 많이 반영된 상태라고 생각하면 된다.

---

## 8-2. critical 시점만 뽑는다

아까 `critical_mask = [0,1,1,0,1,1,0,0]` 였다.

그러면 critical한 hidden은:

* `h2`
* `h3`
* `h5`
* `h6`

이 4개다.

즉 attention은 전체 8개 중에서도
**이 4개의 이벤트성 hidden state를 중요 문맥 후보**로 본다.

---

## 8-3. 왜 attention을 또 하냐

GRU만 쓰면 hidden은 전 시점을 다 섞어서 들고 있다.

그런데 AAForecast는:

> “평상시보다, 이벤트가 있던 시점의 hidden을 미래 예측에 더 직접 반영하고 싶다”

이 철학이 있어서,
critical 시점 hidden들을 다시 모아서 context를 만든다.

---

## 8-4. 직관적 예시

미래 4-step 예측을 해야 하는데,
과거 24개 중 다음 시점들이 중요했다고 하자.

* t=3: Brent 급변 + GPRD_THREAT 동시 상승
* t=6: Brent 급변 + BS_A 상승

그러면 모델은 마지막 hidden만 보는 게 아니라,

* “아, 과거 t=3, t=6 같은 이벤트 시점이 있었지”
* “그 패턴과 연관된 정보를 미래 예측에 더 실어야겠다”

이렇게 동작하도록 만든다.

---

## 8-5. attended_states는 어떻게 생기나

출력도 shape은 똑같다.

* 입력: `hidden_states [B,24,128]`
* 출력: `attended_states [B,24,128]`

즉 각 시점의 hidden을 **이벤트 문맥이 반영된 버전**으로 다시 만든다.

예를 들어 어떤 마지막 시점 `t=8`의 원래 hidden이

`h8 = [0.1, 0.1, 0.0]`

인데, attention이 t3, t6을 중요하게 보면 attended version은

`a8 = [0.9, 0.7, 0.4]`

처럼 더 이벤트 문맥을 반영한 벡터가 될 수 있다.

물론 이 숫자는 toy지만 의미는 이거다.

* `hidden_states[t]` = 순수 GRU 인코딩 결과
* `attended_states[t]` = anomaly context를 반영해 재가중된 결과

---

## 8-6. count_active_channels가 여기서 왜 도움되나

예를 들어 t3와 t6 둘 다 critical이라도,

* t3는 2채널 동시 활성
* t5는 1채널만 활성

이면 t3를 더 중요한 이벤트로 볼 수 있다.

즉 attention 내부에서
`t3` hidden이 `t5`보다 더 큰 weight를 받을 논리가 생긴다.

간단히 말하면:

* mask만 쓰면: “이상치냐 아니냐”
* count/activity까지 쓰면: “얼마나 강한 복합 이벤트냐”

까지 반영 가능하다.

---

# 9. Step 5 — 왜 마지막 4개 hidden만 잘라서 horizon 4개와 맞추나

이 부분이 가장 헷갈릴 수 있는데,
핵심은 **input 24개를 읽고 output 4개를 만들어야 해서, hidden sequence 길이를 forecast horizon 길이와 맞추는 과정**이다.

---

## 9-1. 현재 상태

GRU와 attention까지 지나면 둘 다:

* `hidden_states`: `[B,24,128]`
* `attended_states`: `[B,24,128]`

이다.

그런데 우리가 원하는 최종 출력은:

* `y_hat`: `[B,4,1]`

즉 24개 표현을 그대로 decoder에 넣을 수는 없다.
**4개 horizon에 대응되는 길이 4의 표현**이 필요하다.

---

## 9-2. 여기서는 `_align_horizon()` 이 마지막 4개를 자른다

현재 설정은

* `input_size = 24`
* `h = 4`

그리고 `h <= input_size`

이 경우 가장 단순하게:

* 최근 hidden sequence의 마지막 4개를 사용

즉:

* `hidden_aligned = hidden_states[:, -4:, :]`
* `attended_aligned = attended_states[:, -4:, :]`

shape:

* `[B,4,128]`
* `[B,4,128]`

이렇게 된다.

---

## 9-3. toy 예시로 보면

24개 대신 8개, horizon=4라고 하자.

그러면 hidden이:

* `h1, h2, h3, h4, h5, h6, h7, h8`

있고 마지막 4개는:

* `h5, h6, h7, h8`

이다.

attention 결과도:

* `a5, a6, a7, a8`

이렇게 자른다.

그러면 decoder 입력은 각 horizon slot마다:

* horizon1용 입력: `[h5 ; a5]`
* horizon2용 입력: `[h6 ; a6]`
* horizon3용 입력: `[h7 ; a7]`
* horizon4용 입력: `[h8 ; a8]`

여기서 `;`는 concat이다.

즉 각 시점당 128 + 128 = 256차원.

---

## 9-4. 왜 마지막 4개를 쓰는 게 말이 되나

이 구조는
**“가장 최근 구간의 hidden들이 미래 4-step을 내는 데 가장 직접적인 요약 표현이다”**
라는 가정이다.

즉 모델은 24개 전체를 GRU로 이미 읽었고,
마지막 쪽 hidden일수록 과거 정보가 더 많이 누적되어 있다.

예를 들어:

* `h21`은 1~21 시점 정보 반영
* `h22`는 1~22 시점 정보 반영
* `h23`는 1~23 시점 정보 반영
* `h24`는 1~24 시점 정보 반영

그래서 마지막 4개 hidden을 뽑으면
이미 과거 전체를 요약한 “최근 표현 묶음”이 된다.

---

## 9-5. concat 후 decoder로 간다

이제:

* `hidden_aligned`: `[B,4,128]`
* `attended_aligned`: `[B,4,128]`

concat:

* `decoder_input = [B,4,256]`

toy로 쓰면:

* step1 input = `[h21 ; a21]`
* step2 input = `[h22 ; a22]`
* step3 input = `[h23 ; a23]`
* step4 input = `[h24 ; a24]`

이걸 MLP decoder가 받아서 각 horizon별 scalar를 만든다.

출력:

* `[B,4,1]`

예를 들면:

* horizon1 = 62.02
* horizon2 = 62.51
* horizon3 = 63.03
* horizon4 = 63.28

같은 식이다.

---

# Step 3~5를 한 번에 묶어보면

이렇게 이해하면 된다.

### 입력 24개 시점

Brent + exog가 들어옴

### STAR 수행

target, star exog에서 anomaly mask 생성

예:

* t=20, t=22, t=23 이 critical

### critical/context 생성

* `critical_mask`: 이 시점이 이벤트 시점인지
* `count_active_channels`: 몇 채널이 동시에 터졌는지
* `channel_activity`: 얼마나 강하게 터졌는지

### GRU 인코딩

24개 시점 각각에 대해 hidden 생성

* `hidden_states [B,24,128]`

### anomaly-aware attention

critical한 시점의 hidden을 더 중요하게 모아
이벤트 문맥 반영 버전 생성

* `attended_states [B,24,128]`

### horizon alignment

마지막 4개만 추출

* `hidden[:,20:24]`
* `attended[:,20:24]`

### concat + decoder

각 horizon별로 256차원 입력을 만들어 예측

* `[B,4,256] -> [B,4,1]`

---

# 
