# stability_dh.md

# `stability_dh` 실제 데이터 흐름 해설

이 문서는 `stability_dh` 실험에서
**실제 데이터가 어떤 row로 잘리고, 어떤 변환을 거쳐, 어떤 분해 결과를 만들고, 그게 어떤 형태의 텐서로 모델에 들어가는지**를 최대한 계산 단위로 쪼개서 설명한다.

설명 기준 artifact:
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_stability_dh/config/config.resolved.json`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_stability_dh/summary/result.csv`

---

## 0. 이 실험의 핵심 설정

### 데이터/타깃
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

### CV / training
- `input_size = 64`
- `horizon = 2`
- `n_windows = 1`
- `step_size = 4`
- `transformations_target = diff`
- `transformations_exog = diff`

### AA-Forecast 설정
- backbone: `informer`
- `hidden_size = 128`
- `n_head = 4`
- `encoder_layers = 2`
- `linear_hidden_size = 96`
- `decoder_hidden_size = 128`
- `decoder_layers = 4`
- `season_length = 4`
- `lowess_frac = 0.35`
- `lowess_delta = 0.01`
- `thresh = 3.5`
- `retrieval.enabled = false`
- uncertainty enabled, dropout 후보 12개, sample_count 30

### STAR 그룹
- star hist exog: `GPRD_THREAT`
- non-star hist exog: 나머지 9개
- star tail mode:
  - `GPRD_THREAT` = `upward`

즉 이 실험은,
**10개 exog 중 오직 `GPRD_THREAT` 하나만 STAR decomposition 대상으로 두고,
나머지는 non-star raw/diff 채널로 encoder에 넣는 구조**다.

---

## 1. 마지막 fold에서 모델이 예측한 실제 결과

마지막 fold cutoff는 `2026-02-23` 이고,
그 다음 두 개 주를 예측한다.

| horizon | ds | actual y | predicted y_hat |
|---:|---|---:|---:|
| 1 | 2026-03-02 | 86.64365714 | 77.5370032270 |
| 2 | 2026-03-09 | 98.88672857 | 82.8791667262 |

즉 최종 산출은:
- `h1 = 77.5370`
- `h2 = 82.8792`
- `gap = 5.3422`

---

## 2. 모델 입력 window는 정확히 어디서 어디까지인가

`input_size = 64` 이므로,
모델이 마지막 fold에서 보는 입력 window는 **cutoff 포함 최근 64개 주간 row**다.

- window start: `2024-12-09`
- window end: `2026-02-23`
- horizon forecast dates:
  - `2026-03-02`
  - `2026-03-09`

즉 개념적으로:
- 입력: `2024-12-09 ~ 2026-02-23`
- 출력: `2026-03-02, 2026-03-09`

---

## 3. 실제 raw level tail 12 rows

아래는 cutoff 직전 tail 12 rows의 실제 raw 값이다.

| dt | Brent | GPRD_THREAT | BS_Core_Index_A | GPRD | GPRD_ACT | BS_B | BS_C | OVX | LMEX | BCOM | DXY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-12-08 | 61.6601 | 124.4465 | -0.0718 | 107.9064 | 98.4823 | -0.1544 | -0.3336 | 31.2880 | 4824.14 | 109.7635 | 98.6435 |
| 2025-12-15 | 60.1329 | 152.7953 | 0.0931 | 140.0735 | 127.8944 | -0.3448 | -0.2273 | 34.6080 | 4859.10 | 108.3599 | 98.5015 |
| 2025-12-22 | 61.5232 | 127.6635 | 0.2349 | 129.2938 | 136.3667 | -0.5292 | 0.0744 | 32.4950 | 4984.73 | 111.2127 | 98.0047 |
| 2025-12-29 | 60.9655 | 139.7818 | 0.2432 | 127.0956 | 125.8220 | -0.4229 | -0.2066 | 30.3725 | 5091.38 | 110.3752 | 98.3508 |
| 2026-01-05 | 62.0721 | 230.4503 | 0.3098 | 188.1892 | 131.9713 | -0.2864 | 0.0436 | 33.0100 | 5313.40 | 111.9372 | 98.8177 |
| 2026-01-12 | 64.2552 | 270.2686 | 0.5900 | 199.1509 | 106.0008 | -0.3853 | -0.6355 | 41.7440 | 5433.98 | 114.1818 | 99.2097 |
| 2026-01-19 | 65.1314 | 192.3791 | 1.2587 | 143.4654 | 80.0684 | -0.5270 | 0.1288 | 45.1425 | 5364.74 | 117.6876 | 98.1070 |
| 2026-01-26 | 67.9775 | 179.3480 | 2.8126 | 146.5533 | 97.7555 | -0.7737 | 0.2704 | 51.8120 | 5505.00 | 122.2131 | 96.7202 |
| 2026-02-02 | 67.6057 | 197.6168 | 2.7211 | 146.3858 | 83.5047 | -0.5919 | 1.7066 | 52.8640 | 5341.38 | 117.4815 | 97.5965 |
| 2026-02-09 | 68.3835 | 156.9218 | 1.8195 | 135.0997 | 122.6667 | -0.5482 | 1.5767 | 45.4640 | 5367.40 | 117.8900 | 96.8906 |
| 2026-02-16 | 70.3128 | 100.7189 | 2.5001 | 81.7474 | 57.8866 | -0.4255 | 1.6618 | 51.7200 | 5275.54 | 117.8648 | 97.5935 |
| 2026-02-23 | 72.7244 | 151.5088 | 2.8935 | 106.4258 | 51.1552 | -0.4000 | 1.7742 | 60.2640 | 5456.76 | 120.5224 | 97.7227 |

이건 모델이 직접 보는 값이 아니라,
**diff 변환 전 원본 level tail** 이다.

---

## 4. 실제 diff 입력은 어떻게 만들어지는가

runtime 설정에서
- `transformations_target = diff`
- `transformations_exog = diff`

이므로,
각 column은 직전 시점 차분으로 바뀐다.

수식은 단순하다.

예를 들어 target Brent에 대해
- `diff_t = y_t - y_(t-1)`

예시:
- 2026-01-12 Brent diff
  - raw: `64.25517143`
  - previous raw: `62.0721`
  - diff: `64.25517143 - 62.0721 = 2.18307143`

`GPRD_THREAT`도 동일하다.

예시:
- 2026-02-23 GPRD_THREAT diff
  - raw: `151.5087650844`
  - previous raw: `100.7189107622`
  - diff: `50.7898543222`

---

## 5. 실제 diff tail 12 rows

아래는 같은 tail 12 rows에 대한 실제 diff 값이다.

| dt | Brent_diff | GPRD_THREAT_diff |
|---|---:|---:|
| 2025-12-08 | -1.616943 | -40.701694 |
| 2025-12-15 | -1.527157 | 28.348873 |
| 2025-12-22 | 1.390286 | -25.131877 |
| 2025-12-29 | -0.557714 | 12.118305 |
| 2026-01-05 | 1.106600 | 90.668530 |
| 2026-01-12 | 2.183071 | 39.818267 |
| 2026-01-19 | 0.876200 | -77.889493 |
| 2026-01-26 | 2.846129 | -13.031122 |
| 2026-02-02 | -0.371757 | 18.268886 |
| 2026-02-09 | 0.777757 | -40.694993 |
| 2026-02-16 | 1.929314 | -56.202938 |
| 2026-02-23 | 2.411543 | 50.789854 |

여기서 중요한 관찰은:
- target Brent diff는 대체로 `-1.6 ~ +2.8` 수준
- `GPRD_THREAT_diff` 는 `-77.9 ~ +90.7` 수준까지 튄다

즉 이 실험에서 star 경로가 보는 핵심 burst는,
**Brent 자체보다 `GPRD_THREAT` diff 쪽에서 더 극단적**이다.

---

## 6. STAR 분해는 정확히 어떤 계산을 하는가

코드 기준 (`plugins/aa_forecast/modules.py`) STARFeatureExtractor는 아래 순서로 계산한다.

### 6.1 trend
입력 시계열을 `x_t` 라고 하면,
LOWESS로 부드러운 trend `T_t` 를 만든다.

- `T_t = LOWESS(x_t)`
- 여기서
  - `frac = 0.35`
  - `delta = 0.01 * seq_len`

이 run에서는 seq_len이 diff valid rows 기준으로 63에 가깝기 때문에,
LOWESS delta는 대략 `0.63` 스케일로 들어간다.

### 6.2 detrended
trend를 나눠서 detrended signal을 만든다.

- `D_t = x_t / T_t`

코드에서는 `_safe_divide`를 쓰므로,
trend 절댓값이 아주 작으면 `eps=1e-4` 로 안정화한다.

### 6.3 seasonal
season length 4 기준으로 같은 phase끼리 평균을 낸다.

예를 들어 index를 4주 주기 phase로 나누면:
- phase 0: `t = 0,4,8,...`
- phase 1: `t = 1,5,9,...`
- phase 2: `t = 2,6,10,...`
- phase 3: `t = 3,7,11,...`

각 phase 평균을 해당 위치 seasonal baseline으로 복사한다.

- `S_t = mean(D_phase(t))`

### 6.4 residual
- `R_t = D_t / S_t`

즉 최종 residual은
**(원시 입력) / (LOWESS trend) / (seasonal baseline)**
형태로 된다.

### 6.5 robust z-score
코드는 residual에 대해 median/MAD 기반 robust score를 계산한다.

- residual center = median over time
- `MAD = median(|R_t - median(R)|)`
- signed score:
  - `z_t = 0.6745 * (R_t - median(R)) / MAD`

### 6.6 anomaly mask
- 기본 규칙: `|z_t| > thresh`
- 여기서 `thresh = 3.5`
- 하지만 `GPRD_THREAT` 는 `upward` mode 이므로
  - `z_t > 3.5` 일 때만 anomaly로 본다.
  - 큰 음수는 anomaly로 치지 않는다.

### 6.7 anomalies / cleaned residual
코드 반환값은 아래처럼 나뉜다.

- `anomalies`: anomaly인 시점만 residual 값을 남기고, 나머지는 1
- `residual`: anomaly가 아닌 시점 residual만 남기고, anomaly 시점은 1

즉 anomaly와 non-anomaly residual이 분리된다.

---

## 7. 실제 target diff에 대한 STAR 출력 예시

아래는 target `Com_BrentCrudeOil_diff` 의 tail 12 rows에 대해,
코드 그대로 STARFeatureExtractor를 돌린 결과다.

표 컬럼:
- `value`: diff 입력값
- `trend`: LOWESS trend
- `seasonal`: phase 평균
- `anomalies`: anomaly면 residual 값, 아니면 1
- `residual`: anomaly 아니면 residual 값, anomaly면 1
- `score_signed`: robust signed z-score
- `critical`: anomaly mask

| dt | value | trend | seasonal | anomalies | residual | score_signed | critical |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-12-08 | -1.616943 | 0.117929 | -1.576020 | 1.0 | 8.699829 | 3.051672 | 0 |
| 2025-12-15 | -1.527157 | 0.228918 | 4.712933 | 1.0 | -1.415511 | -0.582885 | 0 |
| 2025-12-22 | 1.390286 | 0.371749 | 1.903204 | 1.0 | 1.965029 | 0.631781 | 0 |
| 2025-12-29 | -0.557714 | 0.522148 | -0.530614 | 1.0 | 2.012981 | 0.649011 | 0 |
| 2026-01-05 | 1.106600 | 0.679601 | -1.576020 | 1.0 | -1.033177 | -0.445508 | 0 |
| 2026-01-12 | 2.183071 | 0.843800 | 4.712933 | 1.0 | 0.548955 | 0.122970 | 0 |
| 2026-01-19 | 0.876200 | 1.013753 | 1.903204 | 1.0 | 0.454136 | 0.088900 | 0 |
| 2026-01-26 | 2.846129 | 1.187893 | -0.530614 | 1.0 | -4.515427 | -1.696720 | 0 |
| 2026-02-02 | -0.371757 | 1.364746 | -1.576020 | 1.0 | 0.172841 | -0.012172 | 0 |
| 2026-02-09 | 0.777757 | 1.542722 | 4.712933 | 1.0 | 0.106971 | -0.035840 | 0 |
| 2026-02-16 | 1.929314 | 1.721289 | 1.903204 | 1.0 | 0.588930 | 0.137333 | 0 |
| 2026-02-23 | 2.411543 | 1.900628 | -0.530614 | 1.0 | -2.391220 | -0.933469 | 0 |

중요한 해석:
- target diff 자체는 tail 12 구간에서 **threshold 3.5를 넘는 anomaly가 안 잡힌다**
- 즉 이 fold에서 spike direction 힌트는 target 자체보다,
  **star exog 쪽에서 더 많이 올 가능성**이 있다.

---

## 8. 실제 star exog `GPRD_THREAT_diff` 에 대한 STAR 출력 예시

같은 방식으로 `GPRD_THREAT_diff` 결과는 아래와 같다.

| dt | value | trend | seasonal | anomalies | residual | score_signed | critical |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-12-08 | -40.701694 | 1.405427 | -7.034438 | 1.0 | 4.116941 | 2.125499 | 0 |
| 2025-12-15 | 28.348873 | 0.863356 | 4.419395 | 7.429898 | 1.0 | 3.919273 | 1 |
| 2025-12-22 | -25.131877 | -0.456796 | 6.127777 | 8.978421 | 1.0 | 4.757709 | 1 |
| 2025-12-29 | 12.118305 | -1.347367 | -13.988150 | 1.0 | 0.642977 | 0.244548 | 0 |
| 2026-01-05 | 90.668530 | -2.031320 | -7.034438 | 1.0 | 6.345250 | 3.331999 | 0 |
| 2026-01-12 | 39.818267 | -2.649915 | 4.419395 | 1.0 | -3.400068 | -1.944528 | 0 |
| 2026-01-19 | -77.889493 | -3.290621 | 6.127777 | 1.0 | 3.862763 | 1.987875 | 0 |
| 2026-01-26 | -13.031122 | -4.015161 | -13.988150 | 1.0 | -0.232016 | -0.229211 | 0 |
| 2026-02-02 | 18.268886 | -4.842699 | -7.034438 | 1.0 | 0.536284 | 0.186780 | 0 |
| 2026-02-09 | -40.694993 | -5.774807 | 4.419395 | 1.0 | 1.594559 | 0.759774 | 0 |
| 2026-02-16 | -56.202938 | -6.814566 | 6.127777 | 1.0 | 1.345916 | 0.625148 | 0 |
| 2026-02-23 | 50.789854 | -7.937887 | -13.988150 | 1.0 | 0.457417 | 0.144077 | 0 |

여기서 진짜 중요한 부분은:
- `GPRD_THREAT` 는 upward mode이므로,
  **양의 robust score가 3.5 초과일 때만 anomaly** 다.
- 그래서 tail 12 구간에서 실제 anomaly로 잡힌 건:
  - `2025-12-15` (`score = 3.919273`)
  - `2025-12-22` (`score = 4.757709`)

즉 이 실험에서 model은 최근 burst window 안에서,
**star exogenous 쪽 상승 이벤트 흔적을 explicit anomaly token처럼 받는다**고 이해할 수 있다.

---

## 9. encoder 입력 feature는 실제로 어떻게 붙는가

이 run에서 입력 feature는 대략 아래 18차원 조각으로 합쳐진다.

### 9.1 target 쪽 5개
1. raw target diff
2. target trend
3. target seasonal
4. target anomalies
5. target residual

### 9.2 non-star exog 9개
6. `BS_Core_Index_A_diff`
7. `GPRD_diff`
8. `GPRD_ACT_diff`
9. `BS_Core_Index_B_diff`
10. `BS_Core_Index_C_diff`
11. `Idx_OVX_diff`
12. `Com_LMEX_diff`
13. `Com_BloombergCommodity_BCOM_diff`
14. `Idx_DxyUSD_diff`

### 9.3 star exog `GPRD_THREAT` 쪽 4개
15. trend
16. seasonal
17. anomalies
18. residual

즉 각 시점마다 모델은 단순 Brent diff 하나만 보는 게 아니라,
**target STAR 4개 + star exog STAR 4개 + non-star diff 9개 + raw target diff** 를 동시에 본다.

---

## 10. encoder 이후 decoder에서 무슨 일이 일어나는가

이 실험의 Informer AA decoder는 단순 linear head가 아니다.

개념적으로는:
1. encoder가 window 전체 representation을 만든다.
2. 여기서
   - event summary
   - event path
   - regime latent
   - pooled context
   를 만든다.
3. decoder input은 이 정보를 이용해 scale/shift/gate로 조정된다.
4. trajectory path가 future 2-step shock profile을 만든다.
5. 최종 forecast는 baseline level 위에 trajectory 계열 uplift가 더해진다.

즉 이 케이스의 `77.54 -> 82.88` 는,
단순 extrapolation이 아니라
**STAR로 추출한 이벤트성 정보 + Informer decoder의 trajectory path** 가 결합된 결과다.

---

## 11. 이 케이스가 왜 상대적으로 잘됐는가

이 실험은 retrieval 없이도 아래를 동시에 만족한다.

1. **우상향 형성**
   - `77.54 -> 82.88`
2. **gap 확보**
   - `+5.34`
3. **star exog를 1개만 써서 구조 불안정 최소화**
4. **작은 Informer 크기**
   - `128 / 96 / 128`
5. **target 쪽 과도한 anomaly보다 exogenous event path를 더 활용**
   - 마지막 tail 구간에서 target anomaly는 약했지만,
   - `GPRD_THREAT` anomaly가 실제로 잡혔다.

즉 이 케이스는
**“Brent 자체의 diff spike보다, exogenous burst를 통해 horizon uplift를 만든 no-retrieval 케이스”**
로 이해하는 게 맞다.

---

## 12. 그런데 왜 아직 목표에는 못 미치는가

실제값 대비 보면:
- h1 actual `86.64` vs pred `77.54`
- h2 actual `98.89` vs pred `82.88`

즉,
- 방향은 맞고
- gap도 충분하지만
- spike amplitude를 끝까지 level forecast로 운반하는 힘이 아직 부족하다.

한 줄로 말하면,
`stability_dh` 의 병목은
**anomaly를 못 찾는 것보다, 찾은 spike 근거를 최종 forecast level까지 충분히 transport하지 못하는 것**이다.

---

## 13. 최종 한 줄 요약

`stability_dh` 는
**최근 64-step diff window + `GPRD_THREAT` 하나만 star 경로로 태운 AA-Informer** 가,
마지막 fold에서 실제 spike 전조를 받아
`77.54 -> 82.88` 을 만든 케이스다.

즉,
- no-retrieval 기준에서 좋은 기준점이고,
- `gap >= 4` 는 만족하지만,
- 절대 레벨 `78 / 85` 는 아직 못 넘은
**근접 성공 케이스** 다.
