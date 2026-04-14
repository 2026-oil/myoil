# AAForecast retrieval 상세 흐름 (STAR 이후, 수식 포함)

## 대상 옵션

아래 설정이 켜진 경우의 retrieval 단계를 설명한다.

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

이 문서는 **STAR 분해가 끝난 뒤** retrieval이 어떤 입력을 받아 어떤 수식으로 이웃을 고르고, 어떻게 `base_prediction`을 `final_prediction`으로 바꿔서 최종 `y_hat`를 만드는지에만 집중한다.

## 전제

- retrieval은 `aa_forecast.uncertainty.enabled=true`를 요구한다.
- retrieval이 enabled일 때는 `use_shape_key` 또는 `use_event_key` 중 적어도 하나는 `true`여야 한다.
- 현재 구현은 `mode=posthoc_blend`, `similarity=cosine`이다.
- 현재 similarity 결합 상수는 코드상 `shape=0.20`, `event=0.80`이다.

관련 validation은 `plugins/aa_forecast/config.py::_normalize_retrieval_config(...)`에서 수행된다.

---

## 큰 흐름

retrieval은 `plugins/aa_forecast/runtime.py::predict_aa_forecast_fold(...)` 안에서 uncertainty 계산 다음에 실행된다.

흐름을 축약하면 아래와 같다.

1. `AAForecast.forward(...)`에서 STAR 기반 내부 상태를 만든다.
2. uncertainty path가 켜져 있으면 dropout 후보들로 반복 예측해 `prediction_mean`, `uncertainty_std`를 만든다.
3. retrieval query를 만들기 위해 마지막 `input_size` 구간에서 shape/event signature를 계산한다.
4. train history 안에서 retrieval bank 후보들을 만들고 동일한 signature를 계산한다.
5. query와 후보 간 similarity를 계산하고, threshold/min-similarity 필터를 통과한 후보만 남긴다.
6. 상위 `top_k` 후보에 softmax weight를 준다.
7. 후보들의 future return을 가중 평균해 `memory_prediction`을 만든다.
8. `base_prediction`과 `memory_prediction`을 blend해서 `final_prediction`을 만든다.
9. `final_prediction`이 최종 `target_predictions[job.model]`이 되고, 이후 forecast CSV의 `y_hat`로 저장된다.

코드 경로는 대략 다음 순서다.

- `predict_aa_forecast_fold(...)`
- `_build_retrieval_signature(...)`
- `_build_event_memory_bank(...)`
- `_build_event_query(...)`
- `_retrieve_event_neighbors(...)`
- `_blend_event_memory_prediction(...)`
- `_write_retrieval_artifacts(...)`

---

## STAR 분해 직후 retrieval 입력으로 무엇이 남는가

retrieval은 raw input 전체를 그대로 쓰지 않는다. 핵심은 STAR 이후의 event 신호와 최근 target path다.

`AAForecast.forward(...)`와 `model._compute_star_outputs(...)`가 만들어 주는 retrieval 핵심 입력은 아래 세 가지다.

### 1. `critical_mask`

- 시점별/채널별로 "이 위치가 critical event인가"를 나타내는 마스크다.
- retrieval에서는 이를 flatten해서 event signature 일부로 사용한다.

### 2. `count_active_channels`

- 각 시점에서 활성화된 critical 채널 수다.
- event intensity의 거친 요약으로 쓰인다.

### 3. `channel_activity`

- STAR가 계산한 활동량 텐서다.
- retrieval에서는 flatten된 activity 전체, channel별 합, channel별 최댓값이 event signature 구성에 들어간다.

### 4. 최근 target diff window

- query/candidate 모두 최근 `input_size` 길이의 transformed target window를 사용한다.
- 이 문맥에서 target은 raw level이 아니라 runtime diff transform 이후 값이다.
- 따라서 retrieval의 shape 비교는 **level path가 아니라 diff path** 위에서 이뤄진다.

---

## 실제 데이터 샘플로 보는 query와 chosen neighbor

이 문서의 수식이 추상적으로 느껴질 수 있어서, 실제 run artifact와 `data/df.csv`의 raw row를 같이 놓고 보자.

retrieval applied 예시에서 query cutoff는:

- `train_end_ds = 2026-03-02`
- retrieval artifact: `aa_forecast/retrieval/20260302T000000.json`

이때 query의 최근 raw 관측 꼬리 일부는 다음과 같다.

```csv
dt,Com_BrentCrudeOil,GPRD_THREAT,BS_Core_Index_A,GPRD
2026-01-26,67.9775,179.34795597621374,2.812635457,146.55332510811942
2026-02-02,67.60574286,197.61684199741907,2.721093748,146.38579777308874
2026-02-09,68.3835,156.9218488420759,1.819512983,135.0996573311942
2026-02-16,70.31281429,100.7189107622419,2.500105145,81.74744279044015
2026-02-23,72.72435714,151.5087650844029,2.893495748,106.42579214913503
2026-03-02,86.64365714,290.84305463518416,4.1328195,226.48289271763392
```

이 raw row들이 retrieval에서 그대로 cosine 비교에 들어가는 것은 아니다. 실제 query는:

1. target `Com_BrentCrudeOil`를 diff transform한 최근 `input_size` 길이 window
2. STAR가 계산한 `critical_mask`, `count_active_channels`, `channel_activity`

를 요약한 shape/event signature로 변환된다.

같은 artifact에서 선택된 top-1 neighbor는:

- `candidate_end_ds = 2024-12-30`
- `candidate_future_end_ds = 2025-01-13`
- `softmax_weight = 1.0`

이 candidate의 raw tail 일부는 다음과 같다.

```csv
dt,Com_BrentCrudeOil,GPRD_THREAT,BS_Core_Index_A,GPRD
2024-11-25,72.44594286,186.03195190429688,-0.468832571,154.47337341308594
2024-12-02,71.92588571,164.7931867327009,-0.793607165,179.8975110735212
2024-12-09,73.48958571,122.31811250959124,-0.758465085,138.68458993094308
2024-12-16,73.21955714,135.511109488351,-0.294751248,161.9764862060547
2024-12-23,73.44568571,126.26148469107493,-0.810762088,124.2609656197684
2024-12-30,75.35178333,117.269592830113,-0.681996982,122.6793191092355
```

그리고 candidate 뒤의 실제 future raw 값은:

- `2025-01-06`: `77.89148333`
- `2025-01-13`: `80.95013333`

따라서 이 candidate의 future return은 raw level 기준으로 다음처럼 만들어진다.

$$
r^{(\mathrm{cand})}_1 = \frac{77.89148333 - 75.35178333}{75.35178333}
$$

$$
r^{(\mathrm{cand})}_2 = \frac{80.95013333 - 75.35178333}{75.35178333}
$$

artifact에 기록된 값은 실제로:

- `future_returns = [0.03370457722118514, 0.07429618454393117]`

이다.

즉 이 예시를 raw 데이터 관점에서 읽으면:

- query는 `2026-03-02`까지의 최근 event burst와 price acceleration을 요약한 signature
- chosen neighbor는 `2024-12-30` 시점의 유사한 event signature
- retrieval은 그 뒤 실제로 있었던 상대 상승 경로를 현재 마지막 raw level `86.64365714`에 다시 적용하려는 시도

로 해석하면 된다.

---

## retrieval signature 수식

### 정규화

$$
\mathrm{normalize}(v)=\frac{v}{\lVert v \rVert_2}
$$

단, norm이 매우 작으면 영벡터를 반환한다.

코드 대응: `_normalize_signature(...)`

### shape signature

최근 `input_size` 길이의 target diff window를 shape signature로 만든다.

$$
s_{\text{shape}} = \mathrm{normalize}(y_{t-L+1:t})
$$

여기서:
- $L$ = `input_size`
- $y_{t-L+1:t}$ = transformed target window

코드 대응: `_build_retrieval_signature(...)`

### event signature

STAR 이후 payload에서 다음 벡터를 이어 붙여 정규화한다.

$$
s_{\text{event}} = \mathrm{normalize}([\text{critical\_mask},\ \text{count\_active},\ \text{channel\_activity},\ \text{activity\_sums},\ \text{activity\_max}])
$$

코드 대응:
- `_require_star_payload(...)`
- `_build_retrieval_signature(...)`

### event score

query/candidate를 retrieval에 쓸지 말지 결정하는 강도 점수다.

$$
\text{event\_score} = \sum \text{count\_active\_channels} + \sum |\text{channel\_activity}|
$$

코드 대응: `_build_retrieval_signature(...)`

---

## candidate bank 구성

`_build_event_memory_bank(...)`는 train history 안에서 과거 window들을 훑으며 candidate를 만든다.

각 candidate에 대해 수행하는 일은 다음과 같다.

1. 길이 `input_size`의 transformed window를 자른다.
2. 그 window로 shape/event signature를 만든다.
3. candidate `event_score`가 `event_score_threshold`보다 작으면 버린다.
4. surviving candidate에 대해 anchor 시점의 raw target과 그 뒤 horizon 구간의 raw future를 사용해 future return을 계산한다.

future return은 raw level 기준으로 아래처럼 만든다.

$$
r^{(i)}_{\text{future}} = \frac{y^{(i)}_{\text{future}} - y^{(i)}_{\text{anchor}}}{\max(|y^{(i)}_{\text{anchor}}|, \epsilon)}
$$

즉 candidate bank는 "과거의 event signature + 그 뒤 실제로 어떤 상대 변화가 일어났는가"를 같이 저장한 메모리다.

코드 대응: `_build_event_memory_bank(...)`

---

## query 구성

`_build_event_query(...)`는 현재 fold에서 마지막 `input_size` 길이의 transformed train window를 꺼내 shape/event signature를 만든다.

즉 query는
- 지금 막 예측 직전에 보이는 target diff path
- 지금 막 관측된 STAR activity pattern

을 요약한 벡터다.

코드 대응: `_build_event_query(...)`

### 실제 query diff 샘플

retrieval skipped 케이스(`train_end_ds = 2026-02-23`)의 최근 transformed target/exog 꼬리 일부는 다음과 같다.

```csv
dt,Com_BrentCrudeOil,GPRD_THREAT,BS_Core_Index_A,GPRD
2026-01-19,0.8761999999999972,-77.88949257986886,0.668659239,-55.68550545828677
2026-01-26,2.8461285700000047,-13.031122480119961,1.5539442609999998,3.087921142578125
2026-02-02,-0.3717571399999997,18.268886021205333,-0.09154170899999992,-0.16752733503068384
2026-02-09,0.7777571399999914,-40.694993155343184,-0.9015807649999998,-11.286140441894531
2026-02-16,1.9293142900000078,-56.202938079833984,0.680592162,-53.35221454075406
2026-02-23,2.4115428499999894,50.789854322161005,0.39339060299999984,24.678349358694888
```

이 케이스의 retrieval artifact에는 다음이 기록돼 있다.

- `query_event_score = 200.67826199531555`
- `event_score_threshold = 400.0`
- `retrieval_applied = false`
- `skip_reason = below_event_threshold`

즉 이 경우는 similarity 계산 이전에 **query event score 자체가 너무 약해서** retrieval이 닫힌다.

반대로 retrieval applied 케이스(`train_end_ds = 2026-03-02`)의 최근 transformed 꼬리 일부는 다음과 같다.

```csv
dt,Com_BrentCrudeOil,GPRD_THREAT,BS_Core_Index_A,GPRD
2026-01-26,2.8461285700000047,-13.031122480119961,1.5539442609999998,3.087921142578125
2026-02-02,-0.3717571399999997,18.268886021205333,-0.09154170899999992,-0.16752733503068384
2026-02-09,0.7777571399999914,-40.694993155343184,-0.9015807649999998,-11.286140441894531
2026-02-16,1.9293142900000078,-56.202938079833984,0.680592162,-53.35221454075406
2026-02-23,2.4115428499999894,50.789854322161005,0.39339060299999984,24.678349358694888
2026-03-02,13.919300000000007,139.33428955078125,1.2393237520000002,120.05710056849888
```

이 케이스의 retrieval artifact에는 다음이 기록돼 있다.

- `query_event_score = 530.2648351192474`
- `retrieval_applied = true`

즉 `2026-03-02` 직전 query는 target diff와 STAR exog diff가 함께 크게 튀면서 threshold를 넘었고, 그래서 retrieval path가 열린다.

---

## similarity 계산

### cosine similarity

shape/event 모두 cosine similarity를 쓴다.

$$
\cos(a,b)=\frac{a \cdot b}{\lVert a \rVert_2 \lVert b \rVert_2}
$$

코드 대응: `_cosine_similarity(...)`

### event bonus

`use_event_key=true`이고 `event_score_log_bonus_alpha > 0`이면, event similarity에 추가 bonus를 붙인다.

$$
\text{bonus} = \min\left(\max\left(\log\frac{\text{candidate\_score}}{\text{query\_score}}, 0\right),\ \text{cap}\right)
$$

$$
\text{event\_component} = \text{event\_similarity} + \alpha \cdot \text{bonus}
$$

현재 옵션에서는:
- $\alpha = 0.15$
- $\text{cap} = 0.1$

즉 candidate event score가 query보다 크면 event similarity에 최대 `0.015`까지 추가 보너스가 붙을 수 있다.

코드 대응: `_retrieve_event_neighbors(...)`

### combined similarity

현재 옵션은:
- `use_shape_key=false`
- `use_event_key=true`

따라서 실제 similarity는 아래처럼 단순해진다.

$$
\text{similarity} = \text{event\_component}
$$

참고로 코드상 일반식은 아래와 같다.

둘 다 켜진 경우:

$$
\text{similarity}=0.20\cdot\text{shape\_similarity}+0.80\cdot\text{event\_component}
$$

shape만 켜진 경우:

$$
\text{similarity}=\text{shape\_similarity}
$$

코드 대응: `_retrieve_event_neighbors(...)`

### min similarity filter

candidate마다 계산된 similarity가 `min_similarity`보다 작으면 버린다.

현재 값은:

$$
\text{min\_similarity}=0.35
$$

따라서 event similarity(+bonus)가 0.35 미만이면 retrieval 후보가 되지 못한다.

---

## top-k 선택과 softmax weight

similarity를 통과한 후보를 내림차순 정렬한 뒤 상위 `top_k`개만 남긴다.

현재 값은:

$$
\text{top\_k}=1
$$

따라서 실제로는 최고 점수 candidate 1개만 남는다.

그 뒤 softmax weight를 계산한다.

$$
z_i = \frac{\text{similarity}_i}{T}
$$

$$
p_i = \frac{e^{z_i-\max z}}{\sum_j e^{z_j-\max z}}
$$

현재 `top_k=1`이면 결과적으로 softmax weight는 사실상 1이 된다.

실제 applied artifact에서도:

- `top_k_used = 1`
- chosen neighbor `softmax_weight = 1.0`

이 확인된다.

코드 대응: `_retrieve_event_neighbors(...)`

---

## memory prediction 생성

선택된 neighbor들의 future return을 softmax weight로 가중 평균해 memory return을 만든다.

$$
r^{(\mathrm{mem})} = \sum_i p_i \cdot r_i^{(\mathrm{future})}
$$

그 다음 현재 fold의 마지막 raw target level을 anchor로 써서 memory prediction을 복원한다.

$$
\hat y^{(\mathrm{mem})} = y_{\text{last}} + \max(|y_{\text{last}}|,\epsilon) \cdot r^{(\mathrm{mem})}
$$

즉 retrieval은 diff target 자체를 직접 blend하는 게 아니라, **과거 event analogue의 상대 변화 패턴**을 현재 마지막 level에 다시 입혀 memory trajectory를 만드는 구조다.

applied 예시에서 실제 값은 다음과 같다.

- 현재 마지막 raw level (`2026-03-02`): `86.64365714`
- chosen neighbor anchor (`2024-12-30`): `75.35178333`
- chosen neighbor future return:
  - `0.03370457722118514`
  - `0.07429618454393117`
- artifact의 `memory_prediction`:
  - step 1: `89.56394497280102`
  - step 2: `93.08095028043454`

즉 이 메모리 경로는 “과거 2024-12-30 이후에 있었던 상대 상승 패턴”을 현재 2026-03-02 레벨 `86.64365714`에 다시 입힌 결과다.

코드 대응: `predict_aa_forecast_fold(...)`

---

## uncertainty gate와 blend weight

먼저 retrieval 코드에서 similarity를 0~1 사이로 clip한 `similarity_scale`을 만든다.

현재 옵션은 `use_uncertainty_gate=false`이므로 uncertainty scale은 horizon마다 1이다.

일반식은 아래와 같다.

uncertainty gate를 켠 경우:

$$
u_h = \frac{\sigma_h}{\max_h \sigma_h}
$$

분모가 작으면:

$$
u_h = 1
$$

현재 설정에서는 gate를 쓰지 않으므로:

$$
u_h = 1
$$

blend weight는 아래처럼 계산된다.

$$
\lambda_h = \text{blend\_floor} + (\text{blend\_max}-\text{blend\_floor}) \cdot \text{similarity\_scale} \cdot u_h
$$

현재 설정에서는:
- `blend_floor = 0.0`
- `blend_max = 1.0`
- `use_uncertainty_gate = false`

따라서 사실상:

$$
\lambda_h = \text{similarity\_scale}
$$

applied artifact에서는:

- `mean_similarity = 0.8919193026076779`
- `blend_weight_by_horizon = [0.8919193026076779, 0.8919193026076779]`

즉 현재 옵션(`blend_floor=0`, `blend_max=1`, `use_uncertainty_gate=false`)에서는 실제로 similarity 값이 그대로 blend weight가 된다.

코드 대응: `_blend_event_memory_prediction(...)`

---

## final prediction

최종 예측은 base prediction과 memory prediction의 convex combination이다.

$$
\hat y_h^{(\mathrm{final})} = (1-\lambda_h)\hat y_h^{(\mathrm{base})} + \lambda_h \hat y_h^{(\mathrm{mem})}
$$

그리고 이 값이 코드에서:

```python
target_predictions[job.model] = final_prediction
```

으로 들어간다.

즉 forecast CSV의 `y_hat`는 applied 케이스에서 `final_prediction`과 동일하다.

코드 대응:
- `_blend_event_memory_prediction(...)`
- `predict_aa_forecast_fold(...)`

---

## 옵션별 영향 표

| 옵션 | 코드 소비 위치 | 수식/조건 역할 | 현재 값의 실제 의미 |
|---|---|---|---|
| `enabled` | `_normalize_retrieval_config`, `predict_aa_forecast_fold` | retrieval 경로 on/off | retrieval path 자체 활성화 |
| `top_k` | `_retrieve_event_neighbors` | 상위 후보 개수 제한 | 최종 neighbor 1개만 사용 |
| `event_score_threshold` | `_build_event_memory_bank`, `_retrieve_event_neighbors` | query/candidate event score 필터 | 강한 event 신호가 있어야 retrieval 참여 |
| `min_similarity` | `_retrieve_event_neighbors` | low-similarity candidate 제거 | 0.35 미만 candidate 제거 |
| `blend_max` | `_blend_event_memory_prediction` | blend weight 상한 | memory prediction이 크게 반영될 수 있음 |
| `use_uncertainty_gate` | `_blend_event_memory_prediction` | uncertainty scale 사용 여부 | std가 blend weight를 조정하지 않음 |
| `use_shape_key` | `_retrieve_event_neighbors` | shape cosine 사용 여부 | target shape cosine은 제외 |
| `use_event_key` | `_retrieve_event_neighbors` | event cosine 사용 여부 | STAR event signature가 핵심 |
| `event_score_log_bonus_alpha` | `_retrieve_event_neighbors` | bonus 크기 | 더 강한 과거 이벤트에 bonus 부여 |
| `event_score_log_bonus_cap` | `_retrieve_event_neighbors` | bonus 상한 | 과도한 보너스 방지 |

---

## worked example 1 — retrieval applied

실제 artifact:

- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/aa_forecast/retrieval/20260302T000000.json`

확인된 값:

- `retrieval_applied = true`
- `top_k_used = 1`
- `blend_weight_by_horizon = [0.8919193026076779, 0.8919193026076779]`
- `base_prediction[0] = 87.18645239056478`
- `memory_prediction[0] = 89.56394497280102`
- `final_prediction[0] = 89.30698391646786`

그리고 같은 run의 forecast CSV:

- `cv/AAForecast_forecasts.csv`
- `y_hat = 89.30698391646786`

같은 row의 step 2도 함께 보면:

- `base_prediction[1] = 85.73390449266961`
- `memory_prediction[1] = 93.08095028043454`
- `final_prediction[1] = 92.28687644791958`

즉 이 예시는 step 1뿐 아니라 step 2에서도 retrieval memory path가 base path보다 더 공격적인 상승 경로를 제시했고, 큰 blend weight 때문에 최종값이 memory 쪽으로 상당히 이동한 케이스다.

즉 이 케이스에서:

1. backbone + uncertainty 단계까지의 `base_prediction`은 `87.1864...`
2. retrieval이 적용됨
3. top-1 neighbor의 future return이 현재 마지막 level에 재적용돼 `memory_prediction`이 만들어짐
4. blend weight `0.8919...`가 적용됨
5. 최종 `final_prediction[0] = 89.30698...`
6. 이 값이 CSV `y_hat`로 기록됨

정리하면:

$$
\text{forecast CSV y\_hat} = \text{retrieval JSON final\_prediction}
$$

---

## worked example 2 — retrieval skipped

실제 artifact:

- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/aa_forecast/retrieval/20260223T000000.json`

확인된 값:

- `retrieval_applied = false`
- `skip_reason = below_event_threshold`

의미:

- 현재 query event score가 `event_score_threshold=400.0`을 넘지 못해 retrieval이 skip된 케이스다.
- 이 경우 후보 검색/softmax/blend가 실질적으로 적용되지 않는다.
- 따라서 `final_prediction == base_prediction`이다.

실제 skip artifact 값은:

- `query_event_score = 200.67826199531555`
- `base_prediction = [73.79478085233299, 74.6368685329311]`
- `memory_prediction = [73.79478085233299, 74.6368685329311]`
- `final_prediction = [73.79478085233299, 74.6368685329311]`

즉 skip 케이스에서는 retrieval summary에 `memory_prediction` 필드가 있더라도 실제로는 base와 동일하게 남아, forecast를 바꾸지 않는다.

---

## artifact 해석법

### 1. retrieval JSON

`aa_forecast/retrieval/<slug>.json`에서 우선 볼 필드:

- `retrieval_attempted`
- `retrieval_applied`
- `skip_reason`
- `top_k_requested`
- `top_k_used`
- `mean_similarity`
- `max_similarity`
- `base_prediction`
- `memory_prediction`
- `final_prediction`
- `blend_weight_by_horizon`

이 파일이 retrieval의 진짜 설명서다.

### 2. neighbors CSV

`aa_forecast/retrieval/<slug>.neighbors.csv`에서 볼 필드:

- `rank`
- `similarity`
- `shape_similarity`
- `event_similarity`
- `softmax_weight`
- `future_return_step_k`

이 파일은 "어떤 과거 이벤트를 참고했는가"를 보여준다.

### 3. forecast CSV

`cv/AAForecast_forecasts.csv`에서 retrieval 관련으로 볼 필드:

- `y_hat`
- `aaforecast_retrieval_enabled`
- `aaforecast_retrieval_applied`
- `aaforecast_retrieval_skip_reason`
- `aaforecast_retrieval_artifact`

이 row를 통해 forecast와 retrieval artifact를 연결할 수 있다.

---

## 주의점

### 1. retrieval enabled와 retrieval applied는 다르다

- `enabled=true`는 기능이 켜졌다는 뜻이다.
- `applied=true`는 그 fold/cutoff에서 실제로 retrieval이 prediction을 수정했다는 뜻이다.

### 2. retrieval on/off run 직접 비교 주의

retrieval on run과 no-retrieval run은 **cutoff window가 다를 수 있다**.

따라서:
- 동일 cutoff가 아닌 두 run의 `y_hat`를 row-by-row 바로 비교하면 안 된다.

비교 전 반드시:
- `train_end_ds`
- `ds`
- `horizon_step`

이 같은지 확인해야 한다.

### 3. skip reason 해석

주요 skip reason:

- `below_event_threshold` — query event score가 약함
- `empty_bank` — threshold를 넘긴 candidate bank가 없음
- `min_similarity` — bank는 있었지만 similarity를 만족한 이웃이 없음

---

## 디버깅 체크리스트

1. `config/config.resolved.json`에서 retrieval/uncertainty/backbone 확인
2. forecast row의 `aaforecast_retrieval_applied` 확인
3. `aaforecast_retrieval_artifact` JSON 열기
4. `base_prediction`과 `final_prediction` 비교
5. skip이면 `skip_reason` 확인
6. applied면 neighbors CSV에서 `top_k_used`, `softmax_weight`, `future_return_step_k` 확인
7. on/off 비교 시 cutoff가 같은지 먼저 확인

---

## 한 줄 요약

현재 설정에서 retrieval은 **shape key를 쓰지 않고 STAR event signature만으로 과거 event analogue를 찾은 뒤**, 그 미래 return 패턴을 현재 마지막 level에 재적용해 memory trajectory를 만들고, 이를 `base_prediction`과 blend해서 최종 `y_hat`를 만든다.
