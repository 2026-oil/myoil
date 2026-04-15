# AAForecast GRU retrieval 동작 정리

> [!NOTE]
> 이 문서는 `uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml` 실행 시  
> **지금 현재 워킹트리 기준으로 retrieval이 실제로 어떻게 계산되는지**를 설명한다.

> [!WARNING]
> 질문에 적힌 `yaml/plugins/aa_forecast/retreival/aa_forecast_parity_gru.yaml`는 이번 실행의 active file이 아니다.  
> 현재 실제 실행 체인은 아래다.
>
> `yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml`
> → `yaml/plugins/aa_forecast/aa_forecast_gru-ret.yaml`
> → `yaml/plugins/retrieval/baseline_retrieval.yaml`

---

## 1. 실제로 어떤 설정이 적용되나

메인 실행은 아래다.

```bash
uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml
```

이 메인 YAML은 AAForecast plugin file만 지정한다.

```yaml
aa_forecast:
  enabled: true
  config_path: yaml/plugins/aa_forecast/aa_forecast_gru-ret.yaml
```

그런데 현재 `aa_forecast_gru-ret.yaml` 안의 retrieval은 inline 값이 아니라, 다시 detail YAML을 가리킨다.

```yaml
retrieval:
  enabled: true
  config_path: ../retrieval/baseline_retrieval.yaml
```

즉 retrieval 설명은 **AAForecast plugin YAML만 보면 안 되고**, 최종적으로는 `baseline_retrieval.yaml`까지 merge된 값을 봐야 한다.

현재 fresh run manifest 기준 resolved retrieval 값은 아래다.

| 항목 | 값 |
| --- | --- |
| task.name | `aaforecast_gru-ret` |
| target | `Com_BrentCrudeOil` |
| input_size | `64` |
| horizon | `2` |
| backbone | `gru` |
| top_k | `5` |
| recency_gap_steps | `8` |
| event_score_threshold | `1.0` |
| min_similarity | `0.55` |
| blend_floor | `0.0` |
| blend_max | `0.25` |
| use_uncertainty_gate | `true` |
| use_shape_key | `true` |
| use_event_key | `true` |
| temperature | `0.1` |
| event_score_log_bonus_alpha | `0.0` |
| event_score_log_bonus_cap | `0.0` |
| STAR exog | `GPRD_THREAT`, `BS_Core_Index_A`, `BS_Core_Index_C` |
| non-STAR exog | `GPRD`, `GPRD_ACT`, `BS_Core_Index_B`, `Idx_OVX`, `Com_LMEX`, `Com_BloombergCommodity_BCOM` |

> [!TIP]
> 이전 설명처럼 `top_k=1`, `blend_max=1.0`, `use_shape_key=false`가 아니다.  
> 현재 설정은 **top-k 5개**, **shape+event 둘 다 사용**, **uncertainty-gated blend**, **blend cap 0.25**다.

---

## 2. retrieval이 전체 파이프라인에서 어디에 들어가나

현재 흐름은 아래 순서다.

1. GRU AAForecast가 기본 예측 `base_prediction`을 만든다.
2. uncertainty가 켜져 있으므로 dropout 후보들에서 mean/std를 계산한다.
3. retrieval이 켜져 있으므로:
   - 과거 학습 구간에서 memory bank를 만든다.
   - 마지막 64-step window를 query로 만든다.
   - query와 bank의 similarity를 계산해 top-5 neighbor를 고른다.
   - top-5 neighbor의 미래 수익률을 softmax 평균해 `memory_prediction`을 만든다.
   - uncertainty std까지 반영한 horizon별 blend weight로 `base_prediction`과 `memory_prediction`을 섞는다.
4. 그 결과가 최종 `final_prediction`이다.

즉 retrieval은 별도 모델이 아니라,
**기본 예측 뒤에 붙는 posthoc memory blend**다.

---

## 3. 코드 기준 계산식

## 3-1. retrieval signature 만들기

각 window마다 retrieval은 아래 3개를 만든다.

- `shape_vector`
- `event_vector`
- `event_score`

핵심 개념은 아래다.

```text
shape_vector = normalize(해당 window의 target 시계열)

event_vector = normalize(
    critical_mask
    ++ count_active_channels
    ++ channel_activity(flatten)
    ++ activity_sums
    ++ activity_max
)

event_score = sum(count_active_channels) + sum(abs(channel_activity))
```

즉:

- `shape_vector` = target 패턴 자체
- `event_vector` = STAR가 잡은 이벤트 구조
- `event_score` = 그 window가 얼마나 eventful한지

---

## 3-2. memory bank 만들기

과거 학습 구간에서 길이 `input_size=64`짜리 window를 밀면서 candidate를 만든다.

candidate 하나는 대략 아래다.

```text
candidate = {
  shape_vector,
  event_vector,
  event_score,
  anchor_target_value,
  future_returns,
}
```

여기서

```text
future_returns = (future_values - anchor_value) / max(abs(anchor_value), 1e-8)
```

이다.

즉 retrieval은 과거 absolute price를 복사하지 않고,
**candidate 끝점(anchor) 이후의 상대 수익률 경로**를 저장한다.

현재 설정은 `event_score_threshold=1.0`라서 threshold는 매우 낮다.
그래서 실제 artifact를 보면 candidate 대부분이 bank에 남는다.

예: `2025-12-01` cutoff

- `candidate_count = 497`
- `eligible_candidate_count = 497`

---

## 3-3. query 만들기

현재 cutoff 직전의 마지막 64-step window를 query로 만든다.

```text
query = {
  shape_vector,
  event_vector,
  event_score,
}
```

현재 설정은 threshold가 `1.0`라서,
query `event_score`가 아주 작지 않은 한 거의 skip되지 않는다.

현재 fresh artifact 기준으로는 4개 cutoff 모두 retrieval이 실제 적용되었다.

---

## 3-4. similarity 계산

현재 설정은 `use_shape_key=true`, `use_event_key=true`, `log bonus=0.0`다.

따라서 neighbor 하나의 similarity는 아래처럼 계산된다.

```text
shape_similarity = cosine(query.shape_vector, candidate.shape_vector)
event_similarity = cosine(query.event_vector, candidate.event_vector)

similarity = 0.20 * shape_similarity + 0.80 * event_similarity
```

여기서 0.20 / 0.80 가중치는 `plugins/retrieval/runtime.py`의 상수다.

즉 현재 retrieval은

- shape도 보고
- event도 보지만
- event를 더 강하게 본다

라고 이해하면 된다.

그 다음

```text
if similarity < 0.55:
    탈락
```

이고, 통과한 후보들 중 상위 5개를 가져온다.

---

## 3-5. softmax weight와 memory prediction

선택된 top-k neighbor들에 대해 softmax weight를 붙인다.

```text
logit_i = similarity_i / temperature
weight_i = softmax(logit_i)
```

현재는 `temperature=0.1`이므로,
상위 이웃에 weight가 더 몰리지만 top-5 전체가 반영된다.

그 다음 horizon별 weighted return을 만든다.

```text
weighted_returns = Σ weight_i * future_returns_i
```

그리고 현재 cutoff의 마지막 실제 타깃값 `current_last_y`에 이 return path를 입힌다.

```text
scale = max(abs(current_last_y), 1e-8)
memory_prediction = current_last_y + scale * weighted_returns
```

즉 retrieval은
**과거 top-5 유사 이벤트들의 미래 상대수익률 평균을, 현재 cutoff 가격 레벨로 옮겨온 것**이다.

---

## 3-6. uncertainty gate가 들어간 horizon별 blend

현재 설정은 `use_uncertainty_gate=true` 이다.

그래서 blend weight는 scalar 하나가 아니라 **horizon별**로 계산된다.

```text
similarity_scale = clip(mean_similarity, 0, 1)
uncertainty_scale_h = std_h / max(std over horizons)

blend_weight_h = blend_floor
               + (blend_max - blend_floor)
               * similarity_scale
               * uncertainty_scale_h

blend_weight_h = clip(blend_weight_h, blend_floor, blend_max)
```

현재 설정값을 넣으면:

```text
blend_floor = 0.0
blend_max = 0.25
```

즉 사실상

```text
blend_weight_h = 0.25 * mean_similarity * uncertainty_scale_h
```

이다.

중요한 점:

- similarity가 높을수록 retrieval 비중이 커진다.
- uncertainty std가 큰 horizon일수록 retrieval 비중이 커진다.
- 그래도 최대 0.25까지만 반영된다.

즉 현재 retrieval은 **memory를 참고하되, base 예측을 완전히 덮어쓰지 않도록 제한**되어 있다.

최종 예측은 아래다.

```text
final_prediction_h
= (1 - blend_weight_h) * base_prediction_h
+ blend_weight_h * memory_prediction_h
```

---

## 4. toy sample 손계산

아래 toy sample은 **현재 설정**인

- `top_k=5`
- `use_shape_key=true`
- `use_event_key=true`
- `blend_max=0.25`
- `use_uncertainty_gate=true`

를 반영한 단순 예시다.

### 4-1. 가정

query와 후보 2개만 있다고 단순화하자.

```text
base_prediction = [100, 104]
current_last_y = 100
mean_similarity는 top-k 평균으로 0.80이라 가정
std_by_horizon = [2, 4]
max_std = 4
uncertainty_scale = [0.5, 1.0]
```

후보 2개가 아래라고 하자.

| 후보 | shape_similarity | event_similarity | similarity | future_returns |
| --- | ---: | ---: | ---: | --- |
| A | 0.90 | 0.80 | `0.2*0.90 + 0.8*0.80 = 0.82` | `[0.03, 0.01]` |
| B | 0.70 | 0.85 | `0.2*0.70 + 0.8*0.85 = 0.82` | `[0.00, -0.02]` |

둘 다 similarity가 같다고 하자.
그러면 softmax weight는 거의 `0.5`, `0.5`다.

---

### 4-2. weighted return 계산

```text
weighted_returns_h1 = 0.5*0.03 + 0.5*0.00 = 0.015
weighted_returns_h2 = 0.5*0.01 + 0.5*(-0.02) = -0.005
```

그래서 memory prediction은

```text
memory_h1 = 100 + 100*0.015 = 101.5
memory_h2 = 100 + 100*(-0.005) = 99.5
```

즉

```text
memory_prediction = [101.5, 99.5]
```

---

### 4-3. horizon별 blend weight 계산

현재 설정에서

```text
blend_weight_h = 0.25 * mean_similarity * uncertainty_scale_h
```

이므로

```text
blend_weight_h1 = 0.25 * 0.80 * 0.50 = 0.10
blend_weight_h2 = 0.25 * 0.80 * 1.00 = 0.20
```

즉 uncertainty가 더 큰 horizon 2가 retrieval을 더 많이 반영한다.

---

### 4-4. final prediction 계산

```text
final_h1 = (1 - 0.10)*100 + 0.10*101.5
         = 90 + 10.15
         = 100.15

final_h2 = (1 - 0.20)*104 + 0.20*99.5
         = 83.2 + 19.9
         = 103.1
```

따라서

```text
final_prediction = [100.15, 103.1]
```

이다.

요점은 현재 retrieval이

- top-k 여러 개를 평균하고
- uncertainty가 큰 horizon에 더 많이 반영되며
- 그래도 cap 0.25 때문에 과격하게 덮어쓰지 않는다는 것

이다.

---

## 5. 실제 run artifact로 본 worked example (`2025-12-01`)

artifact:

- `runs/feature_set_aaforecast_aaforecast_gru-ret/aa_forecast/retrieval/20251201T000000.json`
- `runs/feature_set_aaforecast_aaforecast_gru-ret/aa_forecast/retrieval/20251201T000000.neighbors.csv`
- `runs/feature_set_aaforecast_aaforecast_gru-ret/aa_forecast/uncertainty/20251201T000000.json`

핵심 값:

| 항목 | 값 |
| --- | ---: |
| cutoff | `2025-12-01` |
| current_last_y | `63.277029` |
| query_event_score | `419.330600...` |
| top_k_used | `5` |
| mean_similarity | `0.8665137648...` |
| max_similarity | `0.8769667543...` |
| blend_weight_by_horizon | `[0.1877383789, 0.2166284412]` |
| selected_std_by_horizon | `[0.3690020584, 0.4257858259]` |

상위 5개 neighbor의 softmax 가중합 결과는:

```text
weighted_returns ≈ [0.0000889247, 0.0041405883]
```

따라서 memory prediction은:

```text
memory_h1 = 63.277029 + 63.277029 * 0.0000889247
          ≈ 63.282656

memory_h2 = 63.277029 + 63.277029 * 0.0041405883
          ≈ 63.539033
```

artifact의 값과 맞는다.

```text
memory_prediction = [63.282655..., 63.539032...]
```

base prediction은:

```text
base_prediction = [62.49257278, 66.61139862]
```

이제 horizon별 blend를 적용하면:

```text
final_h1
= (1 - 0.1877383789) * 62.49257278
+ 0.1877383789 * 63.28265589
≈ 62.640902

final_h2
= (1 - 0.2166284412) * 66.61139862
+ 0.2166284412 * 63.53903313
≈ 65.945837
```

artifact의 최종값과 맞는다.

```text
final_prediction = [62.640901..., 65.945836...]
```

또한 uncertainty gate도 눈으로 확인할 수 있다.

- horizon 2의 std가 더 큼
- 그래서 horizon 2의 blend weight가 더 큼

실제로:

```text
uncertainty_scale ≈ [0.8666, 1.0]
blend_weight ≈ 0.25 * 0.8665137648 * uncertainty_scale
```

가 artifact의 `blend_weight_by_horizon`와 맞는다.

---

## 6. 실제 run artifact로 본 두 번째 example (`2025-12-29`)

artifact:

- `runs/feature_set_aaforecast_aaforecast_gru-ret/aa_forecast/retrieval/20251229T000000.json`

핵심 값:

| 항목 | 값 |
| --- | ---: |
| cutoff | `2025-12-29` |
| current_last_y | `60.965500` |
| query_event_score | `362.320683...` |
| threshold | `1.0` |
| retrieval_applied | `true` |
| top_k_used | `5` |
| mean_similarity | `0.8441731638...` |
| blend_weight_by_horizon | `[0.1938392329, 0.2110432909]` |

weighted return은:

```text
weighted_returns ≈ [0.0214506571, 0.0185027088]
```

그래서 memory prediction은:

```text
memory_prediction ≈ [62.273250, 62.093527]
```

base prediction은:

```text
base_prediction = [57.11238632, 57.12405266]
```

최종 prediction은:

```text
final_prediction ≈ [58.112764, 58.172827]
```

즉 `2025-12-29`도 retrieval applied 케이스다.

> [!NOTE]
> fresh artifact 기준으로는  
> `2025-12-01`, `2025-12-29`, `2026-01-26`, `2026-02-23` 네 cutoff 모두 retrieval이 적용되었다.

---

## 7. 한 줄 요약

> [!NOTE]
> 현재 `aaforecast-gru-ret` retrieval은  
> **최근 64-step window와 비슷한 과거 eventful window 5개를 shape+event similarity로 찾고,  
> 그들의 미래 상대수익률 경로를 softmax 평균해 현재 가격 레벨에 옮긴 뒤,  
> uncertainty가 큰 horizon일수록 조금 더 많이(base의 최대 25%까지) 섞는 posthoc 보정**이다.

---

## 8. 확인에 사용한 근거

- command
  - `uv run python main.py --validate-only --config yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml`
- config chain
  - `yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml`
  - `yaml/plugins/aa_forecast/aa_forecast_gru-ret.yaml`
  - `yaml/plugins/retrieval/baseline_retrieval.yaml`
- resolved/runtime artifacts
  - `runs/feature_set_aaforecast_aaforecast_gru-ret/config/config.resolved.json`
  - `runs/feature_set_aaforecast_aaforecast_gru-ret/manifest/run_manifest.json`
  - `runs/feature_set_aaforecast_aaforecast_gru-ret/summary/result.csv`
- retrieval artifacts
  - `runs/feature_set_aaforecast_aaforecast_gru-ret/aa_forecast/retrieval/20251201T000000.json`
  - `runs/feature_set_aaforecast_aaforecast_gru-ret/aa_forecast/retrieval/20251229T000000.json`
- uncertainty artifact
  - `runs/feature_set_aaforecast_aaforecast_gru-ret/aa_forecast/uncertainty/20251201T000000.json`
- code
  - `plugins/aa_forecast/runtime.py`
  - `plugins/retrieval/runtime.py`
  - `plugins/aa_forecast/config.py`
