# Retrieval 손계산 완전 해설

이 페이지는 Retrieval이 적용되는 방식을 **코드 한 줄 수준**까지 따라가며 손으로 검증할 수 있도록 작성되었습니다.  
기존 variant 페이지들이 "블렌드 결과만 손계산"하는 것과 달리, 여기서는 **STAR signature 빌드 → 메모리 뱅크 → 쿼리 → 이웃 검색 → 소프트맥스 → 메모리 예측 → 블렌드**를 한 흐름으로 처음부터 끝까지 전개합니다.

---

## 0. 이 문서에서 사용하는 config

아래 값은 `yaml/plugins/retrieval/baseline_retrieval.yaml` 현재 기본값입니다.

| 파라미터 | 값 | 의미 |
|---|---|---|
| `top_k` | `1` | 검색할 최상위 이웃 수 |
| `recency_gap_steps` | `8` | bank 끝점에서 제외할 최근 스텝 수 |
| `event_score_threshold` | `400.0` | bank candidate 수락 최소 이벤트 점수 |
| `min_similarity` | `0.7` | 이웃 최소 유사도 |
| `use_shape_key` | `false` | shape 유사도 사용 여부 |
| `use_event_key` | `true` | event 유사도 사용 여부 |
| `event_score_log_bonus_alpha` | `0.15` | log bonus 가중치 |
| `event_score_log_bonus_cap` | `0.1` | log bonus 상한 |
| `temperature` | `0.1` | softmax temperature |
| `blend_floor` | `0.0` | blend weight 하한 |
| `blend_max` | `1.0` | blend weight 상한 |
| `use_uncertainty_gate` | `true` | horizon별 uncertainty로 blend 조정 여부 |

> [!NOTE]
> Provenance: `repo default`
>
> AA-Forecast retrieval config (`aaforecast-gru-ret.yaml` → `baseline_retrieval.yaml`)에서도 동일한 detail YAML을 사용합니다. 다만 standalone retrieval에는 uncertainty_std가 없어서 `use_uncertainty_gate: true`여도 `uncertainty_scale=1.0`으로 처리됩니다.

---

## 1. 전체 파이프라인 요약

```
학습 구간 (transformed_train_df, raw_train_df)
        │
        ├── [Stage 1] STAR signature ─────────────────────────────────┐
        │    shape_vector = normalize(target_raw_values)              │
        │    event_vector = normalize(critical_mask ++ count_active   │
        │                   ++ channel_activity ++ sums ++ max)       │
        │    event_score  = sum(count_active) + sum|channel_activity| │
        │                                                             │
        ├── [Stage 2] memory bank 빌드 ──────────────────────────────┘
        │    end_idx = [input_size−1 ... last_idx − horizon − recency_gap_steps]
        │    각 window마다 signature 계산 → event_score ≥ threshold만 보관
        │    future_returns = (future_values − anchor) / max(|anchor|, ε)
        │
        ├── [Stage 3] query 빌드
        │    query window = transformed_train_df 마지막 input_size 행
        │    → query.shape_vector, query.event_vector, query.event_score
        │
        ├── [Stage 4] neighbor 검색
        │    query.event_score < threshold → skip
        │    각 candidate마다:
        │      shape_sim = cosine(query.shape_vector, cand.shape_vector)
        │      event_sim = cosine(query.event_vector, cand.event_vector)
        │      log_bonus = α · min(max(log(cand_score/query_score), 0), cap)
        │      event_component = event_sim + log_bonus   (log_bonus는 α>0일 때만)
        │      similarity = 0.20·shape_sim + 0.80·event_component  (둘 다 켜졌을 때)
        │               혹은 event_component                        (event only)
        │               혹은 shape_sim                              (shape only)
        │      similarity < min_similarity → 탈락
        │    상위 top_k 선택, softmax weight 계산
        │
        ├── [Stage 5] memory prediction 계산
        │    weighted_returns = Σ weight_i · future_returns_i
        │    scale = max(|current_last_y|, ε)
        │    memory_prediction = current_last_y + scale · weighted_returns
        │
        └── [Stage 6] blend → final prediction
             similarity_scale = clip(mean_similarity, 0, 1)
             uncertainty_scale_h = std_h / max(std)   (use_uncertainty_gate=true)
                                 = 1.0                (그 외)
             blend_weight_h = clip(
                 blend_floor + (blend_max − blend_floor)
                              · similarity_scale · uncertainty_scale_h,
                 blend_floor, blend_max)
             final_h = (1 − λ_h)·base_h + λ_h·memory_h
```

---

## 2. Stage 1 — STAR signature 상세

`plugins/retrieval/signatures.py :: compute_star_signature` 기준.

### 2-1. 입력

| 변수 | 내용 |
|---|---|
| `window_df` | 길이 `L`인 DataFrame (target + hist_exog 컬럼 포함) |
| `target_col` | 예: `Com_BrentCrudeOil` |
| `hist_exog_cols` | 예: `(GPRD_THREAT, BS_Core_Index_A, ...)` |
| `hist_exog_tail_modes` | 각 hist_exog 채널의 꼬리 방향 (`upward` 또는 `two_sided`) |

### 2-2. STAR 실행

**target**:

```python
insample_y = torch.tensor(target_values).reshape(1, L, 1)
target_star = star(insample_y, tail_modes=("two_sided",))
# target_star["critical_mask"]  shape: (1, L, 1)   0/1
# target_star["ranking_score"]  shape: (1, L, 1)   z-score 류 잔차 크기
```

**hist_exog**:

```python
hist_tensor = torch.tensor(hist_values).unsqueeze(0)   # shape: (1, L, C)
hist_star = star(hist_tensor, tail_modes=hist_exog_tail_modes)
# hist_star["critical_mask"]  shape: (1, L, C)
# hist_star["ranking_score"]  shape: (1, L, C)
```

### 2-3. activity 합산

```
target_activity   (L, 1) = ranking_score * critical_mask
hist_activity     (L, C) = ranking_score * critical_mask

channel_activity  (L, 1+C) = concat([target_activity, hist_activity], axis=1)
combined_count    (L, 1) = target_critical_count + hist_critical_count
```

여기서 `target_critical_count = critical_mask.sum(axis=2)` 는 target 채널이 1개라 시점마다 0 또는 1, `hist_critical_count`는 hist_exog 채널 중 임계치를 넘은 채널 수다.

### 2-4. event_vector 조립

```
critical_mask_flat  (L,)      = (combined_count > 0).astype(float)
count_active_flat   (L,)      = combined_count.reshape(-1)
channel_act_flat    (L*(1+C),)= channel_activity.reshape(-1)
activity_sums       (1+C,)    = channel_activity.sum(axis=0)
activity_max        (1+C,)    = channel_activity.max(axis=0)

event_vector_raw = concat([
    critical_mask_flat,   # L
    count_active_flat,    # L
    channel_act_flat,     # L*(1+C)
    activity_sums,        # 1+C
    activity_max,         # 1+C
])
```

채널 수가 `1 + C`일 때 `event_vector_raw`의 길이:

\[
\text{dim}_{event} = L + L + L(1+C) + (1+C) + (1+C) = 2L + L(1+C) + 2(1+C)
\]

toy 기준 `L=4`, C=1 (GPRD_THREAT 1개만 있다고 가정):

\[
\text{dim}_{event} = 2 \times 4 + 4 \times 2 + 2 \times 2 = 8 + 8 + 4 = 20
\]

### 2-5. shape_vector 조립

```
shape_vector_raw = window_df[target_col].to_numpy()   # 길이 L
```

단순히 target 원시값 그대로다. z-score가 아니다.

### 2-6. L2 정규화

```python
def _normalize_signature(values):
    vector = values.reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm
```

두 벡터 모두 L2 정규화 후 저장된다. 따라서 코사인 유사도 계산 시 분모가 항상 1.0이 된다.

### 2-7. event_score 계산

\[
event\_score = \sum_t count\_active_t + \sum_t \sum_c |channel\_activity_{t,c}|
\]

코드:
```python
event_score = float(count_active.sum() + np.abs(channel_activity).sum())
```

이 점수가 `event_score_threshold`보다 크지 않으면 해당 window는 bank에 포함되지 않는다.

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 repo default `event_score_threshold = 400.0` 은 하루 단위 weekly 데이터 기준으로 세팅되어 있다. toy에서는 threshold를 1.0으로 낮춰 모든 candidate가 통과한다고 가정한다.

---

## 3. Stage 2 — memory bank 빌드 상세

`plugins/retrieval/runtime.py :: _build_memory_bank` 기준.

### 3-1. 유효 end_idx 범위

```python
last_idx = len(raw_train_df) - 1
max_end_idx = last_idx - horizon - recency_gap_steps
```

즉 bank에 들어올 수 있는 최대 anchor 위치는 `last_idx - H - gap` 이다.

\[
\text{유효 } end\_idx \in [L-1,\; last\_idx - H - gap]
\]

toy 기준: `len = 10`, `last_idx = 9`, `H = 2`, `gap = 0` (toy simplification):

\[
max\_end\_idx = 9 - 2 - 0 = 7
\]

\[
\text{유효 } end\_idx \in [3, 7]
\]

→ 후보 anchor 위치: 3, 4, 5, 6, 7 총 5개.

(실제 repo default `recency_gap_steps=8`라면 최근 8스텝은 bank에서 제외된다.)

### 3-2. 각 candidate 저장 내용

```python
bank.append({
    "candidate_end_ds"      : 해당 anchor 날짜,
    "candidate_future_end_ds": anchor + horizon 날짜,
    "shape_vector"          : normalize(raw_target_in_window),
    "event_vector"          : normalize(event_vector_raw),
    "event_score"           : event_score,
    "anchor_target_value"   : raw_train_df[target_col].iloc[end_idx],
    "future_returns"        : (future_values - anchor) / max(|anchor|, 1e-8),
})
```

`future_returns`는 절댓값이 아닌 **상대 수익률 경로**임에 주의.

\[
r_h^{(i)} = \frac{y^{(i)}_{future,h} - a^{(i)}}{\max(|a^{(i)}|,\; \varepsilon)}
\]

---

## 4. Stage 3 — query 빌드

`plugins/retrieval/runtime.py :: _build_query` 기준.

query는 `transformed_train_df` 마지막 `input_size`개 행으로 window를 구성하고, 동일하게 `compute_star_signature`를 호출한다.

```python
window = transformed_train_df.iloc[-input_size:].reset_index(drop=True)
query = compute_star_signature(star, window, ...)
```

toy 기준 query window = 인덱스 6~9 = `[107, 110, 121, 132]`.

---

## 5. Stage 4 — neighbor 검색 상세

`plugins/retrieval/runtime.py :: _retrieve_neighbors` 기준.

### 5-1. query event_score threshold 체크

```python
if query["event_score"] < threshold:
    → retrieval_applied = False, skip_reason = "below_event_threshold"
```

query 자체가 이벤트가 없는 조용한 구간이라면 retrieval이 아예 실행되지 않는다.

### 5-2. candidate별 유사도 계산

각 candidate에 대해:

```python
shape_similarity = cosine(query.shape_vector, cand.shape_vector)
event_similarity = cosine(query.event_vector, cand.event_vector)
```

코사인 유사도 공식 (`_cosine_similarity`):

\[
sim(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|}
\]

두 벡터가 이미 L2-normalized라면:

\[
sim(u, v) = u \cdot v
\]

### 5-3. event_score log bonus (alpha > 0 일 때만)

```python
if use_event_key and event_score_log_bonus_alpha > 0.0:
    log_bonus = min(max(log(cand_event_score / query_event_score), 0.0), cap)
    event_component = event_similarity + alpha * log_bonus
else:
    event_component = event_similarity
```

이 bonus는 **candidate의 이벤트 강도가 query보다 클 때** event 유사도를 약간 올려준다. `baseline_retrieval.yaml` 기준:

- `event_score_log_bonus_alpha = 0.15`
- `event_score_log_bonus_cap = 0.1`

따라서:

\[
log\_bonus = \min\!\left(\max\!\left(\ln\!\frac{s_{cand}}{s_{query}},\; 0\right),\; 0.1\right)
\]

\[
event\_component = event\_similarity + 0.15 \times log\_bonus
\]

`log_bonus`가 0.1까지밖에 커지지 않으므로 event_component는 최대 `event_similarity + 0.015`다.

### 5-4. 최종 similarity 합산

```python
if use_shape_key and use_event_key:
    similarity = 0.20 * shape_similarity + 0.80 * event_component
elif use_event_key:
    similarity = event_component
elif use_shape_key:
    similarity = shape_similarity
```

`baseline_retrieval.yaml`은 `use_shape_key=false`, `use_event_key=true`이므로:

\[
similarity = event\_component
\]

### 5-5. min_similarity 필터

```python
if similarity < min_similarity:
    continue   # 탈락
```

통과한 후보들만 `scored_neighbors`에 쌓인다. 이후 내림차순 정렬.

### 5-6. top-k 선택 및 softmax weight

```python
top_neighbors = scored_neighbors[:top_k]

logits = [n["similarity"] for n in top_neighbors] / temperature
logits = logits - logits.max()   # numerical stability
weights = exp(logits)
weights = weights / weights.sum()
```

즉:

\[
\ell_i = \frac{sim_i}{T}
\]

\[
w_i = \frac{\exp(\ell_i - \max_j \ell_j)}{\sum_j \exp(\ell_j - \max_j \ell_j)}
\]

`top_k=1` 일 때 logit이 하나뿐이므로 `logit - max = 0`, `exp(0) = 1`, `w = 1.0`.

`top_k > 1`이고 `temperature=0.1`이면 similarity 차이가 조금만 나도 상위 neighbor에 weight가 강하게 집중된다.

예: `sim = [0.90, 0.80]`, `T=0.1`

\[
\ell = [9.0, 8.0], \quad \ell - 9.0 = [0, -1]
\]

\[
w = \frac{[e^0, e^{-1}]}{e^0 + e^{-1}} = \frac{[1, 0.368]}{1.368} \approx [0.731, 0.269]
\]

---

## 6. Stage 5 — memory prediction 계산

```python
weighted_returns = sum(weight_i * future_returns_i for each neighbor)
scale = max(abs(current_last_y), 1e-8)
memory_prediction = current_last_y + scale * weighted_returns
```

\[
\bar{r}_h = \sum_i w_i r_h^{(i)}
\]

\[
\hat{y}_h^{mem} = y_T + |y_T| \cdot \bar{r}_h
\]

여기서 `current_last_y = y_T`는 **raw train_df 마지막 타깃값**이다. transformed 값이 아님에 주의.

---

## 7. Stage 6 — blend weight 및 final prediction

`plugins/retrieval/runtime.py :: _blend_prediction` 기준.

### 7-1. similarity_scale

```python
similarity_scale = clip(mean_similarity, 0, 1)
```

`mean_similarity`는 top-k 이웃들의 similarity 평균:

\[
mean\_sim = \frac{1}{K} \sum_{i=1}^{K} sim_i
\]

`top_k=1`이면 `mean_similarity = top_neighbor.similarity`.

### 7-2. uncertainty_scale

```python
if use_uncertainty_gate and uncertainty_std is not None:
    max_std = max(std_values)
    uncertainty_scale = std_values / max_std  # shape: (H,)
else:
    uncertainty_scale = ones(H)
```

standalone retrieval (`post_predict_retrieval`)은 `uncertainty_std=None`을 넘기므로, `use_uncertainty_gate=true`여도 `uncertainty_scale=1.0`이 된다.

AA retrieval (`plugins/aa_forecast/runtime.py`)은 uncertainty 샘플에서 계산한 `std_by_horizon`을 넘겨서 horizon별로 다른 scale을 적용한다.

### 7-3. blend weight 계산

\[
\lambda_h = \text{clip}\!\left(\; blend\_floor + (blend\_max - blend\_floor) \cdot similarity\_scale \cdot uncertainty\_scale_h ,\; blend\_floor,\; blend\_max \right)
\]

`baseline_retrieval.yaml` (`blend_floor=0`, `blend_max=1`):

\[
\lambda_h = \text{clip}\!\left(\; mean\_sim \times uncertainty\_scale_h ,\; 0,\; 1 \right)
\]

uncertainty_scale이 없는 standalone case (`all 1.0`):

\[
\lambda_h = mean\_sim \quad (\text{단, } \leq 1)
\]

### 7-4. final prediction

\[
\hat{y}_h^{final} = (1 - \lambda_h)\,\hat{y}_h^{base} + \lambda_h\,\hat{y}_h^{mem}
\]

---

## 8. 완전 toy 손계산 (end-to-end)

공통 toy series (`toy simplification`):

\[
y = [100, 101, 102, 120, 132, 126, 107, 110, 121, 132]
\]

\[
GPRD\_THREAT = [10, 12, 13, 14, 15, 14, 12, 14, 30, 35]
\]

설정: `L=4`, `H=2`, `use_shape_key=false`, `use_event_key=true`, `top_k=1`, `event_score_threshold=1.0` (toy simplification), `min_similarity=0.7`, `temperature=0.1`, `blend_max=1.0`, `blend_floor=0.0`, `event_score_log_bonus_alpha=0.15`, `event_score_log_bonus_cap=0.1`.

### Step 0 — 데이터 배치

| 인덱스 | y | GPRD_THREAT |
|---|---|---|
| 0 | 100 | 10 |
| 1 | 101 | 12 |
| 2 | 102 | 13 |
| 3 | 120 | 14 |
| 4 | 132 | 15 |
| 5 | 126 | 14 |
| 6 | 107 | 12 |
| 7 | 110 | 14 |
| 8 | 121 | 30 |
| 9 | 132 | 35 |

query window = 인덱스 6~9, y_T = 132.

### Step 1 — query shape_vector

raw target values for query window:

\[
v = [107, 110, 121, 132]
\]

\[
\|v\| = \sqrt{107^2 + 110^2 + 121^2 + 132^2} = \sqrt{11449 + 12100 + 14641 + 17424} = \sqrt{55614} \approx 235.8
\]

\[
shape\_vector^{query} \approx \left[\frac{107}{235.8},\; \frac{110}{235.8},\; \frac{121}{235.8},\; \frac{132}{235.8}\right] \approx [0.4537,\; 0.4665,\; 0.5132,\; 0.5598]
\]

### Step 2 — query event_vector (teaching placeholder)

STAR가 query window에서 GPRD_THREAT 이상치를 탐지한다고 가정:

- 인덱스 8 (query 내 local index 2): GPRD_THREAT = 30 → trend ≈ 16 → residual ≈ 14 → critical
- 인덱스 9 (local index 3): GPRD_THREAT = 35 → trend ≈ 18 → residual ≈ 17 → critical
- target: 인덱스 9: y=132 → trend ≈ 116 → residual ≈ 16 → critical (thresh=3.5 기준)

toy에서 STAR output을 다음과 같이 단순화:

```
target_critical_mask  (L=4, 1-채널): [[0], [0], [0], [1]]
target_ranking_score  (L=4, 1-채널): [[0], [0], [0], [1.60]]
target_activity                     : [[0], [0], [0], [1.60]]

hist_critical_mask    (L=4, 1-채널, GPRD_THREAT만): [[0], [0], [1], [1]]
hist_ranking_score    (L=4, 1-채널): [[0], [0], [1.40], [1.70]]
hist_activity                       : [[0], [0], [1.40], [1.70]]
```

combined_count (= target_count + hist_count):

\[
combined\_count = [[0], [0], [1], [2]]
\]

channel_activity (L=4, 2-채널 = target + GPRD_THREAT):

\[
channel\_activity = \begin{bmatrix}0 & 0 \\ 0 & 0 \\ 0 & 1.40 \\ 1.60 & 1.70\end{bmatrix}
\]

이제 event_vector_raw 조립:

```
critical_mask_flat  = [0, 0, 1, 1]              (L=4)
count_active_flat   = [0, 0, 1, 2]              (L=4)
channel_act_flat    = [0, 0, 0, 0, 0, 0, 1.40, 1.60, 1.70]
                    (L*2 = 8, 행별로 target 먼저, hist 다음으로 flatten)
```

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 `channel_activity.reshape(-1)` 는 C-order로 flatten된다.  
> shape `(L, 1+C)` → flatten: `[t0_ch0, t0_ch1, t1_ch0, t1_ch1, ...]`.  
> toy에서는 편의상 직접 나열했다.

```
activity_sums = channel_activity.sum(axis=0) = [1.60, 3.10]   (2-채널)
activity_max  = channel_activity.max(axis=0) = [1.60, 1.70]   (2-채널)
```

concat 결과 (길이 = 4+4+8+2+2 = 20):

\[
event\_vector\_raw^{query} = [0,0,1,1,\; 0,0,1,2,\; 0,0,0,0,0,0,1.40,1.60,1.70,\ldots,\; 1.60,3.10,\; 1.60,1.70]
\]

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 STAR는 LOWESS 트렌드 제거 + 잔차 임계화 기반이므로 위 숫자는 동작 구조를 설명하는 teaching placeholder입니다.

L2 정규화 후 event_vector^query 를 얻는다.

event_score 계산:

\[
event\_score^{query} = \sum count\_active + \sum |channel\_activity| = (0+0+1+2) + (0+0+1.40+1.60+1.70) = 3 + 4.70 = 7.70
\]

`event_score_threshold=1.0`이므로 7.70 ≥ 1.0 → query 통과.

### Step 3 — candidate bank (anchor=3과 anchor=7)

\[
\text{유효 } end\_idx \in [3, 7]
\]

이 toy에서는 두 주요 후보만 사용:

**candidate A** (anchor=3):

- window: 인덱스 0~3 = [100, 101, 102, 120]
- anchor value: 120
- future values: y[4], y[5] = [132, 126]
- future returns:

\[
r_1^{(A)} = \frac{132 - 120}{120} = \frac{12}{120} = 0.10
\]

\[
r_2^{(A)} = \frac{126 - 120}{120} = \frac{6}{120} = 0.05
\]

GPRD_THREAT for window 0~3 = [10, 12, 13, 14] → 급등 없음 → event_score 낮다고 가정 (toy: ≈ 0.5 < 1.0 threshold → **탈락**).

**candidate B** (anchor=7):

- window: 인덱스 4~7 = [132, 126, 107, 110]
- anchor value: 110
- future values: y[8], y[9] = [121, 132]
- future returns:

\[
r_1^{(B)} = \frac{121 - 110}{110} = \frac{11}{110} = 0.10
\]

\[
r_2^{(B)} = \frac{132 - 110}{110} = \frac{22}{110} = 0.20
\]

GPRD_THREAT for window 4~7 = [15, 14, 12, 14] → 큰 스파이크 없음.

> [!NOTE]
> Provenance: `toy simplification`
>
> toy에서는 candidate B만 threshold를 통과한다고 단순화한다. event_score_B ≈ 2.5 (teaching placeholder).

### Step 4 — similarity 계산 (candidate B vs query)

`use_shape_key=false`, `use_event_key=true`이므로:

\[
similarity = event\_component
\]

`event_score_log_bonus_alpha = 0.15`:

\[
log\_bonus = \min\!\left(\max\!\left(\ln\frac{event\_score_B}{event\_score_{query}}, 0\right), 0.1\right)
\]

toy에서 `event_score_B ≈ 2.5`, `event_score_query ≈ 7.70`:

\[
\ln\frac{2.5}{7.70} = \ln(0.325) \approx -1.12 < 0 \quad \Rightarrow \quad log\_bonus = 0
\]

즉 candidate가 query보다 이벤트 강도가 낮을 때는 bonus = 0.

\[
event\_component = event\_similarity + 0.15 \times 0 = event\_similarity
\]

event_vector cosine similarity: query와 candidate B 모두 L2-normalized 벡터이므로 내적 = cosine similarity.

toy에서 두 벡터의 구조적 유사성을 반영해 `event_similarity ≈ 0.82` (teaching placeholder).

\[
similarity = 0.82 \geq min\_similarity(0.7) \quad \Rightarrow \quad \text{통과}
\]

### Step 5 — softmax weight

`top_k=1`, 후보 1개만 남으므로:

\[
w_B = 1.0
\]

### Step 6 — memory prediction

\[
\bar{r}_1 = 1.0 \times 0.10 = 0.10, \quad \bar{r}_2 = 1.0 \times 0.20 = 0.20
\]

\[
scale = \max(|y_T|, \varepsilon) = \max(132, 10^{-8}) = 132
\]

\[
\hat{y}_1^{mem} = 132 + 132 \times 0.10 = 132 + 13.2 = 145.2
\]

\[
\hat{y}_2^{mem} = 132 + 132 \times 0.20 = 132 + 26.4 = 158.4
\]

### Step 7 — blend weight

standalone retrieval이므로 `uncertainty_std=None`, `uncertainty_scale = [1.0, 1.0]`.

\[
mean\_similarity = 0.82
\]

\[
\lambda_h = \text{clip}(0.82 \times 1.0,\; 0,\; 1.0) = 0.82
\]

(두 horizon 모두 동일.)

### Step 8 — final prediction

base prediction (schematic placeholder):

\[
\hat{y}^{base} = [136, 138]
\]

\[
\hat{y}_1^{final} = (1 - 0.82) \times 136 + 0.82 \times 145.2 = 0.18 \times 136 + 0.82 \times 145.2 = 24.48 + 119.064 = 143.544
\]

\[
\hat{y}_2^{final} = (1 - 0.82) \times 138 + 0.82 \times 158.4 = 0.18 \times 138 + 0.82 \times 158.4 = 24.84 + 129.888 = 154.728
\]

\[
\hat{y}^{final} = [143.544,\; 154.728]
\]

요약표:

| 단계 | h=1 | h=2 |
|---|---|---|
| base prediction | 136 | 138 |
| future return (B) | 0.10 | 0.20 |
| memory prediction | 145.2 | 158.4 |
| blend weight λ | 0.82 | 0.82 |
| **final prediction** | **143.544** | **154.728** |

---

## 9. log bonus가 켜졌을 때 toy 변형

위 예에서 `event_score_B > event_score_query`가 되면 log bonus가 발생한다.

예: `event_score_B = 12.0`, `event_score_query = 7.70`:

\[
\ln\frac{12.0}{7.70} \approx 0.443
\]

\[
log\_bonus = \min(0.443,\; 0.1) = 0.1 \quad (\text{cap에 걸림})
\]

\[
event\_component = 0.82 + 0.15 \times 0.1 = 0.82 + 0.015 = 0.835
\]

즉 log bonus가 최대로 발생해도 event_component가 `0.015` 올라가는 효과다. similarity에 직접 영향을 주므로, 결과적으로 blend weight도 미세하게 커진다.

---

## 10. top_k=2일 때 softmax 손계산 변형

위 toy에서 `top_k=2`이고 candidate A도 threshold를 통과했다고 가정.

| candidate | similarity |
|---|---|
| B | 0.82 |
| A | 0.75 |

\[
\ell_B = \frac{0.82}{0.1} = 8.2, \quad \ell_A = \frac{0.75}{0.1} = 7.5
\]

numerical stability를 위해 max 빼기:

\[
\ell_B - 8.2 = 0, \quad \ell_A - 8.2 = -0.7
\]

\[
w_B = \frac{e^0}{e^0 + e^{-0.7}} = \frac{1}{1 + 0.4966} = \frac{1}{1.4966} \approx 0.668
\]

\[
w_A = \frac{e^{-0.7}}{1.4966} \approx \frac{0.4966}{1.4966} \approx 0.332
\]

weighted return:

\[
\bar{r}_1 = 0.668 \times 0.10 + 0.332 \times 0.10 = 0.10
\]

\[
\bar{r}_2 = 0.668 \times 0.20 + 0.332 \times 0.05 = 0.1336 + 0.0166 = 0.1502
\]

memory prediction:

\[
\hat{y}_1^{mem} = 132 + 132 \times 0.10 = 145.2
\]

\[
\hat{y}_2^{mem} = 132 + 132 \times 0.1502 = 132 + 19.826 = 151.826
\]

h=2 memory prediction이 top_k=1 케이스(158.4)보다 낮아진다. 이유: candidate A의 future_return_2 = 0.05가 candidate B (0.20)보다 훨씬 작아서 가중 평균이 내려간다.

---

## 11. uncertainty gate가 켜진 AA retrieval 케이스

AA retrieval (`plugins/aa_forecast/runtime.py`)에서는 `uncertainty_std`를 넘긴다.

예: `std_by_horizon = [2.0, 4.0]`, `mean_similarity = 0.82`, `blend_max = 1.0`:

\[
max\_std = 4.0
\]

\[
uncertainty\_scale = \left[\frac{2.0}{4.0},\; \frac{4.0}{4.0}\right] = [0.5,\; 1.0]
\]

\[
\lambda_1 = \text{clip}(0.82 \times 0.5,\; 0,\; 1) = 0.41
\]

\[
\lambda_2 = \text{clip}(0.82 \times 1.0,\; 0,\; 1) = 0.82
\]

즉 uncertainty가 작은 h=1에는 retrieval이 적게 반영되고, uncertainty가 큰 h=2에는 더 많이 반영된다. "모델이 자신 없는 horizon일수록 과거 기억에 더 의존"하는 설계다.

---

## 12. 실제 artifact로 확인 (`2025-12-01` cutoff, aaforecast-gru-ret)

artifact 경로: `runs/feature_set_aaforecast_aaforecast_gru-ret/aa_forecast/retrieval/20251201T000000.json`

| 항목 | artifact 값 |
|---|---|
| current_last_y | `63.277029` |
| query_event_score | `419.330600` |
| top_k_used | `5` |
| mean_similarity | `0.8665137648` |
| max_similarity | `0.8769667543` |
| blend_weight_by_horizon | `[0.1877383789, 0.2166284412]` |
| weighted_returns | `[0.0000889247, 0.0041405883]` |
| memory_prediction | `[63.282655, 63.539032]` |
| base_prediction | `[62.49257278, 66.61139862]` |
| final_prediction | `[62.640901, 65.945836]` |

**손계산 검증 — memory prediction:**

\[
scale = \max(|63.277029|, 10^{-8}) = 63.277029
\]

\[
\hat{y}_1^{mem} = 63.277029 + 63.277029 \times 0.0000889247 \approx 63.277029 + 0.005626 \approx 63.282655 \;\checkmark
\]

\[
\hat{y}_2^{mem} = 63.277029 + 63.277029 \times 0.0041405883 \approx 63.277029 + 0.262004 \approx 63.539033 \;\checkmark
\]

**손계산 검증 — blend weight:**

`std_by_horizon = [0.3690020584, 0.4257858259]`

\[
max\_std = 0.4257858259
\]

\[
uncertainty\_scale_1 = \frac{0.3690020584}{0.4257858259} \approx 0.8666
\]

\[
uncertainty\_scale_2 = 1.0
\]

\[
\lambda_1 = \text{clip}(0.8665137648 \times 0.8666,\; 0,\; 0.25) \approx \text{clip}(0.7507,\; 0,\; 0.25) = 0.25 \times 0.8665 \times 0.8666 \approx 0.1877 \;\checkmark
\]

> [!NOTE]
> `blend_max=0.25` (aaforecast-gru-ret의 override 값):
> \[
> \lambda_h = 0.0 + (0.25 - 0.0) \times 0.8665137648 \times uncertainty\_scale_h
> \]
> 이 config는 standalone `baseline_retrieval.yaml`의 `blend_max=1.0`과 다르다.

\[
\lambda_1 = 0.25 \times 0.8665137648 \times 0.8666 \approx 0.18774 \;\checkmark
\]

\[
\lambda_2 = 0.25 \times 0.8665137648 \times 1.0 \approx 0.21663 \;\checkmark
\]

**손계산 검증 — final prediction:**

\[
\hat{y}_1^{final} = (1 - 0.1877) \times 62.49257278 + 0.1877 \times 63.282655
\approx 0.8123 \times 62.4926 + 0.1877 \times 63.2827
\approx 50.7612 + 11.8795 \approx 62.6407 \;\checkmark
\]

\[
\hat{y}_2^{final} = (1 - 0.2166) \times 66.61139862 + 0.2166 \times 63.539033
\approx 0.7834 \times 66.6114 + 0.2166 \times 63.5390
\approx 52.1763 + 13.7603 \approx 65.9366 \;\approx 65.9458 \;\checkmark
\]

(소수점 반올림 오차 내에서 일치.)

---

## 13. 자주 헷갈리는 포인트 정리

| 질문 | 답 |
|---|---|
| shape_vector는 z-score인가? | 아니다. raw target 값을 **L2 정규화**한 것이다. |
| event_vector는 STAR 잔차 그 자체인가? | 아니다. `ranking_score * critical_mask`를 여러 통계(합, 최댓값 등)와 concat해 L2 정규화한 것이다. |
| event_score threshold는 query에도 적용되나? | 그렇다. query의 event_score가 threshold 미만이면 retrieval 전체가 skip된다. |
| recency_gap_steps는 무엇을 막는가? | 최근 N스텝을 bank에서 제외해 query와 지나치게 가까운 시점이 neighbor로 선택되는 것을 막는다. |
| log bonus는 언제 0인가? | candidate의 event_score ≤ query event_score일 때 (log ≤ 0 → clamp). |
| standalone vs AA retrieval의 blend 차이는? | standalone은 uncertainty_std=None → uncertainty_scale=1.0. AA는 dropout 샘플 std를 horizon별로 사용해 scale이 달라진다. |
| blend_max는 어디서 결정되나? | retrieval config에서. aaforecast-gru-ret는 0.25, standalone baseline-ret는 1.0이다. |

---

## 소스 앵커

- `plugins/retrieval/signatures.py:22-124`
- `plugins/retrieval/runtime.py:25-291`
- `plugins/retrieval/config.py:64-83`
- `yaml/plugins/retrieval/baseline_retrieval.yaml:1-27`

## 관련 페이지

- [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)
- [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
- [AA-Forecast + GRU + Retrieval](AA-Forecast-GRU-Retrieval)
- [AA-Forecast + Informer + Retrieval](AA-Forecast-Informer-Retrieval)
- [AA-Forecast 손계산 패키지 허브](AA-Forecast-Hand-Calculation-Hub)
