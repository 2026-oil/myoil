# AA-Forecast 공통 수식·표기·손계산 부록

이 페이지는 AA-Forecast 손계산 패키지의 **공통 notation / toy series / provenance rule** 을 정의하는 single source of truth 입니다.

## provenance tag legend

| Tag | 의미 |
|---|---|
| `repo default` | 현재 워킹트리의 YAML / runtime default와 직접 대응하는 설명 |
| `toy simplification` | 계산 구조를 이해시키기 위해 숫자나 길이를 축소한 설명 |
| `variant-specific override` | 특정 variant에서만 켜지거나 달라지는 옵션 |

## 공통 toy series

공통 target series:

\[
y = [100, 101, 102, 120, 132, 126, 107, 110, 121, 132]
\]

공통 toy exogenous 예시:

\[
GPRD\_THREAT = [10, 12, 13, 14, 15, 14, 12, 14, 30, 35]
\]

\[
BS\_Core\_Index\_A = [0.1, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3, 1.4, 1.6]
\]

toy default:
- `L = 4`
- `H = 2`
- query window = 마지막 4개 값

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 repo default는 실험에 따라 `input_size=64`, `uncertainty.sample_count=50`, `recency_gap_steps=8` 등 더 큽니다.

### 시리즈 인덱스 테이블

전체 10개 시점을 0-기반 인덱스로 정렬하면 다음과 같습니다.

| 인덱스 \(t\) | \(y_t\) | \(GPRD\_THREAT_t\) | \(BS\_Core\_Index\_A_t\) |
|---|---|---|---|
| 0 | 100 | 10 | 0.1 |
| 1 | 101 | 12 | 0.1 |
| 2 | 102 | 13 | 0.2 |
| 3 | 120 | 14 | 0.3 |
| 4 | 132 | 15 | 0.4 |
| 5 | 126 | 14 | 0.3 |
| 6 | 107 | 12 | 0.2 |
| 7 | 110 | 14 | 0.3 |
| 8 | 121 | 30 | 1.4 |
| 9 | 132 | 35 | 1.6 |

\(T = 9\) (마지막 인덱스), `L=4`, `H=2` 이므로:

- **query 시작 인덱스** \(T - L + 1 = 9 - 4 + 1 = 6\)
- **query window** \(y[6:10] = [107, 110, 121, 132]\)
- **forecast target** \(y[10], y[11]\) — 아직 관측되지 않은 두 시점

## 표기법

| 기호 | 의미 |
|---|---|
| \(L\) | input window length |
| \(H\) | forecast horizon |
| \(Q\) | 현재 query window |
| \(W^{(i)}\) | 과거 candidate window |
| \(a^{(i)}\) | candidate 마지막 anchor 값 |
| \(r_h^{(i)}\) | candidate의 h-step future return |
| \(w_i\) | softmax로 얻은 neighbor weight |
| \(\hat y_h^{base}\) | retrieval 이전 base prediction |
| \(\hat y_h^{mem}\) | retrieval memory prediction |
| \(\hat y_h^{final}\) | 최종 blended prediction |
| \(\lambda_h\) | horizon별 blend weight |

## sliding window 수식

현재 query window:

\[
Q = [y_{T-L+1}, \dots, y_T]
\]

과거 candidate window:

\[
W^{(i)} = [y_{i-L+1}, \dots, y_i]
\]

candidate anchor:

\[
a^{(i)} = y_i
\]

## future return 수식

candidate 뒤의 미래값이 \(y^{(i)}_{future, h}\) 일 때:

\[
r_h^{(i)} = \frac{y^{(i)}_{future,h} - a^{(i)}}{\max(|a^{(i)}|, \epsilon)}
\]

이 수식은 standalone retrieval (`plugins/retrieval/runtime.py`) 과 AA retrieval (`plugins/aa_forecast/runtime.py`) 모두의 핵심 공통 구조입니다.

## weighted return / memory prediction

neighbor 가중 평균 수익률:

\[
\bar r_h = \sum_i w_i r_h^{(i)}
\]

memory prediction:

\[
\hat y_h^{mem} = y_T + |y_T| \bar r_h
\]

## blend 수식

최종 예측:

\[
\hat y_h^{final} = (1-\lambda_h) \hat y_h^{base} + \lambda_h \hat y_h^{mem}
\]

standalone retrieval과 AA retrieval 모두 `blend_floor`, `blend_max`, `mean_similarity`, `uncertainty_scale` 을 이용해 \(\lambda_h\) 를 만듭니다.

## event / shape signature 직관

### shape signature

\[
s_{shape} = normalize(y_{t-L+1:t})
\]

**toy 예시**: query window \(Q = [107, 110, 121, 132]\) 의 z-score 정규화:
\[
\bar{Q} = \frac{107+110+121+132}{4} = 117.5, \quad \sigma_Q = \sqrt{\frac{(107-117.5)^2+(110-117.5)^2+(121-117.5)^2+(132-117.5)^2}{4}} \approx 9.86
\]
\[
s_{shape}^{Q} \approx [-1.065,\ -0.761,\ 0.355,\ 1.471]
\]

candidate A \(= [100, 101, 102, 120]\) 의 shape signature 도 비슷하게 구하면:
\[
\bar{A} = 105.75, \quad \sigma_A \approx 8.26
\]
\[
s_{shape}^{A} \approx [-0.697,\ -0.576,\ -0.454,\ 1.727]
\]

Q와 A의 코사인 유사도 \(\approx 0.89\) 로 높습니다. 두 윈도우 모두 마지막 시점에 급등이 있기 때문입니다.

> [!NOTE]
> Provenance: `toy simplification`
>
> 이 z-score 기반 shape signature는 "모양 직관"을 설명하기 위한 것입니다.  
> 실제 `compute_star_signature` 구현에서 `shape_vector`는 **raw target 값을 L2 정규화**한 벡터로, z-score와 다릅니다.  
> 두 스위치가 모두 켜지면(`use_shape_key=true`, `use_event_key=true`) similarity = 0.20·shape_sim + 0.80·event_component 로 결합됩니다.

### event signature

STAR 이후 payload에서 대략 다음 조각을 normalize 해서 사용합니다.

\[
s_{event} = normalize([critical\_mask, count\_active, channel\_activity, activity\_sums, activity\_max])
\]

**toy 예시**: query window에서 STAR decomposition 결과를 사용해 event signature를 만든다고 할 때:

| 채널 | residual 마지막 값 | toy threshold | critical |
|---|---|---|---|
| target | 16 (= 132 − 116) | 10 | ✓ |
| GPRD_THREAT | 17 (= 35 − 18) | 10 | ✓ |
| BS_Core_Index_A | 1.1 (= 1.6 − 0.5) | 0.5 | ✓ |

여기서 trend는 각 채널의 단순 선형 trend toy 값입니다.

`count_active` (마지막 시점) = 3, `channel_activity` 마지막 값 합산 ≈ 34.1.

### event score

\[
event\_score = \sum count\_active\_channels + \sum |channel\_activity|
\]

**toy**: 위 예에서 \(event\_score \approx 3 + 34.1 = 37.1\) (teaching placeholder).

실제 repo default `event_score_threshold=400.0` 와 비교하면 toy 값은 훨씬 작지만, threshold를 toy에 맞게 낮게 설정한다고 가정합니다.

> [!NOTE]
> Provenance: `repo default`
>
> Retrieval similarity/threshold/blend의 literal 수식은 실제 구현과 직접 대응합니다. 다만 STAR 내부 decomposition 상세는 페이지별로 필요한 만큼만 schematic으로 줄여 설명합니다.

## literal vs schematic 경계

### literal
- window를 어디서 자르는가
- future return을 어떻게 계산하는가
- top-k와 softmax를 어떻게 적용하는가
- blend weight가 어떻게 base와 memory를 섞는가
- YAML toggle이 어떤 branch를 켜고 끄는가

### schematic
- GRU recurrent update의 내부 weight 계산
- Informer attention head들의 내부 score 계산 전체
- AA-Forecast event/path/regime latent representation의 내부 weight 계산

## toy에서 자주 재사용하는 candidate 예시

### 전체 슬라이딩 윈도우 뱅크

`L=4`, `H=2` 로 10개 시점 시리즈에서 candidate를 생성하려면, anchor \(i\) 가 \(i \geq L-1 = 3\) 이면서 그 뒤로 H개 미래가 존재(\(i + H \leq 9\)) 해야 합니다. 즉 \(i \in \{3, 4, 5, 6, 7\}\) 이 가능합니다. 이 패키지에서는 설명을 위해 **anchor=3 (candidate A)** 와 **anchor=7 (candidate B)** 두 개를 사용합니다.

| candidate | anchor \(i\) | window \(y[i-3:i+1]\) | anchor값 \(a^{(i)}\) | future \(y[i+1:i+3]\) |
|---|---|---|---|---|
| A | 3 | \([100, 101, 102, 120]\) | 120 | \([132, 126]\) |
| B | 7 | \([132, 126, 107, 110]\) | 110 | \([121, 132]\) |

query:
\[
Q = [107, 110, 121, 132]
\]

candidate A:
\[
A = [100, 101, 102, 120], \quad a^{(A)} = 120
\]

candidate B:
\[
B = [132, 126, 107, 110], \quad a^{(B)} = 110
\]

### candidate A의 future return

\[
r_1^{(A)} = \frac{132 - 120}{\max(|120|, \epsilon)} = \frac{12}{120} = 0.10
\]
\[
r_2^{(A)} = \frac{126 - 120}{\max(|120|, \epsilon)} = \frac{6}{120} = 0.05
\]

따라서 \(r^{(A)} = [0.10,\ 0.05]\).

### candidate B의 future return

\[
r_1^{(B)} = \frac{121 - 110}{\max(|110|, \epsilon)} = \frac{11}{110} = 0.10
\]
\[
r_2^{(B)} = \frac{132 - 110}{\max(|110|, \epsilon)} = \frac{22}{110} = 0.20
\]

따라서 \(r^{(B)} = [0.10,\ 0.20]\).

두 candidate 모두 h=1 return이 동일(0.10) 하지만, h=2 return에서 갈립니다. B는 “과거에도 2-step 뒤 더 큰 상승이 있었다”는 기억을 담고 있습니다.

이 예시는 retrieval가 “과거의 상대 수익률 경로를 현재 scale에 다시 입힌다”는 직관을 설명할 때 반복 사용합니다.

---

## 메모리 뱅크 Retrieval 손계산 — 전체 흐름 예시

이 섹션은 메모리 뱅크 빌드부터 최종 blended prediction까지를 **단계별 손계산**으로 전개합니다.  
위에서 정의한 공통 toy series와 표기법을 그대로 사용하며, [Retrieval 손계산 완전 해설](Retrieval-Deep-Dive-Hand-Calculation) 페이지의 end-to-end 예시와 숫자가 대응합니다.

### R-0. 이 예시에서 사용하는 config

| 파라미터 | toy 값 | 의미 |
|---|---|---|
| `L` (input_size) | `4` | window 길이 |
| `H` (horizon) | `2` | 예측 horizon |
| `top_k` | `1` | 검색할 최상위 이웃 수 |
| `recency_gap_steps` | `0` | toy simplification (repo default=8) |
| `event_score_threshold` | `1.0` | toy simplification (repo default=400.0) |
| `min_similarity` | `0.7` | 이웃 최소 유사도 |
| `use_shape_key` | `true` | shape 유사도 사용 |
| `use_event_key` | `true` | event 유사도 사용 |
| `event_score_log_bonus_alpha` | `0.15` | log bonus 가중치 |
| `event_score_log_bonus_cap` | `0.1` | log bonus 상한 |
| `temperature` | `0.1` | softmax temperature |
| `blend_floor` | `0.0` | blend weight 하한 |
| `blend_max` | `1.0` | blend weight 상한 |
| `use_uncertainty_gate` | `true` | horizon별 uncertainty로 blend 조정 여부 |

> [!NOTE]
> Provenance: `toy simplification`
>
> `recency_gap_steps=0`, `event_score_threshold=1.0`은 toy에서 계산 구조를 보이기 위한 축소값입니다.  
> 실제 repo default는 각각 `8`과 `400.0`입니다.  
> `use_shape_key=true`, `use_event_key=true`, `use_uncertainty_gate=true` — 이 예시에서는 세 스위치가 모두 켜진 상태의 전체 retrieval 파이프라인을 보입니다.

공통 toy series:

\[
y = [100, 101, 102, 120, 132, 126, 107, 110, 121, 132]
\]
\[
GPRD\_THREAT = [10, 12, 13, 14, 15, 14, 12, 14, 30, 35]
\]

---

### R-1. 메모리 뱅크 빌드

**유효 anchor 범위 계산:**

\[
last\_idx = 9, \quad max\_end\_idx = last\_idx - H - gap = 9 - 2 - 0 = 7
\]

\[
\text{유효 } end\_idx \in [L-1,\; max\_end\_idx] = [3,\; 7]
\]

따라서 anchor 후보: \(i \in \{3, 4, 5, 6, 7\}\), 총 5개.  
이 예시에서는 설명을 위해 anchor=3 (candidate A)과 anchor=7 (candidate B) 두 개를 중점적으로 사용합니다.

#### candidate A (anchor = 3)

window: \(y[0:4] = [100, 101, 102, 120]\)

**shape_vector 계산** (raw target values를 L2 정규화):

\[
v^{(A)} = [100, 101, 102, 120]
\]

\[
\|v^{(A)}\| = \sqrt{100^2 + 101^2 + 102^2 + 120^2} = \sqrt{10000 + 10201 + 10404 + 14400} = \sqrt{45005} \approx 212.1
\]

\[
shape\_vector^{(A)} \approx \left[\frac{100}{212.1},\; \frac{101}{212.1},\; \frac{102}{212.1},\; \frac{120}{212.1}\right] \approx [0.4715,\; 0.4762,\; 0.4809,\; 0.5658]
\]

**event_vector 및 event_score** (teaching placeholder):  
GPRD_THREAT for window 0~3 = \([10, 12, 13, 14]\) — 완만한 상승, 큰 잔차 없음.

toy STAR 결과 (단순화):
```text
target_critical_mask   (L=4): [0, 0, 0, 0]
hist_critical_mask     (L=4): [0, 0, 0, 0]
combined_count                : [0, 0, 0, 0]
channel_activity (L×2)        : 모두 0
```

\[
event\_score^{(A)} = \sum count\_active + \sum |channel\_activity| = 0 + 0 = 0.0
\]

\(event\_score^{(A)} = 0.0 < threshold = 1.0\) → **candidate A 탈락** (bank에 포함되지 않음).

**anchor value, future values, future returns** (참고용):

\[
a^{(A)} = y_3 = 120
\]

\[
r_1^{(A)} = \frac{y_4 - a^{(A)}}{\max(|a^{(A)}|, \varepsilon)} = \frac{132 - 120}{120} = \frac{12}{120} = 0.10
\]

\[
r_2^{(A)} = \frac{y_5 - a^{(A)}}{\max(|a^{(A)}|, \varepsilon)} = \frac{126 - 120}{120} = \frac{6}{120} = 0.05
\]

#### candidate B (anchor = 7)

window: \(y[4:8] = [132, 126, 107, 110]\)

**shape_vector 계산:**

\[
v^{(B)} = [132, 126, 107, 110]
\]

\[
\|v^{(B)}\| = \sqrt{132^2 + 126^2 + 107^2 + 110^2} = \sqrt{17424 + 15876 + 11449 + 12100} = \sqrt{56849} \approx 238.4
\]

\[
shape\_vector^{(B)} \approx \left[\frac{132}{238.4},\; \frac{126}{238.4},\; \frac{107}{238.4},\; \frac{110}{238.4}\right] \approx [0.5537,\; 0.5285,\; 0.4488,\; 0.4614]
\]

**event_vector 및 event_score** (teaching placeholder):  
GPRD_THREAT for window 4~7 = \([15, 14, 12, 14]\) — 뚜렷한 이상치 없지만 toy threshold=1.0에서는 통과.

toy STAR 결과 (단순화):
```text
target_critical_mask   (L=4): [0, 0, 0, 0]
hist_critical_mask     (L=4): [0, 0, 0, 0]
combined_count                : [0, 0, 0, 0]
channel_activity (L×2)        : 모두 0, 단 마지막 시점 소량 활동
```

toy에서 `event_score_B ≈ 2.5` (teaching placeholder).

\[
event\_score^{(B)} = 2.5 \geq threshold = 1.0 \quad \Rightarrow \quad \textbf{candidate B 통과}
\]

**anchor value, future values, future returns:**

\[
a^{(B)} = y_7 = 110
\]

\[
r_1^{(B)} = \frac{y_8 - a^{(B)}}{\max(|a^{(B)}|, \varepsilon)} = \frac{121 - 110}{110} = \frac{11}{110} = 0.10
\]

\[
r_2^{(B)} = \frac{y_9 - a^{(B)}}{\max(|a^{(B)}|, \varepsilon)} = \frac{132 - 110}{110} = \frac{22}{110} = 0.20
\]

**bank 저장 레코드 (candidate B):**

```text
{
  "shape_vector"       : normalize([132, 126, 107, 110]) ≈ [0.5537, 0.5285, 0.4488, 0.4614],
  "event_vector"       : normalize(event_vector_raw_B)  — L2 정규화 완료,
  "event_score"        : 2.5,
  "anchor_target_value": 110,
  "future_returns"     : [0.10, 0.20],
}
```

---

### R-2. Query 빌드

query window = 마지막 \(L=4\)개 행: 인덱스 6~9

\[
Q = [107, 110, 121, 132], \quad y_T = 132
\]

**shape_vector 계산:**

\[
v^{Q} = [107, 110, 121, 132]
\]

\[
\|v^{Q}\| = \sqrt{107^2 + 110^2 + 121^2 + 132^2} = \sqrt{11449 + 12100 + 14641 + 17424} = \sqrt{55614} \approx 235.8
\]

\[
shape\_vector^{Q} \approx \left[\frac{107}{235.8},\; \frac{110}{235.8},\; \frac{121}{235.8},\; \frac{132}{235.8}\right] \approx [0.4537,\; 0.4665,\; 0.5132,\; 0.5598]
\]

**event_vector 및 event_score:**  
query window에서 GPRD_THREAT = \([12, 14, 30, 35]\) — 인덱스 8, 9에 급등.

toy STAR 결과 (단순화):

```text
target_critical_mask   (L=4, 1-채널): [[0], [0], [0], [1]]
target_ranking_score   (L=4, 1-채널): [[0], [0], [0], [1.60]]
target_activity                      : [[0], [0], [0], [1.60]]

hist_critical_mask     (L=4, GPRD_THREAT): [[0], [0], [1], [1]]
hist_ranking_score     (L=4)             : [[0], [0], [1.40], [1.70]]
hist_activity                            : [[0], [0], [1.40], [1.70]]
```

combined_count (= target_count + hist_count):

\[
combined\_count = [0,\; 0,\; 1,\; 2]
\]

channel_activity (\(L=4\), 2-채널 = target + GPRD_THREAT):

\[
channel\_activity = \begin{bmatrix}0 & 0 \\ 0 & 0 \\ 0 & 1.40 \\ 1.60 & 1.70\end{bmatrix}
\]

event_vector_raw 조립:

```text
critical_mask_flat  (L=4)     : [0, 0, 1, 1]
count_active_flat   (L=4)     : [0, 0, 1, 2]
channel_act_flat    (L*2=8)   : [0, 0, 0, 0, 0, 1.40, 1.60, 1.70]   ← C-order flatten
activity_sums       (2-채널)  : [1.60, 3.10]
activity_max        (2-채널)  : [1.60, 1.70]
```

concat 결과 (길이 = 4+4+8+2+2 = 20):

\[
event\_vector\_raw^{Q} = [\underbrace{0,0,1,1}_{\text{critical\_mask}},\; \underbrace{0,0,1,2}_{\text{count\_active}},\; \underbrace{0,0,0,0,0,1.40,1.60,1.70}_{\text{channel\_act}},\; \underbrace{1.60,3.10}_{\text{sums}},\; \underbrace{1.60,1.70}_{\text{max}}]
\]

L2 정규화 후 \(event\_vector^{Q}\) 를 얻는다.

event_score 계산:

\[
event\_score^{Q} = \sum count\_active + \sum |channel\_activity|
= (0+0+1+2) + (0+0+1.40+1.60+1.70) = 3 + 4.70 = 7.70
\]

\(event\_score^{Q} = 7.70 \geq threshold = 1.0\) → **query 통과, retrieval 실행.**

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 STAR는 LOWESS 트렌드 제거 + 잔차 임계화 기반입니다. 위 ranking_score 숫자는 동작 구조를 설명하기 위한 teaching placeholder입니다.

---

### R-3. Neighbor 검색

**bank 상태:**

| candidate | event_score | shape_vector | future_returns |
|---|---|---|---|
| A | 0.0 (탈락) | — | — |
| B | 2.5 (통과) | normalize([132,126,107,110]) | [0.10, 0.20] |

bank에 남은 후보: **candidate B 단독**.

**similarity 계산 (candidate B vs query):**

`use_shape_key=true`, `use_event_key=true`이므로 두 키를 모두 결합합니다:

\[
similarity = 0.20 \times shape\_similarity + 0.80 \times event\_component
\]

**step 1 — shape_similarity:**

shape_vector^Q ≈ [0.4537, 0.4665, 0.5132, 0.5598] (R-2 계산 결과)  
shape_vector^B ≈ [0.5537, 0.5285, 0.4488, 0.4614] (R-1 계산 결과)

두 벡터가 모두 L2 정규화되어 있으므로 코사인 유사도 = 내적:

\[
shape\_similarity = shape\_vector^Q \cdot shape\_vector^{(B)}
\]

\[
= 0.4537 \times 0.5537 + 0.4665 \times 0.5285 + 0.5132 \times 0.4488 + 0.5598 \times 0.4614
\]

\[
\approx 0.2514 + 0.2465 + 0.2303 + 0.2583 = 0.9865 \approx 0.99
\]

> [!NOTE]
> shape_vector는 raw target 값을 L2 정규화한 것(z-score 아님)이므로, 두 윈도우의 값들이 비슷한 수준 범위(100~135)에 있으면 L2 정규화 후 각 성분이 모두 0.45~0.56 범위로 수렴해 코사인이 높게 나옵니다. query [107,110,121,132](상승 추세)와 candidate B [132,126,107,110](하강 추세)는 추세 방향은 반대이지만, 정규화 후 4D 벡터로서 성분 크기가 서로 비슷해 shape_similarity ≈ 0.99입니다.

**step 2 — event_score log bonus:**

\[
log\_bonus = \min\!\left(\max\!\left(\ln\frac{event\_score^{(B)}}{event\_score^{Q}},\; 0\right),\; cap\right) = \min\!\left(\max\!\left(\ln\frac{2.5}{7.70},\; 0\right),\; 0.1\right)
\]

\[
\ln\frac{2.5}{7.70} = \ln(0.3247) \approx -1.12 < 0 \quad \Rightarrow \quad \max(-1.12,\; 0) = 0 \quad \Rightarrow \quad log\_bonus = 0
\]

candidate B의 event_score가 query보다 낮으므로 bonus 없음.

**step 3 — event_component:**

\[
event\_component = event\_similarity + 0.15 \times 0 = event\_similarity
\]

**event_similarity** (L2-normalized 벡터의 내적 = cosine similarity):

query와 candidate B 모두 마지막 1~2 시점에서 이벤트가 집중되는 구조적 유사성이 있음.

\[
event\_similarity \approx 0.82 \quad \text{(teaching placeholder)}
\]

\[
event\_component = 0.82
\]

**step 4 — 최종 combined similarity:**

\[
similarity = 0.20 \times 0.99 + 0.80 \times 0.82 = 0.198 + 0.656 = 0.854
\]

**min_similarity 필터:**

\[
0.854 \geq min\_similarity = 0.7 \quad \Rightarrow \quad \text{candidate B 통과}
\]

scored_neighbors = [B (similarity=0.854)].

---

### R-4. Top-k 선택 및 Softmax 가중치

`top_k=1`이므로 상위 1개만 선택: **candidate B**.

softmax weight 계산:

\[
\ell_B = \frac{similarity_B}{temperature} = \frac{0.854}{0.1} = 8.54
\]

후보가 1개뿐이므로 numerical stability 보정 후:

\[
\ell_B - \max(\ell) = 8.54 - 8.54 = 0
\]

\[
w_B = \frac{e^{0}}{e^{0}} = \frac{1}{1} = 1.0
\]

---

### R-5. Memory Prediction 계산

가중 평균 수익률:

\[
\bar{r}_h = \sum_i w_i \cdot r_h^{(i)} = 1.0 \times r_h^{(B)}
\]

\[
\bar{r}_1 = 1.0 \times 0.10 = 0.10, \quad \bar{r}_2 = 1.0 \times 0.20 = 0.20
\]

현재 마지막 값 및 scale:

\[
y_T = 132, \quad scale = \max(|y_T|,\; \varepsilon) = \max(132,\; 10^{-8}) = 132
\]

memory prediction:

\[
\hat{y}_1^{mem} = y_T + scale \times \bar{r}_1 = 132 + 132 \times 0.10 = 132 + 13.2 = 145.2
\]

\[
\hat{y}_2^{mem} = y_T + scale \times \bar{r}_2 = 132 + 132 \times 0.20 = 132 + 26.4 = 158.4
\]

\[
\hat{y}^{mem} = [145.2,\; 158.4]
\]

---

### R-6. Uncertainty-Gated Blend 및 Final Prediction

**similarity_scale:**

\[
mean\_similarity = \frac{1}{1} \times 0.854 = 0.854
\]

\[
similarity\_scale = \text{clip}(0.854,\; 0,\; 1) = 0.854
\]

**uncertainty_scale (`use_uncertainty_gate=true`):**

`use_uncertainty_gate=true`이면 horizon별 예측 불확실성(std)을 최대값으로 나누어 scale을 계산합니다.

AA retrieval 경로에서 `std_by_horizon = [2.0, 4.0]` 이 넘어온 경우:

\[
max\_std = 4.0
\]

\[
uncertainty\_scale = \left[\frac{2.0}{4.0},\; \frac{4.0}{4.0}\right] = [0.5,\; 1.0]
\]

h=1은 모델 불확실성이 상대적으로 작아 scale=0.5, h=2는 불확실성이 커 scale=1.0.  
"모델이 자신 없는 horizon일수록 과거 기억에 더 의존"하는 설계입니다.

> [!NOTE]
> standalone retrieval (`post_predict_retrieval`)은 `uncertainty_std=None`을 전달하므로, `use_uncertainty_gate=true`여도 `uncertainty_scale=1.0`으로 처리됩니다.  
> AA retrieval (`plugins/aa_forecast/runtime.py`)만 dropout 샘플 std를 horizon별로 전달해 uncertainty gate가 실제로 동작합니다.  
> standalone 경로에서의 비교는 아래 R-9를 참조하세요.

**blend weight:**

`blend_floor=0`, `blend_max=1`:

\[
\lambda_h = \text{clip}\!\left(\; blend\_floor + (blend\_max - blend\_floor) \times similarity\_scale \times uncertainty\_scale_h,\; blend\_floor,\; blend\_max \right)
\]

\[
\lambda_1 = \text{clip}(0 + (1-0) \times 0.854 \times 0.5,\; 0,\; 1) = \text{clip}(0.427,\; 0,\; 1) = 0.427
\]

\[
\lambda_2 = \text{clip}(0 + (1-0) \times 0.854 \times 1.0,\; 0,\; 1) = \text{clip}(0.854,\; 0,\; 1) = 0.854
\]

uncertainty가 작은 h=1에는 retrieval이 적게 반영(λ=0.427), 불확실성이 큰 h=2에는 더 많이 반영(λ=0.854)됩니다.

**final prediction** (base prediction은 schematic placeholder):

\[
\hat{y}^{base} = [136,\; 138]
\]

\[
\hat{y}_1^{final} = (1 - 0.427) \times 136 + 0.427 \times 145.2 = 0.573 \times 136 + 0.427 \times 145.2 = 77.928 + 61.980 = 139.908
\]

\[
\hat{y}_2^{final} = (1 - 0.854) \times 138 + 0.854 \times 158.4 = 0.146 \times 138 + 0.854 \times 158.4 = 20.148 + 135.274 = 155.422
\]

\[
\hat{y}^{final} \approx [139.91,\; 155.42]
\]

---

### R-7. 전체 단계 요약표

| 단계 | 항목 | h=1 | h=2 |
|---|---|---|---|
| R-1 | candidate A event_score | 0.0 (탈락) | — |
| R-1 | candidate B event_score | 2.5 (통과) | — |
| R-1 | candidate B future return \(r_h^{(B)}\) | 0.10 | 0.20 |
| R-2 | query event_score | 7.70 (통과) | — |
| R-3 | shape_similarity (B vs Q) | 0.99 | — |
| R-3 | event_similarity (B vs Q) | 0.82 | — |
| R-3 | log_bonus | 0.0 | — |
| R-3 | combined similarity (0.20×0.99 + 0.80×0.82) | 0.854 (통과) | — |
| R-4 | softmax weight \(w_B\) | 1.0 | — |
| R-5 | \(\bar{r}_h\) | 0.10 | 0.20 |
| R-5 | \(\hat{y}_h^{mem}\) | 145.2 | 158.4 |
| R-6 | similarity_scale | 0.854 | — |
| R-6 | uncertainty_scale (AA 경로, std=[2.0, 4.0]) | 0.5 | 1.0 |
| R-6 | blend weight \(\lambda_h\) | 0.427 | 0.854 |
| R-6 | base prediction | 136 | 138 |
| **R-6** | **final prediction** | **139.91** | **155.42** |

---

### R-8. 변형 시나리오 — top_k=2 (candidate A도 통과할 경우)

candidate A가 event_score threshold를 통과하고, 각 similarity가 다음과 같다고 가정합니다.

| candidate | shape_similarity | event_similarity | combined (0.20·shape + 0.80·event) |
|---|---|---|---|
| B | 0.99 | 0.82 | 0.854 |
| A | 1.00 | 0.75 | 0.800 |

> [!NOTE]
> candidate A (anchor=3, window=[100,101,102,120])의 shape_similarity는 L2 정규화 기준으로 계산합니다.  
> shape_vector^A ≈ [0.4715, 0.4762, 0.4809, 0.5658] (R-1 참고), shape_vector^Q ≈ [0.4537, 0.4665, 0.5132, 0.5598] (R-2 참고).  
> 내적 = 0.4715×0.4537 + 0.4762×0.4665 + 0.4809×0.5132 + 0.5658×0.5598 ≈ 0.2139 + 0.2221 + 0.2468 + 0.3167 ≈ 0.9995 ≈ 1.00.  
> A와 Q 모두 값이 100~135 범위에 있어 L2 정규화 후 코사인이 1에 가깝습니다. event_similarity ≈ 0.75는 teaching placeholder입니다.

softmax (`temperature=0.1`):

\[
\ell_B = \frac{0.854}{0.1} = 8.54, \quad \ell_A = \frac{0.800}{0.1} = 8.00
\]

numerical stability 보정:

\[
\ell_B - 8.54 = 0, \quad \ell_A - 8.54 = -0.54
\]

\[
w_B = \frac{e^0}{e^0 + e^{-0.54}} = \frac{1}{1 + 0.5827} = \frac{1}{1.5827} \approx 0.632
\]

\[
w_A = \frac{e^{-0.54}}{1.5827} \approx \frac{0.5827}{1.5827} \approx 0.368
\]

가중 평균 수익률:

\[
\bar{r}_1 = 0.632 \times 0.10 + 0.368 \times 0.10 = 0.10
\]

\[
\bar{r}_2 = 0.632 \times 0.20 + 0.368 \times 0.05 = 0.1264 + 0.0184 = 0.1448
\]

memory prediction:

\[
\hat{y}_1^{mem} = 132 + 132 \times 0.10 = 145.2
\]

\[
\hat{y}_2^{mem} = 132 + 132 \times 0.1448 = 132 + 19.11 = 151.11
\]

h=2 memory prediction이 top_k=1 케이스(158.4)보다 낮아집니다. candidate A의 \(r_2^{(A)} = 0.05\)가 candidate B(0.20)보다 훨씬 낮아서 가중 평균이 내려가기 때문입니다.

---

### R-9. 변형 시나리오 — standalone retrieval에서의 uncertainty gate 비교

standalone retrieval (`post_predict_retrieval`) 경로에서는 `uncertainty_std=None`이 전달되므로, `use_uncertainty_gate=true`여도 uncertainty_scale이 항상 1.0이 됩니다.

\[
uncertainty\_scale = [1.0,\; 1.0]
\]

이 경우 blend weight는 similarity_scale만으로 결정됩니다:

\[
\lambda_1 = \lambda_2 = \text{clip}(0 + 1.0 \times 0.854 \times 1.0,\; 0,\; 1) = 0.854
\]

base prediction \(\hat{y}^{base} = [136,\; 138]\) 기준:

\[
\hat{y}_1^{final} = (1 - 0.854) \times 136 + 0.854 \times 145.2 = 0.146 \times 136 + 0.854 \times 145.2 = 19.856 + 124.001 = 143.857
\]

\[
\hat{y}_2^{final} = (1 - 0.854) \times 138 + 0.854 \times 158.4 = 0.146 \times 138 + 0.854 \times 158.4 = 20.148 + 135.274 = 155.422
\]

주 시나리오(R-6, AA retrieval, std=[2.0, 4.0])와 비교:

| 경로 | \(\lambda_1\) | \(\lambda_2\) | \(\hat{y}_1^{final}\) | \(\hat{y}_2^{final}\) |
|---|---|---|---|---|
| AA retrieval (uncertainty gate 활성화, std=[2.0, 4.0]) | 0.427 | 0.854 | 139.91 | 155.42 |
| standalone (uncertainty gate 미적용, scale=1.0) | 0.854 | 0.854 | 143.86 | 155.42 |

h=2 최종 예측은 동일하지만, h=1은 AA retrieval 경로에서 uncertainty gate가 blend를 줄여 base 쪽으로 당깁니다.  
"모델이 자신 있는 h=1은 base에 더 의존, 자신 없는 h=2는 memory에 더 의존"하는 설계 의도가 수치로 드러납니다.

---

### R-10. 자주 헷갈리는 포인트

| 질문 | 답 |
|---|---|
| shape_vector는 z-score인가? | 아니다. raw target 값을 **L2 정규화**한 것이다. z-score가 아니다. |
| event_vector는 STAR 잔차 그 자체인가? | 아니다. `ranking_score × critical_mask`를 여러 통계(합, 최댓값 등)와 concat해 L2 정규화한 것이다. |
| event_score threshold는 query에도 적용되나? | 그렇다. query의 event_score가 threshold 미만이면 retrieval 전체가 skip된다. |
| recency_gap_steps는 무엇을 막는가? | 최근 N스텝을 bank에서 제외해 query와 지나치게 가까운 시점이 neighbor로 선택되는 것을 막는다. |
| log bonus는 언제 0인가? | candidate event_score ≤ query event_score일 때 (ln ≤ 0 → clamp to 0). |
| use_shape_key=true일 때 similarity 가중치는? | `0.20 × shape_similarity + 0.80 × event_component` (두 키가 모두 켜졌을 때). shape 키만 켜지면 `similarity = shape_similarity`, event 키만 켜지면 `similarity = event_component`. |
| shape_similarity가 높게 나오는 이유는? | shape_vector는 raw 값의 L2 정규화이므로, 두 윈도우의 값들이 비슷한 수준 범위(예: 100~135)에 있으면 L2 정규화 후 각 성분이 비슷한 크기로 수렴해 코사인이 높게 나온다. 추세 방향이 반대인 두 윈도우도 값의 절대 수준이 비슷하면 shape_similarity가 높을 수 있다. 방향(추세 패턴)을 의미있게 비교하려면 z-score 기반 시그니처가 필요하지만 현재 구현은 L2 정규화를 사용한다. |
| standalone vs AA retrieval의 blend 차이는? | standalone은 `uncertainty_std=None` → `uncertainty_scale=1.0`. AA는 dropout 샘플 std를 horizon별로 사용해 scale이 달라진다. |
| future return의 분모는 무엇인가? | `max(|anchor_value|, ε)` — 0에 가까운 anchor에 대한 수치 안정화 장치다. |
| 메모리 예측의 기준점은 무엇인가? | `current_last_y = y_T` — raw train_df 마지막 타깃값. transformed 값이 아니다. |

---

## 소스 앵커

- `plugins/retrieval/signatures.py:22-124`
- `plugins/retrieval/runtime.py:31-71`
- `plugins/retrieval/runtime.py:123-235`
- `plugins/retrieval/runtime.py:248-281`
- `plugins/aa_forecast/runtime.py:1110-1196`
- `plugins/aa_forecast/runtime.py:1439-1496`
- `yaml/plugins/retrieval/baseline_retrieval.yaml:1-27`

## 관련 페이지

- [AA-Forecast 손계산 패키지 허브](AA-Forecast-Hand-Calculation-Hub)
- [Retrieval 손계산 완전 해설](Retrieval-Deep-Dive-Hand-Calculation)
- [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
- [AA-Forecast + GRU + Retrieval](AA-Forecast-GRU-Retrieval)
- [AA-Forecast + Informer + Retrieval](AA-Forecast-Informer-Retrieval)
