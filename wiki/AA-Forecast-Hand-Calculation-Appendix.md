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
> `use_shape_key=false` 가 설정된 경우에는 이 shape signature가 아니라 아래의 event signature를 사용합니다. 형태 유사도가 아니라 이벤트 유사도로 neighbor를 뽑는 것입니다.

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

## 소스 앵커

Source: `plugins/retrieval/runtime.py:31-71`, `plugins/retrieval/runtime.py:123-235`, `plugins/retrieval/runtime.py:248-281`, `plugins/aa_forecast/runtime.py:1110-1196`, `plugins/aa_forecast/runtime.py:1439-1496`, `yaml/plugins/retrieval/baseline_retrieval.yaml:1-27`

## 관련 페이지

- [AA-Forecast 손계산 패키지 허브](AA-Forecast-Hand-Calculation-Hub)
- [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
- [AA-Forecast + GRU + Retrieval](AA-Forecast-GRU-Retrieval)
- [AA-Forecast + Informer + Retrieval](AA-Forecast-Informer-Retrieval)
