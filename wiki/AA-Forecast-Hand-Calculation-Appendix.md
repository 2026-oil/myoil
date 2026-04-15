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

```math
y = [100, 101, 102, 120, 132, 126, 107, 110, 121, 132]
```

공통 toy exogenous 예시:

```math
GPRD\_THREAT = [10, 12, 13, 14, 15, 14, 12, 14, 30, 35]
```

```math
BS\_Core\_Index\_A = [0.1, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3, 1.4, 1.6]
```

toy default:
- `L = 4`
- `H = 2`
- query window = 마지막 4개 값

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 repo default는 실험에 따라 `input_size=64`, `uncertainty.sample_count=50`, `recency_gap_steps=8` 등 더 큽니다.

## 표기법

| 기호 | 의미 |
|---|---|
| $`L`$ | input window length |
| $`H`$ | forecast horizon |
| $`Q`$ | 현재 query window |
| $`W^{(i)}`$ | 과거 candidate window |
| $`a^{(i)}`$ | candidate 마지막 anchor 값 |
| $`r_h^{(i)}`$ | candidate의 h-step future return |
| $`w_i`$ | softmax로 얻은 neighbor weight |
| $`\hat y_h^{base}`$ | retrieval 이전 base prediction |
| $`\hat y_h^{mem}`$ | retrieval memory prediction |
| $`\hat y_h^{final}`$ | 최종 blended prediction |
| $`\lambda_h`$ | horizon별 blend weight |

## sliding window 수식

현재 query window:

```math
Q = [y_{T-L+1}, \dots, y_T]
```

과거 candidate window:

```math
W^{(i)} = [y_{i-L+1}, \dots, y_i]
```

candidate anchor:

```math
a^{(i)} = y_i
```

## future return 수식

candidate 뒤의 미래값이 $`y^{(i)}_{future, h}`$ 일 때:

```math
r_h^{(i)} = \frac{y^{(i)}_{future,h} - a^{(i)}}{\max(|a^{(i)}|, \epsilon)}
```

이 수식은 standalone retrieval (`plugins/retrieval/runtime.py`) 과 AA retrieval (`plugins/aa_forecast/runtime.py`) 모두의 핵심 공통 구조입니다.

## weighted return / memory prediction

neighbor 가중 평균 수익률:

```math
\bar r_h = \sum_i w_i r_h^{(i)}
```

memory prediction:

```math
\hat y_h^{mem} = y_T + |y_T| \bar r_h
```

## blend 수식

최종 예측:

```math
\hat y_h^{final} = (1-\lambda_h) \hat y_h^{base} + \lambda_h \hat y_h^{mem}
```

standalone retrieval과 AA retrieval 모두 `blend_floor`, `blend_max`, `mean_similarity`, `uncertainty_scale` 을 이용해 $`\lambda_h`$ 를 만듭니다.

## event / shape signature 직관

### shape signature

```math
s_{shape} = normalize(y_{t-L+1:t})
```

### event signature

STAR 이후 payload에서 대략 다음 조각을 normalize 해서 사용합니다.

```math
s_{event} = normalize([critical\_mask, count\_active, channel\_activity, activity\_sums, activity\_max])
```

### event score

```math
event\_score = \sum count\_active\_channels + \sum |channel\_activity|
```

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

query:
```math
Q = [107, 110, 121, 132]
```

candidate A:
```math
A = [100, 101, 102, 120]
```

candidate B:
```math
B = [132, 126, 107, 110]
```

candidate B의 future return:
```math
[(121-110)/110, (132-110)/110] = [0.10, 0.20]
```

이 예시는 retrieval가 “과거의 상대 수익률 경로를 현재 scale에 다시 입힌다”는 직관을 설명할 때 반복 사용합니다.

## 소스 앵커

Source: `plugins/retrieval/runtime.py:31-71`, `plugins/retrieval/runtime.py:123-235`, `plugins/retrieval/runtime.py:248-281`, `plugins/aa_forecast/runtime.py:1110-1196`, `plugins/aa_forecast/runtime.py:1439-1496`, `yaml/plugins/retrieval/baseline_retrieval.yaml:1-27`

## 관련 페이지

- [AA-Forecast 손계산 패키지 허브](AA-Forecast-Hand-Calculation-Hub)
- [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
- [AA-Forecast + GRU + Retrieval](AA-Forecast-GRU-Retrieval)
- [AA-Forecast + Informer + Retrieval](AA-Forecast-Informer-Retrieval)
