# AA-Forecast 베이스라인 + Retrieval

## 1. Variant definition

이 페이지는 `baseline-ret.yaml` 을 설명합니다. baseline과 동일하게 command 하나가 `GRU`, `Informer` 두 model로 fan-out 되지만, 이번에는 **standalone retrieval plugin** 이 예측 뒤에 붙습니다.

## 2. Command / config

```bash
uv run python main.py --config yaml/experiment/feature_set_aaforecast/baseline-ret.yaml
```

- main config: `yaml/experiment/feature_set_aaforecast/baseline-ret.yaml`
- jobs route: `yaml/jobs/main/gru_informer.yaml`
- retrieval detail: `yaml/plugins/retrieval/baseline_retrieval.yaml`

## 3. What is ON / OFF

| Switch | 상태 |
|---|---|
| retrieval | On |
| aa_forecast | Off |
| use_shape_key | Off |
| use_event_key | On |
| top_k | 1 |
| use_uncertainty_gate | On |
| blend_max | 1.0 |

> [!NOTE]
> Provenance: `repo default`
>
> 위 값들은 `baseline_retrieval.yaml` 의 현재 detail config와 직접 대응합니다.

## 4. Pipeline delta vs prior variant

baseline 대비 추가되는 것은 아래뿐입니다.

1. 마지막 input window에서 query signature를 만듭니다.
2. 과거 history에서 candidate bank를 만듭니다.
3. similarity가 가장 높은 neighbor를 뽑습니다 (`top_k=1`).
4. 그 neighbor의 future return을 현재 scale에 다시 입혀 `memory_prediction` 을 만듭니다.
5. base prediction과 memory prediction을 섞어 `final_prediction` 을 만듭니다.

## 5. What is literal vs schematic on this page

### literal
- query/candidate window
- event score threshold
- similarity filtering
- future return
- softmax weight (`top_k=1` 이므로 사실상 1.0)
- uncertainty-gated blend

### schematic
- baseline GRU / Informer가 base prediction을 만드는 learned path

## 6. Core formulas used in this variant

future return:
```math
r_h^{(i)} = \frac{y^{(i)}_{future,h} - a^{(i)}}{\max(|a^{(i)}|, \epsilon)}
```

memory prediction:
```math
\hat y_h^{mem} = y_T + |y_T|\bar r_h
```

final blend:
```math
\hat y_h^{final} = (1-\lambda_h)\hat y_h^{base} + \lambda_h\hat y_h^{mem}
```

현재 detail config는 `use_uncertainty_gate: true`, `blend_max: 1.0` 이므로 toy에서는:
```math
\lambda_h = mean\_similarity \times uncertainty\_scale_h
```
처럼 읽으면 됩니다.

## 7. Toy sample setup

공통 toy target series:
```math
[100, 101, 102, 120, 132, 126, 107, 110, 121, 132]
```

query:
```math
Q = [107, 110, 121, 132]
```

candidate B:
```math
B = [132, 126, 107, 110]
```

candidate B의 future return:
```math
[(121-110)/110, (132-110)/110] = [0.10, 0.20]
```

## 8. Step-by-step hand calculation

### Step 1 — current last value (literal)

```math
y_T = 132
```

### Step 2 — retrieved return path (literal)

`top_k=1` 이므로 weighted return도 그대로:
```math
\bar r = [0.10, 0.20]
```

### Step 3 — memory prediction (literal)

```math
\hat y^{mem} = 132 + 132 \times [0.10, 0.20] = [145.2, 158.4]
```

### Step 4 — uncertainty gate (literal)

toy에서 uncertainty scale을
```math
[0.5, 1.0]
```

그리고 mean similarity를
```math
0.887
```

로 두면,
```math
\lambda = [0.4435, 0.887]
```

### Step 5 — GRU subsection (schematic base + literal blend)

baseline GRU base output을 teaching용으로
```math
\hat y^{GRU}_{base} = [136, 138]
```

라고 두면,
```math
\hat y^{GRU}_{final} = [140.0802, 156.0948]
```

### Step 6 — Informer subsection (schematic base + literal blend)

baseline Informer base output을 teaching용으로
```math
\hat y^{Informer}_{base} = [135, 140]
```

라고 두면,
```math
\hat y^{Informer}_{final} = [139.5237, 156.3208]
```

## 9. Interpretation

- retrieval은 base model을 대체하지 않습니다.
- 먼저 base prediction을 만든 뒤, 과거 유사 사건의 future return을 꺼내 와서 posthoc으로 섞습니다.
- `blend_max=1.0` 이므로 toy에서는 retrieval 비중이 커질 수 있지만, 실제 적용 정도는 similarity와 uncertainty scale에 달려 있습니다.

## 10. Grounding notes to repo surfaces

- `baseline-ret.yaml` 는 top-level retrieval을 켭니다.
- `baseline_retrieval.yaml` 는 `top_k=1`, `use_event_key=true`, `use_shape_key=false`, `use_uncertainty_gate=true` 입니다.
- retrieval literal 수식은 standalone retrieval runtime과 직접 대응합니다.

## 11. Provenance tags summary

- `repo default`: retrieval on/off, top_k, blend_max, uncertainty gate
- `toy simplification`: query/candidate 숫자, uncertainty scale, base outputs
- `variant-specific override`: retrieval branch 활성화

## 관련 페이지

- [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)
- [AA-Forecast 베이스라인 (GRU / Informer)](AA-Forecast-Baseline-GRU-Informer)
- [AA-Forecast + GRU + Retrieval](AA-Forecast-GRU-Retrieval)
- [AA-Forecast + Informer + Retrieval](AA-Forecast-Informer-Retrieval)

Source: `yaml/experiment/feature_set_aaforecast/baseline-ret.yaml:1-24`, `yaml/jobs/main/gru_informer.yaml:1-15`, `yaml/plugins/retrieval/baseline_retrieval.yaml:1-27`, `plugins/retrieval/runtime.py:31-71`, `plugins/retrieval/runtime.py:123-281`
