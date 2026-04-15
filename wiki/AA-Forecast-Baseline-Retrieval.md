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
\[
r_h^{(i)} = \frac{y^{(i)}_{future,h} - a^{(i)}}{\max(|a^{(i)}|, \epsilon)}
\]

memory prediction:
\[
\hat y_h^{mem} = y_T + |y_T|\bar r_h
\]

final blend:
\[
\hat y_h^{final} = (1-\lambda_h)\hat y_h^{base} + \lambda_h\hat y_h^{mem}
\]

현재 detail config는 `use_uncertainty_gate: true`, `blend_max: 1.0` 이므로 toy에서는:
\[
\lambda_h = mean\_similarity \times uncertainty\_scale_h
\]
처럼 읽으면 됩니다.

## 7. Toy sample setup

공통 toy target series (10개 시점):

\[
y = [100, 101, 102, 120, 132, 126, 107, 110, 121, 132]
\]

query (마지막 4개, 인덱스 6~9):
\[
Q = [107, 110, 121, 132], \quad y_T = 132
\]

**candidate bank** (두 개의 주요 후보):

| candidate | anchor \(i\) | window \(y[i-3:i+1]\) | \(a^{(i)}\) | future \(y[i+1:i+3]\) |
|---|---|---|---|---|
| A | 3 | \([100, 101, 102, 120]\) | 120 | \([132, 126]\) |
| B | 7 | \([132, 126, 107, 110]\) | 110 | \([121, 132]\) |

candidate A의 future return:
\[
r_1^{(A)} = \frac{132-120}{120} = 0.10, \quad r_2^{(A)} = \frac{126-120}{120} = 0.05
\]

candidate B의 future return:
\[
r_1^{(B)} = \frac{121-110}{110} = 0.10, \quad r_2^{(B)} = \frac{132-110}{110} = 0.20
\]

`use_event_key=true` 기반 similarity 계산 결과, **candidate B가 top-1 neighbor** 로 선택됩니다 (\(sim \approx 0.887\), teaching placeholder).

> [!NOTE]
> Provenance: `toy simplification`
>
> `use_shape_key=false` 이므로 형태 유사도가 아니라 event signature 유사도로 neighbor를 뽑습니다. toy에서는 candidate B의 event_key 유사도가 더 높다고 가정합니다.

## 8. Step-by-step hand calculation

### Step 1 — query window 및 candidate 확인 (literal)

마지막 4개로 query window를 자릅니다:
\[
Q = [107, 110, 121, 132], \quad y_T = 132
\]

candidate bank에서 event_key 기반 top-1 neighbor를 선택합니다. toy에서 **candidate B** 가 선택됩니다:
- anchor: \(a^{(B)} = 110\)
- future: \([121, 132]\)

### Step 2 — future return 계산 (literal)

candidate B의 anchor \(a^{(B)} = 110\) 과 future \([121, 132]\) 로부터:

\[
r_1^{(B)} = \frac{121 - 110}{\max(|110|, \epsilon)} = \frac{11}{110} = 0.10
\]
\[
r_2^{(B)} = \frac{132 - 110}{\max(|110|, \epsilon)} = \frac{22}{110} = 0.20
\]

`top_k=1` 이므로 softmax weight = 1.0, 가중 평균 수익률 = return 그대로:
\[
\bar r = [0.10, 0.20]
\]

### Step 3 — memory prediction (literal)

현재 scale \(y_T = 132\) 에 수익률 경로를 입힙니다:

\[
\hat y_1^{mem} = 132 + |132| \times 0.10 = 132 + 13.2 = 145.2
\]
\[
\hat y_2^{mem} = 132 + |132| \times 0.20 = 132 + 26.4 = 158.4
\]
\[
\hat y^{mem} = [145.2,\ 158.4]
\]

### Step 4 — blend weight (uncertainty gate) 계산 (literal)

toy에서 다음 값을 가정합니다:
- `mean_similarity` = 0.887 (neighbor similarity 평균, teaching placeholder)
- `uncertainty_scale` = \([0.5, 1.0]\) (horizon별 불확실성 스케일, teaching placeholder)

\(\lambda_h = \min(mean\_similarity \times uncertainty\_scale_h,\ blend\_max)\) 이므로:

\[
\lambda_1 = \min(0.887 \times 0.5,\ 1.0) = \min(0.4435,\ 1.0) = 0.4435
\]
\[
\lambda_2 = \min(0.887 \times 1.0,\ 1.0) = \min(0.887,\ 1.0) = 0.887
\]
\[
\lambda = [0.4435,\ 0.887]
\]

### Step 5 — GRU subsection: blend 계산 (schematic base + literal blend)

baseline GRU base output (schematic placeholder):
\[
\hat y^{GRU}_{base} = [136, 138]
\]

최종 blend:
\[
\hat y_1^{GRU, final} = (1 - 0.4435) \times 136 + 0.4435 \times 145.2 = 0.5565 \times 136 + 0.4435 \times 145.2
= 75.684 + 64.3962 = 140.0802
\]
\[
\hat y_2^{GRU, final} = (1 - 0.887) \times 138 + 0.887 \times 158.4 = 0.113 \times 138 + 0.887 \times 158.4
= 15.594 + 140.5008 = 156.0948
\]
\[
\hat y^{GRU}_{final} = [140.0802,\ 156.0948]
\]

### Step 6 — Informer subsection: blend 계산 (schematic base + literal blend)

baseline Informer base output (schematic placeholder):
\[
\hat y^{Informer}_{base} = [135, 140]
\]

최종 blend:
\[
\hat y_1^{Informer, final} = 0.5565 \times 135 + 0.4435 \times 145.2 = 75.1275 + 64.3962 = 139.5237
\]
\[
\hat y_2^{Informer, final} = 0.113 \times 140 + 0.887 \times 158.4 = 15.82 + 140.5008 = 156.3208
\]
\[
\hat y^{Informer}_{final} = [139.5237,\ 156.3208]
\]

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
