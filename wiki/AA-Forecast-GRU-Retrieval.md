# AA-Forecast + GRU + Retrieval

## 1. Variant definition

이 페이지는 `aaforecast-gru-ret.yaml` 을 설명합니다. 즉 **AA base path** 와 **retrieval memory blend** 가 둘 다 켜진 full pipeline 입니다.

## 2. Command / config

```bash
uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml
```

- main config: `yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml`
- AA plugin config: `yaml/plugins/aa_forecast/aa_forecast_gru-ret.yaml`
- retrieval detail: `yaml/plugins/retrieval/baseline_retrieval.yaml`

## 3. What is ON / OFF

| Switch | 상태 |
|---|---|
| aa_forecast | On |
| backbone | GRU |
| retrieval | On |
| uncertainty | On |
| retrieval top_k | 1 |

## 4. Pipeline delta vs prior variant

AA-GRU 대비 추가되는 단계는 retrieval memory branch입니다.

1. AA-GRU가 먼저 base prediction을 만듭니다.
2. 같은 fold에서 retrieval query / bank / neighbors를 계산합니다.
3. retrieved future return을 현재 scale에 적용해 memory prediction을 만듭니다.
4. base prediction과 memory prediction을 uncertainty-gated blend로 섞습니다.

## 5. What is literal vs schematic on this page

### literal
- AA branch의 decomposition 구조
- retrieval branch의 future return / memory prediction / blend
- ON/OFF switches

### schematic
- AA-GRU learned base prediction 자체

## 6. Core formulas used in this variant

AA-GRU base output을
\[
\hat y_h^{AA-GRU, base}
\]

retrieval memory output을
\[
\hat y_h^{mem}
\]

라 두면,

\[
\hat y_h^{final} = (1-\lambda_h) \hat y_h^{AA-GRU, base} + \lambda_h \hat y_h^{mem}
\]

즉 full pipeline의 핵심은 “AA가 만든 base prediction” 을 retrieval이 posthoc으로 다시 끌어당긴다는 점입니다.

## 7. Toy sample setup

query window (인덱스 6~9):
\[
Q = [107, 110, 121, 132], \quad y_T = 132
\]

**AA branch** (target STAR decomposition):

| 채널 | window | toy trend | residual | critical mask |
|---|---|---|---|---|
| target | \([107,110,121,132]\) | \([107,110,113,116]\) | \([0,0,8,16]\) | \([0,0,0,1]\) |
| GPRD_THREAT | \([12,14,30,35]\) | \([12,14,16,18]\) | \([0,0,14,17]\) | \([0,0,1,1]\) |

**retrieval branch** (candidate B 선택됨, `top_k=1`):

| 항목 | 값 |
|---|---|
| anchor \(a^{(B)}\) | 110 |
| future | \([121, 132]\) |
| \(r_1^{(B)}\) | \(11/110 = 0.10\) |
| \(r_2^{(B)}\) | \(22/110 = 0.20\) |
| \(y_T\) | 132 |

AA-GRU schematic base output:
\[
\hat y^{AA-GRU, base} = [140, 146]
\]

uncertainty gate:
- mean_similarity = 0.887 (teaching placeholder)
- uncertainty_scale = \([0.5, 1.0]\)
- \(\lambda = [0.4435, 0.887]\)

## 8. Step-by-step hand calculation

### Step 1 — AA decomposition (literal)

target: residual \([0, 0, 8, 16]\), threshold=10, critical mask \([0,0,0,1]\)

GPRD_THREAT: residual \([0, 0, 14, 17]\), threshold=10, critical mask \([0,0,1,1]\)

10채널 event-aware feature block → GRU encoder 입력.

### Step 2 — AA base prediction (schematic)

GRU가 event-aware 10채널 텐서를 처리하여 base prediction을 만듭니다. teaching용 placeholder:
\[
\hat y^{AA-GRU, base} = [140, 146]
\]

(baseline GRU \([136,138]\) 보다 높은 이유: STAR event 신호가 상승 예측을 강화)

### Step 3 — retrieval memory prediction (literal)

candidate B의 future return \(\bar r = [0.10, 0.20]\) 을 현재 scale에 입힙니다:

\[
\hat y_1^{mem} = 132 + |132| \times 0.10 = 132 + 13.2 = 145.2
\]
\[
\hat y_2^{mem} = 132 + |132| \times 0.20 = 132 + 26.4 = 158.4
\]
\[
\hat y^{mem} = [145.2,\ 158.4]
\]

### Step 4 — blend weight 계산 (literal)

\[
\lambda_1 = \min(0.887 \times 0.5,\ 1.0) = 0.4435
\]
\[
\lambda_2 = \min(0.887 \times 1.0,\ 1.0) = 0.887
\]

### Step 5 — final blend 산술 (literal)

h=1:
\[
\hat y_1^{final} = (1 - 0.4435) \times 140 + 0.4435 \times 145.2
= 0.5565 \times 140 + 0.4435 \times 145.2
= 77.91 + 64.3962 = 142.3062
\]

h=2:
\[
\hat y_2^{final} = (1 - 0.887) \times 146 + 0.887 \times 158.4
= 0.113 \times 146 + 0.887 \times 158.4
= 16.498 + 140.5008 = 156.9988
\]
\[
\hat y^{AA-GRU}_{final} = [142.3062,\ 156.9988]
\]

## 9. Interpretation

- retrieval이 없을 때 AA-GRU는 event-aware base path만 갖습니다.
- retrieval이 켜지면 “AA가 만든 base” 와 “과거 유사 사건 memory” 가 동시에 작동합니다.
- 그래서 이 variant는 가장 직관적으로 **내부 event model + 외부 nearest-neighbor memory** 의 결합입니다.

## 10. Grounding notes to repo surfaces

- `aa_forecast_gru-ret.yaml` 는 backbone GRU + retrieval on
- detail retrieval 값은 `baseline_retrieval.yaml` 에서 읽습니다
- AA retrieval path는 `plugins/aa_forecast/runtime.py` 의 retrieval helper들과 대응합니다

## 11. Provenance tags summary

- `repo default`: AA on, GRU backbone, retrieval on, top_k=1
- `toy simplification`: `[140, 146]`, `[145.2, 158.4]`, `[142.3062, 156.9988]`
- `variant-specific override`: AA + retrieval 동시 활성화

## 관련 페이지

- [AA-Forecast + GRU](AA-Forecast-GRU)
- [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
- [AA-Forecast + Informer + Retrieval](AA-Forecast-Informer-Retrieval)
- [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)

Source: `yaml/experiment/feature_set_aaforecast/aaforecast-gru-ret.yaml:1-24`, `yaml/plugins/aa_forecast/aa_forecast_gru-ret.yaml:1-24`, `yaml/plugins/retrieval/baseline_retrieval.yaml:1-27`, `plugins/aa_forecast/runtime.py:1110-1196`, `plugins/aa_forecast/runtime.py:1439-1496`
