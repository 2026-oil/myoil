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

query:
\[
Q = [107, 110, 121, 132]
\]

retrieved return path:
\[
\bar r = [0.10, 0.20]
\]

memory prediction:
\[
\hat y^{mem} = [145.2, 158.4]
\]

AA-GRU schematic base output:
\[
\hat y^{AA-GRU, base} = [140, 146]
\]

uncertainty-gated blend weight:
\[
\lambda = [0.4435, 0.887]
\]

## 8. Step-by-step hand calculation

### Step 1 — AA decomposition (literal structure)

AA-GRU 페이지와 동일하게 target / star exog decomposition을 통해 event-aware feature block을 만듭니다.

### Step 2 — AA base prediction (schematic)

teaching용으로
\[
\hat y^{AA-GRU, base} = [140, 146]
\]

라고 둡니다.

### Step 3 — retrieval memory prediction (literal)

\[
\hat y^{mem} = [145.2, 158.4]
\]

### Step 4 — final blend (literal)

\[
\hat y^{AA-GRU}_{final} = (1-\lambda) \hat y^{AA-GRU, base} + \lambda \hat y^{mem}
\]

따라서
\[
\hat y^{AA-GRU}_{final} = [142.3062, 156.9988]
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
