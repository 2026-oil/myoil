# AA-Forecast + Informer + Retrieval

## 1. Variant definition

이 페이지는 `aaforecast-informer-ret.yaml` 을 설명합니다. AA-Forecast Informer base path 위에 retrieval memory branch가 붙는 full pipeline 입니다.

## 2. Command / config

```bash
uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret.yaml
```

- main config: `yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret.yaml`
- AA plugin config: `yaml/plugins/aa_forecast/aa_forecast_informer-ret.yaml`
- retrieval detail: `yaml/plugins/retrieval/baseline_retrieval.yaml`

## 3. What is ON / OFF

| Switch | 상태 |
|---|---|
| aa_forecast | On |
| backbone | Informer |
| retrieval | On |
| uncertainty | On |
| retrieval top_k | 1 |

## 4. Pipeline delta vs prior variant

AA-Informer 대비 새로 추가되는 단계는 retrieval memory branch 뿐입니다.

## 5. What is literal vs schematic on this page

### literal
- retrieval memory calculation
- uncertainty-gated blend
- AA decomposition이 retrieval query를 event-aware하게 만든다는 구조

### schematic
- AA-Informer base prediction 그 자체
- attention-based learned layer 내부 계산

## 6. Core formulas used in this variant

\[
\hat y_h^{final} = (1-\lambda_h) \hat y_h^{AA-Informer, base} + \lambda_h \hat y_h^{mem}
\]

이 페이지의 핵심은 baseline-ret와 같은 retrieval 수식을 쓰되, \(\hat y_h^{base}\) 가 plain Informer가 아니라 **AA-Informer base** 라는 점입니다.

## 7. Toy sample setup

- memory prediction:
\[
[145.2, 158.4]
\]
- AA-Informer schematic base output:
\[
[141, 149]
\]
- blend weight:
\[
[0.4435, 0.887]
\]

## 8. Step-by-step hand calculation

### Step 1 — AA base path (schematic)

teaching용 AA-Informer base output:
\[
\hat y^{AA-Informer, base} = [141, 149]
\]

### Step 2 — retrieval memory path (literal)

\[
\hat y^{mem} = [145.2, 158.4]
\]

### Step 3 — final blend (literal)

\[
\hat y^{AA-Informer}_{final} = (1-\lambda) \hat y^{AA-Informer, base} + \lambda \hat y^{mem}
\]

즉
\[
\hat y^{AA-Informer}_{final} = [142.8627, 157.3378]
\]

## 9. Interpretation

- full pipeline 수식 자체는 AA-GRU+Retrieval과 동일합니다.
- 달라지는 것은 AA base path가 Informer attention backbone 위에 있다는 점입니다.
- 즉 retrieval memory는 동일한 방식으로 작동하지만, base prediction의 성격이 다릅니다.

## 10. Grounding notes to repo surfaces

- `aa_forecast_informer-ret.yaml` 는 backbone Informer + retrieval on
- retrieval detail 값은 `baseline_retrieval.yaml` 을 따라갑니다
- uncertainty and decoder params는 plugin YAML에 명시됩니다

## 11. Provenance tags summary

- `repo default`: Informer backbone, retrieval on, uncertainty on
- `toy simplification`: `[141, 149]`, `[145.2, 158.4]`, `[142.8627, 157.3378]`
- `variant-specific override`: AA + Informer + retrieval 동시 활성화

## 관련 페이지

- [AA-Forecast + Informer](AA-Forecast-Informer)
- [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
- [AA-Forecast + GRU + Retrieval](AA-Forecast-GRU-Retrieval)
- [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)

Source: `yaml/experiment/feature_set_aaforecast/aaforecast-informer-ret.yaml:1-24`, `yaml/plugins/aa_forecast/aa_forecast_informer-ret.yaml:1-35`, `yaml/plugins/retrieval/baseline_retrieval.yaml:1-27`, `plugins/aa_forecast/runtime.py:1110-1196`, `plugins/aa_forecast/runtime.py:1439-1496`
