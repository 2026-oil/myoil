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

query window (인덱스 6~9):
\[
Q = [107, 110, 121, 132], \quad y_T = 132
\]

**AA branch** (target STAR decomposition, AA-Informer 페이지와 동일):

| 채널 | window | toy trend | residual | critical mask |
|---|---|---|---|---|
| target | \([107,110,121,132]\) | \([107,110,113,116]\) | \([0,0,8,16]\) | \([0,0,0,1]\) |
| GPRD_THREAT | \([12,14,30,35]\) | \([12,14,16,18]\) | \([0,0,14,17]\) | \([0,0,1,1]\) |
| BS_Core_Index_A | \([0.2,0.3,1.4,1.6]\) | \([0.2,0.3,0.4,0.5]\) | \([0,0,1.0,1.1]\) | \([0,0,1,1]\) |

**retrieval branch** (AA-GRU+Retrieval 과 동일한 candidate B):

| 항목 | 값 |
|---|---|
| anchor \(a^{(B)}\) | 110 |
| \(r^{(B)}\) | \([0.10, 0.20]\) |
| memory prediction | \([145.2, 158.4]\) |

AA-Informer schematic base output:
\[
\hat y^{AA-Informer, base} = [141, 149]
\]

blend weight:
\[
\lambda = [0.4435, 0.887]
\]

## 8. Step-by-step hand calculation

### Step 1 — AA decomposition + Informer base prediction (schematic)

STAR decomposition: target critical mask \([0,0,0,1]\), GPRD critical mask \([0,0,1,1]\).

10채널 event-aware 텐서 → Informer encoder (2 layers, ProbSparse attention) → decoder (season_length=4).

Teaching용 AA-Informer base output:
\[
\hat y^{AA-Informer, base} = [141, 149]
\]

(AA-GRU의 \([140,146]\) 보다 h=2에서 더 높은 이유: Informer attention이 GPRD 두 시점 burst를 더 강하게 반영)

### Step 2 — retrieval memory path (literal)

candidate B anchor \(a^{(B)}=110\), future \([121, 132]\):

\[
\hat y_1^{mem} = 132 + 132 \times 0.10 = 145.2
\]
\[
\hat y_2^{mem} = 132 + 132 \times 0.20 = 158.4
\]
\[
\hat y^{mem} = [145.2,\ 158.4]
\]

### Step 3 — blend weight 계산 (literal)

\[
\lambda_1 = \min(0.887 \times 0.5,\ 1.0) = 0.4435
\]
\[
\lambda_2 = \min(0.887 \times 1.0,\ 1.0) = 0.887
\]

### Step 4 — final blend 산술 (literal)

h=1:
\[
\hat y_1^{final} = (1 - 0.4435) \times 141 + 0.4435 \times 145.2
= 0.5565 \times 141 + 0.4435 \times 145.2
= 78.4665 + 64.3962 = 142.8627
\]

h=2:
\[
\hat y_2^{final} = (1 - 0.887) \times 149 + 0.887 \times 158.4
= 0.113 \times 149 + 0.887 \times 158.4
= 16.837 + 140.5008 = 157.3378
\]
\[
\hat y^{AA-Informer}_{final} = [142.8627,\ 157.3378]
\]

**비교**: GRU+Retrieval의 \([142.3062, 156.9988]\) 대비 h=1에서 0.5562, h=2에서 0.3390 더 높습니다. AA base prediction의 차이(141 vs 140, 149 vs 146)가 최종 blend에 그대로 반영됩니다.

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
