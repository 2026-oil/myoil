# AA-Forecast + GRU

## 1. Variant definition

이 페이지는 `aaforecast-gru.yaml` 을 설명합니다. baseline 대비 retrieval은 꺼져 있고, 대신 **AA-Forecast stage** 가 켜져서 target과 일부 exogenous를 STAR/LOWESS 기반으로 분해한 뒤 GRU backbone으로 보냅니다.

## 2. Command / config

```bash
uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml
```

- main config: `yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml`
- plugin config: `yaml/plugins/aa_forecast/aa_forecast_gru.yaml`

## 3. What is ON / OFF

| Switch | 상태 |
|---|---|
| retrieval | Off |
| aa_forecast | On |
| backbone | GRU |
| uncertainty | On (`sample_count=50`) |
| star_anomaly_tails.upward | `GPRD_THREAT`, `BS_Core_Index_A`, `BS_Core_Index_C` |

## 4. Pipeline delta vs prior variant

baseline 대비 바뀌는 것은 다음입니다.

1. target과 selected star exog에 LOWESS + STAR decomposition이 들어갑니다.
2. non-star exog는 raw/diff history feature로 남습니다.
3. AA feature block이 GRU backbone 입력에 concat 됩니다.
4. retrieval 없이 AA base prediction만 최종 output이 됩니다.

## 5. What is literal vs schematic on this page

### literal
- 어떤 series가 star path에 들어가는가
- `lowess_frac`, `lowess_delta`, `thresh`
- toy residual / anomaly mask 계산
- feature block이 raw/non-star/star 조각으로 나뉜다는 구조

### schematic
- GRU hidden state update
- AA internal event/path/regime latent calculation 전체
- uncertainty 반복 sampling의 learned output 값 그 자체

## 6. Core formulas used in this variant

LOWESS trend를 toy에선
\[
T_t = LOWESS(y_t)
\]

로 표기합니다.

toy residual:
\[
residual_t = y_t - T_t
\]

toy anomaly mask는 설명용으로
\[
critical_t = \mathbb{1}(|residual_t| > threshold)
\]

처럼 둡니다.

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 STAR 분해의 robust anomaly 처리 전체를 이 한 줄로 환원한 것은 설명을 위한 축약입니다.

## 7. Toy sample setup

toy에서는 전체 10개 시점 시리즈의 마지막 4개를 query window로 사용합니다.

target query window (인덱스 6~9):
\[
Q_{target} = [107, 110, 121, 132]
\]

exogenous query windows:
\[
GPRD\_THREAT = [12, 14, 30, 35]
\]
\[
BS\_Core\_Index\_A = [0.2, 0.3, 1.4, 1.6]
\]

toy trend 단순화:

| 채널 | window | toy trend \(T\) | residual \(= window - T\) |
|---|---|---|---|
| target | \([107, 110, 121, 132]\) | \([107, 110, 113, 116]\) | \([0, 0, 8, 16]\) |
| GPRD_THREAT | \([12, 14, 30, 35]\) | \([12, 14, 16, 18]\) | \([0, 0, 14, 17]\) |
| BS_Core_Index_A | \([0.2, 0.3, 1.4, 1.6]\) | \([0.2, 0.3, 0.4, 0.5]\) | \([0.0, 0.0, 1.0, 1.1]\) |

toy threshold (각 채널):

| 채널 | threshold (toy) |
|---|---|
| target | 10 |
| GPRD_THREAT | 10 |
| BS_Core_Index_A | 0.5 |

## 8. Step-by-step hand calculation

### Step 1 — target STAR block (literal)

target query window \([107, 110, 121, 132]\) 에 대한 toy trend/residual 계산:

- **trend**: \(T = [107, 110, 113, 116]\) (linear, step=3)
- **residual**: \(residual = [0, 0, 8, 16]\)
- threshold = 10
- **critical mask**: \([0, 0, 0, 1]\) (마지막 시점만 \(|16| > 10\) 이므로 critical)

event burst: 마지막 시점의 residual 16은 trend를 크게 벗어나는 상승 사건입니다.

### Step 2 — exogenous STAR block (literal)

`GPRD_THREAT` 와 `BS_Core_Index_A`, `BS_Core_Index_C` 는 star path로 들어갑니다. toy에서는 이 중 하나만 확대해도 “event burst가 star block으로 들어간다”는 계산 흐름을 이해할 수 있습니다.

### Step 3 — encoder feature composition (literal structure)

AA-GRU 입력은 개념적으로 아래 조각의 concat 입니다.

1. raw / transformed target
2. non-star hist exog
3. target STAR outputs
4. star hist exog STAR outputs

즉 baseline보다 “event-aware feature block” 이 추가됩니다.

### Step 4 — event score 요약 (literal)

위 STAR decomposition 결과로부터 event_score 를 구합니다:

- count_active (마지막 시점): target(1) + GPRD(1) + BS(1) = 3
- channel_activity 합산 (마지막 시점): |16| + |17| + |1.1| = 34.1

\[
event\_score \approx 3 + 34.1 = 37.1 \quad \text{(toy placeholder)}
\]

이 event_score 가 threshold를 넘으면 AA retrieval이 event_key 로 neighbor를 찾습니다 (retrieval이 on 일 때).

### Step 5 — GRU learned layer (schematic)

10채널 event-aware 텐서를 GRU가 받아 learned recurrence 로 미래 2-step을 만듭니다. teaching용 base output:
\[
\hat y^{AA-GRU}_{base} = [140, 146]
\]

이 숫자 자체는 literal 재현값이 아니라 schematic placeholder 입니다. baseline GRU \([136, 138]\) 보다 높은 이유는 event-aware feature 덕분에 모델이 더 강한 상승 신호를 포착했기 때문이라고 직관적으로 이해합니다.

## 9. Interpretation

- AA-GRU는 retrieval 없이도 event-aware feature composition만으로 base forecast를 바꿉니다.
- 따라서 이 페이지의 핵심은 “memory bank를 더한다”가 아니라 “입력 feature를 event-aware하게 바꾼다” 입니다.
- literal로 따라갈 수 있는 곳은 decomposition과 feature composition까지이고, 이후 GRU forecast는 schematic 영역입니다.

## 10. Grounding notes to repo surfaces

- `aa_forecast_gru.yaml` 는 `model: gru`
- retrieval은 off
- uncertainty는 on, `sample_count=50`
- star anomaly tails는 upward 세 변수로 묶입니다

## 11. Provenance tags summary

- `repo default`: GRU backbone, uncertainty on, star tails
- `toy simplification`: trend/residual/anomaly 계산 예시, `[140, 146]`
- `variant-specific override`: AA-Forecast on, retrieval off

## 관련 페이지

- [AA-Forecast 베이스라인 (GRU / Informer)](AA-Forecast-Baseline-GRU-Informer)
- [AA-Forecast + Informer](AA-Forecast-Informer)
- [AA-Forecast + GRU + Retrieval](AA-Forecast-GRU-Retrieval)
- [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)

Source: `yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml:1-26`, `yaml/plugins/aa_forecast/aa_forecast_gru.yaml:1-32`
