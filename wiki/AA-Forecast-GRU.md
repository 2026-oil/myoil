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
$$
T_t = LOWESS(y_t)
$$

로 표기합니다.

toy residual:
$$
residual_t = y_t - T_t
$$

toy anomaly mask는 설명용으로
$$
critical_t = \mathbb{1}(|residual_t| > threshold)
$$

처럼 둡니다.

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 STAR 분해의 robust anomaly 처리 전체를 이 한 줄로 환원한 것은 설명을 위한 축약입니다.

## 7. Toy sample setup

target toy window:
$$
[100, 101, 102, 120]
$$

toy trend를 단순화해서
$$
T = [100, 101, 102, 103]
$$

라고 두면,
$$
residual = [0, 0, 0, 17]
$$

즉 마지막 시점이 event처럼 보입니다.

toy exogenous `GPRD_THREAT` 는
$$
[10, 12, 14, 35]
$$

로 두고 마지막 시점 burst를 강조합니다.

## 8. Step-by-step hand calculation

### Step 1 — target STAR block (literal)

trend / residual toy 계산:
- trend: `[100, 101, 102, 103]`
- residual: `[0, 0, 0, 17]`
- critical mask: `[0, 0, 0, 1]`

### Step 2 — exogenous STAR block (literal)

`GPRD_THREAT` 와 `BS_Core_Index_A`, `BS_Core_Index_C` 는 star path로 들어갑니다. toy에서는 이 중 하나만 확대해도 “event burst가 star block으로 들어간다”는 계산 흐름을 이해할 수 있습니다.

### Step 3 — encoder feature composition (literal structure)

AA-GRU 입력은 개념적으로 아래 조각의 concat 입니다.

1. raw / transformed target
2. non-star hist exog
3. target STAR outputs
4. star hist exog STAR outputs

즉 baseline보다 “event-aware feature block” 이 추가됩니다.

### Step 4 — GRU learned layer (schematic)

그 다음부터는 GRU backbone이 learned recurrence로 미래 2-step을 만듭니다. teaching용으로 base output을
$$
\hat y^{AA-GRU}_{base} = [140, 146]
$$

라고 적을 수 있지만, 이 숫자 자체는 literal 재현값이 아니라 schematic placeholder 입니다.

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
