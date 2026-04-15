# AA-Forecast + Informer

## 1. Variant definition

이 페이지는 `aaforecast-informer.yaml` 을 설명합니다. AA-Forecast stage는 켜져 있지만 retrieval은 꺼져 있고, backbone만 GRU 대신 Informer 입니다.

## 2. Command / config

```bash
uv run python main.py --config yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml
```

- main config: `yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml`
- plugin config: `yaml/plugins/aa_forecast/aa_forecast_informer.yaml`

## 3. What is ON / OFF

| Switch | 상태 |
|---|---|
| retrieval | Off |
| aa_forecast | On |
| backbone | Informer |
| uncertainty | On (`sample_count=50`) |
| decoder_hidden_size | 192 |
| season_length | 4 |

## 4. Pipeline delta vs prior variant

AA-GRU 대비 바뀌는 것은 backbone입니다.

1. target / star exog decomposition은 유지됩니다.
2. event-aware feature composition도 유지됩니다.
3. 단, forecast를 만드는 learned layer가 recurrent GRU가 아니라 attention 기반 Informer 입니다.

## 5. What is literal vs schematic on this page

### literal
- 어떤 입력 조각이 AA block을 만드는가
- STAR decomposition을 통해 어떤 event 신호가 생기는가
- retrieval이 off 라는 사실

### schematic
- attention head score / sparse attention 내부 계산
- decoder path의 learned weight 계산 전체

## 6. Core formulas used in this variant

Informer 페이지도 literal 핵심은 decomposition 쪽입니다.

\[
T_t = LOWESS(y_t)
\]

\[
residual_t = y_t - T_t
\]

\[
critical_t = \mathbb{1}(|residual_t| > threshold)
\]

그리고 그 결과가 event-aware feature block으로 encoder/decoder conditioning 에 들어갑니다.

## 7. Toy sample setup

AA-GRU 페이지와 동일한 query window를 사용합니다 (마지막 4개, 인덱스 6~9):

| 채널 | window | toy trend \(T\) | residual | toy threshold | critical mask |
|---|---|---|---|---|---|
| target | \([107,110,121,132]\) | \([107,110,113,116]\) | \([0,0,8,16]\) | 10 | \([0,0,0,1]\) |
| GPRD_THREAT | \([12,14,30,35]\) | \([12,14,16,18]\) | \([0,0,14,17]\) | 10 | \([0,0,1,1]\) |
| BS_Core_Index_A | \([0.2,0.3,1.4,1.6]\) | \([0.2,0.3,0.4,0.5]\) | \([0,0,1.0,1.1]\) | 0.5 | \([0,0,1,1]\) |

decomposition 결과는 GRU 페이지와 동일하며, 달라지는 것은 이 feature block을 처리하는 backbone 입니다.

## 8. Step-by-step hand calculation

### Step 1 — decomposition (literal)

target 채널:
- residual 마지막 값 = 16, threshold=10 → critical=1
- event burst 감지: 마지막 시점 target 급등

GPRD_THREAT 채널:
- residual 마지막 두 값 = [14, 17], threshold=10 → critical=[1,1]
- geopolitical event burst 감지

BS_Core_Index_A 채널:
- residual 마지막 두 값 = [1.0, 1.1], threshold=0.5 → critical=[1,1]
- 리스크 지수 급등 감지

### Step 2 — feature composition (literal structure)

Informer도 baseline보다 더 많은 event-aware feature 조각을 입력받습니다. 즉 “무엇을 보고 예측하는가” 는 baseline Informer보다 richer 합니다.

### Step 3 — Informer encoder-decoder (schematic)

10채널 텐서가 Informer encoder를 통과합니다:

1. **encoder** (2 layers): ProbSparse self-attention으로 전체 L=4 시점의 컨텍스트 벡터를 만듭니다
2. **decoder**: `season_length=4` seasonal prior + encoder 컨텍스트로 H=2 예측 생성

teaching용 base output:
\[
\hat y^{AA-Informer}_{base} = [141, 149]
\]

GRU의 \([140, 146]\) 과 다소 다른 이유는 attention 기반 backbone이 long-range 패턴을 다르게 가중하기 때문입니다.

> [!NOTE]
> Provenance: `toy simplification`

## 9. Interpretation

- AA-Informer는 retrieval uplift 없이도 AA decomposition 덕분에 baseline Informer와 다른 forecast path를 갖습니다.
- literal hand calculation은 decomposition / feature composition / branch ON/OFF 까지입니다.
- 최종 forecast 값은 Informer learned layer가 내는 schematic output으로 받아들여야 합니다.

## 10. Grounding notes to repo surfaces

- `aa_forecast_informer.yaml` 는 `model: informer`
- retrieval off
- uncertainty on, dropout candidates 명시
- backbone params는 `hidden_size=128`, `n_head=8`, `encoder_layers=2`, `decoder_hidden_size=192`

## 11. Provenance tags summary

- `repo default`: Informer backbone, uncertainty on, retrieval off
- `toy simplification`: trend/residual example, `[141, 149]`
- `variant-specific override`: backbone이 Informer로 바뀜

## 관련 페이지

- [AA-Forecast 베이스라인 (GRU / Informer)](AA-Forecast-Baseline-GRU-Informer)
- [AA-Forecast + GRU](AA-Forecast-GRU)
- [AA-Forecast + Informer + Retrieval](AA-Forecast-Informer-Retrieval)
- [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)

Source: `yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml:1-26`, `yaml/plugins/aa_forecast/aa_forecast_informer.yaml:1-34`
