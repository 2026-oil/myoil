# AA-Forecast 손계산 패키지 허브

이 패키지는 `feature_set_aaforecast` 계열 6개 실행을 **수식 + toy sample + 손계산** 기준으로 읽기 위한 위키 묶음입니다.

핵심 목표는 세 가지입니다.

1. 어떤 command/config가 어떤 파이프라인을 켜는지 한눈에 보이게 하기
2. **literal로 손계산 가능한 층**과 **learned model이라 schematic으로만 설명해야 하는 층**을 분리하기
3. GRU / Informer / Retrieval / AA-Forecast가 어디서 갈라지는지 비교 가능하게 만들기

## 먼저 알아둘 규칙

- 이 묶음은 코드 투어 문서가 아닙니다.
- 모든 페이지는 같은 toy series와 같은 표기법을 재사용합니다.
- 모든 주요 설명 블록에는 provenance tag가 붙습니다.
  - `repo default`
  - `toy simplification`
  - `variant-specific override`

## 독서 순서

1. [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)
2. [AA-Forecast 베이스라인 (GRU / Informer)](AA-Forecast-Baseline-GRU-Informer)
3. [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
4. [AA-Forecast + GRU](AA-Forecast-GRU)
5. [AA-Forecast + Informer](AA-Forecast-Informer)
6. [AA-Forecast + GRU + Retrieval](AA-Forecast-GRU-Retrieval)
7. [AA-Forecast + Informer + Retrieval](AA-Forecast-Informer-Retrieval)

## 비교표

| 페이지 | command / config | Base model | Retrieval | AA-Forecast | 손계산 중심 포인트 |
|---|---|---|---|---|---|
| Baseline | `baseline.yaml` | GRU + Informer | Off | Off | window selection, 동일 입력이 두 backbone으로 fan-out |
| Baseline + Retrieval | `baseline-ret.yaml` | GRU + Informer | On | Off | query/bank, similarity, future return, blend |
| AA + GRU | `aaforecast-gru.yaml` | GRU | Off | On | STAR 분해, AA feature composition, opaque GRU learned layer |
| AA + Informer | `aaforecast-informer.yaml` | Informer | Off | On | STAR 분해, AA feature composition, opaque Informer learned layer |
| AA + GRU + Retrieval | `aaforecast-gru-ret.yaml` | GRU | On | On | AA base prediction + retrieval memory blend |
| AA + Informer + Retrieval | `aaforecast-informer-ret.yaml` | Informer | On | On | AA base prediction + retrieval memory blend |

## 무엇이 literal이고 무엇이 schematic인가

### literal로 설명하는 층
- 마지막 `input_size` window를 자르는 규칙
- anchor와 future return 계산
- retrieval similarity filtering / top-k / softmax weighting
- blend equation
- ON/OFF toggle이 파이프라인에 미치는 변화

### schematic으로 설명하는 층
- GRU hidden state update 자체
- Informer attention 내부 score 계산 전체
- AA-Forecast latent event/path/regime 표현의 내부 weight 계산

즉 이 패키지는 **“어떤 값을 정확히 손으로 다시 계산할 수 있는가”** 와 **“여기부터는 학습된 모델이 계산한 출력으로 받아들여야 하는가”** 를 명시적으로 나눕니다.

## baseline command가 왜 두 model subsection을 가지는가

`baseline.yaml` 과 `baseline-ret.yaml` 는 둘 다 `yaml/jobs/main/gru_informer.yaml` 을 참조합니다. 즉 command는 하나지만 실제 실행 view는 **GRU path** 와 **Informer path** 두 갈래입니다. 이 때문에 baseline 계열 페이지는 command-level page이면서 동시에 model subsection 두 개를 가집니다.

## 공통 toy series

이 패키지는 아래 toy target series를 재사용합니다.

$$
[100, 101, 102, 120, 132, 126, 107, 110, 121, 132]
$$

공통 가정:
- `L = 4`
- `H = 2`
- query window는 마지막 4개 값
- retrieval 예시는 이해를 위해 작은 window로 축소

> [!NOTE]
> Provenance: `toy simplification`
>
> 실제 repo default에서는 `input_size=64`, `recency_gap_steps=8`, `event_score_threshold=400.0` 같은 값이 등장합니다. 본 패키지의 toy는 계산 경로를 이해하기 위한 축소판입니다.

## 패키지 내부 링크 지도

- 허브는 비교와 길찾기만 담당합니다.
- 공통 수식과 표기법은 [부록](AA-Forecast-Hand-Calculation-Appendix) 이 single source of truth 입니다.
- 상세 페이지는 “이전 variant 대비 무엇이 달라졌는가”에 집중합니다.

## 소스 앵커

Source: `yaml/experiment/feature_set_aaforecast/baseline.yaml:1-23`, `yaml/experiment/feature_set_aaforecast/baseline-ret.yaml:1-24`, `yaml/jobs/main/gru_informer.yaml:1-15`, `yaml/plugins/aa_forecast/aa_forecast_gru.yaml:1-32`, `yaml/plugins/aa_forecast/aa_forecast_informer.yaml:1-34`, `yaml/plugins/retrieval/baseline_retrieval.yaml:1-27`

## 관련 페이지

- [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)
- [AA-Forecast 베이스라인 (GRU / Informer)](AA-Forecast-Baseline-GRU-Informer)
- [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
