# AA-Forecast 베이스라인 (GRU / Informer)

## 1. Variant definition

이 페이지는 `baseline.yaml` 이 만드는 **command-level baseline** 을 설명합니다. 중요한 점은 이 config가 `yaml/jobs/main/gru_informer.yaml` 을 참조하므로, 하나의 command 안에 **GRU path** 와 **Informer path** 가 함께 들어 있다는 것입니다.

## 2. Command / config

```bash
uv run python main.py --config yaml/experiment/feature_set_aaforecast/baseline.yaml
```

- main config: `yaml/experiment/feature_set_aaforecast/baseline.yaml`
- jobs route: `yaml/jobs/main/gru_informer.yaml`
- models: `GRU`, `Informer`

## 3. What is ON / OFF

| Switch | 상태 |
|---|---|
| retrieval | Off |
| aa_forecast | Off |
| base models | GRU + Informer fan-out |
| shared hist exog | On |

> [!NOTE]
> Provenance: `repo default`
>
> `baseline.yaml` 에는 `retrieval.enabled: false` 가 명시되어 있고, jobs route는 `gru_informer.yaml` 입니다.

## 4. Pipeline delta vs prior variant

이 페이지는 기준점(reference origin)입니다. 이후 모든 페이지는 여기에서 무엇이 더 켜졌는지 설명합니다.

## 5. What is literal vs schematic on this page

### literal
- 최근 `L`개 시점의 target / exogenous window를 자르는 규칙
- 같은 입력 window가 GRU와 Informer 두 모델로 fan-out 되는 사실

### schematic
- GRU 내부 hidden state update
- Informer 내부 attention score 계산
- base forecast $\hat y^{base}$ 를 학습된 모델이 어떻게 만드는지의 세부 weight 계산

## 6. Core formulas used in this variant

baseline에는 retrieval이 없으므로 literal 수식은 주로 window 정의입니다.

$$
Q = [y_{T-L+1}, \dots, y_T]
$$

그리고 각 model은 같은 $Q$ 를 받아 서로 다른 learned function을 적용합니다.

$$
\hat y_h^{GRU} = f_{GRU}(Q, X)
$$

$$
\hat y_h^{Informer} = f_{Informer}(Q, X)
$$

여기서 $X$ 는 같은 기간의 hist exogenous window 입니다.

> [!NOTE]
> Provenance: `toy simplification`
>
> 위 두 식은 structure를 설명하는 schematic 식입니다. 실제 weight를 손으로 계산하겠다는 뜻은 아닙니다.

## 7. Toy sample setup

공통 toy target series:
$$
[100, 101, 102, 120, 132, 126, 107, 110, 121, 132]
$$

toy에서는 `L=4`, `H=2` 로 둡니다.

## 8. Step-by-step hand calculation

### Step 1 — query window 자르기 (literal)

마지막 4개 값만 취하면:

$$
Q = [107, 110, 121, 132]
$$

이건 GRU와 Informer가 **공통으로 받는 baseline input window** 입니다.

### Step 2 — retrieval / AA branch 확인 (literal)

- retrieval = off
- aa_forecast = off

따라서 현재 페이지에서는 memory bank, STAR decomposition, blend stage가 없습니다.

### Step 3 — GRU subsection (schematic)

같은 query window를 GRU가 받아 horizon 2-step base forecast를 낸다고 씁니다.

예시적 표기:

$$
\hat y^{GRU}_{base} = [136, 138]
$$

이 숫자 자체는 **손으로 재현한 repo output이 아니라**, “baseline learned layer가 이렇게 두 값을 낸다”는 teaching용 placeholder 입니다.

> [!NOTE]
> Provenance: `toy simplification`

### Step 4 — Informer subsection (schematic)

같은 query window를 Informer도 받지만, attention 기반 backbone이기 때문에 다른 learned output을 낼 수 있습니다.

예시적 표기:

$$
\hat y^{Informer}_{base} = [135, 140]
$$

### Step 5 — 해석

이 baseline 페이지에서 literal로 손으로 다시 계산할 수 있는 부분은 **입력 window와 실행 분기** 까지입니다. 최종 forecast 숫자는 두 backbone의 learned function이 담당합니다.

## 9. Interpretation

- baseline은 모든 후속 페이지의 출발점입니다.
- retrieval이 꺼져 있기 때문에 “과거 유사 사건을 다시 불러와 섞는” 단계가 없습니다.
- AA-Forecast가 꺼져 있기 때문에 STAR 분해와 event/path/regime 설명도 없습니다.
- 같은 command 안에 GRU/Informer가 둘 다 들어 있다는 점이 이 페이지의 핵심입니다.

## 10. Grounding notes to repo surfaces

- `baseline.yaml` 는 retrieval off
- `gru_informer.yaml` 는 `GRU`, `Informer` 두 모델을 포함
- 따라서 이 페이지는 “single config, dual model fan-out” 을 문서 구조에 직접 반영해야 합니다.

## 11. Provenance tags summary

- `repo default`: baseline config shape, retrieval off, jobs fan-out
- `toy simplification`: $[136, 138]$, $[135, 140]$ 같은 schematic base outputs
- `variant-specific override`: 없음 (기준점 페이지)

## 관련 페이지

- [AA-Forecast 공통 수식·표기·손계산 부록](AA-Forecast-Hand-Calculation-Appendix)
- [AA-Forecast 베이스라인 + Retrieval](AA-Forecast-Baseline-Retrieval)
- [AA-Forecast + GRU](AA-Forecast-GRU)
- [AA-Forecast + Informer](AA-Forecast-Informer)

Source: `yaml/experiment/feature_set_aaforecast/baseline.yaml:1-23`, `yaml/jobs/main/gru_informer.yaml:1-15`
