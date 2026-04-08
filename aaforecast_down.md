# AAForecast 하방 편향 분석 보고서

## 1. 결론 요약

`runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast`의 예측이 **전 구간에서 항상 하방으로 깔리는 문제는 아니고**, **2026-01-12 cutoff 이후부터 급격히 하방 편향으로 전환**된다. 가장 중요한 결론은 아래 5가지다.

1. **1차 원인은 late-regime 급등 자체를 학습 구간이 충분히 보여주지 못한 것**이다.  
   - 2026-02-09 cutoff 기준 train tail의 주간 변화폭은 대체로 `-1.617 ~ +2.846` 수준인데, 실제 미래는 `+2.412`, `+13.919`, `+12.243`으로 급가속한다.
   - 즉, 마지막 fold의 3~4 step 급등은 train window 안에서 사실상 등장하지 않는 패턴이다.

2. **하지만 데이터 난이도만의 문제는 아니다. AAForecast 경로가 parity GRU보다 일관되게 더 낮게 예측한다.**  
   - AAForecast는 **16/16 forecast point 전부에서 GRU보다 낮은 예측값**을 냈다.
   - 전체 평균 오차는 `AAForecast = -4.673`, `GRU = +0.388`이다.
   - 즉, 하방 편향은 “공통 데이터 난이도” 위에 **AAForecast 전용 downward shift**가 추가된 형태다.

3. **AAForecast의 STAR 분해는 이 run에서 seasonal period를 명시적으로 받지 못해 모델 기본값 `season_length=12`로 동작했을 가능성이 높다.**  
   - run config / plugin config / params override 어디에도 `season_length`가 없다.
   - `neuralforecast/models/aaforecast/model.py`와 `plugins/aa_forecast/modules.py`의 기본값은 `12`다.
   - 주간 Brent 시계열에 12-step seasonal assumption을 넣으면 trend/seasonal/residual/anomaly 분해가 왜곡될 수 있고, 그 왜곡은 AAForecast 쪽에만 존재한다.

4. **AAForecast의 anomaly-context는 최근 상승장의 “직전 패턴”을 충분히 잡지 못한다.**  
   - 마지막 cutoff(2026-02-09)의 최근 96-step 중 active anomaly는 8개뿐이고, 최근 20-step에서는 3개뿐이다.
   - 최근 active dates는 `2026-01-05`, `2026-01-12`, `2026-01-26` 정도이며, 실제 폭등 직전/직후 구조를 충분히 대표하지 못한다.
   - attention이 sparse anomaly에만 의존하는 구조라, 급등 직전의 충분한 상승 문맥이 없으면 보수적(낮은) 출력으로 수렴하기 쉽다.

5. **uncertainty 경로는 사실상 붕괴되어 있고, 상방 보정을 전혀 제공하지 못했다.**  
   - 모든 cutoff / 모든 horizon에서 `selected_std_by_horizon = 0.0`, `selected_dropout_by_horizon = [0.1, 0.1, 0.1, 0.1]`이다.
   - candidate별 mean은 달라지지만, 같은 dropout 내부 sample std는 모두 0이다.
   - 즉, uncertainty stage는 상방 tail 리스크를 반영하지 못했고, 결과적으로 point forecast를 보수적으로 유지시켰다.

---

## 2. 이번 분석에서 확인한 파일

### Run artifacts
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/cv/AAForecast_forecasts.csv`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/cv/AAForecast_metrics_by_cutoff.csv`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/cv/AAForecast_rolling_origin_metrics.csv`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/aa_forecast/context/*.csv`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/aa_forecast/context/*.json`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/aa_forecast/uncertainty/*.json`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/aa_forecast/uncertainty/*.csv`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/config/config.resolved.json`
- `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast/models/AAForecast/folds/*/loss_curve_every_10_global_steps.csv`

### Parity comparison
- `runs/feature_set_aaforecast_brentoil_case1_parity_gru/cv/GRU_forecasts.csv`
- `runs/feature_set_aaforecast_brentoil_case1_parity_gru/cv/GRU_rolling_origin_metrics.csv`
- `runs/feature_set_aaforecast_brentoil_case1_parity_gru/config/config.resolved.json`
- `runs/feature_set_aaforecast_brentoil_case1_parity_gru/models/GRU/folds/*/loss_curve_every_10_global_steps.csv`

### Code / config surfaces
- `yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml`
- `yaml/plugins/aa_forecast_brentoil_case1_parity.yaml`
- `neuralforecast/models/aaforecast/model.py`
- `neuralforecast/models/aaforecast/gru.py`
- `plugins/aa_forecast/modules.py`
- `plugins/aa_forecast/runtime.py`
- `runtime_support/forecast_models.py`
- `tests/test_gru_aaforecast_parity.py`

---

## 3. 관측된 현상: “항상 하방”이 아니라, 후반 fold에서 하방으로 무너진다

### 3-1. fold별 실제값 vs AAForecast 예측

| cutoff | actual mean | AAForecast mean | mean error |
|---|---:|---:|---:|
| 2025-11-17 | 61.932 | 65.641 | +3.709 |
| 2025-12-15 | 62.204 | 64.152 | +1.948 |
| 2026-01-12 | 67.275 | 62.241 | -5.033 |
| 2026-02-09 | 82.142 | 62.825 | -19.317 |

핵심은 다음이다.
- **초기 2개 fold는 오히려 상방 예측**이다.
- **하방 편향은 2026-01-12 cutoff부터 시작**된다.
- **2026-02-09 cutoff에서 편향이 폭발적으로 커진다.**

특히 마지막 fold의 실제/예측은 아래와 같다.

- actual: `[70.313, 72.724, 86.644, 98.887]`
- AAForecast: `[61.285, 67.393, 59.910, 62.712]`
- error: `[-9.028, -5.331, -26.734, -36.175]`

즉, 문제는 단순한 noise가 아니라 **late horizon에서 level 자체를 못 따라가는 현상**이다.

---

## 4. parity GRU와 비교했을 때 무엇이 다른가

### 4-1. AAForecast는 모든 시점에서 GRU보다 더 낮다

전체 16개 예측 포인트를 합치면:
- `AAForecast mean error = -4.673`
- `GRU mean error = +0.388`
- `AAForecast mean abs error = 7.643`
- `GRU mean abs error = 7.457`
- **AAForecast prediction < GRU prediction: 16 / 16 points**

cutoff별 평균 차이(`AAForecast - GRU`)도 전부 음수다.

| cutoff | AAForecast mean | GRU mean | AAForecast - GRU |
|---|---:|---:|---:|
| 2025-11-17 | 65.641 | 69.172 | -3.530 |
| 2025-12-15 | 64.152 | 68.421 | -4.269 |
| 2026-01-12 | 62.241 | 69.078 | -6.837 |
| 2026-02-09 | 62.825 | 68.435 | -5.610 |

이건 매우 중요하다. 같은 dataset / 같은 training regime / 같은 parity hyperparameter를 쓰는데도, **AAForecast 추가 경로(STAR decomposition + anomaly sparse attention + uncertainty replacement)가 일관되게 level을 아래로 당긴다**는 뜻이다.

### 4-2. 데이터 난이도는 공통이지만, 하방 방향성은 AAForecast가 더 강하다

마지막 cutoff(2026-02-09)에서:
- GRU error: `[-7.075, -2.220, -16.435, -29.099]`
- AAForecast error: `[-9.028, -5.331, -26.734, -36.175]`

즉,
- **late-regime 급등을 못 맞히는 문제는 두 모델 공통**이고,
- **그 급등 구간에서 추가로 더 아래로 미는 건 AAForecast 특유의 현상**이다.

---

## 5. 원인 1: 마지막 fold의 급등은 train tail이 보여준 패턴보다 훨씬 가파르다

2026-02-09 cutoff 기준 train tail(최근 12주)은 아래와 같다.

- `2025-11-24`: 62.659
- `2025-12-01`: 63.277
- `2025-12-08`: 61.660
- `2025-12-15`: 60.133
- `2025-12-22`: 61.523
- `2025-12-29`: 60.966
- `2026-01-05`: 62.072
- `2026-01-12`: 64.255
- `2026-01-19`: 65.131
- `2026-01-26`: 67.978
- `2026-02-02`: 67.606
- `2026-02-09`: 68.384

train에서 주간 증가폭은 대략:
- `+0.618`, `-1.617`, `-1.527`, `+1.390`, `-0.558`, `+1.107`, `+2.183`, `+0.876`, `+2.846`, `-0.372`, `+0.778`

반면 미래 4 step은:
- values: `[70.313, 72.724, 86.644, 98.887]`
- diffs: `[+2.412, +13.919, +12.243]`

즉, 마지막 2개 horizon의 점프는 **최근 입력 구간에서 거의 보이지 않는 acceleration**이다.

이 때문에 모델이 “평균적 continuation”을 택하면 자연스럽게 낮게 예측하게 된다. 실제로 AAForecast의 마지막 cutoff 예측은 `[61.285, 67.393, 59.910, 62.712]`로 **68 전후 plateau에 머무르며 폭등 구간으로 진입하지 못한다.**

이 부분은 **가장 강하게 확인된 1차 원인**이다.

---

## 6. 원인 2: AAForecast 경로는 parity GRU 대비 구조적으로 더 낮은 level을 만든다

AAForecast는 parity GRU 위에 다음을 추가한다.

1. `STARFeatureExtractor`를 통해 target/hist_exog를 `trend`, `seasonal`, `anomalies`, `residual`로 분해
2. anomaly 기반 `critical_mask` 생성
3. `CriticalSparseAttention`으로 anomaly timestep 중심의 context 주입
4. uncertainty 단계에서 dropout 후보들 중 선택된 mean으로 최종 point forecast 대체

관련 코드는 다음에 있다.
- `neuralforecast/models/aaforecast/model.py`
- `plugins/aa_forecast/modules.py`
- `plugins/aa_forecast/runtime.py`

특히 `forward()`를 보면 AAForecast는 raw `insample_y`와 `hist_exog` 외에,
- `target_trend`
- `target_seasonal`
- `target_anomalies`
- `target_residual`
- STAR hist exog의 decomposition 결과들
을 encoder input에 추가한다.

그리고 attention은 `critical_mask`가 가리키는 sparse timestep을 중심으로 작동한다.

이 구조 자체가 나쁘다는 뜻은 아니지만, **이번 parity run에서는 결과적으로 모든 forecast point에서 GRU보다 낮은 값을 내므로**, 실제 down-bias의 differential source는 AAForecast 추가 경로라고 보는 게 맞다.

즉, **“데이터가 어려웠다”는 설명만으로는 부족하고, “AAForecast가 그 어려운 상황에서 더 보수적으로 반응했다”가 정확한 해석**이다.

---

## 7. 원인 3: 이 run에서 season_length가 명시되지 않아, STAR가 기본값 12로 동작했을 가능성이 매우 높다

이건 이번 분석에서 확인한 **가장 구체적인 구조적 리스크**다.

### 7-1. run/config 어디에도 `season_length`가 없다

다음 파일들에서 `season_length`를 확인했지만, run에 반영된 흔적이 없다.
- `runs/.../config/config.resolved.json`
- `runs/.../aa_forecast/config/stage_config.json`
- `yaml/plugins/aa_forecast_brentoil_case1_parity.yaml`
- `yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml`
- `plugins/aa_forecast/runtime.py::_aa_params_override`

즉 runtime이 AAForecast에 override로 넣는 값은
- `top_k`
- STAR exog grouping
- tail modes
- `lowess_frac`
- `lowess_delta`
- uncertainty params
- `scaler_type`
정도이고, **`season_length`는 포함되지 않는다.**

### 7-2. 코드 기본값은 12다

- `neuralforecast/models/aaforecast/model.py` → `season_length: int = 12`
- `plugins/aa_forecast/modules.py` → `season_length: int = 12`

따라서 이 run의 AAForecast는 별도 주입이 없으면 **12-step seasonal decomposition**으로 동작한다.

### 7-3. 왜 이게 하방 편향을 키울 수 있나

이번 데이터는 weekly Brent 시계열이다. 이때 12-step seasonality는 다음 문제를 만들 수 있다.
- trend 추정이 실제 구조보다 과도하게 smooth해질 수 있음
- seasonal 평균이 최근 레짐 변화보다 더 짧은 반복성을 강제할 수 있음
- residual / anomaly 분해가 “상승 레짐 진입”을 구조 변화가 아니라 residual noise처럼 다룰 수 있음
- 결국 attention이 보는 critical timestep의 질이 나빠짐

이 항목은 **단독으로 전부를 설명하는 확정 원인이라기보다**, AAForecast 쪽으로만 존재하는 **강한 구조적 amplifier**로 보는 것이 정확하다.

---

## 8. 원인 4: anomaly-context가 sparse하고, 최근 급등 직전 패턴을 충분히 대표하지 못한다

AAForecast는 `critical_mask`에 기반해 attention을 건다. 즉, **어떤 시점을 anomaly로 뽑느냐**가 예측 방향에 큰 영향을 준다.

마지막 cutoff(`2026-02-09`)의 context artifact를 보면:
- full context points: `580`
- active points: `178`
- 최근 96-step 중 active anomaly: `8`
- 최근 20-step 중 active anomaly: `3`

최근 96-step에서 active로 잡힌 날짜는:
- `2024-04-15`
- `2025-04-07`
- `2025-06-09`
- `2025-06-16`
- `2025-06-23`
- `2026-01-05`
- `2026-01-12`
- `2026-01-26`

여기서 보이는 문제는 두 가지다.

1. **최근 상승장 직전 구간에서 active anomaly가 아주 촘촘하지 않다.**  
   - 마지막 20-step에 anomaly가 3개뿐이다.

2. **선택된 anomaly 중 상당수는 훨씬 오래된 시점이다.**  
   - 최근 상승장의 local acceleration을 설명하기보다, 과거 다른 레짐의 event를 섞어 attention하게 된다.

AAForecast는 `top_k=0.05`로 anomaly를 고른다. 즉 최근 window에서 선택되는 critical context가 매우 sparse하다. 급등 같은 **연속적 regime transition**보다 **몇 개의 isolated event**에 더 민감한 구조라, 이 케이스에서는 상승장 continuation보다 보수적 continuation을 택하기 쉬웠다.

---

## 9. 원인 5: uncertainty 경로가 실제로는 작동하지 않아 상방 리스크를 반영하지 못했다

uncertainty artifacts를 보면, 모든 cutoff와 모든 horizon에서:
- `selected_std_by_horizon = [0.0, 0.0, 0.0, 0.0]`
- `selected_dropout_by_horizon = [0.1, 0.1, 0.1, 0.1]`

또한 candidate stats를 보면 dropout별 prediction mean은 달라지는데,
- 같은 dropout 내부 sample std는 항상 `0.0`
- 그래서 selection은 항상 첫 후보인 `0.1`로 고정된다.

이건 의미상 다음과 같다.
- uncertainty head는 **실질적인 forecast distribution**을 제공하지 못했다.
- point forecast replacement는 사실상 **단일 deterministic path**에 가깝다.
- 급등 구간에서 upper-tail 가능성을 남겨두지 못하고, 평균적인 보수 예측을 그대로 내보낸다.

이건 base model의 down-bias를 만들어낸 1차 원인이라기보다, **하방 편향을 완화할 기회가 있었는데 그 경로가 전혀 기능하지 못한 상태**라고 보는 게 맞다.

---

## 10. loss curve / training dynamics 관찰

AAForecast의 fold별 마지막 validation loss는 대략 아래 수준이었다.
- fold 0: `val_loss 17.77 ~ 19.90`
- fold 1: `28.60 ~ 28.71`
- fold 2: `50.45 ~ 57.79`
- fold 3: `41.28 ~ 45.05`

GRU는 대응 fold에서 대체로 더 높은 validation loss를 보이는 구간도 있다.
즉,
- **AAForecast가 단순히 optimization failure로 망가진 것은 아니다.**
- in-fold validation 기준으로는 GRU보다 더 “좋아 보이는” fold도 있다.
- 그런데 실제 미래 급등 구간에서는 더 낮게 예측한다.

이건 오히려 이번 문제를 **optimization 실패**보다 **representation / decomposition / context selection mismatch**로 보는 쪽에 힘을 실어준다.

---

## 11. 최종 해석

이번 run의 “전체적인 하방 예측”은 사실 아래 조합으로 보는 것이 가장 정확하다.

### (A) 공통 원인
- 마지막 fold의 급등이 train tail 대비 너무 가파르다.
- 그래서 parity GRU도 하방으로 틀린다.

### (B) AAForecast 전용 추가 원인
- AAForecast는 STAR decomposition + sparse anomaly attention 때문에 parity GRU보다 항상 더 낮은 level을 낸다.
- 그 추가 downward shift가 16/16 포인트 전부에서 확인된다.
- 여기에 `season_length` 미주입(기본 12 사용 가능성), sparse anomaly selection, 비작동 uncertainty가 겹치면서 **late-regime 상승을 따라가는 대신 plateau형 저수준 예측으로 고정**된다.

즉, 가장 짧게 말하면:

> **이 run의 하방 편향은 “late surge 자체의 예측 난도” + “AAForecast 경로가 parity GRU보다 더 보수적으로 작동한 구조적 downward shift”의 결합 결과다.**

---

## 12. 원인 우선순위 정리

### 확신도 높음
1. **late-regime 급등의 out-of-pattern acceleration**
2. **AAForecast 경로가 parity GRU보다 일관되게 더 낮은 level을 만드는 differential bias**

### 확신도 중상
3. **`season_length` 미주입으로 인한 STAR decomposition mismatch 가능성**
4. **sparse anomaly-context가 최근 급등 직전 문맥을 충분히 잡지 못한 점**

### 확신도 중간
5. **uncertainty stage 붕괴로 상방 리스크 반영이 불가능했던 점**

---

## 13. 반례 / 해석 시 주의점

- 이 모델이 **항상** 하방 예측한 것은 아니다. 초반 2개 fold는 상방 예측이다.
- 따라서 “AAForecast는 원래 무조건 낮게 찍는다”는 식의 일반화는 과하다.
- 보다 정확한 표현은:  
  **“이 case1 parity run에서는 2026-01 이후 레짐 전환 구간에서 AAForecast의 추가 경로가 GRU 대비 일관된 downward shift를 만들어 냈다.”**

