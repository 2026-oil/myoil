# 01. 핵심쟁점

---

본 실험은 **XGB, LGBM** 모델로 Brent / WTI 각각에 대해,
단변량 예측과 `BS_Core_Index_Integrated`를 추가한 다변량 예측을 비교하여
블랙스완 지수가 유가 예측에 유의미한 정보를 주는지 검토한다.

이번 비교의 핵심은 다음과 같다.

- **Brent와 WTI 모두에서 `BS_Core_Index_Integrated`를 hist exogenous 변수로 추가했지만, 단변량 대비 성능이 악화됐다.**
- **즉 이번 tree direct 설정에서는 블랙스완 지수가 두 타깃 모두에 대해 유의미한 증분 정보를 만들지 못했다.**

# 02. 데이터 및 모델 세팅

---

- **예측 타깃:**
  - `Com_BrentCrudeOil` (BrentCrude)
  - `Com_CrudeOil` (WTI)
- **예측 단위:** 주간 예측
- **평가 구조:** 6개 rolling TSCV (`horizon=8`, `step_size=8`, `gap=0`)
- **overlap_eval_policy:** `by_cutoff_mean`
- **training.loss:** `mae`
- **실험 모델군:** `xgboost`, `lightgbm`
- **공통 lag 설정:** `[1, 2, 3, 4, 8, 12, 26, 52]`
- **residual:** 비활성화 (`residual.enabled: false`)

# 03. 실험 설계 및 적용

---

- 각 타깃(Brent / WTI)에 대해 **uni vs mul 한 쌍씩** 비교했다.
- 두 run의 차이는 **`hist_exog_cols`에 `BS_Core_Index_Integrated`를 추가했는지 여부**뿐이다.
- 모델 종류, tree 파라미터, training, CV 구조는 동일하다.
- 따라서 성능 차이는 본질적으로 **추가 exogenous 변수의 정보 가치**로 해석할 수 있다.

# 04. 실험(모델링) 결과

## 04-01. 전체 leaderboard 비교

| Target | Model | 단변량 MAPE | 다변량 MAPE | Δ MAPE | 단변량 RMSE | 다변량 RMSE | 단변량 MAE | 다변량 MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Brent | xgboost | 6.51% | 10.54% | +4.02%p | 5.699 | 9.558 | 4.716 | 7.684 |
| Brent | lightgbm | 7.32% | 9.23% | +1.92%p | 6.335 | 8.392 | 5.221 | 6.705 |
| WTI | xgboost | 7.58% | 12.12% | +4.54%p | 6.334 | 10.021 | 5.234 | 8.277 |
| WTI | lightgbm | 8.13% | 13.28% | +5.15%p | 6.603 | 10.560 | 5.575 | 8.930 |

### 인사이트

- Brent에서는 exog 추가 후
  - xgboost: **MAPE +4.02%p** 악화
  - lightgbm: **MAPE +1.92%p** 악화
- WTI에서는 exog 추가 후
  - xgboost: **MAPE +4.54%p** 악화
  - lightgbm: **MAPE +5.15%p** 악화
- 즉 **두 타깃 모두에서 mul이 uni보다 명확하게 열위**였다.

## 04-02. fold별 오차 변화

### **BrentCrude**

#### xgboost

| Fold | Cutoff | Δ MAPE (Mul-Uni) |
| --- | --- | --- |
| 0 | 2025-04-07 | +4.93%p |
| 1 | 2025-06-02 | +0.33%p |
| 2 | 2025-07-28 | +3.43%p |
| 3 | 2025-09-22 | +1.03%p |
| 4 | 2025-11-17 | +3.62%p |
| 5 | 2026-01-12 | +10.78%p |

- xgboost는 **6/6 fold 전부**에서 uni가 mul보다 더 좋았다.

#### lightgbm

| Fold | Cutoff | Δ MAPE (Mul-Uni) |
| --- | --- | --- |
| 0 | 2025-04-07 | -1.76%p |
| 1 | 2025-06-02 | +0.27%p |
| 2 | 2025-07-28 | +1.67%p |
| 3 | 2025-09-22 | +1.58%p |
| 4 | 2025-11-17 | +1.96%p |
| 5 | 2026-01-12 | +7.78%p |

- lightgbm는 **RMSE 기준 6/6 fold**, **MAE 기준 5/6 fold**에서 uni가 mul보다 더 좋았다.

### **WTI**

#### xgboost

| Fold | Cutoff | Δ MAPE (Mul-Uni) |
| --- | --- | --- |
| 0 | 2025-04-07 | +8.26%p |
| 1 | 2025-06-02 | -5.92%p |
| 2 | 2025-07-28 | +7.42%p |
| 3 | 2025-09-22 | +4.11%p |
| 4 | 2025-11-17 | +2.52%p |
| 5 | 2026-01-12 | +10.85%p |

- xgboost는 **5/6 fold**에서 uni가 mul보다 더 좋았고, 개선된 fold는 `2025-06-02` 하나뿐이었다.

#### lightgbm

| Fold | Cutoff | Δ MAPE (Mul-Uni) |
| --- | --- | --- |
| 0 | 2025-04-07 | +12.53%p |
| 1 | 2025-06-02 | -5.66%p |
| 2 | 2025-07-28 | +6.10%p |
| 3 | 2025-09-22 | +4.89%p |
| 4 | 2025-11-17 | +3.42%p |
| 5 | 2026-01-12 | +9.65%p |

- lightgbm도 **5/6 fold**에서 uni가 mul보다 더 좋았고, 개선된 fold는 `2025-06-02` 하나뿐이었다.

# 05. 결과 분석 및 얻게 된 인사이트

---

이번 비교를 수치로 요약하면 다음과 같다.

1. **Brent와 WTI 모두에서 입력 변수는 8개에서 16개로 늘었지만 MAPE는 개선되지 않았다.**
   - Brent: MAPE **+1.92%p ~ +4.02%p** 악화
   - WTI: MAPE **+4.54%p ~ +5.15%p** 악화

2. **추가된 8개 exog lag는 두 타깃 모두에서 target lag보다 훨씬 약한 신호였다.**
   - Brent: target lag 상관이 exog lag 상관보다 **약 4.6배 ~ 7.9배** 강함
   - WTI: target lag 상관이 exog lag 상관보다 **약 3.8배 ~ 6.1배** 강함

3. **악화는 특정 fold 하나의 우연이 아니라 반복 패턴이다.**
   - Brent: xgboost **6/6**, lightgbm RMSE **6/6** fold에서 uni 우세
   - WTI: xgboost/lightgbm 모두 **5/6 fold**에서 uni 우세

4. **가장 큰 실패 구간은 두 타깃 모두 마지막 fold의 급등 horizon이었다.**
   - Brent xgboost 마지막 fold 절대오차 합계: `mul 162.376 / uni 88.881`
   - WTI xgboost 마지막 fold 절대오차 합계: `mul 153.117 / uni 84.838`

따라서 이번 실험의 결론은 다음과 같다.

- **`BS_Core_Index_Integrated`는 현재 tree direct 설정에서 Brent와 WTI 모두에 대해 유의미한 증분 정보를 주지 못했다.**
- **두 타깃 모두 target lag만 사용하는 단변량 구성이 더 안정적이고 더 낮은 오차를 기록했다.**

# 06. 향후 Action Plan

---

- Brent / WTI tree 실험의 기본선은 당분간 **uni 구성**으로 유지한다.
- `BS_Core_Index_Integrated`를 계속 검증하려면, 현재처럼 단순 lag 추가가 아니라 **변환 / 정규화 / interaction / lag 선택 축소** 방식으로 재설계한다.
- tree 계열에서는 exog를 넣을 때 모든 lag를 그대로 넣기보다, 상관 또는 중요도 기준으로 **소수 lag만 선별**하는 방향이 필요하다.
- 후속 검증은 `fold_idx=5` 같은 급등 구간을 별도로 나눠, exog가 turning point 검출에 실제 도움을 주는지 다시 확인한다.

# Appendix. 근거 파일

---

## A-1. 모델 하이퍼파라미터 설정

### **xgboost**

```yaml
lags: [1, 2, 3, 4, 8, 12, 26, 52]
n_estimators: 128
max_depth: 4
subsample: 0.8
colsample_bytree: 0.8
requested_mode: learned_fixed
validated_mode: learned_fixed
```

### **lightgbm**

```yaml
lags: [1, 2, 3, 4, 8, 12, 26, 52]
n_estimators: 128
max_depth: 4
num_leaves: 15
min_child_samples: 20
feature_fraction: 0.8
requested_mode: learned_fixed
validated_mode: learned_fixed
```

## A-2. 공통 Training / CV 설정

```yaml
train_protocol: expanding_window_tscv
input_size: 96
batch_size: 32
valid_batch_size: 64
windows_batch_size: 1024
inference_windows_batch_size: 1024
model_step_size: 1
max_steps: 2000
val_size: 16
val_check_steps: 20
min_steps_before_early_stop: 500
early_stop_patience_steps: 3
loss: mae
```

```yaml
horizon: 8
step_size: 8
n_windows: 6
gap: 0
max_train_size: null
overlap_eval_policy: by_cutoff_mean
```

## A-3. 근거 파일

- `runs/tree_oil_mul_brent_mul_tree/summary/leaderboard.csv`
- `runs/tree_oil_uni_brent_uni_tree/summary/leaderboard.csv`
- `runs/tree_oil_mul_wti_mul_tree/summary/leaderboard.csv`
- `runs/tree_oil_uni_wti_uni_tree/summary/leaderboard.csv`
- `runs/tree_oil_mul_brent_mul_tree/scheduler/workers/xgboost/cv/xgboost_metrics_by_cutoff.csv`
- `runs/tree_oil_uni_brent_uni_tree/scheduler/workers/xgboost/cv/xgboost_metrics_by_cutoff.csv`
- `runs/tree_oil_mul_brent_mul_tree/scheduler/workers/lightgbm/cv/lightgbm_metrics_by_cutoff.csv`
- `runs/tree_oil_uni_brent_uni_tree/scheduler/workers/lightgbm/cv/lightgbm_metrics_by_cutoff.csv`
- `runs/tree_oil_mul_wti_mul_tree/scheduler/workers/xgboost/cv/xgboost_metrics_by_cutoff.csv`
- `runs/tree_oil_uni_wti_uni_tree/scheduler/workers/xgboost/cv/xgboost_metrics_by_cutoff.csv`
- `runs/tree_oil_mul_wti_mul_tree/scheduler/workers/lightgbm/cv/lightgbm_metrics_by_cutoff.csv`
- `runs/tree_oil_uni_wti_uni_tree/scheduler/workers/lightgbm/cv/lightgbm_metrics_by_cutoff.csv`
- `runs/tree_oil_mul_brent_mul_tree/scheduler/workers/xgboost/cv/xgboost_forecasts.csv`
- `runs/tree_oil_uni_brent_uni_tree/scheduler/workers/xgboost/cv/xgboost_forecasts.csv`
- `runs/tree_oil_mul_wti_mul_tree/scheduler/workers/xgboost/cv/xgboost_forecasts.csv`
- `runs/tree_oil_uni_wti_uni_tree/scheduler/workers/xgboost/cv/xgboost_forecasts.csv`
- `yaml/experiment/tree_oil_mul/brent_mul.yaml`
- `yaml/experiment/tree_oil_uni/brent_uni.yaml`
- `yaml/experiment/tree_oil_mul/wti_mul.yaml`
- `yaml/experiment/tree_oil_uni/wti_uni.yaml`
- `data/df.csv`
