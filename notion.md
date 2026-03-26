# 01. 핵심쟁점

---

`feature_set_bs` 실험 결과를 기준으로, 블랙스완 지수(BS_Core_Index_A/B/C/Integrated)를 포함한 feature set이 케이스별·타깃별로 어떤 성능 차이를 보였는지 정리한다.

**케이스별 평균 성능**, **타깃별 유효 feature set**, **모델별 일관성**을 함께 확인한다.

# 02. 데이터 및 모델 세팅

---

- **예측 타깃:** WTI / Brent Oil (F) Weekly Avg
- **예측 단위:** 주간 예측
- **평가 구조:** 12개 rolling TSCV(h=8, step=8, gap=0)
- **overlap_eval_policy:** by_cutoff_mean
- **loss:** mse
- **residual:** 비활성화 (`residual.enabled: false`)
- **실험 모델군:** TimeXer / TSMixerx / Naive / iTransformer / LSTM

## 02-01. 케이스별 hist_exog_cols

### **BrentCrude**

- **Case 1**
- 
    
    ```yaml
    ...
    hist_exog_cols:
      - Com_Gasoline
      - Com_Steel
      - Bonds_US_Spread_10Y_1Y
      - Bonds_CHN_Spread_30Y_5Y
      - EX_USD_BRL
      - Com_Cheese
      - Bonds_BRZ_Spread_10Y_1Y
      - Com_Cu_Gold_Ratio
      - Idx_OVX
      - Com_Oil_Spread
      - Com_LME_Zn_Spread
      - Idx_CSI300
      - Bonds_CHN_Spread_5Y_1Y
      - Com_LME_Cu_Spread
      - Com_LME_Pb_Spread
      - Com_LME_Al_Spread
      - BS_Core_Index_A
      - BS_Core_Index_B
      - BS_Core_Index_C
      - BS_Core_Index_Integrated
    ...
    ```
    
- **Case 2**
- 
    
    ```yaml
    ...
    hist_exog_cols:
      - Com_Gasoline
      - Com_BloombergCommodity_BCOM
      - Com_LME_Ni_Cash
      - Com_Coal
      - Com_Cotton
      - Com_LME_Al_Cash
      - Bonds_KOR_10Y
      - Com_Barley
      - Com_Canola
      - Com_LMEX
      - Com_LME_Ni_Inv
      - Com_Corn
      - Com_Wheat
      - BS_Core_Index_A
      - BS_Core_Index_B
      - BS_Core_Index_C
      - BS_Core_Index_Integrated
    ...
    ```
    
- **Case 3**
- 
    
    ```yaml
    ...
    hist_exog_cols:
      - Com_Gasoline
      - Com_BloombergCommodity_BCOM
      - Com_LME_Ni_Cash
      - Com_Coal
      - Com_LME_Al_Cash
      - Bonds_KOR_10Y
      - Com_LMEX
      - Com_LME_Ni_Inv
      - BS_Core_Index_A
      - BS_Core_Index_B
      - BS_Core_Index_C
      - BS_Core_Index_Integrated
    ...
    ```
    
- **Case 4**
- 
    
    ```yaml
    ...
    hist_exog_cols:
      - Com_Gasoline
      - Com_BloombergCommodity_BCOM
      - Com_LME_Ni_Cash
      - Com_Coal
      - Com_Cotton
      - Com_LME_Al_Cash
      - Bonds_KOR_10Y
      - Com_Barley
      - Com_Canola
      - Com_LMEX
      - Com_LME_Ni_Inv
      - Com_Corn
      - Com_Wheat
      - Com_NaturalGas
      - Idx_OVX
      - Com_Gold
      - BS_Core_Index_A
      - BS_Core_Index_B
      - BS_Core_Index_C
      - BS_Core_Index_Integrated
    ...
    ```
    
### **WTI**

- **Case 1**
- 
    
    ```yaml
    ...
    hist_exog_cols:
      - Com_Gasoline
      - Com_LME_Zn_Inv
      - Com_OrangeJuice
      - Com_Cheese
      - Bonds_BRZ_1Y
      - Idx_OVX
      - Com_Cu_Gold_Ratio
      - Com_LME_Sn_Inv
      - Idx_CSI300
      - Com_LME_Zn_Spread
      - Bonds_CHN_Spread_5Y_2Y
      - Com_LME_Al_Spread
      - Bonds_CHN_Spread_2Y_1Y
      - Com_Oil_Spread
      - Bonds_CHN_Spread_10Y_5Y
      - BS_Core_Index_A
      - BS_Core_Index_B
      - BS_Core_Index_C
      - BS_Core_Index_Integrated
    ...
    ```
    
- **Case 2**
- 
    
    ```yaml
    ...
    hist_exog_cols:
      - Com_Gasoline
      - Com_BloombergCommodity_BCOM
      - Com_LME_Ni_Cash
      - Com_Coal
      - Com_Canola
      - Com_Cotton
      - Com_LME_Al_Cash
      - Com_LMEX
      - Bonds_KOR_10Y
      - Com_PalmOil
      - Com_Barley
      - Com_Corn
      - Com_Oat
      - Com_Wheat
      - Com_Soybeans
      - Com_LME_Ni_Inv
      - BS_Core_Index_A
      - BS_Core_Index_B
      - BS_Core_Index_C
      - BS_Core_Index_Integrated
    ...
    ```
    
- **Case 3**
- 
    
    ```yaml
    ...
    hist_exog_cols:
      - Com_Gasoline
      - Com_BloombergCommodity_BCOM
      - Com_LME_Ni_Cash
      - Com_Coal
      - Com_LME_Al_Cash
      - Com_LMEX
      - Bonds_KOR_10Y
      - Com_LME_Ni_Inv
      - BS_Core_Index_A
      - BS_Core_Index_B
      - BS_Core_Index_C
      - BS_Core_Index_Integrated
    ...
    ```
    
- **Case 4**
- 
    
    ```yaml
    ...
    hist_exog_cols:
      - Com_Gasoline
      - Com_BloombergCommodity_BCOM
      - Com_LME_Ni_Cash
      - Com_Coal
      - Com_Canola
      - Com_Cotton
      - Com_LME_Al_Cash
      - Com_LMEX
      - Bonds_KOR_10Y
      - Com_PalmOil
      - Com_Barley
      - Com_Corn
      - Com_Oat
      - Com_Wheat
      - Com_Soybeans
      - Com_LME_Ni_Inv
      - Com_NaturalGas
      - Idx_OVX
      - Com_Gold
      - BS_Core_Index_A
      - BS_Core_Index_B
      - BS_Core_Index_C
      - BS_Core_Index_Integrated
    ...
    ```
    
# 03. 실험 설계 및 적용

---

- 각 타깃(BrentCrude, WTI)을 독립적인 forecasting 문제로 학습 및 평가했다.
- 각 케이스는 **hist_exog_cols 구성만 다르고**, 모델군과 평가 구조는 동일하게 유지했다.
- 공통 평가 설정은 **12개 rolling TSCV(h=8, step=8, gap=0)** 이다.
- `leaderboard.csv` 기준으로 각 모델의 평균 Fold 성능(MAPE, nRMSE, MAE, R2)을 정리했다.

# 04. 실험(모델링) 결과

### 인사이트

- BrentCrude는 **Case 1**이 가장 낮은 평균 MAPE(7.47%)를 기록했고, Case 2~4는 7.67%~7.71% 범위에 모였다.
- WTI는 **Case 2**가 가장 낮은 평균 MAPE(7.63%)를 기록했으며, Case 1의 평균 MAPE(10.36%)가 전체 평균을 가장 크게 끌어올렸다.
- 학습 모델만 기준으로 보면 BrentCrude에서는 **iTransformer**(평균 MAPE 6.86%)가 가장 안정적이었고, WTI에서는 **iTransformer**(평균 MAPE 7.60%)가 가장 낮았다.
- 두 타깃 모두 **Naive**가 nRMSE 기준 최상위였고, **TimeXer**는 BrentCrude Case 3(11.42%)와 WTI Case 1(15.81%)에서 큰 오차를 보였다.

## 04-01. 케이스별 평균 성능 비교

| Case | 구분 | Target | Mean MAPE | △ MAPE | Mean nRMSE | Mean MAE | Mean R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case 1 | 적용전 (`feature_set`) | BrentCrude | 7.56% |  | 1.07 | 5.51 | -25.89 |
|  | 적용후 (`feature_set_bs`) | BrentCrude | 7.47% | -0.09% | 1.03 | 5.45 | -19.14 |
| Case 2 | 적용전 (`feature_set`) | BrentCrude | 8.22% |  | 1.14 | 5.96 | -19.72 |
|  | 적용후 (`feature_set_bs`) | BrentCrude | 7.70% | -0.52% | 1.06 | 5.58 | -18.73 |
| Case 3 | 적용전 (`feature_set`) | BrentCrude | 7.54% |  | 1.10 | 5.50 | -19.48 |
|  | 적용후 (`feature_set_bs`) | BrentCrude | 7.71% | +0.16% | 1.12 | 5.59 | -21.40 |
| Case 4 | 적용전 (`feature_set`) | BrentCrude | 7.66% |  | 1.07 | 5.60 | -19.10 |
|  | 적용후 (`feature_set_bs`) | BrentCrude | 7.67% | +0.01% | 1.05 | 5.57 | -15.95 |
| Case 1 | 적용전 (`feature_set`) | WTI | 10.72% |  | 1.35 | 7.29 | -22.99 |
|  | 적용후 (`feature_set_bs`) | WTI | 10.36% | -0.35% | 1.40 | 7.07 | -29.37 |
| Case 2 | 적용전 (`feature_set`) | WTI | 8.92% |  | 1.22 | 6.15 | -18.03 |
|  | 적용후 (`feature_set_bs`) | WTI | 7.63% | -1.29% | 1.05 | 5.28 | -13.75 |
| Case 3 | 적용전 (`feature_set`) | WTI | 8.98% |  | 1.18 | 6.20 | -19.98 |
|  | 적용후 (`feature_set_bs`) | WTI | 8.28% | -0.70% | 1.17 | 5.72 | -19.95 |
| Case 4 | 적용전 (`feature_set`) | WTI | 8.12% |  | 1.11 | 5.57 | -16.23 |
|  | 적용후 (`feature_set_bs`) | WTI | 8.45% | +0.33% | 1.17 | 5.82 | -22.15 |

**BrentCrude.**

- 타깃 평균 MAPE는 `feature_set` 7.74%에서 `feature_set_bs` 7.64%로 0.11%p 낮아졌다.
- 케이스별로는 Case 2 개선폭(-0.52%p)이 가장 컸고, Case 3(+0.16%p)와 Case 4(+0.01%p)는 악화됐다.

**WTI.**

- 타깃 평균 MAPE는 `feature_set` 9.19%에서 `feature_set_bs` 8.68%로 0.50%p 낮아졌다.
- 개선은 Case 2(-1.29%p), Case 3(-0.70%p), Case 1(-0.35%p)에 나타났고, Case 4(+0.33%p)는 악화됐다.

## 04-02. 케이스별 세부 결과

### **Case 1 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.40% | 0.69 | 3.98 | -6.17 |
| 2 | TimeXer | 6.40% | 0.94 | 4.64 | -13.56 |
| 3 | iTransformer | 7.69% | 1.03 | 5.59 | -14.07 |
| 4 | LSTM | 8.57% | 1.10 | 6.48 | -15.15 |
| 5 | TSMixerx | 9.31% | 1.40 | 6.54 | -46.74 |
| Mean |  | 7.47% | 1.03 | 5.45 | -19.14 |

### **Case 2 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.40% | 0.69 | 3.98 | -6.17 |
| 2 | iTransformer | 6.50% | 0.92 | 4.75 | -12.10 |
| 3 | LSTM | 8.54% | 1.08 | 6.46 | -14.85 |
| 4 | TSMixerx | 7.85% | 1.24 | 5.58 | -36.65 |
| 5 | TimeXer | 10.20% | 1.37 | 7.13 | -23.87 |
| Mean |  | 7.70% | 1.06 | 5.58 | -18.73 |

### **Case 3 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.40% | 0.69 | 3.98 | -6.17 |
| 2 | iTransformer | 6.59% | 0.96 | 4.85 | -13.51 |
| 3 | TSMixerx | 6.55% | 0.96 | 4.74 | -14.29 |
| 4 | LSTM | 8.57% | 1.09 | 6.48 | -14.97 |
| 5 | TimeXer | 11.42% | 1.91 | 7.88 | -58.06 |
| Mean |  | 7.71% | 1.12 | 5.59 | -21.40 |

### **Case 4 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.40% | 0.69 | 3.98 | -6.17 |
| 2 | TSMixerx | 6.19% | 0.83 | 4.54 | -9.93 |
| 3 | iTransformer | 6.65% | 0.95 | 4.83 | -12.48 |
| 4 | LSTM | 8.60% | 1.10 | 6.49 | -15.24 |
| 5 | TimeXer | 11.50% | 1.71 | 8.01 | -35.92 |
| Mean |  | 7.67% | 1.05 | 5.57 | -15.95 |

### **Case 1 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.75% | 0.76 | 4.04 | -7.90 |
| 2 | iTransformer | 7.35% | 0.95 | 5.11 | -10.77 |
| 3 | LSTM | 9.36% | 1.06 | 6.76 | -14.84 |
| 4 | TSMixerx | 13.54% | 2.04 | 9.03 | -70.63 |
| 5 | TimeXer | 15.81% | 2.18 | 10.40 | -42.71 |
| Mean |  | 10.36% | 1.40 | 7.07 | -29.37 |

### **Case 2 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.75% | 0.76 | 4.04 | -7.90 |
| 2 | TSMixerx | 6.64% | 0.93 | 4.58 | -11.35 |
| 3 | iTransformer | 7.63% | 1.05 | 5.26 | -13.95 |
| 4 | LSTM | 9.36% | 1.07 | 6.76 | -14.87 |
| 5 | TimeXer | 8.78% | 1.43 | 5.78 | -20.69 |
| Mean |  | 7.63% | 1.05 | 5.28 | -13.75 |

### **Case 3 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.75% | 0.76 | 4.04 | -7.90 |
| 2 | LSTM | 9.29% | 1.07 | 6.72 | -14.84 |
| 3 | iTransformer | 7.93% | 1.16 | 5.45 | -17.43 |
| 4 | TSMixerx | 7.73% | 1.18 | 5.26 | -21.33 |
| 5 | TimeXer | 10.72% | 1.65 | 7.15 | -38.26 |
| Mean |  | 8.28% | 1.17 | 5.72 | -19.95 |

### **Case 4 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.75% | 0.76 | 4.04 | -7.90 |
| 2 | iTransformer | 7.49% | 1.01 | 5.17 | -12.35 |
| 3 | TSMixerx | 7.33% | 1.05 | 4.99 | -15.36 |
| 4 | LSTM | 9.31% | 1.06 | 6.73 | -14.62 |
| 5 | TimeXer | 12.37% | 1.98 | 8.19 | -60.51 |
| Mean |  | 8.45% | 1.17 | 5.82 | -22.15 |

## 04-03. 블랙스완 적용 후 모형별 통합 Table

## **BrentCrude**

### **Naive**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 5.40% | 0.69 | 3.98 | -6.17 |
| Case 2 | 5.40% | 0.69 | 3.98 | -6.17 |
| Case 3 | 5.40% | 0.69 | 3.98 | -6.17 |
| Case 4 | 5.40% | 0.69 | 3.98 | -6.17 |
| Average | 5.40% | 0.69 | 3.98 | -6.17 |

### **TimeXer**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 6.34% | 0.90 | 4.59 | -11.13 |
| Case 2 | 7.14% | 1.00 | 5.15 | -14.14 |
| Case 3 | 7.04% | 0.99 | 5.09 | -13.28 |
| Case 4 | 7.46% | 1.04 | 5.37 | -15.50 |
| Average | 7.00% | 0.99 | 5.05 | -13.51 |

### **TSMixerx**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 9.31% | 1.40 | 6.54 | -46.74 |
| Case 2 | 7.85% | 1.24 | 5.58 | -36.65 |
| Case 3 | 6.55% | 0.96 | 4.74 | -14.29 |
| Case 4 | 6.19% | 0.83 | 4.54 | -9.93 |
| Average | 7.47% | 1.11 | 5.35 | -26.91 |

### **iTransformer**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 6.63% | 0.90 | 4.79 | -11.84 |
| Case 2 | 6.10% | 0.88 | 4.45 | -11.72 |
| Case 3 | 5.94% | 0.86 | 4.37 | -9.99 |
| Case 4 | 6.31% | 0.91 | 4.59 | -11.73 |
| Average | 6.24% | 0.89 | 4.55 | -11.32 |

### **LSTM**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 8.52% | 1.08 | 6.44 | -14.87 |
| Case 2 | 8.86% | 1.10 | 6.66 | -15.03 |
| Case 3 | 8.53% | 1.08 | 6.45 | -14.86 |
| Case 4 | 7.85% | 1.00 | 5.89 | -13.32 |
| Average | 8.44% | 1.07 | 6.36 | -14.52 |

## **WTI**

### **Naive**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 5.75% | 0.76 | 4.04 | -7.90 |
| Case 2 | 5.75% | 0.76 | 4.04 | -7.90 |
| Case 3 | 5.75% | 0.76 | 4.04 | -7.90 |
| Case 4 | 5.75% | 0.76 | 4.04 | -7.90 |
| Average | 5.75% | 0.76 | 4.04 | -7.90 |

### **TimeXer**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 7.52% | 1.06 | 5.11 | -15.38 |
| Case 2 | 7.68% | 1.06 | 5.21 | -15.09 |
| Case 3 | 7.59% | 1.05 | 5.16 | -14.39 |
| Case 4 | 7.22% | 1.00 | 4.90 | -13.24 |
| Average | 7.50% | 1.04 | 5.10 | -14.52 |

### **TSMixerx**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 13.54% | 2.04 | 9.03 | -70.63 |
| Case 2 | 6.64% | 0.93 | 4.58 | -11.35 |
| Case 3 | 7.73% | 1.18 | 5.26 | -21.33 |
| Case 4 | 7.33% | 1.05 | 4.99 | -15.36 |
| Average | 8.81% | 1.30 | 5.96 | -29.67 |

### **iTransformer**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 7.25% | 0.98 | 4.98 | -12.38 |
| Case 2 | 7.65% | 1.11 | 5.23 | -17.47 |
| Case 3 | 7.45% | 1.06 | 5.11 | -14.95 |
| Case 4 | 7.68% | 1.11 | 5.23 | -17.18 |
| Average | 7.51% | 1.07 | 5.14 | -15.49 |

### **LSTM**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 9.34% | 1.06 | 6.75 | -14.62 |
| Case 2 | 8.08% | 0.95 | 5.77 | -12.50 |
| Case 3 | 6.90% | 0.85 | 4.82 | -9.57 |
| Case 4 | 8.32% | 0.97 | 5.95 | -12.76 |
| Average | 8.16% | 0.96 | 5.82 | -12.36 |

# 05. 결과 분석 및 얻게 된 인사이트

---

**타깃 평균 요약표**

| 구분 | Mean MAPE | Mean nRMSE | Mean MAE | Mean R2 |
| --- | --- | --- | --- | --- |
| BrentCrude 평균 | 6.91% | 0.95 | 5.06 | -14.48 |
| WTI 평균 | 7.55% | 1.03 | 5.21 | -15.99 |
| 전체 평균 | 7.23% | 0.99 | 5.13 | -15.24 |

BrentCrude는 평균적으로 6.91%의 MAPE를 보였고, Case 3~4 구간에서 성능이 가장 안정화됐다. 모델 단위로는 iTransformer와 TimeXer가 Naive 다음으로 낮은 평균 오차를 유지했다.

WTI는 평균적으로 7.55%의 MAPE를 보였으며, Case 2~4는 비슷한 수준으로 수렴했지만 Case 1의 TSMixerx 급등이 전체 평균을 악화시켰다. WTI 학습 모델 중에는 TimeXer와 iTransformer의 평균 MAPE가 가장 낮고 서로 거의 비슷했다.

공통적으로 Naive가 가장 낮은 nRMSE를 보였기 때문에, 후속 실험에서는 학습 모델이 Naive 대비 어떤 구간에서 추가 가치를 만드는지 구간 단위로 점검할 필요가 있다. 또한 TSMixerx는 일부 케이스에서 개선 여지가 보였지만 변동성이 크므로 하이퍼파라미터 또는 입력 feature 구성 재점검이 필요하다.

# 06. 향후 Action Plan

---

- BrentCrude는 Case 3~4 feature 구성을 우선 후보로 삼아 HPO 실험을 확장한다.
- WTI는 Case 2~4를 중심으로 재검증하되, Case 1에서 크게 흔들린 TSMixerx/TimeXer 입력 구성을 우선 점검한다.
- Naive 대비 학습 모델의 우위가 드러나는 구간을 fold 단위로 다시 확인해, 모델별 강·약세 구간을 분리 분석한다.

# Appendix. 공통 설정 및 모델 파라미터

---

- 공통적으로 `residual.enabled: false` 상태의 `feature_set_bs` 실험이다.
- 모든 케이스의 모델 구성은 동일하고, case 차이는 주로 `hist_exog_cols`에 있다.
- 아래 설정은 현재 top-level `runs/feature_set_bs_*_case*_bs/config/config.resolved.json`에서 확인한 실제 공통 하이퍼파라미터다.

## A-1. 공통 Training 설정

```yaml
train_protocol: expanding_window_tscv
input_size: 64
season_length: 52
batch_size: 32
valid_batch_size: 64
windows_batch_size: 1024
inference_windows_batch_size: 1024
learning_rate: 0.001
scaler_type: null
model_step_size: 8
max_steps: 1000
val_size: 8
val_check_steps: 50
early_stop_patience_steps: 5
num_lr_decays: -1
loss: mse
```

## A-2. 공통 CV 설정

```yaml
horizon: 8
step_size: 8
n_windows: 12
gap: 0
max_train_size: null
overlap_eval_policy: by_cutoff_mean
```

## A-3. 블랙스완 적용 후 공통 Jobs 설정

```yaml
- model: TimeXer
  params:
    patch_len: 16
    hidden_size: 768
    n_heads: 16
    e_layers: 4
    d_ff: 1024
    factor: 8
    dropout: 0.2
    use_norm: true
- model: TSMixerx
  params:
    n_block: 2
    ff_dim: 64
    dropout: 0.1
    revin: true
- model: Naive
  params: {}
- model: iTransformer
  params:
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 256
    dropout: 0.0
- model: LSTM
  params:
    encoder_hidden_size: 64
    decoder_hidden_size: 64
    encoder_n_layers: 4
    context_size: 10
```
