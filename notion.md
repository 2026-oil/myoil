# 01. 핵심쟁점

---

`feature_set_bs` 실험 결과를 기준으로, 블랙스완 지수(BS_Core_Index_A/B/C/Integrated)를 포함한 feature set이 케이스별·타깃별로 어떤 성능 차이를 보였는지 현재 `runs/feature_set_*` 및 `runs/feature_set_bs_*` 산출물 기준으로 다시 정리한다.

본 문서는 **`feature_set_bs` 중심 보고서**를 유지하되, 각 표와 해석에는 대응하는 `feature_set` 결과를 비교 기준으로 함께 반영한다.

# 02. 데이터 및 모델 세팅

---

- **예측 타깃:** WTI / Brent Oil (F) Weekly Avg
- **예측 단위:** 주간 예측
- **평가 구조:** 6개 rolling TSCV(h=8, step=8, gap=0)
- **overlap_eval_policy:** `by_cutoff_mean`
- **loss:** `mse`
- **residual:** 비활성화 (`residual.enabled: false`)
- **실험 모델군:** TimeXer / TSMixerx / Naive / iTransformer / LSTM
- **jobs fan-out:** 각 케이스는 `jobs_1 ~ jobs_4` 네 개 route로 실행되며, 본 문서의 모델 행은 해당 route들의 현재 leaderboard를 모델 기준으로 평균한 값이다.

## 02-01. 케이스별 hist_exog_cols

### **BrentCrude**

- **Case 1**
- 
    ```yaml
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
    ```

- **Case 2**
- 
    ```yaml
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
    ```

- **Case 3**
- 
    ```yaml
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
    ```

- **Case 4**
- 
    ```yaml
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
    ```

### **WTI**

- **Case 1**
- 
    ```yaml
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
    ```

- **Case 2**
- 
    ```yaml
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
    ```

- **Case 3**
- 
    ```yaml
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
    ```

- **Case 4**
- 
    ```yaml
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
    ```

# 03. 실험 설계 및 적용

---

- 각 타깃(BrentCrude, WTI)을 독립적인 forecasting 문제로 학습 및 평가했다.
- 각 케이스는 **hist_exog_cols 구성만 다르고**, 모델군과 공통 training/CV 설정은 동일하게 유지했다.
- 각 케이스는 `jobs_1 ~ jobs_4` 네 개 route로 fan-out 되며, route별로 동일한 모델군을 서로 다른 하이퍼파라미터로 실행한다.
- `04-01`의 케이스 평균은 현재 `runs/*_jobs_*`의 모델 평균을 기준으로 계산했다.
- `04-02`, `04-03`의 모델 행은 각 케이스/모델에 대해 `jobs_1 ~ jobs_4` leaderboard 값을 평균한 대표값이다.

# 04. 실험(모델링) 결과

### 인사이트

- `feature_set_bs` 기준 BrentCrude의 최저 케이스 평균 MAPE는 **Case 3 (7.02%)**였다.
- `feature_set_bs` 기준 WTI의 최저 케이스 평균 MAPE는 **Case 2 (7.74%)**였다.
- 케이스 평균 기준 가장 큰 개선은 **WTI Case 2 (-0.23%p)**, 가장 큰 악화는 **WTI Case 1 (+0.36%p)**였다.
- 학습 모델만 놓고 보면 `feature_set_bs`에서 BrentCrude는 **LSTM (6.11%)**, WTI는 **LSTM (6.10%)**가 가장 낮은 평균 MAPE를 기록했다.

## 04-01. 케이스별 평균 성능 비교

| Case | 구분 | Target | Mean MAPE | △ MAPE | Mean nRMSE | Mean MAE | Mean R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Case 1 | 적용전 (`feature_set`) | BrentCrude | 8.73% |  | 1.27 | 6.08 | -42.63 |
|  | 적용후 (`feature_set_bs`) | BrentCrude | 8.64% | -0.09% | 1.24 | 6.02 | -37.63 |
| Case 2 | 적용전 (`feature_set`) | BrentCrude | 7.15% |  | 0.94 | 5.06 | -13.47 |
|  | 적용후 (`feature_set_bs`) | BrentCrude | 7.32% | +0.17% | 1.03 | 5.15 | -19.59 |
| Case 3 | 적용전 (`feature_set`) | BrentCrude | 6.91% |  | 0.92 | 4.89 | -12.30 |
|  | 적용후 (`feature_set_bs`) | BrentCrude | 7.02% | +0.11% | 0.97 | 4.96 | -13.85 |
| Case 4 | 적용전 (`feature_set`) | BrentCrude | 6.98% |  | 0.91 | 4.95 | -11.25 |
|  | 적용후 (`feature_set_bs`) | BrentCrude | 7.23% | +0.25% | 0.99 | 5.10 | -17.61 |
| Case 1 | 적용전 (`feature_set`) | WTI | 9.87% |  | 1.42 | 6.53 | -32.87 |
|  | 적용후 (`feature_set_bs`) | WTI | 10.23% | +0.36% | 1.54 | 6.75 | -44.02 |
| Case 2 | 적용전 (`feature_set`) | WTI | 7.97% |  | 1.15 | 5.31 | -17.37 |
|  | 적용후 (`feature_set_bs`) | WTI | 7.74% | -0.23% | 1.15 | 5.16 | -17.20 |
| Case 3 | 적용전 (`feature_set`) | WTI | 7.72% |  | 1.14 | 5.15 | -17.07 |
|  | 적용후 (`feature_set_bs`) | WTI | 7.76% | +0.03% | 1.17 | 5.16 | -17.40 |
| Case 4 | 적용전 (`feature_set`) | WTI | 7.97% |  | 1.13 | 5.31 | -16.32 |
|  | 적용후 (`feature_set_bs`) | WTI | 7.84% | -0.13% | 1.17 | 5.21 | -18.28 |

**BrentCrude.**

- `feature_set_bs` 기준 타깃 평균 MAPE는 7.55%이고, 최저 케이스는 Case 3(7.02% )였다.
- baseline 대비 평균 MAPE 변화는 +0.11%였고, 개선 케이스는 Case 1였다.
- 가장 큰 악화는 Case 4에서 나타났고, 해당 케이스의 ΔMAPE는 +0.25%였다.

**WTI.**

- `feature_set_bs` 기준 타깃 평균 MAPE는 8.39%이고, 최저 케이스는 Case 2(7.74% )였다.
- baseline 대비 평균 MAPE 변화는 +0.01%였고, 개선 케이스는 Case 2, Case 4였다.
- 가장 큰 악화는 Case 1에서 나타났고, 해당 케이스의 ΔMAPE는 +0.36%였다.

## 04-02. 케이스별 세부 결과

### **Case 1 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 6.00% | 0.78 | 4.36 | -9.52 |
| 2 | LSTM | 6.07% | 0.85 | 4.24 | -9.79 |
| 3 | TimeXer | 6.45% | 0.85 | 4.56 | -8.84 |
| 4 | iTransformer | 8.14% | 1.01 | 5.79 | -12.61 |
| 5 | TSMixerx | 16.56% | 2.73 | 11.16 | -147.38 |
| Mean |  | 8.64% | 1.24 | 6.02 | -37.63 |

- baseline 대비 케이스 평균 ΔMAPE는 -0.09%였다.

### **Case 2 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 6.00% | 0.78 | 4.36 | -9.52 |
| 2 | LSTM | 6.08% | 0.85 | 4.24 | -9.85 |
| 3 | iTransformer | 6.98% | 0.91 | 4.99 | -11.19 |
| 4 | TimeXer | 7.29% | 0.93 | 5.15 | -10.63 |
| 5 | TSMixerx | 10.25% | 1.69 | 7.00 | -56.76 |
| Mean |  | 7.32% | 1.03 | 5.15 | -19.59 |

- baseline 대비 케이스 평균 ΔMAPE는 +0.17%였다.

### **Case 3 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 6.00% | 0.78 | 4.36 | -9.52 |
| 2 | LSTM | 6.23% | 0.88 | 4.35 | -10.34 |
| 3 | iTransformer | 6.92% | 0.93 | 4.95 | -11.49 |
| 4 | TimeXer | 7.41% | 0.95 | 5.23 | -11.70 |
| 5 | TSMixerx | 8.54% | 1.29 | 5.91 | -26.21 |
| Mean |  | 7.02% | 0.97 | 4.96 | -13.85 |

- baseline 대비 케이스 평균 ΔMAPE는 +0.11%였다.

### **Case 4 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 6.00% | 0.78 | 4.36 | -9.52 |
| 2 | LSTM | 6.07% | 0.85 | 4.24 | -9.80 |
| 3 | iTransformer | 7.07% | 0.92 | 5.05 | -10.76 |
| 4 | TimeXer | 7.27% | 0.92 | 5.14 | -10.37 |
| 5 | TSMixerx | 9.74% | 1.48 | 6.73 | -47.61 |
| Mean |  | 7.23% | 0.99 | 5.10 | -17.61 |

- baseline 대비 케이스 평균 ΔMAPE는 +0.25%였다.

### **Case 1 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | LSTM | 6.09% | 0.83 | 4.06 | -8.71 |
| 2 | Naive | 6.34% | 0.96 | 4.36 | -12.95 |
| 3 | iTransformer | 8.05% | 1.09 | 5.41 | -13.51 |
| 4 | TimeXer | 10.13% | 1.47 | 6.67 | -24.16 |
| 5 | TSMixerx | 20.53% | 3.37 | 13.23 | -160.78 |
| Mean |  | 10.23% | 1.54 | 6.75 | -44.02 |

- baseline 대비 케이스 평균 ΔMAPE는 +0.36%였다.

### **Case 2 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | LSTM | 6.04% | 0.84 | 4.02 | -8.77 |
| 2 | Naive | 6.34% | 0.96 | 4.36 | -12.95 |
| 3 | TimeXer | 8.64% | 1.19 | 5.72 | -15.15 |
| 4 | iTransformer | 8.64% | 1.27 | 5.73 | -18.48 |
| 5 | TSMixerx | 9.03% | 1.48 | 5.94 | -30.63 |
| Mean |  | 7.74% | 1.15 | 5.16 | -17.20 |

- baseline 대비 케이스 평균 ΔMAPE는 -0.23%였다.

### **Case 3 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | LSTM | 6.25% | 0.91 | 4.16 | -9.65 |
| 2 | Naive | 6.34% | 0.96 | 4.36 | -12.95 |
| 3 | TimeXer | 8.61% | 1.19 | 5.69 | -15.24 |
| 4 | iTransformer | 8.06% | 1.25 | 5.35 | -17.79 |
| 5 | TSMixerx | 9.52% | 1.56 | 6.21 | -31.39 |
| Mean |  | 7.76% | 1.17 | 5.16 | -17.40 |

- baseline 대비 케이스 평균 ΔMAPE는 +0.03%였다.

### **Case 4 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | LSTM | 6.01% | 0.85 | 4.00 | -8.85 |
| 2 | Naive | 6.34% | 0.96 | 4.36 | -12.95 |
| 3 | TimeXer | 8.48% | 1.17 | 5.62 | -14.50 |
| 4 | iTransformer | 8.68% | 1.29 | 5.76 | -19.10 |
| 5 | TSMixerx | 9.68% | 1.61 | 6.29 | -36.01 |
| Mean |  | 7.84% | 1.17 | 5.21 | -18.28 |

- baseline 대비 케이스 평균 ΔMAPE는 -0.13%였다.

## 04-03. 블랙스완 적용 후 모형별 통합 Table

## **BrentCrude**

### **Naive**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 6.00% | 0.78 | 4.36 | -9.52 |
| Case 2 | 6.00% | 0.78 | 4.36 | -9.52 |
| Case 3 | 6.00% | 0.78 | 4.36 | -9.52 |
| Case 4 | 6.00% | 0.78 | 4.36 | -9.52 |
| Average | 6.00% | 0.78 | 4.36 | -9.52 |

### **TimeXer**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 6.45% | 0.85 | 4.56 | -8.84 |
| Case 2 | 7.29% | 0.93 | 5.15 | -10.63 |
| Case 3 | 7.41% | 0.95 | 5.23 | -11.70 |
| Case 4 | 7.27% | 0.92 | 5.14 | -10.37 |
| Average | 7.10% | 0.91 | 5.02 | -10.38 |

### **TSMixerx**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 16.56% | 2.73 | 11.16 | -147.38 |
| Case 2 | 10.25% | 1.69 | 7.00 | -56.76 |
| Case 3 | 8.54% | 1.29 | 5.91 | -26.21 |
| Case 4 | 9.74% | 1.48 | 6.73 | -47.61 |
| Average | 11.27% | 1.80 | 7.70 | -69.49 |

### **iTransformer**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 8.14% | 1.01 | 5.79 | -12.61 |
| Case 2 | 6.98% | 0.91 | 4.99 | -11.19 |
| Case 3 | 6.92% | 0.93 | 4.95 | -11.49 |
| Case 4 | 7.07% | 0.92 | 5.05 | -10.76 |
| Average | 7.28% | 0.94 | 5.19 | -11.51 |

### **LSTM**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 6.07% | 0.85 | 4.24 | -9.79 |
| Case 2 | 6.08% | 0.85 | 4.24 | -9.85 |
| Case 3 | 6.23% | 0.88 | 4.35 | -10.34 |
| Case 4 | 6.07% | 0.85 | 4.24 | -9.80 |
| Average | 6.11% | 0.86 | 4.27 | -9.95 |

## **WTI**

### **Naive**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 6.34% | 0.96 | 4.36 | -12.95 |
| Case 2 | 6.34% | 0.96 | 4.36 | -12.95 |
| Case 3 | 6.34% | 0.96 | 4.36 | -12.95 |
| Case 4 | 6.34% | 0.96 | 4.36 | -12.95 |
| Average | 6.34% | 0.96 | 4.36 | -12.95 |

### **TimeXer**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 10.13% | 1.47 | 6.67 | -24.16 |
| Case 2 | 8.64% | 1.19 | 5.72 | -15.15 |
| Case 3 | 8.61% | 1.19 | 5.69 | -15.24 |
| Case 4 | 8.48% | 1.17 | 5.62 | -14.50 |
| Average | 8.97% | 1.26 | 5.93 | -17.26 |

### **TSMixerx**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 20.53% | 3.37 | 13.23 | -160.78 |
| Case 2 | 9.03% | 1.48 | 5.94 | -30.63 |
| Case 3 | 9.52% | 1.56 | 6.21 | -31.39 |
| Case 4 | 9.68% | 1.61 | 6.29 | -36.01 |
| Average | 12.19% | 2.01 | 7.92 | -64.70 |

### **iTransformer**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 8.05% | 1.09 | 5.41 | -13.51 |
| Case 2 | 8.64% | 1.27 | 5.73 | -18.48 |
| Case 3 | 8.06% | 1.25 | 5.35 | -17.79 |
| Case 4 | 8.68% | 1.29 | 5.76 | -19.10 |
| Average | 8.36% | 1.22 | 5.56 | -17.22 |

### **LSTM**

| Case | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- |
| Case 1 | 6.09% | 0.83 | 4.06 | -8.71 |
| Case 2 | 6.04% | 0.84 | 4.02 | -8.77 |
| Case 3 | 6.25% | 0.91 | 4.16 | -9.65 |
| Case 4 | 6.01% | 0.85 | 4.00 | -8.85 |
| Average | 6.10% | 0.86 | 4.06 | -9.00 |

# 05. 결과 분석 및 얻게 된 인사이트

---

**타깃 평균 요약표 (`feature_set_bs` 기준)**

| 구분 | Mean MAPE | Mean nRMSE | Mean MAE | Mean R2 |
| --- | --- | --- | --- | --- |
| BrentCrude 평균 | 7.55% | 1.06 | 5.31 | -22.17 |
| WTI 평균 | 8.39% | 1.26 | 5.57 | -24.23 |
| 전체 평균 | 7.97% | 1.16 | 5.44 | -23.20 |

BrentCrude는 `feature_set_bs` 기준 평균 MAPE 7.55%를 기록했고, baseline 대비 변화는 +0.11%였다. 케이스 단위로는 Case 1만 개선됐고, Case 4의 악화 폭이 가장 컸다.

WTI는 `feature_set_bs` 기준 평균 MAPE 8.39%를 기록했고, baseline 대비 변화는 +0.01%였다. 개선은 Case 2와 Case 4에서 관찰됐지만, Case 1의 악화가 전체 평균을 다시 끌어올렸다.

학습 모델 기준으로는 BrentCrude와 WTI 모두 LSTM이 가장 낮은 평균 MAPE를 보였고(각각 6.11%, 6.10%), TimeXer/iTransformer는 중간권, TSMixerx는 케이스별 변동성이 가장 컸다.

Naive는 두 타깃 모두 가장 낮은 평균 오차를 유지했기 때문에, 후속 실험에서는 BS 지표가 학습 모델에만 추가 가치를 만든 구간이 어디인지 fold 또는 route 단위로 더 세분해서 보는 것이 필요하다.

# 06. 향후 Action Plan

---

- BrentCrude는 BS 추가가 현재 평균 기준으로 Case 1에서만 개선됐으므로, 후속 실험은 Case 1의 BS feature 유지 여부와 Case 2~4의 route별 악화 원인 분리를 우선한다.
- WTI는 Case 2와 Case 4에서만 개선이 확인됐으므로, 해당 두 케이스를 중심으로 BS 지표의 기여도를 재검증한다.
- LSTM은 두 타깃 모두 학습 모델 중 가장 낮은 평균 MAPE를 보였으므로, HPO 또는 추가 feature 실험의 우선순위를 높인다.
- TSMixerx는 jobs route별 편차가 커서 평균값이 크게 흔들리므로, route별 하이퍼파라미터 민감도 점검을 별도 분석으로 분리한다.

# Appendix. 공통 설정 및 모델 파라미터

---

- 현재 문서는 `feature_set_bs` 실행을 중심으로 작성했지만, 비교값은 대응하는 `feature_set` run을 함께 사용했다.
- 모든 케이스의 공통 Training/CV 설정은 `runs/feature_set_bs_brentoil_case1_bs_jobs_1/config/config.resolved.json` 등 최신 resolved config에서 확인했다.
- Jobs 설정은 공통 한 벌이 아니라 `jobs_1 ~ jobs_4` 네 개 route로 fan-out 된다.

## A-1. 공통 Training 설정

```yaml
train_protocol: expanding_window_tscv
input_size: 64
batch_size: 32
valid_batch_size: 64
windows_batch_size: 1024
inference_windows_batch_size: 1024
optimizer:
  name: adamw
  kwargs: {}
lr_scheduler:
  name: OneCycleLR
  max_lr: 0.001
  pct_start: 0.3
  div_factor: 25.0
  final_div_factor: 10000.0
  anneal_strategy: cos
  three_phase: false
  cycle_momentum: false
scaler_type: null
model_step_size: 8
max_steps: 2000
val_size: 24
val_check_steps: 20
min_steps_before_early_stop: 500
early_stop_patience_steps: 3
loss: mse
```

## A-2. 공통 CV 설정

```yaml
horizon: 8
step_size: 8
n_windows: 6
gap: 0
max_train_size: null
overlap_eval_policy: by_cutoff_mean
```

## A-3. 블랙스완 적용 후 공통 Jobs 설정

현재 `feature_set` / `feature_set_bs` 케이스는 아래 네 개 jobs route를 공통으로 참조한다.

### `jobs_1.yaml`

```yaml
- model: TimeXer
  params:
    patch_len: 8
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 128
    factor: 1
    dropout: 0.05
    use_norm: true
- model: TSMixerx
  params:
    n_block: 3
    ff_dim: 64
    dropout: 0.05
    revin: true
- model: Naive
  params: {}
- model: iTransformer
  params:
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 192
    dropout: 0.0
- model: LSTM
  params:
    encoder_hidden_size: 64
    decoder_hidden_size: 64
    encoder_n_layers: 4
    context_size: 8
```

### `jobs_2.yaml`

```yaml
- model: TimeXer
  params:
    patch_len: 16
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 128
    factor: 1
    dropout: 0.1
    use_norm: true
- model: TSMixerx
  params:
    n_block: 2
    ff_dim: 64
    dropout: 0.10
    revin: true
- model: Naive
  params: {}
- model: iTransformer
  params:
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 256
    dropout: 0.05
- model: LSTM
  params:
    encoder_hidden_size: 64
    decoder_hidden_size: 64
    encoder_n_layers: 4
    context_size: 10
```

### `jobs_3.yaml`

```yaml
- model: TimeXer
  params:
    patch_len: 8
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 128
    factor: 1
    dropout: 0.15
    use_norm: true
- model: TSMixerx
  params:
    n_block: 2
    ff_dim: 96
    dropout: 0.10
    revin: true
- model: Naive
  params: {}
- model: iTransformer
  params:
    hidden_size: 64
    n_heads: 4
    e_layers: 3
    d_ff: 256
    dropout: 0.0
- model: LSTM
  params:
    encoder_hidden_size: 64
    decoder_hidden_size: 64
    encoder_n_layers: 3
    context_size: 16
```

### `jobs_4.yaml`

```yaml
- model: TimeXer
  params:
    patch_len: 16
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 128
    factor: 1
    dropout: 0.2
    use_norm: true
- model: TSMixerx
  params:
    n_block: 3
    ff_dim: 64
    dropout: 0.25
    revin: true
- model: Naive
  params: {}
- model: iTransformer
  params:
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 384
    dropout: 0.15
- model: LSTM
  params:
    encoder_hidden_size: 64
    decoder_hidden_size: 64
    encoder_n_layers: 4
    context_size: 12
```
