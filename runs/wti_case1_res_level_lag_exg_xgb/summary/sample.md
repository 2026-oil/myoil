# 02. 데이터 및 모델 세팅

---

## **Case 1 | WTI**

- 아래는 case별 hist_exog_cols만 남기고, 공통 training/jobs 상세는 Appendix 첨부 하였음.

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
...
```

# 03. 실험 설계 및 적용

---

- 타깃: WTI
- 각 타깃을 독립적인 forecasting 문제로 학습/평가
- 평가는 12개 rolling TSCV(h=8, step=8, gap=0) 구조로 설계했다.
- overlap_eval_policy: by_cutoff_mean

# 04. 실험(모델링) 결과

### 04-01. 세부 결과

---

- 각 case별로 BrentCrude/WTI의 last_fold_all_models plot과 leaderboard.csv 전체 모델 결과를 아래에 정리했다.

## **Case 1 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.75% | 0.76 | 4.04 | -7.90 |
| 2 | DLinear | 5.48% | 0.81 | 3.81 | -8.53 |
| 3 | iTransformer | 6.94% | 0.86 | 4.83 | -9.60 |
| 4 | PatchTST | 5.91% | 0.92 | 4.06 | -14.71 |
| 5 | LSTM | 9.24% | 1.06 | 6.67 | -14.60 |
| 6 | Autoformer | 34.12% | 4.71 | 22.31 | -375.40 |
| 7 | NHITS | 36.29% | 6.13 | 24.52 | -667.54 |

### 각 모형별 Table

- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | WTI | 5.75% | 0.76 | 4.04 | -7.90 |
- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | WTI | 5.48% | 0.81 | 3.81 | -8.53 |
- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | WTI | 6.94% | 0.86 | 4.83 | -9.60 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | WTI | 5.91% | 0.92 | 4.06 | -14.71 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | WTI | 9.24% | 1.06 | 6.67 | -14.60 |
- Autoformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | WTI | 34.12% | 4.71 | 22.31 | -375.40 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | WTI | 36.29% | 6.13 | 24.52 | -667.54 |
