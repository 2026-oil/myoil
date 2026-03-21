# 02. 데이터 및 모델 세팅

---

## **Case 1 | BrentCrude**

- 아래는 case별 hist_exog_cols만 남기고, 공통 training/jobs 상세는 Appendix 첨부 하였음.

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
...
```

# 03. 실험 설계 및 적용

---

- 타깃: BrentCrude
- 각 타깃을 독립적인 forecasting 문제로 학습/평가
- 평가는 12개 rolling TSCV(h=8, step=8, gap=0) 구조로 설계했다.
- overlap_eval_policy: by_cutoff_mean

# 04. 실험(모델링) 결과

### 04-01. 세부 결과

---

- 각 case별로 BrentCrude/WTI의 last_fold_all_models plot과 leaderboard.csv 전체 모델 결과를 아래에 정리했다.

## **Case 1 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.40% | 0.69 | 3.98 | -6.17 |
| 2 | PatchTST | 4.88% | 0.69 | 3.63 | -6.41 |
| 3 | DLinear | 5.28% | 0.76 | 3.88 | -8.07 |
| 4 | iTransformer | 6.83% | 0.90 | 5.01 | -10.50 |
| 5 | NHITS | 8.60% | 1.06 | 6.25 | -13.76 |
| 6 | LSTM | 15.89% | 2.13 | 11.87 | -89.01 |
| 7 | Autoformer | 32.56% | 4.79 | 22.60 | -402.41 |

### 각 모형별 Table

- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | BrentCrude | 5.40% | 0.69 | 3.98 | -6.17 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | BrentCrude | 4.88% | 0.69 | 3.63 | -6.41 |
- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | BrentCrude | 5.28% | 0.76 | 3.88 | -8.07 |
- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | BrentCrude | 6.83% | 0.90 | 5.01 | -10.50 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | BrentCrude | 8.60% | 1.06 | 6.25 | -13.76 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | BrentCrude | 15.89% | 2.13 | 11.87 | -89.01 |
- Autoformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 | BrentCrude | 32.56% | 4.79 | 22.60 | -402.41 |
