# 02. 데이터 및 모델 세팅

---

## **Case 1 HPT | BrentCrude**

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
- 평가는 5개 rolling TSCV(h=8, step=8, gap=0) 구조로 설계했다.
- overlap_eval_policy: by_cutoff_mean

# 04. 실험(모델링) 결과

### 04-01. 세부 결과

---

- 각 case별로 BrentCrude/WTI의 last_fold_all_models plot과 leaderboard.csv 전체 모델 결과를 아래에 정리했다.

## **Case 1 HPT | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | DLinear | 4.83% | 0.65 | 3.62 | -6.31 |
| 2 | LSTM | 5.84% | 0.72 | 4.29 | -7.81 |
| 3 | iTransformer | 5.05% | 0.75 | 3.68 | -7.75 |
| 4 | NHITS | 5.89% | 0.84 | 4.27 | -11.02 |
| 5 | Naive | 6.70% | 0.86 | 4.91 | -11.42 |
| 6 | PatchTST | 7.73% | 1.09 | 5.33 | -20.84 |

### 각 모형별 Table

- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | BrentCrude | 4.83% | 0.65 | 3.62 | -6.31 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | BrentCrude | 5.84% | 0.72 | 4.29 | -7.81 |
- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | BrentCrude | 5.05% | 0.75 | 3.68 | -7.75 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | BrentCrude | 5.89% | 0.84 | 4.27 | -11.02 |
- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | BrentCrude | 6.70% | 0.86 | 4.91 | -11.42 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | BrentCrude | 7.73% | 1.09 | 5.33 | -20.84 |
