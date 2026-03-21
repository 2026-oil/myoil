# 02. 데이터 및 모델 세팅

---

## **Case 1 HPT | WTI**

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
- 평가는 5개 rolling TSCV(h=8, step=8, gap=0) 구조로 설계했다.
- overlap_eval_policy: by_cutoff_mean

# 04. 실험(모델링) 결과

### 04-01. 세부 결과

---

- 각 case별로 BrentCrude/WTI의 last_fold_all_models plot과 leaderboard.csv 전체 모델 결과를 아래에 정리했다.

## **Case 1 HPT | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | iTransformer | 6.41% | 0.68 | 4.58 | -4.64 |
| 2 | LSTM | 5.75% | 0.90 | 3.99 | -8.89 |
| 3 | DLinear | 5.33% | 0.90 | 3.76 | -11.63 |
| 4 | NHITS | 6.83% | 1.06 | 4.74 | -13.77 |
| 5 | Naive | 7.10% | 1.08 | 4.92 | -15.50 |
| 6 | PatchTST | 7.03% | 1.23 | 4.80 | -22.52 |

### 각 모형별 Table

- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | WTI | 6.41% | 0.68 | 4.58 | -4.64 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | WTI | 5.75% | 0.90 | 3.99 | -8.89 |
- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | WTI | 5.33% | 0.90 | 3.76 | -11.63 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | WTI | 6.83% | 1.06 | 4.74 | -13.77 |
- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | WTI | 7.10% | 1.08 | 4.92 | -15.50 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 1 HPT | WTI | 7.03% | 1.23 | 4.80 | -22.52 |
