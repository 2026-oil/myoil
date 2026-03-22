# 02. 데이터 및 모델 세팅

---

## **Case 4 HPT | WTI**

- 아래는 case별 hist_exog_cols만 남기고, 공통 training/jobs 상세는 Appendix 첨부 하였음.

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

## **Case 4 HPT | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | DLinear | 5.52% | 0.84 | 3.85 | -8.18 |
| 2 | PatchTST | 6.41% | 0.86 | 4.39 | -7.47 |
| 3 | iTransformer | 5.86% | 0.96 | 4.06 | -11.85 |
| 4 | LSTM | 6.24% | 1.02 | 4.23 | -10.64 |
| 5 | Naive | 7.10% | 1.08 | 4.92 | -15.50 |
| 6 | NHITS | 6.25% | 1.31 | 4.23 | -34.63 |

### 각 모형별 Table

- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | WTI | 5.52% | 0.84 | 3.85 | -8.18 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | WTI | 6.41% | 0.86 | 4.39 | -7.47 |
- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | WTI | 5.86% | 0.96 | 4.06 | -11.85 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | WTI | 6.24% | 1.02 | 4.23 | -10.64 |
- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | WTI | 7.10% | 1.08 | 4.92 | -15.50 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | WTI | 6.25% | 1.31 | 4.23 | -34.63 |
