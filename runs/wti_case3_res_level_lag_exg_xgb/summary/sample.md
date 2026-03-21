# 02. 데이터 및 모델 세팅

---

## **Case 3 | WTI**

- 아래는 case별 hist_exog_cols만 남기고, 공통 training/jobs 상세는 Appendix 첨부 하였음.

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

## **Case 3 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | DLinear | 5.48% | 0.81 | 3.81 | -8.53 |
| 2 | iTransformer | 6.29% | 0.90 | 4.37 | -16.70 |
| 3 | PatchTST | 5.91% | 0.92 | 4.06 | -14.71 |
| 4 | LSTM | 12.50% | 1.74 | 8.40 | -47.02 |
| 5 | Autoformer | 38.58% | 4.96 | 25.45 | -427.29 |
| 6 | NHITS | 49.64% | 6.37 | 33.73 | -695.77 |

### 각 모형별 Table

- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 | WTI | 5.48% | 0.81 | 3.81 | -8.53 |
- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 | WTI | 6.29% | 0.90 | 4.37 | -16.70 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 | WTI | 5.91% | 0.92 | 4.06 | -14.71 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 | WTI | 12.50% | 1.74 | 8.40 | -47.02 |
- Autoformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 | WTI | 38.58% | 4.96 | 25.45 | -427.29 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 | WTI | 49.64% | 6.37 | 33.73 | -695.77 |
