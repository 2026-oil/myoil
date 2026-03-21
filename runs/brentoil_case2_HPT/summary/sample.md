# 02. 데이터 및 모델 세팅

---

## **Case 2 HPT | BrentCrude**

- 아래는 case별 hist_exog_cols만 남기고, 공통 training/jobs 상세는 Appendix 첨부 하였음.

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

## **Case 2 HPT | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | DLinear | 4.83% | 0.65 | 3.62 | -6.31 |
| 2 | LSTM | 4.61% | 0.70 | 3.36 | -6.61 |
| 3 | iTransformer | 5.75% | 0.77 | 4.25 | -9.66 |
| 4 | NHITS | 5.90% | 0.83 | 4.35 | -11.23 |
| 5 | Naive | 6.70% | 0.86 | 4.91 | -11.42 |
| 6 | PatchTST | 7.73% | 1.09 | 5.33 | -20.84 |

### 각 모형별 Table

- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | BrentCrude | 4.83% | 0.65 | 3.62 | -6.31 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | BrentCrude | 4.61% | 0.70 | 3.36 | -6.61 |
- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | BrentCrude | 5.75% | 0.77 | 4.25 | -9.66 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | BrentCrude | 5.90% | 0.83 | 4.35 | -11.23 |
- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | BrentCrude | 6.70% | 0.86 | 4.91 | -11.42 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | BrentCrude | 7.73% | 1.09 | 5.33 | -20.84 |
