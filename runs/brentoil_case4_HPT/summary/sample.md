# 02. 데이터 및 모델 세팅

---

## **Case 4 HPT | BrentCrude**

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
  - Com_NaturalGas
  - Idx_OVX
  - Com_Gold
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

## **Case 4 HPT | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | DLinear | 5.22% | 0.63 | 3.84 | -4.28 |
| 2 | PatchTST | 5.31% | 0.69 | 3.91 | -6.18 |
| 3 | LSTM | 5.95% | 0.79 | 4.20 | -10.04 |
| 4 | NHITS | 5.66% | 0.86 | 4.04 | -11.68 |
| 5 | Naive | 6.70% | 0.86 | 4.91 | -11.42 |
| 6 | iTransformer | 6.40% | 0.87 | 4.64 | -11.94 |

### 각 모형별 Table

- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | BrentCrude | 5.22% | 0.63 | 3.84 | -4.28 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | BrentCrude | 5.31% | 0.69 | 3.91 | -6.18 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | BrentCrude | 5.95% | 0.79 | 4.20 | -10.04 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | BrentCrude | 5.66% | 0.86 | 4.04 | -11.68 |
- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | BrentCrude | 6.70% | 0.86 | 4.91 | -11.42 |
- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 HPT | BrentCrude | 6.40% | 0.87 | 4.64 | -11.94 |
