# 02. 데이터 및 모델 세팅

---

## **Case 3 HPT | BrentCrude**

- 아래는 case별 hist_exog_cols만 남기고, 공통 training/jobs 상세는 Appendix 첨부 하였음.

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

## **Case 3 HPT | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | iTransformer | 4.42% | 0.61 | 3.26 | -4.54 |
| 2 | NHITS | 3.84% | 0.62 | 2.89 | -5.89 |
| 3 | DLinear | 5.22% | 0.63 | 3.84 | -4.28 |
| 4 | PatchTST | 5.31% | 0.69 | 3.91 | -6.18 |
| 5 | Naive | 6.70% | 0.86 | 4.91 | -11.42 |
| 6 | LSTM | 8.44% | 1.32 | 5.73 | -34.22 |

### 각 모형별 Table

- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 HPT | BrentCrude | 4.42% | 0.61 | 3.26 | -4.54 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 HPT | BrentCrude | 3.84% | 0.62 | 2.89 | -5.89 |
- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 HPT | BrentCrude | 5.22% | 0.63 | 3.84 | -4.28 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 HPT | BrentCrude | 5.31% | 0.69 | 3.91 | -6.18 |
- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 HPT | BrentCrude | 6.70% | 0.86 | 4.91 | -11.42 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 3 HPT | BrentCrude | 8.44% | 1.32 | 5.73 | -34.22 |
