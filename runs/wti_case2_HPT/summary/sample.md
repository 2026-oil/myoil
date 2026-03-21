# 02. 데이터 및 모델 세팅

---

## **Case 2 HPT | WTI**

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

## **Case 2 HPT | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | iTransformer | 4.90% | 0.50 | 3.52 | -1.84 |
| 2 | DLinear | 5.33% | 0.90 | 3.76 | -11.63 |
| 3 | NHITS | 6.79% | 1.05 | 4.69 | -13.97 |
| 4 | LSTM | 7.11% | 1.05 | 4.74 | -12.15 |
| 5 | Naive | 7.10% | 1.08 | 4.92 | -15.50 |
| 6 | PatchTST | 7.03% | 1.23 | 4.80 | -22.52 |

### 각 모형별 Table

- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | WTI | 4.90% | 0.50 | 3.52 | -1.84 |
- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | WTI | 5.33% | 0.90 | 3.76 | -11.63 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | WTI | 6.79% | 1.05 | 4.69 | -13.97 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | WTI | 7.11% | 1.05 | 4.74 | -12.15 |
- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | WTI | 7.10% | 1.08 | 4.92 | -15.50 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 HPT | WTI | 7.03% | 1.23 | 4.80 | -22.52 |
