# 02. 데이터 및 모델 세팅

---

## **Case 2 | WTI**

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
- 평가는 12개 rolling TSCV(h=8, step=8, gap=0) 구조로 설계했다.
- overlap_eval_policy: by_cutoff_mean

# 04. 실험(모델링) 결과

### 04-01. 세부 결과

---

- 각 case별로 BrentCrude/WTI의 last_fold_all_models plot과 leaderboard.csv 전체 모델 결과를 아래에 정리했다.

## **Case 2 | WTI**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | iTransformer | 6.29% | 0.71 | 4.44 | -5.77 |
| 2 | Naive | 5.75% | 0.76 | 4.04 | -7.90 |
| 3 | DLinear | 5.48% | 0.81 | 3.81 | -8.53 |
| 4 | PatchTST | 5.91% | 0.92 | 4.06 | -14.71 |
| 5 | LSTM | 9.52% | 1.10 | 6.64 | -19.83 |
| 6 | NHITS | 24.25% | 3.93 | 16.67 | -238.13 |
| 7 | Autoformer | 42.86% | 5.53 | 28.51 | -500.75 |

### 각 모형별 Table

- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 | WTI | 6.29% | 0.71 | 4.44 | -5.77 |
- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 | WTI | 5.75% | 0.76 | 4.04 | -7.90 |
- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 | WTI | 5.48% | 0.81 | 3.81 | -8.53 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 | WTI | 5.91% | 0.92 | 4.06 | -14.71 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 | WTI | 9.52% | 1.10 | 6.64 | -19.83 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 | WTI | 24.25% | 3.93 | 16.67 | -238.13 |
- Autoformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 2 | WTI | 42.86% | 5.53 | 28.51 | -500.75 |
