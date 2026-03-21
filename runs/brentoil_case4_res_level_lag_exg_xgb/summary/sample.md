# 02. 데이터 및 모델 세팅

---

## **Case 4 | BrentCrude**

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
- 평가는 12개 rolling TSCV(h=8, step=8, gap=0) 구조로 설계했다.
- overlap_eval_policy: by_cutoff_mean

# 04. 실험(모델링) 결과

### 04-01. 세부 결과

---

- 각 case별로 BrentCrude/WTI의 last_fold_all_models plot과 leaderboard.csv 전체 모델 결과를 아래에 정리했다.

## **Case 4 | BrentCrude**

| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| 1 | Naive | 5.40% | 0.69 | 3.98 | -6.17 |
| 2 | PatchTST | 4.88% | 0.69 | 3.63 | -6.41 |
| 3 | DLinear | 5.28% | 0.76 | 3.88 | -8.07 |
| 4 | iTransformer | 6.37% | 0.77 | 4.73 | -10.58 |
| 5 | LSTM | 9.11% | 1.29 | 6.71 | -27.87 |
| 6 | NHITS | 22.91% | 3.24 | 16.20 | -162.02 |
| 7 | Autoformer | 31.17% | 3.88 | 22.62 | -212.57 |

### 각 모형별 Table

- Naive

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 | BrentCrude | 5.40% | 0.69 | 3.98 | -6.17 |
- PatchTST

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 | BrentCrude | 4.88% | 0.69 | 3.63 | -6.41 |
- DLinear

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 | BrentCrude | 5.28% | 0.76 | 3.88 | -8.07 |
- iTransformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 | BrentCrude | 6.37% | 0.77 | 4.73 | -10.58 |
- LSTM

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 | BrentCrude | 9.11% | 1.29 | 6.71 | -27.87 |
- NHITS

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 | BrentCrude | 22.91% | 3.24 | 16.20 | -162.02 |
- Autoformer

    | Case | MAPE | nRMSE | MAE | R2 |
    | --- | --- | --- | --- | --- |
    | Case 4 | BrentCrude | 31.17% | 3.88 | 22.62 | -212.57 |
