# 02. 데이터 및 모델 세팅

---

## **Case 1 | BrentCrude**

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
    
    ### **Case 1 | WTI**
    
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
    
    ## **Case 2 | BrentCrude**
    
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
    
    ### **Case 2 | WTI**
    
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
    
    ## **Case 3 | BrentCrude**
    
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
    
    ### **Case 3 | WTI**
    
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
    
    ## **Case 4 | BrentCrude**
    
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
    
    ### **Case 4 | WTI**
    
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

- 각 타깃을 독립적인 단변량 forecasting 문제로 학습/평가
- 평가는 24개 rolling TSCV(h=12, step=4) 후 마지막 12주 holdout으로 최종 비교하는 구조로 설계했다.
- 다변량을 요하는 itransformer 이번 실험에서 제외

# 04. 실험(모델링) 결과

### 04-01. 세부 결과

---

- 각 case별로 BrentCrude/WTI의 last_fold_all_models plot과 leaderboard.csv 전체 모델 결과를 아래에 정리했다.
    
    각 case별로 BrentCrude와 WTI의 `last_fold_all_models`와 leaderboard를 함께 정리했다.
    
    ## **Case 1 | BrentCrude**
    
    ![](attachment:1a985a54-8d4b-4d60-969a-5606eb389f8b:last_fold_all_models.png)
    
    | Rank (MSE) | Model | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- | --- |
    | 1 | PatchTST | 4.885% | 3.638 | 33.247 | 4.545 |
    | 2 | DLinear | 5.308% | 3.906 | 33.355 | 4.798 |
    | 3 | Naive | 5.399% | 3.984 | 34.916 | 4.765 |
    | 4 | iTransformer | 6.626% | 4.863 | 43.589 | 5.718 |
    | 5 | NHITS | 8.474% | 6.066 | 61.553 | 7.031 |
    | 6 | LSTM | 11.402% | 8.197 | 104.889 | 8.922 |
    | 7 | Autoformer | 28.811% | 20.363 | 519.374 | 20.951 |
    
    ## **Case 1 | WTI**
    
    ![](attachment:ddf4abbf-2376-4887-8933-8f3b9b83652e:last_fold_all_models.png)
    
    | Rank (MSE) | Model | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- | --- |
    | 1 | DLinear | 5.483% | 3.810 | 31.393 | 4.745 |
    | 2 | Naive | 5.749% | 4.035 | 35.383 | 4.832 |
    | 3 | PatchTST | 5.905% | 4.065 | 37.248 | 4.902 |
    | 4 | iTransformer | 6.943% | 4.827 | 43.568 | 5.612 |
    | 5 | LSTM | 10.315% | 6.995 | 83.092 | 8.077 |
    | 6 | Autoformer | 31.724% | 21.335 | 602.595 | 22.024 |
    | 7 | NHITS | 31.649% | 22.201 | 882.620 | 26.625 |
    
    ---
    
    ## **Case 2 | BrentCrude**
    
    ![](attachment:059b45f4-b030-4495-b983-01eff61acdb8:last_fold_all_models.png)
    
    | Rank (MSE) | Model | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- | --- |
    | 1 | PatchTST | 4.885% | 3.638 | 33.247 | 4.545 |
    | 2 | DLinear | 5.308% | 3.906 | 33.355 | 4.798 |
    | 3 | Naive | 5.399% | 3.984 | 34.916 | 4.765 |
    | 4 | LSTM | 7.349% | 5.296 | 46.924 | 6.094 |
    | 5 | iTransformer | 6.643% | 4.896 | 55.328 | 5.594 |
    | 6 | Autoformer | 32.366% | 23.287 | 676.025 | 23.856 |
    | 7 | NHITS | 29.211% | 20.943 | 754.585 | 24.880 |
    
    ## **Case 2 | WTI**
    
    ![](attachment:3aef46fc-1b22-46b7-939d-4608ca979d9a:last_fold_all_models.png)
    
    | Rank (MSE) | Model | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- | --- |
    | 1 | DLinear | 5.483% | 3.810 | 31.393 | 4.745 |
    | 2 | Naive | 5.749% | 4.035 | 35.383 | 4.832 |
    | 3 | PatchTST | 5.905% | 4.065 | 37.248 | 4.902 |
    | 4 | iTransformer | 6.286% | 4.436 | 48.649 | 5.121 |
    | 5 | LSTM | 11.792% | 7.957 | 91.358 | 8.591 |
    | 6 | NHITS | 24.173% | 16.609 | 448.317 | 18.920 |
    | 7 | Autoformer | 39.398% | 26.083 | 897.385 | 26.491 |
    
    ---
    
    ## **Case 3 | BrentCrude**
    
    ![](attachment:059948e7-8a3e-454a-bfa0-d6d24ffe9e94:last_fold_all_models.png)
    
    | Rank (MSE) | Model | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- | --- |
    | 1 | PatchTST | 4.881% | 3.635 | 33.141 | 4.536 |
    | 2 | DLinear | 5.281% | 3.881 | 33.154 | 4.765 |
    | 3 | Naive | 5.399% | 3.984 | 34.916 | 4.765 |
    | 4 | iTransformer | 5.932% | 4.349 | 43.524 | 5.097 |
    | 5 | LSTM | 8.349% | 5.883 | 49.119 | 6.423 |
    | 6 | Autoformer | 35.888% | 25.771 | 810.117 | 26.219 |
    | 7 | NHITS | 36.143% | 26.305 | 1035.760 | 30.063 |
    
    ## **Case 3 | WTI**
    
    ![](attachment:ad99f0a9-41f9-4e9c-885f-60c2bd2f92a0:last_fold_all_models.png)
    
    | Rank (MSE) | Model | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- | --- |
    | 1 | DLinear | 5.483% | 3.810 | 31.393 | 4.745 |
    | 2 | Naive | 5.749% | 4.035 | 35.383 | 4.832 |
    | 3 | PatchTST | 5.905% | 4.065 | 37.248 | 4.902 |
    | 4 | iTransformer | 6.287% | 4.371 | 43.490 | 5.122 |
    | 5 | LSTM | 9.518% | 6.396 | 57.689 | 6.994 |
    | 6 | Autoformer | 34.896% | 23.377 | 704.675 | 23.896 |
    | 7 | NHITS | 44.908% | 31.425 | 1640.670 | 35.782 |
    
    ---
    
    ## **Case 4 | BrentCrude**
    
    ![](attachment:088ba727-f615-429e-93c6-732e8900bc9c:last_fold_all_models.png)
    
    | Rank (MSE) | Model | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- | --- |
    | 1 | PatchTST | 4.881% | 3.635 | 33.141 | 4.536 |
    | 2 | DLinear | 5.281% | 3.881 | 33.154 | 4.765 |
    | 3 | Naive | 5.399% | 3.984 | 34.916 | 4.765 |
    | 4 | LSTM | 7.670% | 5.469 | 47.096 | 6.180 |
    | 5 | iTransformer | 6.367% | 4.728 | 49.941 | 5.490 |
    | 6 | NHITS | 16.913% | 12.174 | 233.572 | 14.707 |
    | 7 | Autoformer | 30.803% | 22.337 | 641.273 | 22.965 |
    
    ## **Case 4 | WTI**
    
    ![](attachment:e8cddb0a-7f48-4c8d-a4b9-4caef1edc605:last_fold_all_models.png)
    
    | Rank (MSE) | Model | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- | --- |
    | 1 | DLinear | 5.483% | 3.810 | 31.393 | 4.745 |
    | 2 | Naive | 5.749% | 4.035 | 35.383 | 4.832 |
    | 3 | PatchTST | 5.905% | 4.065 | 37.248 | 4.902 |
    | 4 | iTransformer | 6.732% | 4.742 | 49.645 | 5.459 |
    | 5 | LSTM | 10.825% | 7.331 | 76.245 | 8.169 |
    | 6 | NHITS | 25.226% | 17.094 | 479.856 | 18.863 |
    | 7 | Autoformer | 38.528% | 25.520 | 887.874 | 26.109 |

### 각 모형별 Table

- PatchTST
    - WTI
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 5.905% | 4.065 | 37.248 | 4.902 |
    | 2 | 5.905% | 4.065 | 37.248 | 4.902 |
    | 3 | 5.905% | 4.065 | 37.248 | 4.902 |
    | 4 | 5.905% | 4.065 | 37.248 | 4.902 |
    - Brent
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 4.885% | 3.638 | 33.247 | 4.545 |
    | 2 | 4.885% | 3.638 | 33.247 | 4.545 |
    | 3 | 4.881% | 3.635 | 33.141 | 4.536 |
    | 4 | 4.881% | 3.635 | 33.141 | 4.536 |
- DLinear
    - WTI
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 5.483% | 3.810 | 31.393 | 4.745 |
    | 2 | 5.483% | 3.810 | 31.393 | 4.745 |
    | 3 | 5.483% | 3.810 | 31.393 | 4.745 |
    | 4 | 5.483% | 3.810 | 31.393 | 4.745 |
    - Brent
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 5.308% | 3.906 | 33.355 | 4.798 |
    | 2 | 5.308% | 3.906 | 33.355 | 4.798 |
    | 3 | 5.281% | 3.881 | 33.154 | 4.765 |
    | 4 | 5.281% | 3.881 | 33.154 | 4.765 |
- Naive
    - WTI
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 5.749% | 4.035 | 35.383 | 4.832 |
    | 2 | 5.749% | 4.035 | 35.383 | 4.832 |
    | 3 | 5.749% | 4.035 | 35.383 | 4.832 |
    | 4 | 5.749% | 4.035 | 35.383 | 4.832 |
    - Brent
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 5.399% | 3.984 | 34.916 | 4.765 |
    | 2 | 5.399% | 3.984 | 34.916 | 4.765 |
    | 3 | 5.399% | 3.984 | 34.916 | 4.765 |
    | 4 | 5.399% | 3.984 | 34.916 | 4.765 |
- iTransformer
    - WTI
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 6.943% | 4.827 | 43.568 | 5.612 |
    | 2 | 6.286% | 4.436 | 48.649 | 5.121 |
    | 3 | 6.287% | 4.371 | 43.490 | 5.122 |
    | 4 | 6.732% | 4.742 | 49.645 | 5.459 |
    - Brent
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 6.626% | 4.863 | 43.589 | 5.718 |
    | 2 | 6.643% | 4.896 | 55.328 | 5.594 |
    | 3 | 5.932% | 4.349 | 43.524 | 5.097 |
    | 4 | 6.367% | 4.728 | 49.941 | 5.490 |
- LSTM
    - WTI
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 10.315% | 6.995 | 83.092 | 8.077 |
    | 2 | 11.792% | 7.957 | 91.358 | 8.591 |
    | 3 | 9.518% | 6.396 | 57.689 | 6.994 |
    | 4 | 10.825% | 7.331 | 76.245 | 8.169 |
    - Brent
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 11.402% | 8.197 | 104.889 | 8.922 |
    | 2 | 7.349% | 5.296 | 46.924 | 6.094 |
    | 3 | 8.349% | 5.883 | 49.119 | 6.423 |
    | 4 | 7.670% | 5.469 | 47.096 | 6.180 |
- NHITS
    - WTI
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 31.649% | 22.201 | 882.620 | 26.625 |
    | 2 | 24.173% | 16.609 | 448.317 | 18.920 |
    | 3 | 44.908% | 31.425 | 1640.670 | 35.782 |
    | 4 | 25.226% | 17.094 | 479.856 | 18.863 |
    - Brent
    
    | Case | MAPE | MAE | MSE | RMSE |
    | --- | --- | --- | --- | --- |
    | 1 | 8.474% | 6.066 | 61.553 | 7.031 |
    | 2 | 29.211% | 20.943 | 754.585 | 24.880 |
    | 3 | 36.143% | 26.305 | 1035.760 | 30.063 |
    | 4 | 16.913% | 12.174 | 233.572 | 14.707 |
- Autoformer
    - WTI
    
    | Case | RMSE | MAPE | MAE | MSE |
    | --- | --- | --- | --- | --- |
    | 1 | 22.024 | 31.724% | 21.335 | 602.595 |
    | 2 | 26.491 | 39.398% | 26.083 | 897.385 |
    | 3 | 23.896 | 34.896% | 23.377 | 704.675 |
    | 4 | 26.109 | 38.528% | 25.520 | 887.874 |
    - Brent
    
    | Case | RMSE | MAPE | MAE | MSE |
    | --- | --- | --- | --- | --- |
    | 1 | 20.951 | 28.811% | 20.363 | 519.374 |
    | 2 | 23.856 | 32.366% | 23.287 | 676.025 |
    | 3 | 26.219 | 35.888% | 25.771 | 810.117 |
    | 4 | 22.965 | 30.803% | 22.337 | 641.273 |

# **Appendix**

실험에 진행된 하이퍼파라미터는 다음과 같다.

```yaml
training:
  train_protocol: expanding_window_tscv
  input_size: 64
  season_length: 52
  batch_size: 32
  valid_batch_size: 32
  windows_batch_size: 1024
  inference_windows_batch_size: 1024
  learning_rate: 0.001
  max_steps: 1000
  val_size: 8
  val_check_steps: 100
  early_stop_patience_steps: 5
  loss: mse
jobs:
- model: LSTM
  params:
    encoder_hidden_size: 64
    decoder_hidden_size: 64
    encoder_n_layers: 2
    context_size: 10
- model: NHITS
  params:
    n_pool_kernel_size:
    - 2
    - 2
    - 1
    n_freq_downsample:
    - 24
    - 12
    - 1
    dropout_prob_theta: 0.0
- model: DLinear
  params:
    moving_avg_window: 7
- model: Autoformer
  params:
    hidden_size: 64
    dropout: 0.1
    factor: 3
    n_head: 4
- model: PatchTST
  params:
    hidden_size: 64
    n_heads: 4
    encoder_layers: 2
    patch_len: 16
    dropout: 0.1
- model: iTransformer
  params:
    hidden_size: 64
    n_heads: 4
    e_layers: 2
    d_ff: 256
    dropout: 0.1
- model: Naive
  params: {}
```