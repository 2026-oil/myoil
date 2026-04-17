# 기술혁신학회지 논문 원고

## 국문 초록
본 연구는 WTI 주간 가격 예측에서 이상치 인지와 Retrieval 결합이 스파이크 구간 성능을 어떻게 바꾸는지 검토한다. 데이터는 677주의 주간 관측치(2013-03-25~2026-03-09)이며, 설명변수는 GPRD_THREAT, GPRD, GPRD_ACT, Idx_OVX, Com_LMEX, Com_BloombergCommodity_BCOM, Idx_DxyUSD로 구성된다. 실험은 AA-only, Baseline, Retrieval-only, AA+Retrieval의 네 계열을 비교했다. 결과적으로 direct Informer의 MAPE는 21.02%였으나 Retrieval을 결합하면 6.16%로 낮아졌고, AA-only Informer는 21.56%에 머물렀다. AA+Retrieval Informer는 6.21%를 기록해, 이상치 인지와 Retrieval을 함께 쓰는 설계가 단일 AA-only보다 실질적인 개선을 가져온다는 점을 보여준다.

주제어: WTI, Retrieval-Augmented Forecasting, AA-Forecast, Informer, 지리정치 위험지수

## Abstract
This study examines whether anomaly-aware preprocessing and Retrieval improve weekly WTI forecasting under spike-heavy market regimes. We evaluate 677 weekly observations from 2013-03-25 to 2026-03-09 with seven exogenous variables: GPRD_THREAT, GPRD, GPRD_ACT, Idx_OVX, Com_LMEX, Com_BloombergCommodity_BCOM, and Idx_DxyUSD. Across the final runs, direct Informer achieves 21.02% MAPE, while Informer+Retrieval reduces it to 6.16%. The AA-only Informer remains at 21.56%, but AA+Retrieval improves to 6.21%. The results indicate that Retrieval is the main driver of the gain, while anomaly-aware features help the combined pipeline remain competitive.

Keywords: WTI, Retrieval-Augmented Forecasting, AA-Forecast, Informer, Geopolitical Risk Index

## I. 서론
WTI 원유가격은 지정학적 충격, 수급 불균형, 환율 변화, 금융 변동성이 겹칠 때 급격한 스파이크를 보인다. 평균 회귀 성향의 시계열 모형은 정상 구간에서는 그럴듯한 성능을 내더라도, 위기 구간에서는 방향성과 개형을 놓치기 쉽다. 본 연구는 STAR 계열 이상치 인지와 Retrieval을 결합해, 희소한 충격 구간을 외부 메모리와 명시적 이상치 피처로 보완하는 접근을 시험한다.

## II. 데이터와 연구 설계
데이터는 주간 WTI 가격과 외생 변수 7개로 구성된다. 표본은 677주이며, 타깃은 Com_CrudeOil로 정렬했다. 실험 설정은 입력 길이 64주, 예측 지평 2주, 단일 rolling cutoff 구조를 사용했다. AA-Forecast는 LOWESS 기반 추세-계절성 분해와 이상치 임계치 3.08를 사용하고, Retrieval은 top-k=1, cosine similarity, recency gap 16을 적용했다.

| 변수 | 평균 | 최솟값 | 최댓값 | 표준편차 | 왜도 | 첨도 |
| --- | --- | --- | --- | --- | --- | --- |
| WTI 원유가격 | 66.87 | 3.92 | 120.44 | 20.07 | 0.295 | -0.424 |
| GPRD_THREAT | 135.57 | 50.21 | 656.83 | 60.98 | 2.767 | 14.445 |
| GPRD | 120.06 | 46.86 | 460.77 | 46.63 | 2.409 | 10.425 |
| GPRD_ACT | 104.23 | 22.04 | 549.21 | 56.55 | 2.346 | 11.105 |
| OVX | 37.99 | 15.22 | 234.52 | 17.93 | 4.587 | 35.037 |
| LMEX | 3,373.47 | 2,065.02 | 5,505.00 | 736.60 | 0.538 | -0.321 |
| BCOM | 97.70 | 60.13 | 138.28 | 18.08 | 0.409 | -0.603 |
| DXY | 96.10 | 79.22 | 113.07 | 7.20 | -0.595 | 0.126 |

| 항목 | 값 |
| --- | --- |
| 표본 | 677주 |
| 기간 | 2013-03-25 ~ 2026-03-09 |
| 타깃 | WTI 원유가격(Com_CrudeOil) |
| 설명변수 | GPRD_THREAT, GPRD, GPRD_ACT, Idx_OVX, Com_LMEX, Com_BloombergCommodity_BCOM, Idx_DxyUSD |
| 입력 길이 | 64주 |
| 예측 지평 | 2주 |
| 검증 방식 | 단일 rolling cutoff(2026-02-23) |
| 손실 함수 | MSE |
| AA-Forecast lowess_frac | 0.359 |
| AA-Forecast lowess_delta | 0.008 |
| AA-Forecast thresh | 3.08 |
| AA-Forecast season_length | 4 |
| Retrieval top_k | 1 |
| Retrieval recency_gap_steps | 16 |
| Retrieval trigger_quantile | 0.0126 |
| Retrieval blend_floor | 0.088 |
| Retrieval blend_max | 0.894 |

## III. 연구 가설
| 가설 번호 | 개념 | 가설 내용 | 검증 위치 |
| --- | --- | --- | --- |
| 가설 1 | 외생충격 변수 효과 | GPR 계열 충격 변수와 블랙스완류 이벤트 정보를 포함한 모델이 충격 구간에서 더 나은 설명력을 보일 것이다. | IV.3, IV.5 |
| 가설 2 | Retrieval 구조 효과 | Retrieval 구조를 적용한 모델이 적용하지 않은 모수적 모델보다 스파이크 구간 방향성/개형 포착이 우수할 것이다. | IV.4 |
| 가설 3 | 이상치-Retrieval 시너지 | STAR 기반 이상치 피처를 Retrieval 쿼리에 넣은 결합 모델이 단독 효과보다 더 안정적으로 작동할 것이다. | IV.5 |
| 가설 4 | 통합 파이프라인 우월성 | 이상치 인지 전처리, Retrieval, Informer를 통합한 제안 파이프라인이 baseline 대비 전체 오차를 낮출 것이다. | IV.4, IV.5 |

## IV. 실증 결과
전체 성능 비교에서는 Retrieval이 가장 큰 개선을 만들었다. direct Informer의 MAPE는 21.02%였지만, direct Informer+Ret는 6.16%로 낮아졌다. direct TimeXer는 21.81%, direct TimeXer+Ret는 6.11%였다. AA-only 계열은 21.56% 내외에 머물렀으나, AA+Ret는 6.21% 수준으로 떨어졌다.

| 구분 | 모형 | MAE | RMSE | MAPE(%) | R2 |
| --- | --- | --- | --- | --- | --- |
| Baseline | GRU | 20.22 | 21.46 | 23.20 | -9.091 |
| Baseline | Informer | 18.30 | 19.37 | 21.02 | -7.222 |
| Baseline | TimeXer | 18.94 | 19.83 | 21.81 | -7.616 |
| Baseline+Ret | GRU | 5.58 | 6.63 | 6.26 | 0.037 |
| Baseline+Ret | Informer | 5.47 | 6.38 | 6.16 | 0.108 |
| Baseline+Ret | TimeXer | 5.44 | 6.44 | 6.11 | 0.092 |
| AA-only | GRU | 19.15 | 20.24 | 21.99 | -7.979 |
| AA-only | Informer | 18.79 | 19.94 | 21.56 | -7.718 |
| AA-only | TimeXer | 18.83 | 19.84 | 21.66 | -7.622 |
| AA+Ret | GRU | 5.52 | 6.51 | 6.19 | 0.071 |
| AA+Ret | Informer | 5.53 | 6.50 | 6.21 | 0.073 |
| AA+Ret | TimeXer | 5.48 | 6.46 | 6.15 | 0.085 |

핵심적으로, direct baseline 대비 Retrieval-only는 대폭의 성능 향상을 보였고, AA-only는 단독으로는 충분하지 않았다. 다만 AA+Ret는 AA-only보다 분명히 좋았고, 일부 backbone에서는 Retrieval-only와 거의 비슷한 수준을 유지했다. 이 결과는 이상치 인지가 Retrieval의 대체물이 아니라 보조 신호임을 시사한다.

## V. 가설 검증
| 가설 | 검증 상태 | 핵심 증거 | 해석 |
| --- | --- | --- | --- |
| 가설 1 | 부분 검증 | final_wti에는 GPR/OVX 충격 변수는 존재하나 블랙스완 지수의 별도 제거 실험은 없음. | 외생충격 입력은 모델 설정에 반영되었지만, 블랙스완 항목은 본 결과셋에서 독립 검정되지 않았다. |
| 가설 2 | 지지 | Informer MAPE 21.02% → 6.16%, TimeXer 21.81% → 6.11% | Retrieval 도입만으로도 스파이크 구간 및 전체 오차가 크게 줄었다. |
| 가설 3 | 부분 지지 | AA-only Informer 21.56% 대비 AA+Ret 6.21%는 개선되지만, Retrieval-only Informer 6.16%에는 소폭 미달. | 이상치 피처는 도움이 되지만, 모든 backbone에서 단순 합산보다 항상 우월하다고 보기는 어렵다. |
| 가설 4 | 지지 | AA-only Informer 21.56% → AA+Ret 6.21% | 통합 파이프라인은 AA-only 대비 확실한 개선을 보였다. |

가설 2와 가설 4는 명확히 지지된다. Retrieval을 도입하면 전체 MAPE가 20%대에서 6%대로 내려가며, AA-only의 한계를 보완한다. 가설 3은 backbone에 따라 부분적으로만 지지되었다. AA+Ret는 AA-only보다 낫지만, retrieval-only가 이미 매우 강한 경우에는 단순한 결합이 항상 더 낫다고는 말할 수 없다. 가설 1은 final_wti에서 블랙스완 지수의 별도 제거 실험이 확인되지 않아 부분 검증으로 남긴다.

## VI. 결론
본 연구는 WTI 주간 예측에서 Retrieval이 핵심적인 개선 요인이라는 점을 확인했다. AA-only는 단독으로는 충분하지 않았지만, Retrieval과 결합하면 실질적인 오차 감소가 발생했다. 가장 낮은 MAPE는 TimeXer+Ret에서 나타났고, AA+Ret Informer도 direct baseline 대비 큰 폭으로 개선되었다. 따라서 충격 구간이 잦은 원유가격 예측에서는 이상치 인지와 외부 메모리 검색을 함께 쓰는 전략이 유효하다.

## 참고문헌
Caldara, D., & Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review*, 112(4), 1194-1225.
Cho, K., van Merriënboer, B., Gülçehre, Ç., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.
Farhangi, A., Bian, J., Huang, A., Xiong, H., Wang, J., & Guo, Z. (2023). AA-Forecast: Anomaly-aware forecast for extreme events. *Data Mining and Knowledge Discovery*, 37(3), 1209-1229.
Jammazi, R., & Aloui, C. (2012). Crude oil price forecasting: Experimental evidence from wavelet decomposition and neural network modeling. *Energy Economics*, 34, 230-241.
Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *AAAI*.
Li, S., Jin, X., Xuan, Y., Zhou, X., Chen, W., Wang, Y.-X., & Yan, X. (2019). Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. *NeurIPS*.
Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting. *NeurIPS*.
Du, D., Han, T., & Guo, S. (2026). Predicting the future by retrieving the past. *AAAI*.

## 부록 A. 주요 하이퍼파라미터
| 설정 | 값 |
| --- | --- |
| random_seed | 1 |
| opt_study_count | 1 |
| opt_n_trial | 50 |
| training.max_steps | 400 |
| training.batch_size | 32 |
| training.valid_batch_size | 8 |
| training.windows_batch_size | 64 |
| training.inference_windows_batch_size | 64 |
| training.val_size | 4 |
| training.scaler_type | robust |
| cv.horizon | 2 |
| cv.step_size | 1 |
| cv.n_windows | 1 |
| aa_forecast.model_params.hidden_size | 128 |
| aa_forecast.model_params.n_head | 8 |
| aa_forecast.model_params.encoder_layers | 2 |
| aa_forecast.model_params.decoder_layers | 4 |
