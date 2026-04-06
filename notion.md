# 01. Brent Case 1 실험 배경 및 기록 목적

---

- 본 문서는 **Brent case 1**에 대해 수행한 feature-subset 실험 6건의 결과를 정리한 **실험 기록 문서**이다.
- 이번 기록의 출발점은, 기존 **NEC 실험에서 뒤쪽 h7~8 급등 구간 대응이 충분하지 않다**는 문제의식이었다.
- 이에 따라 동일한 Brent case 1 조건에서 변수 조합을 먼저 NEC로 점검하고, 후속으로 **AAForecast 아키텍처 실험**을 추가 수행하였다.
- 본 문서의 목적은 “어떤 실험이 어떤 결과를 냈는가”를 남기는 것이며, **NEC와 AAForecast를 정면 비교하는 의사결정 문서**로 쓰는 것은 본 범위에 포함하지 않는다.

# 02. 실험 개요 및 공통 세팅

---

## 02-01. 공통 실험 조건

- **Target**: `Com_BrentCrudeOil`
- **Dataset**: `data/df.csv`
- **입력 길이(`input_size`)**: 96
- **예측 horizon**: 8
- **CV**: `n_windows=2`, `step_size=4`, `gap=0`
- **Train protocol**: `expanding_window_tscv`
- **Loss**: `mse`
- **Scaler**: `robust`
- **Batch 실행 기준일**: `2026-04-02`
- **Batch 결과**: 6개 실험 모두 PASS

## 02-02. 변수셋 설계

| 변수셋 | 설명 | 제외된 변수군 |
| --- | --- | --- |
| `BS + GPR` | 블랙스완 지수 계열 + 지정학 리스크 계열을 모두 포함한 10개 변수셋 | 없음 |
| `BS` | `BS + GPR`에서 지정학 리스크 계열 제거 | `GPRD_THREAT`, `GPRD`, `GPRD_ACT` |
| `GPR` | `BS + GPR`에서 블랙스완 지수 계열 제거 | `BS_Core_Index_A`, `BS_Core_Index_B`, `BS_Core_Index_C` |

## 02-03. 실행한 6개 실험

| 구분 | 실험명 |
| --- | --- |
| NEC | NEC - BS + GPR |
| NEC | NEC - BS |
| NEC | NEC - GPR |
| AAForecast | AAForecast - BS + GPR |
| AAForecast | AAForecast - BS |
| AAForecast | AAForecast - GPR |

# 03. NEC feature-subset 실험 기록

---

## 03-01. 실험 의도

- NEC 실험은 **후행 급등 구간 대응 한계가 어느 변수군 제거에서 더 크게 드러나는지**를 확인하기 위한 기록 단계로 수행하였다.
- 본 문서에서는 NEC 결과를 “AAForecast와의 승부”로 해석하지 않고, **후속 아키텍처 실험을 열어준 전 단계 기록**으로 정리한다.

## 03-02. NEC 결과 요약

| 변수셋 | MAPE | MAE | RMSE | nRMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BS + GPR` | **4.84%** | 4.067 | 6.584 | 26.18% | 0.320 |
| `BS` | 13.53% | 10.231 | 13.343 | 89.05% | -5.755 |
| `GPR` | 15.79% | 11.782 | 14.781 | 104.02% | -8.530 |

## 03-03. NEC 실험 기록 메모

- NEC 내부에서는 **`BS + GPR`이 가장 낮은 MAPE(4.84%)**를 기록했다.
- `BS`는 `BS + GPR` 대비 **+8.69%p**, `GPR`은 **+10.95%p** 만큼 MAPE가 악화되었다.
- `BS`, `GPR`은 모두 R²가 큰 폭의 음수로 내려가, 해당 변수군 제거가 NEC 실험 안정성을 크게 약화시키는 방향으로 나타났다.
- 따라서 NEC 단계 기록에서는 **GPRD 계열과 블랙스완 지수 계열을 모두 유지한 `BS + GPR` 조합**이 후속 실험의 기준점이 되었다.

# 04. AAForecast 후속 실험 기록

---

## 04-01. 실험 의도

- NEC 단계에서 확인한 문제의식을 바탕으로, 동일한 Brent case 1 / 동일한 feature-subset 축(`BS + GPR`, `BS`, `GPR`)에서 **AAForecast 아키텍처를 후속 실험**으로 수행하였다.
- 이 단계의 목적은 NEC 결과를 뒤집는 경쟁 비교가 아니라, **후행 급등 구간 대응 문제를 다른 아키텍처에서 어떻게 기록할 수 있는지 확인하는 것**이었다.

## 04-02. AAForecast 결과 요약

| 변수셋 | MAPE | MAE | RMSE | nRMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BS + GPR` | **4.62%** | 3.395 | 4.935 | 37.96% | -0.351 |
| `BS` | 6.56% | 4.600 | 4.998 | 39.68% | -0.510 |
| `GPR` | 8.94% | 6.085 | 6.248 | 51.51% | -1.617 |

## 04-03. AAForecast 실험 기록 메모

- AAForecast 내부에서도 **`BS + GPR`이 가장 낮은 MAPE(4.62%)**를 기록했다.
- `BS`는 `BS + GPR` 대비 **+1.94%p**, `GPR`은 **+4.32%p** 만큼 MAPE가 상승했다.
- 즉, AAForecast 후속 실험에서도 **변수 제거보다 `BS + GPR` 유지 조건이 가장 안정적인 기록**으로 남았다.
- 이번 후속 실험의 기록 결과는, NEC 단계에서 제기된 문제의식 이후 **AAForecast 아키텍처 실험이 실제로 유의미한 결과를 남겼다**는 점을 문서화하는 데 있다.

# 05. 실험 기록 요약 및 인사이트

---

- 이번 배치(`2026-04-02`)에서는 **6개 실험이 모두 정상 종료**되었다.
- NEC 기록에서는 `BS + GPR`이 가장 좋았고, `GPRD` 계열 또는 **블랙스완 지수 계열**을 제거한 경우 성능 저하 폭이 컸다.
- 후속 AAForecast 기록에서도 동일하게 `BS + GPR`이 가장 좋았고, 변수군 제거 시 MAPE가 일관되게 악화되었다.
- 따라서 이번 문서에서 남겨둘 핵심은 다음 두 가지다.
  1. **Brent case 1 기준 feature-subset 실험에서는 `BS + GPR`이 두 실험군 내부에서 모두 최상위였다.**
  2. **NEC의 한계를 문제의식으로 삼아 수행한 AAForecast 후속 실험이 실제로 좋은 결과를 기록했다.**
- 다만 본 문서는 실험 기록 문서이므로, 여기서 곧바로 **“향후엔 무조건 AAForecast를 채택한다”**는 식의 결론까지 확장하지는 않는다.

# 06. Appendix

---

## 06-01. 변수셋 상세

### `BS + GPR`
- `Idx_OVX`
- `Com_Oil_Spread`
- `BS_Core_Index_A`
- `BS_Core_Index_B`
- `BS_Core_Index_C`
- `Com_LMEX`
- `Com_BloombergCommodity_BCOM`
- `GPRD_THREAT`
- `GPRD`
- `GPRD_ACT`

### `BS`
- `Idx_OVX`
- `Com_Oil_Spread`
- `BS_Core_Index_A`
- `BS_Core_Index_B`
- `BS_Core_Index_C`
- `Com_LMEX`
- `Com_BloombergCommodity_BCOM`

### `GPR`
- `Idx_OVX`
- `Com_Oil_Spread`
- `Com_LMEX`
- `Com_BloombergCommodity_BCOM`
- `GPRD_THREAT`
- `GPRD`
- `GPRD_ACT`

## 06-02. 공통 Training / CV / Scheduler 설정

- **Training**
  - `input_size=96`
  - `batch_size=32`
  - `valid_batch_size=64`
  - `windows_batch_size=1024`
  - `inference_windows_batch_size=1024`
  - `max_steps=1000`
  - `val_size=16`
  - `val_check_steps=1`
  - `min_steps_before_early_stop=400`
  - `early_stop_patience_steps=20`
  - `loss=mse`
  - `optimizer=adamw`
  - `scaler_type=robust`
- **CV**
  - `horizon=8`
  - `step_size=4`
  - `n_windows=2`
  - `gap=0`
  - `overlap_eval_policy=by_cutoff_mean`
- **Scheduler**
  - `gpu_ids=[0, 1]`
  - `max_concurrent_jobs=2`
  - `worker_devices=1`

## 06-03. 하이퍼파라미터 (6개 실험 전체)

> 하이퍼파라미터 관련 내용은 Appendix의 가장 하단에만 정리한다. 이번 6개 실험은 실험별 feature subset은 다르지만, 모델/학습 하이퍼파라미터는 실질적으로 동일 그룹으로 묶인다.

### 06-03-01. AAForecast 3개 실험

적용 실험
- AAForecast - BS + GPR
- AAForecast - BS
- AAForecast - GPR

공통 AAForecast model params
- `encoder_hidden_size=256`
- `encoder_n_layers=3`
- `encoder_dropout=0.2`
- `decoder_hidden_size=256`
- `decoder_layers=3`
- `season_length=4`
- `trend_kernel_size=3`
- `anomaly_threshold=4.0`

### 06-03-02. NEC 3개 실험

적용 실험
- NEC - BS + GPR
- NEC - BS
- NEC - GPR

공통 NEC preprocessing / inference
- `preprocessing.mode=diff_std`
- `gmm_components=3`
- `epsilon=1.5`
- `inference.mode=soft_weighted_inverse`
- `threshold=0.5`
- `validation.windows=8`

공통 NEC branch params
- **classifier / extreme (`TSMixerx`)**
  - `n_block=3`
  - `ff_dim=96`
  - `dropout=0.1`
  - `revin=true`
- **classifier only**
  - `alpha=2.0`
  - `beta=0.5`
  - `oversample_extreme_windows=true`
- **normal (`LSTM`)**
  - `encoder_hidden_size=128`
  - `decoder_hidden_size=128`
  - `encoder_n_layers=2`
  - `decoder_layers=2`
  - `oversample_extreme_windows=false`
- **extreme (`TSMixerx`)**
  - `oversample_extreme_windows=true`

변동 항목
- NEC 3개 실험 간 차이는 **branch별 변수 목록(feature subset)** 뿐이며, 위 하이퍼파라미터 자체는 동일하다.
