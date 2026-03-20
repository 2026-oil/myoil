# Leak Check — `main.py` + `baseline-brentoil.yaml`

## Scope
점검 대상은 두 경로입니다.

1. **원본 baseline 경로**: `baseline-brentoil.yaml`
2. **임시 residual 경로**: 원본은 건드리지 않고, 최소 단위 probe용 임시 config로 `residual.enabled=true`

사용자 제약에 따라 **full-data / 대형 학습은 하지 않았고**, 아래 조합으로 점검했습니다.

- 코드 경로 점검
- 기존 타깃 테스트 재실행
- 최소 단위 runtime probe

## Final verdict
- **직접적인 미래 데이터 leakage는 발견하지 못했습니다.**
- 다만 **평가 윈도우 중복(overlapping test windows)** 은 존재합니다.
  - 이것은 **train/test leakage는 아니고**, `step_size < horizon` 때문에 생기는 **metric 해석 주의사항**입니다.
  - 현재 `cv.overlap_eval_policy: by_cutoff_mean` 은 config에 있지만 실제 runtime에서 소비되지 않아, fold 평균 metric은 일부 날짜를 중복 반영할 수 있습니다.

## Checklist

| Check | Baseline as-is | Residual temp probe | Verdict | Evidence |
|---|---|---|---|---|
| `main.py`가 별도 우회 경로 없이 동일 runtime으로 들어가는가 | Yes | Yes | PASS | `main.py`는 bootstrap 후 `residual.runtime.main` 호출 |
| outer TSCV에서 fold train/test가 index 기준으로 겹치지 않는가 | Yes | Yes | PASS | baseline split summary + runtime split code |
| fold replay 학습이 fold train slice만 사용하고, actual은 fold test slice에서만 읽는가 | Yes | Yes | PASS | `residual/runtime.py:_fit_and_predict_fold` |
| exact YAML에서 future exogenous가 미래에서 새어들 수 있는가 | `futr_exog_cols=[]` | probe도 `[]` | PASS | exact config + adapters code |
| exact baseline의 multivariate fallback 모델(iTransformer/TimeMixer/SMamba)이 train 밖 데이터를 adapter에 섞는가 | inspection + tiny probe | tiny iTransformer probe | PASS | routing summary + multivariate adapter summary + iTransformer probe |
| scaling / normalization 통계가 fold train 밖에서 fit되는가 | Not observed | Not observed | PASS | `NeuralForecast._scalers_fit_transform` + fold train-only `nf.fit(...)` |
| residual backcast panel이 outer-fold test를 학습에 섞는가 | N/A | No | PASS | code + targeted pytest + probe artifacts |
| residual eval panel이 outer-fold train 안쪽을 평가에 섞는가 | N/A | No | PASS | code + probe artifacts |
| residual XGBoost feature frame에 fold id / corrected target 같은 누수 컬럼이 들어가는가 | N/A | No | PASS | plugin code + targeted pytest |
| Optuna main tuning이 fold별 test 바깥을 미리 학습/평가에 쓰는가 | targeted tests + inspection | targeted tests + inspection | PASS (targeted) | `_tune_main_job` + auto-mode tests |
| Optuna residual tuning이 fold-local backcast/eval 밖의 데이터를 쓰는가 | config-mode test + inspection | config-mode test + inspection | PARTIAL (no end-to-end runtime) | `_score_residual_params` + mode-selection test |
| 평가 metric이 날짜 중복 없이 완전 disjoint하게 집계되는가 | No | probe에서는 1-step overlap 존재 가능 | CAUTION | `step_size < horizon`, config field unused |
| 원본 `baseline-brentoil.yaml`이 변경되지 않았는가 | Yes | Yes | PASS | SHA256 before/after identical |

## Detailed findings

### 1) `main.py` 자체는 leakage source가 아님
`main.py`는 환경 bootstrap 후 `residual.runtime.main`으로 위임하는 thin wrapper입니다. 자체적으로 데이터를 읽거나 split하지 않습니다.

- Evidence:
  - `main.py`
  - `uv run pytest --no-cov tests/test_residual_main.py -q` → `3 passed`

### 2) Outer TSCV split은 train/test disjoint
핵심 split 로직은 `residual/runtime.py`의 `_build_tscv_splits` 입니다.

- `train_idx = range(train_start, train_end)`
- `test_idx = range(test_start, test_end)`
- `test_start = train_end + gap`

따라서 같은 fold 안에서는 train/test index가 직접 겹치지 않습니다.

실제 `baseline-brentoil.yaml` 기준 정적 계산 결과:

- rows: `584`
- `horizon=8`, `step_size=4`, `n_windows=24`, `gap=0`
- `all_train_test_disjoint = true`
- first fold:
  - train: `2015-01-05` → `2024-04-08`
  - test: `2024-04-15` → `2024-06-03`
- last fold:
  - train: `2015-01-05` → `2026-01-12`
  - test: `2026-01-19` → `2026-03-09`

- Evidence:
  - `residual/runtime.py:562-603`
  - `/tmp/neuralforecast-leak-audit-20260320/baseline_split_summary.json`
  - targeted pytest: `test_build_tscv_splits_uses_configured_step_size`, `test_runtime_outer_cv_cutoffs_follow_step_size`

### 3) 실제 fold replay 경로도 train slice / test slice를 분리함
`_fit_and_predict_fold` 는 아래처럼 동작합니다.

- `train_df = source_df.iloc[train_idx]`
- `future_df = source_df.iloc[test_idx]`
- `nf.fit(adapter_inputs.fit_df, ...)`
- `target_actuals = future_df[target_col]`

즉 fit은 train slice 기반이고, 정답은 test slice에서만 읽습니다.

- Evidence:
  - `residual/runtime.py:372-408`

### 4) exact baseline YAML에서는 future exog 누수 경로가 비활성
`baseline-brentoil.yaml`은 다음과 같습니다.

- `hist_exog_cols`: 있음
- `futr_exog_cols: []`
- `static_exog_cols: []`

`build_univariate_inputs` 에서는:

- fit용 `fit_df`는 `source_df`(여기서는 fold train slice)에서 생성
- future exog는 `futr_exog_cols` 가 있을 때만 `future_df`에서 생성
- static exog도 현재 YAML에서는 비어 있음

즉 **이 exact YAML 기준으로는 future exogenous leakage path가 열려 있지 않습니다.**

추가로 exact baseline job set에는 `iTransformer`, `TimeMixer`, `SMamba`처럼 `_should_use_multivariate(...)` 경로를 타는 모델이 있습니다. 이 경로에서는 `build_multivariate_inputs(...)` 가 `train_df`의 `target + hist_exog` 컬럼만 melt 해서 multi-series `fit_df` 를 만들고, 현재 YAML처럼 `futr_exog_cols=[]` 인 경우 `futr_df` 를 만들지 않습니다.

실제 tiny `iTransformer` probe에서도:

- `uses_multivariate_adapter = true`
- `fit_unique_id_count = 17`
- `has_futr_df = false`
- `fit_max_ds == train_max_dt`

였고, residual 경계 summary 역시 두 fold 모두

- `backcast_all_ds_le_outer_train_end = true`
- `eval_all_ds_gt_outer_train_end = true`

였습니다. 즉 **exact baseline의 multivariate fallback 경로에서도 train 밖 데이터를 adapter/fit 쪽으로 끌어오는 증거는 없었습니다.**

- Evidence:
  - `baseline-brentoil.yaml`
  - `residual/runtime.py:322-330`
  - `residual/adapters.py:71-121`
  - `/tmp/neuralforecast-leak-audit-20260320/baseline_job_routing.json`
  - `/tmp/neuralforecast-leak-audit-20260320/itransformer_multivariate_adapter_summary.json`
  - `/tmp/neuralforecast-leak-audit-20260320/residual_probe_itransformer_boundary_summary.json`

### 5) Scaling / normalization도 fold train 경계 안에서 fit됨
`NeuralForecast._scalers_fit_transform` 은 `nf.fit(...)` 에 전달된 dataset에 대해서만 scaler를 fit합니다. 이 runtime에서는 `nf.fit(...)` 자체가 fold train slice에서 만든 `adapter_inputs.fit_df` 로 호출됩니다.

따라서 local scaler / static scaler 통계가 fold test를 보고 fit되는 경로는 보이지 않았습니다.

- Evidence:
  - `residual/runtime.py:395-400`
  - `neuralforecast/core.py:288-317`

### 6) Residual path도 outer-fold 경계를 넘지 않음
Residual은 `outer fold` 학습이 끝난 뒤 다음 두 패널로 나뉩니다.

- `backcast_panel`: outer fold의 **train 내부**에서 여러 inner cutoff를 만들어 residual 학습용 row 생성
- `eval_panel`: outer fold의 **test horizon**만 사용

코드상:

- `backcast_panel` 은 `train_df` 내부 prefix/history만 사용 (`history_df`, `future_df` 모두 `train_df`에서 slicing)
- `eval_panel` 은 outer fold 예측과 actual만 사용
- plugin은 `fit(backcast_panel)` 후 `predict(eval_panel)` 만 수행

이건 **inner backcast는 허용하지만 outer test leakage는 없는 구조**입니다.

- Evidence:
  - `residual/runtime.py:693-746`
  - `residual/runtime.py:749-893`
  - targeted pytest:
    - `test_apply_residual_plugin_uses_fold_local_backcasts_only`
    - `test_xgboost_plugin_predicts_panel_and_writes_checkpoint`

### 7) 최소 runtime residual probe에서도 경계가 맞음
실행한 최소 probe:

- base: `baseline-brentoil.yaml`의 컬럼 구조 사용
- data: 실제 `data/df.csv` 앞 24행 slice
- job: `DLinear` 1개
- residual: `xgboost`, fixed params
- CPU, `max_steps=1`, `horizon=2`, `n_windows=2`, `max_train_size=16`

결과:

- run success: `{"ok": true, "executed_jobs": ["DLinear"] ...}`
- residual artifacts 생성:
  - `corrected_folds.csv`
  - `folds/fold_000/backcast_panel.csv`
  - `folds/fold_000/corrected_eval.csv`
  - `folds/fold_001/backcast_panel.csv`
  - `folds/fold_001/corrected_eval.csv`
- boundary summary:
  - `fold_000`: `backcast_all_ds_le_outer_train_end = true`, `eval_all_ds_gt_outer_train_end = true`
  - `fold_001`: `backcast_all_ds_le_outer_train_end = true`, `eval_all_ds_gt_outer_train_end = true`

즉 probe artifact 기준으로도 **residual 학습 row는 outer train 안에만 있고**, **corrected eval row는 outer cutoff 이후만 포함**했습니다.

추가로 exact baseline YAML의 `hist_exog_cols` 가 **supports_hist_exog=True 모델에 실제로 연결되는지** 확인하기 위해 **LSTM 최소 probe**도 별도로 실행했습니다. `build_model(...)` 은 지원 모델에 대해서만 `hist_exog_list` 를 넘기며(`residual/models.py:184-192`), 이 probe 역시 `residual=true`, CPU, `max_steps=1`, 2 folds 조건에서 성공했고, 두 fold 모두

- `backcast_all_ds_le_outer_train_end = true`
- `eval_all_ds_gt_outer_train_end = true`

를 만족했습니다. 따라서 **hist exog를 실제로 받는 learned model 경로에서도 같은 fold 경계가 유지**되는 것을 확인했습니다.

- Evidence:
  - `/tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe.yaml`
  - `/tmp/neuralforecast-leak-audit-20260320/residual-probe-run/manifest/run_manifest.json`
  - `/tmp/neuralforecast-leak-audit-20260320/residual-probe-run/residual/DLinear/diagnostics.json`
  - `/tmp/neuralforecast-leak-audit-20260320/residual_probe_artifact_summary.json`
  - `/tmp/neuralforecast-leak-audit-20260320/residual_probe_boundary_summary.json`
  - `/tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe_lstm.yaml`
  - `/tmp/neuralforecast-leak-audit-20260320/residual_probe_lstm_boundary_summary.json`
- `residual/models.py:184-192`

### 8) Residual XGBoost feature frame에는 명백한 leakage 컬럼이 없음
Residual plugin feature는 다음 4개입니다.

- `horizon_step`
- `y_hat_base`
- `cutoff_day`
- `ds_day`

여기에는

- `fold_idx`
- `y`
- `residual_target`
- corrected target

같은 직접 누수 컬럼이 feature로 포함되지 않습니다.

주의할 점은 `ds_day` / `cutoff_day` 는 **미래 시점의 캘린더 정보**입니다. 하지만 forecast timestamp 자체는 예측 시점에 이미 알려진 값이므로 일반적인 시계열 설정에서는 leakage로 보지 않습니다.

- Evidence:
  - `residual/plugins/xgboost.py:47-115`
  - targeted pytest: `test_xgboost_plugin_predicts_panel_and_writes_checkpoint`

### 9) Optuna tuning 경로도 fold-local 평가 구조
#### Main model auto tuning
`_tune_main_job` 는 trial마다 각 fold에 대해 `_fit_and_predict_fold(...)` 를 다시 호출하고, fold MSE 평균을 objective로 사용합니다. 즉 tuning objective도 fold test 기준입니다.

여기에 대해 추가로 최소 단위 targeted test를 다시 돌렸습니다. 통과한 테스트는 다음을 보장합니다.

- `test_runtime_auto_mode_records_selector_provenance_and_modes`
  - learned-auto 실행 시 `requested_mode=learned_auto_requested`, `validated_mode=learned_auto`
  - `best_params.json`, `optuna_study_summary.json` artifact 생성
- `test_runtime_auto_mode_records_training_selector_provenance_and_artifacts`
  - training-auto 경로에서 `training_override` 가 실제 fold 실행에 전달됨
  - `training_best_params.json` 과 training study artifact 생성
- `test_effective_config_pins_val_size_to_horizon_for_training_auto`
  - training-auto일 때 `val_size` 가 `cv.horizon` 으로 고정되어 fold 경계를 유지

- Fresh verification:
  - `UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' uv run pytest --no-cov tests/test_residual_config.py -k 'test_load_app_config_marks_auto_requested_and_validated_modes or test_runtime_auto_mode_records_selector_provenance_and_modes or test_runtime_auto_mode_records_training_selector_provenance_and_artifacts or test_effective_config_pins_val_size_to_horizon_for_training_auto' -q`
  - result: `4 passed`

- Evidence:
  - `residual/runtime.py:421-503`
  - `tests/test_residual_config.py:1562-1625`
  - `tests/test_residual_config.py:1628-1724`
  - `tests/test_residual_config.py:1727-1751`

#### Residual auto tuning
`_score_residual_params` 는 이미 만들어진 `fold_payloads` 안의 `backcast_panel` / `eval_panel` 만 사용합니다. 또한 config normalization 테스트에서 residual auto 모드가 `requested_mode=residual_auto_requested`, `validated_mode=residual_auto` 로 정상 표기되는 것을 확인했습니다.

다만 residual auto에 대해서는 **end-to-end tiny runtime Optuna 재생산까지는 하지 않았습니다.** 그래서 결론은 **code+config evidence상 leak path not found** 이지만, **runtime-proven verdict는 partial** 로 남깁니다.

- Evidence:
  - `residual/runtime.py:506-533`
  - `tests/test_residual_config.py:1352-1373`

### 10) 진짜 조심할 점: evaluation overlap은 있음
현재 baseline config는

- `horizon=8`
- `step_size=4`

이므로 인접 fold test window가 4시점씩 겹칩니다.
실제 계산 결과도 `consecutive_test_overlap_sizes_first5 = [4, 4, 4, 4, 4]` 였습니다.

이건 **train leakage는 아닙니다.** 다만:

- fold 평균 metric이 일부 날짜를 중복 반영할 수 있고
- `cv.overlap_eval_policy: by_cutoff_mean` 이 config에 있어도 현재 runtime 검색 기준 사용처가 보이지 않았습니다

따라서 **모델 비교 metric 해석에는 주의가 필요**합니다.

- Evidence:
  - `/tmp/neuralforecast-leak-audit-20260320/baseline_split_summary.json`
  - `rg -n "overlap_eval_policy|by_cutoff_mean|overlap" residual neuralforecast tests`

## Commands run

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY'
from pathlib import Path
import json
import pandas as pd
from residual.config import load_app_config
from residual.runtime import _build_tscv_splits
repo = Path('.').resolve()
loaded = load_app_config(repo, config_path=repo/'baseline-brentoil.yaml')
df = pd.read_csv(repo / loaded.config.dataset.path)
df = df.sort_values(loaded.config.dataset.dt_col).reset_index(drop=True)
splits = _build_tscv_splits(len(df), loaded.config.cv)
summary = []
for fold_idx, (train_idx, test_idx) in enumerate(splits):
    train_dates = df.loc[train_idx, loaded.config.dataset.dt_col].tolist()
    test_dates = df.loc[test_idx, loaded.config.dataset.dt_col].tolist()
    summary.append({
        'fold_idx': fold_idx,
        'train_len': len(train_idx),
        'test_len': len(test_idx),
        'train_start': train_dates[0],
        'train_end': train_dates[-1],
        'test_start': test_dates[0],
        'test_end': test_dates[-1],
        'index_overlap': bool(set(train_idx).intersection(test_idx)),
    })
consecutive_test_overlap = []
for i in range(len(splits)-1):
    consecutive_test_overlap.append(len(set(splits[i][1]) & set(splits[i+1][1])))
result = {
    'rows': len(df),
    'cv': {
        'horizon': loaded.config.cv.horizon,
        'step_size': loaded.config.cv.step_size,
        'n_windows': loaded.config.cv.n_windows,
        'gap': loaded.config.cv.gap,
        'max_train_size': loaded.config.cv.max_train_size,
    },
    'folds': summary,
    'all_train_test_disjoint': all(not item['index_overlap'] for item in summary),
    'consecutive_test_overlap_sizes': consecutive_test_overlap,
}
Path('/tmp/neuralforecast-leak-audit-20260320/baseline_split_summary.json').write_text(
    json.dumps(result, indent=2), encoding='utf-8'
)
PY

UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' \
  uv run python main.py --validate-only --config baseline-brentoil.yaml \
  --jobs Naive --output-root /tmp/neuralforecast-leak-audit-20260320/baseline-validate-only

UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' \
  uv run python main.py --validate-only \
  --config /tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe.yaml \
  --jobs DLinear --output-root /tmp/neuralforecast-leak-audit-20260320/residual-probe-validate

UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' \
  uv run python main.py \
  --config /tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe.yaml \
  --jobs DLinear --output-root /tmp/neuralforecast-leak-audit-20260320/residual-probe-run

UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' \
  uv run python main.py \
  --config /tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe_lstm.yaml \
  --jobs LSTM --output-root /tmp/neuralforecast-leak-audit-20260320/residual-probe-lstm-run

UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' \
  uv run python main.py \
  --config /tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe_itransformer.yaml \
  --jobs iTransformer --output-root /tmp/neuralforecast-leak-audit-20260320/residual-probe-itransformer-run

UV_CACHE_DIR=/tmp/uv-cache uv run pytest --no-cov tests/test_residual_main.py -q

UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' \
  uv run pytest --no-cov tests/test_residual_config.py \
  -k 'test_load_app_config_marks_auto_requested_and_validated_modes or test_runtime_auto_mode_records_selector_provenance_and_modes or test_runtime_auto_mode_records_training_selector_provenance_and_artifacts or test_effective_config_pins_val_size_to_horizon_for_training_auto' -q

UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES='' \
  uv run pytest --no-cov tests/test_residual_main.py tests/test_residual_config.py \
  -k 'build_tscv_splits_uses_configured_step_size or runtime_outer_cv_cutoffs_follow_step_size or apply_residual_plugin_uses_fold_local_backcasts_only or xgboost_plugin_predicts_panel_and_writes_checkpoint' -q
```

## Artifacts
- `leak_check.md`
- `/tmp/neuralforecast-leak-audit-20260320/baseline_split_summary.json`
- `/tmp/neuralforecast-leak-audit-20260320/baseline_job_routing.json`
- `/tmp/neuralforecast-leak-audit-20260320/baseline-validate-only/config/capability_report.json`
- `/tmp/neuralforecast-leak-audit-20260320/baseline-validate-only/manifest/run_manifest.json`
- `/tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_mini.csv`
- `/tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe.yaml`
- `/tmp/neuralforecast-leak-audit-20260320/residual-probe-run/residual/DLinear/diagnostics.json`
- `/tmp/neuralforecast-leak-audit-20260320/residual_probe_artifact_summary.json`
- `/tmp/neuralforecast-leak-audit-20260320/residual_probe_boundary_summary.json`
- `/tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe_lstm.yaml`
- `/tmp/neuralforecast-leak-audit-20260320/residual_probe_lstm_boundary_summary.json`
- `/tmp/neuralforecast-leak-audit-20260320/residual-probe-lstm-run/residual/LSTM/diagnostics.json`
- `/tmp/neuralforecast-leak-audit-20260320/baseline_brentoil_residual_probe_itransformer.yaml`
- `/tmp/neuralforecast-leak-audit-20260320/itransformer_multivariate_adapter_summary.json`
- `/tmp/neuralforecast-leak-audit-20260320/residual_probe_itransformer_boundary_summary.json`
- `/tmp/neuralforecast-leak-audit-20260320/residual-probe-itransformer-run/residual/iTransformer/diagnostics.json`
- `/tmp/neuralforecast-leak-audit-20260320/test_residual_config_auto_refs.txt`

## Bottom line
현재 코드 기준으로는:

- **baseline path:** 미래 test row가 학습으로 직접 섞이는 leakage 증거 없음
- **residual path:** backcast는 outer train 내부, eval은 outer test만 사용 → leakage 증거 없음
- **Optuna path:** main learned-auto/training-auto는 targeted test + code 기준 문제 없음, residual-auto는 code+config 기준 문제 없음(단, end-to-end runtime은 partial)
- **주의점:** overlapping evaluation windows는 존재하므로, 이건 leakage가 아니라 **metric interpretation caveat** 로 봐야 함
