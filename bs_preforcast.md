# bs_preforcast

이 문서는 현재 저장소에서 구현된 `bs_preforcast` 설정과 동작 방식을 설명합니다.

> 이름은 코드/설정과 동일하게 `bs_preforcast`로 유지합니다.

---

## 1. 목적

`bs_preforcast`는 **main 예측 전에 별도 stage1을 먼저 실행**해서, 지정한 `bs_*` 계열 컬럼을 먼저 예측한 뒤 그 결과를 main stage 입력에 주입하는 기능입니다.

주요 목적:

- `bs_*` 계열 입력을 main stage 앞단에서 별도 모델로 예측
- 예측 결과를 main stage에 연결
- stage1을 독립 설정 파일로 관리
- stage1 전용 artifact / provenance / search-space 유지

---

## 2. 메인 YAML에서 쓰는 설정

메인 YAML에서는 아래 top-level block만 둡니다.

```yaml
bs_preforcast:
  enabled: true
  config_path: bs_preforcast.yaml
```

### 필드 설명

- `enabled`
  - `true`면 stage1 활성화
  - `false`면 기존 main stage만 실행

- `config_path`
  - 독립 `bs_preforcast` 설정 YAML 경로
  - 생략하면 repo root의 `bs_preforcast.yaml` 사용

### `config_path`에 대해

현재 구현은 `config_path`를 **생략하면 기본값으로 repo root의 `bs_preforcast.yaml`**을 사용합니다.

즉 보통은 메인 YAML에 `enabled`만 두고, 다른 파일을 쓰고 싶을 때만 `config_path`를 명시합니다.

예:

```yaml
bs_preforcast:
  enabled: true
  config_path: custom_bs_preforcast.yaml
```

### 제거된 방식

이전 `routing.univariable_config`, `routing.multivariable_config` 방식은 제거되었습니다.
또한 main YAML에 `using_futr_exog`, `target_columns`, `task`를 직접 쓰는 방식도 허용되지 않습니다.

---

## 3. 독립 파일 `bs_preforcast.yaml`

권장 방식은 repo root의 `bs_preforcast.yaml`를 두는 것입니다.

메인 YAML은 이 파일을 자동으로 읽고, stage1 실행 시 이 파일 안의

- top-level `bs_preforcast`
- `common`
- `univariable`
- `multivariable`

section을 merge해서 사용합니다.

예:

```yaml
bs_preforcast:
  using_futr_exog: true
  target_columns:
    - bs_a
    - bs_b
  task:
    multivariable: false

common:
  dataset:
    path: data/df.csv
    dt_col: dt
    hist_exog_cols: []
    futr_exog_cols: []
    static_exog_cols: []
  runtime:
    random_seed: 1
  training:
    train_protocol: expanding_window_tscv
    input_size: 64
    season_length: 52
    batch_size: 32
    valid_batch_size: 64
    windows_batch_size: 1024
    inference_windows_batch_size: 1024
    learning_rate: 0.001
    model_step_size: 8
    max_steps: 200
    val_size: 8
    val_check_steps: 50
    early_stop_patience_steps: 5
    loss: mse
  cv:
    horizon: 8
    step_size: 8
    n_windows: 12
    gap: 0
    overlap_eval_policy: by_cutoff_mean
  scheduler:
    gpu_ids: [0, 1]
    max_concurrent_jobs: 2
    worker_devices: 1
    parallelize_single_job_tuning: false
  residual:
    enabled: false
    model: xgboost
    params: {}

univariable:
  task:
    name: bs_preforcast_univariable
  dataset:
    target_col: BS_Core_Index_A
  jobs:
    - model: TimeXer
      params: {}

multivariable:
  task:
    name: bs_preforcast_multivariable
  dataset:
    target_col: BS_Core_Index_A
    hist_exog_cols: [BS_Core_Index_B]
  jobs:
    - model: TimeXer
      params: {}
```

### 독립 파일이 소유하는 것

- `bs_preforcast.using_futr_exog`
- `bs_preforcast.target_columns`
- `bs_preforcast.task.multivariable`
- `dataset`
- `runtime`
- `training`
- `cv`
- `scheduler`
- `jobs`

즉 실제 어떤 `bs_*` 컬럼을 돌릴지와 `futr_exog`/`multivariable` 의도까지 모두 linked YAML이 결정합니다.

---

## 4. section 선택 규칙

### `task.multivariable: false`

- `common + univariable`를 merge
- `target_columns` 각각에 대해 개별 stage run 수행

예:

```yaml
target_columns:
  - bs_a
  - bs_b
```

이면 stage1 run은 2개 생깁니다.

### `task.multivariable: true`

- `common + multivariable`를 merge
- `target_columns` 전체를 하나의 stage run으로 처리

이때 내부 stage payload는:

- 첫 번째 target → `dataset.target_col`
- 나머지 target → `dataset.hist_exog_cols`

로 구성됩니다.

---

## 5. main stage 주입 방식

주입 모드는 두 가지입니다.

### A. `futr_exog`

조건:

- `using_futr_exog: true`
- 그리고 main 모델이 `futr_exog` 지원

동작:

- `bs_preforcast_futr__<column>` 컬럼 생성
- 이를 `futr_exog_cols`로 주입

예:

- `bs_a` -> `bs_preforcast_futr__bs_a`

### B. `lag_derived`

조건:

- `using_futr_exog: false`

동작:

- stage1 예측값을 새 컬럼으로 생성
- 이 컬럼을 `hist_exog_cols` 쪽으로 붙임
- 즉 main 모델은 이를 future exog가 아니라 **lag/history feature처럼** 보게 됨

현재 컬럼 이름은 futr 경로와 동일하게:

- `bs_preforcast_futr__<column>`

을 사용하지만,

- futr 모드에서는 `futr_exog_cols`
- lag-derived 모드에서는 `hist_exog_cols`

로 연결됩니다.

### fail-fast 규칙

- `using_futr_exog: true`인데 main 모델이 `futr_exog`를 지원하지 않으면 `lag_derived`로 내리지 않고 즉시 실패합니다.
- stage1 forecast 값이 없거나 비어 있으면 마지막 값 대체를 하지 않고 즉시 실패합니다.
- tree direct stage가 예측에 필요한 최소 history를 못 가지면 마지막 값 대체 없이 즉시 실패합니다.

---

## 6. stage1 모델 후보군

현재 `SUPPORTED_BS_PREFORCAST_MODELS` 범위:

### 통계

- `ARIMA`
- `ES`

### ML

- `xgboost`
- `lightgbm`

### NF 모델

- `LSTM`
- `TSMixerx`
- `TimeXer`
- `TFT`
- 기타 `SUPPORTED_AUTO_MODEL_NAMES`에 포함된 NF 모델

### 주의

- univariable direct-model 기본값은 `yaml/bs_preforcast_jobs_default.yaml`가 소유합니다.
- baseline-only job (`Naive`, `SeasonalNaive`, `HistoricAverage`)는 stage1 actual execution에서 지원하지 않음
- statistical / tree model은 direct-run execution 경로
- NF-native 모델은 기존 `main.py` runtime subprocess 경로 재사용

---

## 7. stage1 search-space

repo root `search_space.yaml`에 아래 section이 추가됩니다.

```yaml
bs_preforcast_models:
  ARIMA:
    order:
      type: categorical
      choices: ["[1, 0, 0]", "[1, 1, 0]", "[2, 1, 0]"]
    season_length:
      type: categorical
      choices: [1, 4, 8, 12]
  xgboost:
    lags:
      type: categorical
      choices:
        - "[1, 2, 3]"
        - "[1, 2, 3, 6, 12]"
        - "[1, 2, 3, 6, 12, 24]"

bs_preforcast_training:
  global:
    input_size:
      type: categorical
      choices: [48, 64, 96]
```

### 규칙

- stage1 auto mode는 `bs_preforcast_models` / `bs_preforcast_training`만 참조
- main `models` / `training` section과 분리
- stage1 `jobs[*].params`가 비어 있으면 auto mode
- tuned params는 run artifact에만 저장되고 source YAML은 rewrite하지 않음

---

## 8. `ARIMA`, `ES`, tree direct 파라미터

현재 statistical stage model은 `season_length` selector를 지원하고, tree direct model은 명시적 `lags` list를 지원합니다.

예:

```yaml
bs_preforcast_models:
  ARIMA:
    order:
      type: categorical
      choices: ["[1, 0, 0]", "[1, 1, 0]", "[2, 1, 0]"]
    season_length:
      type: categorical
      choices: [1, 4, 8, 12]
  ES:
    season_length:
      type: categorical
      choices: [1, 4, 8, 12]
  xgboost:
    lags:
      type: categorical
      choices:
        - "[1, 2, 3]"
        - "[1, 2, 3, 6, 12]"
```

실행 시 statistical model은 direct predictor의 `season_length`를 사용하고,
`xgboost` / `lightgbm`는 `skforecast.direct.ForecasterDirect`에 `lags`를 그대로 전달합니다.

---

## 9. artifact / provenance

`bs_preforcast.enabled: true`면 main run root 아래에 stage subtree가 생깁니다.

예:

```text
runs/<task>/bs_preforcast/
  config/
    config.resolved.json
    capability_report.json
  manifest/
    run_manifest.json
  summary/
    dashboard.md
  artifacts/
    bs_preforcast_forecasts.csv
  runs/
    <target-or-multivariable>/
```

main artifact에는 아래가 기록됩니다.

- `config_path`
- selected stage config path
- injection mode (`futr_exog` / `lag_derived`)
- target columns
- stage1 artifact paths
- stage1 run roots
- stage1 best params 소비 경로

---

## 10. validate-only 동작

`--validate-only`일 때도 아래를 확인/기록합니다.

- main `bs_preforcast` config normalize
- `bs_preforcast.yaml` 존재 여부
- stage1 config normalize
- stage1 dedicated search-space
- main/stage1 provenance
- dashboard / forecast artifact 경로

즉 full training 없이도 구조 검증이 가능합니다.

---

## 11. direct stage variant fail-fast 규칙

direct stage variant는

- 데이터 길이 `<= horizon`

이면 자동 보정하지 않고 바로 오류를 냅니다.

오류 메시지:

```text
bs_preforcast stage direct run needs more than horizon rows
```

즉,

- 짧은 데이터
- 큰 horizon

조합은 **명시적 오류가 정답**입니다.

---

## 12. dependency

stage statistical model 지원을 위해:

- `statsforecast`
- `statsmodels`

를 사용합니다.

또한 stage ML model을 위해:

- `xgboost`
- `lightgbm`

도 사용합니다.

환경 반영:

```bash
uv sync
```

---

## 13. 운영 예시

### validate-only

```bash
uv run python main.py --config path/to/main.yaml --validate-only
```

### 특정 main job 실행

```bash
uv run python main.py --config path/to/main.yaml --jobs DLinear --output-root runs/bs-preforcast-smoke
```

### 기대 효과

- stage1 실행
- stage1 artifact 생성
- main stage에 futr 또는 lag-derived 형태로 주입

---

## 14. 현재 구현 기준 요약

- `bs_preforcast`는 main 앞단 stage1
- `bs_preforcast.yaml` 독립 파일 기반
- main YAML은 `enabled`, `config_path`만 소유
- linked YAML의 top-level `bs_preforcast`가 `using_futr_exog`, `target_columns`, `task.multivariable`를 소유
- `futr_exog` 지원 모델이면 future exog 주입
- `using_futr_exog: true`인데 지원하지 않으면 fail-fast
- `using_futr_exog: false`일 때만 lag/history 쪽으로 주입
- statistical / tree / NF-native stage model 모두 지원 경로 존재
- learned-auto stage job은 materialized `best_params.json`을 fold-time injection에서 재사용
- 짧은 데이터 + 큰 horizon direct stage는 fail-fast
- tree direct stage short-history도 fail-fast
