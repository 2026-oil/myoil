# bs_preforcast

이 문서는 현재 저장소에서 구현된 `bs_preforcast` 설정과 실제 동작 방식을 설명합니다.

> 주의: 이름은 현재 코드/설정과 동일하게 `bs_preforcast`로 유지합니다.

---

## 1. 목적

`bs_preforcast`는 **main 예측 전에 별도 stage1을 먼저 실행**해서, 지정한 `bs_*` 계열 컬럼을 horizon 기준으로 예측한 뒤 그 결과를 main stage 입력에 주입하는 기능입니다.

현재 목적은 다음과 같습니다.

- `bs_*` 컬럼을 main 모델이 직접 원본 값만 보지 않게 하기
- stage1 결과를 main stage에 연결하기
- uni / multi route를 별도 YAML로 관리하기
- stage1도 별도 artifact / provenance / search-space를 가지게 하기

---

## 2. 메인 config에서 쓰는 설정

메인 YAML에 아래 top-level block을 둡니다.

```yaml
bs_preforcast:
  enabled: true
  config_path: bs_preforcast.yaml
  using_futr_exog: true
  target_columns:
    - bs_a
    - bs_b
  task:
    multivariable: false
```

### 필드 설명

- `enabled`
  - `true`면 stage1을 활성화합니다.
  - `false`면 기존 main 실행만 수행합니다.

- `using_futr_exog`
  - `true`면 **가능할 때** main 모델의 `futr_exog` 입력면을 사용합니다.
  - `false`면 `lag_derived` 경로를 사용합니다.

- `target_columns`
  - stage1이 예측할 대상 컬럼 목록입니다.
  - **자동 탐색하지 않습니다.**
  - main YAML이 이 목록의 **최종 소유자**입니다.

- `task.multivariable`
  - `false`: `target_columns`의 각 컬럼을 **독립 단변량 stage1 실행**
  - `true`: `target_columns` 전체를 **공동 다변량 stage1 실행**

- `config_path`
  - 독립 `bs_preforcast` 설정 파일 경로입니다.
  - 권장값은 repo root의 `bs_preforcast.yaml`입니다.

- `config_path`만 지원합니다.
  - 이전 `routing.univariable_config`, `routing.multivariable_config` 방식은 제거되었습니다.

---

## 3. 독립 `bs_preforcast.yaml`의 역할

권장 방식은 repo root의 `bs_preforcast.yaml`처럼 **독립 파일**을 두는 것입니다.

메인 YAML은 `config_path`로 이 파일을 가리키고, stage1 실행은 이 파일 안의 `common + (univariable|multivariable)` section merge 결과를 사용합니다.

예:

```yaml
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
    input_size: 64
    season_length: 52
    batch_size: 32
    learning_rate: 0.001
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

### 독립 `bs_preforcast.yaml`이 소유하는 것

- `dataset`
- `runtime`
- `training`
- `cv`
- `scheduler`
- `jobs`

### 독립 `bs_preforcast.yaml`이 소유하지 않는 것

- `bs_preforcast.target_columns`

즉, `bs_preforcast.yaml`은 stage1 실행 설정만 정의하고, **실제 어떤 `bs_*` 컬럼을 돌릴지는 main YAML이 결정**합니다.

또한 독립 config 파일 안에 다시 top-level `bs_preforcast:` block을 넣으면 오류가 납니다.

---

## 4. 독립 config 파일 내부 section 선택 규칙

### `task.multivariable: false`

- `config_path`로 지정한 독립 파일에서 `univariable` section을 선택
- `target_columns` 각각에 대해
  - temp config 생성
  - 개별 stage run 수행

즉:

- `bs_a`
- `bs_b`

두 개가 있으면 stage1 run도 2개 생깁니다.

### `task.multivariable: true`

- `config_path`로 지정한 독립 파일에서 `multivariable` section을 선택
- `target_columns` 전체를 하나의 stage run으로 처리합니다.

이때 stage payload에서는:

- 첫 번째 컬럼이 `dataset.target_col`
- 나머지가 `dataset.hist_exog_cols`

형태로 변환됩니다.

---

## 5. main stage 주입 방식

주입 모드는 두 가지입니다.

### A. `futr_exog`

조건:

- `using_futr_exog: true`
- 그리고 main 모델이 `futr_exog` 지원

동작:

- `bs_preforcast_futr__<column>` 형태의 새 컬럼 생성
- 이 컬럼을 `futr_exog_cols`로 주입

예:

- `bs_a` → `bs_preforcast_futr__bs_a`

### B. `lag_derived`

조건:

- `using_futr_exog: false`
- 또는 main 모델이 `futr_exog`를 지원하지 않음

동작:

- stage1 예측값을 새 컬럼으로 생성
- 이 컬럼을 `hist_exog_cols` 쪽으로 붙임
- 즉 main 모델은 이 값을 future exog가 아니라 **lag/history feature처럼** 보게 됩니다

현재 컬럼 이름은 futr 경로와 동일하게:

- `bs_preforcast_futr__<column>`

을 사용하지만,

- futr 모드에서는 `futr_exog_cols`
- lag-derived 모드에서는 `hist_exog_cols`

로 연결됩니다.

---

## 6. stage1 모델 후보군

현재 `SUPPORTED_BS_PREFORCAST_MODELS`에는 다음 범주가 포함됩니다.

### 통계

- `AutoARIMA`
- `ES`

### ML

- `xgboost`
- `lightgbm`

### NF 모델

- `LSTM`
- `TSMixerx`
- `TimeXer`
- `TFT`
- 기타 `SUPPORTED_AUTO_MODEL_NAMES`에 포함된 NF 모델들

### 주의

- stage1 actual execution에서는 **baseline-only job (`Naive`, `SeasonalNaive`, `HistoricAverage`)는 지원하지 않습니다.**
- statistical / tree model은 direct-run fallback 경로를 탑니다.
- NF-native 모델은 기존 `main.py` runtime subprocess 경로를 재사용합니다.

---

## 7. stage1 search-space

repo root `search_space.yaml`에 아래 두 section이 추가되었습니다.

```yaml
bs_preforcast_models:
  AutoARIMA:
    season_length:
      type: categorical
      choices: [1, 4, 8, 12]

bs_preforcast_training:
  global:
    input_size:
      type: categorical
      choices: [48, 64, 96]
```

### 규칙

- stage1 auto mode는 `bs_preforcast_models` / `bs_preforcast_training`만 봅니다.
- main `models` / `training` section과 분리되어 있습니다.
- route YAML의 `jobs[*].params`가 비어 있으면 stage1 auto mode가 됩니다.
- tuned params는 **run artifact에만 기록**되고 source YAML은 rewrite하지 않습니다.

---

## 8. `AutoARIMA`, `ES` 파라미터

현재 statistical stage model은 stage 전용 selector인 `stage_season_length`를 지원합니다.

예:

```yaml
bs_preforcast_models:
  AutoARIMA:
    stage_season_length:
      type: categorical
      choices: [1, 4, 8, 12]
  ES:
    stage_season_length:
      type: categorical
      choices: [1, 4, 8, 12]
```

실행 시에는 이 selector가 stage predictor 쪽 `season_length`로 매핑됩니다.
즉 search-space 이름은 main/runtime training knob와 충돌하지 않도록 분리되고,
실제 statistical model 실행 시점에만 `season_length`로 해석됩니다.

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

main artifact에도 아래 정보가 기록됩니다.

- selected route path
- injection mode (`futr_exog` / `lag_derived`)
- target columns
- stage1 artifact paths
- stage1 run roots
- stage1 best params artifact 소비 경로

---

## 10. validate-only 동작

`--validate-only`일 때도 아래는 확인/기록됩니다.

- main `bs_preforcast` config normalize
- route YAML 존재 여부
- stage1 config normalize
- stage1 dedicated search-space
- main/stage1 provenance
- dashboard / forecast artifact 경로

즉 실제 full training 없이도 구조 검증이 가능합니다.

---

## 11. direct stage variant의 fail-fast 규칙

`bs_preforcast` direct stage variant는

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

stage statistical model 지원을 위해 다음 dependency가 추가되었습니다.

- `statsforecast`
- `statsmodels`

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

- stage1 route 실행
- stage1 artifact 생성
- main stage에 futr 또는 lag-derived 형태로 주입

---

## 14. 현재 구현 기준 요약

- `bs_preforcast`는 main 앞단 stage1
- route YAML은 stage1 실행 설정
- main YAML이 `target_columns`를 소유
- `futr_exog` 지원 모델이면 future exog 주입
- 그렇지 않으면 lag/history 쪽으로 주입
- statistical / tree / NF-native stage model 모두 지원 경로 존재
- learned-auto stage job은 materialized `best_params.json`을 fold-time injection에서 재사용
- 짧은 데이터 + 큰 horizon direct stage는 fail-fast

## 15. 현재 남아 있는 제한

현재 구현 기준으로는 다음이 남아 있습니다.

- `AutoARIMA` / `ES`의 **learned_auto non-validate runtime path**는 validate-only/selector 수준보다 더 넓은 실증이 아직 부족합니다.

즉,

- config normalization
- validate-only artifact 생성
- fold-time `best_params.json` 소비

까지는 검증되어 있지만,

- full non-validate 장기 실행 경로는 후속 검증 여지가 있습니다.
