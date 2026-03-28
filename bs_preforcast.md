# bs_preforcast

이 문서는 현재 저장소에서 구현된 `bs_preforcast` 설정과 동작 방식을 설명합니다.

> 이름은 코드/설정과 동일하게 `bs_preforcast`로 유지합니다.
> Python 패키지 구현 경로는 `plugins/bs_preforcast/` 이고, import 경로는 `plugins.bs_preforcast...` 입니다. YAML 설정 경로 `yaml/plugins/...` 와는 별개입니다.

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
  config_path: yaml/plugins/bs_preforcast.yaml
```

### 필드 설명

- `enabled`
  - `true`면 stage1 활성화
  - `false`면 기존 main stage만 실행

- `config_path`
  - 독립 `bs_preforcast` 설정 YAML 경로
  - 생략하면 repo root의 `yaml/plugins/bs_preforcast.yaml` 사용

### `config_path`에 대해

현재 구현은 `config_path`를 **생략하면 기본값으로 repo root의 `yaml/plugins/bs_preforcast.yaml`**을 사용합니다.

즉 보통은 메인 YAML에 `enabled`만 두고, 다른 파일을 쓰고 싶을 때만 `config_path`를 명시합니다.

예:

```yaml
bs_preforcast:
  enabled: true
  config_path: yaml/plugins/custom_bs_preforcast.yaml
```

### 제거된 방식

이전 `routing.univariable_config`, `routing.multivariable_config` 방식은 제거되었습니다.
또한 main YAML에 `target_columns`, `task`를 직접 쓰는 방식도 허용되지 않습니다.

---

## 3. 독립 파일 `yaml/plugins/bs_preforcast.yaml`

권장 방식은 repo root의 `yaml/plugins/bs_preforcast.yaml`를 두는 것입니다.

현재 contract는 **plugin-only YAML** 입니다.  
즉 stage1 linked YAML은 아래만 소유합니다.

- top-level `bs_preforcast`
- `jobs`

그리고 main YAML의 아래 section을 그대로 상속합니다.

- `task`
- `dataset`
- `runtime`
- `training`
- `cv`
- `scheduler`
- `residual`

예:

```yaml
bs_preforcast:
  target_columns:
    - BS_Core_Index_Integrated
  task:
    multivariable: false
  exog_columns: []

jobs:
  - model: TimeXer
    params:
      patch_len: 16
      hidden_size: 768
      n_heads: 16
      e_layers: 4
      d_ff: 1024
```

### 독립 파일이 소유하는 것

- `bs_preforcast.target_columns`
- `bs_preforcast.exog_columns`
- `bs_preforcast.task.multivariable`
- `jobs`

### main YAML에서 상속하는 것

- `task`
- `dataset`
- `runtime`
- `training`
- `cv`
- `scheduler`
- `residual`

즉 linked YAML은 **bs 대상 컬럼 / exog 컬럼 / multivariable 여부 / stage1 job**만 결정하고, 나머지 실행 contract는 main YAML이 소유합니다.

---

## 4. 실행 규칙

### `task.multivariable: false`

- `target_columns` 각각에 대해 개별 stage run을 수행합니다.
- plugin-only contract에서는 linked YAML에 **정확히 1개의 fixed-param job**만 둡니다.
- shared catalog가 필요하면 `yaml/jobs/bs_preforcast/bs_preforcast_jobs_uni.yaml`에서 `yaml/jobs/bs_preforcast/uni/*.yaml` single-job route들로 fanout시키고, runtime은 route별 top-level run으로 분리 실행합니다.

### `task.multivariable: true`

- `target_columns` 전체를 하나의 stage run으로 처리합니다.
- direct-model (`ARIMA`, `ES`, `xgboost`, `lightgbm`) 경로는 multivariable mode를 지원하지 않고 즉시 실패합니다.
- shared catalog가 필요하면 `yaml/jobs/bs_preforcast/bs_preforcast_jobs_multi.yaml`에서 `yaml/jobs/bs_preforcast/multi/*.yaml` single-job route들로 fanout시키고, baseline-only stage job(`Naive` 등)은 catalog에 넣지 않습니다.

---

## 5. main stage 주입 방식

주입 모드는 main 모델 capability에 따라 자동 선택됩니다.

### A. `futr_exog`

조건:

- main 모델이 `futr_exog` 지원

동작:

- target 컬럼명을 그대로 유지
- 해당 컬럼을 `hist_exog_cols`에서 제거하고 `futr_exog_cols`로 이동
- train 구간은 과거 실제값을 유지하고, future 구간은 plugin 예측값으로 채움

예:

- `bs_a` -> 컬럼 이름은 그대로 `bs_a`
- futr-capable main 모델이면 `bs_a ∈ futr_exog_cols`, `bs_a ∉ hist_exog_cols`

### B. `lag_derived`

조건:

- main 모델이 `futr_exog` 미지원

동작:

- target 컬럼명을 그대로 유지
- 해당 컬럼을 `hist_exog_cols`에 둔 채 사용
- train의 마지막 horizon 구간과 future horizon 구간을 plugin 예측값 기준으로 overwrite
- 즉 main 모델은 같은 이름의 컬럼을 **lag/history feature처럼** 보게 됨

정리하면:

- futr 모드에서는 target 컬럼이 `futr_exog_cols`로 이동
- lag-derived 모드에서는 target 컬럼이 `hist_exog_cols`에 남음
- source YAML을 rewrite하지 않고 **runtime에서 effective exog membership만 바뀜**

### fail-fast 규칙

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

- `yaml/jobs/bs_preforcast_jobs_default.yaml`는 direct-model 후보/예시 레퍼런스이며, plugin-only linked YAML 자체는 그중 **정확히 1개의 fixed-param job**만 선택해 적어야 합니다.
- baseline-only job (`Naive`, `SeasonalNaive`, `HistoricAverage`)는 stage1 actual execution에서 지원하지 않음
- statistical / tree model은 direct-run execution 경로
- NF-native 모델은 기존 `main.py` runtime subprocess 경로 재사용

---

## 7. stage1 search-space

repo root `yaml/HPO/search_space.yaml`에 아래 section이 추가됩니다.

```yaml
bs_preforcast_models:
  ARIMA:
    order:
      type: categorical
      choices: ["[1, 0, 0]", "[1, 1, 0]", "[2, 1, 0]"]
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

- plugin-only direct stage는 linked YAML에 **정확히 1개의 fixed-param job**만 허용합니다.
- main `models` / `training` section과는 독립적으로 direct-model defaults를 관리합니다.
- 여러 direct-model 후보를 실험하려면 `yaml/jobs/bs_preforcast/bs_preforcast_jobs_uni.yaml` 같은 catalog에서 single-job route 파일들을 fanout시키거나, 별도 실험 family에서 config를 나눠 관리해야 합니다.

---

## 8. `ARIMA`, `ES`, tree direct 파라미터

현재 statistical stage model은 capability에 따라 자동 injection mode를 선택하고, tree direct model은 명시적 `lags` list를 지원합니다.

예:

```yaml
bs_preforcast_models:
  ARIMA:
    order:
      type: categorical
      choices: ["[1, 0, 0]", "[1, 1, 0]", "[2, 1, 0]"]
  ES:
  xgboost:
    lags:
      type: categorical
      choices:
        - "[1, 2, 3]"
        - "[1, 2, 3, 6, 12]"
```

실행 시 `xgboost` / `lightgbm`는 `skforecast.direct.ForecasterDirect`에 `lags`를 그대로 전달합니다.

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
```

main artifact에는 아래가 기록됩니다.

- `config_path`
- selected stage config path
- job별 injection result (`futr_exog` / `lag_derived`)
- target columns
- stage1 forecast artifact path
- metadata shell (`stage1_run_roots == []`, `stage1_selected_jobs_path == null`)

---

## 10. validate-only 동작

`--validate-only`일 때도 아래를 확인/기록합니다.

- main `bs_preforcast` config normalize
- `yaml/plugins/bs_preforcast.yaml` 존재 여부
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
- `yaml/plugins/bs_preforcast.yaml` 독립 파일 기반
- main YAML은 `enabled`, `config_path`만 소유
- linked YAML의 top-level `bs_preforcast`가 `target_columns`, `task.multivariable`를 소유
- `futr_exog` 지원 모델이면 future exog 주입
- main 모델 capability에 따라 job별 injection mode를 자동 선택
- futr 미지원 모델은 lag/history 쪽으로 주입
- statistical / tree / NF-native stage model 모두 지원 경로 존재
- plugin-only contract에서는 linked YAML에 정확히 1개의 fixed-param stage job만 둠
- 짧은 데이터 + 큰 horizon direct stage는 fail-fast
- tree direct stage short-history도 fail-fast
