# jobs defaults 레퍼런스

기준 파일: `yaml/jobs/bs_preforcast_jobs_default.yaml`

이 파일은 `bs_preforcast` stage에서 참조할 수 있는 **direct-model 기본 파라미터 anchor** 입니다. tuneable range가 아니라 baseline 성격의 고정 기본값입니다.

Source: `yaml/jobs/bs_preforcast_jobs_default.yaml:1-36`, `tests/test_bs_preforcast_config.py:330-439`

## 현재 내용

```yaml
- model: ARIMA
  params:
    order: [1, 1, 0]
    include_mean: true
    include_drift: false
- model: ES
  params:
    trend: add
    damped_trend: false
- model: xgboost
  params:
    lags: [1, 2, 3, 6, 12]
    n_estimators: 64
    max_depth: 4
    subsample: 1.0
    colsample_bytree: 1.0
- model: lightgbm
  params:
    lags: [1, 2, 3, 6, 12]
    n_estimators: 96
    max_depth: 6
    num_leaves: 31
    min_child_samples: 20
```

Source: `yaml/jobs/bs_preforcast_jobs_default.yaml:1-36`

## 무엇을 위한 파일인가

- **ARIMA / ES**: 통계 direct-model 기본값
- **xgboost / lightgbm**: lag 기반 tree-model 기본값
- repo 기본 plugin route는 이 defaults YAML을 사용해 네 개의 stage job을 materialize할 수 있습니다.

Source: `yaml/jobs/bs_preforcast_jobs_default.yaml:1-36`, `tests/test_bs_preforcast_config.py:330-439`

## 모델별 필드 설명

### `ARIMA`

현재 기본값:
- `order: [1, 1, 0]`
- `include_mean: true`
- `include_drift: false`

필드별 의미:
- `order`
  - 의미: ARIMA `(p, d, q)` order triplet
  - 현재 파일 값: `[1, 1, 0]`
  - 현재 repo search-space 기준 가능한 categorical 후보값: `[1, 0, 0]`, `[1, 1, 0]`, `[2, 1, 0]`
- `include_mean`
  - 의미: 평균항 포함 여부
  - 현재 파일 값: `true`
  - 가능한 categorical 값: `true`, `false`
- `include_drift`
  - 의미: drift 포함 여부
  - 현재 파일 값: `false`
  - 가능한 categorical 값: `false`, `true`

주의:
- stage에서는 `AutoARIMA`가 아니라 `ARIMA`를 사용해야 합니다.
- runtime은 `order`가 3개 숫자 리스트/튜플인지, 음수가 없는지 검사합니다.

Source: `yaml/jobs/bs_preforcast_jobs_default.yaml:1-8`, `yaml/HPO/search_space.yaml:401-424`, `plugins/bs_preforcast/plugin.py:201-205`, `plugins/bs_preforcast/runtime.py:354-359`, `plugins/bs_preforcast/runtime.py:587-598`

### `ES`

현재 기본값:
- `trend: add`
- `damped_trend: false`

필드별 의미:
- `trend`
  - 의미: 추세 항 사용 방식
  - 현재 파일 값: `add`
  - 현재 repo search-space 기준 가능한 categorical 값: `null`, `add`
- `damped_trend`
  - 의미: damped trend 사용 여부
  - 현재 파일 값: `false`
  - 가능한 categorical 값: `false`, `true`

주의:
- runtime은 seasonal component를 더 이상 허용하지 않습니다.

Source: `yaml/jobs/bs_preforcast_jobs_default.yaml:9-12`, `yaml/HPO/search_space.yaml:425-435`, `plugins/bs_preforcast/runtime.py:628-637`

### `xgboost`

현재 기본값:
- `lags: [1, 2, 3, 6, 12]`
- `n_estimators: 64`
- `max_depth: 4`
- `subsample: 1.0`
- `colsample_bytree: 1.0`

필드별 의미:
- `lags`
  - 의미: lag feature 집합
  - 현재 파일 값: `[1, 2, 3, 6, 12]`
  - 현재 repo search-space 기준 가능한 categorical 후보값: `[1, 2, 3]`, `[1, 2, 3, 6, 12]`, `[1, 2, 3, 6, 12, 24]`
- `n_estimators`
  - 의미: boosting tree 개수
  - 현재 파일 값: `64`
  - 가능한 categorical 값: `16`, `32`, `64`
- `max_depth`
  - 의미: tree 최대 깊이
  - 현재 파일 값: `4`
  - search-space 범위: `low=2`, `high=6`, `step=1`
- `subsample`
  - 의미: row subsampling 비율
  - 현재 파일 값: `1.0`
  - 참고: 이 값은 defaults 파일에는 있지만 현재 `bs_preforcast_models.xgboost` search-space에서는 별도 후보를 제공하지 않습니다.
- `colsample_bytree`
  - 의미: feature subsampling 비율
  - 현재 파일 값: `1.0`
  - 참고: 이 값도 defaults 파일에는 있지만 stage search-space에서는 별도 후보를 제공하지 않습니다.

주의:
- runtime은 `lags`를 양의 정수 list/int로 강제합니다.
- history 길이가 최대 lag보다 짧으면 direct tree stage가 실패할 수 있습니다.

Source: `yaml/jobs/bs_preforcast_jobs_default.yaml:13-24`, `yaml/HPO/search_space.yaml:436-464`, `plugins/bs_preforcast/runtime.py:364-390`, `plugins/bs_preforcast/runtime.py:658-677`

### `lightgbm`

현재 기본값:
- `lags: [1, 2, 3, 6, 12]`
- `n_estimators: 96`
- `max_depth: 6`
- `num_leaves: 31`
- `min_child_samples: 20`

필드별 의미:
- `lags`
  - 의미: lag feature 집합
  - 현재 파일 값: `[1, 2, 3, 6, 12]`
  - 현재 repo search-space 기준 가능한 categorical 후보값: `[1, 2, 3]`, `[1, 2, 3, 6, 12]`, `[1, 2, 3, 6, 12, 24]`
- `n_estimators`
  - 의미: boosting tree 개수
  - 현재 파일 값: `96`
  - 가능한 categorical 값: `32`, `64`, `96`
- `max_depth`
  - 의미: tree 최대 깊이
  - 현재 파일 값: `6`
  - 가능한 categorical 값: `4`, `6`, `-1`
- `num_leaves`
  - 의미: leaf 수 제한
  - 현재 파일 값: `31`
  - 가능한 categorical 값: `15`, `31`, `63`
- `min_child_samples`
  - 의미: child node 최소 샘플 수
  - 현재 파일 값: `20`
  - 가능한 categorical 값: `10`, `20`, `40`

주의:
- `feature_fraction`은 search-space에는 있지만 defaults 파일에는 없는 선택형 파라미터입니다.
- `lags` 제약은 xgboost와 동일하게 적용됩니다.

Source: `yaml/jobs/bs_preforcast_jobs_default.yaml:25-36`, `yaml/HPO/search_space.yaml:465-510`, `plugins/bs_preforcast/runtime.py:364-390`, `plugins/bs_preforcast/runtime.py:658-689`

## 언제 그대로 쓰고, 언제 바꾸나

그대로 쓰기 좋은 경우:
- direct-model stage를 baseline처럼 빠르게 검증할 때
- validate-only와 구조 검증이 우선일 때
- learned model보다 direct-model 비교 기준선이 필요할 때

바꾸는 것이 맞는 경우:
- 특정 target/horizon에 맞춰 lag 구조를 조정해야 할 때
- tree model 복잡도(`max_depth`, `num_leaves`)를 조절하고 싶을 때
- fixed params와 auto/search-space mode를 의도적으로 구분하고 싶을 때

## safe authoring notes

- jobs defaults는 **기본 anchor** 로 이해하는 것이 안전합니다.
- 실제 per-run override는 linked plugin YAML의 stage jobs 또는 별도 jobs YAML에서 일어날 수 있습니다.
- categorical 후보가 search-space에 정의된 필드는, defaults 값을 바꾸기 전에 그 후보 집합과의 관계를 같이 보세요.
- params를 비워 auto mode로 가는 경우 search-space 계약도 함께 확인해야 합니다.

Source: `app_config.py:1708-1735`, `yaml/HPO/search_space.yaml:401-510`

## common mistakes

- defaults 파일 값을 “유일한 허용값”으로 오해함
- `AutoARIMA`를 direct stage 모델로 넣으려 함
- `lags`를 문자열로 쓰거나 음수/0을 넣음
- defaults 파일과 search-space 후보값의 역할 차이를 구분하지 않음

## 관련 페이지

- [bs_preforcast plugin YAML 레퍼런스](YAML-Reference-bs_preforcast-plugin-YAML)
- [HPO search space 레퍼런스](YAML-Reference-HPO-search-space)
