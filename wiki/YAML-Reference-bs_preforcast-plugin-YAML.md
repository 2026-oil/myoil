# bs_preforcast plugin YAML 레퍼런스

대상 파일:
- `yaml/plugins/bs_preforcast.yaml`
- `yaml/plugins/bs_preforcast_multi.yaml`
- `yaml/plugins/bs_preforcast_uni.yaml`

이 파일군은 **plugin-linked YAML** 입니다. 메인 YAML에서 plugin을 켠 뒤, 실제 stage 세부 설정은 이쪽이 소유합니다.

Source: `yaml/plugins/bs_preforcast.yaml:1-17`, `yaml/plugins/bs_preforcast_multi.yaml:1-14`, `yaml/plugins/bs_preforcast_uni.yaml:1-7`

## ownership 경계

| surface | 소유 주체 | 메모 |
|---|---|---|
| 메인 YAML의 `bs_preforcast.enabled`, `bs_preforcast.config_path` | composition-root YAML | plugin on/off와 linked file 선택 |
| linked YAML의 `target_columns`, `task.multivariable`, `hist_columns`, `jobs` | plugin YAML | stage-specific authoring surface |
| 공통 runtime/training/cv/scheduler 기본값 | `yaml/setting/setting.yaml` | 별도 shared-default layer |

Source: `yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml:24-31`, `yaml/plugins/bs_preforcast.yaml:1-17`, `app_config.py:1639-1667`

## 현재 inline 예시

```yaml
bs_preforcast:
  target_columns:
    - BS_Core_Index_Integrated
  task:
    multivariable: false
  hist_columns: []
jobs:
  - model: TimeXer
    params:
      patch_len: 16
      hidden_size: 768
      n_heads: 16
      e_layers: 4
      d_ff: 1024
      factor: 8
      dropout: 0.2
      use_norm: true
```

Source: `yaml/plugins/bs_preforcast.yaml:1-17`

## 필드 레퍼런스

### `bs_preforcast.target_columns`
- stage1에서 먼저 예측할 target 컬럼 목록
- 현재 repo 예시 값: `BS_Core_Index_Integrated`
- 타입: 문자열 리스트

### `bs_preforcast.task.multivariable`
- multivariable stage 실행 여부
- 가능한 categorical 값: `true`, `false`
- 현재 대표 예시:
  - `yaml/plugins/bs_preforcast_multi.yaml` → `true`
  - `yaml/plugins/bs_preforcast_uni.yaml` / `bs_preforcast.yaml` → `false`

### `bs_preforcast.hist_columns`
- stage 학습에 함께 쓰는 history feature 컬럼
- 타입: 문자열 리스트
- 제약: `target_columns`와 겹치면 fail-fast

Source: `yaml/plugins/bs_preforcast_multi.yaml:1-14`, `yaml/plugins/bs_preforcast_uni.yaml:1-7`, `plugins/bs_preforcast/config.py:121-125`, `tests/test_bs_preforcast_config.py:234-266`

### `jobs[*].model`
- stage1에서 사용할 모델명
- 현재 repo에서 직접 확인 가능한 categorical 값:
  - direct-model 쪽: `ARIMA`, `ES`, `xgboost`, `lightgbm`
  - learned/stage model search-space 쪽: `LSTM`, `NHITS`, `TSMixerx`, `TimeXer`, `TFT`
- 금지 값: `AutoARIMA`

Source: `yaml/jobs/bs_preforcast_jobs_default.yaml:1-36`, `yaml/HPO/search_space.yaml:401-651`, `plugins/bs_preforcast/plugin.py:201-205`, `tests/test_bs_preforcast_config.py:635-663`

### `jobs[*].params`
- stage model별 파라미터
- direct-model은 jobs defaults 또는 search-space 후보와 연결될 수 있음
- learned model은 `bs_preforcast_models`, `bs_preforcast_training`과 연결될 수 있음

## 주요 fail-fast 규칙

### 1) linked route must exist
`config_path`가 가리키는 파일이 없으면 로딩 단계에서 실패합니다.

### 2) top-level `bs_preforcast` block required
linked YAML에 해당 블록이 없으면 실패합니다.

### 3) `hist_columns` / `target_columns` overlap forbidden

### 4) `AutoARIMA` forbidden in stage

### 5) complex search-space choices must be native YAML lists
문자열로 리스트를 흉내 내면 안 됩니다.

Source: `plugins/bs_preforcast/plugin.py:87-105`, `plugins/bs_preforcast/config.py:121-149`, `tests/test_bs_preforcast_config.py:442-570`, `tests/test_bs_preforcast_config.py:614-615`

## safe authoring notes

- 메인 YAML은 `enabled`와 `config_path` 중심으로 두고, 상세는 linked YAML에 유지하세요.
- `task.multivariable`은 `true/false` 이진 분기이므로, stage 목적에 맞게 명시적으로 선택하세요.
- `jobs[*].model`은 현재 repo가 실제로 지원하는 값 집합 안에서만 고르는 것이 안전합니다.
- 수정 후에는 validate-only를 먼저 실행하세요.

## 관련 페이지

- [메인 실험 YAML 안전하게 작성하기](Authoring-a-Main-Experiment-YAML-Safely)
- [플러그인 구조](Plugin-Architecture)
- [검증 절차와 주요 fail-fast 오류](Validation-Workflow-and-Common-Fail-Fast-Errors)
- [HPO search space 레퍼런스](YAML-Reference-HPO-search-space)
