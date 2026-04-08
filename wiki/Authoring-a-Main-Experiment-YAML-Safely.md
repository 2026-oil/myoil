# 메인 실험 YAML 안전하게 작성하기

이 페이지는 **운영자가 실제로 처음 복사하고 수정해야 하는 메인 실험 YAML**을 기준으로 설명합니다.

## 보통 가장 먼저 수정하는 파일

대표 예시:

```yaml
yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml
```

이 파일에는 현재 repo의 composition-root 패턴이 그대로 들어 있습니다.

- `task.name`: 실행 식별자
- `dataset.*`: 데이터 경로와 컬럼
- `jobs`: 별도 jobs YAML 연결
- `bs_preforcast.enabled` + `config_path`: linked plugin YAML 활성화

Source: `yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml:1-31`

## 메인 실험 YAML이 의미하는 것

메인 실험 YAML은 실행의 **조립 지점(composition root)** 입니다. 정규화 이후에는 `AppConfig`의 최상위 구조로 해석됩니다.

포함되는 주요 영역:
- `task`
- `dataset`
- `runtime`
- `training`
- `training_search`
- `cv`
- `scheduler`
- `jobs`
- `stage_plugin_config`

Source: `app_config.py:283-295`

## 안전한 이해 방식

메인 YAML은 다음을 담당합니다.

- 이번 run의 정체성을 정함
- 메인 dataset과 jobs/plugin 활성화를 정함
- 어떤 linked plugin YAML을 쓸지 정함
- jobs 및 search/tuning 입력을 가리킴
- 하지만 모든 세부 필드를 한 파일에 다 넣을 필요는 없음

## 무엇이 무엇을 소유하나

| 파일/블록 | 소유 영역 | 메모 |
|---|---|---|
| 메인 실험 YAML | `task`, `dataset`, `jobs`, plugin activation | 가장 먼저 복사/수정할 파일 |
| `yaml/setting/setting.yaml` | shared `runtime`, `training`, `cv`, `scheduler` 기본값 | 공통 기본값 레이어 |
| `yaml/plugins/bs_preforcast*.yaml` | linked stage-plugin 설정 | stage-specific 필드와 stage jobs |
| `yaml/jobs/bs_preforcast_jobs_default.yaml` | direct-model 기본값 | `ARIMA`, `ES`, `xgboost`, `lightgbm` 기준값 |
| `yaml/HPO/search_space.yaml` | auto-tuning 후보 공간 | params를 비웠을 때 탐색 계약으로 사용 |

Source: `yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml:1-31`, `yaml/setting/setting.yaml:1-37`, `yaml/plugins/bs_preforcast.yaml:1-17`, `yaml/jobs/bs_preforcast_jobs_default.yaml:1-36`, `yaml/HPO/search_space.yaml:1-715`, `app_config.py:68-95`, `app_config.py:1708-1755`

## `bs_preforcast`를 켤 때 안전한 절차

메인 YAML에서는 보통 아래처럼 plugin을 활성화만 합니다.

```yaml
bs_preforcast:
  enabled: true
  config_path: yaml/plugins/bs_preforcast.yaml
```

실제 stage 세부 설정은 linked plugin YAML에서 관리합니다.

Source: `README.md:156-171`, `plugins/bs_preforcast/plugin.py:69-118`

## validate-only 루프

실행 전에 가장 먼저 권장되는 명령은 다음입니다.

```bash
uv run python main.py --validate-only --config yaml/experiment/feature_set/brentoil-case1.yaml
uv run python main.py --validate-only --config yaml/experiment/feature_set/brentoil-case1.yaml --setting yaml/setting/setting.yaml
uv run python main.py --validate-only --config yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml
```

이 명령들은 현재 README가 권장하는 기본 검증 루프입니다. 실제 성공 여부는 실행 시점 환경에서 다시 확인해야 합니다.

Source: `README.md:39-49`, `README.md:85-97`, `yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml:1-31`

### validate-only가 보장하는 것

성공한 validate-only는 적어도 다음을 뜻합니다.

- YAML 파싱 가능
- config 정규화 가능
- linked path 해석 가능
- plugin route validation 통과
- full training 없이 runtime payload 구성 가능

Source: `main.py:67-119`, `app_config.py:1620-1814`

## Safe authoring checklist

1. 검증된 기존 메인 YAML 하나를 기준 파일로 고른다.
2. 변경하려는 값이 어느 파일의 ownership인지 먼저 확인한다.
3. linked plugin YAML과 메인 YAML의 역할을 섞지 않는다.
4. validate-only를 먼저 돌린다.
5. fail-fast 오류를 모두 없앤 뒤 실제 실행한다.

## 자주 하는 실수

### 1) 메인 YAML에 plugin 세부 필드를 다 넣으려는 경우
메인 YAML은 activation 중심이고, plugin 세부는 linked YAML이 소유합니다.

### 2) `jobs`와 stage jobs를 같은 레이어로 생각하는 경우
메인 jobs와 plugin-linked stage jobs는 역할이 다를 수 있습니다.

### 3) validate-only 없이 바로 실행하는 경우
오류를 늦게 발견하게 됩니다. 먼저 validate-only를 통과시키는 것이 안전합니다.

## 다음에 볼 페이지

- [main.py 실행 흐름](Execution-Flow-from-main.py)
- [설정 로딩과 YAML 소유권 맵](Config-Loading-and-YAML-Authority-Map)
- [플러그인 구조](Plugin-Architecture)
- [검증 절차와 주요 fail-fast 오류](Validation-Workflow-and-Common-Fail-Fast-Errors)
