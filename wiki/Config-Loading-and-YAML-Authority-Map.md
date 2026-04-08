# 설정 로딩과 YAML 소유권 맵

이 페이지는 **YAML이 어떻게 runtime state로 바뀌는지**, 그리고 **어느 파일이 어떤 설정을 소유하는지**를 설명합니다.

## 1. 정규화 결과물: `AppConfig`와 `LoadedConfig`

`AppConfig`는 YAML 내용을 typed config로 정규화한 결과물입니다.
`LoadedConfig`는 여기에 source path, hash, search-space metadata, stage plugin loaded payload 등을 추가한 실행용 구조입니다.

Source: `app_config.py:283-350`

## 2. shared settings merge

YAML app config는 필요 시 `yaml/setting/setting.yaml`의 shared setting을 병합합니다. 운영자가 `--setting`으로 명시 지정할 수도 있고, repo 기본 shared settings가 자동 사용될 수도 있습니다.

Source: `app_config.py:1635-1666`

## 3. jobs resolution / fanout

`jobs`는 inline job list일 수도 있고, 외부 YAML 참조일 수도 있습니다. loader는 이를 해석해 fanout spec 또는 concrete jobs payload로 바꿉니다.

Source: `app_config.py:1667-1707`

## 4. search space 자동 요청 규칙

loader는 search space를 항상 로드하지 않습니다. 다음 같은 경우 `yaml/HPO/search_space.yaml`이 필요하다고 판단할 수 있습니다.

- non-baseline job의 `params`가 비어 있음
- probed stage job의 `params`가 비어 있음

Source: `app_config.py:1708-1735`

## 5. stage plugin probe/load

payload에 plugin key가 있으면 registry가 active plugin을 찾고, plugin이 route를 검증한 뒤, 필요 시 linked YAML과 search space까지 포함해 stage를 load합니다.

Source: `app_config.py:1672-1692`, `app_config.py:1749-1786`, `plugin_contracts/stage_registry.py:45-50`

## 6. normalized payload와 metadata

최종적으로 loader는 아래 같은 metadata를 포함한 normalized payload를 구성합니다.

- resolved config content
- search space path / sha256
- shared settings path / sha256
- stage plugin metadata

Source: `app_config.py:1787-1814`

## ownership 표

| 파일/영역 | 누가 소유? | 의미 |
|---|---|---|
| 메인 실험 YAML | 운영자/실험 정의 | 실행 조립, dataset, jobs 연결, plugin on/off |
| `yaml/setting/setting.yaml` | shared defaults | 공통 runtime/training/cv/scheduler 값 |
| `yaml/plugins/*.yaml` | plugin-linked config | plugin 세부 필드와 stage jobs |
| `yaml/jobs/*.yaml` | jobs catalog/defaults | 모델/params 묶음 |
| `yaml/HPO/search_space.yaml` | tuning contract | auto mode일 때 탐색 가능한 값 범위 |

## 운영자가 자주 착각하는 지점

### 착각 1) 메인 YAML이 모든 세부 설정을 다 소유한다고 생각함
아닙니다. 메인 YAML은 composition root이고, shared settings / plugin linked YAML / jobs defaults / HPO search space가 각자 ownership을 가집니다.

### 착각 2) search space가 항상 로드된다고 생각함
아닙니다. auto tuning이 필요한 경우에만 의미가 생깁니다.

### 착각 3) plugin validation은 실행 시점에만 터진다고 생각함
많은 오류는 `load_app_config()` 단계에서 fail-fast로 차단됩니다.

## 다음에 볼 페이지

- [main.py 실행 흐름](Execution-Flow-from-main.py)
- [플러그인 구조](Plugin-Architecture)
- [메인 실험 YAML 안전하게 작성하기](Authoring-a-Main-Experiment-YAML-Safely)
