# NeuralForecast 운영자 위키

이 위키는 **운영자가 YAML/config를 안전하게 작성·수정·검증**할 수 있도록 만든 실무 중심 문서입니다.

이 저장소는 단순한 upstream `neuralforecast` 패키지가 아니라, 다음을 함께 포함한 **실험/실행 하네스**입니다.

- `main.py` 기반 실행 진입점
- `app_config.py` 기반 설정 정규화/검증
- `bs_preforcast` stage plugin
- `yaml/setting`, `yaml/plugins`, `yaml/jobs`, `yaml/HPO/search_space` 기반 설정 체계

Source: `main.py:46-135`, `app_config.py:283-350`, `app_config.py:1619-1814`

## 먼저 읽을 페이지

1. **[메인 실험 YAML 안전하게 작성하기](Authoring-a-Main-Experiment-YAML-Safely)**
2. **[검증 절차와 주요 fail-fast 오류](Validation-Workflow-and-Common-Fail-Fast-Errors)**

## 핵심 흐름 문서

- **[main.py 실행 흐름](Execution-Flow-from-main.py)**
- **[설정 로딩과 YAML 소유권 맵](Config-Loading-and-YAML-Authority-Map)**
- **[플러그인 구조](Plugin-Architecture)**

## YAML 레퍼런스 문서

- **[setting.yaml 레퍼런스](YAML-Reference-setting.yaml)**
- **[bs_preforcast plugin YAML 레퍼런스](YAML-Reference-bs_preforcast-plugin-YAML)**
- **[jobs defaults 레퍼런스](YAML-Reference-jobs-defaults)**
- **[HPO search space 레퍼런스](YAML-Reference-HPO-search-space)**

## 이 위키가 다루는 것

- 메인 실험 YAML을 어디서 시작해서 어떻게 수정해야 하는지
- `main.py`가 어떻게 config loading과 runtime dispatch로 이어지는지
- linked plugin YAML이 어떻게 연결되는지
- `setting.yaml`, plugin YAML, jobs defaults, `yaml/HPO/search_space.yaml`의 역할 차이
- validate-only가 무엇을 보장하고 어떤 오류를 미리 막는지

## 이 위키가 1차로 다루지 않는 것

- `neuralforecast/` 모델 내부 구현 상세
- `tests/` 자체 설명
- `experiments/` 문서화
- `runs/`, `lightning_logs/` 등 생성 산출물 설명
- 제거된 legacy feature의 내부 설명

## 빠른 지도

- **작성 시작점:** 메인 실험 YAML
- **실행 흐름:** `main.py` → `load_app_config()` → `run_loaded_config()`
- **검증 루프:** validate-only → fail-fast 오류 수정 → 실제 실행
- **세부 필드 조회:** shared settings / plugin YAML / jobs defaults / HPO search space

## 소스 앵커

- `main.py:36-43` — 공개 CLI에서 `--output-root` 차단
- `main.py:67-119` — config load 및 jobs fanout handoff
- `README.md:39-49` — validate-only 사용법
- `README.md:156-184` — `bs_preforcast` 및 대체 YAML 작성 경로
- `yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml:1-31` — 실제 composition-root 예시
