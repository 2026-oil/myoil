# main.py 실행 흐름

이 페이지는 **CLI 진입점에서 runtime handoff까지**만 설명합니다.

## 1. bootstrap과 `.venv` 재실행

`main.py`는 workspace root와 `.venv/bin/python` 경로를 확인하고, 필요하면 현재 프로세스를 `.venv` Python으로 재실행합니다.

운영자 관점에서 중요한 이유:
- `uv run ...`이 기본 실행 경로지만
- bootstrap 구조를 알면 환경 문제를 더 빨리 진단할 수 있습니다.

Source: `main.py:10-13`, `main.py:25-33`, `main.py:122-135`

## 2. `--output-root` 차단

공개 CLI에서는 `--output-root`를 직접 받지 않습니다. 일반 실행의 run root는 `task.name`과 내부 route 규칙으로 유도됩니다.

Source: `main.py:36-43`

## 3. parser surface

현재 `main.py` 주요 옵션:
- `--config`
- `--config-path`
- `--config-toml`
- `--setting`
- `--validate-only`
- `--jobs`
- 내부 hidden 옵션들 (`--output-root`, `--internal-jobs-route`, `--internal-stage`)

운영자가 주로 신경 써야 하는 것은 `--config`, `--setting`, `--validate-only`, `--jobs` 입니다.

Source: `main.py:46-64`

## 4. `load_app_config()` handoff

`_run_cli()`는 parser 결과를 받아 `load_app_config()`를 호출합니다.

이 단계에서 일어나는 핵심 일:
- 설정 파일 로드
- shared setting merge
- jobs fanout 여부 판정
- stage plugin probe/load 준비

Source: `main.py:67-87`, `app_config.py:1619-1814`

## 5. jobs fanout 처리

`jobs`가 여러 job-file route로 풀리면, `main.py`는 fanout spec을 순회하며 각 variant를 `run_loaded_config()`에 넘깁니다.

`--validate-only`일 때는 fanout 결과를 JSON으로 출력할 수 있습니다.

Source: `main.py:88-117`

## 6. 최종 runtime dispatch

fanout이 없으면 정규화된 `LoadedConfig`를 그대로 `run_loaded_config()`에 넘깁니다.

즉 `main.py`의 핵심 책임은:
- CLI 입력 수집
- bootstrap 보장
- config loading 시작
- fanout 여부에 따른 dispatch

입니다.

Source: `main.py:118-135`

## 운영자가 기억할 핵심

- 설정의 실제 의미 해석은 대부분 `app_config.py`에서 일어납니다.
- `main.py`는 **진입점 + dispatch orchestration** 입니다.
- 첫 검증은 항상 `--validate-only`가 안전합니다.

## 다음에 볼 페이지

- [설정 로딩과 YAML 소유권 맵](Config-Loading-and-YAML-Authority-Map)
- [메인 실험 YAML 안전하게 작성하기](Authoring-a-Main-Experiment-YAML-Safely)
