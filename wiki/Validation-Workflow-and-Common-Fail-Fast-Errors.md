# 검증 절차와 주요 fail-fast 오류

이 페이지는 **실행 전에 설정 오류를 빠르게 잡는 방법**에 집중합니다.

## 기본 validate-only 명령

```bash
uv run python main.py --validate-only --config yaml/experiment/feature_set/brentoil-case1.yaml
uv run python main.py --validate-only --config yaml/experiment/feature_set/brentoil-case1.yaml --setting yaml/setting/setting.yaml
uv run python main.py --validate-only --config yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml
```

이 세 명령은 현재 README가 권장하는 기본 validate-only 루프입니다. 실제 성공 여부는 실행 시점 환경에서 다시 확인해야 합니다.

Source: `README.md:39-49`, `README.md:85-97`, `yaml/experiment/feature_set_bs_preforcast/brentoil-case3.yaml:1-31`

## validate-only가 의미하는 것

성공한 validate-only는 적어도 다음을 뜻합니다.

- composition-root YAML이 문법적으로 유효함
- main config가 정규화 가능함
- shared settings merge 가능함
- linked plugin YAML path가 해석됨
- stage plugin route validation 통과
- full training 없이 run payload 구성 가능

Source: `main.py:67-119`, `app_config.py:1639-1814`

## plugin activation 패턴

메인 실험 YAML에서 `bs_preforcast`를 켜는 기본 패턴은 아래와 같습니다.

```yaml
bs_preforcast:
  enabled: true
  config_path: yaml/plugins/bs_preforcast.yaml
```

이 블록은 **plugin activation만 담당**합니다. 실제 stage 세부 필드는 linked plugin YAML이 소유합니다.

Source: `README.md:156-171`, `plugins/bs_preforcast/plugin.py:69-118`

## 주요 fail-fast 오류

### 1) config path 누락 또는 제거된 `--output-root` 사용
- `main.py`는 config path가 필요합니다.
- 공개 CLI에서는 `--output-root`가 허용되지 않습니다.

Source: `main.py:36-43`, `main.py:77-81`

### 2) linked `bs_preforcast` YAML 경로가 해석되지 않음
`bs_preforcast`가 켜져 있으면 linked config path가 실제로 존재해야 합니다.

Source: `plugins/bs_preforcast/plugin.py:86-95`

### 3) routed YAML에 top-level `bs_preforcast` 블록이 없음
plugin은 routed 파일에 해당 블록이 없으면 바로 실패합니다.

Source: `plugins/bs_preforcast/plugin.py:99-109`

### 4) `hist_columns`와 `target_columns`가 겹침
config loading 단계에서 overlap을 거부합니다.

Source: `plugins/bs_preforcast/config.py:121-125`, `tests/test_bs_preforcast_config.py:234-266`

### 5) search space에 문자열 리스트를 넣음
search-space의 list choice는 문자열이 아니라 **native YAML list** 여야 합니다.

Source: `tests/test_bs_preforcast_config.py:526-570`, `tests/test_bs_preforcast_config.py:614-615`

### 6) `bs_preforcast` stage에서 `AutoARIMA` 사용
현재 이 stage에서는 `AutoARIMA`를 허용하지 않습니다.

Source: `plugins/bs_preforcast/plugin.py:202-206`, `tests/test_bs_preforcast_config.py:573-663`

### 7) 빈 params로 인해 search-space awareness 발생
jobs params가 비어 있으면 loader가 `yaml/HPO/search_space.yaml`을 요청할 수 있습니다.

Source: `app_config.py:1708-1735`

## 실전 troubleshooting 루프

1. 메인 composition-root YAML부터 수정
2. linked plugin YAML path 재확인
3. ownership 맵 재확인
4. validate-only 재실행
5. 그 다음에만 narrowed job 또는 full run 실행

## 왜 이 repo는 이렇게 엄격한가

현재 runtime은 조용히 추측하지 않고 **빨리 실패하는 쪽**을 택합니다.

- `main.py`는 CLI surface를 엄격하게 유지
- `app_config.py`는 실행 전에 정규화/검증 수행
- plugin 코드는 linked YAML ownership과 search-space 가정을 검증
- README도 validate-only를 첫 안전장치로 권장

Source: `main.py:36-43`, `app_config.py:1620-1814`, `README.md:39-49`, `README.md:156-184`

## 참고: 대체 작성 경로

repo는 Excel-to-YAML 생성 경로도 지원하지만, 이 위키 1차 흐름의 핵심은 **직접 YAML을 안전하게 작성/검증하는 운영 루프**입니다.

Source: `README.md:175-184`

## 관련 페이지

- [메인 실험 YAML 안전하게 작성하기](Authoring-a-Main-Experiment-YAML-Safely)
- [설정 로딩과 YAML 소유권 맵](Config-Loading-and-YAML-Authority-Map)
- [bs_preforcast plugin YAML 레퍼런스](YAML-Reference-bs_preforcast-plugin-YAML)
