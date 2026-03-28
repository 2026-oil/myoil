# 플러그인 구조

이 repo의 stage plugin 시스템은 **plugin-neutral contract** 와 **concrete implementation** 으로 나뉩니다.

## 1. registry 역할

`plugin_contracts/stage_registry.py`는 등록된 stage plugin들을 관리합니다.

핵심 역할:
- plugin 등록
- key 기반 lookup
- active plugin 탐색
- lazy discovery (`plugins.bs_preforcast.plugin` import)

즉, main runtime은 concrete plugin을 직접 알지 않고 registry를 통해 접근합니다.

Source: `plugin_contracts/stage_registry.py:15-20`, `plugin_contracts/stage_registry.py:30-50`, `plugin_contracts/stage_registry.py:54-67`

## 2. `StagePlugin` contract

`plugin_contracts/stage_plugin.py`의 `StagePlugin` protocol은 stage plugin이 제공해야 할 surface를 정의합니다.

주요 lifecycle:
- identity: 어떤 top-level YAML key가 plugin을 켜는가
- config lifecycle: normalize / enabled / serialize
- route validation & stage load
- search-space integration
- runtime hooks
- manifest / validation payload
- fanout helpers

Source: `plugin_contracts/stage_plugin.py:19-213`

## 3. concrete example: `bs_preforcast`

`plugins/bs_preforcast/plugin.py`는 현재 repo의 실제 구현 예시입니다.

### identity / config lifecycle
- `config_key = "bs_preforcast"`
- `normalize_config()`로 main YAML의 plugin block 정규화
- `config_to_dict()`로 normalized payload serialize

Source: `plugins/bs_preforcast/plugin.py:29-63`

### route validation
plugin이 켜져 있으면 linked YAML(`config_path`)를 해석하고, 파일 존재 여부, top-level `bs_preforcast` 블록 존재 여부, linked YAML 구조를 확인합니다.

Source: `plugins/bs_preforcast/plugin.py:69-118`

### stage loading / normalized metadata
linked YAML과 search-space 계약까지 포함한 stage-loaded config를 만들고, 이후 normalized payload에 stage1 metadata를 붙입니다.

Source: `plugins/bs_preforcast/plugin.py:120-166`

### search-space integration
`bs_preforcast`는 자체 section key를 가집니다.

- `bs_preforcast_models`
- `bs_preforcast_training`

또한 stage model로 `AutoARIMA`를 허용하지 않습니다.

Source: `plugins/bs_preforcast/plugin.py:172-211`

### runtime hook delegation
실제 fold input 준비와 stage materialization은 plugin이 runtime 함수로 위임합니다.

Source: `plugins/bs_preforcast/plugin.py:217-255`

## 4. runtime에서 실제로 일어나는 일

`plugins/bs_preforcast/runtime.py` 기준으로 plugin stage는 다음을 수행합니다.

- fold forecast 계산
- `futr_exog` 또는 `lag_derived` injection 준비
- stage artifacts / manifest / capability report 기록
- main normalized payload와 manifest에 stage metadata 병합

Source: `plugins/bs_preforcast/runtime.py:773-875`, `plugins/bs_preforcast/runtime.py:878-936`

## 운영자에게 중요한 의미

운영자 관점에서 plugin은 **메인 YAML에서 켜고, linked YAML에서 상세를 관리하며, loader/runtime이 fail-fast로 검증하는 확장 포인트**입니다.

즉:
- 메인 YAML = plugin activation
- linked YAML = plugin-owned details
- runtime = plugin contract에 따라 stage 처리

## 자주 하는 실수

- main YAML에 plugin 세부 필드를 과하게 넣음
- `config_path`만 맞추면 끝이라고 생각함
- plugin-linked jobs와 main jobs의 역할 차이를 구분하지 않음

## 다음에 볼 페이지

- [메인 실험 YAML 안전하게 작성하기](Authoring-a-Main-Experiment-YAML-Safely)
- [bs_preforcast plugin YAML 레퍼런스](YAML-Reference-bs_preforcast-plugin-YAML)
