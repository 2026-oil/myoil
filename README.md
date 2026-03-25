# NeuralForecast wrapper README

이 문서는 이 저장소의 **wrapper 실행 방식과 `config.yaml` 설정**만 다룹니다.
upstream Nixtla 일반 소개가 아니라, 현재 이 checkout에서 실제로 쓰는 운영 기준 문서입니다.

---

## 1. 초기 셋업

### `uv` 설치

`uv`가 없다면 먼저 설치합니다.

- macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 새 셸을 열고 확인합니다.

```bash
uv --version
```

### 저장소 환경 준비

```bash
cd neuralforecast
uv sync --group dev
```

최소 실행 확인:

```bash
uv run python main.py --validate-only
```

설명:

- `uv sync --group dev`로 현재 repo의 Python 환경과 개발 도구를 맞춥니다.
- 실행/테스트/검증은 기본적으로 `uv run ...` 기준으로 봅니다.

---

## 2. 실행 진입점

현재 실행 진입점은 루트의 `main.py`입니다.

- 엔트리포인트: `main.py`
- 실제 런타임: `residual/runtime.py`

역할:

- 프로젝트 가상환경 Python으로 bootstrap
- 이후 `residual.runtime.main()`으로 제어 전달

즉, 사용자는 아래처럼 실행하면 됩니다.

```bash
cd neuralforecast
uv run python main.py --validate-only
```

주요 인자:

- `--config <path>`
- `--config-path <path>`
- `--config-toml <path>`
- `--validate-only`
- `--jobs <job-name...>`
- `--output-root <path>`

예시:

### 전체 config 검증

```bash
cd neuralforecast
uv run python main.py --validate-only --config config.yaml
```

### 특정 job만 실행

```bash
cd neuralforecast
uv run python main.py --config config.yaml --jobs TFT --output-root runs/single-job-smoke
```

`--output-root`를 생략한 단일 job 재실행은 기존 scheduler run을 재사용할 수 있습니다.
같은 config source path와 같은 resolved config signature로 생성된 scheduler-backed run이 이미 있으면, runtime은 가장 최근 run의
`scheduler/workers/<job>` 경로를 자동 재사용하고 해당 job의 기존 산출물을 지운 뒤 fresh rerun을 수행합니다.
그 후 부모 run의 `summary/leaderboard.csv`, `summary/sample.md`, `summary/last_fold_*.png`도 다시 생성합니다.
매칭되는 scheduler run이 없으면 현재처럼 기본 single-job output root를 사용합니다.

### 멀티 job scheduler 실행

```bash
cd neuralforecast
uv run python main.py --config examples/real_smoke.yaml --output-root runs/two-gpu-smoke
```

현재 동작 기준:

- 단일 job 실행은 runtime 경로를 직접 탑니다.
- 다중 job 실행은 scheduler launch plan + subprocess worker 경로를 탑니다.

---

## 3. 설정 파일 개요

지원 형식:

- YAML
- TOML

우선순위:

1. `--config`
2. `--config-path`
3. `--config-toml`
4. 미지정 시 repo 루트의 `config.yaml` / `config.yml` / `config.toml`

현재 대표 예시 파일:

- `config.yaml`
- `examples/real_smoke.yaml`

top-level section:

- `dataset`
- `runtime`
- `training`
- `cv`
- `scheduler`
- `residual`
- `jobs`

---

## 3.5 Excel 템플릿으로 YAML 만들기

직접 YAML을 손으로 수정하기 어렵다면 루트의 `xl_2_yaml.py`를 사용할 수 있습니다.

핵심 기능:

- 빈 Excel 템플릿 생성
- Excel workbook -> `yaml/<family>/...` YAML 자동 생성
- 생성 직후 내부 runtime validate-only helper 경로(`load_app_config` + job/adapter validation) 수행
- 기존 YAML -> Excel workbook 역변환

지원 family(현재 템플릿 dropdown 기준):

- `feature_set`
- `feature_set_HPT`
- `feature_set_HPT_c3`
- `feature_set_HPT_n100`
- `feature_set_residual`
- `bomb`
- `bomb_trans`
- `univar`
- `blackswan`
- `jaeho_feature_set`

### 템플릿 생성

```bash
cd neuralforecast
uv run python xl_2_yaml.py template /tmp/nf-template.xlsx
```

### Excel -> YAML 생성

```bash
cd neuralforecast
uv run python xl_2_yaml.py /tmp/nf-template.xlsx
```

또는 명시적으로:

```bash
cd neuralforecast
uv run python xl_2_yaml.py generate /tmp/nf-template.xlsx
```

특정 catalog row만 생성하고 싶으면:

```bash
cd neuralforecast
uv run python xl_2_yaml.py generate /tmp/nf-template.xlsx --catalog-id cfg1 --catalog-id cfg2
```

기본 동작:

- `Catalog.family`는 지원 family 목록에 있어야만 합니다.
- `Catalog.family` + `file_target`/`config_stem`으로 최종 경로 결정
- 출력은 `yaml/<family>/...` 아래로 자동 배치
- 충돌 경로는 fail-fast
- 임시 staging 후 내부 runtime validation이 모두 통과해야 최종 경로로 promote

### YAML -> Excel 역변환

```bash
cd neuralforecast
uv run python xl_2_yaml.py reverse yaml/feature_set/wti-case3.yaml --output /tmp/wti-case3.xlsx
```

여러 YAML도 한 workbook으로 역변환할 수 있습니다.

```bash
cd neuralforecast
uv run python xl_2_yaml.py reverse \
  yaml/feature_set/wti-case3.yaml \
  yaml/feature_set/brentoil-case3.yaml \
  --output /tmp/case3-batch.xlsx
```

### workbook schema 요약

필수 core sheet:

- `Catalog`
- `Task`
- `Dataset`
- `Runtime`
- `Training`
- `CV`
- `Scheduler`
- `Residual`
- `Jobs`
- `SearchSpace`

추가 adapter sheet:

- `Adapter.<family>` 형태

중요 규칙:

- `Catalog` 한 row가 최종 YAML 1개를 뜻합니다.
- 다른 sheet row는 모두 `catalog_id`로 `Catalog` row에 귀속됩니다.
- `Jobs`는 동일 `catalog_id` 아래 여러 row를 가질 수 있습니다.
- adapter override는 allowlist 밖 field를 수정할 수 없습니다.
- reverse conversion은 adapter provenance를 복원하지 않고, 확인 가능한 YAML 의미를 core sheet 기준으로 정규화합니다.
- blank cell은 기본적으로 "omit" 의미이고, 명시적으로 적은 default 값은 재생성 시 유지됩니다.

## 4. `config.yaml` 설정표

### 4.1 `dataset`

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `path` | string | yes | 입력 CSV 경로 |
| `target_col` | string | yes | 공통 타깃 컬럼 |
| `dt_col` | string | no | 시간 컬럼명, 기본값 `dt` |
| `freq` | string \| null | no | 시계열 주기. 생략 시 runtime이 자동 추론 |
| `hist_exog_cols` | list[string] | no | 과거 exogenous 컬럼 목록 |
| `futr_exog_cols` | list[string] | no | 미래 exogenous 컬럼 목록 |
| `static_exog_cols` | list[string] | no | static exogenous 컬럼 목록 |

### 4.2 `runtime`

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `random_seed` | int | no | 공통 랜덤 시드 |

### 4.3 `training`

> 현재 learned model은 **expanding-window TS-CV의 각 fold마다 별도로 학습**됩니다.

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `train_protocol` | string | no | 학습/eval 프로토콜 선언값. 현재 `expanding_window_tscv` |
| `input_size` | int | no | 모델 입력 길이 |
| `season_length` | int | no | seasonality 길이 |
| `batch_size` | int | no | 학습 배치 크기 |
| `valid_batch_size` | int | no | validation 배치 크기 |
| `windows_batch_size` | int | no | training windows batch 크기 |
| `inference_windows_batch_size` | int | no | inference windows batch 크기 |
| `learning_rate` | float | no | 공통 learning rate |
| `max_steps` | int | no | 최대 학습 step |
| `val_size` | int | no | 각 fit에서 내부 validation 길이 |
| `val_check_steps` | int | no | validation check 주기 |
| `early_stop_patience_steps` | int | no | early stopping patience |
| `loss` | string | no | 공통 loss. 현재 `mse`만 지원 |

### 4.4 `cv`

> 현재 CV는 **expanding-window time-series cross-validation**입니다.  
> 각 fold는 cutoff까지의 전체 이력을 학습에 사용하고, 그 다음 `horizon` 구간을 예측합니다.

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `horizon` | int | no | 예측 길이 |
| `step_size` | int | no | fold 내부 backcast cutoff stride |
| `n_windows` | int | no | CV fold 수 |
| `gap` | int | no | TSCV train/test 사이 gap. 기본 `0` |
| `max_train_size` | int/null | no | TSCV 최대 train 길이. 기본 `null` |
| `overlap_eval_policy` | string | no | 겹치는 예측 구간 집계 정책. 현재 `by_cutoff_mean` |

### 4.5 `scheduler`

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `gpu_ids` | list[int] | no | 사용할 GPU lane 목록 |
| `max_concurrent_jobs` | int | no | 동시 실행 가능한 최대 job 수 |
| `worker_devices` | int | no | worker당 device 수. 현재 `1`만 허용 |

### 4.6 `residual`

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | no | residual 보정 사용 여부 |
| `model` | string | no | residual 모델명. 현재 `xgboost` |
| `params` | object | no | residual 모델 파라미터 |

### 4.7 `jobs`

`jobs`는 **모델 단위 실행 목록**입니다.
공통 training control은 `training:`에 두고, `jobs`에는 learned model의
아키텍처별/가족별로 달라져야 하는 값만 둡니다.
baseline (`Naive`, `SeasonalNaive`, `HistoricAverage`)은 fairness normalization 대상이 아닙니다.

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `model` | string | yes | 실행할 모델 이름 |
| `params` | object | no | 해당 모델의 개별 override |

핵심 규칙:

- `dataset.target_col`은 config 전체에서 한 번만 정의합니다.
- exogenous 컬럼들도 `dataset`에서 한 번만 정의합니다.
- `jobs`에서는 모델 이름이 유일해야 합니다.
- `training:`에 있는 공통 key를 `jobs[*].params`에 다시 쓰면 안 됩니다.
- semantic commonality는 허용하지만, 모델 API가 다르면 key는 각자 유지합니다.
- API가 다른 key들 사이에는 aliasing이나 canonicalization을 하지 않습니다.
- 예: transformer 계열 `hidden_size`와 LSTM의 `encoder_hidden_size`/`decoder_hidden_size`는 같은 fairness 축으로 정렬할 수 있지만 하나의 key로 강제 통합하지 않습니다.
- conservative fairness 기준에서는 직접 대응되는 축만 맞추고, `PatchTST.n_heads`, `PatchTST.patch_len`, `FEDformer.modes`, NHITS 구조 knob들은 모델 로컬 예외로 둡니다.
- README에서는 `params` 내부의 모델별 세부 override는 별도로 풀어쓰지 않습니다.

---

## 5. 현재 `config.yaml` 예시

### 상단 구조 예시

```yaml
dataset:
  path: ../df.csv
  target_col: Com_CrudeOil
  dt_col: dt
  hist_exog_cols: []
  futr_exog_cols: []
  static_exog_cols: []

runtime:
  random_seed: 1

training:
  train_protocol: expanding_window_tscv
  input_size: 64
  season_length: 52
  batch_size: 32
  valid_batch_size: 32
  windows_batch_size: 1024
  inference_windows_batch_size: 1024
  learning_rate: 0.001
  max_steps: 1000
  val_size: 12
  val_check_steps: 100
  early_stop_patience_steps: -1
  loss: mse

cv:
  horizon: 12
  step_size: 4
  n_windows: 24
  gap: 0
  max_train_size:
  overlap_eval_policy: by_cutoff_mean

scheduler:
  gpu_ids: [0, 1]
  max_concurrent_jobs: 2
  worker_devices: 1

residual:
  enabled: true
  model: xgboost
  params: { ... }
```

### `jobs` 예시

모델별 상세 `params` 값은 여기서 전부 나열하지 않고, 구조만 유지합니다.
예시는 한 개만 둡니다.

```yaml
jobs:
  - model: TFT
    params: { ... }
```

---

## 6. residual 관리

residual 관련 코드는 `residual/` 아래에 모여 있습니다.

주요 파일:

- `residual/config.py`: YAML/TOML을 typed config로 정규화
- `residual/adapters.py`: `fit_df`, `futr_df`, `static_df`, `channel_map` 생성
- `residual/models.py`: 모델 capability 검증 및 공통 설정 적용
- `residual/manifest.py`: manifest / provenance 기록
- `residual/scheduler.py`: GPU lane 계획 및 subprocess worker 실행
- `residual/runtime.py`: 전체 실행 진입점

현재 residual 평가는 별도 holdout 없이 **config-driven TSCV fold**만 사용합니다.

현재 기준으로 README에서 확실히 말할 수 있는 것:

- residual은 wrapper의 1급 설정 영역입니다.
- `residual.enabled`, `residual.model`, `residual.params`로 관리합니다.
- manifest / provenance와 함께 실행 당시 설정을 추적합니다.

manifest에 기록되는 대표 항목:

- `manifest_version`
- `artifact_schema_version`
- `evaluation_protocol_version`
- `config_source_type`
- `config_source_path`
- `config_resolved_path`
- `config_input_sha256`
- `config_resolved_sha256`
- `entrypoint_version`
- `compat_mode`
- `training.loss`

### 새로운 residual 모델 추가 방법

새 residual 모델을 추가할 때는 **기존 base forecast / job / scheduler 구조는 건드리지 않고**
`residual` 레이어에만 새 플러그인을 붙이면 됩니다.

핵심 수정 지점:

1. `residual/plugins/<new_model>.py`
   - 새 `ResidualPlugin` 구현 추가
2. `residual/plugins/__init__.py`
   - 새 플러그인 export
3. `residual/registry.py`
   - `config.residual.model` 값에 따라 플러그인 선택
4. `residual/config.py`
   - `SUPPORTED_RESIDUAL_MODELS`에 새 이름 추가

설정 예시:

```yaml
residual:
  enabled: true
  model: mlp
  params:
    lookback: 8
    hidden_size: 32
    epochs: 50
    learning_rate: 0.001
```

스키마 예시:

```python
from dataclasses import dataclass
from typing import Any

import pandas as pd

from residual.plugins_base import ResidualContext, ResidualPlugin


@dataclass(frozen=True)
class _MLPConfig:
    lookback: int = 8
    hidden_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001


class MLPResidualPlugin(ResidualPlugin):
    name = "mlp"

    def __init__(
        self,
        *,
        lookback: int = 8,
        hidden_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 0.001,
    ):
        self.config = _MLPConfig(
            lookback=lookback,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
        ...

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        ...

    def metadata(self) -> dict[str, Any]:
        return {
            "plugin": self.name,
            "lookback": self.config.lookback,
            "hidden_size": self.config.hidden_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
        }
```

registry 연결 예시:

```python
from .plugins import MLPResidualPlugin, XGBoostResidualPlugin

def build_residual_plugin(config: Any) -> ResidualPlugin:
    ...
    if name == "xgboost":
        return XGBoostResidualPlugin(...)
    if name == "mlp":
        return MLPResidualPlugin(...)
    raise ValueError(f"Unsupported residual model: {name}")
```

정리:

- residual 모델 추가만 할 때는 `jobs:`는 그대로 둡니다.
- 새 residual 모델은 `residual.model`과 `residual.params`만 추가하면 됩니다.
- 자세한 단계별 문서는 `residual_guide.md`를 참고하세요.

---

## 7. 현재 확인된 실행 예시

### validate-only

```bash
cd neuralforecast
uv run python main.py --validate-only
```

### single job smoke

```bash
cd neuralforecast
uv run python main.py --config examples/real_smoke.yaml --jobs TFT --output-root runs/single-job-smoke
```

`examples/real_smoke.yaml` keeps a small `max_steps=10` smoke budget and a tiny
`xgboost` residual config; non-smoke configs should set their own training
budget under `training.max_steps`.

### two-gpu scheduler smoke

```bash
cd neuralforecast
uv run python main.py --config examples/real_smoke.yaml --output-root runs/two-gpu-smoke
```

관련 산출물 예시:

- `runs/two-gpu-smoke/scheduler/events.jsonl`
- `runs/two-gpu-smoke/scheduler/workers/*/summary.json`
- `runs/two-gpu-smoke/scheduler/workers/*/stdout.log`
- `runs/two-gpu-smoke/scheduler/workers/*/stderr.log`

---

## 8. 요약

이 저장소에서 현재 기억해야 할 핵심은 아래입니다.

- 실행 진입점: `main.py`
- 설정 중심: `config.yaml` / `config.toml`
- 공통 loss: `training.loss = mse`
- CV 방식: **expanding-window TS-CV**
- residual 관리 중심: `residual/`
- multi-job 실행: scheduler가 GPU lane 분배
- 각 worker는 `devices=1`만 사용

upstream 라이브러리 자체가 필요하면 아래를 참고하면 됩니다.

- https://github.com/Nixtla/neuralforecast
