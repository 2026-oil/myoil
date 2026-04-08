# NeuralForecast wrapper README

이 문서는 이 저장소의 **wrapper 실행 방식과 명시적 app config 설정**만 다룹니다.
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
uv run python main.py --validate-only --config yaml/experiment/feature_set/brentoil-case1.yaml
```

설명:

- `uv sync --group dev`로 현재 repo의 Python 환경과 개발 도구를 맞춥니다.
- 실행/테스트/검증은 기본적으로 `uv run ...` 기준으로 봅니다.

---

## 2. 실행 진입점

현재 실행 진입점은 루트의 `main.py`입니다.

- 엔트리포인트: `main.py`
- 실제 런타임: `runtime_support/runner.py`

역할:

- 프로젝트 가상환경 Python으로 bootstrap
- 이후 `runtime_support.runner.main()`으로 제어 전달

즉, 사용자는 아래처럼 실행하면 됩니다.

```bash
cd neuralforecast
uv run python main.py --validate-only --config yaml/experiment/feature_set/brentoil-case1.yaml
```

`main.py`는 더 이상 repo 루트 기본 config를 자동 탐색하지 않습니다.
항상 `--config` / `--config-path` / `--config-toml` 중 하나를 명시해야 합니다.

주요 인자:

- `--config <path>`
- `--config-path <path>`
- `--config-toml <path>`
- `--setting <path>`
- `--validate-only`
- `--jobs <job-name...>`

예시:

### 전체 config 검증

```bash
cd neuralforecast
uv run python main.py --validate-only --config yaml/experiment/feature_set/brentoil-case1.yaml
```

### 특정 shared setting으로 검증

```bash
cd neuralforecast
uv run python main.py --validate-only --config yaml/experiment/feature_set/brentoil-case1.yaml --setting yaml/setting/setting.yaml
```

### 특정 job만 실행

```bash
cd neuralforecast
uv run python main.py --config yaml/experiment/feature_set/brentoil-case1.yaml --jobs TFT
```

단일 job 재실행은 기존 scheduler run을 재사용할 수 있습니다.
같은 config source path와 같은 resolved config signature로 생성된 scheduler-backed run이 이미 있으면, runtime은 가장 최근 run의
`scheduler/workers/<job>` 경로를 자동 재사용하고 해당 job의 기존 산출물을 지운 뒤 fresh rerun을 수행합니다.
그 후 부모 run의 `summary/leaderboard.csv`, `summary/sample.md`, `summary/last_fold_*.png`도 다시 생성합니다.
매칭되는 scheduler run이 없으면 현재처럼 기본 single-job output root를 사용합니다.

### 멀티 job scheduler 실행

```bash
cd neuralforecast
uv run python main.py --config yaml/experiment/feature_set/brentoil-case1.yaml
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

미지정 시에는 에러가 나며, 명시적 config 경로가 필요합니다.

현재 대표 예시 파일:

- `yaml/experiment/feature_set/brentoil-case1.yaml`
- `yaml/experiment/feature_set/wti-case1.yaml`

top-level section:

- `dataset`
- `runtime`
- `training`
- `cv`
- `scheduler`
- `jobs`

### `bs_preforcast`

`bs_preforcast`는 main stage 전에 별도 stage1을 실행해 `bs_*` 계열 컬럼을 먼저 예측한 뒤, 그 결과를 main 입력에 주입하는 기능입니다.

권장 방식은 main YAML에서 독립 config 파일을 가리키는 구조입니다.

```yaml
bs_preforcast:
  enabled: true
  config_path: yaml/plugins/bs_preforcast.yaml
```

Python 패키지 구현은 `plugins/bs_preforcast/` 아래에 있으며, 코드 import는 `plugins.bs_preforcast...` 를 기준으로 맞춥니다.

상세 설정/동작/아티팩트 설명은 별도 문서를 보세요:

- `bs_preforcast.md`
- `yaml/jobs/bs_preforcast_jobs_default.yaml` (univariable direct-model 기본 파라미터)
- `yaml/jobs/bs_preforcast/bs_preforcast_jobs_uni.yaml` (univariable stage1 fanout catalog; 실제 단일 stage job은 `yaml/jobs/bs_preforcast/uni/*.yaml`)
- `yaml/jobs/bs_preforcast/bs_preforcast_jobs_multi.yaml` (multivariable stage1 fanout catalog; 실제 단일 stage job은 `yaml/jobs/bs_preforcast/multi/*.yaml`)

---

## 4. app config 설정표

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
| `input_size` | int | no | 모델 입력 길이 |
| `batch_size` | int | no | 학습 배치 크기 |
| `valid_batch_size` | int | no | validation 배치 크기 |
| `windows_batch_size` | int | no | training windows batch 크기 |
| `inference_windows_batch_size` | int | no | inference windows batch 크기 |
| `lr_scheduler` | mapping | no | OneCycleLR 공통 스케줄러 설정 |
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

### 4.6 `jobs`

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

## 5. 명시적 YAML 예시

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
  input_size: 64
  batch_size: 32
  valid_batch_size: 32
  windows_batch_size: 1024
  inference_windows_batch_size: 1024
  lr_scheduler:
    name: OneCycleLR
    max_lr: 0.001
    pct_start: 0.3
    div_factor: 25.0
    final_div_factor: 10000.0
    anneal_strategy: cos
    three_phase: false
    cycle_momentum: false
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

관련 산출물 예시:

- `runs/feature_set_brentoil_case1/scheduler/events.jsonl`
- `runs/feature_set_brentoil_case1/scheduler/workers/*/summary.json`
- `runs/feature_set_brentoil_case1/scheduler/workers/*/stdout.log`
- `runs/feature_set_brentoil_case1/scheduler/workers/*/stderr.log`

---

## 8. 요약

이 저장소에서 현재 기억해야 할 핵심은 아래입니다.

- 실행 진입점: `main.py`
- 설정 중심: 명시적 `--config` / `--config-path` / `--config-toml`
- 공통 loss: `training.loss = mse`
- CV 방식: **expanding-window TS-CV**
- multi-job 실행: scheduler가 GPU lane 분배
- 각 worker는 `devices=1`만 사용

upstream 라이브러리 자체가 필요하면 아래를 참고하면 됩니다.

- https://github.com/Nixtla/neuralforecast
