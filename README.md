# NeuralForecast wrapper README

이 문서는 `/home/sonet/.openclaw/workspace/research/neuralforecast` 기준의 **커스텀 wrapper 실행 방식**만 설명한다.

이 README는 upstream Nixtla 소개 문서가 아니라, 현재 이 저장소에서 **실제로 동작하는 방식**에 맞춘 운영 문서다.

---

## 1. 초기 셋업

처음에는 이 저장소 루트에서 환경부터 맞춘 뒤 실행하는 것을 권장한다.

경로:
- `/home/sonet/.openclaw/workspace/research/neuralforecast`

권장 초기 셋업:

```bash
cd /home/sonet/.openclaw/workspace/research/neuralforecast
uv sync --group dev
```

최소 실행 확인:

```bash
cd /home/sonet/.openclaw/workspace/research/neuralforecast
uv run python main.py --validate-only
```

설명:
- `uv sync --group dev`로 현재 repo의 Python 환경과 개발 도구를 맞춘다.
- 이후 실행/테스트/검증은 `uv run ...` 기준으로 보는 것이 가장 안전하다.
- README에서는 내부 환경 디렉터리 자체를 직접 호출하는 방식보다, `uv` 기반 진입을 기준으로 설명한다.

---

## 2. `main.py`

현재 실행 진입점은 루트의 `main.py`다.

경로:
- `neuralforecast/main.py`

역할:
- 현재 repo 루트를 기준으로 bootstrap을 수행한다.
- `PYTHONPATH`에 현재 repo 루트를 넣는다.
- bootstrap이 끝나면 `residual.runtime.main()`으로 제어를 넘긴다.

즉, 실제 사용자는 아래처럼 실행하면 된다.

```bash
cd /home/sonet/.openclaw/workspace/research/neuralforecast
uv run python main.py --validate-only
```

현재 지원하는 주요 인자:
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
uv run python main.py --config config.yaml --jobs smoke_tft --output-root runs/single-job-smoke
```

### 멀티 job scheduler 실행
```bash
cd neuralforecast
uv run python main.py --config examples/real_smoke.yaml --output-root runs/two-gpu-smoke
```

현재 동작 기준으로:
- 단일 job 실행은 실제 runtime 경로를 탄다.
- 다중 job 실행은 scheduler launch plan + subprocess worker 실행 경로를 탄다.

---

## 3. config 작성법

현재 config는 **typed internal config model**로 읽힌다.

지원 형식:
- YAML
- TOML

주의:
- `dataset.freq`는 선택 사항이다.
- 생략하면 runtime이 `dt_col`에서 주기를 자동 추론한다.
- 자동 추론이 실패하는 비정규 시계열이면 그때만 `freq`를 명시하면 된다.

우선순위/입력 방식:
- `--config config.yaml`
- `--config-path config.yaml`
- `--config-toml config.toml`
- 아무 것도 주지 않으면 repo 루트에서 `config.yaml`, `config.yml`, `config.toml` 순으로 찾는다.

현재 기본 예시 파일:
- `neuralforecast/config.yaml`
- `neuralforecast/examples/real_smoke.yaml`

### top-level 구조
현재 지원하는 top-level section:
- `dataset`
- `runtime`
- `training`
- `cv`
- `scheduler`
- `residual`
- `jobs`

### 필수적으로 이해해야 하는 것

#### `dataset`
예시:
```yaml
dataset:
  path: ../df.csv
  dt_col: dt
```

의미:
- `path`: 입력 데이터 CSV 경로
- `target_col`: 모든 모델이 공통으로 다룰 단일 타깃 컬럼
- `dt_col`: 시간 컬럼 이름
- `freq`: 시계열 주기 (선택 사항, 생략하면 `dt_col`에서 자동 추론)

#### `training`
예시:
```yaml
training:
  input_size: 64
  season_length: 52
  batch_size: 32
  valid_batch_size: 32
  windows_batch_size: 1024
  inference_windows_batch_size: 1024
  learning_rate: 0.001
  max_steps: 100
  loss: mse
```

중요:
- `loss`는 **모든 모델 공통 설정**이다.
- 현재 v1에서 지원하는 공통 loss는 **`mse` 하나**다.
- 이 값은 resolved config와 manifest에도 기록된다.

#### `cv`
예시:
```yaml
cv:
  horizon: 12
  step_size: 4
  n_windows: 24
  final_holdout: 12
  overlap_eval_policy: by_cutoff_mean
```

의미:
- `horizon`: 예측 길이
- `step_size`: fold 간 이동 폭
- `n_windows`: CV fold 수
- `final_holdout`: 최종 홀드아웃 길이
- `overlap_eval_policy`: 현재 `by_cutoff_mean` 사용

#### `scheduler`
예시:
```yaml
scheduler:
  gpu_ids: [0, 1]
  max_concurrent_jobs: 2
  worker_devices: 1
```

의미:
- `gpu_ids`: 사용할 GPU lane
- `max_concurrent_jobs`: 동시 실행 job 수
- `worker_devices`: 현재 **항상 1이어야 함**

현재 설계상:
- worker는 `CUDA_VISIBLE_DEVICES`로 lane을 고정한다.
- 각 worker는 `devices=1`만 허용한다.

#### `jobs`
`jobs`는 `model` 단위로만 관리한다.

예시:
```yaml
jobs:
  - model: TFT
    params: {}

  - model: iTransformer
    params: {}
```

핵심 규칙:
- `dataset.target_col`은 config 전체에서 한 번만 설정한다.
- `hist_exog_cols`, `futr_exog_cols`, `static_exog_cols`도 `dataset` 단에서 한 번만 설정한다.
- `jobs`에서는 모델 이름만 고유하게 적는다.
- 같은 모델을 두 번 이상 넣으면 안 된다.
- `params`에는 해당 모델 하이퍼파라미터만 넣는다.
- `hist_exog_cols`, `futr_exog_cols`, `static_exog_cols`가 모두 비어 있으면 wrapper는 단변량 경로로 처리한다.
- 셋 중 하나라도 채워져 있으면 외부 변수 포함 경로로 처리한다.
- 다변량 모델(`iTransformer` 등)은 dataset-level exog를 채널로 해석한다.

### 실제 예시 파일
현재 real smoke용 예시는 아래다.

```yaml
dataset:
  path: ../../df.csv
  target_col: Com_CrudeOil
  dt_col: dt
  hist_exog_cols: []
  futr_exog_cols: []
  static_exog_cols: []
runtime:
  random_seed: 1
training:
  input_size: 8
  season_length: 52
  batch_size: 16
  valid_batch_size: 16
  windows_batch_size: 32
  inference_windows_batch_size: 32
  learning_rate: 0.001
  max_steps: 1
  val_size: 0
  loss: mse
cv:
  horizon: 2
  step_size: 2
  n_windows: 1
  final_holdout: 2
  overlap_eval_policy: by_cutoff_mean
scheduler:
  gpu_ids: [0, 1]
  max_concurrent_jobs: 2
  worker_devices: 1
residual:
  enabled: false
  train_source: oof_cv
jobs:
  - name: smoke_tft
    model: TFT
    job_type: univariate_with_exog
    target_col: Com_CrudeOil
    hist_exog_cols: []
    futr_exog_cols: []
    static_exog_cols: []
    params: {}
  - name: smoke_itransformer
    model: iTransformer
    job_type: multivariate_channels
    target_col: Com_CrudeOil
    channel_cols:
      - Com_BrentCrudeOil
    params: {}
```

---

## 4. residual 관리

현재 residual 관련 코드는 아래에 모여 있다.

경로:
- `neuralforecast/residual/`

파일 역할:
- `config.py`
  - YAML/TOML을 typed config로 정규화
- `adapters.py`
  - `fit_df`, `futr_df`, `static_df`, `channel_map` 생성
- `models.py`
  - 모델 capability 검증
  - 공통 loss(`mse`) 적용
- `manifest.py`
  - manifest / provenance 기록
- `scheduler.py`
  - GPU lane 계획 및 subprocess worker 실행
- `runtime.py`
  - 전체 실행 진입점

### residual section 자체
현재 config에는 아래처럼 들어간다.

```yaml
residual:
  enabled: true
  train_source: oof_cv
```

현재 실제로 동작하는 수준에서 말하면:
- `residual` section은 **typed config에 포함**된다.
- `train_source`는 현재
  - `insample_backcast`
  - `oof_cv`
  중 하나로 정규화된다.
- 이 값은 runtime config/manifest 관점에서 관리된다.

### 지금 문서화 가능한 현재 상태
현재 기준으로 README에서 확실히 말할 수 있는 것은:
- residual 관련 코드의 **관리 단위는 `neuralforecast/residual/`** 이다.
- config에서 `residual.enabled`, `residual.train_source`를 관리한다.
- manifest/provenance 체계와 함께 wrapper 설정 일부로 유지된다.

즉, residual은 지금 이 repo에서 **wrapper의 1급 설정 영역**으로 관리되고 있다.

### manifest / provenance
현재 run manifest에는 다음이 기록된다.
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

즉 residual/평가 방식까지 포함해, 실행 당시 설정을 추적하는 구조다.

---

## 5. 현재 확인된 실행 예시

### validate-only
```bash
cd /home/sonet/.openclaw/workspace/research/neuralforecast
uv run python main.py --validate-only
```

확인 결과 예:
```json
{"ok": true, "jobs": ["crudeoil_tft", "crudeoil_itransformer"]}
```

### single job smoke
```bash
cd /home/sonet/.openclaw/workspace/research/neuralforecast
uv run python main.py --config examples/real_smoke.yaml --jobs smoke_tft --output-root runs/single-job-smoke
```

### two-gpu scheduler smoke
```bash
cd /home/sonet/.openclaw/workspace/research/neuralforecast
uv run python main.py --config examples/real_smoke.yaml --output-root runs/two-gpu-smoke
```

확인 결과 예:
- `smoke_tft` → `gpu_id: 0`, `devices: 1`
- `smoke_itransformer` → `gpu_id: 1`, `devices: 1`

관련 산출물:
- `runs/two-gpu-smoke/scheduler/events.jsonl`
- `runs/two-gpu-smoke/scheduler/workers/*/summary.json`
- `runs/two-gpu-smoke/scheduler/workers/*/{stdout,stderr}.log`

---

## 6. 정리

이 저장소에서 지금 기준으로 기억하면 된다.

- 실행 진입점: `neuralforecast/main.py`
- 설정 중심: `config.yaml` / `config.toml`
- 공통 loss: `training.loss = mse`
- residual 관리 중심: `neuralforecast/residual/`
- multi-job 실행: scheduler가 GPU lane 분배
- 각 worker는 `devices=1`만 사용

upstream Nixtla 라이브러리 자체 설명은 필요하면 아래를 참고하면 된다.
- https://github.com/Nixtla/neuralforecast
