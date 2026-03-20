# `main.py --config baseline-wti.yaml` 실행 흐름 상세 설명

## 기준 명령

```bash
uv run python main.py --config baseline-wti.yaml
```

이 문서는 위 명령을 실행했을 때 현재 저장소 코드 기준으로 **어떤 순서로 어떤 로직이 수행되는지**를 상세하게 설명한다.

---

## 1. 한 줄 요약

이 실행은 단순히 `baseline-wti.yaml`을 읽어서 모델 하나를 학습하는 흐름이 아니다.

실제로는 아래처럼 동작한다.

1. `main.py`가 먼저 `.venv` 파이썬으로 재실행(re-exec)한다.
2. `residual.runtime.main()`으로 진입한다.
3. `baseline-wti.yaml`을 읽고 `search_space.yaml`도 함께 읽는다.
4. `task.name=baseline_wti`를 기준으로 기본 출력 경로를 `runs/baseline_wti`로 잡는다.
5. 전체 job 23개를 정규화한다.
6. `Naive` 1개는 baseline 고정 모드, 나머지 22개는 `params: {}` 이므로 Optuna 자동튜닝 모드가 된다.
7. residual은 `enabled: false` 이므로 residual 보정 단계는 아예 수행하지 않는다.
8. job 수가 1개보다 많기 때문에 루트 프로세스는 스케줄러 역할만 하고, 각 모델별 worker 프로세스를 다시 `main.py`로 띄운다.
9. 각 worker는 자기 모델 하나에 대해 시계열 cross-validation을 수행한다.
10. learned model은 Optuna 튜닝 후 fold 학습/예측을 수행하고, `Naive`는 별도 baseline 경로로 평가한다.

---

## 2. 실제 입력 config 요약

`baseline-wti.yaml`의 핵심 값은 다음과 같다.

### task
- `task.name: baseline_wti`

### dataset
- `path: data/df.csv`
- `target_col: Com_CrudeOil`
- `dt_col: dt`
- `hist_exog_cols`: 88개
- `futr_exog_cols`: 0개
- `static_exog_cols`: 0개

### training
- `input_size: 64`
- `season_length: 52`
- `batch_size: 32`
- `valid_batch_size: 32`
- `windows_batch_size: 1024`
- `inference_windows_batch_size: 1024`
- `learning_rate: 0.001`
- `max_steps: 1000`
- `val_size: 12`
- `val_check_steps: 100`
- `early_stop_patience_steps: -1`
- `loss: mse`

### cv
- `horizon: 8`
- `step_size: 4`
- `n_windows: 24`
- `gap: 0`
- `max_train_size: null`
- `overlap_eval_policy: by_cutoff_mean`

### scheduler
- `gpu_ids: [0, 1]`
- `max_concurrent_jobs: 2`
- `worker_devices: 1`

### residual
- `enabled: false`
- `model: xgboost`
- `params: {}`

### jobs
총 23개:
- learned model 22개
- baseline model 1개 (`Naive`)

---

## 3. 현재 데이터셋 상태

`data/df.csv`를 현재 기준으로 확인하면:

- 행 수: 584
- 열 수: 91
- `dt` 최소값: `2015-01-05`
- `dt` 최대값: `2026-03-09`
- `Com_CrudeOil` non-null 개수: 584
- 날짜 빈도 추론 결과: `W-MON`

즉 이 config는 **월요일 기준 주간 시계열** 데이터에 대해 WTI(`Com_CrudeOil`)를 예측하는 설정이다.

---

## 4. 엔트리포인트: `main.py`

실행은 `main.py`에서 시작한다.

### 4-1. `.venv` 파이썬으로 재실행
`main.py`는 현재 실행 중인 인터프리터가 저장소의 `.venv/bin/python`이 아니면 그쪽으로 다시 `execvpe` 한다.

의도는 다음과 같다.
- 저장소 전용 가상환경 사용 강제
- `optuna`, `pandas`, `torch`, `neuralforecast` 등 의존성 누락 방지
- 런타임 환경을 일정하게 맞춤

### 4-2. `PYTHONPATH` 설정
`WORKSPACE_ROOT`를 `PYTHONPATH`에 넣는다.

의도는 다음과 같다.
- 로컬 패키지 import 보장
- `residual.*`, `neuralforecast.*` import 안정화

### 4-3. 실제 진입점 전환
그 뒤 실제 로직은 다음으로 넘어간다.

```python
from residual.runtime import main as residual_main
return residual_main(args)
```

즉 `main.py`는 사실상 **bootstrap wrapper** 역할이다.

---

## 5. CLI 파싱: `residual.runtime.main()`

런타임은 아래 인자를 받는다.

- `--config`
- `--config-path`
- `--config-toml`
- `--validate-only`
- `--jobs`
- `--output-root`

현재 실행은 보통 다음 형태다.

```bash
uv run python main.py --config baseline-wti.yaml
```

따라서:
- `config_path = baseline-wti.yaml`
- `validate_only = False`
- `jobs = None`
- `output_root = None`

즉 **config에 있는 모든 job을 실행 대상으로 선택**한다.

---

## 6. config 로드와 정규화

`load_app_config()`가 `baseline-wti.yaml`을 읽고 내부 dataclass 구조로 정규화한다.

### 6-1. 경로 해석
- `baseline-wti.yaml`은 YAML 파일로 인식된다.
- `dataset.path: data/df.csv`는 절대경로로 변환된다.

실제 해석 결과:
- `/home/sonet/.openclaw/workspace/research/neuralforecast/data/df.csv`

### 6-2. search space 자동 로드
`jobs[*].params`를 보면 learned model들이 전부 `{}` 이다.

이 코드베이스에서 learned model이 빈 params를 가지면 의미는:
- 사용자가 하이퍼파라미터를 직접 고정하지 않았다.
- 따라서 Optuna 자동탐색이 필요하다.

그래서 `load_app_config()`는 자동으로 `search_space.yaml`도 함께 읽는다.

### 6-3. task 기반 출력 경로 결정
`task.name=baseline_wti` 이므로, `--output-root`를 명시하지 않으면 기본 출력 경로는:

```text
runs/baseline_wti
```

가 된다.

이 규칙은 다음처럼 동작한다.
- `task.name` 문자열을 sanitize
- 영숫자/`-`/`_`/`.` 외 문자는 `-`로 치환
- 비어 있으면 `runs/validation`
- 현재는 `baseline_wti` 그대로 사용 가능

---

## 7. 각 job의 mode 판정

이 config에서 실제 판정 결과는 아래와 같다.

### baseline model
- `Naive`
  - `requested_mode = baseline_fixed`
  - `validated_mode = baseline_fixed`

### learned model 22개
전부 아래 형태다.
- `requested_mode = learned_auto_requested`
- `validated_mode = learned_auto`

즉 다음 모델들은 모두 **Optuna 자동튜닝 후 본 평가** 경로를 탄다.

- `RNN`
- `GRU`
- `LSTM`
- `NHITS`
- `DLinear`
- `NLinear`
- `TFT`
- `VanillaTransformer`
- `Informer`
- `Autoformer`
- `FEDformer`
- `PatchTST`
- `iTransformer`
- `TimeLLM`
- `TimeMixer`
- `TimesNet`
- `TimeXer`
- `NonstationaryTransformer`
- `xLSTMMixer`
- `Mamba`
- `SMamba`
- `CMamba`

### residual mode
현재 config는:
- `residual.enabled = false`

그래서:
- `requested_mode = residual_disabled`
- `validated_mode = residual_disabled`

즉 residual plugin (`xgboost`)은 아예 실행되지 않는다.

---

## 8. 루트 실행 초기에 생성되는 파일

실제 본 학습 전에 루트 run 디렉터리 아래에 기본 아티팩트를 먼저 만든다.

### 생성 파일
- `runs/baseline_wti/config/config.resolved.json`
- `runs/baseline_wti/config/capability_report.json`
- `runs/baseline_wti/manifest/run_manifest.json`

### 의미
#### `config.resolved.json`
정규화된 최종 config 스냅샷이다.
- 상대경로 → 절대경로 변환 반영
- requested/validated mode 반영
- search space 경로/hash 반영

#### `capability_report.json`
각 모델이 어떤 입력 모드/능력을 갖는지 기록한다.
예:
- multivariate 여부
- exogenous 지원 여부
- auto 지원 여부

#### `run_manifest.json`
이번 실행의 provenance 문서다.
- config source path
- config hash
- search_space hash
- job별 validated mode
- residual mode
- entrypoint version

---

## 9. 학습 전에 수행하는 adapter 사전 검증

런타임은 바로 학습하지 않고 먼저 `_validate_adapters()`를 수행한다.

여기서 하는 일:
- `data/df.csv`를 실제로 읽는다.
- 각 모델별로 필요한 입력 포맷을 만들 수 있는지 확인한다.
- 필요한 컬럼이 빠졌는지 검사한다.

### 9-1. univariate 모델 입력 형식
univariate 모델은 대략 다음 wide 형태를 쓴다.
- `unique_id = Com_CrudeOil`
- `ds = dt`
- `y = Com_CrudeOil`
- hist exog 88개를 같은 row에 컬럼으로 붙임

즉 target 하나를 예측하되, 과거 외생변수 88개를 입력 피처로 사용한다.

### 9-2. multivariate 모델 입력 형식
multivariate 모델은 target과 exogenous를 모두 여러 시계열 채널로 간주해서 long/panel 형태로 바꾼다.

현재 config에서 multivariate로 판정된 모델:
- `iTransformer`
- `TimeMixer`
- `TimeXer`
- `xLSTMMixer`
- `SMamba`
- `CMamba`

이때 사용되는 series 수는:
- target 1개 + hist exog 88개 = 총 89 series

즉 이 모델들은 내부적으로 **89변수 multivariate panel**을 입력으로 받는다.

---

## 10. single-job가 아니라 scheduler 모드로 감

`--jobs`를 따로 주지 않았기 때문에 선택된 job 수는 23개다.

코드는 다음처럼 분기한다.
- job 수가 1개면 `_run_single_job()` 직접 수행
- job 수가 2개 이상이면 스케줄러 모드 진입

현재는 23개이므로 **루트 프로세스는 직접 학습하지 않고 worker orchestration만 수행**한다.

---

## 11. 스케줄러의 worker 배치 방식

스케줄러는 `build_launch_plan()`으로 GPU를 round-robin 배정한다.

현재 설정:
- `gpu_ids = [0, 1]`
- `max_concurrent_jobs = 2`

### 실제 launch plan 예시
- `RNN -> GPU 0`
- `GRU -> GPU 1`
- `LSTM -> GPU 0`
- `NHITS -> GPU 1`
- ...
- `Naive -> GPU 0`

중요한 점은 두 가지다.

### 11-1. 동시 실행 수 제한
코드상 `max_concurrent_jobs=2` 이므로 **동시에 최대 2개 worker만 활성화**된다.

즉 launch plan은 23개지만 실제 실행은:
- 2개 시작
- 하나 끝나면 다음 것 시작
- 이런 식으로 순차적으로 진행

### 11-2. worker도 다시 `main.py`를 호출함
worker는 내부적으로 아래와 비슷한 명령으로 재귀적으로 실행된다.

```bash
python main.py \
  --jobs RNN \
  --output-root runs/baseline_wti/scheduler/workers/RNN \
  --config baseline-wti.yaml
```

즉 루트 프로세스와 worker 프로세스 모두 동일 엔트리포인트를 쓰되,
worker는 `--jobs <단일모델>`을 받아 single-job 경로로 들어간다.

### 11-3. worker 환경 변수
worker 실행 시 환경변수로 다음이 설정된다.
- `CUDA_VISIBLE_DEVICES=<gpu_id>`
- `NEURALFORECAST_WORKER_DEVICES=1`

즉 각 worker는 자기에게 할당된 GPU 하나만 보게 된다.

---

## 12. worker 하나 내부에서 실제로 일어나는 일

예를 들어 `TimeLLM` worker 하나를 기준으로 보면:

1. 다시 `main.py` 진입
2. `.venv` python 보정
3. 다시 config 로드
4. `--jobs TimeLLM`으로 job 하나만 선택
5. `output_root = runs/baseline_wti/scheduler/workers/TimeLLM`
6. worker 전용 resolved config / capability / manifest 생성
7. `_run_single_job()` 수행

즉 worker 하나는 **자기 모델 하나에 대한 독립 run 디렉터리**를 갖는다.

---

## 13. `_run_single_job()`에서 공통으로 하는 일

각 worker는 자기 모델에 대해 아래를 수행한다.

### 13-1. 원본 CSV 재로딩
- `data/df.csv`를 다시 읽는다.
- `dt_col` 기준 정렬한다.

### 13-2. frequency 추론
`dataset.freq`가 config에 명시되지 않았으므로 `pd.infer_freq()`를 쓴다.
현재 데이터는 `W-MON`으로 추론된다.

### 13-3. TSCV split 생성
현재 설정은:
- `horizon=8`
- `n_windows=24`
- `gap=0`
- `max_train_size=null`

실제 split 결과는 다음과 같다.

- fold 0: train 392 / test 8
- fold 1: train 400 / test 8
- ...
- fold 23: train 576 / test 8

즉 **확장형(expanding) 학습 구간 + 고정 8-step 평가 구간**이 24번 반복된다.

---

## 14. 중요한 디테일: `cv.step_size=4`는 direct CV split에는 거의 안 쓰임

config만 보면 `step_size=4`가 있으니 fold가 4칸씩 이동한다고 생각하기 쉽다.
하지만 현재 코드 경로는 그렇지 않다.

### 실제 동작
`n_windows > 1`이면 `TimeSeriesSplit`를 사용하고, 여기서는:
- `n_splits`
- `test_size`
- `gap`
- `max_train_size`

만 직접 사용한다.

즉 direct CV split 생성 자체에는 `step_size`가 관여하지 않는다.

### `step_size`가 쓰이는 곳
현재 코드에서 `step_size`는 주로:
- residual backcast panel 생성 로직
- single-split 특수 경로 계산

쪽에 의미가 있다.

그런데 이번 config는 `residual.enabled=false`이므로, 체감상 `step_size=4`는 거의 실효가 없다.

---

## 15. baseline 모델(`Naive`) 경로

`Naive`는 learned model과 완전히 다르게 돈다.

### 특징
- `build_model()`을 호출하지 않는다.
- `NeuralForecast.fit()`을 하지 않는다.
- baseline 전용 `_baseline_cross_validation()`를 사용한다.

### 예측 방식
`Naive`는 각 fold에서:
- train 구간 마지막 관측값을
- 미래 horizon 전체에 그대로 반복 예측한다.

### 생성 결과 예
현재 실제로 확인된 worker 산출물:
- `runs/baseline_wti/scheduler/workers/Naive/config/config.resolved.json`
- `runs/baseline_wti/scheduler/workers/Naive/config/capability_report.json`
- `runs/baseline_wti/scheduler/workers/Naive/manifest/run_manifest.json`
- `runs/baseline_wti/scheduler/workers/Naive/cv/Naive_forecasts.csv`
- `runs/baseline_wti/scheduler/workers/Naive/cv/Naive_metrics_by_cutoff.csv`
- `runs/baseline_wti/scheduler/workers/Naive/models/Naive/fit_summary.json`
- `runs/baseline_wti/scheduler/workers/Naive/stdout.log`
- `runs/baseline_wti/scheduler/workers/Naive/stderr.log`

`stdout.log`에는 실제로 아래처럼 남아 있다.

```json
{"ok": true, "executed_jobs": ["Naive"]}
```

즉 baseline 경로는 매우 가볍고, 산출물도 이미 정상 생성된 상태다.

---

## 16. learned model 경로

learned model은 대체로 다음 순서로 실행된다.

### 16-1. Optuna 자동튜닝
`validated_mode = learned_auto` 이므로 먼저 `_tune_main_job()`으로 들어간다.

여기서:
- sampler = TPE
- seed = `runtime.random_seed` 또는 환경변수 override
- trial 수 = 기본 5 (`NEURALFORECAST_OPTUNA_NUM_TRIALS` 없으면 5)

각 trial마다:
1. `search_space.yaml`에서 해당 모델의 탐색 파라미터 후보를 읽고
2. 파라미터를 샘플링한 뒤
3. 모든 fold(24개)에 대해 fit/predict를 수행하고
4. 각 fold의 MSE 평균을 objective로 사용한다.

즉 learned_auto 모델 하나당 기본적으로:
- Optuna 5 trials × 24 folds = 120회 fold fit
- best params 확정 후 본 평가 24 folds 추가
- 총 약 144회 fit

이것이 모델 22개에 대해 반복되므로 실행량이 매우 크다.

### 16-2. best params 저장
튜닝이 끝나면 worker run dir 아래에 보통 다음이 생긴다.
- `models/<model>/best_params.json`
- `models/<model>/optuna_study_summary.json`

그리고 manifest에도 해당 파일 경로가 반영된다.

### 16-3. best params로 본 cross-validation 재실행
튜닝이 끝나면 best params를 반영한 `effective_job`으로 실제 fold 예측을 다시 돈다.

각 fold에서:
1. train/test slice 분리
2. adapter input 생성
3. `build_model()` 호출
4. `NeuralForecast.fit()`
5. `nf.predict()`
6. target series(`Com_CrudeOil`) 기준 예측/실측 비교
7. MAE/MSE/RMSE 계산
8. forecast CSV / metrics CSV row 누적

---

## 17. 모델 생성 시 공통으로 들어가는 실제 인자

learned model 생성 시 공통으로 넣는 핵심 파라미터는 다음과 같다.

- `h = 8`
- `input_size = 64`
- `max_steps = 1000`
- `learning_rate = 0.001`
- `val_check_steps = 100`
- `early_stop_patience_steps = -1`
- `batch_size = 32`
- `valid_batch_size = 32`
- `windows_batch_size = 1024`
- `inference_windows_batch_size = 1024`
- `random_seed = 1`
- `loss = MSE()`
- `valid_loss = MSE()`
- `alias = <job.model>`
- `accelerator = 'gpu'` 또는 `'cpu'`
- `devices = 1`
- `enable_checkpointing = False`

그리고 모델 capability에 따라 아래도 추가된다.
- `hist_exog_list`
- `futr_exog_list`
- `stat_exog_list`
- `n_series`

### 예: multivariate 모델
multivariate 모델은 `n_series=89` 같은 값이 들어갈 수 있다.

### 예: univariate + hist exog 지원 모델
`RNN`, `GRU`, `LSTM`, `NHITS`, `TFT` 등은 hist exog 88개가 실제 입력 리스트로 들어간다.

---

## 18. residual 보정이 이번 실행에서 왜 완전히 생략되는가

코드상 learned model 본 평가가 끝나면 원래는 `_apply_residual_plugin()`으로 들어갈 수 있다.

하지만 이번 config는:

```yaml
residual:
  enabled: false
```

이므로 다음 로직에서 즉시 return 된다.

- baseline model이면 return
- residual disabled이면 return

즉 이번 실행에서는 아래가 생성되지 않는다.
- `residual/<model>/folds/...`
- `backcast_panel.csv`
- `corrected_eval.csv`
- residual `best_params.json`
- residual `plugin_metadata.json`
- residual `diagnostics.json`

즉 이번 run은 이름에 residual runtime이 들어가긴 하지만, **실제론 pure baseline/learned CV runtime**에 가깝다.

---

## 19. worker별 산출물 구조

worker 하나가 성공적으로 끝나면 보통 아래 구조가 생긴다.

```text
runs/baseline_wti/scheduler/workers/<MODEL>/
  config/
    config.resolved.json
    capability_report.json
  manifest/
    run_manifest.json
  cv/
    <MODEL>_forecasts.csv
    <MODEL>_metrics_by_cutoff.csv
  models/
    <MODEL>/
      fit_summary.json
      best_params.json                  # learned_auto일 때
      optuna_study_summary.json         # learned_auto일 때
  stdout.log
  stderr.log
  summary.json
```

### `summary.json`
스케줄러가 worker 종료 후 기록한다.
내용은 대략:
- `job_name`
- `gpu_id`
- `devices`
- `cuda_visible_devices`
- `returncode`
- `stdout_path`
- `stderr_path`
- `completed_at`

### scheduler 레벨 파일
루트 scheduler 디렉터리에는:
- `scheduler/launch_plan.json`
- `scheduler/events.jsonl`

가 생긴다.

`events.jsonl`에는 `worker_started`, `worker_completed` 이벤트가 쌓인다.

---

## 20. 현재 확인 가능한 실제 실행 흔적

현재 `runs/baseline_wti/` 아래에는 실제 실행 흔적이 남아 있다.

### 확인된 사실
- launch plan은 23개 job을 포함한다.
- `events.jsonl`에는 worker 시작 이벤트들이 기록돼 있다.
- `Naive` worker는 실제 산출물이 존재하고 stdout도 성공이다.
- 일부 learned model worker는 아직 산출물이 없거나 실패 흔적이 있다.

### 예: `TimeXer` 실패 흔적
현재 `runs/baseline_wti/scheduler/workers/TimeXer/stderr.log`에는 다음 계열의 에러가 남아 있다.

- `RuntimeError: Efficient attention cannot produce valid seed and offset outputs when the batch size exceeds (65535).`

즉 **코드 흐름 자체는 정상 진입했지만, 모델 내부 torch attention 경로에서 런타임 실패**가 발생할 수 있다.

이 말은 곧:
- 실행 흐름 설명과
- 실제 모든 모델이 성공하는지는
서로 다른 문제라는 뜻이다.

---

## 21. 이 config에서 실제로 무거운 부분

이 실행이 오래 걸리는 핵심 이유는 다음이다.

### 21-1. 모델 수가 많음
- 총 23 jobs

### 21-2. 대부분 learned_auto
- 22개가 Optuna 자동튜닝

### 21-3. fold 수가 많음
- 각 모델당 24 folds

### 21-4. trial 수가 있음
- 기본 5 trials

즉 대략 learned_auto 모델 하나당 144회 수준의 fit이 발생할 수 있다.
22개면 총 계산량은 매우 커진다.

### 21-5. 일부 모델은 multivariate 89-series 입력
특히 아래 모델들은 입력 차원이 더 크다.
- `iTransformer`
- `TimeMixer`
- `TimeXer`
- `xLSTMMixer`
- `SMamba`
- `CMamba`

이 모델들은 89개 series panel을 처리하므로 메모리/연산 부담이 상대적으로 크다.

---

## 22. config에 있지만 현재 경로에서 체감 영향이 낮은 값

아래 값들은 config에 보이지만, 현재 코드 경로 기준으로 사용자가 기대하는 방식과 다를 수 있다.

### `training.train_protocol: expanding_window_tscv`
실제 분기 로직에서 이 문자열을 보고 실행 경로를 바꾸지는 않는다.
지금 구현은 사실상 CV 로직이 고정되어 있다.

### `training.season_length: 52`
현재 `build_model()` 공통 인자에서는 직접 주입되지 않는다.
개별 모델이 별도 파라미터로 필요로 하면 Optuna/best params나 job params로 들어가야 의미가 생긴다.

### `cv.step_size: 4`
앞서 설명했듯이 direct TSCV split 생성에는 직접 관여하지 않는다.

즉 config를 읽을 때 이 값들이 “중요해 보이지만”, 현재 런타임 구현에서는 생각보다 실질 영향이 제한적이다.

---

## 23. 실행 완료 시 루트 프로세스가 하는 일

루트 프로세스는 모든 worker 결과를 모아서 마지막에 다음처럼 판단한다.

### 전부 성공
모든 worker의 `returncode == 0`이면:

```json
{
  "ok": true,
  "scheduled_jobs": [...],
  "worker_results": [...]
}
```

를 출력한다.

### 하나라도 실패
하나라도 `returncode != 0`이면:

```json
{
  "ok": false,
  "worker_results": [...]
}
```

형태로 종료한다.

즉 최종 성공 조건은 **모든 worker 성공**이다.

---

## 24. 최종 흐름을 순서도로 다시 정리하면

```text
uv run python main.py --config baseline-wti.yaml
  ↓
main.py
  ↓
필요 시 .venv/bin/python 으로 re-exec
  ↓
residual.runtime.main()
  ↓
baseline-wti.yaml 로드
  ↓
(search_space.yaml 자동 로드)
  ↓
config 정규화 + mode 판정
  ↓
output_root = runs/baseline_wti
  ↓
resolved config / capability / manifest 생성
  ↓
adapter 사전검증
  ↓
job 23개 선택
  ↓
스케줄러 모드 진입
  ↓
launch_plan 생성 (GPU 0/1 round-robin)
  ↓
max_concurrent_jobs=2 로 worker 순차 실행
  ↓
worker별 main.py --jobs <MODEL> 재호출
  ↓
각 worker가 자기 모델 하나에 대해
  - CSV 로드
  - freq 추론
  - 24-fold TSCV 생성
  - baseline 또는 learned_auto 경로 수행
  - forecast/metrics/manifest/summary 산출물 저장
  ↓
모든 worker returncode 수집
  ↓
전부 성공이면 ok=true, 하나라도 실패면 ok=false
```

---

## 25. 이 config를 읽을 때 실전적으로 기억할 핵심 포인트

1. **출력 디렉터리는 기본적으로 `runs/baseline_wti`다.**
2. **실행은 모델 하나가 아니라 23개 job 전체를 돈다.**
3. **22개 learned model은 전부 Optuna 자동튜닝이다.**
4. **residual은 꺼져 있으므로 residual 관련 보정/산출물은 없다.**
5. **CV는 24 folds × horizon 8이다.**
6. **동시에 최대 2개 worker만 돈다.**
7. **multivariate 모델은 89 series를 입력으로 쓴다.**
8. **`step_size=4`, `season_length=52`, `train_protocol`은 현재 구현에서 체감 영향이 제한적이다.**
9. **최종 성공 조건은 모든 worker가 성공하는 것이다.**
10. **실제 실행에서는 모델별 런타임 실패가 발생할 수 있다.**

---

## 26. 참고한 코드/아티팩트

### 코드
- `main.py`
- `residual/config.py`
- `residual/runtime.py`
- `residual/scheduler.py`
- `residual/models.py`
- `residual/adapters.py`
- `residual/manifest.py`
- `residual/optuna_spaces.py`

### 현재 확인한 런타임 아티팩트
- `runs/baseline_wti/manifest/run_manifest.json`
- `runs/baseline_wti/scheduler/launch_plan.json`
- `runs/baseline_wti/scheduler/events.jsonl`
- `runs/baseline_wti/scheduler/workers/Naive/...`
- `runs/baseline_wti/scheduler/workers/TimeXer/stderr.log`

---

## 27. 다음에 더 파고들 수 있는 주제

원하면 이어서 아래도 추가 설명 가능하다.

1. **모델 하나 예시로 fold 1개 내부에서 데이터가 어떻게 변형되는지**
   - 예: `TimeLLM` 또는 `iTransformer`

2. **왜 어떤 모델은 실패하고 어떤 모델은 성공하는지**
   - GPU 메모리/attention/multivariate shape 관점

3. **이 config를 실제로 더 빠르게 돌리려면 어디를 줄여야 하는지**
   - jobs 수
   - n_windows
   - Optuna trial 수
   - multivariate 모델 제외
