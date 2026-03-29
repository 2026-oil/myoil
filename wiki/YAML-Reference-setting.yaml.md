# setting.yaml 레퍼런스

기준 파일: `yaml/setting/setting.yaml`

이 파일은 여러 실험 YAML에 공통으로 적용할 **shared defaults** 를 담습니다. composition root가 아니라, 정규화 전에 `load_app_config()`가 병합하는 shared-default 레이어입니다.

Source: `yaml/setting/setting.yaml:1-37`, `app_config.py:68-95`, `app_config.py:1639-1667`

## 이 파일이 소유하는 것

`app_config.py` 기준 shared-settings owned path는 명시돼 있습니다.

- `runtime.random_seed`
- `training.input_size`
- `training.batch_size`
- `training.valid_batch_size`
- `training.windows_batch_size`
- `training.inference_windows_batch_size`
- `training.lr_scheduler`
- `training.max_steps`
- `training.val_size`
- `training.val_check_steps`
- `training.model_step_size`
- `training.early_stop_patience_steps`
- `training.loss`
- `training.optimizer`
- `cv.gap`
- `cv.horizon`
- `cv.step_size`
- `cv.n_windows`
- `cv.max_train_size`
- `cv.overlap_eval_policy`
- `scheduler.gpu_ids`
- `scheduler.max_concurrent_jobs`
- `scheduler.worker_devices`

Source: `app_config.py:68-92`

## 현재 파일 구조

```yaml
runtime:
  random_seed: 1
training:
  input_size: 64
  batch_size: 32
  valid_batch_size: 64
  windows_batch_size: 1024
  inference_windows_batch_size: 1024
  max_steps: 1000
  val_size: 8
  val_check_steps: 50
  model_step_size: 8
  early_stop_patience_steps: 5
  loss: mse
  optimizer:
    name: adamw
    kwargs: {}
  lr_scheduler:
    name: OneCycleLR
    max_lr: 0.001
    pct_start: 0.3
    div_factor: 25.0
    final_div_factor: 10000.0
    anneal_strategy: cos
    three_phase: false
    cycle_momentum: false
cv:
  gap: 0
  horizon: 8
  step_size: 8
  n_windows: 6
  max_train_size: null
  overlap_eval_policy: by_cutoff_mean
scheduler:
  gpu_ids: [0, 1]
  max_concurrent_jobs: 2
  worker_devices: 1
```

Source: `yaml/setting/setting.yaml:1-37`

## 필드별 설명

### `runtime.random_seed`
- 의미: 공통 재현성 seed
- 현재 파일 값: `1`
- 타입: 정수

Source: `yaml/setting/setting.yaml:1-2`

### `training.input_size`
- 의미: 기본 입력 윈도 길이
- 현재 파일 값: `64`
- 관련 categorical 후보값 예시: `48`, `64`, `96` (`training.global` search-space 기준)

### `training.batch_size`
- 의미: 기본 학습 배치 크기
- 현재 파일 값: `32`
- 관련 categorical 후보값 예시: `16`, `32`, `64`, `128`

### `training.valid_batch_size`
- 의미: validation 배치 크기
- 현재 파일 값: `64`
- 비고: `search_space.yaml`에는 별도 categorical 후보가 정의돼 있지 않아 현재는 설정값 자체를 기준으로 봐야 합니다.

### `training.windows_batch_size`
- 의미: 학습 시 window 샘플 처리 단위
- 현재 파일 값: `1024`
- 비고: 현재 search-space 후보 정의 없음

### `training.inference_windows_batch_size`
- 의미: 추론 시 window 배치 단위
- 현재 파일 값: `1024`
- 비고: 현재 search-space 후보 정의 없음

### `training.max_steps`
- 의미: 최대 학습 step 수
- 현재 파일 값: `1000`
- 비고: 현재 search-space 후보 정의 없음

### `training.val_size`
- 의미: validation 분할 크기
- 현재 파일 값: `8`
- 비고: 현재 search-space 후보 정의 없음

### `training.val_check_steps`
- 의미: validation 체크 주기
- 현재 파일 값: `50`
- 비고: 현재 search-space 후보 정의 없음

### `training.model_step_size`
- 의미: 모델 step size 기본값
- 현재 파일 값: `8`
- categorical 후보값: `4`, `8`

### `training.early_stop_patience_steps`
- 의미: early stopping patience
- 현재 파일 값: `5`
- 비고: 현재 search-space 후보 정의 없음

### `training.loss`
- 의미: 손실 함수
- 현재 파일 값: `mse`
- 가능한 값: `mse`, `exloss`

Source: `app_config.py:37`, `app_config.py:196`, `app_config.py:1172-1178`

### `training.optimizer.name`
- 의미: optimizer 종류
- 현재 파일 값: `adamw`
- 가능한 categorical 값: `adamw`, `ademamix`, `mars`, `soap`

### `training.optimizer.kwargs`
- 의미: optimizer 추가 kwargs
- 현재 파일 값: `{}`
- 타입: mapping

Source: `app_config.py:38`, `app_config.py:170-172`, `app_config.py:705-723`

### `training.lr_scheduler.name`
- 의미: lr scheduler 종류
- 현재 파일 값: `OneCycleLR`
- 가능한 값: `OneCycleLR`만 허용

### `training.lr_scheduler.anneal_strategy`
- 의미: annealing 방식
- 현재 파일 값: `cos`
- 가능한 categorical 값: `cos`, `linear`

### `training.lr_scheduler.max_lr`
- 의미: peak learning rate
- 현재 파일 값: `0.001`
- 제약: `> 0`

### `training.lr_scheduler.pct_start`
- 의미: warmup 비율
- 현재 파일 값: `0.3`
- 제약: `0 < value < 1`

### `training.lr_scheduler.div_factor`
- 의미: initial lr divisor
- 현재 파일 값: `25.0`
- 제약: `> 1`

### `training.lr_scheduler.final_div_factor`
- 의미: final lr divisor
- 현재 파일 값: `10000.0`
- 제약: `> 1`

### `training.lr_scheduler.three_phase`
- 의미: 3-phase scheduler 사용 여부
- 현재 파일 값: `false`
- 가능한 값: `true`, `false`

### `training.lr_scheduler.cycle_momentum`
- 의미: cycle momentum 사용 여부
- 현재 파일 값: `false`
- 가능한 값: `true`, `false`

Source: `yaml/setting/setting.yaml:18-26`, `app_config.py:156-166`, `app_config.py:630-691`

### `cv.gap`
- 의미: cutoff와 평가 구간 사이 간격
- 현재 파일 값: `0`

### `cv.horizon`
- 의미: 예측 horizon
- 현재 파일 값: `8`

### `cv.step_size`
- 의미: 윈도 이동 크기
- 현재 파일 값: `8`

### `cv.n_windows`
- 의미: CV 윈도 개수
- 현재 파일 값: `6`

### `cv.max_train_size`
- 의미: train size 상한
- 현재 파일 값: `null`

### `cv.overlap_eval_policy`
- 의미: 겹치는 평가 구간 집계 정책
- 현재 파일 값: `by_cutoff_mean`
- 가능한 categorical 값: `by_cutoff_mean`만 허용

Source: `yaml/setting/setting.yaml:27-33`, `app_config.py:216-222`

### `scheduler.gpu_ids`
- 의미: 사용 가능한 GPU id 목록
- 현재 파일 값: `[0, 1]`

### `scheduler.max_concurrent_jobs`
- 의미: 동시에 실행할 최대 job 수
- 현재 파일 값: `2`

### `scheduler.worker_devices`
- 의미: worker 하나가 사용할 device 수
- 현재 파일 값: `1`

Source: `yaml/setting/setting.yaml:34-37`, `app_config.py:226-230`

## safe authoring notes

- shared default를 바꾸면 여러 실험에 동시에 영향이 갈 수 있으므로 `--setting`을 포함한 validate-only를 먼저 돌리는 것이 안전합니다.
- categorical 값이 코드에서 제한되는 필드(`loss`, `optimizer.name`, `lr_scheduler.name`, `anneal_strategy`, `cv.overlap_eval_policy`)는 허용값 밖으로 벗어나면 fail-fast 될 수 있습니다.

## 관련 페이지

- [메인 실험 YAML 안전하게 작성하기](Authoring-a-Main-Experiment-YAML-Safely)
- [설정 로딩과 YAML 소유권 맵](Config-Loading-and-YAML-Authority-Map)
- [HPO search space 레퍼런스](YAML-Reference-HPO-search-space)
