# HPO search space 레퍼런스

기준 파일: `yaml/HPO/search_space.yaml`

이 파일은 **auto tuning / search-space 계약**입니다. 모든 실험이 항상 이 파일을 쓰는 것은 아니고, loader가 auto mode가 필요하다고 판단할 때 의미를 가집니다.

Source: `yaml/HPO/search_space.yaml:1-715`, `app_config.py:1708-1735`, `plugins/bs_preforcast/plugin.py:179-212`

## loader가 이 파일을 읽는 경우

- non-baseline job의 `params`가 비어 있음
- probed stage job의 `params`가 비어 있음

즉, 이 파일은 **항상 적용되는 기본값 파일**이 아니라 **auto mode에서 탐색 가능한 후보 집합**입니다.

Source: `app_config.py:1708-1735`

## 공통 schema 읽는 법

- `type: categorical` → `choices` 중 하나를 고름
- `type: int` → `low/high/step` 범위에서 정수 탐색
- `type: float` → `low/high` 범위에서 실수 탐색

복합 리스트 choice는 반드시 **문자열이 아니라 native YAML list** 여야 합니다.

좋은 예: `[[1, 0, 0], [1, 1, 0]]`  / 나쁜 예: `"[1, 1, 0]"`

Source: `tests/test_bs_preforcast_config.py:442-570`, `tests/test_bs_preforcast_config.py:614-615`

## `models`

메인 learned-model search surface 입니다.

### `LSTM`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `encoder_hidden_size` | `categorical` | `256`, `512`, `768` |
| `encoder_n_layers` | `categorical` | `4`, `6`, `8` |
| `inference_input_size` | `categorical` | `32`, `64`, `128` |
| `encoder_dropout` | `categorical` | `0.0`, `0.1`, `0.2`, `0.3` |
| `decoder_hidden_size` | `categorical` | `256`, `512`, `768` |
| `decoder_layers` | `categorical` | `2`, `4` |
| `context_size` | `categorical` | `16`, `32`, `64` |

### `iTransformer`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `hidden_size` | `categorical` | `256`, `512`, `768` |
| `n_heads` | `categorical` | `16`, `32` |
| `e_layers` | `categorical` | `4`, `8` |
| `d_ff` | `categorical` | `1024`, `2048` |
| `d_layers` | `categorical` | `4`, `8` |
| `factor` | `categorical` | `4`, `8` |
| `dropout` | `categorical` | `0.0`, `0.1`, `0.2` |
| `use_norm` | `categorical` | `true`, `false` |

### `TSMixerx`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `n_block` | `categorical` | `4`, `8` |
| `ff_dim` | `categorical` | `256`, `512`, `1024` |
| `dropout` | `categorical` | `0.1`, `0.2`, `0.3` |
| `revin` | `categorical` | `true`, `false` |

### `TimeXer`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `patch_len` | `categorical` | `8`, `16` |
| `hidden_size` | `categorical` | `256`, `512`, `768` |
| `n_heads` | `categorical` | `16`, `32` |
| `e_layers` | `categorical` | `4`, `8` |
| `d_ff` | `categorical` | `512`, `1024` |
| `factor` | `categorical` | `4`, `8` |
| `dropout` | `categorical` | `0.1`, `0.2`, `0.3` |
| `use_norm` | `categorical` | `true` |

### `NonstationaryTransformer`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `hidden_size` | `categorical` | `64`, `128`, `256` |
| `dropout` | `categorical` | `0.0`, `0.1`, `0.2`, `0.3` |
| `n_head` | `categorical` | `4`, `8` |
| `conv_hidden_size` | `categorical` | `64`, `128`, `256` |
| `encoder_layers` | `categorical` | `1`, `2`, `3` |
| `decoder_layers` | `categorical` | `1`, `2` |

Source: `yaml/HPO/search_space.yaml:1-188`

## `training.global`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `input_size` | `categorical` | `48`, `64`, `96` |
| `batch_size` | `categorical` | `16`, `32`, `64`, `128` |
| `scaler_type` | `categorical` | `null` |
| `model_step_size` | `categorical` | `4`, `8` |

Source: `yaml/HPO/search_space.yaml:189-212`

## `training.per_model`

### `LSTM`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `input_size` | `categorical` | `24`, `48`, `96` |
| `batch_size` | `categorical` | `32`, `64` |
| `scaler_type` | `categorical` | `null` |
| `model_step_size` | `categorical` | `4`, `8` |

### `TSMixerx`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `input_size` | `categorical` | `48`, `64`, `72` |
| `batch_size` | `categorical` | `16` |
| `scaler_type` | `categorical` | `null` |
| `model_step_size` | `categorical` | `4`, `8` |

### `TimeXer`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `input_size` | `categorical` | `48`, `64`, `96` |
| `batch_size` | `categorical` | `16`, `32`, `64`, `128` |
| `scaler_type` | `categorical` | `null` |
| `model_step_size` | `categorical` | `4`, `8` |

### `iTransformer`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `input_size` | `categorical` | `48`, `64`, `96` |
| `batch_size` | `categorical` | `16` |
| `scaler_type` | `categorical` | `null` |
| `model_step_size` | `categorical` | `4` |

### `NonstationaryTransformer`

| 필드 | 타입 | 가능한 값 / 범위 |
|---|---|---|
| `input_size` | `categorical` | `48`, `64`, `96` |
| `batch_size` | `categorical` | `16`, `32`, `64` |
| `scaler_type` | `categorical` | `null` |
| `model_step_size` | `categorical` | `4`, `8` |

Source: `yaml/HPO/search_space.yaml:213-318`
