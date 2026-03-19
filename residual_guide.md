# Residual Guide

이 문서는 **새 residual 모델을 추가하는 최소 절차**를 설명합니다.

목표:

- 기존 `jobs` / forecast model / scheduler 구조는 유지
- residual correction 레이어에만 새 모델 추가
- `config.yaml`에서 `residual.model`만 바꿔 쓸 수 있게 만들기

---

## 1. 현재 구조

현재 residual 확장 포인트는 아래 4곳입니다.

1. `residual/plugins_base.py`
   - 플러그인 인터페이스 정의
2. `residual/plugins/`
   - residual 모델 구현
3. `residual/registry.py`
   - `config.residual.model` → 실제 플러그인 매핑
4. `residual/config.py`
   - 지원 residual 모델 이름 검증

현재 내장 residual 모델:

- `xgboost`

즉 새 모델 추가는 보통 아래 흐름입니다.

1. 새 플러그인 파일 작성
2. 플러그인 export
3. registry 연결
4. config 지원 목록 추가

---

## 2. 플러그인 인터페이스

모든 residual 모델은 `ResidualPlugin` 인터페이스를 따라야 합니다.

```python
class ResidualPlugin(ABC):
    name: str

    @abstractmethod
    def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
        ...

    @abstractmethod
    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        ...
```

### 입력 데이터 의미

#### `fit(panel_df, context)`

`panel_df`는 residual 학습용 panel 데이터입니다. 대표 컬럼:

- `model_name`
- `unique_id`
- `fold_idx`
- `cutoff`
- `train_end_ds`
- `ds`
- `horizon_step`
- `y`
- `y_hat_base`
- `residual_target`

`residual_target = y - y_hat_base`

#### `predict(panel_df)`

fold evaluation panel residual 예측입니다.

반환에는 최소한 아래가 있어야 합니다.

- 기존 입력 컬럼들
- `residual_hat`

현재 XGBoost residual 모델은 panel 전체 컬럼을 다 feature로 쓰지 않고,
실질적으로 아래 의미의 입력만 사용합니다.

- `cutoff`
- `ds`
- `horizon_step`
- `y_hat_base`

구현에서는 XGBoost 입력을 위해 `cutoff` / `ds`를 숫자형 날짜 값으로 변환합니다.

#### `metadata()`

run artifact에 남길 설정 요약입니다.

예:

- plugin 이름
- 주요 하이퍼파라미터
- lookback / hidden size / epochs 등

---

## 3. 새 residual 모델 추가 순서

예시로 `mlp` residual 모델을 추가한다고 가정합니다.

### Step 1) 플러그인 파일 추가

새 파일:

- `residual/plugins/mlp.py`

예시 스키마:

```python
from __future__ import annotations

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
        self._trained = False

    def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
        ordered = panel_df.sort_values(["fold_idx", "ds"]).reset_index(drop=True)
        # TODO: residual_target 기반 학습 로직 구현
        self._trained = True

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("Residual plugin is not trained")
        ordered = panel_df.sort_values(["fold_idx", "ds"]).reset_index(drop=True).copy()
        ordered["residual_hat"] = 0.0
        return ordered

    def metadata(self) -> dict[str, Any]:
        return {
            "plugin": self.name,
            "lookback": self.config.lookback,
            "hidden_size": self.config.hidden_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
        }
```

주의:

- 최소 구현이라도 `predict`는 **반드시 DataFrame 반환**
- 결과 컬럼에 `residual_hat` 포함
- `fit` 호출 전 예측 시 명확히 에러를 내는 편이 안전

---

### Step 2) 플러그인 export 추가

파일:

- `residual/plugins/__init__.py`

예:

```python
from .mlp import MLPResidualPlugin
from .xgboost import XGBoostResidualPlugin

__all__ = [
    "MLPResidualPlugin",
    "XGBoostResidualPlugin",
]
```

---

### Step 3) registry 연결

파일:

- `residual/registry.py`

예시:

```python
from .plugins import MLPResidualPlugin, XGBoostResidualPlugin


def build_residual_plugin(config: Any) -> ResidualPlugin:
    if is_dataclass(config):
        config = asdict(config)

    name = str(config.get("model", "xgboost")).lower()
    params = dict(config.get("params", {}))

    if name == "xgboost":
        return XGBoostResidualPlugin(
            n_estimators=int(params.get("n_estimators", 32)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.1)),
        )

    if name == "mlp":
        return MLPResidualPlugin(
            lookback=int(params.get("lookback", 8)),
            hidden_size=int(params.get("hidden_size", 32)),
            epochs=int(params.get("epochs", 50)),
            learning_rate=float(params.get("learning_rate", 0.001)),
        )

    raise ValueError(f"Unsupported residual model: {name}")
```

핵심:

- `config.residual.model` 문자열과 registry branch 이름이 일치해야 함
- `params` parsing은 registry에서 명시적으로 처리하는 편이 지금 구조와 가장 잘 맞음

---

### Step 4) config 지원 목록 추가

파일:

- `residual/config.py`

현재는 `SUPPORTED_RESIDUAL_MODELS`로 허용 모델을 검증합니다.

예:

```python
SUPPORTED_RESIDUAL_MODELS = {"xgboost", "mlp"}
```

이걸 안 바꾸면 config에서 아래처럼 적어도 초기 로딩에서 막힙니다.

```yaml
residual:
  model: mlp
```

---

## 4. config.yaml 설정 예시

새 residual 모델을 붙일 때는 `jobs`를 바꿀 필요가 없습니다.

즉 아래만 바꾸면 됩니다.

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

설명:

- `enabled`: residual correction on/off
- `model`: registry에서 선택할 residual 모델 이름
- `params`: 해당 residual 모델 전용 하이퍼파라미터

중요:

- `jobs:`는 **base forecasting model 실행 목록**
- `residual:`은 **그 예측 위에 덧붙는 correction model**

즉 residual 모델을 늘린다고 `jobs`를 새 형식으로 바꾸는 게 아닙니다.

---

## 5. 어떤 컬럼을 기준으로 학습하나

현재 runtime은 fold-specific CV residual panel 하나만 residual 학습에 사용합니다.

runtime이 CV 예측을 모아서 아래 개념의 테이블을 만듭니다.

- `y`
- `y_hat_base`
- `residual_target = y - y_hat_base`
- `fold_count`

이 panel은 겹치는 시점이 있으면 날짜 단위로 집계된 뒤 residual plugin으로 전달됩니다.

따라서 새 residual 모델은 보통 **`residual_target` 시계열 문제**만 풀면 됩니다.

---

## 6. 구현시 권장 체크리스트

새 residual 모델을 추가할 때는 아래를 확인하세요.

- [ ] `ResidualPlugin` 4개 메서드 구현
- [ ] `predict` 반환값에 `residual_hat` 포함
- [ ] `metadata()`에 핵심 파라미터 기록
- [ ] `residual/plugins/__init__.py` export 추가
- [ ] `residual/registry.py` branch 추가
- [ ] `residual/config.py` 지원 모델 목록 추가
- [ ] `config.yaml`에서 `residual.model` 이름과 registry 이름 일치 확인

권장 추가 작업:

- [ ] 최소 smoke test 추가
- [ ] `validate-only`로 config 로딩 확인
- [ ] 단일 job 실행으로 residual artifact 생성 확인

---

## 7. 최소 smoke 확인 예시

config에 새 모델을 연결한 뒤:

```bash
cd neuralforecast
uv run python main.py --validate-only --config config.yaml
```

그 다음 단일 job으로 확인:

```bash
cd neuralforecast
uv run python main.py --config config.yaml --jobs TFT --output-root runs/single-job-smoke
```

확인 포인트:

- `runs/.../residual/<job>/plugin_metadata.json`
- `runs/.../residual/<job>/corrected_folds.csv`
- `runs/.../residual/<job>/diagnostics.json`

---

## 8. 추천 원칙

새 residual 모델을 추가할 때는 아래 원칙이 가장 안전합니다.

1. **config 이름은 짧고 명확하게**
   - 예: `xgboost`, `mlp`
2. **registry에서 파라미터 파싱을 명시적으로**
   - 지금 구조와 가장 잘 맞음
3. **plugin은 residual correction만 책임지게**
   - base forecast 로직과 섞지 않기
4. **`jobs` 구조는 건드리지 않기**
   - residual은 상위 correction layer
5. **artifact-friendly metadata 남기기**
   - 후속 분석/디버깅이 쉬워짐

---

## 9. 한 줄 요약

새 residual 모델 추가는 보통 아래 4단계면 충분합니다.

1. `residual/plugins/<name>.py` 작성
2. `residual/plugins/__init__.py` export
3. `residual/registry.py` 연결
4. `residual/config.py` 지원 목록 추가

그 다음 `config.yaml`에서:

```yaml
residual:
  model: <name>
  params: { ... }
```

로 바꾸면 됩니다.
