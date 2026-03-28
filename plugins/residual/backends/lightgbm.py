from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from ._base import BaseResidualPlugin
from tuning.search_space import (
    DEFAULT_RESIDUAL_PARAMS_BY_MODEL,
    RESIDUAL_INTERNAL_OPTIMIZER_DEFAULTS,
)


@dataclass(frozen=True)
class _LightGBMConfig:
    n_estimators: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["n_estimators"]
    max_depth: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["max_depth"]
    num_leaves: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["num_leaves"]
    min_child_samples: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["min_child_samples"]
    feature_fraction: float = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["feature_fraction"]
    cpu_threads: int | None = None


class LightGBMResidualPlugin(BaseResidualPlugin):
    name = "lightgbm"

    def __init__(
        self,
        *,
        n_estimators: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["n_estimators"],
        max_depth: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["max_depth"],
        num_leaves: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["num_leaves"],
        min_child_samples: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["min_child_samples"],
        feature_fraction: float = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["feature_fraction"],
        cpu_threads: int | None = None,
        feature_config: Any = None,
    ):
        super().__init__(
            config=_LightGBMConfig(
                n_estimators=n_estimators,
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                feature_fraction=feature_fraction,
                cpu_threads=cpu_threads,
            ),
            feature_config=feature_config,
        )

    def _train_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.model = LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            feature_fraction=self.config.feature_fraction,
            learning_rate=RESIDUAL_INTERNAL_OPTIMIZER_DEFAULTS["lightgbm"]["shrinkage_rate"],
            random_state=0,
            n_jobs=self.config.cpu_threads,
            verbosity=-1,
        )
        self.model.fit(features, target)
        joblib.dump(self.model, self._checkpoint_path)

    def _predict_values(self, features: pd.DataFrame) -> Any:
        return self.model.predict(features).astype(float)
