from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from residual.models.backends._base import BaseResidualPlugin
from residual.optuna_spaces import RESIDUAL_DEFAULTS


@dataclass(frozen=True)
class _RandomForestConfig:
    n_estimators: int = int(RESIDUAL_DEFAULTS["randomforest"]["n_estimators"])
    max_depth: int = int(RESIDUAL_DEFAULTS["randomforest"]["max_depth"])
    min_samples_leaf: int = int(RESIDUAL_DEFAULTS["randomforest"]["min_samples_leaf"])
    max_features: str = str(RESIDUAL_DEFAULTS["randomforest"]["max_features"])
    cpu_threads: int | None = None


class RandomForestResidualPlugin(BaseResidualPlugin):
    name = "randomforest"

    def __init__(
        self,
        *,
        n_estimators: int = int(RESIDUAL_DEFAULTS["randomforest"]["n_estimators"]),
        max_depth: int = int(RESIDUAL_DEFAULTS["randomforest"]["max_depth"]),
        min_samples_leaf: int = int(
            RESIDUAL_DEFAULTS["randomforest"]["min_samples_leaf"]
        ),
        max_features: str = str(RESIDUAL_DEFAULTS["randomforest"]["max_features"]),
        cpu_threads: int | None = None,
        feature_config: Any = None,
    ):
        super().__init__(
            config=_RandomForestConfig(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                cpu_threads=cpu_threads,
            ),
            feature_config=feature_config,
        )

    def _train_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=0,
            n_jobs=self.config.cpu_threads,
        )
        self.model.fit(features, target)
        with self._checkpoint_path.open("wb") as handle:
            pickle.dump(self.model, handle)

    def _predict_values(self, features: pd.DataFrame) -> Any:
        return self.model.predict(features).astype(float)
