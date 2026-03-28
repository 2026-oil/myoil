from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from xgboost import Booster, DMatrix, train as xgb_train

from ._base import BaseResidualPlugin
from tuning.search_space import RESIDUAL_DEFAULTS, RESIDUAL_INTERNAL_OPTIMIZER_DEFAULTS


@dataclass(frozen=True)
class _XGBoostConfig:
    n_estimators: int = RESIDUAL_DEFAULTS["xgboost"]["n_estimators"]
    max_depth: int = RESIDUAL_DEFAULTS["xgboost"]["max_depth"]
    subsample: float = RESIDUAL_DEFAULTS["xgboost"]["subsample"]
    colsample_bytree: float = RESIDUAL_DEFAULTS["xgboost"]["colsample_bytree"]
    cpu_threads: int | None = None


class XGBoostResidualPlugin(BaseResidualPlugin):
    name = "xgboost"

    def __init__(
        self,
        *,
        n_estimators: int = RESIDUAL_DEFAULTS["xgboost"]["n_estimators"],
        max_depth: int = RESIDUAL_DEFAULTS["xgboost"]["max_depth"],
        subsample: float = RESIDUAL_DEFAULTS["xgboost"]["subsample"],
        colsample_bytree: float = RESIDUAL_DEFAULTS["xgboost"]["colsample_bytree"],
        cpu_threads: int | None = None,
        feature_config: Any = None,
    ):
        super().__init__(
            config=_XGBoostConfig(
                n_estimators=n_estimators,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                cpu_threads=cpu_threads,
            ),
            feature_config=feature_config,
        )

    def _train_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        dtrain = DMatrix(features, label=target)
        self.model: Booster = xgb_train(
            params={
                "objective": "reg:squarederror",
                "max_depth": self.config.max_depth,
                "eta": RESIDUAL_INTERNAL_OPTIMIZER_DEFAULTS["xgboost"]["eta"],
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bytree,
                "verbosity": 0,
                "seed": 0,
                **(
                    {"nthread": int(self.config.cpu_threads)}
                    if self.config.cpu_threads is not None
                    else {}
                ),
            },
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
        )
        self.model.save_model(self._checkpoint_path)

    def _predict_values(self, features: pd.DataFrame) -> Any:
        return self.model.predict(DMatrix(features)).astype(float)
