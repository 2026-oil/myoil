from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import Booster, DMatrix, train as xgb_train

from residual.optuna_spaces import DEFAULT_RESIDUAL_PARAMS
from residual.plugins_base import ResidualContext, ResidualPlugin


@dataclass(frozen=True)
class _XGBoostConfig:
    n_estimators: int = DEFAULT_RESIDUAL_PARAMS["n_estimators"]
    max_depth: int = DEFAULT_RESIDUAL_PARAMS["max_depth"]
    learning_rate: float = DEFAULT_RESIDUAL_PARAMS["learning_rate"]
    subsample: float = DEFAULT_RESIDUAL_PARAMS["subsample"]
    colsample_bytree: float = DEFAULT_RESIDUAL_PARAMS["colsample_bytree"]


class XGBoostResidualPlugin(ResidualPlugin):
    name = "xgboost"

    def __init__(
        self,
        *,
        n_estimators: int = DEFAULT_RESIDUAL_PARAMS["n_estimators"],
        max_depth: int = DEFAULT_RESIDUAL_PARAMS["max_depth"],
        learning_rate: float = DEFAULT_RESIDUAL_PARAMS["learning_rate"],
        subsample: float = DEFAULT_RESIDUAL_PARAMS["subsample"],
        colsample_bytree: float = DEFAULT_RESIDUAL_PARAMS["colsample_bytree"],
    ):
        self.config = _XGBoostConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )
        self.model: Booster | None = None
        self._trained = False
        self._fallback_value = 0.0
        self._checkpoint_path: Path | None = None

    @staticmethod
    def _prepare_panel(panel_df: pd.DataFrame) -> pd.DataFrame:
        panel = panel_df.copy()
        for column in ("cutoff", "ds"):
            panel[column] = pd.to_datetime(panel[column])
        panel["horizon_step"] = panel["horizon_step"].astype(int)
        panel["y_hat_base"] = panel["y_hat_base"].astype(float)
        panel["cutoff_day"] = panel["cutoff"].astype("int64") // 86_400_000_000_000
        panel["ds_day"] = panel["ds"].astype("int64") // 86_400_000_000_000
        return panel

    @classmethod
    def _feature_frame(cls, panel_df: pd.DataFrame) -> pd.DataFrame:
        prepared = cls._prepare_panel(panel_df)
        return prepared[
            [
                "horizon_step",
                "y_hat_base",
                "cutoff_day",
                "ds_day",
            ]
        ].astype(float)

    def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
        self._checkpoint_path = context.output_dir / "model.ubj"
        context.output_dir.mkdir(parents=True, exist_ok=True)
        if panel_df.empty:
            self._trained = False
            self._fallback_value = 0.0
            return
        train_panel = panel_df.dropna(subset=["residual_target"]).reset_index(drop=True)
        if train_panel.empty:
            self._trained = False
            self._fallback_value = 0.0
            return
        target = train_panel["residual_target"].astype(float)
        self._fallback_value = float(target.mean())
        features = self._feature_frame(train_panel)
        dtrain = DMatrix(features, label=target)
        self.model = xgb_train(
            params={
                "objective": "reg:squarederror",
                "max_depth": self.config.max_depth,
                "eta": self.config.learning_rate,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bytree,
                "verbosity": 0,
                "nthread": 1,
                "seed": 0,
            },
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
        )
        self.model.save_model(self._checkpoint_path)
        self._trained = True

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        ordered = panel_df.copy()
        if ordered.empty:
            ordered["residual_hat"] = pd.Series(dtype=float)
            return ordered
        if self._trained and self.model is not None:
            features = self._feature_frame(ordered)
            ordered["residual_hat"] = self.model.predict(DMatrix(features)).astype(
                float
            )
        else:
            ordered["residual_hat"] = float(self._fallback_value)
        return ordered

    def metadata(self) -> dict[str, Any]:
        return {
            "plugin": self.name,
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "checkpoint_path": str(self._checkpoint_path)
            if self._checkpoint_path
            else None,
        }
