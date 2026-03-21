from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import Booster, DMatrix, train as xgb_train

from residual.features import ResidualFeatureConfig, build_residual_feature_frame
from residual.optuna_spaces import RESIDUAL_DEFAULTS
from residual.plugins_base import ResidualContext, ResidualPlugin


@dataclass(frozen=True)
class _XGBoostConfig:
    n_estimators: int = RESIDUAL_DEFAULTS["xgboost"]["n_estimators"]
    max_depth: int = RESIDUAL_DEFAULTS["xgboost"]["max_depth"]
    learning_rate: float = RESIDUAL_DEFAULTS["xgboost"]["learning_rate"]
    subsample: float = RESIDUAL_DEFAULTS["xgboost"]["subsample"]
    colsample_bytree: float = RESIDUAL_DEFAULTS["xgboost"]["colsample_bytree"]


class XGBoostResidualPlugin(ResidualPlugin):
    name = "xgboost"

    def __init__(
        self,
        *,
        n_estimators: int = RESIDUAL_DEFAULTS["xgboost"]["n_estimators"],
        max_depth: int = RESIDUAL_DEFAULTS["xgboost"]["max_depth"],
        learning_rate: float = RESIDUAL_DEFAULTS["xgboost"]["learning_rate"],
        subsample: float = RESIDUAL_DEFAULTS["xgboost"]["subsample"],
        colsample_bytree: float = RESIDUAL_DEFAULTS["xgboost"]["colsample_bytree"],
        feature_config: Any = None,
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
        self._feature_config_override = feature_config
        self._resolved_feature_config: ResidualFeatureConfig | None = None
        self._feature_columns: tuple[str, ...] | None = None

    def _feature_frame(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        feature_frame = build_residual_feature_frame(
            panel_df,
            feature_config=(
                self._resolved_feature_config
                if self._resolved_feature_config is not None
                else self._feature_config_override
            ),
            required_columns=self._feature_columns,
        )
        return feature_frame.frame

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
        feature_frame = build_residual_feature_frame(
            train_panel,
            feature_config=(
                self._feature_config_override
                if self._feature_config_override is not None
                else context.config
            ),
        )
        self._resolved_feature_config = feature_frame.resolved_config
        self._feature_columns = feature_frame.columns
        features = feature_frame.frame
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
