from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from residual.optuna_spaces import RESIDUAL_DEFAULTS
from residual.plugins_base import ResidualContext, ResidualPlugin


@dataclass(frozen=True)
class _RandomForestConfig:
    n_estimators: int = int(RESIDUAL_DEFAULTS["randomforest"]["n_estimators"])
    max_depth: int = int(RESIDUAL_DEFAULTS["randomforest"]["max_depth"])
    min_samples_leaf: int = int(RESIDUAL_DEFAULTS["randomforest"]["min_samples_leaf"])
    max_features: str = str(RESIDUAL_DEFAULTS["randomforest"]["max_features"])


class RandomForestResidualPlugin(ResidualPlugin):
    name = "randomforest"

    def __init__(self, *, n_estimators: int = 200, max_depth: int = 6, min_samples_leaf: int = 2, max_features: str = "sqrt"):
        self.config = _RandomForestConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        self.model: RandomForestRegressor | None = None
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
        return prepared[["horizon_step", "y_hat_base", "cutoff_day", "ds_day"]].astype(float)

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
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=0,
            n_jobs=1,
        )
        self.model.fit(features, target)
        with self._checkpoint_path.open("wb") as handle:
            pickle.dump(self.model, handle)
        self._trained = True

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        ordered = panel_df.copy()
        if ordered.empty:
            ordered["residual_hat"] = pd.Series(dtype=float)
            return ordered
        if self._trained and self.model is not None:
            features = self._feature_frame(ordered)
            ordered["residual_hat"] = self.model.predict(features).astype(float)
        else:
            ordered["residual_hat"] = float(self._fallback_value)
        return ordered

    def metadata(self) -> dict[str, Any]:
        return {
            "plugin": self.name,
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "min_samples_leaf": self.config.min_samples_leaf,
            "max_features": self.config.max_features,
            "checkpoint_path": str(self._checkpoint_path) if self._checkpoint_path else None,
        }
