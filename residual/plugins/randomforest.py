from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from residual.features import ResidualFeatureConfig, build_residual_feature_frame
from residual.optuna_spaces import RESIDUAL_DEFAULTS
from residual.plugins_base import ResidualContext, ResidualPlugin


@dataclass(frozen=True)
class _RandomForestConfig:
    n_estimators: int = int(RESIDUAL_DEFAULTS["randomforest"]["n_estimators"])
    max_depth: int = int(RESIDUAL_DEFAULTS["randomforest"]["max_depth"])
    min_samples_leaf: int = int(RESIDUAL_DEFAULTS["randomforest"]["min_samples_leaf"])
    max_features: str = str(RESIDUAL_DEFAULTS["randomforest"]["max_features"])
    cpu_threads: int | None = None


class RandomForestResidualPlugin(ResidualPlugin):
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
        self.config = _RandomForestConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            cpu_threads=cpu_threads,
        )
        self.model: RandomForestRegressor | None = None
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
            "cpu_threads": self.config.cpu_threads,
            "checkpoint_path": str(self._checkpoint_path) if self._checkpoint_path else None,
        }
