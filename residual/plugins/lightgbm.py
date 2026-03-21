from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from residual.features import ResidualFeatureConfig, build_residual_feature_frame
from residual.optuna_spaces import DEFAULT_RESIDUAL_PARAMS_BY_MODEL
from residual.plugins_base import ResidualContext, ResidualPlugin


@dataclass(frozen=True)
class _LightGBMConfig:
    n_estimators: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["n_estimators"]
    max_depth: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["max_depth"]
    learning_rate: float = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["learning_rate"]
    num_leaves: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["num_leaves"]
    min_child_samples: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["min_child_samples"]
    feature_fraction: float = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["feature_fraction"]


class LightGBMResidualPlugin(ResidualPlugin):
    name = "lightgbm"

    def __init__(
        self,
        *,
        n_estimators: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["n_estimators"],
        max_depth: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["max_depth"],
        learning_rate: float = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["learning_rate"],
        num_leaves: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["num_leaves"],
        min_child_samples: int = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["min_child_samples"],
        feature_fraction: float = DEFAULT_RESIDUAL_PARAMS_BY_MODEL["lightgbm"]["feature_fraction"],
        feature_config: Any = None,
    ):
        self.config = _LightGBMConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            feature_fraction=feature_fraction,
        )
        self.model: LGBMRegressor | None = None
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
        self.model = LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            feature_fraction=self.config.feature_fraction,
            random_state=0,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(features, target)
        joblib.dump(self.model, self._checkpoint_path)
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
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "min_child_samples": self.config.min_child_samples,
            "feature_fraction": self.config.feature_fraction,
            "checkpoint_path": str(self._checkpoint_path) if self._checkpoint_path else None,
        }
