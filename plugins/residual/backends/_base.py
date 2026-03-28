from __future__ import annotations

from abc import abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from residual.features import FlatResidualFeatureConfig, build_residual_feature_frame
from residual.plugins_base import ResidualContext, ResidualPlugin


class BaseResidualPlugin(ResidualPlugin):
    """Shared scaffold for tree-based residual plugins (XGBoost, LightGBM, RandomForest)."""

    def __init__(self, *, config: Any, feature_config: Any = None) -> None:
        self.config = config
        self.model: Any = None
        self._trained = False
        self._fallback_value = 0.0
        self._checkpoint_path: Path | None = None
        self._feature_config_override = feature_config
        self._resolved_feature_config: FlatResidualFeatureConfig | None = None
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
        self._checkpoint_path = context.output_dir / self._checkpoint_filename()
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
        self._train_model(feature_frame.frame, target)
        self._trained = True

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        ordered = panel_df.copy()
        if ordered.empty:
            ordered["residual_hat"] = pd.Series(dtype=float)
            return ordered
        if self._trained and self.model is not None:
            features = self._feature_frame(ordered)
            ordered["residual_hat"] = self._predict_values(features)
        else:
            ordered["residual_hat"] = float(self._fallback_value)
        return ordered

    def metadata(self) -> dict[str, Any]:
        return {
            "plugin": self.name,
            **asdict(self.config),
            "checkpoint_path": str(self._checkpoint_path)
            if self._checkpoint_path
            else None,
        }

    def _checkpoint_filename(self) -> str:
        return "model.ubj"

    @abstractmethod
    def _train_model(self, features: pd.DataFrame, target: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict_values(self, features: pd.DataFrame) -> Any:
        raise NotImplementedError
