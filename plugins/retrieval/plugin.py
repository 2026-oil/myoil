"""StagePlugin implementation for standalone retrieval."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import config as _cfg

if TYPE_CHECKING:
    from app_config import AppConfig, LoadedConfig
    from tuning.search_space import SearchSpaceContract


class RetrievalStagePlugin:
    """Standalone retrieval plugin.

    This plugin does *not* own top-level job execution.  Standard models
    (GRU, Informer, etc.) go through the normal fit/predict path and the
    retrieval step is applied afterwards via :meth:`post_predict_fold`.
    """

    @property
    def config_key(self) -> str:
        return "retrieval"

    # ------------------------------------------------------------------
    # Config lifecycle
    # ------------------------------------------------------------------

    def default_config(self) -> _cfg.RetrievalPluginConfig:
        return _cfg.RetrievalPluginConfig()

    def normalize_config(
        self,
        payload: Any,
        *,
        unknown_keys: Any,
        coerce_bool: Any,
        coerce_optional_path_string: Any,
    ) -> _cfg.RetrievalPluginConfig:
        return _cfg.normalize_retrieval_plugin_config(
            payload,
            unknown_keys=unknown_keys,
            coerce_bool=coerce_bool,
            coerce_optional_path_string=coerce_optional_path_string,
        )

    def is_enabled(self, config: Any) -> bool:
        return isinstance(config, _cfg.RetrievalPluginConfig) and bool(config.enabled)

    def config_to_dict(self, config: Any) -> dict[str, Any] | None:
        if not self.is_enabled(config):
            return None
        return _cfg.retrieval_config_to_dict(config)

    # ------------------------------------------------------------------
    # Route validation & stage loading (no-ops for retrieval)
    # ------------------------------------------------------------------

    def validate_route(
        self,
        repo_root: Path,
        source_path: Path,
        config: Any,
        *,
        load_document: Any,
        unknown_keys: Any,
        coerce_bool: Any,
        coerce_name_tuple: Any,
    ) -> dict[str, Any] | None:
        del coerce_name_tuple
        if not self.is_enabled(config):
            return None
        if config.config_path is not None:
            from app_config import _resolve_relative_config_reference

            resolved = _resolve_relative_config_reference(
                repo_root, source_path, config.config_path,
            )
            if not resolved.exists():
                raise FileNotFoundError(
                    f"retrieval config_path does not exist: {resolved}"
                )
            suffix = resolved.suffix.lower()
            if suffix in {".yaml", ".yml"}:
                doc_type = "yaml"
            elif suffix == ".toml":
                doc_type = "toml"
            else:
                raise ValueError(
                    f"retrieval config_path must be .yaml/.yml/.toml, got {suffix!r}"
                )
            detail_payload = load_document(resolved, doc_type)
            _cfg.normalize_retrieval_detail_payload(
                detail_payload,
                unknown_keys=unknown_keys,
                coerce_bool=coerce_bool,
            )
        return {}

    def load_stage(
        self,
        repo_root: Path,
        *,
        source_path: Path,
        source_type: str,
        config: Any,
        search_space_contract: SearchSpaceContract | None,
    ) -> _cfg.RetrievalPluginConfig | None:
        del source_type, search_space_contract
        if not self.is_enabled(config) or config.config_path is None:
            return None
        from app_config import (
            _coerce_bool,
            _load_document,
            _resolve_relative_config_reference,
            _unknown_keys,
        )

        resolved = _resolve_relative_config_reference(
            repo_root, source_path, config.config_path,
        )
        suffix = resolved.suffix.lower()
        doc_type = "yaml" if suffix in {".yaml", ".yml"} else "toml"
        detail_payload = _load_document(resolved, doc_type, repo_root=repo_root)
        return _cfg.normalize_retrieval_detail_payload(
            detail_payload,
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
        )

    def apply_stage_to_config(self, config: AppConfig, stage_loaded: Any) -> AppConfig:
        if not isinstance(stage_loaded, _cfg.RetrievalPluginConfig):
            return config
        configured_path = getattr(config.stage_plugin_config, "config_path", None)
        if configured_path is None:
            return replace(config, stage_plugin_config=stage_loaded)
        return replace(
            config,
            stage_plugin_config=replace(stage_loaded, config_path=configured_path),
        )

    def stage_normalized_payload(
        self,
        config: AppConfig,
        stage_loaded: Any,
    ) -> dict[str, Any]:
        del config, stage_loaded
        return {}

    # ------------------------------------------------------------------
    # Search-space integration (passthrough / empty)
    # ------------------------------------------------------------------

    def supported_models(self) -> set[str]:
        return set()

    def stage_only_param_registry(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {}

    def normalize_search_space_sections(
        self,
        payload: dict[str, Any],
        *,
        normalize_model_section: Any,
        normalize_training_section: Any,
    ) -> dict[str, Any]:
        del payload, normalize_model_section, normalize_training_section
        return {}

    def model_search_space_key(self) -> str:
        return "models"

    def training_search_space_key(self) -> str:
        return "training"

    def validate_model(self, model_name: str) -> None:
        pass

    def owns_top_level_job(self, model_name: str) -> bool:
        del model_name
        return False

    def capabilities_for(self, model_name: str) -> dict[str, Any]:
        raise ValueError(
            f"retrieval plugin does not own top-level job execution for model {model_name!r}"
        )

    def model_search_space_fallback_key(self) -> str | None:
        return None

    def training_search_space_fallback_key(self) -> str | None:
        return None

    # ------------------------------------------------------------------
    # Runtime hooks
    # ------------------------------------------------------------------

    def prepare_fold_inputs(
        self,
        loaded: LoadedConfig,
        job: Any,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        *,
        run_root: Path | None,
    ) -> tuple[LoadedConfig, pd.DataFrame, pd.DataFrame, str]:
        del job, run_root
        return loaded, train_df, future_df, "retrieval"

    def predict_fold(
        self,
        loaded: LoadedConfig,
        job: Any,
        *,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        run_root: Path | None,
        params_override: dict[str, Any] | None = None,
        training_override: dict[str, Any] | None = None,
        fold_idx: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, Any | None]:
        del fold_idx
        raise NotImplementedError(
            "retrieval plugin does not own top-level jobs; "
            "prediction is handled by post_predict_fold"
        )

    def post_predict_fold(
        self,
        loaded: LoadedConfig,
        job: Any,
        *,
        target_predictions: pd.DataFrame,
        train_df: pd.DataFrame,
        transformed_train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        fitted_model: Any | None,
        run_root: Path | None,
    ) -> pd.DataFrame:
        """Apply retrieval as a post-prediction step."""
        from .runtime import post_predict_retrieval

        plugin_cfg: _cfg.RetrievalPluginConfig = loaded.config.stage_plugin_config
        if not plugin_cfg.enabled:
            return target_predictions

        config = loaded.config
        target_col = config.dataset.target_col
        dt_col = config.dataset.dt_col
        hist_exog_cols = config.dataset.hist_exog_cols

        # Derive input_size from the fitted model when available.
        input_size: int | None = None
        if fitted_model is not None:
            models_list = getattr(fitted_model, "models", None)
            if models_list:
                input_size = int(getattr(models_list[0], "input_size", 0))
        if not input_size:
            input_size = config.training.input_size

        horizon = config.cv.horizon

        pred_col = job.model
        for col in target_predictions.columns:
            if col.startswith(job.model):
                pred_col = col
                break

        return post_predict_retrieval(
            plugin_cfg=plugin_cfg,
            target_predictions=target_predictions,
            train_df=train_df,
            transformed_train_df=transformed_train_df,
            future_df=future_df,
            target_col=target_col,
            dt_col=dt_col,
            hist_exog_cols=hist_exog_cols,
            prediction_col=pred_col,
            input_size=input_size,
            horizon=horizon,
            run_root=run_root,
        )

    def materialize_stage(
        self,
        loaded: LoadedConfig,
        selected_jobs: Any,
        *,
        run_root: Path,
        main_resolved_path: Path,
        main_capability_path: Path,
        main_manifest_path: Path,
        entrypoint_version: str,
        validate_only: bool,
    ) -> None:
        del loaded, selected_jobs, run_root, main_resolved_path
        del main_capability_path, main_manifest_path, entrypoint_version, validate_only

    # ------------------------------------------------------------------
    # Manifest / validation payload
    # ------------------------------------------------------------------

    def manifest_block(self, loaded: LoadedConfig) -> dict[str, Any]:
        cfg = loaded.config.stage_plugin_config
        if not self.is_enabled(cfg):
            return {}
        return _cfg.retrieval_config_to_dict(cfg)

    def validation_payload(
        self,
        loaded: LoadedConfig,
        selected_jobs: Any,
        caps_by_model: dict[str, Any],
    ) -> dict[str, Any]:
        del selected_jobs, caps_by_model
        cfg = loaded.config.stage_plugin_config
        if not self.is_enabled(cfg):
            return {"retrieval_enabled": False}
        return {"retrieval_enabled": True, **_cfg.retrieval_config_to_dict(cfg)}

    # ------------------------------------------------------------------
    # Fanout helpers
    # ------------------------------------------------------------------

    def fanout_config_keys(self) -> set[str]:
        return {"retrieval"}

    def fanout_filter_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("retrieval fanout_filter_payload requires a dict payload")
        return {
            key: value
            for key, value in payload.items()
            if key in _cfg.RETRIEVAL_PLUGIN_MAIN_KEYS
        }

    def fanout_stage_payload(self, loaded: LoadedConfig) -> dict[str, Any] | None:
        raw = loaded.normalized_payload.get("retrieval")
        if not isinstance(raw, dict):
            return None
        if not raw.get("enabled"):
            return None
        return None


_plugin = RetrievalStagePlugin()

from plugin_contracts.stage_registry import register_stage_plugin  # noqa: E402

register_stage_plugin(_plugin)
