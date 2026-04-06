"""StagePlugin implementation for AA-Forecast experiment routing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import config as _cfg

if TYPE_CHECKING:
    from app_config import AppConfig, LoadedConfig
    from tuning.search_space import SearchSpaceContract


class AAForecastStagePlugin:
    @property
    def config_key(self) -> str:
        return "aa_forecast"

    def default_config(self) -> _cfg.AAForecastPluginConfig:
        return _cfg.AAForecastPluginConfig()

    def normalize_config(
        self,
        payload: Any,
        *,
        unknown_keys: Any,
        coerce_bool: Any,
        coerce_optional_path_string: Any,
    ) -> _cfg.AAForecastPluginConfig:
        return _cfg.normalize_aa_forecast_config(
            payload,
            unknown_keys=unknown_keys,
            coerce_bool=coerce_bool,
            coerce_optional_path_string=coerce_optional_path_string,
        )

    def is_enabled(self, config: Any) -> bool:
        return isinstance(config, _cfg.AAForecastPluginConfig) and bool(config.enabled)

    def config_to_dict(self, config: Any) -> dict[str, Any] | None:
        if not self.is_enabled(config):
            return None
        return _cfg.aa_forecast_to_dict(config)

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
        del coerce_name_tuple, load_document
        if not self.is_enabled(config):
            return None
        if config.config_path is not None:
            from app_config import _resolve_relative_config_reference

            stage_source_path = _resolve_relative_config_reference(
                repo_root,
                source_path,
                config.config_path,
            )
            if not stage_source_path.exists():
                raise FileNotFoundError(
                    f"aa_forecast selected route does not exist: {stage_source_path}"
                )
        linked = (
            _cfg.load_aa_forecast_stage1(
                repo_root,
                source_path=source_path,
                source_type="yaml",
                aa_forecast=config,
                search_space_contract=None,
            ).config
        )
        return {
            "jobs": _cfg.aa_forecast_jobs_payload(linked),
            "training_search": _cfg.aa_forecast_training_search_payload(linked),
        }

    def load_stage(
        self,
        repo_root: Path,
        *,
        source_path: Path,
        source_type: str,
        config: Any,
        search_space_contract: SearchSpaceContract | None,
    ) -> _cfg.AAForecastStageLoadedConfig:
        return _cfg.load_aa_forecast_stage1(
            repo_root,
            source_path=source_path,
            source_type=source_type,
            aa_forecast=config,
            search_space_contract=search_space_contract,
        )

    def apply_stage_to_config(self, config: AppConfig, stage_loaded: Any) -> AppConfig:
        return replace(config, stage_plugin_config=stage_loaded.config)

    def stage_normalized_payload(
        self,
        config: AppConfig,
        stage_loaded: Any,
    ) -> dict[str, Any]:
        aa_cfg = config.stage_plugin_config
        tuning = _cfg.aa_forecast_plugin_tuning_public_dict(aa_cfg)
        payload = {
            "aa_forecast": {
                "stage1": {
                    "source_path": str(stage_loaded.source_path),
                    "source_type": stage_loaded.source_type,
                    "config_input_sha256": stage_loaded.input_hash,
                    "config_resolved_sha256": stage_loaded.resolved_hash,
                    "search_space_path": (
                        str(stage_loaded.search_space_path)
                        if stage_loaded.search_space_path is not None
                        else None
                    ),
                    "search_space_sha256": stage_loaded.search_space_hash,
                    **tuning,
                }
            }
        }
        return payload

    def supported_models(self) -> set[str]:
        return {"AAForecast"}

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
        if model_name != "AAForecast":
            raise ValueError("aa_forecast stage supports only the AAForecast model")

    def owns_top_level_job(self, model_name: str) -> bool:
        return model_name == "AAForecast"

    def capabilities_for(self, model_name: str) -> dict[str, Any]:
        if model_name != "AAForecast":
            raise ValueError(
                f"aa_forecast does not own top-level job execution for model {model_name!r}"
            )
        return {
            "name": model_name,
            "multivariate": False,
            "supports_hist_exog": True,
            "supports_futr_exog": False,
            "supports_stat_exog": False,
            "requires_n_series": False,
            "single_device_only": True,
        }

    def model_search_space_fallback_key(self) -> str | None:
        return None

    def training_search_space_fallback_key(self) -> str | None:
        return None

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
        return loaded, train_df, future_df, "aa_forecast"

    def predict_fold(
        self,
        loaded: LoadedConfig,
        job: Any,
        *,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        run_root: Path | None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, Any | None]:
        from .runtime import predict_aa_forecast_fold

        return predict_aa_forecast_fold(
            loaded,
            job,
            train_df=train_df,
            future_df=future_df,
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
        from .runtime import materialize_aa_forecast_stage

        materialize_aa_forecast_stage(
            loaded=loaded,
            selected_jobs=selected_jobs,
            run_root=run_root,
            main_resolved_path=main_resolved_path,
            main_capability_path=main_capability_path,
            main_manifest_path=main_manifest_path,
            entrypoint_version=entrypoint_version,
            validate_only=validate_only,
        )

    def manifest_block(self, loaded: LoadedConfig) -> dict[str, Any]:
        aa_cfg = loaded.config.stage_plugin_config
        selected = _cfg.aa_forecast_resolved_selected_path(
            aa_cfg, loaded.stage_plugin_loaded
        )
        return _cfg.aa_forecast_plugin_state_dict(
            aa_cfg, selected_config_path=selected
        )

    def validation_payload(
        self,
        loaded: LoadedConfig,
        selected_jobs: Any,
        caps_by_model: dict[str, Any],
    ) -> dict[str, Any]:
        aa_cfg = loaded.config.stage_plugin_config
        selected = _cfg.aa_forecast_resolved_selected_path(
            aa_cfg, loaded.stage_plugin_loaded
        )
        base = _cfg.aa_forecast_plugin_state_dict(
            aa_cfg, selected_config_path=selected
        )
        return {
            **base,
            "handled_jobs": [
                {
                    "model": job.model,
                    "requested_mode": job.requested_mode,
                    "validated_mode": job.validated_mode,
                    "supports_hist_exog": bool(caps_by_model[job.model].supports_hist_exog),
                }
                for job in selected_jobs
            ],
        }

    def fanout_config_keys(self) -> set[str]:
        return {"aa_forecast"}

    def fanout_filter_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("aa_forecast fanout_filter_payload requires a dict payload")
        return {
            key: value
            for key, value in payload.items()
            if key in _cfg.AA_FORECAST_MAIN_KEYS
        }

    def fanout_stage_payload(self, loaded: LoadedConfig) -> dict[str, Any] | None:
        raw = loaded.normalized_payload.get("aa_forecast")
        if not isinstance(raw, dict):
            return None
        if not raw.get("enabled"):
            return None
        stage1 = raw.get("stage1")
        if stage1 is None or not isinstance(stage1, dict):
            raise ValueError(
                "aa_forecast is enabled but normalized_payload is missing aa_forecast.stage1; "
                "cannot fan out jobs without stage metadata"
            )
        return {"aa_forecast": {"stage1": stage1}}


_plugin = AAForecastStagePlugin()

from plugin_contracts.stage_registry import register_stage_plugin  # noqa: E402

register_stage_plugin(_plugin)
