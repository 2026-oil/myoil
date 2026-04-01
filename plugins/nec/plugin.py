"""StagePlugin implementation for NEC."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import config as _cfg
from .runtime import materialize_nec_stage, predict_nec_fold

if TYPE_CHECKING:
    from app_config import AppConfig, LoadedConfig
    from tuning.search_space import SearchSpaceContract


class NecStagePlugin:
    @property
    def config_key(self) -> str:
        return "nec"

    def default_config(self) -> _cfg.NecConfig:
        return _cfg.NecConfig()

    def normalize_config(self, payload: Any, *, unknown_keys: Any, coerce_bool: Any, coerce_optional_path_string: Any) -> _cfg.NecConfig:
        return _cfg.normalize_nec_config(
            payload,
            unknown_keys=unknown_keys,
            coerce_bool=coerce_bool,
            coerce_optional_path_string=coerce_optional_path_string,
        )

    def is_enabled(self, config: Any) -> bool:
        return bool(getattr(config, "enabled", False))

    def config_to_dict(self, config: Any) -> dict[str, Any] | None:
        if not self.is_enabled(config):
            return None
        return _cfg.nec_to_dict(config)

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
        if not self.is_enabled(config):
            return None
        from app_config import _resolve_relative_config_reference

        selected_config_path = config.config_path
        if selected_config_path is None:
            raise ValueError("nec enabled but config_path was not resolved")
        stage_source_path = _resolve_relative_config_reference(repo_root, source_path, selected_config_path)
        if not stage_source_path.exists():
            raise FileNotFoundError(f"nec selected route does not exist: {stage_source_path}")
        stage_source_type = "toml" if stage_source_path.suffix.lower() == ".toml" else "yaml"
        stage_payload = load_document(stage_source_path, stage_source_type)
        if not isinstance(stage_payload, dict):
            raise ValueError("nec config_path must resolve to a mapping with a top-level nec block")
        _cfg.normalize_linked_nec_config(
            stage_payload.get("nec"),
            unknown_keys=unknown_keys,
            coerce_bool=coerce_bool,
            coerce_name_tuple=coerce_name_tuple,
        )
        return {"jobs": [], "residual": {}}

    def load_stage(self, repo_root: Path, *, source_path: Path, source_type: str, config: Any, search_space_contract: SearchSpaceContract | None) -> _cfg.NecStageLoadedConfig:
        return _cfg.load_nec_stage1(
            repo_root,
            source_path=source_path,
            source_type=source_type,
            nec=config,
            search_space_contract=search_space_contract,
        )

    def apply_stage_to_config(self, config: AppConfig, stage_loaded: Any) -> AppConfig:
        return replace(config, stage_plugin_config=stage_loaded.config)

    def stage_normalized_payload(self, config: AppConfig, stage_loaded: Any) -> dict[str, Any]:
        nec_cfg = config.stage_plugin_config
        return {
            "nec": {
                "stage1": {
                    "source_path": str(stage_loaded.source_path),
                    "source_type": stage_loaded.source_type,
                    "config_input_sha256": stage_loaded.input_hash,
                    "config_resolved_sha256": stage_loaded.resolved_hash,
                    "history_steps": nec_cfg.history_steps,
                    "hist_columns": list(nec_cfg.hist_columns),
                    "preprocessing_mode": nec_cfg.preprocessing.mode,
                    "probability_feature": nec_cfg.preprocessing.probability_feature,
                    "gmm_components": nec_cfg.preprocessing.gmm_components,
                    "epsilon": nec_cfg.preprocessing.epsilon,
                    "shared_training_scaler": config.training.scaler_type,
                    "shared_scaler_override_active": config.training.scaler_type not in (None, "identity"),
                    "paper_overrides_shared_scaler": True,
                }
            }
        }

    def supported_models(self) -> set[str]:
        return {"NEC"}

    def stage_only_param_registry(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {}

    def normalize_search_space_sections(self, payload: dict[str, Any], *, normalize_model_section: Any, normalize_training_section: Any) -> dict[str, Any]:
        del payload, normalize_model_section, normalize_training_section
        return {"nec_models": {}, "nec_training": []}

    def model_search_space_key(self) -> str:
        return "nec_models"

    def training_search_space_key(self) -> str:
        return "nec_training"

    def validate_model(self, model_name: str) -> None:
        if model_name != "NEC":
            raise ValueError("nec stage supports only the NEC top-level model")

    def model_search_space_fallback_key(self) -> str | None:
        return None

    def training_search_space_fallback_key(self) -> str | None:
        return None

    def prepare_fold_inputs(self, loaded: LoadedConfig, job: Any, train_df: pd.DataFrame, future_df: pd.DataFrame, *, run_root: Path | None) -> tuple[LoadedConfig, pd.DataFrame, pd.DataFrame, str]:
        del job, run_root
        return loaded, train_df, future_df, "nec"

    def materialize_stage(self, loaded: LoadedConfig, selected_jobs: Any, *, run_root: Path, main_resolved_path: Path, main_capability_path: Path, main_manifest_path: Path, entrypoint_version: str, validate_only: bool) -> None:
        materialize_nec_stage(
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
        nec_cfg = loaded.config.stage_plugin_config
        stage1 = loaded.stage_plugin_loaded
        return {
            "enabled": nec_cfg.enabled,
            "config_path": nec_cfg.config_path,
            "history_steps": nec_cfg.history_steps,
            "hist_columns": list(nec_cfg.hist_columns),
            "preprocessing_mode": nec_cfg.preprocessing.mode,
            "probability_feature": nec_cfg.preprocessing.probability_feature,
            "gmm_components": nec_cfg.preprocessing.gmm_components,
            "epsilon": nec_cfg.preprocessing.epsilon,
            "selected_config_path": str(stage1.source_path) if stage1 is not None else nec_cfg.config_path,
        }

    def validation_payload(self, loaded: LoadedConfig, selected_jobs: Any, caps_by_model: dict[str, Any]) -> dict[str, Any]:
        nec_cfg = loaded.config.stage_plugin_config
        stage1 = loaded.stage_plugin_loaded
        return {
            "enabled": nec_cfg.enabled,
            "config_path": nec_cfg.config_path,
            "selected_config_path": str(stage1.source_path) if stage1 is not None else nec_cfg.config_path,
            "history_steps": nec_cfg.history_steps,
            "hist_columns": list(nec_cfg.hist_columns),
            "preprocessing_mode": nec_cfg.preprocessing.mode,
            "probability_feature": nec_cfg.preprocessing.probability_feature,
            "gmm_components": nec_cfg.preprocessing.gmm_components,
            "epsilon": nec_cfg.preprocessing.epsilon,
            "shared_training_scaler": loaded.config.training.scaler_type,
            "handled_jobs": [
                {
                    "model": job.model,
                    "requested_mode": job.requested_mode,
                    "validated_mode": job.validated_mode,
                    "plugin_owned": self.owns_top_level_job(job.model),
                    "supports_hist_exog": bool(caps_by_model[job.model].supports_hist_exog),
                }
                for job in selected_jobs
            ],
        }

    def fanout_config_keys(self) -> set[str]:
        return {"nec"}

    def fanout_filter_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, dict):
            return {key: value for key, value in payload.items() if key in {"enabled", "config_path"}}
        return payload

    def fanout_stage_payload(self, loaded: LoadedConfig) -> dict[str, Any] | None:
        if loaded.stage_plugin_loaded is None:
            return None
        return {"nec": {"stage1": loaded.normalized_payload.get("nec", {}).get("stage1")}}

    def owns_top_level_job(self, model_name: str) -> bool:
        return model_name == "NEC"

    def handles_job_model(self, model_name: str) -> bool:
        return self.owns_top_level_job(model_name)

    def job_capabilities(self, model_name: str) -> dict[str, Any]:
        if not self.owns_top_level_job(model_name):
            raise ValueError(f"Unsupported NEC plugin model: {model_name}")
        return {
            "name": model_name,
            "multivariate": False,
            "supports_hist_exog": False,
            "supports_futr_exog": False,
            "supports_stat_exog": False,
            "requires_n_series": False,
            "single_device_only": True,
        }

    def capabilities_for(self, model_name: str) -> dict[str, Any]:
        return self.job_capabilities(model_name)

    def predict_fold(self, loaded: LoadedConfig, job: Any, train_df: pd.DataFrame, future_df: pd.DataFrame, *, run_root: Path | None, params_override: dict[str, Any] | None = None, training_override: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.Series, Any, pd.DataFrame, Any | None]:
        del params_override, training_override
        return predict_nec_fold(loaded, job, train_df, future_df, run_root=run_root)


_plugin = NecStagePlugin()

from plugin_contracts.stage_registry import register_stage_plugin  # noqa: E402

register_stage_plugin(_plugin)
