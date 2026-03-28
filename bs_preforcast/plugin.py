"""StagePlugin implementation for bs_preforcast.

Registers itself with :mod:`residual.stage_registry` on import so that
``residual/`` never needs a direct reference to ``bs_preforcast``.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from . import config as _cfg
from . import search_space as _ss

if TYPE_CHECKING:
    from residual.config import AppConfig, LoadedConfig
    from residual.optuna_spaces import SearchSpaceContract


class BsPreforcastStagePlugin:
    """Concrete :class:`~residual.stage_plugin.StagePlugin` for bs_preforcast."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def config_key(self) -> str:
        return "bs_preforcast"

    # ------------------------------------------------------------------
    # Config lifecycle
    # ------------------------------------------------------------------

    def default_config(self) -> _cfg.BsPreforcastConfig:
        return _cfg.BsPreforcastConfig()

    def normalize_config(
        self,
        payload: Any,
        *,
        unknown_keys: Any,
        coerce_bool: Any,
        coerce_optional_path_string: Any,
    ) -> _cfg.BsPreforcastConfig:
        return _cfg.normalize_bs_preforcast_config(
            payload,
            unknown_keys=unknown_keys,
            coerce_bool=coerce_bool,
            coerce_optional_path_string=coerce_optional_path_string,
        )

    def is_enabled(self, config: Any) -> bool:
        return bool(getattr(config, "enabled", False))

    def config_to_dict(self, config: Any) -> dict[str, Any] | None:
        payload = asdict(config)
        if not payload.get("enabled", False):
            return None
        payload["target_columns"] = list(payload["target_columns"])
        return payload

    # ------------------------------------------------------------------
    # Route validation & stage loading
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
        if not self.is_enabled(config):
            return None
        from residual.config import (
            _resolve_jobs_reference,
            _resolve_relative_config_reference,
        )

        selected_config_path = config.config_path
        if selected_config_path is None:
            raise ValueError("bs_preforcast enabled but config_path was not resolved")
        stage_source_path = _resolve_relative_config_reference(
            repo_root, source_path, selected_config_path
        )
        if not stage_source_path.exists():
            raise FileNotFoundError(
                f"bs_preforcast selected route does not exist: {stage_source_path}"
            )
        stage_source_type = (
            "toml" if stage_source_path.suffix.lower() == ".toml" else "yaml"
        )
        stage_raw_payload = load_document(stage_source_path, stage_source_type)
        if stage_raw_payload.get("bs_preforcast") in (None, {}):
            raise ValueError(
                "bs_preforcast routed YAML must define a top-level bs_preforcast block"
            )
        _cfg.normalize_linked_bs_preforcast_config(
            stage_raw_payload.get("bs_preforcast"),
            unknown_keys=unknown_keys,
            coerce_bool=coerce_bool,
            coerce_name_tuple=coerce_name_tuple,
        )
        return {
            "jobs": _resolve_jobs_reference(
                repo_root,
                source_path=stage_source_path,
                jobs_value=dict(stage_raw_payload).get("jobs", []),
            ),
            "residual": {},
        }

    def load_stage(
        self,
        repo_root: Path,
        *,
        source_path: Path,
        source_type: str,
        config: Any,
        search_space_contract: SearchSpaceContract | None,
    ) -> _cfg.BsPreforcastStageLoadedConfig:
        return _cfg.load_bs_preforcast_stage1(
            repo_root,
            source_path=source_path,
            source_type=source_type,
            bs_preforcast=config,
            search_space_contract=search_space_contract,
        )

    def apply_stage_to_config(self, config: AppConfig, stage_loaded: Any) -> AppConfig:
        return replace(
            config,
            stage_plugin_config=stage_loaded.config.stage_plugin_config,
        )

    def stage_normalized_payload(
        self,
        config: AppConfig,
        stage_loaded: Any,
    ) -> dict[str, Any]:
        bs_cfg = config.stage_plugin_config
        return {
            "bs_preforcast": {
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
                    "target_columns": list(bs_cfg.target_columns),
                    "multivariable": bs_cfg.task.multivariable,
                },
            },
        }

    # ------------------------------------------------------------------
    # Search-space integration
    # ------------------------------------------------------------------

    def supported_models(self) -> set[str]:
        return _ss.SUPPORTED_BS_PREFORCAST_MODELS

    def stage_only_param_registry(self) -> dict[str, dict[str, dict[str, Any]]]:
        return _ss.BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY

    def normalize_search_space_sections(
        self,
        payload: dict[str, Any],
        *,
        normalize_model_section: Any,
        normalize_training_section: Any,
    ) -> dict[str, Any]:
        models, training = _ss.normalize_bs_preforcast_sections(
            payload,
            normalize_model_section=normalize_model_section,
            normalize_training_section=normalize_training_section,
        )
        return {
            "bs_preforcast_models": models,
            "bs_preforcast_training": training,
        }

    def model_search_space_key(self) -> str:
        return "bs_preforcast_models"

    def training_search_space_key(self) -> str:
        return "bs_preforcast_training"

    def validate_model(self, model_name: str) -> None:
        if model_name == "AutoARIMA":
            raise ValueError(
                "bs_preforcast stage no longer supports AutoARIMA; use ARIMA instead"
            )

    def model_search_space_fallback_key(self) -> str | None:
        return "models"

    def training_search_space_fallback_key(self) -> str | None:
        return "training"

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
    ) -> tuple[LoadedConfig, pd.DataFrame, pd.DataFrame, Any]:
        from .runtime import prepare_bs_preforcast_fold_inputs

        return prepare_bs_preforcast_fold_inputs(
            loaded, job, train_df, future_df, run_root=run_root
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
        from .runtime import materialize_bs_preforcast_stage

        materialize_bs_preforcast_stage(
            loaded=loaded,
            selected_jobs=selected_jobs,
            run_root=run_root,
            main_resolved_path=main_resolved_path,
            main_capability_path=main_capability_path,
            main_manifest_path=main_manifest_path,
            entrypoint_version=entrypoint_version,
            validate_only=validate_only,
        )

    # ------------------------------------------------------------------
    # Manifest / validation payload
    # ------------------------------------------------------------------

    def manifest_block(self, loaded: LoadedConfig) -> dict[str, Any]:
        bs_cfg = loaded.config.stage_plugin_config
        stage1 = loaded.stage_plugin_loaded
        return {
            "enabled": bs_cfg.enabled,
            "config_path": bs_cfg.config_path,
            "target_columns": list(bs_cfg.target_columns),
            "multivariable": bs_cfg.task.multivariable,
            "selected_config_path": (
                loaded.normalized_payload.get("bs_preforcast", {}).get(
                    "selected_config_path", bs_cfg.config_path
                )
            ),
            "stage1": (
                {
                    "source_path": str(stage1.source_path),
                    "source_type": stage1.source_type,
                    "config_input_sha256": stage1.input_hash,
                    "config_resolved_sha256": stage1.resolved_hash,
                    "search_space_path": (
                        str(stage1.search_space_path)
                        if stage1.search_space_path is not None
                        else None
                    ),
                    "search_space_sha256": stage1.search_space_hash,
                }
                if stage1 is not None
                else None
            ),
        }

    def validation_payload(
        self,
        loaded: LoadedConfig,
        selected_jobs: Any,
        caps_by_model: dict[str, Any],
    ) -> dict[str, Any]:
        bs_cfg = loaded.config.stage_plugin_config
        stage1 = loaded.stage_plugin_loaded
        selected_bs_config_path = (
            str(stage1.source_path) if stage1 is not None else bs_cfg.config_path
        )
        return {
            "enabled": bs_cfg.enabled,
            "config_path": bs_cfg.config_path,
            "target_columns": list(bs_cfg.target_columns),
            "multivariable": bs_cfg.task.multivariable,
            "selected_config_path": selected_bs_config_path,
            "job_injection_results": [
                {
                    "model": job.model,
                    "injection_mode": (
                        "futr_exog"
                        if caps_by_model[job.model].supports_futr_exog
                        else "lag_derived"
                    ),
                    "supports_futr_exog": bool(
                        caps_by_model[job.model].supports_futr_exog
                    ),
                }
                for job in selected_jobs
            ],
        }

    # ------------------------------------------------------------------
    # Fanout helpers
    # ------------------------------------------------------------------

    def fanout_config_keys(self) -> set[str]:
        return {"bs_preforcast"}

    def fanout_filter_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, dict):
            return {
                key: value
                for key, value in payload.items()
                if key in {"enabled", "config_path"}
            }
        return payload

    def fanout_stage_payload(self, loaded: LoadedConfig) -> dict[str, Any] | None:
        if loaded.stage_plugin_loaded is None:
            return None
        return {
            "bs_preforcast": {
                "stage1": loaded.normalized_payload.get("bs_preforcast", {}).get(
                    "stage1"
                ),
            },
        }


_plugin = BsPreforcastStagePlugin()

from residual.stage_registry import register_stage_plugin  # noqa: E402

register_stage_plugin(_plugin)
