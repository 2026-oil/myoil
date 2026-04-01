"""StagePlugin implementation for NEC."""
from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import config as _cfg
from .runtime import materialize_nec_stage, predict_nec_fold

if TYPE_CHECKING:
    from app_config import AppConfig, LoadedConfig
    from tuning.search_space import SearchSpaceContract

_PROTECTED_BRANCH_MODEL_PARAMS = {
    "h",
    "input_size",
    "inference_input_size",
    "hist_exog_list",
    "futr_exog_list",
    "stat_exog_list",
    "loss",
    "valid_loss",
    "scaler_type",
    "random_seed",
    "alias",
    "optimizer",
    "optimizer_kwargs",
    "dataloader_kwargs",
    "n_series",
    "enable_checkpointing",
    "enable_progress_bar",
    "logger",
}


def _catalog_and_capabilities(model_name: str) -> tuple[set[str], Any]:
    import neuralforecast.models as nf_models
    from runtime_support.forecast_models import MODEL_CLASSES, capabilities_for

    exported = {name for name in getattr(nf_models, "__all__", []) if name in MODEL_CLASSES}
    if model_name not in exported:
        if model_name in getattr(nf_models, "__all__", []):
            raise ValueError(
                f"nec branch model {model_name!r} is exported by neuralforecast.models but is not buildable through the shared runtime catalog"
            )
        raise ValueError(f"nec branch model {model_name!r} is not exported by neuralforecast.models")
    return exported, capabilities_for(model_name)


def _validate_branch_model_params(branch_name: str, branch_cfg: _cfg.NecBranchConfig) -> None:
    from runtime_support.forecast_models import MODEL_CLASSES

    model_cls = MODEL_CLASSES[branch_cfg.model]
    signature = inspect.signature(model_cls.__init__)
    explicit_params = {name for name in signature.parameters if name != "self"}
    protected = sorted(set(branch_cfg.model_params) & _PROTECTED_BRANCH_MODEL_PARAMS)
    if protected:
        raise ValueError(
            f"nec.{branch_name}.model_params cannot override NEC-controlled parameter(s): {', '.join(protected)}"
        )
    unknown = sorted(set(branch_cfg.model_params) - explicit_params)
    if unknown:
        raise ValueError(
            f"nec.{branch_name}.model_params contains unsupported parameter(s) for {branch_cfg.model}: {', '.join(unknown)}"
        )


def _validate_branch_contract(branch_name: str, branch_cfg: _cfg.NecBranchConfig) -> dict[str, Any]:
    _exported, caps = _catalog_and_capabilities(branch_cfg.model)
    if caps.multivariate:
        raise ValueError(f"nec.{branch_name}.model={branch_cfg.model} is multivariate and not supported by NEC branches")
    needs_hist_exog = bool(branch_cfg.variables) or branch_name in {"classifier", "extreme"}
    if needs_hist_exog and not caps.supports_hist_exog:
        raise ValueError(
            f"nec.{branch_name}.model={branch_cfg.model} is incompatible: branch requires hist exog inputs but the model does not support hist exog"
        )
    if branch_name == "classifier" and branch_cfg.model in {"NBEATS", "NLinear", "DLinear", "TimesNet"}:
        # Models without hist exog support already fail above; keep this branch-specific guard explicit for readability.
        raise ValueError(f"nec.classifier.model={branch_cfg.model} is incompatible with the classifier adapter contract")
    _validate_branch_model_params(branch_name, branch_cfg)
    return {
        "model": branch_cfg.model,
        "variables": list(branch_cfg.variables),
        "compatible": True,
        "multivariate": bool(caps.multivariate),
        "supports_hist_exog": bool(caps.supports_hist_exog),
        "supports_futr_exog": bool(caps.supports_futr_exog),
        "supports_stat_exog": bool(caps.supports_stat_exog),
        "classifier_adapter": branch_name == "classifier",
    }


def _branch_compatibility_payload(config: AppConfig, nec_cfg: _cfg.NecConfig) -> dict[str, Any]:
    del config
    payload: dict[str, Any] = {}
    for branch_name, branch_cfg in _cfg.nec_branch_configs(nec_cfg).items():
        payload[branch_name] = _validate_branch_contract(branch_name, branch_cfg)
    return payload


def _validate_branch_variables(config: AppConfig, nec_cfg: _cfg.NecConfig) -> None:
    header = pd.read_csv(config.dataset.path, nrows=0)
    dataset_columns = set(header.columns)
    target_col = config.dataset.target_col
    for branch_name, branch_cfg in _cfg.nec_branch_configs(nec_cfg).items():
        if target_col in branch_cfg.variables:
            raise ValueError(f"nec.{branch_name}.variables must not include dataset.target_col {target_col!r}")
        missing = [column for column in branch_cfg.variables if column not in dataset_columns]
        if missing:
            raise ValueError(
                f"nec.{branch_name}.variables reference missing dataset column(s): {', '.join(missing)}"
            )


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
        normalized = _cfg.normalize_linked_nec_config(
            stage_payload.get("nec"),
            unknown_keys=unknown_keys,
            coerce_bool=coerce_bool,
            coerce_name_tuple=coerce_name_tuple,
        )
        for branch_name, branch_cfg in _cfg.nec_branch_configs(normalized).items():
            _validate_branch_contract(branch_name, branch_cfg)
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
        nec_cfg = stage_loaded.config
        _validate_branch_variables(config, nec_cfg)
        _branch_compatibility_payload(config, nec_cfg)
        return replace(config, stage_plugin_config=nec_cfg)

    def stage_normalized_payload(self, config: AppConfig, stage_loaded: Any) -> dict[str, Any]:
        nec_cfg = config.stage_plugin_config
        compatibility = _branch_compatibility_payload(config, nec_cfg)
        return {
            "nec": {
                "stage1": {
                    "source_path": str(stage_loaded.source_path),
                    "source_type": stage_loaded.source_type,
                    "config_input_sha256": stage_loaded.input_hash,
                    "config_resolved_sha256": stage_loaded.resolved_hash,
                    "branches": {
                        name: {
                            "model": branch.model,
                            "model_params": dict(branch.model_params),
                            "variables": list(branch.variables),
                            **compatibility[name],
                        }
                        for name, branch in _cfg.nec_branch_configs(nec_cfg).items()
                    },
                    "active_hist_columns": list(_cfg.nec_active_hist_columns(nec_cfg)),
                    "history_steps_source": "training.input_size",
                    "history_steps_value": config.training.input_size,
                    "preprocessing_mode": nec_cfg.preprocessing.mode,
                    "inference_mode": nec_cfg.inference.mode,
                    "inference_threshold": nec_cfg.inference.threshold,
                    "probability_feature_forced": True,
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
            "selected_config_path": str(stage1.source_path) if stage1 is not None else nec_cfg.config_path,
            "branches": {
                name: {
                    "model": branch.model,
                    "model_params": dict(branch.model_params),
                    "variables": list(branch.variables),
                }
                for name, branch in _cfg.nec_branch_configs(nec_cfg).items()
            },
            "active_hist_columns": list(_cfg.nec_active_hist_columns(nec_cfg)),
            "history_steps_source": "training.input_size",
            "history_steps_value": loaded.config.training.input_size,
            "preprocessing_mode": nec_cfg.preprocessing.mode,
            "inference_mode": nec_cfg.inference.mode,
            "inference_threshold": nec_cfg.inference.threshold,
            "probability_feature_forced": True,
            "gmm_components": nec_cfg.preprocessing.gmm_components,
            "epsilon": nec_cfg.preprocessing.epsilon,
        }

    def validation_payload(self, loaded: LoadedConfig, selected_jobs: Any, caps_by_model: dict[str, Any]) -> dict[str, Any]:
        nec_cfg = loaded.config.stage_plugin_config
        stage1 = loaded.stage_plugin_loaded
        compatibility = _branch_compatibility_payload(loaded.config, nec_cfg)
        return {
            "enabled": nec_cfg.enabled,
            "config_path": nec_cfg.config_path,
            "selected_config_path": str(stage1.source_path) if stage1 is not None else nec_cfg.config_path,
            "branches": {
                name: {
                    "model": branch.model,
                    "model_params": dict(branch.model_params),
                    "variables": list(branch.variables),
                    **compatibility[name],
                }
                for name, branch in _cfg.nec_branch_configs(nec_cfg).items()
            },
            "active_hist_columns": list(_cfg.nec_active_hist_columns(nec_cfg)),
            "history_steps_source": "training.input_size",
            "history_steps_value": loaded.config.training.input_size,
            "preprocessing_mode": nec_cfg.preprocessing.mode,
            "inference_mode": nec_cfg.inference.mode,
            "inference_threshold": nec_cfg.inference.threshold,
            "probability_feature_forced": True,
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
