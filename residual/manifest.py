from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json

from .config import (
    DEFAULT_ARTIFACT_SCHEMA_VERSION,
    DEFAULT_EVALUATION_PROTOCOL_VERSION,
    DEFAULT_MANIFEST_VERSION,
    LoadedConfig,
)
from .features import hist_exog_lag_feature_name


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def residual_feature_policy_payload(feature_config: Any) -> dict[str, Any]:
    return _json_ready(_coerce_mapping(feature_config))


def residual_active_feature_columns(feature_config: Any) -> list[str]:
    payload = residual_feature_policy_payload(feature_config)
    lag_payload = _coerce_mapping(payload.get("lag_features"))
    exog_payload = _coerce_mapping(payload.get("exog_sources"))

    columns: list[str] = []
    if payload.get("include_horizon_step", True):
        columns.append("horizon_step")
    if payload.get("include_base_prediction", True):
        columns.append("y_hat_base")
    if payload.get("include_date_features", False):
        columns.extend(["cutoff_day", "ds_day"])

    lag_sources = [str(item) for item in lag_payload.get("sources", [])]
    lag_steps = [int(item) for item in lag_payload.get("steps", [])]
    for source in lag_sources:
        for step in lag_steps:
            columns.append(f"{source}_lag_{step}")

    for column in exog_payload.get("hist", []):
        name = hist_exog_lag_feature_name(str(column))
        if name not in columns:
            columns.append(name)
    for group in ("futr", "static"):
        for column in exog_payload.get(group, []):
            name = str(column)
            if name not in columns:
                columns.append(name)
    return columns


def build_manifest(
    loaded: LoadedConfig,
    *,
    compat_mode: str,
    entrypoint_version: str,
    resolved_config_path: Path,
) -> dict[str, Any]:
    return {
        'manifest_version': DEFAULT_MANIFEST_VERSION,
        'artifact_schema_version': DEFAULT_ARTIFACT_SCHEMA_VERSION,
        'evaluation_protocol_version': DEFAULT_EVALUATION_PROTOCOL_VERSION,
        'config_source_type': loaded.source_type,
        'config_source_path': str(loaded.source_path),
        'config_resolved_path': str(resolved_config_path),
        'config_input_sha256': loaded.input_hash,
        'config_resolved_sha256': loaded.resolved_hash,
        'search_space_path': str(loaded.search_space_path)
        if loaded.search_space_path
        else None,
        'search_space_sha256': loaded.search_space_hash,
        'entrypoint_version': entrypoint_version,
        'compat_mode': compat_mode,
        'jobs': [
            {
                'model': job.model,
                'requested_mode': job.requested_mode,
                'validated_mode': job.validated_mode,
                'selected_search_params': list(job.selected_search_params),
            }
            for job in loaded.config.jobs
        ],
        'training_search': {
            'requested_mode': loaded.config.training_search.requested_mode,
            'validated_mode': loaded.config.training_search.validated_mode,
            'selected_search_params': list(
                loaded.config.training_search.selected_search_params
            ),
        },
        'residual': {
            'model': loaded.config.residual.model,
            'target': loaded.config.residual.target,
            'requested_mode': loaded.config.residual.requested_mode,
            'validated_mode': loaded.config.residual.validated_mode,
            'selected_search_params': list(loaded.config.residual.selected_search_params),
            'feature_policy': residual_feature_policy_payload(
                loaded.config.residual.features
            ),
            'active_feature_columns': residual_active_feature_columns(
                loaded.config.residual.features
            ),
        },
        'training': {'loss': loaded.config.training.loss},
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
