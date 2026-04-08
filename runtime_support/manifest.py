from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from app_config import (
    DEFAULT_ARTIFACT_SCHEMA_VERSION,
    DEFAULT_EVALUATION_PROTOCOL_VERSION,
    DEFAULT_MANIFEST_VERSION,
    LoadedConfig,
)
from plugin_contracts.stage_registry import get_active_stage_plugin


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


def _stage_plugin_manifest_block(loaded: LoadedConfig) -> dict[str, Any]:
    result = get_active_stage_plugin(loaded.config)
    if result is None:
        return {}
    plugin, _ = result
    return {plugin.config_key: plugin.manifest_block(loaded)}


def build_manifest(
    loaded: LoadedConfig,
    *,
    compat_mode: str,
    entrypoint_version: str,
    resolved_config_path: Path,
    optuna_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "manifest_version": DEFAULT_MANIFEST_VERSION,
        "artifact_schema_version": DEFAULT_ARTIFACT_SCHEMA_VERSION,
        "evaluation_protocol_version": DEFAULT_EVALUATION_PROTOCOL_VERSION,
        "config_source_type": loaded.source_type,
        "config_source_path": str(loaded.source_path),
        "config_resolved_path": str(resolved_config_path),
        "config_input_sha256": loaded.input_hash,
        "config_resolved_sha256": loaded.resolved_hash,
        "search_space_path": str(loaded.search_space_path)
        if loaded.search_space_path
        else None,
        "search_space_sha256": loaded.search_space_hash,
        "shared_settings_path": (
            str(loaded.shared_settings_path)
            if loaded.shared_settings_path is not None
            else None
        ),
        "shared_settings_sha256": loaded.shared_settings_hash,
        "entrypoint_version": entrypoint_version,
        "compat_mode": compat_mode,
        "jobs": [
            {
                "model": job.model,
                "requested_mode": job.requested_mode,
                "validated_mode": job.validated_mode,
                "selected_search_params": list(job.selected_search_params),
            }
            for job in loaded.config.jobs
        ],
        "training_search": {
            "requested_mode": loaded.config.training_search.requested_mode,
            "validated_mode": loaded.config.training_search.validated_mode,
            "selected_search_params": list(
                loaded.config.training_search.selected_search_params
            ),
        },
        **_stage_plugin_manifest_block(loaded),
        "training": {"loss": loaded.config.training.loss},
    }
    if optuna_payload is not None:
        manifest["optuna"] = _json_ready(dict(optuna_payload))
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
