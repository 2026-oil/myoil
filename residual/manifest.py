from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .config import (
    DEFAULT_ARTIFACT_SCHEMA_VERSION,
    DEFAULT_EVALUATION_PROTOCOL_VERSION,
    DEFAULT_MANIFEST_VERSION,
    LoadedConfig,
)


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
        'entrypoint_version': entrypoint_version,
        'compat_mode': compat_mode,
        'jobs': [job.name for job in loaded.config.jobs],
        'training': {'loss': loaded.config.training.loss},
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
