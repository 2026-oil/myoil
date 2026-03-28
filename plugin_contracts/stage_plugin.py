"""Generic stage-plugin protocol for pre-main-stage execution pipelines.

residual/ has zero knowledge of any concrete stage plugin (e.g. bs_preforcast).
All stage-specific behaviour is provided through this protocol and dispatched
via :mod:`residual.stage_registry`.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from app_config import AppConfig, LoadedConfig
    from tuning.search_space import SearchSpaceContract


@runtime_checkable
class StagePlugin(Protocol):
    """Interface every stage plugin must satisfy."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def config_key(self) -> str:
        """Top-level YAML key that activates this plugin (e.g. ``"bs_preforcast"``)."""
        ...

    # ------------------------------------------------------------------
    # Config lifecycle
    # ------------------------------------------------------------------

    def default_config(self) -> Any:
        """Return a disabled/empty default config dataclass."""
        ...

    def normalize_config(
        self,
        payload: Any,
        *,
        unknown_keys: Any,
        coerce_bool: Any,
        coerce_optional_path_string: Any,
    ) -> Any:
        """Normalize the raw YAML mapping into a typed config dataclass."""
        ...

    def is_enabled(self, config: Any) -> bool:
        """Return ``True`` if the stage is active for the given config."""
        ...

    def config_to_dict(self, config: Any) -> dict[str, Any] | None:
        """Serialize config for the normalized payload.

        Return ``None`` to omit the section entirely.
        """
        ...

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
        """Probe the routed config early (before full normalization).

        Return a small ``stage_payload_probe`` dict that the caller can use
        to decide whether a search-space is needed, or ``None`` if the
        stage is disabled.
        """
        ...

    def load_stage(
        self,
        repo_root: Path,
        *,
        source_path: Path,
        source_type: str,
        config: Any,
        search_space_contract: SearchSpaceContract | None,
    ) -> Any:
        """Fully load and validate the stage-1 config.

        Return an opaque loaded-stage object stored on ``LoadedConfig``.
        """
        ...

    def apply_stage_to_config(self, config: AppConfig, stage_loaded: Any) -> AppConfig:
        """Return an updated ``AppConfig`` after stage-1 has been loaded."""
        ...

    def stage_normalized_payload(
        self,
        config: AppConfig,
        stage_loaded: Any,
    ) -> dict[str, Any]:
        """Return extra keys to merge into ``normalized_payload`` after stage loading."""
        ...

    # ------------------------------------------------------------------
    # Search-space integration
    # ------------------------------------------------------------------

    def supported_models(self) -> set[str]:
        ...

    def stage_only_param_registry(self) -> dict[str, dict[str, dict[str, Any]]]:
        ...

    def normalize_search_space_sections(
        self,
        payload: dict[str, Any],
        *,
        normalize_model_section: Any,
        normalize_training_section: Any,
    ) -> dict[str, Any]:
        """Return extra search-space section entries (e.g. ``bs_preforcast_models``)."""
        ...

    def model_search_space_key(self) -> str:
        """Key used to look up model specs in the search-space (e.g. ``"bs_preforcast_models"``)."""
        ...

    def training_search_space_key(self) -> str:
        """Key used to look up training specs (e.g. ``"bs_preforcast_training"``)."""
        ...

    def validate_model(self, model_name: str) -> None:
        """Raise ``ValueError`` for disallowed models in stage scope."""
        ...

    def model_search_space_fallback_key(self) -> str | None:
        """Optional fallback key when the primary key has no entry (e.g. ``"models"``)."""
        ...

    def training_search_space_fallback_key(self) -> str | None:
        """Optional fallback key for training (e.g. ``"training"``)."""
        ...

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
        ...

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
        ...

    # ------------------------------------------------------------------
    # Manifest / validation payload
    # ------------------------------------------------------------------

    def manifest_block(self, loaded: LoadedConfig) -> dict[str, Any]:
        ...

    def validation_payload(
        self,
        loaded: LoadedConfig,
        selected_jobs: Any,
        caps_by_model: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    # ------------------------------------------------------------------
    # Fanout helpers
    # ------------------------------------------------------------------

    def fanout_config_keys(self) -> set[str]:
        """Keys to preserve when slicing payload for jobs fanout."""
        ...

    def fanout_filter_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a cleaned payload dict suitable for fanout re-normalization."""
        ...

    def fanout_stage_payload(
        self,
        loaded: LoadedConfig,
    ) -> dict[str, Any] | None:
        """Extra stage entries to merge into fanout normalized payload."""
        ...
