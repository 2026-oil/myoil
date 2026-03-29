from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app_config import JobConfig, LoadedConfig


class DefaultBsPreforcastPlugin:
    name = "default"

    def resolve_injection_mode(
        self,
        loaded: LoadedConfig,
        *,
        selected_jobs: Iterable[Any],
    ) -> str:
        from plugins.bs_preforcast.runtime import resolve_bs_preforcast_injection_mode

        return resolve_bs_preforcast_injection_mode(loaded, selected_jobs=selected_jobs)

    def prepare_fold_inputs(
        self,
        loaded: LoadedConfig,
        job: JobConfig,
        train_df: pd.DataFrame,
        future_df: pd.DataFrame,
        *,
        run_root: Path | None = None,
    ) -> tuple[LoadedConfig, pd.DataFrame, pd.DataFrame, str]:
        from plugins.bs_preforcast.runtime import prepare_bs_preforcast_fold_inputs

        return prepare_bs_preforcast_fold_inputs(
            loaded,
            job,
            train_df,
            future_df,
            run_root=run_root,
        )

    def materialize_stage(
        self,
        *,
        loaded: LoadedConfig,
        selected_jobs: Iterable[JobConfig],
        run_root: Path,
        main_resolved_path: Path,
        main_capability_path: Path,
        main_manifest_path: Path,
        entrypoint_version: str,
        validate_only: bool,
    ) -> None:
        from plugins.bs_preforcast.runtime import materialize_bs_preforcast_stage

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

    def load_stage_config(
        self,
        repo_root: Path,
        loaded: LoadedConfig,
    ) -> LoadedConfig | None:
        from plugins.bs_preforcast.runtime import load_bs_preforcast_stage_config

        return load_bs_preforcast_stage_config(repo_root, loaded)
