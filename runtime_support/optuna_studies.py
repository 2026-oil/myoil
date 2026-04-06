from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
import copy
import json

from app_config import LoadedConfig

STUDY_SEED_STRIDE = 10_000


@dataclass(frozen=True)
class StudySelection:
    study_count: int
    configured_selected_study_index: int | None
    cli_selected_study_index: int | None
    selected_study_index: int | None
    canonical_projection_study_index: int
    execute_study_indices: tuple[int, ...]

    @property
    def multi_study_enabled(self) -> bool:
        return self.study_count > 1

    @property
    def targeted(self) -> bool:
        return self.selected_study_index is not None


@dataclass(frozen=True)
class StudyContext:
    selection: StudySelection
    stage: str
    job_name: str
    study_index: int
    sampler_seed: int
    proposal_flow_id: str
    study_name: str
    study_label: str
    study_root: Path
    storage_path: Path
    summary_path: Path
    best_params_path: Path
    training_best_params_path: Path | None
    visuals_root: Path
    visual_inventory_path: Path

    @property
    def trials_root(self) -> Path:
        return self.study_root / "trials" / self.job_name / self.stage


def _runtime_payload(loaded: LoadedConfig) -> Mapping[str, Any]:
    runtime_payload = loaded.normalized_payload.get("runtime")
    if isinstance(runtime_payload, dict):
        return runtime_payload
    return {}


def _coerce_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer when provided")
    return int(value)


def _coerce_positive_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a positive integer")
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def resolve_study_selection(
    loaded: LoadedConfig,
    *,
    cli_selected_study: int | None = None,
) -> StudySelection:
    runtime_payload = _runtime_payload(loaded)
    runtime_config = getattr(loaded.config, "runtime", None)
    configured_count = runtime_payload.get(
        "opt_study_count",
        getattr(runtime_config, "opt_study_count", None),
    )
    study_count = _coerce_positive_int(
        configured_count,
        field_name="runtime.opt_study_count",
        default=1,
    )
    configured_selected = _coerce_optional_int(
        runtime_payload.get(
            "opt_selected_study",
            getattr(runtime_config, "opt_selected_study", None),
        ),
        field_name="runtime.opt_selected_study",
    )
    cli_selected = _coerce_optional_int(
        cli_selected_study,
        field_name="--optuna-study",
    )
    effective_selected = cli_selected if cli_selected is not None else configured_selected
    if effective_selected is not None and not 1 <= effective_selected <= study_count:
        raise ValueError(
            f"selected Optuna study {effective_selected} must be within 1..{study_count}"
        )
    execute_indices = (
        (effective_selected,)
        if effective_selected is not None
        else tuple(range(1, study_count + 1))
    )
    return StudySelection(
        study_count=study_count,
        configured_selected_study_index=configured_selected,
        cli_selected_study_index=cli_selected,
        selected_study_index=effective_selected,
        canonical_projection_study_index=(
            effective_selected if effective_selected is not None else 1
        ),
        execute_study_indices=tuple(int(index) for index in execute_indices),
    )


def study_label(study_index: int) -> str:
    return f"study-{int(study_index):02d}"


def study_root(run_root: Path, study_index: int) -> Path:
    return run_root / "studies" / study_label(study_index)


def trial_dir_name(trial_number: int) -> str:
    return f"trial-{int(trial_number):04d}"


def build_selection_runtime_payload(
    loaded: LoadedConfig,
    *,
    selected_study_index: int | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(loaded.normalized_payload)
    runtime_payload = payload.setdefault("runtime", {})
    runtime_payload.setdefault("opt_study_count", 1)
    runtime_payload["opt_selected_study"] = selected_study_index
    return payload


def clone_loaded_with_selected_study(
    loaded: LoadedConfig,
    *,
    selected_study_index: int | None,
) -> LoadedConfig:
    return replace(
        loaded,
        normalized_payload=build_selection_runtime_payload(
            loaded, selected_study_index=selected_study_index
        ),
    )


def loaded_with_study_selection_override(
    loaded: LoadedConfig,
    selected_study_index: int | None,
) -> LoadedConfig:
    selection = resolve_study_selection(
        loaded,
        cli_selected_study=selected_study_index,
    )
    updated = clone_loaded_with_selected_study(
        loaded,
        selected_study_index=selection.selected_study_index,
    )
    return replace(
        updated,
        config=replace(
            updated.config,
            runtime=replace(
                updated.config.runtime,
                opt_study_count=selection.study_count,
                opt_selected_study=selection.selected_study_index,
            ),
        ),
    )


def _task_name(loaded: LoadedConfig) -> str:
    return loaded.config.task.name or loaded.source_path.stem or "run"


def build_study_context(
    loaded: LoadedConfig,
    *,
    selection: StudySelection | None = None,
    run_root: Path | None = None,
    stage_root: Path | None = None,
    stage: str,
    job_name: str,
    study_index: int,
    base_seed: int | None = None,
    worker_index: int = 0,
) -> StudyContext:
    selection = resolve_study_selection(loaded) if selection is None else selection
    if run_root is None:
        if stage_root is None:
            raise ValueError("run_root or stage_root is required for build_study_context")
        run_root = stage_root
    if base_seed is None:
        base_seed = int(getattr(loaded.config.runtime, "random_seed", 1))
    label = study_label(study_index)
    stage_root = study_root(run_root, study_index)
    sampler_seed = int(base_seed) + (study_index - 1) * STUDY_SEED_STRIDE + int(worker_index)
    proposal_flow_id = sha256(
        "::".join(
            [
                _task_name(loaded),
                loaded.active_jobs_route_slug or "",
                job_name,
                stage,
                str(study_index),
                str(base_seed),
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    parts = [
        part
        for part in (
            "neuralforecast",
            _task_name(loaded),
            loaded.active_jobs_route_slug or "",
            job_name,
            stage,
            loaded.input_hash[:12],
            label,
        )
        if part
    ]
    stage_dir = stage_root / "optuna" / job_name / stage
    return StudyContext(
        selection=selection,
        stage=stage,
        job_name=job_name,
        study_index=study_index,
        sampler_seed=sampler_seed,
        proposal_flow_id=proposal_flow_id,
        study_name="::".join(parts),
        study_label=label,
        study_root=stage_root,
        storage_path=stage_dir / ".optuna" / f"{stage}.journal",
        summary_path=stage_dir / "optuna_study_summary.json",
        best_params_path=stage_dir / "best_params.json",
        training_best_params_path=stage_dir / "training_best_params.json",
        visuals_root=stage_root / "visuals" / job_name / stage,
        visual_inventory_path=stage_root / "visuals" / job_name / stage / "artifact_inventory.json",
    )


def study_catalog_entry(context: StudyContext) -> dict[str, Any]:
    return {
        "study_index": context.study_index,
        "study_label": context.study_label,
        "study_root": str(context.study_root.resolve()),
        "study_name": context.study_name,
        "storage_path": str(context.storage_path.resolve()),
        "summary_path": str(context.summary_path.resolve()),
        "best_params_path": str(context.best_params_path.resolve()),
        "training_best_params_path": (
            str(context.training_best_params_path.resolve())
            if context.training_best_params_path is not None
            else None
        ),
        "sampler_seed": context.sampler_seed,
        "proposal_flow_id": context.proposal_flow_id,
        "selected": context.study_index == context.selection.selected_study_index,
        "canonical_projection": (
            context.study_index == context.selection.canonical_projection_study_index
        ),
    }


def selection_manifest_payload(
    selection: StudySelection,
    *,
    study_catalog_path: Path | None = None,
    cross_study_visuals_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "study_count": selection.study_count,
        "configured_selected_study_index": selection.configured_selected_study_index,
        "cli_selected_study_index": selection.cli_selected_study_index,
        "selected_study_index": selection.selected_study_index,
        "canonical_projection_study_index": selection.canonical_projection_study_index,
        "execute_study_indices": list(selection.execute_study_indices),
        "study_catalog_path": (
            str(study_catalog_path.resolve()) if study_catalog_path is not None else None
        ),
        "cross_study_visualizations_root": (
            str(cross_study_visuals_root.resolve())
            if cross_study_visuals_root is not None
            else None
        ),
    }


def build_study_catalog_payload(
    run_root: Path,
    selection: StudySelection,
    *,
    study_summaries: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    study_summaries = study_summaries or {}
    for study_index in range(1, selection.study_count + 1):
        entry = {
            "study_index": study_index,
            "study_label": study_label(study_index),
            "study_root": str(study_root(run_root, study_index).resolve()),
            "executed": study_index in selection.execute_study_indices,
            "selected": study_index == selection.selected_study_index,
            "canonical_projection": study_index
            == selection.canonical_projection_study_index,
        }
        summary = study_summaries.get(study_index)
        if summary is not None:
            entry["summary"] = json.loads(json.dumps(summary))
        entries.append(entry)
    return {
        "study_count": selection.study_count,
        "selected_study_index": selection.selected_study_index,
        "canonical_projection_study_index": selection.canonical_projection_study_index,
        "entries": entries,
    }


def write_study_catalog(
    target: Path,
    payload_or_contexts: Mapping[str, Any] | Sequence[StudyContext],
    *,
    selected_study_index: int | None = None,
    canonical_projection_study_index: int | None = None,
    extra_entries: Mapping[int, Mapping[str, Any]] | None = None,
) -> Path:
    if isinstance(payload_or_contexts, Mapping):
        path = target
        payload = dict(payload_or_contexts)
    else:
        contexts = list(payload_or_contexts)
        path = target / "study_catalog.json"
        selection = contexts[0].selection if contexts else StudySelection(
            study_count=0,
            configured_selected_study_index=None,
            cli_selected_study_index=None,
            selected_study_index=None,
            canonical_projection_study_index=0,
            execute_study_indices=(),
        )
        payload = build_study_catalog_payload(
            target,
            selection,
            study_summaries={
                context.study_index: {
                    **(extra_entries or {}).get(context.study_index, {}),
                    "study_name": context.study_name,
                    "storage_path": str(context.storage_path.resolve()),
                    "sampler_seed": context.sampler_seed,
                    "proposal_flow_id": context.proposal_flow_id,
                }
                for context in contexts
            },
        )
        if selected_study_index is not None:
            payload["selected_study_index"] = selected_study_index
        if canonical_projection_study_index is not None:
            payload["canonical_projection_study_index"] = (
                canonical_projection_study_index
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
