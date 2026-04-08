#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_config import JobConfig, LoadedConfig, load_app_config
from runtime_support.optuna_studies import (
    StudySelection,
    build_study_catalog_payload,
    build_study_context,
    study_catalog_entry,
    write_study_catalog,
)
from runtime_support.optuna_visuals import (
    build_study_visualizations,
    write_cross_study_visualizations,
)
from runtime_support.runner import _copy_projection_file, _load_main_tuning_result

DEFAULT_SOURCE_RUN_ROOT = (
    ROOT / "runs" / "feature_set_aaforecast_aa_forecast_brentoil"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "runs" / "feature_set_aaforecast_aa_forecast_brentoil_til3"
)
DEFAULT_MODEL = "AAForecast"
ALLOWED_STUDIES = (1, 2, 3)
FLAT_COPY_RELATIVE_PATHS = (
    Path("study_catalog.json"),
    Path("optuna_study_summary.json"),
    Path("visualizations/cross_study_summary.json"),
    Path("visualizations/cross_study_leaderboard.csv"),
    Path("visualizations/cross_study_dashboard.html"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the AAForecast study-1~3 summary bundle from scheduler-local "
            "persisted Optuna state using the canonical runner reload path."
        )
    )
    parser.add_argument(
        "--source-run-root",
        default=str(DEFAULT_SOURCE_RUN_ROOT),
        help="Source parent run root. Defaults to the approved aa_forecast_brentoil run.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Target til3 output root. Defaults to the approved til3 run root.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to reconstruct. Defaults to AAForecast.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Reconstruct and validate studies 1,2,3 from scheduler-local state without "
            "writing output artifacts."
        ),
    )
    return parser.parse_args()


def _resolve_repo_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _study_label(study_index: int) -> str:
    return f"study-{study_index:02d}"


def _synthetic_selection() -> StudySelection:
    return StudySelection(
        study_count=len(ALLOWED_STUDIES),
        configured_selected_study_index=None,
        cli_selected_study_index=None,
        selected_study_index=None,
        canonical_projection_study_index=1,
        execute_study_indices=ALLOWED_STUDIES,
    )


def _load_parent_config(source_run_root: Path) -> LoadedConfig:
    manifest_path = source_run_root / "manifest" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    config_source_path = Path(str(manifest["config_source_path"]))
    shared_settings_path = manifest.get("shared_settings_path")
    return load_app_config(
        ROOT,
        config_path=config_source_path,
        shared_settings_path=shared_settings_path,
    )


def _select_job(loaded: LoadedConfig, model_name: str) -> JobConfig:
    matches = [job for job in loaded.config.jobs if job.model == model_name]
    if not matches:
        available = ", ".join(job.model for job in loaded.config.jobs)
        raise ValueError(
            f"Requested model {model_name!r} was not found. Available jobs: {available}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Requested model {model_name!r} resolved to {len(matches)} jobs; "
            "the til3 helper requires a single job"
        )
    job = matches[0]
    if job.validated_mode != "learned_auto":
        raise ValueError(
            f"Model {model_name!r} must be in learned_auto mode, got {job.validated_mode!r}"
        )
    return job


def _worker_summary_paths(source_run_root: Path, study_index: int, model_name: str) -> list[Path]:
    return sorted(
        (
            source_run_root
            / "scheduler"
            / _study_label(study_index)
            / "workers"
        ).glob(f"{model_name}#*/summary.json")
    )


def _budget_path(source_run_root: Path, study_index: int, model_name: str) -> Path:
    return (
        source_run_root
        / "scheduler"
        / "models"
        / model_name
        / "studies"
        / _study_label(study_index)
        / "optuna"
        / model_name
        / "main-search"
        / ".optuna"
        / "main-search.journal.budget.json"
    )


def _require_source_study_is_eligible(
    source_run_root: Path,
    *,
    study_index: int,
    model_name: str,
) -> dict[str, Any]:
    worker_paths = _worker_summary_paths(source_run_root, study_index, model_name)
    if not worker_paths:
        raise FileNotFoundError(
            f"study {_study_label(study_index)} is missing worker summaries under "
            f"{source_run_root / 'scheduler'}"
        )
    worker_summaries = [_read_json(path) for path in worker_paths]
    nonzero = [
        summary
        for summary in worker_summaries
        if int(summary.get("returncode", 1)) != 0
    ]
    if nonzero:
        raise RuntimeError(
            f"study {_study_label(study_index)} has non-zero worker return codes: "
            f"{[summary.get('returncode') for summary in nonzero]}"
        )

    budget_payload = _read_json(_budget_path(source_run_root, study_index, model_name))
    reserved = int(budget_payload["reserved_trial_count"])
    target = int(budget_payload["target_trial_count"])
    if reserved != target:
        raise RuntimeError(
            f"study {_study_label(study_index)} is incomplete: "
            f"reserved_trial_count={reserved} target_trial_count={target}"
        )

    return {
        "study_index": study_index,
        "worker_summary_paths": [str(path.resolve()) for path in worker_paths],
        "budget_path": str(
            _budget_path(source_run_root, study_index, model_name).resolve()
        ),
        "worker_count": len(worker_paths),
        "reserved_trial_count": reserved,
        "target_trial_count": target,
    }


def _write_json(path: Path, payload: MappingLike) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


MappingLike = dict[str, Any]


def _copy_flat_outputs(output_root: Path, model_root: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for relative_path in FLAT_COPY_RELATIVE_PATHS:
        source = model_root / relative_path
        if not source.exists():
            raise FileNotFoundError(f"missing nested artifact for flat copy: {source}")
        target = output_root / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[source.name] = str(target.resolve())
    return copied


def _validate_generated_outputs(
    *,
    output_root: Path,
    model_root: Path,
    study_catalog: dict[str, Any],
) -> None:
    if study_catalog.get("study_count") != len(ALLOWED_STUDIES):
        raise RuntimeError("generated study catalog has an unexpected study_count")
    if study_catalog.get("selected_study_index") is not None:
        raise RuntimeError("generated study catalog must keep selected_study_index=null")
    if study_catalog.get("canonical_projection_study_index") != 1:
        raise RuntimeError(
            "generated study catalog must keep canonical_projection_study_index=1"
        )
    entry_indices = [int(entry["study_index"]) for entry in study_catalog["entries"]]
    if entry_indices != list(ALLOWED_STUDIES):
        raise RuntimeError(f"generated study catalog contains unexpected entries: {entry_indices}")

    forbidden_labels = ("study-04", "study-05")
    nested_paths = [
        model_root / "study_catalog.json",
        model_root / "optuna_study_summary.json",
        model_root / "visualizations" / "cross_study_summary.json",
        model_root / "visualizations" / "cross_study_leaderboard.csv",
        model_root / "visualizations" / "cross_study_dashboard.html",
    ]
    flat_paths = [output_root / relative_path.name for relative_path in FLAT_COPY_RELATIVE_PATHS]
    for path in [*nested_paths, *flat_paths]:
        if not path.exists():
            raise FileNotFoundError(f"expected artifact is missing: {path}")
        payload = path.read_text(encoding="utf-8")
        for forbidden in forbidden_labels:
            if forbidden in payload:
                raise RuntimeError(f"unexpected {forbidden} leakage detected in {path}")

    for relative_path in FLAT_COPY_RELATIVE_PATHS:
        nested_path = model_root / relative_path
        flat_path = output_root / relative_path.name
        if nested_path.read_bytes() != flat_path.read_bytes():
            raise RuntimeError(
                f"flat output {flat_path} does not match nested source {nested_path}"
            )


def regenerate_til3_summary(
    *,
    source_run_root: Path,
    output_root: Path,
    model_name: str,
    dry_run: bool,
) -> dict[str, Any]:
    loaded = _load_parent_config(source_run_root)
    job = _select_job(loaded, model_name)
    selection = _synthetic_selection()
    source_model_root = source_run_root / "scheduler" / "models" / model_name
    target_model_root = output_root / "models" / model_name

    eligibility: list[dict[str, Any]] = []
    per_study_summary: dict[int, dict[str, Any]] = {}
    per_study_paths: dict[int, dict[str, Path]] = {}

    for study_index in ALLOWED_STUDIES:
        eligibility.append(
            _require_source_study_is_eligible(
                source_run_root,
                study_index=study_index,
                model_name=model_name,
            )
        )
        source_context = build_study_context(
            loaded,
            selection=selection,
            stage_root=source_model_root,
            stage="main-search",
            job_name=model_name,
            study_index=study_index,
        )
        best_params, best_training_params, study_summary = _load_main_tuning_result(
            loaded,
            job,
            source_model_root,
            study_context=source_context,
        )
        per_study_summary[study_index] = study_summary

        target_context = build_study_context(
            loaded,
            selection=selection,
            stage_root=target_model_root,
            stage="main-search",
            job_name=model_name,
            study_index=study_index,
        )
        study_root = target_context.study_root
        best_params_path = study_root / "best_params.json"
        training_best_params_path = study_root / "training_best_params.json"
        study_summary_path = study_root / "optuna_study_summary.json"
        metadata_path = study_root / "metadata.json"
        per_study_paths[study_index] = {
            "best_params": best_params_path,
            "training_best_params": training_best_params_path,
            "study_summary": study_summary_path,
            "metadata": metadata_path,
        }

        if dry_run:
            continue

        _write_json(best_params_path, best_params)
        _write_json(training_best_params_path, best_training_params)
        _write_json(study_summary_path, study_summary)
        metadata_payload = study_catalog_entry(source_context)
        metadata_payload.update(
            {
                "study_root": str(study_root.resolve()),
                "summary_path": str(study_summary_path.resolve()),
                "best_params_path": str(best_params_path.resolve()),
                "training_best_params_path": str(
                    training_best_params_path.resolve()
                ),
            }
        )
        _write_json(metadata_path, metadata_payload)
        build_study_visualizations(target_context, study_summary)

    result: dict[str, Any] = {
        "source_run_root": str(source_run_root.resolve()),
        "output_root": str(output_root.resolve()),
        "model": model_name,
        "allowed_studies": list(ALLOWED_STUDIES),
        "eligibility": eligibility,
        "canonical_reconstruction_path": "_load_main_tuning_result()->_collect_main_tuning_result()",
        "synthetic_metadata": {
            "study_count": selection.study_count,
            "selected_study_index": selection.selected_study_index,
            "canonical_projection_study_index": selection.canonical_projection_study_index,
        },
    }
    if dry_run:
        result["dry_run"] = True
        result["study_indices"] = sorted(per_study_summary)
        return result

    projection_paths = per_study_paths[selection.canonical_projection_study_index]
    target_model_root.mkdir(parents=True, exist_ok=True)
    _copy_projection_file(projection_paths["best_params"], target_model_root / "best_params.json")
    _copy_projection_file(
        projection_paths["training_best_params"],
        target_model_root / "training_best_params.json",
    )
    _copy_projection_file(
        projection_paths["study_summary"],
        target_model_root / "optuna_study_summary.json",
    )

    study_catalog = build_study_catalog_payload(
        target_model_root,
        selection,
        study_summaries=per_study_summary,
    )
    write_study_catalog(target_model_root / "study_catalog.json", study_catalog)
    write_cross_study_visualizations(run_root=target_model_root, study_catalog=study_catalog)
    flat_outputs = _copy_flat_outputs(output_root, target_model_root)
    _validate_generated_outputs(
        output_root=output_root,
        model_root=target_model_root,
        study_catalog=study_catalog,
    )

    result["nested_root"] = str(target_model_root.resolve())
    result["nested_study_catalog_path"] = str(
        (target_model_root / "study_catalog.json").resolve()
    )
    result["flat_outputs"] = flat_outputs
    result["study_indices"] = sorted(per_study_summary)
    return result


def main() -> int:
    args = parse_args()
    source_run_root = _resolve_repo_path(args.source_run_root)
    output_root = _resolve_repo_path(args.output_root)
    payload = regenerate_til3_summary(
        source_run_root=source_run_root,
        output_root=output_root,
        model_name=args.model,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
