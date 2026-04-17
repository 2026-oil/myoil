from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import fcntl
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
from neuralforecast import NeuralForecast
from optuna.trial import TrialState

from runtime_support.adapters import build_multivariate_inputs, build_univariate_inputs
from plugin_contracts.stage_registry import get_active_stage_plugin
from app_config import (
    JobConfig,
    LoadedConfig,
    load_app_config,  # noqa: F401 - re-exported for bootstrap/tests
    loaded_config_for_jobs_fanout,  # noqa: F401 - compatibility export
)
from runtime_support.manifest import (
    build_manifest,
    _atomic_write_text,
    _manifest_lock,
    write_manifest,
)
from runtime_support.forecast_models import (
    BASELINE_MODEL_NAMES,
    ModelCapabilities,
    build_model,
    is_direct_top_level_model,
    validate_job,
)
from tuning import (
    DEFAULT_OPTUNA_STUDY_DIRECTION,
    SUPPORTED_MODEL_AUTO_MODEL_NAMES,
    LEGACY_TRAINING_SELECTOR_TO_CONFIG_FIELD,
    build_optuna_sampler,
    optuna_num_trials,
    optuna_seed,
    suggest_model_params,
    suggest_training_params,
    training_param_registry_for_model,
    training_range_source_for_model,
)
from runtime_support.progress import (
    ConsoleProgressRenderer,
    ModelProgressState,
    emit_progress_event,
)
from runtime_support.scheduler import (
    build_device_groups,
    build_launch_plan,
    build_tuning_launch_plan,
    run_parallel_jobs,
)
from runtime_support.optuna_studies import (
    StudyContext,
    StudySelection,
    build_study_catalog_payload,
    build_study_context,
    loaded_with_study_selection_override,
    resolve_study_selection,
    selection_manifest_payload,
    study_catalog_entry,
    study_label,
    trial_dir_name,
    write_study_catalog,
)
from runtime_support.optuna_visuals import (
    build_cross_study_visualizations,
    build_study_visualizations,
    write_cross_study_visualizations,
)
from neuralforecast.models.bs_preforcast_direct import (
    DirectPredictionResult,
    normalized_direct_job_params,
    predict_univariate_direct,
)

ENTRYPOINT_VERSION = "neuralforecast-runtime-v1"
LOSS_CURVE_PLOT_FILENAME = "loss_curve.png"
LOSS_CURVE_SAMPLE_FILENAME = "loss_curve_every_10_global_steps.csv"
LOSS_CURVE_SMOOTHED_SAMPLE_FILENAME = "loss_curve_smoothed_every_10_global_steps.csv"
LOSS_CURVE_VALIDATION_FILENAME = "loss_curve_validation_points.csv"
SUMMARY_LOSS_ARTIFACTS_FILENAME = "loss_curve_artifacts.csv"
SUMMARY_RESULTS_FILENAME = "result.csv"
TRIAL_PREDICTIONS_FILENAME = "predictions.csv"
TRIAL_FOLD_PLOT_FILENAME = "plot.png"
TRIAL_FOLD_METRICS_FILENAME = "metrics.json"
TRIAL_FOLD_CHECKPOINT_FILENAME = "checkpoint.pt"
LOSS_CURVE_SAMPLE_EVERY_N_STEPS = 10
LOSS_CURVE_TRAIN_SMOOTHING_WINDOW = 5


class _OptunaTrialFailure(RuntimeError):
    """Recoverable per-trial failure that should not abort the whole study."""


@dataclass(frozen=True)
class _CurveFrameCarrier:
    curve_frame: pd.DataFrame

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ProgressLogger:
    def __init__(
        self,
        job_name: str,
        total_steps: int,
        *,
        model_index: int = 1,
        total_models: int = 1,
    ) -> None:
        self.state = ModelProgressState(
            job_name=job_name,
            model_index=model_index,
            total_models=total_models,
            total_steps=max(total_steps, 1),
        )
        self._structured_mode = (
            os.environ.get("NEURALFORECAST_PROGRESS_MODE") == "structured"
        )
        self._renderer = None if self._structured_mode else ConsoleProgressRenderer()

    def _publish(
        self,
        event: str,
        *,
        fold_idx: int | None = None,
        total_folds: int | None = None,
        phase: str | None = None,
        detail: str | None = None,
        status: str | None = None,
        step_increment: int = 0,
    ) -> None:
        self.state.event = event
        self.state.completed_steps = min(
            self.state.total_steps,
            max(0, self.state.completed_steps + step_increment),
        )
        self.state.current_fold = fold_idx
        if total_folds is not None:
            self.state.total_folds = total_folds
        if phase is not None:
            self.state.phase = phase
        self.state.detail = detail
        if status is not None:
            self.state.status = status
        if self._structured_mode:
            emit_progress_event(self.state)
            return
        assert self._renderer is not None
        self._renderer.render([self.state])

    def model_started(self, *, total_folds: int, detail: str | None = None) -> None:
        self._publish(
            "model-start",
            total_folds=total_folds,
            detail=detail,
            status="running",
        )

    def fold_started(
        self, fold_idx: int, *, total_folds: int, phase: str | None = None
    ) -> None:
        self._publish(
            "fold-start",
            fold_idx=fold_idx,
            total_folds=total_folds,
            phase=phase,
            status="running",
        )

    def fold_completed(
        self,
        fold_idx: int,
        *,
        total_folds: int,
        phase: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._publish(
            "fold-done",
            fold_idx=fold_idx,
            total_folds=total_folds,
            phase=phase,
            detail=detail,
            status="running",
            step_increment=1,
        )

    def error(
        self,
        fold_idx: int | None,
        *,
        total_folds: int | None = None,
        phase: str | None = None,
        exc: BaseException,
    ) -> None:
        self._publish(
            "error",
            fold_idx=fold_idx,
            total_folds=total_folds,
            phase=phase,
            detail=f"{type(exc).__name__}: {exc}",
            status="failed",
        )
        if self._renderer is not None:
            self._renderer.close()

    def model_finished(self, *, detail: str | None = None) -> None:
        self._publish(
            "model-done",
            phase=self.state.phase,
            detail=detail,
            status="completed",
        )
        if self._renderer is not None:
            self._renderer.close()


def _selected_jobs(loaded: LoadedConfig, names: list[str] | None):
    if not names:
        return list(loaded.config.jobs)
    allowed = set(names)
    return [job for job in loaded.config.jobs if job.model in allowed]


def _safe_output_root_part(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value
    ).strip(".-")


def _default_output_root(repo_root: Path, loaded: LoadedConfig) -> Path:
    task_name = (loaded.config.task.name or "").strip()
    if not task_name:
        return repo_root / "runs" / "validation"
    config_parent = loaded.source_path.parent
    parent_name = (
        repo_root.name
        if config_parent.resolve() == repo_root.resolve()
        else config_parent.name
    )
    safe_parts = [
        part
        for part in (
            _safe_output_root_part(parent_name),
            _safe_output_root_part(task_name),
            _safe_output_root_part(loaded.active_jobs_route_slug or ""),
        )
        if part
    ]
    safe_name = "_".join(safe_parts)
    return repo_root / "runs" / (safe_name or "validation")


def _resolve_run_roots(
    repo_root: Path,
    loaded: LoadedConfig,
    *,
    output_root: str | None,
) -> dict[str, Path]:
    if output_root is not None:
        explicit_root = Path(output_root)
        return {"run_root": explicit_root, "summary_root": explicit_root}
    default_root = _default_output_root(repo_root, loaded)
    return {"run_root": default_root, "summary_root": default_root}


def _remove_existing_artifact(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _prune_model_run_artifacts(run_root: Path, model_name: str) -> None:
    workers_root = run_root / "scheduler" / "workers"
    targets = [
        run_root / "cv" / f"{model_name}_forecasts.csv",
        run_root / "cv" / f"{model_name}_metrics_by_cutoff.csv",
        run_root / "models" / model_name,
        workers_root / model_name,
    ]
    for target in targets:
        _remove_existing_artifact(target)
    if not workers_root.exists():
        return
    for worker_root in sorted(workers_root.glob(f"{model_name}#*")):
        _remove_existing_artifact(worker_root)


def _sync_study_roots(
    source_models_dir: Path,
    target_models_dir: Path,
    *,
    study_indices: Sequence[int],
) -> None:
    target_studies_root = target_models_dir / "studies"
    target_studies_root.mkdir(parents=True, exist_ok=True)
    for study_index in study_indices:
        label = study_label(study_index)
        source_study_root = source_models_dir / "studies" / label
        if not source_study_root.exists():
            raise FileNotFoundError(
                f"parallel tuning study root missing: {source_study_root}"
            )
        target_study_root = target_studies_root / label
        _remove_existing_artifact(target_study_root)
        shutil.copytree(source_study_root, target_study_root)


def _should_prune_model_run_artifacts(job: JobConfig, *, main_stage: str) -> bool:
    if main_stage == "tune-main-only":
        return False
    return job.validated_mode != "learned_auto"


def _build_resolved_artifacts(
    repo_root: Path, loaded: LoadedConfig, output_root: Path
) -> dict[str, Path]:
    run_root = output_root.resolve()
    study_selection = resolve_study_selection(loaded)
    resolved_path = run_root / "config" / "config.resolved.json"
    capability_path = run_root / "config" / "capability_report.json"
    manifest_path = run_root / "manifest" / "run_manifest.json"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    capability_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(loaded.normalized_payload, indent=2), encoding="utf-8"
    )
    manifest = build_manifest(
        loaded,
        compat_mode="dual_read",
        entrypoint_version=ENTRYPOINT_VERSION,
        resolved_config_path=resolved_path,
        optuna_payload=selection_manifest_payload(study_selection),
    )
    write_manifest(manifest_path, manifest)
    return {
        "run_root": run_root,
        "resolved_path": resolved_path,
        "capability_path": capability_path,
        "manifest_path": manifest_path,
    }

def _validate_jobs(loaded: LoadedConfig, selected_jobs, capability_path: Path) -> None:
    payload = {}
    caps_by_model: dict[str, Any] = {}
    for job in selected_jobs:
        caps = _job_capabilities_for(loaded, job)
        caps_by_model[job.model] = caps
        payload[job.model] = {
            **caps.__dict__,
            "requested_mode": job.requested_mode,
            "validated_mode": job.validated_mode,
            "supports_auto": job.model in SUPPORTED_MODEL_AUTO_MODEL_NAMES,
            "search_space_entry_found": bool(job.selected_search_params),
            "selected_search_params": list(job.selected_search_params),
            "unknown_search_params": [],
            "validation_error": None,
        }
    payload["training_search"] = {
        "requested_mode": loaded.config.training_search.requested_mode,
        "validated_mode": loaded.config.training_search.validated_mode,
        "supports_auto": bool(loaded.config.training_search.selected_search_params),
        "search_space_entry_found": bool(
            loaded.config.training_search.selected_search_params
        ),
        "selected_search_params": list(
            loaded.config.training_search.selected_search_params
        ),
        "unknown_search_params": [],
        "validation_error": None,
    }
    stage_result = get_active_stage_plugin(loaded.config)
    if stage_result is not None:
        plugin, _ = stage_result
        payload[plugin.config_key] = plugin.validation_payload(
            loaded, selected_jobs, caps_by_model
        )
    capability_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plugin_owned_top_level_job(loaded: LoadedConfig, model_name: str):
    stage_result = get_active_stage_plugin(loaded.config)
    if stage_result is None:
        return None
    plugin, _ = stage_result
    owns_top_level_job = getattr(plugin, "owns_top_level_job", None)
    if callable(owns_top_level_job) and owns_top_level_job(model_name):
        return plugin
    return None


def _job_capabilities_for(loaded: LoadedConfig, job: JobConfig) -> ModelCapabilities:
    plugin = _plugin_owned_top_level_job(loaded, job.model)
    if plugin is not None:
        plugin_caps = getattr(plugin, "capabilities_for", None)
        if callable(plugin_caps):
            payload = plugin_caps(job.model)
            return ModelCapabilities(**payload)
        return ModelCapabilities(
            name=job.model,
            multivariate=False,
            supports_hist_exog=False,
            supports_futr_exog=False,
            supports_stat_exog=False,
            requires_n_series=False,
        )
    return validate_job(job)


def _update_manifest_artifacts(
    manifest_path: Path,
    *,
    job_name: str,
    study_catalog_path: Path | None = None,
    selected_study_index: int | None = None,
    canonical_projection_study_index: int | None = None,
    model_best_params_path: Path | None = None,
    model_study_summary_path: Path | None = None,
    training_best_params_path: Path | None = None,
    training_study_summary_path: Path | None = None,
    training_range_source: str | None = None,
) -> None:
    with _manifest_lock(manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for job in manifest.get("jobs", []):
            if job.get("model") == job_name:
                if study_catalog_path is not None:
                    job["optuna_study_catalog_path"] = str(study_catalog_path)
                if selected_study_index is not None:
                    job["selected_study_index"] = int(selected_study_index)
                if canonical_projection_study_index is not None:
                    job["canonical_projection_study_index"] = int(
                        canonical_projection_study_index
                    )
                if model_best_params_path is not None:
                    job["model_best_params_path"] = str(model_best_params_path)
                if model_study_summary_path is not None:
                    job["model_optuna_study_summary_path"] = str(model_study_summary_path)
                if training_best_params_path is not None:
                    job["training_best_params_path"] = str(training_best_params_path)
                if training_study_summary_path is not None:
                    job["training_optuna_study_summary_path"] = str(
                        training_study_summary_path
                    )
                if training_range_source is not None:
                    job["training_range_source"] = training_range_source
                break
        if study_catalog_path is not None:
            manifest.setdefault("optuna", {})["study_catalog_path"] = str(
                study_catalog_path
            )
        if selected_study_index is not None:
            manifest.setdefault("optuna", {})["selected_study_index"] = int(
                selected_study_index
            )
        if canonical_projection_study_index is not None:
            manifest.setdefault("optuna", {})["canonical_projection_study_index"] = int(
                canonical_projection_study_index
            )
        if training_best_params_path is not None:
            manifest.setdefault("training_search", {})["best_params_path"] = str(
                training_best_params_path
            )
        if training_study_summary_path is not None:
            manifest.setdefault("training_search", {})["optuna_study_summary_path"] = str(
                training_study_summary_path
            )
        if training_range_source is not None:
            training_payload = manifest.setdefault("training_search", {})
            training_payload.setdefault("training_range_source_by_job", {})[job_name] = (
                training_range_source
            )
            if len(manifest.get("jobs", [])) == 1:
                training_payload["training_range_source"] = training_range_source
            else:
                training_payload.pop("training_range_source", None)
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))


def _effective_config(
    loaded: LoadedConfig, training_override: dict[str, Any] | None = None
):
    if not training_override:
        if loaded.config.training_search.validated_mode != "training_auto":
            return loaded.config
        training_override = {}
    if loaded.config.training_search.validated_mode == "training_auto":
        training_override = {**training_override, "val_size": loaded.config.cv.horizon}
    normalized_override = {
        LEGACY_TRAINING_SELECTOR_TO_CONFIG_FIELD.get(key, key): value
        for key, value in training_override.items()
    }
    return replace(
        loaded.config,
        training=replace(loaded.config.training, **normalized_override),
    )


def _should_use_multivariate(loaded: LoadedConfig, job: JobConfig) -> bool:
    if _plugin_owned_top_level_job(loaded, job.model) is not None:
        return False
    caps = _job_capabilities_for(loaded, job)
    configured_native_exog = (
        (bool(loaded.config.dataset.hist_exog_cols) and caps.supports_hist_exog)
        or (bool(loaded.config.dataset.futr_exog_cols) and caps.supports_futr_exog)
        or (bool(loaded.config.dataset.static_exog_cols) and caps.supports_stat_exog)
    )
    return bool(caps.multivariate and not configured_native_exog)


def _validate_adapters(loaded: LoadedConfig, selected_jobs) -> None:
    source_df = pd.read_csv(loaded.config.dataset.path)
    dt_col = loaded.config.dataset.dt_col
    future_df = source_df if loaded.config.dataset.futr_exog_cols else None
    for job in selected_jobs:
        if _plugin_owned_top_level_job(loaded, job.model) is not None:
            continue
        if _should_use_multivariate(loaded, job):
            build_multivariate_inputs(
                source_df,
                job,
                dataset=loaded.config.dataset,
                dt_col=dt_col,
                future_df=future_df,
            )
        else:
            build_univariate_inputs(
                source_df,
                job,
                dataset=loaded.config.dataset,
                dt_col=dt_col,
                future_df=future_df,
            )


def _build_adapter_inputs(
    loaded: LoadedConfig,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    job: JobConfig,
    dt_col: str,
):
    if _should_use_multivariate(loaded, job):
        return build_multivariate_inputs(
            train_df,
            job,
            dataset=loaded.config.dataset,
            dt_col=dt_col,
            future_df=future_df,
        )
    return build_univariate_inputs(
        train_df,
        job,
        dataset=loaded.config.dataset,
        dt_col=dt_col,
        future_df=future_df,
    )


def _resolve_freq(loaded: LoadedConfig, source_df: pd.DataFrame) -> str:
    if loaded.config.dataset.freq:
        return loaded.config.dataset.freq
    inferred = pd.infer_freq(pd.to_datetime(source_df[loaded.config.dataset.dt_col]))
    if inferred is None:
        raise ValueError(
            "Could not infer frequency from dataset.dt_col; set dataset.freq explicitly"
        )
    return inferred


@dataclass(frozen=True)
class _FoldDiffContext:
    target_col: str
    target_diff_order: int = 0
    target_anchor: float | None = None
    target_first_diff_anchor: float | None = None
    hist_exog_diff_order: int = 0
    hist_exog_cols: tuple[str, ...] = ()


def _runtime_diff_order(mode: str | None) -> int:
    if mode is None:
        return 0
    if mode == "diff":
        return 1
    if mode == "diff-diff":
        return 2
    raise ValueError(f"Unsupported runtime transformation mode: {mode}")


def _apply_diff_transform(series: pd.Series, order: int) -> pd.Series:
    transformed = series.reset_index(drop=True).astype(float)
    for _ in range(order):
        transformed = transformed.diff()
    return transformed


def _has_target_diff(loaded: LoadedConfig) -> bool:
    return _runtime_diff_order(loaded.config.runtime.transformations_target) > 0


def _has_hist_exog_diff(loaded: LoadedConfig) -> bool:
    return bool(
        _runtime_diff_order(loaded.config.runtime.transformations_exog) > 0
        and loaded.config.dataset.hist_exog_cols
    )


def _has_any_runtime_diff(loaded: LoadedConfig) -> bool:
    return _has_target_diff(loaded) or _has_hist_exog_diff(loaded)


def _build_fold_diff_context(
    loaded: LoadedConfig,
    train_df: pd.DataFrame,
    *,
    target_col: str | None = None,
) -> _FoldDiffContext | None:
    target_diff_order = _runtime_diff_order(
        loaded.config.runtime.transformations_target
    )
    hist_exog_diff_order = (
        _runtime_diff_order(loaded.config.runtime.transformations_exog)
        if loaded.config.dataset.hist_exog_cols
        else 0
    )
    if target_diff_order == 0 and hist_exog_diff_order == 0:
        return None
    active_target_col = target_col or loaded.config.dataset.target_col
    required_rows = max(target_diff_order, hist_exog_diff_order) + 1
    if len(train_df) < required_rows:
        raise ValueError(
            "runtime.transformations_target/exog requires at least "
            f"{required_rows} training rows per fold for the configured "
            "diff order"
        )
    target_anchor = None
    target_first_diff_anchor = None
    if target_diff_order > 0:
        target_values = train_df[active_target_col].reset_index(drop=True).astype(float)
        target_anchor = float(target_values.iloc[-1])
        if target_diff_order > 1:
            target_first_diff_anchor = float(
                _apply_diff_transform(target_values, 1).iloc[-1]
            )
    return _FoldDiffContext(
        target_col=active_target_col,
        target_diff_order=target_diff_order,
        target_anchor=target_anchor,
        target_first_diff_anchor=target_first_diff_anchor,
        hist_exog_diff_order=hist_exog_diff_order,
        hist_exog_cols=(
            loaded.config.dataset.hist_exog_cols if hist_exog_diff_order > 0 else ()
        ),
    )


def _transform_training_frame(
    train_df: pd.DataFrame,
    diff_context: _FoldDiffContext | None,
) -> pd.DataFrame:
    normalized = train_df.reset_index(drop=True).copy()
    if diff_context is None:
        return normalized
    trim_rows = max(
        diff_context.target_diff_order,
        diff_context.hist_exog_diff_order,
    )
    if diff_context.target_diff_order > 0:
        normalized[diff_context.target_col] = _apply_diff_transform(
            normalized[diff_context.target_col],
            diff_context.target_diff_order,
        )
    for column in diff_context.hist_exog_cols:
        if column in normalized.columns:
            normalized[column] = _apply_diff_transform(
                normalized[column],
                diff_context.hist_exog_diff_order,
            )
    transformed = normalized.iloc[trim_rows:].reset_index(drop=True)
    if transformed.empty:
        raise ValueError(
            "runtime.transformations_target/exog removed all training rows; "
            "need more rows than the configured diff order"
        )
    return transformed


def _transform_training_series(
    history: pd.Series,
    diff_context: _FoldDiffContext | None,
) -> pd.Series:
    normalized = history.reset_index(drop=True).astype(float)
    if diff_context is None or diff_context.target_diff_order == 0:
        return normalized
    transformed = _apply_diff_transform(
        normalized,
        diff_context.target_diff_order,
    ).iloc[diff_context.target_diff_order :].reset_index(drop=True)
    if transformed.empty:
        raise ValueError(
            "runtime.transformations_target/exog removed all training rows; "
            "need more rows than the configured diff order"
        )
    return transformed


def _restore_prediction_series(
    predictions: pd.Series,
    diff_context: _FoldDiffContext | None,
) -> pd.Series:
    restored = predictions.reset_index(drop=True).astype(float)
    if diff_context is None or diff_context.target_diff_order == 0:
        return restored
    if diff_context.target_anchor is None:
        raise ValueError("diff target restoration requires a target anchor")
    if diff_context.target_diff_order == 1:
        return (restored.cumsum() + diff_context.target_anchor).astype(float)
    if diff_context.target_diff_order == 2:
        if diff_context.target_first_diff_anchor is None:
            raise ValueError(
                "second-order diff target restoration requires a first-diff anchor"
            )
        level = float(diff_context.target_anchor)
        first_diff = float(diff_context.target_first_diff_anchor)
        restored_levels: list[float] = []
        for second_diff in restored:
            first_diff += float(second_diff)
            level += first_diff
            restored_levels.append(level)
        return pd.Series(restored_levels, dtype=float)
    raise ValueError(
        "Unsupported runtime target diff order: "
        f"{diff_context.target_diff_order}"
    )


def _restore_target_predictions(
    target_predictions: pd.DataFrame,
    *,
    prediction_col: str,
    diff_context: _FoldDiffContext | None,
) -> pd.DataFrame:
    restored = target_predictions.reset_index(drop=True).copy()
    restored[prediction_col] = _restore_prediction_series(
        restored[prediction_col],
        diff_context,
    ).to_numpy()
    return restored


def _compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual = actual.reset_index(drop=True)
    predicted = predicted.reset_index(drop=True)
    err = actual - predicted
    mae = float(err.abs().mean())
    mse = float((err**2).mean())
    rmse = mse**0.5
    # Use per-fold actual range normalization so nRMSE remains scale-aware and
    # naturally yields blank report cells for constant-target folds.
    actual_range = float(actual.max() - actual.min()) if not actual.empty else float("nan")
    nrmse = float("nan") if np.isclose(actual_range, 0.0) else rmse / actual_range
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float((err.abs() / actual.abs()).mean())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    ss_res = float((err**2).sum())
    r2 = float("nan") if np.isclose(ss_tot, 0.0) else 1.0 - (ss_res / ss_tot)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "NRMSE": nrmse,
        "R2": r2,
    }


def _maybe_post_predict_fold(
    loaded: LoadedConfig,
    job: JobConfig,
    *,
    target_predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    transformed_train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    fitted_model: Any | None,
    run_root: Path | None,
) -> pd.DataFrame:
    """Call the active stage plugin's ``post_predict_fold`` if it exists."""
    stage_result = get_active_stage_plugin(loaded.config)
    if stage_result is None:
        return target_predictions
    plugin, _ = stage_result
    post_predict = getattr(plugin, "post_predict_fold", None)
    if not callable(post_predict):
        return target_predictions
    return post_predict(
        loaded, job,
        target_predictions=target_predictions,
        train_df=train_df,
        transformed_train_df=transformed_train_df,
        future_df=future_df,
        fitted_model=fitted_model,
        run_root=run_root,
    )


def _fit_and_predict_fold(
    loaded: LoadedConfig,
    job: JobConfig,
    *,
    run_root: Path | None = None,
    source_df: pd.DataFrame,
    freq: str,
    train_idx: list[int],
    test_idx: list[int],
    params_override: dict[str, Any] | None = None,
    training_override: dict[str, Any] | None = None,
    fold_idx: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, Any | None]:
    effective_loaded = loaded
    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col
    train_df = source_df.iloc[train_idx].reset_index(drop=True)
    future_df = source_df.iloc[test_idx].reset_index(drop=True)
    plugin = _plugin_owned_top_level_job(loaded, job.model)
    if plugin is not None:
        return plugin.predict_fold(
            loaded,
            job,
            train_df=train_df,
            future_df=future_df,
            run_root=run_root,
            params_override=params_override,
            training_override=training_override,
            fold_idx=fold_idx,
        )
    stage_result = get_active_stage_plugin(loaded.config)
    if stage_result is not None:
        plugin, _ = stage_result
        effective_loaded, train_df, future_df, _ = plugin.prepare_fold_inputs(
            loaded,
            job,
            train_df,
            future_df,
            run_root=run_root,
        )
    effective_config = _effective_config(effective_loaded, training_override)
    diff_context = _build_fold_diff_context(effective_loaded, train_df)
    transformed_train_df = _transform_training_frame(train_df, diff_context)
    if is_direct_top_level_model(job.model):
        stage_loaded = SimpleNamespace(config=effective_loaded.config)
        merged_params = {**job.params, **(params_override or {})}
        direct_job = replace(
            job,
            params=normalized_direct_job_params(job.model, merged_params),
        )
        direct_result = predict_univariate_direct(
            stage_loaded,
            direct_job,
            target_column=target_col,
            train_df=transformed_train_df,
            future_df=future_df,
        )
        prediction_values = direct_result
        curve_source: Any | None = None
        if isinstance(direct_result, DirectPredictionResult):
            prediction_values = direct_result.predictions
            if direct_result.curve_frame is not None and not direct_result.curve_frame.empty:
                curve_source = _CurveFrameCarrier(direct_result.curve_frame.copy())
        target_predictions = pd.DataFrame(
            {
                "unique_id": [target_col] * len(future_df),
                "ds": pd.to_datetime(future_df[dt_col]).reset_index(drop=True),
                job.model: prediction_values,
            }
        )
        target_predictions = _restore_target_predictions(
            target_predictions,
            prediction_col=job.model,
            diff_context=diff_context,
        )
        target_predictions = _maybe_post_predict_fold(
            loaded, job,
            target_predictions=target_predictions,
            train_df=train_df,
            transformed_train_df=transformed_train_df,
            future_df=future_df,
            fitted_model=None,
            run_root=run_root,
        )
        target_actuals = future_df[target_col].reset_index(drop=True)
        train_end_ds = pd.to_datetime(train_df[dt_col].iloc[-1])
        return target_predictions, target_actuals, train_end_ds, train_df, curve_source
    adapter_inputs = _build_adapter_inputs(
        effective_loaded,
        transformed_train_df,
        future_df,
        job,
        dt_col,
    )
    model = build_model(
        effective_config,
        job,
        n_series=adapter_inputs.metadata.get("n_series"),
        params_override=params_override,
    )
    if hasattr(model, "set_star_precompute_context"):
        model.set_star_precompute_context(
            enabled=True,
            fold_key=json.dumps(
                {
                    "job": job.model,
                    "train_rows": len(transformed_train_df),
                    "train_end": str(train_df[dt_col].iloc[-1]),
                    "params_override": params_override or {},
                    "training_override": training_override or {},
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
        )
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(
        adapter_inputs.fit_df,
        static_df=adapter_inputs.static_df,
        val_size=effective_config.training.val_size,
    )
    predictions = _predict_with_fitted_model(nf, adapter_inputs)
    pred_col = _prediction_column(predictions, job.model)
    target_predictions = predictions[
        predictions["unique_id"] == target_col
    ].reset_index(drop=True)
    target_predictions = _restore_target_predictions(
        target_predictions,
        prediction_col=pred_col,
        diff_context=diff_context,
    )
    target_predictions = _maybe_post_predict_fold(
        loaded, job,
        target_predictions=target_predictions,
        train_df=train_df,
        transformed_train_df=transformed_train_df,
        future_df=future_df,
        fitted_model=nf,
        run_root=run_root,
    )
    target_actuals = future_df[target_col].reset_index(drop=True)
    train_end_ds = pd.to_datetime(train_df[dt_col].iloc[-1])
    return target_predictions, target_actuals, train_end_ds, train_df, nf


def _trial_metrics_summary(
    study: optuna.Study, *, objective_metric: str = "mean_fold_mape"
) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    finished_trial_count = 0
    for trial in study.trials:
        state = trial.state.name.lower()
        state_counts[state] = state_counts.get(state, 0) + 1
        if trial.state.is_finished():
            finished_trial_count += 1
    best_value: float | None = None
    best_trial_number: int | None = None
    if any(trial.state == TrialState.COMPLETE for trial in study.trials):
        best_value = float(study.best_value)
        best_trial_number = int(study.best_trial.number)
    return {
        "direction": study.direction.name.lower(),
        "trial_count": len(study.trials),
        "finished_trial_count": finished_trial_count,
        "state_counts": state_counts,
        "best_value": best_value,
        "best_trial_number": best_trial_number,
        "objective_metric": objective_metric,
    }


def _mean_fold_metric(values: Sequence[float], *, metric_name: str = "metric") -> float:
    if not values:
        raise ValueError("fold metric values must be non-empty")
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"fold {metric_name} values must include at least one finite value")
    return float(finite.mean())


def _objective_stage_label(loaded: LoadedConfig) -> str:
    return "tuning_pre_replay_direct_predictions"


def _selected_optuna_study(args: argparse.Namespace | None) -> int | None:
    value = getattr(args, "optuna_study", None) if args is not None else None
    return None if value is None else int(value)


def _active_study_selection(
    loaded: LoadedConfig,
    args: argparse.Namespace | None = None,
) -> StudySelection:
    return resolve_study_selection(
        loaded,
        cli_selected_study=_selected_optuna_study(args),
    )


def _study_context(
    loaded: LoadedConfig,
    *,
    run_root: Path,
    stage: str,
    job_name: str,
    worker_index: int = 0,
) -> StudyContext:
    selection = _active_study_selection(loaded)
    selected_index = selection.selected_study_index
    if selected_index is None:
        selected_index = selection.canonical_projection_study_index
    return build_study_context(
        loaded,
        selection=selection,
        run_root=run_root,
        stage=stage,
        job_name=job_name,
        study_index=selected_index,
        base_seed=optuna_seed(loaded.config.runtime.random_seed),
        worker_index=worker_index,
    )


def _open_persistent_study(
    study_context: StudyContext,
    *,
    sampler: optuna.samplers.BaseSampler,
) -> tuple[optuna.Study, dict[str, Any]]:
    study_context.storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(study_context.storage_path))
    )
    study = optuna.create_study(
        storage=storage,
        study_name=study_context.study_name,
        load_if_exists=True,
        sampler=sampler,
        direction=DEFAULT_OPTUNA_STUDY_DIRECTION,
    )
    metadata = {
        "study_index": study_context.study_index,
        "study_label": study_context.study_label,
        "study_name": study_context.study_name,
        "storage_backend": "journal",
        "storage_path": str(study_context.storage_path.resolve()),
        "sampler_seed": study_context.sampler_seed,
        "proposal_flow_id": study_context.proposal_flow_id,
        "canonical_projection_study_index": (
            study_context.selection.canonical_projection_study_index
        ),
        "selected_study_index": study_context.selection.selected_study_index,
    }
    study.set_user_attr("study_index", study_context.study_index)
    study.set_user_attr("sampler_seed", study_context.sampler_seed)
    study.set_user_attr("proposal_flow_id", study_context.proposal_flow_id)
    return study, metadata


def _finished_trial_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state.is_finished())


def _shared_budget_state_path(study_metadata: dict[str, Any]) -> Path:
    storage_path = Path(str(study_metadata["storage_path"]))
    return storage_path.with_suffix(f"{storage_path.suffix}.budget.json")


def _reserve_shared_optuna_trial_slot(
    study: optuna.Study,
    *,
    target_trial_count: int,
    study_metadata: dict[str, Any],
) -> int | None:
    budget_path = _shared_budget_state_path(study_metadata)
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    with budget_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        state = json.loads(raw) if raw else {}
        finished_trial_count = _finished_trial_count(study)
        reserved_count = max(
            int(state.get("reserved_trial_count", 0)), finished_trial_count
        )
        if reserved_count >= target_trial_count:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return None
        slot_index = reserved_count
        state.update(
            {
                "reserved_trial_count": reserved_count + 1,
                "target_trial_count": target_trial_count,
                "updated_at": _now_iso(),
            }
        )
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(state, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return slot_index


def _optimize_study_with_resume(
    study: optuna.Study,
    *,
    objective,
    target_trial_count: int,
) -> dict[str, int]:
    existing_trial_count = len(study.trials)
    existing_finished_trial_count = _finished_trial_count(study)
    remaining_trial_count = max(target_trial_count - existing_finished_trial_count, 0)
    if remaining_trial_count > 0:
        study.optimize(
            objective,
            n_trials=remaining_trial_count,
            show_progress_bar=False,
            catch=(_OptunaTrialFailure,),
        )
    return {
        "requested_trial_count": target_trial_count,
        "existing_trial_count_before_optimize": existing_trial_count,
        "existing_finished_trial_count_before_optimize": (
            existing_finished_trial_count
        ),
        "remaining_trial_count": remaining_trial_count,
    }


def _parallel_optimize_study_with_shared_budget(
    study: optuna.Study,
    *,
    objective,
    target_trial_count: int,
    study_metadata: dict[str, Any],
) -> dict[str, int]:
    existing_trial_count = len(study.trials)
    existing_finished_trial_count = _finished_trial_count(study)
    reserved_slots = 0
    while True:
        slot_index = _reserve_shared_optuna_trial_slot(
            study,
            target_trial_count=target_trial_count,
            study_metadata=study_metadata,
        )
        if slot_index is None:
            break
        reserved_slots += 1
        study.optimize(
            objective,
            n_trials=1,
            show_progress_bar=False,
            catch=(_OptunaTrialFailure,),
        )
    return {
        "requested_trial_count": target_trial_count,
        "existing_trial_count_before_optimize": existing_trial_count,
        "existing_finished_trial_count_before_optimize": (
            existing_finished_trial_count
        ),
        "remaining_trial_count": max(
            target_trial_count - existing_finished_trial_count, 0
        ),
        "reserved_trial_slots": reserved_slots,
    }


def _require_complete_best_trial(
    study: optuna.Study, *, label: str
) -> optuna.FrozenTrial:
    if not any(trial.state == TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError(
            f"{label} finished without a successful Optuna trial; inspect study summary for failed/pruned states"
        )
    return study.best_trial


def _trial_dir_from_context(study_context: StudyContext, trial_number: int) -> Path:
    return study_context.study_root / "trials" / trial_dir_name(trial_number)


def _write_trial_result(
    trial_dir: Path,
    *,
    status: str,
    study_context: StudyContext,
    payload: dict[str, Any],
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "status": status,
        "study_index": study_context.study_index,
        "study_name": study_context.study_name,
        "proposal_flow_id": study_context.proposal_flow_id,
        **payload,
    }
    (trial_dir / "trial_result.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


def _main_job_objective(
    loaded: LoadedConfig,
    job: JobConfig,
    *,
    study_context: StudyContext,
    source_df: pd.DataFrame,
    freq: str,
    splits: list[tuple[list[int], list[int]]],
    progress: _ProgressLogger | None = None,
) -> Any:
    trial_count = optuna_num_trials(loaded.config.runtime.opt_n_trial)

    def objective(trial: optuna.Trial) -> float:
        trial_dir = _trial_dir_from_context(study_context, trial.number)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("study_index", study_context.study_index)
        trial.set_user_attr("sampler_seed", study_context.sampler_seed)
        trial.set_user_attr("proposal_flow_id", study_context.proposal_flow_id)
        candidate_params = suggest_model_params(
            job.model,
            job.selected_search_params,
            trial,
            param_specs=(
                None
                if loaded.search_space_payload is None
                else loaded.search_space_payload["models"][job.model]
            ),
        )
        candidate_params = _sanitize_aaforecast_trial_params(
            model_name=job.model,
            candidate_params=candidate_params,
        )
        candidate_training_params = suggest_training_params(
            loaded.config.training_search.selected_search_params,
            trial,
            model_name=job.model,
            param_specs=training_param_registry_for_model(
                job.model,
                search_space_payload=loaded.search_space_payload,
            ),
        )
        trial.set_user_attr("best_params", candidate_params)
        trial.set_user_attr("best_training_params", candidate_training_params)
        fold_mape: list[float] = []
        trial_prediction_frames: list[pd.DataFrame] = []
        _write_trial_result(
            trial_dir,
            status="running",
            study_context=study_context,
            payload={
                "trial_number": trial.number,
                "best_params": candidate_params,
                "best_training_params": candidate_training_params,
                "fold_mape": fold_mape,
            },
        )
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            phase = f"tune-trial-{trial.number + 1}/{trial_count}"
            if progress is not None:
                progress.fold_started(
                    fold_idx, total_folds=len(splits), phase=phase
                )
            try:
                fold_root = _trial_fold_artifact_dir(trial_dir, fold_idx)
                fit_kwargs: dict[str, Any] = {
                    "run_root": fold_root,
                    "source_df": source_df,
                    "freq": freq,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "params_override": candidate_params,
                    "training_override": candidate_training_params,
                    "fold_idx": fold_idx,
                }
                (
                    target_predictions,
                    target_actuals,
                    train_end_ds,
                    train_df,
                    nf,
                ) = _fit_and_predict_fold(
                    loaded,
                    job,
                    **fit_kwargs,
                )
                pred_col = _prediction_column(target_predictions, job.model)
                metrics = _compute_metrics(target_actuals, target_predictions[pred_col])
                metric = metrics["MAPE"]
                completed_fold_mape = fold_mape + [metric]
                interim_metric = _mean_fold_metric(
                    completed_fold_mape, metric_name="mape"
                )
                fold_prediction_frame = _write_trial_fold_artifacts(
                    trial_dir=trial_dir,
                    fold_root=fold_root,
                    model_name=job.model,
                    fold_idx=fold_idx,
                    train_end_ds=train_end_ds,
                    target_predictions=target_predictions,
                    target_actuals=target_actuals,
                    metrics=metrics,
                    fitted_model=nf,
                )
                trial_prediction_frames.append(fold_prediction_frame)
                pd.concat(trial_prediction_frames, ignore_index=True).to_csv(
                    trial_dir / TRIAL_PREDICTIONS_FILENAME,
                    index=False,
                )
                fold_mape.append(metric)
                trial.set_user_attr("fold_mape", fold_mape.copy())
                _write_trial_result(
                    trial_dir,
                    status="running",
                    study_context=study_context,
                    payload={
                        "trial_number": trial.number,
                        "best_params": candidate_params,
                        "best_training_params": candidate_training_params,
                        "fold_mape": fold_mape.copy(),
                        "last_completed_fold": fold_idx,
                        "interim_metric": interim_metric,
                    },
                )
                trial.report(interim_metric, step=fold_idx)
                if trial.should_prune():
                    trial.set_user_attr("pruned_after_fold", fold_idx)
                    if progress is not None:
                        progress.fold_completed(
                            fold_idx,
                            total_folds=len(splits),
                            phase=phase,
                            detail=f"pruned mean_mape={interim_metric:.4f}",
                        )
                    raise optuna.TrialPruned(
                        f"Pruned after fold {fold_idx} with mean_mape={interim_metric:.4f}"
                    )
                if progress is not None:
                    progress.fold_completed(
                        fold_idx,
                        total_folds=len(splits),
                        phase=phase,
                        detail=f"mape={metric:.4f}",
                    )
            except optuna.TrialPruned:
                _write_trial_result(
                    trial_dir,
                    status="pruned",
                    study_context=study_context,
                    payload={
                        "trial_number": trial.number,
                        "best_params": candidate_params,
                        "best_training_params": candidate_training_params,
                        "fold_mape": fold_mape,
                    },
                )
                raise
            except Exception as exc:
                if progress is not None:
                    progress.error(
                        fold_idx,
                        total_folds=len(splits),
                        phase=phase,
                        exc=exc,
                    )
                trial.set_user_attr(
                    "failure_reason",
                    f"fold={fold_idx} {type(exc).__name__}: {exc}",
                )
                _write_trial_result(
                    trial_dir,
                    status="failed",
                    study_context=study_context,
                    payload={
                        "trial_number": trial.number,
                        "best_params": candidate_params,
                        "best_training_params": candidate_training_params,
                        "fold_mape": fold_mape,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise _OptunaTrialFailure(
                    f"{job.model} tuning failed on fold {fold_idx}: {type(exc).__name__}: {exc}"
                ) from exc
        metric = _mean_fold_metric(fold_mape, metric_name="mape")
        trial.set_user_attr("fold_mape", fold_mape)
        _write_trial_result(
            trial_dir,
            status="complete",
            study_context=study_context,
            payload={
                "trial_number": trial.number,
                "best_params": candidate_params,
                "best_training_params": candidate_training_params,
                "fold_mape": fold_mape,
                "objective_value": metric,
            },
        )
        return metric

    return objective


def _sanitize_aaforecast_trial_params(
    *, model_name: str, candidate_params: dict[str, Any]
) -> dict[str, Any]:
    if model_name != "AAForecast":
        return candidate_params
    sanitized = dict(candidate_params)
    sanitized.pop("use_shape_key", None)
    if sanitized.get("use_event_key") is False:
        sanitized["use_event_key"] = True
    return sanitized


def _collect_main_tuning_result(
    study: optuna.Study,
    *,
    loaded: LoadedConfig,
    job: JobConfig,
    study_metadata: dict[str, Any],
    optimize_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    best_trial = _require_complete_best_trial(
        study, label=f"{job.model} main Optuna study"
    )
    best_params = _normalize_legacy_best_params_for_replay(
        job,
        dict(best_trial.user_attrs["best_params"]),
    )
    best_training_params = dict(best_trial.user_attrs["best_training_params"])
    summary = {
        **_trial_metrics_summary(study, objective_metric="mean_fold_mape"),
        **study_metadata,
        **optimize_metadata,
        "requested_mode": job.requested_mode,
        "validated_mode": job.validated_mode,
        "selected_search_params": list(job.selected_search_params),
        "selected_training_search_params": list(
            loaded.config.training_search.selected_search_params
        ),
        "training_range_source": training_range_source_for_model(
            job.model,
            search_space_payload=loaded.search_space_payload,
        ),
        "best_params": best_params,
        "best_training_params": best_training_params,
        "fold_mape": best_trial.user_attrs["fold_mape"],
        "objective_stage": _objective_stage_label(loaded),
    }
    return best_params, best_training_params, summary


def _normalize_legacy_best_params_for_replay(
    job: JobConfig, best_params: dict[str, Any]
) -> dict[str, Any]:
    if job.model != "AAForecast" or "top_k" not in best_params:
        return best_params
    normalized = dict(best_params)
    legacy_top_k = normalized.pop("top_k")
    normalized.setdefault("thresh", legacy_top_k)
    return normalized


def _tune_main_job(
    loaded: LoadedConfig,
    job: JobConfig,
    models_dir: Path,
    *,
    study_context: StudyContext,
    source_df: pd.DataFrame,
    freq: str,
    splits: list[tuple[list[int], list[int]]],
    progress: _ProgressLogger | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sampler = build_optuna_sampler(study_context.sampler_seed)
    objective = _main_job_objective(
        loaded,
        job,
        study_context=study_context,
        source_df=source_df,
        freq=freq,
        splits=splits,
        progress=progress,
    )
    study, study_metadata = _open_persistent_study(
        study_context,
        sampler=sampler,
    )
    optimize_metadata = _optimize_study_with_resume(
        study,
        objective=objective,
        target_trial_count=optuna_num_trials(loaded.config.runtime.opt_n_trial),
    )
    return _collect_main_tuning_result(
        study,
        loaded=loaded,
        job=job,
        study_metadata=study_metadata,
        optimize_metadata=optimize_metadata,
    )


def _parallel_tune_main_job_worker(
    loaded: LoadedConfig,
    job: JobConfig,
    models_dir: Path,
    *,
    study_context: StudyContext,
    source_df: pd.DataFrame,
    freq: str,
    splits: list[tuple[list[int], list[int]]],
    progress: _ProgressLogger | None = None,
) -> dict[str, Any]:
    worker_index = int(os.environ.get("NEURALFORECAST_OPTUNA_WORKER_INDEX", "0"))
    worker_study_context = replace(
        study_context,
        sampler_seed=study_context.sampler_seed + worker_index,
    )
    sampler = build_optuna_sampler(worker_study_context.sampler_seed)
    objective = _main_job_objective(
        loaded,
        job,
        study_context=worker_study_context,
        source_df=source_df,
        freq=freq,
        splits=splits,
        progress=progress,
    )
    study, study_metadata = _open_persistent_study(
        worker_study_context,
        sampler=sampler,
    )
    return _parallel_optimize_study_with_shared_budget(
        study,
        objective=objective,
        target_trial_count=optuna_num_trials(loaded.config.runtime.opt_n_trial),
        study_metadata=study_metadata,
    )


def _load_main_tuning_result(
    loaded: LoadedConfig,
    job: JobConfig,
    models_dir: Path,
    *,
    study_context: StudyContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sampler = build_optuna_sampler(study_context.sampler_seed)
    study, study_metadata = _open_persistent_study(
        study_context,
        sampler=sampler,
    )
    optimize_metadata = {
        "requested_trial_count": optuna_num_trials(loaded.config.runtime.opt_n_trial),
        "existing_trial_count_before_optimize": len(study.trials),
        "existing_finished_trial_count_before_optimize": _finished_trial_count(study),
        "remaining_trial_count": max(
            optuna_num_trials(loaded.config.runtime.opt_n_trial)
            - _finished_trial_count(study),
            0,
        ),
    }
    return _collect_main_tuning_result(
        study,
        loaded=loaded,
        job=job,
        study_metadata=study_metadata,
        optimize_metadata=optimize_metadata,
    )


def _copy_projection_file(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_stage_study_catalog(
    stage_root: Path,
    selection: StudySelection,
    *,
    summary_by_study: Mapping[int, Mapping[str, Any]] | None = None,
) -> Path:
    catalog_path = stage_root / "study_catalog.json"
    write_study_catalog(
        catalog_path,
        build_study_catalog_payload(
            stage_root,
            selection,
            study_summaries=summary_by_study,
        ),
    )
    return catalog_path


def _initialize_study_catalogs(
    run_root: Path,
    loaded: LoadedConfig,
    jobs: Sequence[JobConfig],
) -> None:
    selection = resolve_study_selection(loaded)
    for job in jobs:
        _write_stage_study_catalog(run_root / "models" / job.model, selection)


def _baseline_cross_validation(
    loaded: LoadedConfig,
    train_df: pd.DataFrame,
    splits: list[tuple[list[int], list[int]]],
    model_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics_rows = []
    forecast_rows = []
    values = train_df["y"].reset_index(drop=True)
    dates = pd.to_datetime(train_df["ds"]).reset_index(drop=True)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        history = values.iloc[train_idx].reset_index(drop=True)
        future_actual = values.iloc[test_idx].reset_index(drop=True)
        future_dates = dates.iloc[test_idx].reset_index(drop=True)
        train_end_ds = dates.iloc[train_idx[-1]]
        diff_context = _build_fold_diff_context(
            loaded,
            train_df.iloc[train_idx].reset_index(drop=True),
            target_col="y",
        )
        working_history = _transform_training_series(history, diff_context)
        if model_name == "Naive":
            pred_values = pd.Series(
                [float(working_history.iloc[-1])] * len(future_actual)
            )
        elif model_name == "SeasonalNaive":
            season = min(len(working_history), len(future_actual))
            tail = working_history.iloc[-season:].reset_index(drop=True)
            pred_values = pd.Series(
                (list(tail) * ((len(future_actual) // len(tail)) + 1))[
                    : len(future_actual)
                ]
            )
        else:
            pred_values = pd.Series(
                [float(working_history.mean())] * len(future_actual)
            )
        pred_values = _restore_prediction_series(pred_values, diff_context)
        metrics = _compute_metrics(future_actual, pred_values)
        metrics_rows.append(
            {"fold_idx": fold_idx, "cutoff": str(train_end_ds), **metrics}
        )
        for idx, ds in enumerate(future_dates):
            forecast_rows.append(
                {
                    "model": model_name,
                    "fold_idx": fold_idx,
                    "cutoff": str(train_end_ds),
                    "train_end_ds": str(train_end_ds),
                    "unique_id": train_df["unique_id"].iloc[0],
                    "ds": str(ds),
                    "horizon_step": idx + 1,
                    "y": float(future_actual.iloc[idx]),
                    "y_hat": float(pred_values.iloc[idx]),
                }
            )
    return metrics_rows, forecast_rows


def _predict_with_fitted_model(nf: NeuralForecast, adapter_inputs) -> pd.DataFrame:
    predict_kwargs = {
        "df": adapter_inputs.fit_df,
        "static_df": adapter_inputs.static_df,
    }
    if adapter_inputs.futr_df is not None:
        predict_kwargs["futr_df"] = adapter_inputs.futr_df
    return nf.predict(**predict_kwargs)


def _prediction_column(predictions: pd.DataFrame, model_name: str) -> str:
    if model_name in predictions.columns:
        return model_name
    raise KeyError(f"Could not find prediction column for {model_name}")


def _trial_fold_artifact_dir(trial_dir: Path, fold_idx: int) -> Path:
    return trial_dir / "folds" / f"fold_{fold_idx:03d}"


def _serialize_artifact_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _trial_prediction_extra_columns(
    prediction_frame: pd.DataFrame,
    row_idx: int,
    *,
    prediction_col: str,
) -> dict[str, object]:
    extras: dict[str, object] = {}
    ignored = {"unique_id", "ds", prediction_col}
    for column in prediction_frame.columns:
        column_name = str(column)
        if column_name in ignored:
            continue
        output_name = column_name
        if column_name.startswith(f"{prediction_col}__"):
            output_name = "y_hat_" + column_name.removeprefix(f"{prediction_col}__")
        extras[output_name] = _serialize_artifact_value(
            prediction_frame.iloc[row_idx][column]
        )
    return extras


def _trial_fold_prediction_frame(
    *,
    model_name: str,
    fold_idx: int,
    train_end_ds: pd.Timestamp,
    target_predictions: pd.DataFrame,
    target_actuals: pd.Series,
) -> pd.DataFrame:
    pred_col = _prediction_column(target_predictions, model_name)
    unique_id = (
        str(target_predictions["unique_id"].iloc[0])
        if "unique_id" in target_predictions.columns and not target_predictions.empty
        else ""
    )
    rows: list[dict[str, object]] = []
    for row_idx, ds in enumerate(target_predictions["ds"]):
        actual_value = target_actuals.iloc[row_idx]
        rows.append(
            {
                "model": model_name,
                "fold_idx": fold_idx,
                "train_end_ds": str(pd.Timestamp(train_end_ds)),
                "unique_id": unique_id,
                "ds": str(pd.Timestamp(ds)),
                "horizon_step": row_idx + 1,
                "y": None if pd.isna(actual_value) else float(actual_value),
                "y_hat": float(target_predictions[pred_col].iloc[row_idx]),
                **_trial_prediction_extra_columns(
                    target_predictions,
                    row_idx,
                    prediction_col=pred_col,
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_trial_fold_metrics_json(
    fold_root: Path,
    *,
    model_name: str,
    fold_idx: int,
    train_end_ds: pd.Timestamp,
    metrics: Mapping[str, float],
    row_count: int,
) -> Path:
    metrics_path = fold_root / TRIAL_FOLD_METRICS_FILENAME
    payload = {
        "model": model_name,
        "fold_idx": fold_idx,
        "train_end_ds": str(pd.Timestamp(train_end_ds)),
        "row_count": row_count,
        **{name: float(value) for name, value in metrics.items()},
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics_path


def _write_trial_fold_plot(
    fold_root: Path,
    *,
    model_name: str,
    target_predictions: pd.DataFrame,
    target_actuals: pd.Series,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_col = _prediction_column(target_predictions, model_name)
    ds = pd.to_datetime(target_predictions["ds"]).reset_index(drop=True)
    actual_series = target_actuals.reset_index(drop=True).astype(float)
    predicted_series = target_predictions[pred_col].reset_index(drop=True).astype(float)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(ds, actual_series, label="actual", linewidth=1.8, linestyle="--")
    axis.plot(ds, predicted_series, label=model_name, linewidth=1.8)
    axis.set_title(f"{model_name} fold predictions")
    axis.set_xlabel("ds")
    axis.set_ylabel("y")
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()
    plot_path = fold_root / TRIAL_FOLD_PLOT_FILENAME
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def _write_trial_fold_checkpoint(
    fold_root: Path,
    *,
    fitted_model: Any | None,
) -> Path:
    checkpoint_path = fold_root / TRIAL_FOLD_CHECKPOINT_FILENAME
    if fitted_model is None:
        raise TypeError("trial fold checkpoint export requires a fitted model object")
    models = getattr(fitted_model, "models", None)
    if models and hasattr(models[0], "save"):
        models[0].save(str(checkpoint_path))
        return checkpoint_path
    if hasattr(fitted_model, "save"):
        fitted_model.save(str(checkpoint_path))
        return checkpoint_path
    raise TypeError(
        f"trial fold checkpoint export does not support {type(fitted_model).__name__}"
    )


def _write_trial_fold_artifacts(
    *,
    trial_dir: Path,
    fold_root: Path,
    model_name: str,
    fold_idx: int,
    train_end_ds: pd.Timestamp,
    target_predictions: pd.DataFrame,
    target_actuals: pd.Series,
    metrics: Mapping[str, float],
    fitted_model: Any | None,
) -> pd.DataFrame:
    del trial_dir
    fold_root.mkdir(parents=True, exist_ok=True)
    prediction_frame = _trial_fold_prediction_frame(
        model_name=model_name,
        fold_idx=fold_idx,
        train_end_ds=train_end_ds,
        target_predictions=target_predictions,
        target_actuals=target_actuals,
    )
    prediction_frame.to_csv(fold_root / TRIAL_PREDICTIONS_FILENAME, index=False)
    _write_trial_fold_plot(
        fold_root,
        model_name=model_name,
        target_predictions=target_predictions,
        target_actuals=target_actuals,
    )
    _write_trial_fold_metrics_json(
        fold_root,
        model_name=model_name,
        fold_idx=fold_idx,
        train_end_ds=train_end_ds,
        metrics=metrics,
        row_count=len(prediction_frame),
    )
    _write_trial_fold_checkpoint(fold_root, fitted_model=fitted_model)
    return prediction_frame


def _artifact_model_name(path: Path, suffix: str) -> str:
    return path.name.removesuffix(suffix)


def _summary_job_roots(run_root: Path) -> list[Path]:
    roots: list[Path] = []
    if (run_root / "cv").exists():
        roots.append(run_root)
    workers_root = run_root / "scheduler" / "workers"
    if workers_root.exists():
        roots.extend(
            path for path in sorted(workers_root.iterdir()) if (path / "cv").exists()
        )
    return roots


def _load_metrics_for_summary(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in _summary_job_roots(run_root):
        for path in sorted((root / "cv").glob("*_metrics_by_cutoff.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            model_name = _artifact_model_name(path, "_metrics_by_cutoff.csv")
            if "model" not in frame.columns:
                frame["model"] = model_name
            rows.extend(frame.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _normalize_summary_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    return timestamp


def _normalize_summary_window_frame(
    frame: pd.DataFrame, *, frame_name: str
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "fold_idx" not in frame.columns:
        raise ValueError(f"{frame_name} must contain a fold_idx column")
    if "cutoff" not in frame.columns:
        raise ValueError(f"{frame_name} must contain a cutoff column")
    normalized = frame.copy()
    normalized["fold_idx"] = pd.to_numeric(normalized["fold_idx"], errors="coerce")
    if normalized["fold_idx"].isna().any():
        raise ValueError(f"{frame_name} contains invalid fold_idx values")
    normalized["fold_idx"] = normalized["fold_idx"].astype(int)

    try:
        normalized["normalized_cutoff"] = normalized["cutoff"].map(
            _normalize_summary_timestamp
        )
    except Exception as exc:
        raise ValueError(f"{frame_name} contains invalid cutoff values") from exc
    if normalized["normalized_cutoff"].isna().any():
        raise ValueError(f"{frame_name} contains invalid cutoff values")
    return normalized


def _normalize_summary_metrics_frame(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_summary_window_frame(
        metrics_frame, frame_name="summary metrics"
    )


def _build_leaderboard(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    if metrics_frame.empty or "model" not in metrics_frame.columns:
        columns = [
            "rank",
            "model",
            "mean_fold_mae",
            "mean_fold_mse",
            "mean_fold_rmse",
            "fold_count",
            "mean_fold_mape",
            "mean_fold_nrmse",
            "mean_fold_r2",
        ]
        return pd.DataFrame(columns=columns)

    def _safe_mean(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return float(values.mean())

    agg_fields: dict[str, tuple[str, object]] = {
        "mean_fold_mae": ("MAE", "mean"),
        "mean_fold_mse": ("MSE", "mean"),
        "mean_fold_rmse": ("RMSE", "mean"),
        "fold_count": ("fold_idx", "nunique"),
    }
    if "MAPE" in metrics_frame.columns:
        agg_fields["mean_fold_mape"] = ("MAPE", _safe_mean)
    if "NRMSE" in metrics_frame.columns:
        agg_fields["mean_fold_nrmse"] = ("NRMSE", _safe_mean)
    if "R2" in metrics_frame.columns:
        agg_fields["mean_fold_r2"] = ("R2", _safe_mean)
    sort_fields = ["mean_fold_nrmse", "mean_fold_mse", "mean_fold_mae", "model"]
    active_sort_fields = [field for field in sort_fields if field in agg_fields or field == "model"]
    ascending = [True] * (len(active_sort_fields) - 1) + [True]
    leaderboard = (
        metrics_frame.groupby("model", as_index=False)
        .agg(**agg_fields)
        .sort_values(
            by=active_sort_fields,
            ascending=ascending,
            kind="stable",
        )
        .reset_index(drop=True)
    )
    leaderboard.insert(0, "rank", leaderboard.index + 1)
    return leaderboard


def _write_leaderboard_workbook(leaderboard: pd.DataFrame, workbook_path: Path) -> None:
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(workbook_path, index=False)


def _load_last_fold_forecasts(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in _summary_job_roots(run_root):
        for path in sorted((root / "cv").glob("*_forecasts.csv")):
            if path.name.endswith("_rolling_origin_forecasts.csv"):
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            model_name = _artifact_model_name(path, "_forecasts.csv")
            if "model" not in frame.columns:
                frame["model"] = model_name
            frame["_summary_source_root"] = str(root)
            frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="coerce")
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts = forecasts.dropna(subset=["fold_idx"]).copy()
    forecasts["fold_idx"] = forecasts["fold_idx"].astype(int)
    return forecasts


def _write_summary_results_csv(forecasts: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [
        "model",
        "requested_mode",
        "validated_mode",
        "fold_idx",
        "cutoff",
        "train_end_ds",
        "unique_id",
        "ds",
        "horizon_step",
        "y",
        "y_hat",
    ]
    result_frame = forecasts.copy()
    if "_summary_source_root" in result_frame.columns:
        result_frame = result_frame.drop(columns=["_summary_source_root"])
    if "normalized_cutoff" in result_frame.columns:
        result_frame = result_frame.drop(columns=["normalized_cutoff"])
    sort_columns = [
        column
        for column in ("model", "fold_idx", "cutoff", "ds", "horizon_step")
        if column in result_frame.columns
    ]
    if sort_columns:
        result_frame = result_frame.sort_values(sort_columns, kind="stable").reset_index(
            drop=True
        )
    front_columns = [column for column in ordered_columns if column in result_frame.columns]
    remaining_columns = [
        column for column in result_frame.columns if column not in front_columns
    ]
    result_frame = result_frame[front_columns + remaining_columns]
    result_frame.to_csv(output_path, index=False)
    return output_path


def _write_summary_metrics_csv(metrics: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_frame = metrics.copy()
    if "normalized_cutoff" in metric_frame.columns:
        metric_frame = metric_frame.drop(columns=["normalized_cutoff"])
    ordered_columns = [
        "model",
        "fold_idx",
        "cutoff",
        "MAE",
        "MSE",
        "RMSE",
        "MAPE",
        "NRMSE",
        "R2",
    ]
    sort_columns = [
        column for column in ("model", "fold_idx", "cutoff") if column in metric_frame.columns
    ]
    if sort_columns:
        metric_frame = metric_frame.sort_values(sort_columns, kind="stable").reset_index(
            drop=True
        )
    front_columns = [column for column in ordered_columns if column in metric_frame.columns]
    remaining_columns = [
        column for column in metric_frame.columns if column not in front_columns
    ]
    metric_frame = metric_frame[front_columns + remaining_columns]
    metric_frame.to_csv(output_path, index=False)
    return output_path


def _build_summary_plot_bundle(
    run_root: Path,
    summary_dir: Path,
    leaderboard: pd.DataFrame,
    scoped_forecasts: pd.DataFrame,
    *,
    title_prefix: str,
    include_future_only_predictions: bool = False,
) -> dict[str, str]:
    plot_paths: dict[str, str] = {}
    if leaderboard.empty:
        ordered_models = list(dict.fromkeys(scoped_forecasts["model"].tolist()))
    else:
        ordered_models = [
            model
            for model in leaderboard["model"].tolist()
            if model in set(scoped_forecasts["model"])
        ]
    plot_specs = [
        ("all_models", ordered_models, f"{title_prefix} (all models)", None),
        (
            "all_models_window_16",
            ordered_models,
            f"{title_prefix} (all models, input window=16)",
            16,
        ),
    ]
    for slug, models, title, history_steps_override in plot_specs:
        if not models:
            continue
        plot_path = summary_dir / f"last_fold_{slug}.png"
        _plot_last_fold_overlay(
            scoped_forecasts,
            models,
            plot_path,
            title=title,
            run_root=run_root,
            history_steps_override=history_steps_override,
            include_future_only_predictions=include_future_only_predictions,
        )
        plot_paths[slug] = str(plot_path)
    return plot_paths


def _summary_fold_dir(summary_dir: Path, fold_idx: int) -> Path:
    return summary_dir / "folds" / f"fold_{fold_idx:03d}"


def _write_summary_fold_plot(
    run_root: Path,
    fold_root: Path,
    fold_forecasts: pd.DataFrame,
    selected_models: list[str],
    *,
    fold_idx: int,
) -> Path:
    plot_path = fold_root / "plot.png"
    _plot_last_fold_overlay(
        fold_forecasts,
        selected_models,
        plot_path,
        title=f"Fold {fold_idx:03d} predictions (all models)",
        run_root=run_root,
    )
    return plot_path


def _write_per_fold_summary_bundles(
    run_root: Path,
    summary_dir: Path,
    leaderboard: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> Path | None:
    if forecasts.empty:
        return None
    normalized_forecasts = _normalize_summary_window_frame(
        forecasts, frame_name="summary forecasts"
    )
    normalized_metrics = _normalize_summary_metrics_frame(metrics_frame)
    forecast_fold_indices = sorted(normalized_forecasts["fold_idx"].unique().tolist())
    metric_fold_indices = sorted(normalized_metrics["fold_idx"].unique().tolist())
    if forecast_fold_indices != metric_fold_indices:
        raise ValueError(
            "summary per-fold bundle generation requires matching forecast/metric fold coverage"
        )

    folds_root = summary_dir / "folds"
    folds_root.mkdir(parents=True, exist_ok=True)
    leaderboard_models = leaderboard["model"].tolist() if "model" in leaderboard.columns else []
    for fold_idx in forecast_fold_indices:
        fold_root = _summary_fold_dir(summary_dir, fold_idx)
        fold_root.mkdir(parents=True, exist_ok=True)
        fold_forecasts = normalized_forecasts[
            normalized_forecasts["fold_idx"] == fold_idx
        ].copy()
        fold_metrics = normalized_metrics[
            normalized_metrics["fold_idx"] == fold_idx
        ].copy()
        if fold_forecasts.empty:
            raise ValueError(
                f"summary per-fold bundle generation requires forecast rows for fold {fold_idx}"
            )
        if fold_metrics.empty:
            raise ValueError(
                f"summary per-fold bundle generation requires metric rows for fold {fold_idx}"
            )
        _write_summary_results_csv(fold_forecasts, fold_root / "predictions.csv")
        _write_summary_metrics_csv(fold_metrics, fold_root / "metrics.csv")
        fold_model_set = set(fold_forecasts["model"].tolist())
        ordered_models = [model for model in leaderboard_models if model in fold_model_set]
        ordered_models.extend(
            model
            for model in dict.fromkeys(fold_forecasts["model"].tolist())
            if model not in set(ordered_models)
        )
        _write_summary_fold_plot(
            run_root,
            fold_root,
            fold_forecasts,
            ordered_models,
            fold_idx=fold_idx,
        )
    return folds_root


def _write_summary_bundle(
    run_root: Path,
    summary_dir: Path,
    metrics_frame: pd.DataFrame,
    *,
    scoped_forecasts: pd.DataFrame | None = None,
    title_prefix: str,
    include_future_only_predictions: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    leaderboard = _build_leaderboard(metrics_frame)
    workbook_path = summary_dir / "leaderboard.csv"
    _write_leaderboard_workbook(leaderboard, workbook_path)
    artifact_paths: dict[str, str] = {
        "leaderboard": str(workbook_path),
    }
    if scoped_forecasts is not None and not scoped_forecasts.empty:
        artifact_paths.update(
            _build_summary_plot_bundle(
                run_root,
                summary_dir,
                leaderboard,
                scoped_forecasts,
                title_prefix=title_prefix,
                include_future_only_predictions=include_future_only_predictions,
            )
        )
    return leaderboard, artifact_paths


def _load_summary_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "manifest" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"summary generation requires {manifest_path} to exist"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _summary_repo_root(config_source_path: str | os.PathLike[str]) -> Path:
    config_path = Path(config_source_path).expanduser().resolve()
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _load_summary_loaded_config(run_root: Path) -> LoadedConfig:
    manifest = _load_summary_manifest(run_root)
    config_source_path = manifest.get("config_source_path")
    if not config_source_path:
        raise ValueError(
            "summary generation requires manifest.config_source_path to be set"
        )
    repo_root = _summary_repo_root(config_source_path)
    return load_app_config(repo_root, config_path=str(config_source_path))


def _summary_overlay_actual_frames(
    run_root: Path,
    forecasts: pd.DataFrame,
    *,
    history_steps_override: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if forecasts.empty or "train_end_ds" not in forecasts.columns:
        return pd.DataFrame(), pd.DataFrame()
    loaded = _load_summary_loaded_config(run_root)
    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col
    if history_steps_override is None:
        history_steps = int(getattr(loaded.config.training, "input_size", 0) or 0)
    else:
        history_steps = int(history_steps_override)
    if history_steps < 1:
        return pd.DataFrame(), pd.DataFrame()
    source_df = pd.read_csv(loaded.config.dataset.path)
    if source_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    source_df = source_df.sort_values(dt_col).reset_index(drop=True)
    normalized_ds = pd.Index(
        pd.to_datetime(source_df[dt_col]).map(_normalize_summary_timestamp)
    )
    train_end_values = (
        pd.Series(forecasts["train_end_ds"])
        .dropna()
        .map(_normalize_summary_timestamp)
        .dropna()
        .drop_duplicates()
    )
    if train_end_values.empty:
        return pd.DataFrame(), pd.DataFrame()
    if len(train_end_values) != 1:
        raise ValueError(
            "summary overlay requires a single train_end_ds value for the scoped forecast frame"
        )
    train_end_ds = train_end_values.iloc[0]
    matching = np.flatnonzero(normalized_ds == train_end_ds)
    if len(matching) == 0:
        raise ValueError(
            f"summary overlay could not locate train_end_ds {train_end_ds} in source dataset"
        )
    anchor_idx = int(matching[-1])
    history_start = max(0, anchor_idx - history_steps + 1)
    history_frame = source_df.iloc[history_start : anchor_idx + 1][[dt_col, target_col]].copy()
    history_frame.rename(columns={dt_col: "ds", target_col: "y"}, inplace=True)
    history_frame["ds"] = pd.Index(
        pd.to_datetime(history_frame["ds"]).map(_normalize_summary_timestamp)
    )

    output_frame = forecasts[["ds", "y"]].copy()
    output_frame["ds"] = pd.Index(
        pd.to_datetime(output_frame["ds"]).map(_normalize_summary_timestamp)
    )
    output_frame["y"] = pd.to_numeric(output_frame["y"], errors="coerce")
    output_frame = (
        output_frame.drop_duplicates(subset=["ds"])
        .sort_values("ds", kind="stable")
        .reset_index(drop=True)
    )
    return history_frame.reset_index(drop=True), output_frame


def _plot_last_fold_overlay(
    forecasts: pd.DataFrame,
    selected_models: list[str],
    plot_path: Path,
    *,
    title: str,
    run_root: Path,
    history_steps_override: int | None = None,
    include_future_only_predictions: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _normalize_optional_path(
        value: object, *, source_root: Path | None
    ) -> Path | None:
        if value is None or pd.isna(value):
            return None
        candidate = Path(str(value))
        if candidate.is_absolute():
            return candidate
        if source_root is not None:
            return source_root / candidate
        return run_root / candidate

    def _load_aaforecast_context_frame(selected_frame: pd.DataFrame) -> pd.DataFrame | None:
        if "aaforecast_context_artifact" not in selected_frame.columns:
            return None
        aaf_frame = selected_frame[selected_frame["model"] == "AAForecast"].copy()
        if aaf_frame.empty:
            return None
        artifact_paths: list[Path] = []
        for row in aaf_frame.to_dict(orient="records"):
            source_root = None
            if row.get("_summary_source_root"):
                source_root = Path(str(row["_summary_source_root"]))
            resolved = _normalize_optional_path(
                row.get("aaforecast_context_artifact"),
                source_root=source_root,
            )
            if resolved is not None:
                artifact_paths.append(resolved)
        if not artifact_paths:
            return None
        unique_paths = list(dict.fromkeys(artifact_paths))
        if len(unique_paths) != 1:
            raise ValueError(
                "AAForecast summary plot requires a single context artifact per scoped window"
            )
        context_path = unique_paths[0]
        if not context_path.exists():
            raise FileNotFoundError(
                f"AAForecast context artifact does not exist: {context_path}"
            )
        context_frame = pd.read_csv(context_path)
        required_columns = {"ds", "context_active"}
        if not required_columns.issubset(context_frame.columns):
            raise ValueError(
                "AAForecast context artifact must contain ds and context_active columns"
            )
        context_frame = context_frame.copy()
        context_frame["ds"] = pd.Index(
            [pd.Timestamp(value) for value in context_frame["ds"]]
        )
        context_frame["context_active"] = pd.to_numeric(
            context_frame["context_active"], errors="coerce"
        ).fillna(0)
        return context_frame

    def _selected_train_end_ds(selected_frame: pd.DataFrame) -> pd.Timestamp | None:
        if "train_end_ds" not in selected_frame.columns:
            return None
        train_end_values = (
            pd.Series(selected_frame["train_end_ds"])
            .dropna()
            .map(_normalize_summary_timestamp)
            .dropna()
            .drop_duplicates()
        )
        if train_end_values.empty:
            return None
        if len(train_end_values) != 1:
            raise ValueError(
                "summary plot requires a single train_end_ds value for the scoped forecast frame"
            )
        return pd.Timestamp(train_end_values.iloc[0])

    def _anchor_point(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["ds", "y"])
        return frame.tail(1)[["ds", "y"]].copy()

    def _connected_plot_frame(
        anchor_frame: pd.DataFrame,
        frame: pd.DataFrame,
        *,
        value_col: str,
    ) -> pd.DataFrame:
        plot_frame = frame[["ds", value_col]].copy()
        plot_frame[value_col] = pd.to_numeric(plot_frame[value_col], errors="coerce")
        plot_frame = plot_frame.dropna(subset=[value_col]).reset_index(drop=True)
        if anchor_frame.empty or plot_frame.empty:
            return plot_frame
        anchor = anchor_frame.rename(columns={"y": value_col})
        return pd.concat([anchor, plot_frame], ignore_index=True)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    selected = forecasts[forecasts["model"].isin(selected_models)].copy()
    if selected.empty:
        return
    selected["ds"] = pd.Index([pd.Timestamp(value) for value in selected["ds"]])
    try:
        input_actual_frame, output_actual_frame = _summary_overlay_actual_frames(
            run_root,
            selected,
            history_steps_override=history_steps_override,
        )
    except FileNotFoundError:
        input_actual_frame = pd.DataFrame()
        output_actual_frame = pd.DataFrame()
    actual_anchor_frame = _anchor_point(input_actual_frame)
    context_frame = _load_aaforecast_context_frame(selected)
    train_end_ds = _selected_train_end_ds(selected)
    if context_frame is not None:
        if train_end_ds is not None:
            context_frame = context_frame[
                context_frame["ds"] <= train_end_ds
            ].reset_index(drop=True)
        if not input_actual_frame.empty:
            input_start = input_actual_frame["ds"].min()
            input_end = input_actual_frame["ds"].max()
            context_frame = context_frame[
                (context_frame["ds"] >= input_start) & (context_frame["ds"] <= input_end)
            ].reset_index(drop=True)
    if context_frame is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        price_ax = ax
        context_ax = None
    else:
        fig, (price_ax, context_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
    if not input_actual_frame.empty:
        price_ax.plot(
            input_actual_frame["ds"],
            input_actual_frame["y"],
            label="actual (input)",
            linewidth=2.0,
            color="black",
        )
    if not output_actual_frame.empty and output_actual_frame["y"].notna().any():
        observed_output = _connected_plot_frame(
            actual_anchor_frame,
            output_actual_frame,
            value_col="y",
        )
        price_ax.plot(
            observed_output["ds"],
            observed_output["y"],
            label="actual (output)",
            linewidth=1.8,
            linestyle="--",
            color="dimgray",
        )
    for model_name in selected_models:
        model_frame = (
            selected[selected["model"] == model_name]
            .sort_values("ds")
            .reset_index(drop=True)
        )
        if model_frame.empty:
            continue
        if "y" in model_frame.columns and not include_future_only_predictions:
            model_frame = model_frame[model_frame["y"].notna()].reset_index(drop=True)
        if model_frame.empty:
            continue
        connected_model_frame = _connected_plot_frame(
            actual_anchor_frame,
            model_frame,
            value_col="y_hat",
        )
        if connected_model_frame.empty:
            continue
        prediction_point_indices = list(range(1, len(connected_model_frame)))
        price_ax.plot(
            connected_model_frame["ds"],
            connected_model_frame["y_hat"],
            label=model_name,
            linewidth=1.8,
            marker="o",
            markersize=5,
            markevery=prediction_point_indices if prediction_point_indices else None,
        )
    price_ax.set_title(title)
    price_ax.set_ylabel("y")
    price_ax.legend(loc="best")
    if context_ax is None:
        price_ax.set_xlabel("ds")
    else:
        context_ax.step(
            context_frame["ds"],
            context_frame["context_active"],
            where="mid",
            label="AAForecast anomaly context",
            linewidth=1.6,
        )
        context_ax.set_ylabel("context")
        context_ax.set_xlabel("context ds")
        context_ax.set_yticks([0, 1])
        context_ax.set_yticklabels(["normal", "anomaly"])
        context_ax.set_ylim(-0.05, 1.05)
        context_ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _learned_fold_artifact_dir(run_root: Path, model_name: str, fold_idx: int) -> Path:
    return run_root / "models" / model_name / "folds" / f"fold_{fold_idx:03d}"


def _normalize_curve_frame(curve_frame: pd.DataFrame) -> pd.DataFrame:
    if curve_frame.empty:
        return pd.DataFrame(columns=["global_step", "train_loss", "val_loss"])
    normalized = curve_frame.copy()
    if "global_step" not in normalized.columns:
        raise ValueError("loss curve frame must contain a global_step column")
    if "train_loss" not in normalized.columns:
        raise ValueError("loss curve frame must contain a train_loss column")
    if "val_loss" not in normalized.columns:
        normalized["val_loss"] = np.nan
    normalized["global_step"] = pd.to_numeric(
        normalized["global_step"], errors="coerce"
    )
    normalized["train_loss"] = pd.to_numeric(
        normalized["train_loss"], errors="coerce"
    )
    normalized["val_loss"] = pd.to_numeric(normalized["val_loss"], errors="coerce")
    normalized = normalized.dropna(subset=["global_step"]).copy()
    normalized["global_step"] = normalized["global_step"].astype(int)
    return normalized[["global_step", "train_loss", "val_loss"]].sort_values(
        "global_step", kind="stable"
    ).reset_index(drop=True)


def _trajectory_frame(nf: Any) -> pd.DataFrame:
    if nf is None:
        return pd.DataFrame()
    if isinstance(nf, pd.DataFrame):
        return _normalize_curve_frame(nf)
    if hasattr(nf, "curve_frame"):
        curve_frame = getattr(nf, "curve_frame")
        if isinstance(curve_frame, pd.DataFrame):
            return _normalize_curve_frame(curve_frame)
    models = getattr(nf, "models", None)
    if not models:
        return pd.DataFrame()
    model = models[0]
    train_points = list(getattr(model, "train_trajectories", []))
    val_points = list(getattr(model, "valid_trajectories", []))
    if not train_points or not val_points:
        return pd.DataFrame()

    train_frame = pd.DataFrame(train_points, columns=["global_step", "train_loss"])
    val_frame = pd.DataFrame(val_points, columns=["global_step", "val_loss"])
    merged = train_frame.merge(val_frame, on="global_step", how="outer").sort_values(
        "global_step", kind="stable"
    )
    return _normalize_curve_frame(merged)


def _loss_curve_series(
    curve_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_series = (
        curve_frame[["global_step", "train_loss"]]
        .dropna(subset=["train_loss"])
        .reset_index(drop=True)
    )
    train_smoothed_column = (
        "train_loss_smoothed"
        if "train_loss_smoothed" in curve_frame.columns
        else "train_loss"
    )
    train_smoothed_series = (
        curve_frame[["global_step", train_smoothed_column]]
        .dropna(subset=[train_smoothed_column])
        .rename(columns={train_smoothed_column: "train_loss_smoothed"})
        .reset_index(drop=True)
    )
    val_series = (
        curve_frame[["global_step", "val_loss"]]
        .dropna(subset=["val_loss"])
        .reset_index(drop=True)
    )
    return train_series, train_smoothed_series, val_series


def _curve_frame_with_smoothed_train(
    curve_frame: pd.DataFrame,
    *,
    smoothing_window: int = LOSS_CURVE_TRAIN_SMOOTHING_WINDOW,
) -> pd.DataFrame:
    normalized = _normalize_curve_frame(curve_frame)
    if normalized.empty:
        return normalized.assign(train_loss_smoothed=pd.Series(dtype=float))
    window = max(1, int(smoothing_window))
    enriched = normalized.copy()
    enriched["train_loss_smoothed"] = (
        enriched["train_loss"].rolling(window=window, min_periods=1).mean()
    )
    return enriched[
        ["global_step", "train_loss", "train_loss_smoothed", "val_loss"]
    ].reset_index(drop=True)


def _sample_loss_curve_frame(
    curve_frame: pd.DataFrame,
    *,
    every_n_steps: int = LOSS_CURVE_SAMPLE_EVERY_N_STEPS,
) -> pd.DataFrame:
    if curve_frame.empty:
        return curve_frame.copy()
    sampled = curve_frame.copy()
    sampled["global_step"] = pd.to_numeric(sampled["global_step"], errors="coerce")
    sampled = sampled.dropna(subset=["global_step"]).copy()
    sampled["global_step"] = sampled["global_step"].astype(int)
    sampled = sampled[sampled["global_step"] % every_n_steps == 0].reset_index(drop=True)
    return sampled


def _validation_aligned_curve_frame(curve_frame: pd.DataFrame) -> pd.DataFrame:
    if curve_frame.empty:
        return curve_frame.copy()
    aligned = curve_frame[curve_frame["val_loss"].notna()].copy()
    columns = ["global_step", "train_loss", "val_loss"]
    if "train_loss_smoothed" in aligned.columns:
        columns.insert(2, "train_loss_smoothed")
    return aligned[columns].reset_index(drop=True)


def _configure_loss_curve_axis(axis: Any, curve_frame: pd.DataFrame) -> None:
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    loss_values = curve_frame[["train_loss", "val_loss"]].to_numpy(dtype=float).ravel()
    positive_finite_mask = np.isfinite(loss_values) & (loss_values > 0)
    if not positive_finite_mask.any():
        return

    axis.set_yscale("log", base=10)
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))


def _write_loss_curve_artifact(
    run_root: Path,
    model_name: str,
    fold_idx: int,
    *,
    nf: NeuralForecast | None,
) -> Path | None:
    if nf is None:
        return None
    curve_frame = _curve_frame_with_smoothed_train(_trajectory_frame(nf))
    if curve_frame.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fold_root = _learned_fold_artifact_dir(run_root, model_name, fold_idx)
    fold_root.mkdir(parents=True, exist_ok=True)
    figure_path = fold_root / LOSS_CURVE_PLOT_FILENAME
    sampled_curve_path = fold_root / LOSS_CURVE_SAMPLE_FILENAME
    smoothed_curve_path = fold_root / LOSS_CURVE_SMOOTHED_SAMPLE_FILENAME
    validation_curve_path = fold_root / LOSS_CURVE_VALIDATION_FILENAME
    sampled_curve = _sample_loss_curve_frame(curve_frame)
    sampled_curve[["global_step", "train_loss", "val_loss"]].to_csv(
        sampled_curve_path, index=False
    )
    sampled_curve[
        ["global_step", "train_loss", "train_loss_smoothed", "val_loss"]
    ].to_csv(smoothed_curve_path, index=False)
    _validation_aligned_curve_frame(curve_frame).to_csv(validation_curve_path, index=False)
    train_series, train_smoothed_series, val_series = _loss_curve_series(curve_frame)
    figure, axis = plt.subplots(figsize=(10, 5))
    if not train_series.empty:
        axis.plot(
            train_series["global_step"],
            train_series["train_loss"],
            label="train_loss_raw",
            linewidth=1.0,
            alpha=0.3,
        )
    axis.plot(
        train_smoothed_series["global_step"],
        train_smoothed_series["train_loss_smoothed"],
        label=f"train_loss_smoothed_w{LOSS_CURVE_TRAIN_SMOOTHING_WINDOW}",
        linewidth=2.0,
        zorder=2,
    )
    axis.plot(
        val_series["global_step"],
        val_series["val_loss"],
        label="val_loss",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        zorder=3,
    )
    axis.set_title(f"{model_name} fold {fold_idx:03d} loss curve")
    axis.set_xlabel("global_step")
    axis.set_ylabel("loss")
    _configure_loss_curve_axis(axis, curve_frame)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path


def _should_build_summary_artifacts() -> bool:
    return os.environ.get("NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS") != "1"


def _prune_legacy_per_window_summary_artifacts(summary_dir: Path) -> None:
    for path in summary_dir.glob("test_*"):
        _remove_existing_artifact(path)


def _prune_legacy_summary_markdown(summary_dir: Path) -> None:
    _remove_existing_artifact(summary_dir / "sample.md")


def _build_summary_artifacts(run_root: Path) -> dict[str, str]:
    summary_dir = run_root / "summary"
    _prune_legacy_per_window_summary_artifacts(summary_dir)
    _prune_legacy_summary_markdown(summary_dir)
    metrics = _load_metrics_for_summary(run_root)
    if metrics.empty:
        return {}
    forecasts = _load_last_fold_forecasts(run_root)
    leaderboard, plot_paths = _write_summary_bundle(
        run_root,
        summary_dir,
        metrics,
        title_prefix="Last fold predictions",
    )
    loss_artifact_summary_path = _write_loss_artifact_summary(run_root)
    if loss_artifact_summary_path is not None:
        plot_paths["loss_artifacts"] = str(loss_artifact_summary_path)
    if forecasts.empty:
        return plot_paths
    result_path = _write_summary_results_csv(
        forecasts,
        summary_dir / SUMMARY_RESULTS_FILENAME,
    )
    plot_paths["result"] = str(result_path)
    normalized_forecasts = _normalize_summary_window_frame(
        forecasts, frame_name="summary forecasts"
    )
    last_fold = int(normalized_forecasts["fold_idx"].max())
    last_fold_forecasts = normalized_forecasts[
        normalized_forecasts["fold_idx"] == last_fold
    ].copy()
    plot_paths.update(
        _build_summary_plot_bundle(
            run_root,
            summary_dir,
            leaderboard,
            last_fold_forecasts,
            title_prefix="Last fold predictions",
        )
    )
    fold_bundles_root = _write_per_fold_summary_bundles(
        run_root,
        summary_dir,
        leaderboard,
        metrics,
        forecasts,
    )
    if fold_bundles_root is not None:
        plot_paths["fold_bundles_root"] = str(fold_bundles_root)
    return plot_paths


def _load_loss_artifacts_for_summary(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    artifact_specs = (
        {
            "curve_variant": "raw_sampled",
            "filename": LOSS_CURVE_SAMPLE_FILENAME,
            "source_granularity": "step_sampled",
            "sample_every_n_steps": LOSS_CURVE_SAMPLE_EVERY_N_STEPS,
            "smoothing_window": None,
        },
        {
            "curve_variant": "smoothed_sampled",
            "filename": LOSS_CURVE_SMOOTHED_SAMPLE_FILENAME,
            "source_granularity": "step_sampled",
            "sample_every_n_steps": LOSS_CURVE_SAMPLE_EVERY_N_STEPS,
            "smoothing_window": LOSS_CURVE_TRAIN_SMOOTHING_WINDOW,
        },
        {
            "curve_variant": "validation_aligned",
            "filename": LOSS_CURVE_VALIDATION_FILENAME,
            "source_granularity": "validation",
            "sample_every_n_steps": None,
            "smoothing_window": LOSS_CURVE_TRAIN_SMOOTHING_WINDOW,
        },
    )
    for root in _summary_job_roots(run_root):
        for spec in artifact_specs:
            for csv_path in sorted(root.glob(f"models/*/folds/fold_*/{spec['filename']}")):
                sampled = pd.read_csv(csv_path)
                if "global_step" in sampled.columns:
                    valid_steps = pd.to_numeric(
                        sampled["global_step"], errors="coerce"
                    ).dropna()
                else:
                    valid_steps = pd.Series(dtype=float)
                rows.append(
                    {
                        "model": csv_path.parents[2].name,
                        "fold_idx": int(csv_path.parent.name.removeprefix("fold_")),
                        "curve_variant": spec["curve_variant"],
                        "source_granularity": spec["source_granularity"],
                        "sample_every_n_steps": spec["sample_every_n_steps"],
                        "smoothing_window": spec["smoothing_window"],
                        "sample_count": int(len(sampled)),
                        "first_global_step": (
                            int(valid_steps.iloc[0]) if not valid_steps.empty else None
                        ),
                        "last_global_step": (
                            int(valid_steps.iloc[-1]) if not valid_steps.empty else None
                        ),
                        "loss_csv_path": str(csv_path.relative_to(run_root)),
                    }
                )
    return pd.DataFrame(rows)


def _prediction_additive_fields(
    prediction_frame: pd.DataFrame,
    row_idx: int,
    *,
    prefix: str = "aaforecast_",
) -> dict[str, object]:
    extras: dict[str, object] = {}
    for column in prediction_frame.columns:
        if not str(column).startswith(prefix):
            continue
        value = prediction_frame.iloc[row_idx][column]
        if pd.isna(value):
            extras[str(column)] = None
        elif isinstance(value, np.generic):
            extras[str(column)] = value.item()
        else:
            extras[str(column)] = value
    return extras


def _write_loss_artifact_summary(run_root: Path) -> Path | None:
    loss_artifacts = _load_loss_artifacts_for_summary(run_root)
    if loss_artifacts.empty:
        return None
    summary_path = run_root / "summary" / SUMMARY_LOSS_ARTIFACTS_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    loss_artifacts.to_csv(summary_path, index=False)
    return summary_path


def _build_tscv_splits(total_rows: int, cv_config) -> list[tuple[list[int], list[int]]]:
    if cv_config.step_size < 1:
        raise ValueError("cv.step_size must be at least 1")

    required_rows = (
        cv_config.gap
        + cv_config.horizon
        + cv_config.step_size * (cv_config.n_windows - 1)
        + 1
    )
    if total_rows < required_rows:
        raise ValueError(
            "Dataset is too short for the configured TSCV policy "
            f"(n_windows={cv_config.n_windows}, horizon={cv_config.horizon}, "
            f"step_size={cv_config.step_size}, gap={cv_config.gap}, "
            f"max_train_size={cv_config.max_train_size})"
        )

    splits: list[tuple[list[int], list[int]]] = []
    for fold_idx in range(cv_config.n_windows):
        train_end = (
            total_rows
            - cv_config.gap
            - cv_config.horizon
            - cv_config.step_size * (cv_config.n_windows - 1 - fold_idx)
        )
        if train_end <= 0:
            raise ValueError(
                "Dataset is too short for the configured TSCV policy "
                f"(n_windows={cv_config.n_windows}, horizon={cv_config.horizon}, "
                f"step_size={cv_config.step_size}, gap={cv_config.gap}, "
                f"max_train_size={cv_config.max_train_size})"
            )
        train_start = 0
        if cv_config.max_train_size is not None:
            train_start = max(0, train_end - cv_config.max_train_size)
        test_start = train_end + cv_config.gap
        test_end = test_start + cv_config.horizon
        splits.append(
            (list(range(train_start, train_end)), list(range(test_start, test_end)))
        )
    return splits


def _run_single_job(
    loaded: LoadedConfig,
    job: JobConfig,
    run_root: Path,
    *,
    manifest_path: Path,
    main_stage: str = "full",
) -> None:
    source_df = pd.read_csv(loaded.config.dataset.path)
    source_df = source_df.sort_values(loaded.config.dataset.dt_col).reset_index(
        drop=True
    )
    freq = _resolve_freq(loaded, source_df)
    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col
    splits = _build_tscv_splits(len(source_df), loaded.config.cv)

    cv_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    effective_job = job
    effective_training_params: dict[str, Any] = {}
    replay_params_override: dict[str, Any] | None = None
    models_dir = run_root / "models" / job.model
    models_dir.mkdir(parents=True, exist_ok=True)
    total_steps = len(splits)
    if job.validated_mode == "learned_auto":
        execute_study_count = len(resolve_study_selection(loaded).execute_study_indices)
        if main_stage == "tune-main-only":
            total_steps *= optuna_num_trials(loaded.config.runtime.opt_n_trial)
        else:
            total_steps *= (
                optuna_num_trials(loaded.config.runtime.opt_n_trial)
                * execute_study_count
            ) + 1
    progress = _ProgressLogger(job.model, total_steps)
    progress.model_started(
        total_folds=len(splits),
        detail=f"mode={job.validated_mode} stage={main_stage} output_root={run_root}",
    )

    if job.validated_mode == "learned_auto":
        selection = resolve_study_selection(loaded)
        study_contexts = [
            build_study_context(
                loaded,
                stage_root=models_dir,
                stage="main-search",
                job_name=job.model,
                study_index=study_index,
            )
            for study_index in selection.execute_study_indices
        ]
        per_study_summary: dict[int, dict[str, Any]] = {}
        per_study_projection_sources: dict[int, dict[str, Path]] = {}
        if main_stage == "tune-main-only":
            if len(study_contexts) != 1:
                raise ValueError(
                    "parallel tuning workers must execute a single resolved Optuna study"
                )
            _parallel_tune_main_job_worker(
                loaded,
                job,
                models_dir,
                study_context=study_contexts[0],
                source_df=source_df,
                freq=freq,
                splits=splits,
                progress=progress,
            )
            progress.model_finished(detail="main-tuning-complete")
            return
        projection_context: StudyContext | None = None
        best_params: dict[str, Any] | None = None
        best_training_params: dict[str, Any] | None = None
        study_summary: dict[str, Any] | None = None
        for study_context in study_contexts:
            if main_stage == "replay-only":
                (
                    current_best_params,
                    current_best_training_params,
                    current_study_summary,
                ) = _load_main_tuning_result(
                    loaded,
                    job,
                    models_dir,
                    study_context=study_context,
                )
            else:
                (
                    current_best_params,
                    current_best_training_params,
                    current_study_summary,
                ) = _tune_main_job(
                    loaded,
                    job,
                    models_dir,
                    study_context=study_context,
                    source_df=source_df,
                    freq=freq,
                    splits=splits,
                    progress=progress,
                )
            study_context.study_root.mkdir(parents=True, exist_ok=True)
            study_best_params_path = study_context.study_root / "best_params.json"
            study_training_best_params_path = (
                study_context.study_root / "training_best_params.json"
            )
            study_summary_path = study_context.study_root / "optuna_study_summary.json"
            study_best_params_path.write_text(
                json.dumps(current_best_params, indent=2), encoding="utf-8"
            )
            study_training_best_params_path.write_text(
                json.dumps(current_best_training_params, indent=2),
                encoding="utf-8",
            )
            study_summary_path.write_text(
                json.dumps(current_study_summary, indent=2), encoding="utf-8"
            )
            (study_context.study_root / "metadata.json").write_text(
                json.dumps(study_catalog_entry(study_context), indent=2),
                encoding="utf-8",
            )
            build_study_visualizations(study_context, current_study_summary)
            per_study_summary[study_context.study_index] = current_study_summary
            per_study_projection_sources[study_context.study_index] = {
                "best_params": study_best_params_path,
                "training_best_params": study_training_best_params_path,
                "study_summary": study_summary_path,
            }
            if (
                projection_context is None
                or study_context.study_index
                == selection.canonical_projection_study_index
            ):
                projection_context = study_context
                best_params = current_best_params
                best_training_params = current_best_training_params
                study_summary = current_study_summary
        if projection_context is None or best_params is None or best_training_params is None or study_summary is None:
            raise RuntimeError("failed to resolve canonical Optuna study projection")
        projection_sources = per_study_projection_sources[projection_context.study_index]
        _copy_projection_file(projection_sources["best_params"], models_dir / "best_params.json")
        _copy_projection_file(
            projection_sources["training_best_params"],
            models_dir / "training_best_params.json",
        )
        study_summary_path = models_dir / "optuna_study_summary.json"
        _copy_projection_file(projection_sources["study_summary"], study_summary_path)
        build_cross_study_visualizations(
            models_dir,
            study_contexts,
            per_study_summary,
        )
        _write_stage_study_catalog(
            models_dir,
            selection,
            summary_by_study=per_study_summary,
        )
        _update_manifest_artifacts(
            manifest_path,
            job_name=job.model,
            study_catalog_path=models_dir / "study_catalog.json",
            selected_study_index=selection.selected_study_index,
            canonical_projection_study_index=selection.canonical_projection_study_index,
            model_best_params_path=models_dir / "best_params.json",
            model_study_summary_path=study_summary_path,
            training_best_params_path=models_dir / "training_best_params.json",
            training_study_summary_path=study_summary_path,
            training_range_source=training_range_source_for_model(
                effective_job.model,
                search_space_payload=loaded.search_space_payload,
            ),
        )
        if _plugin_owned_top_level_job(loaded, job.model) is not None:
            # Plugin-owned models may tune stage-specific keys that are invalid
            # as base model constructor kwargs. Keep base params untouched and
            # forward tuned values through per-fold trial overrides instead.
            effective_job = replace(job, params=dict(job.params))
            replay_params_override = dict(best_params)
        else:
            effective_job = replace(job, params=best_params)
        effective_training_params = best_training_params

    if effective_job.model in BASELINE_MODEL_NAMES:
        train_series = source_df[[dt_col, target_col]].copy()
        train_series.rename(columns={dt_col: "ds", target_col: "y"}, inplace=True)
        train_series["ds"] = pd.to_datetime(train_series["ds"])
        train_series.insert(0, "unique_id", target_col)
        baseline_metrics, baseline_forecasts = _baseline_cross_validation(
            loaded,
            train_series,
            splits,
            effective_job.model,
        )
        for fold_idx, baseline_metric in enumerate(baseline_metrics):
            progress.fold_started(fold_idx, total_folds=len(splits), phase="baseline")
            progress.fold_completed(
                fold_idx,
                total_folds=len(splits),
                phase="baseline",
                detail=f"mse={baseline_metric['MSE']:.4f}",
            )
        metrics_rows.extend(baseline_metrics)
        cv_rows.extend(baseline_forecasts)
    else:
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            progress.fold_started(fold_idx, total_folds=len(splits), phase="replay")
            try:
                fit_kwargs: dict[str, Any] = {
                    "source_df": source_df,
                    "freq": freq,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "params_override": replay_params_override,
                    "training_override": effective_training_params,
                    "fold_idx": fold_idx,
                }
                if get_active_stage_plugin(loaded.config) is not None:
                    fit_kwargs["run_root"] = run_root
                target_predictions, target_actuals, train_end_ds, train_df, nf = (
                    _fit_and_predict_fold(
                        loaded,
                        effective_job,
                        **fit_kwargs,
                    )
                )
            except Exception as exc:
                progress.error(
                    fold_idx, total_folds=len(splits), phase="replay", exc=exc
                )
                raise
            _write_loss_curve_artifact(
                run_root,
                effective_job.model,
                fold_idx,
                nf=nf,
            )
            metrics = _compute_metrics(
                target_actuals, target_predictions[effective_job.model]
            )
            progress.fold_completed(
                fold_idx,
                total_folds=len(splits),
                phase="replay",
                detail=f"mse={metrics['MSE']:.4f}",
            )
            metrics_rows.append(
                {
                    "model": effective_job.model,
                    "requested_mode": job.requested_mode,
                    "validated_mode": job.validated_mode,
                    "fold_idx": fold_idx,
                    "cutoff": str(train_end_ds),
                    **metrics,
                }
            )
            for row_idx, ds in enumerate(target_predictions["ds"]):
                cv_rows.append(
                    {
                        "model": effective_job.model,
                        "requested_mode": job.requested_mode,
                        "validated_mode": job.validated_mode,
                        "fold_idx": fold_idx,
                        "cutoff": str(train_end_ds),
                        "train_end_ds": str(train_end_ds),
                        "unique_id": target_col,
                        "ds": str(ds),
                        "horizon_step": row_idx + 1,
                        "y": float(target_actuals.iloc[row_idx]),
                        "y_hat": float(target_predictions[effective_job.model].iloc[row_idx]),
                        **_prediction_additive_fields(target_predictions, row_idx),
                    }
                )


    cv_dir = run_root / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(cv_dir / f"{effective_job.model}_forecasts.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(
        cv_dir / f"{effective_job.model}_metrics_by_cutoff.csv", index=False
    )
    worker_devices = int(
        os.environ.get(
            "NEURALFORECAST_WORKER_DEVICES",
            loaded.config.scheduler.worker_devices,
        )
    )
    resolved_devices = (
        min(loaded.config.training.devices, worker_devices)
        if loaded.config.training.devices is not None
        else worker_devices
    )
    resolved_strategy = loaded.config.training.strategy
    if resolved_strategy is None and resolved_devices > 1:
        resolved_strategy = "ddp-gloo-auto"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "fit_summary.json").write_text(
        json.dumps(
            {
                "model": effective_job.model,
                "requested_mode": job.requested_mode,
                "validated_mode": job.validated_mode,
                "selected_search_params": list(job.selected_search_params),
                "selected_training_search_params": list(
                    loaded.config.training_search.selected_search_params
                ),
                "devices": resolved_devices,
                "precision": loaded.config.training.precision,
                "strategy": resolved_strategy,
                "dataloader_kwargs": loaded.config.training.dataloader_kwargs,
                "loss": loaded.config.training.loss,
                "evaluation_policy": "tscv_only",
                "tuning_objective_metric": (
                    "mean_fold_mape_on_direct_predictions"
                    if job.validated_mode == "learned_auto"
                    else None
                ),
                "fold_count": len(splits),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    progress.model_finished(detail="run-complete")


def _should_parallelize_single_job_tuning(loaded: LoadedConfig, job: JobConfig) -> bool:
    if job.validated_mode != "learned_auto":
        return False
    if os.environ.get("NEURALFORECAST_WORKER_DEVICES"):
        return False
    if not loaded.config.scheduler.parallelize_single_job_tuning:
        return False
    return len(build_device_groups(loaded.config)) > 1


def _run_single_job_with_parallel_tuning(
    repo_root: Path,
    loaded: LoadedConfig,
    job: JobConfig,
    run_root: Path,
    *,
    manifest_path: Path,
) -> list[dict[str, object]]:
    if _should_prune_model_run_artifacts(job, main_stage="full"):
        _prune_model_run_artifacts(run_root, job.model)
    selection = resolve_study_selection(loaded)
    scheduler_dir = run_root / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for study_index in selection.execute_study_indices:
        study_loaded = loaded_with_study_selection_override(loaded, study_index)
        study_scheduler_dir = scheduler_dir / f"study-{study_index:02d}"
        study_scheduler_dir.mkdir(parents=True, exist_ok=True)
        tune_launches = build_tuning_launch_plan(
            study_loaded.config,
            job_name=job.model,
            worker_count=min(
                len(build_device_groups(study_loaded.config)),
                study_loaded.config.scheduler.max_concurrent_jobs,
            ),
            selected_study=study_index,
        )
        (study_scheduler_dir / "launch_plan.json").write_text(
            json.dumps([launch.__dict__ for launch in tune_launches], indent=2),
            encoding="utf-8",
        )
        worker_results = run_parallel_jobs(
            repo_root, study_loaded, tune_launches, study_scheduler_dir
        )
        results.extend(worker_results)
        if any(int(result["returncode"]) != 0 for result in worker_results):
            raise SystemExit(
                json.dumps(
                    {
                        "ok": False,
                        "worker_results": results,
                        "summary_artifacts": {},
                    }
                )
            )
    _sync_study_roots(
        scheduler_dir / "models" / job.model,
        run_root / "models" / job.model,
        study_indices=selection.execute_study_indices,
    )
    projection_loaded = loaded_with_study_selection_override(
        loaded,
        selection.canonical_projection_study_index,
    )
    _run_single_job(
        projection_loaded,
        job,
        run_root,
        manifest_path=manifest_path,
        main_stage="replay-only",
    )
    return results


def run_loaded_config(
    repo_root: Path,
    loaded: LoadedConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    loaded = loaded_with_study_selection_override(loaded, getattr(args, "optuna_study", None))
    selected_jobs = _selected_jobs(loaded, args.jobs)
    resolved_roots = _resolve_run_roots(
        repo_root,
        loaded,
        output_root=args.output_root,
    )
    paths = _build_resolved_artifacts(repo_root, loaded, resolved_roots["run_root"])
    _validate_jobs(loaded, selected_jobs, paths["capability_path"])
    _validate_adapters(loaded, selected_jobs)
    _initialize_study_catalogs(paths["run_root"], loaded, selected_jobs)
    selection = resolve_study_selection(loaded)
    for job in selected_jobs:
        stage_root = paths["run_root"] / "models" / job.model
        stage_catalog = build_study_catalog_payload(stage_root, selection)
        write_cross_study_visualizations(run_root=stage_root, study_catalog=stage_catalog)
        _update_manifest_artifacts(
            paths["manifest_path"],
            job_name=job.model,
            study_catalog_path=stage_root / "study_catalog.json",
            selected_study_index=selection.selected_study_index,
            canonical_projection_study_index=selection.canonical_projection_study_index,
        )
    stage_result = get_active_stage_plugin(loaded.config)
    if stage_result is not None:
        stage_plugin, _ = stage_result
        stage_plugin.materialize_stage(
            loaded=loaded,
            selected_jobs=selected_jobs,
            run_root=paths["run_root"],
            main_resolved_path=paths["resolved_path"],
            main_capability_path=paths["capability_path"],
            main_manifest_path=paths["manifest_path"],
            entrypoint_version=ENTRYPOINT_VERSION,
            validate_only=args.validate_only,
        )
    if args.validate_only:
        return {
            "ok": True,
            "jobs": [job.model for job in selected_jobs],
            "run_root": str(paths["run_root"]),
            "jobs_route": loaded.active_jobs_route_slug,
        }
    if len(selected_jobs) == 1:
        if (
            args.internal_stage == "full"
            and _should_parallelize_single_job_tuning(loaded, selected_jobs[0])
        ):
            worker_results = _run_single_job_with_parallel_tuning(
                repo_root,
                loaded,
                selected_jobs[0],
                paths["run_root"],
                manifest_path=paths["manifest_path"],
            )
        else:
            if _should_prune_model_run_artifacts(
                selected_jobs[0], main_stage=args.internal_stage
            ):
                _prune_model_run_artifacts(paths["run_root"], selected_jobs[0].model)
            _run_single_job(
                loaded,
                selected_jobs[0],
                paths["run_root"],
                manifest_path=paths["manifest_path"],
                main_stage=args.internal_stage,
            )
            worker_results = []
        summary_artifacts = (
            _build_summary_artifacts(resolved_roots["summary_root"])
            if args.internal_stage != "tune-main-only"
            and _should_build_summary_artifacts()
            else {}
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "executed_jobs": [selected_jobs[0].model],
                    "worker_results": worker_results,
                    "summary_artifacts": summary_artifacts,
                }
            )
        )
        return {
            "ok": True,
            "executed_jobs": [selected_jobs[0].model],
            "worker_results": worker_results,
            "summary_artifacts": summary_artifacts,
            "run_root": str(paths["run_root"]),
            "jobs_route": loaded.active_jobs_route_slug,
        }
    selection = resolve_study_selection(loaded)
    launches = build_launch_plan(loaded.config, selected_jobs, selected_study=selection.selected_study_index)
    scheduler_dir = paths["run_root"] / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    (scheduler_dir / "launch_plan.json").write_text(
        json.dumps([launch.__dict__ for launch in launches], indent=2), encoding="utf-8"
    )
    results = run_parallel_jobs(repo_root, loaded, launches, scheduler_dir)
    summary_artifacts = _build_summary_artifacts(paths["run_root"])
    if any(int(result["returncode"]) != 0 for result in results):
        raise SystemExit(
            json.dumps(
                {
                    "ok": False,
                    "worker_results": results,
                    "summary_artifacts": summary_artifacts,
                }
            )
        )
    print(
        json.dumps(
            {
                "ok": True,
                "scheduled_jobs": [launch.__dict__ for launch in launches],
                "worker_results": results,
                "summary_artifacts": summary_artifacts,
            }
        )
    )
    return {
        "ok": True,
        "scheduled_jobs": [launch.__dict__ for launch in launches],
        "worker_results": results,
        "summary_artifacts": summary_artifacts,
        "run_root": str(paths["run_root"]),
        "jobs_route": loaded.active_jobs_route_slug,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by tests to invoke the runner without venv bootstrap."""
    from main import _run_cli

    return _run_cli(argv, repo_root=Path(__file__).resolve().parents[1])
