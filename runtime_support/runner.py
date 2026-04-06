from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
import fcntl
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
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
from plugins.residual import build_residual_plugin
from plugins.residual.base import ResidualContext
from plugins.residual.features import (
    build_residual_feature_frame,
    hist_exog_lag_feature_name,
)
from runtime_support.manifest import (
    build_manifest,
    residual_active_feature_columns,
    residual_feature_policy_payload,
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
    DEFAULT_RESIDUAL_PARAMS_BY_MODEL,
    DEFAULT_OPTUNA_STUDY_DIRECTION,
    RESIDUAL_DEFAULTS,
    SUPPORTED_MODEL_AUTO_MODEL_NAMES,
    SUPPORTED_RESIDUAL_MODELS,
    LEGACY_TRAINING_SELECTOR_TO_CONFIG_FIELD,
    build_optuna_sampler,
    optuna_num_trials,
    optuna_seed,
    suggest_model_params,
    suggest_residual_params,
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

ENTRYPOINT_VERSION = "neuralforecast-residual-v1"
SUMMARY_REPORT_FILENAME = "sample.md"
LOSS_CURVE_PLOT_FILENAME = "loss_curve.png"
LOSS_CURVE_SAMPLE_FILENAME = "loss_curve_every_10_global_steps.csv"
SUMMARY_LOSS_ARTIFACTS_FILENAME = "loss_curve_artifacts.csv"
LOSS_CURVE_SAMPLE_EVERY_N_STEPS = 10
TARGET_DISPLAY_NAMES = {
    "Com_BrentCrudeOil": "BrentCrude",
    "Com_CrudeOil": "WTI",
}
DEFAULT_SAMPLE_STRUCTURE = {
    "section_setup": "# 02. 데이터 및 모델 세팅",
    "setup_intro": "- 아래는 case별 hist_exog_cols만 남기고, 공통 training/jobs 상세는 Appendix 첨부 하였음.",
    "section_design": "# 03. 실험 설계 및 적용",
    "section_results": "# 04. 실험(모델링) 결과",
    "section_results_detail": "### 04-01. 세부 결과",
    "results_intro": "- 각 run의 leaderboard.csv 기준 결과를 아래에 정리했다.",
    "section_model_tables": "### 각 모형별 Table",
}
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
        run_root / "residual" / model_name,
        workers_root / model_name,
    ]
    for target in targets:
        _remove_existing_artifact(target)
    if not workers_root.exists():
        return
    for worker_root in sorted(workers_root.glob(f"{model_name}#*")):
        _remove_existing_artifact(worker_root)


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
        if loaded.config.residual.enabled and (
            is_direct_top_level_model(job.model)
            or _plugin_owned_top_level_job(loaded, job.model) is not None
        ):
            raise ValueError(
                "Top-level direct/plugin-managed models do not yet support residual-enabled runs; "
                "disable residual or use a learned top-level model / bs_preforcast path."
            )
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
    payload["residual"] = {
        "model": loaded.config.residual.model,
        "target": loaded.config.residual.target,
        "requested_mode": loaded.config.residual.requested_mode,
        "validated_mode": loaded.config.residual.validated_mode,
        "supports_auto": loaded.config.residual.model in SUPPORTED_RESIDUAL_MODELS,
        "search_space_entry_found": bool(loaded.config.residual.selected_search_params),
        "selected_search_params": list(loaded.config.residual.selected_search_params),
        "unknown_search_params": [],
        "validation_error": None,
        "feature_policy": residual_feature_policy_payload(
            loaded.config.residual.features
        ),
        "active_feature_columns": residual_active_feature_columns(
            loaded.config.residual.features
        ),
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
    residual_best_params_path: Path | None = None,
    residual_study_summary_path: Path | None = None,
    residual_feature_policy: dict[str, Any] | None = None,
    residual_active_feature_columns: Sequence[str] | None = None,
) -> None:
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
            if residual_best_params_path is not None:
                job["residual_best_params_path"] = str(residual_best_params_path)
            if residual_study_summary_path is not None:
                job["residual_optuna_study_summary_path"] = str(
                    residual_study_summary_path
                )
            break
    if residual_best_params_path is not None:
        manifest.setdefault("residual", {})["best_params_path"] = str(
            residual_best_params_path
        )
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
    if residual_study_summary_path is not None:
        manifest.setdefault("residual", {})["optuna_study_summary_path"] = str(
            residual_study_summary_path
        )
    if residual_feature_policy is not None:
        manifest.setdefault("residual", {})["feature_policy"] = residual_feature_policy
    if residual_active_feature_columns is not None:
        manifest.setdefault("residual", {})["active_feature_columns"] = list(
            residual_active_feature_columns
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
    anchor: float | None = None
    transform_target: bool = True
    hist_exog_cols: tuple[str, ...] = ()


def _has_target_diff(loaded: LoadedConfig) -> bool:
    return loaded.config.runtime.transformations_target == "diff"


def _has_hist_exog_diff(loaded: LoadedConfig) -> bool:
    return bool(
        loaded.config.runtime.transformations_exog == "diff"
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
    target_diff = _has_target_diff(loaded)
    exog_diff = _has_hist_exog_diff(loaded)
    if not target_diff and not exog_diff:
        return None
    active_target_col = target_col or loaded.config.dataset.target_col
    if len(train_df) < 2:
        raise ValueError(
            "runtime.transformations_target/exog=diff requires at least 2 training rows per fold"
        )
    return _FoldDiffContext(
        target_col=active_target_col,
        anchor=float(train_df[active_target_col].iloc[-1]) if target_diff else None,
        transform_target=target_diff,
        hist_exog_cols=loaded.config.dataset.hist_exog_cols if exog_diff else (),
    )


def _transform_training_frame(
    train_df: pd.DataFrame,
    diff_context: _FoldDiffContext | None,
) -> pd.DataFrame:
    normalized = train_df.reset_index(drop=True).copy()
    if diff_context is None:
        return normalized
    if diff_context.transform_target:
        normalized[diff_context.target_col] = (
            normalized[diff_context.target_col].astype(float).diff()
        )
    for column in diff_context.hist_exog_cols:
        if column in normalized.columns:
            normalized[column] = normalized[column].astype(float).diff()
    transformed = normalized.iloc[1:].reset_index(drop=True)
    if transformed.empty:
        raise ValueError(
            "runtime.transformations_target/exog=diff removed all training rows; need at least 2 rows"
        )
    return transformed


def _transform_training_series(
    history: pd.Series,
    diff_context: _FoldDiffContext | None,
) -> pd.Series:
    normalized = history.reset_index(drop=True).astype(float)
    if diff_context is None or not diff_context.transform_target:
        return normalized
    transformed = normalized.diff().iloc[1:].reset_index(drop=True)
    if transformed.empty:
        raise ValueError(
            "runtime.transformations_target/exog=diff removed all training rows; need at least 2 rows"
        )
    return transformed


def _restore_prediction_series(
    predictions: pd.Series,
    diff_context: _FoldDiffContext | None,
) -> pd.Series:
    restored = predictions.reset_index(drop=True).astype(float)
    if diff_context is None or not diff_context.transform_target:
        return restored
    if diff_context.anchor is None:
        raise ValueError("diff target restoration requires a target anchor")
    return (restored.cumsum() + diff_context.anchor).astype(float)


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
    if loaded.config.residual.enabled:
        return "tuning_pre_replay_corrected_predictions"
    return "tuning_pre_replay_direct_predictions"


def _residual_plugin_config(
    loaded: LoadedConfig, params: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": loaded.config.residual.model,
        "params": params,
        "cpu_threads": loaded.config.residual.cpu_threads,
    }


def _suggest_prefixed_residual_params(
    loaded: LoadedConfig,
    trial: optuna.Trial,
) -> dict[str, Any]:
    if loaded.search_space_payload is None:
        raise ValueError("residual auto tuning requires loaded search-space specs")
    registry = loaded.search_space_payload["residual"][loaded.config.residual.model]
    suggested = DEFAULT_RESIDUAL_PARAMS_BY_MODEL[loaded.config.residual.model].copy()
    for name in loaded.config.residual.selected_search_params:
        suggested[name] = suggest_residual_params(
            loaded.config.residual.model,
            (name,),
            trial,
            param_specs=registry,
            name_prefix="residual__",
        )[name]
    return suggested


def _score_main_trial_with_residual(
    loaded: LoadedConfig,
    job: JobConfig,
    *,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    fold_idx: int,
    train_end_ds: pd.Timestamp,
    target_predictions: pd.DataFrame,
    target_actuals: pd.Series,
    nf: NeuralForecast,
    residual_params: dict[str, Any],
    scratch_root: Path,
) -> float:
    backcast_panel = _build_fold_backcast_panel(
        loaded,
        job,
        nf,
        train_df,
        loaded.config.dataset.dt_col,
        loaded.config.dataset.target_col,
        fold_idx,
    )
    if backcast_panel.empty:
        logging.getLogger(__name__).warning(
            "backcast panel is empty for fold %d of %s; "
            "falling back to direct MAPE instead of residual-corrected MAPE",
            fold_idx,
            job.model,
        )
        return _compute_metrics(target_actuals, target_predictions[job.model])["MAPE"]
    eval_panel = _build_fold_eval_panel(
        loaded,
        job,
        fold_idx,
        train_end_ds,
        target_predictions,
        target_actuals,
        future_df,
        train_df,
    )
    with TemporaryDirectory(dir=scratch_root, prefix="residual-main-objective-") as tmpdir:
        plugin = build_residual_plugin(_residual_plugin_config(loaded, residual_params))
        plugin.fit(
            backcast_panel,
            _build_residual_context(
                loaded,
                job,
                Path(tmpdir),
                model_name=job.model,
            ),
        )
        predicted = plugin.predict(eval_panel.copy())
    corrected = eval_panel.reset_index(drop=True).copy()
    corrected["residual_hat"] = predicted["residual_hat"].astype(float).values
    corrected["y_hat_corrected"] = reconstruct_corrected_forecast(
        corrected,
        loaded.config.residual.target,
    )
    return _compute_metrics(corrected["y"], corrected["y_hat_corrected"])["MAPE"]


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
        reserved_count = max(int(state.get("reserved_trial_count", 0)), finished_trial_count)
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
        "remaining_trial_count": max(target_trial_count - existing_finished_trial_count, 0),
        "reserved_trial_slots": reserved_slots,
    }


def _require_complete_best_trial(study: optuna.Study, *, label: str) -> optuna.FrozenTrial:
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
        candidate_training_params = suggest_training_params(
            loaded.config.training_search.selected_search_params,
            trial,
            model_name=job.model,
            param_specs=training_param_registry_for_model(
                job.model,
                search_space_payload=loaded.search_space_payload,
            ),
        )
        candidate_residual_params: dict[str, Any] | None = None
        if loaded.config.residual.enabled:
            if loaded.config.residual.validated_mode == "residual_auto":
                candidate_residual_params = _suggest_prefixed_residual_params(
                    loaded, trial
                )
            else:
                candidate_residual_params = {
                    **RESIDUAL_DEFAULTS[loaded.config.residual.model],
                    **loaded.config.residual.params,
                }
        trial.set_user_attr("best_params", candidate_params)
        trial.set_user_attr("best_training_params", candidate_training_params)
        if candidate_residual_params is not None:
            trial.set_user_attr("best_residual_params", candidate_residual_params)
        fold_mape: list[float] = []
        _write_trial_result(
            trial_dir,
            status="running",
            study_context=study_context,
            payload={
                "trial_number": trial.number,
                "best_params": candidate_params,
                "best_training_params": candidate_training_params,
                "best_residual_params": candidate_residual_params,
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
                fit_kwargs: dict[str, Any] = {
                    "source_df": source_df,
                    "freq": freq,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "params_override": candidate_params,
                    "training_override": candidate_training_params,
                }
                if get_active_stage_plugin(loaded.config) is not None:
                    fit_kwargs["run_root"] = None
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
                if candidate_residual_params is None:
                    metric = _compute_metrics(
                        target_actuals, target_predictions[job.model]
                    )["MAPE"]
                else:
                    future_df = source_df.iloc[test_idx].reset_index(drop=True)
                    metric = _score_main_trial_with_residual(
                        loaded,
                        job,
                        train_df=train_df,
                        future_df=future_df,
                        fold_idx=fold_idx,
                        train_end_ds=train_end_ds,
                        target_predictions=target_predictions,
                        target_actuals=target_actuals,
                        nf=nf,
                        residual_params=candidate_residual_params,
                        scratch_root=trial_dir,
                    )
                fold_mape.append(metric)
                interim_metric = _mean_fold_metric(fold_mape, metric_name="mape")
                trial.set_user_attr("fold_mape", fold_mape.copy())
                _write_trial_result(
                    trial_dir,
                    status="running",
                    study_context=study_context,
                    payload={
                        "trial_number": trial.number,
                        "best_params": candidate_params,
                        "best_training_params": candidate_training_params,
                        "best_residual_params": candidate_residual_params,
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
                        "best_residual_params": candidate_residual_params,
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
                        "best_residual_params": candidate_residual_params,
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
                "best_residual_params": candidate_residual_params,
                "fold_mape": fold_mape,
                "objective_value": metric,
            },
        )
        return metric

    return objective


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
    best_params = dict(best_trial.user_attrs["best_params"])
    best_training_params = dict(best_trial.user_attrs["best_training_params"])
    best_residual_params = (
        dict(best_trial.user_attrs["best_residual_params"])
        if "best_residual_params" in best_trial.user_attrs
        else None
    )
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
    if best_residual_params is not None:
        summary["best_residual_params"] = best_residual_params
    return best_params, best_training_params, summary


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
        if loaded.config.residual.enabled:
            _write_stage_study_catalog(run_root / "residual" / job.model, selection)


def _score_residual_params(
    loaded: LoadedConfig,
    job: JobConfig,
    params: dict[str, Any],
    fold_payloads: list[dict[str, Any]],
    *,
    trial: optuna.Trial,
    study_context: StudyContext,
) -> float:
    trial.set_user_attr("study_index", study_context.study_index)
    trial.set_user_attr("sampler_seed", study_context.sampler_seed)
    trial.set_user_attr("proposal_flow_id", study_context.proposal_flow_id)
    mape_scores: list[float] = []
    for step_idx, payload in enumerate(fold_payloads):
        fold_idx = int(payload["fold_idx"])
        try:
            payload["trial_dir"].mkdir(parents=True, exist_ok=True)
            plugin = build_residual_plugin(_residual_plugin_config(loaded, params))
            plugin.fit(
                payload["backcast_panel"],
                _build_residual_context(
                    loaded,
                    job,
                    payload["trial_dir"],
                    model_name=job.model,
                ),
            )
            predicted = plugin.predict(payload["eval_panel"].copy())
            corrected = payload["eval_panel"].reset_index(drop=True).copy()
            corrected["residual_hat"] = predicted["residual_hat"].astype(float).values
            corrected["y_hat_corrected"] = reconstruct_corrected_forecast(
                corrected,
                loaded.config.residual.target,
            )
            mape_scores.append(
                _compute_metrics(corrected["y"], corrected["y_hat_corrected"])["MAPE"]
            )
            interim_metric = _mean_fold_metric(mape_scores, metric_name="mape")
            trial.set_user_attr("fold_mape", mape_scores.copy())
            trial.report(interim_metric, step=step_idx)
            if trial.should_prune():
                trial.set_user_attr("pruned_after_fold", fold_idx)
                raise optuna.TrialPruned(
                    f"Pruned residual trial after fold {fold_idx} with mean_mape={interim_metric:.4f}"
                )
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr(
                "failure_reason",
                f"fold={fold_idx} {type(exc).__name__}: {exc}",
            )
            raise _OptunaTrialFailure(
                f"{job.model} residual tuning failed on fold {fold_idx}: {type(exc).__name__}: {exc}"
            ) from exc
    return _mean_fold_metric(mape_scores, metric_name="mape")


def _residual_trajectory_group_keys() -> list[str]:
    return ["fold_idx", "cutoff"]


def _prepare_ordered_trajectory_frame(panel_df: pd.DataFrame) -> pd.DataFrame:
    ordered = panel_df.copy()
    ordered["__orig_order"] = np.arange(len(ordered))
    return ordered.sort_values(
        _residual_trajectory_group_keys() + ["horizon_step", "__orig_order"],
        kind="stable",
    ).reset_index(drop=True)


def build_residual_target(panel_df: pd.DataFrame, mode: str) -> pd.Series:
    if panel_df.empty:
        return pd.Series(dtype=float, index=panel_df.index)
    if mode == "level":
        return (
            panel_df["y"].astype(float) - panel_df["y_hat_base"].astype(float)
        ).rename("residual_target")

    ordered = _prepare_ordered_trajectory_frame(panel_df)
    group_keys = _residual_trajectory_group_keys()
    group_index = [ordered[key] for key in group_keys]
    step_index = ordered.groupby(group_keys, sort=False).cumcount()
    y = ordered["y"].astype(float)
    y_hat_base = ordered["y_hat_base"].astype(float)
    residual_target = (y.groupby(group_index).diff() - y_hat_base.groupby(group_index).diff()).where(
        step_index > 0,
        y - y_hat_base,
    )
    ordered["residual_target"] = residual_target.astype(float)
    restored = ordered.sort_values("__orig_order", kind="stable")
    return pd.Series(
        restored["residual_target"].to_numpy(),
        index=panel_df.index,
        name="residual_target",
    )


def reconstruct_corrected_forecast(panel_df: pd.DataFrame, mode: str) -> pd.Series:
    if panel_df.empty:
        return pd.Series(dtype=float, index=panel_df.index)
    if mode == "level":
        return (
            panel_df["y_hat_base"].astype(float)
            + panel_df["residual_hat"].astype(float)
        ).rename("y_hat_corrected")

    ordered = _prepare_ordered_trajectory_frame(panel_df)
    group_keys = _residual_trajectory_group_keys()
    ordered["y_hat_corrected"] = (
        ordered["y_hat_base"].astype(float)
        + ordered.groupby(group_keys, sort=False)["residual_hat"].cumsum().astype(float)
    )
    restored = ordered.sort_values("__orig_order", kind="stable")
    return pd.Series(
        restored["y_hat_corrected"].to_numpy(),
        index=panel_df.index,
        name="y_hat_corrected",
    )


def _prediction_column(predictions: pd.DataFrame, model_name: str) -> str:
    if model_name in predictions.columns:
        return model_name
    raise KeyError(f"Could not find prediction column for {model_name}")


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
        train_df, job, dataset=loaded.config.dataset, dt_col=dt_col, future_df=future_df
    )


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


def _iter_backcast_cutoff_indices(
    train_length: int,
    input_size: int,
    horizon: int,
    step_size: int,
) -> list[int]:
    if train_length < input_size + horizon:
        return []
    return list(range(input_size - 1, train_length - horizon, step_size))


def _build_residual_context(
    loaded: LoadedConfig,
    job: JobConfig,
    residual_root: Path,
    *,
    model_name: str | None = None,
) -> ResidualContext:
    return ResidualContext(
        job_name=job.model,
        model_name=model_name or job.model,
        output_dir=residual_root,
        config=loaded.normalized_payload,
    )


def _predict_with_fitted_model(nf: NeuralForecast, adapter_inputs) -> pd.DataFrame:
    predict_kwargs = {
        "df": adapter_inputs.fit_df,
        "static_df": adapter_inputs.static_df,
    }
    if adapter_inputs.futr_df is not None:
        predict_kwargs["futr_df"] = adapter_inputs.futr_df
    return nf.predict(**predict_kwargs)


def _selected_residual_exog_columns(loaded: LoadedConfig) -> list[str]:
    columns: list[str] = []
    for column in loaded.config.residual.features.exog_sources.hist:
        lagged = hist_exog_lag_feature_name(column)
        if lagged not in columns:
            columns.append(lagged)
    for column in (
        *loaded.config.residual.features.exog_sources.futr,
        *loaded.config.residual.features.exog_sources.static,
    ):
        if column not in columns:
            columns.append(column)
    return columns


def _residual_panel_feature_values(
    loaded: LoadedConfig,
    row_source: pd.Series,
    *,
    static_source_df: pd.DataFrame,
) -> dict[str, object]:
    values: dict[str, object] = {}
    for column in loaded.config.residual.features.exog_sources.hist:
        values[hist_exog_lag_feature_name(column)] = static_source_df[column].iloc[-1]
    for column in loaded.config.residual.features.exog_sources.futr:
        values[column] = row_source[column]
    for column in loaded.config.residual.features.exog_sources.static:
        values[column] = static_source_df[column].iloc[-1]
    return values


def _canonical_panel_columns(
    include_target: bool = True,
    *,
    feature_source_columns: Sequence[str] = (),
) -> list[str]:
    columns = [
        "model_name",
        "fold_idx",
        "panel_split",
        "unique_id",
        "cutoff",
        "train_end_ds",
        "ds",
        "horizon_step",
        "y_hat_base",
    ]
    if include_target:
        columns.extend(["y", "residual_target"])
    for column in feature_source_columns:
        if column not in columns:
            columns.append(column)
    return columns


def _fold_artifact_dir(residual_root: Path, fold_idx: int) -> Path:
    return residual_root / "folds" / f"fold_{fold_idx:03d}"


def _build_fold_eval_panel(
    loaded: LoadedConfig,
    job: JobConfig,
    fold_idx: int,
    train_end_ds: object,
    target_predictions: pd.DataFrame,
    actuals: pd.Series,
    future_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(train_end_ds)
    feature_source_columns = _selected_residual_exog_columns(loaded)
    rows: list[dict[str, object]] = []
    for row_idx, ds in enumerate(target_predictions["ds"]):
        y_hat_base = float(target_predictions[job.model].iloc[row_idx])
        y = float(actuals.iloc[row_idx])
        row_source = future_df.iloc[row_idx]
        rows.append(
            {
                "model_name": job.model,
                "fold_idx": fold_idx,
                "panel_split": "fold_eval",
                "unique_id": target_predictions["unique_id"].iloc[row_idx],
                "cutoff": cutoff,
                "train_end_ds": cutoff,
                "ds": pd.to_datetime(ds),
                "horizon_step": row_idx + 1,
                "y": y,
                "y_hat_base": y_hat_base,
                "residual_target": 0.0,
                **_residual_panel_feature_values(
                    loaded,
                    row_source,
                    static_source_df=train_df,
                ),
            }
        )
    panel = pd.DataFrame(
        rows,
        columns=_canonical_panel_columns(
            feature_source_columns=feature_source_columns,
        ),
    ).reset_index(drop=True)
    panel["residual_target"] = build_residual_target(
        panel,
        loaded.config.residual.target,
    )
    return panel


def _build_fold_backcast_panel(
    loaded: LoadedConfig,
    job: JobConfig,
    nf: NeuralForecast,
    train_df: pd.DataFrame,
    dt_col: str,
    target_col: str,
    fold_idx: int,
) -> pd.DataFrame:
    cutoff_indices = _iter_backcast_cutoff_indices(
        train_length=len(train_df),
        input_size=loaded.config.training.input_size,
        horizon=loaded.config.cv.horizon,
        step_size=loaded.config.cv.step_size,
    )
    feature_source_columns = _selected_residual_exog_columns(loaded)
    rows: list[dict[str, object]] = []
    for cutoff_idx in cutoff_indices:
        history_df = train_df.iloc[: cutoff_idx + 1].reset_index(drop=True)
        future_df = train_df.iloc[
            cutoff_idx + 1 : cutoff_idx + 1 + loaded.config.cv.horizon
        ].reset_index(drop=True)
        if _has_any_runtime_diff(loaded) and len(history_df) < 2:
            continue
        diff_context = _build_fold_diff_context(
            loaded,
            history_df,
            target_col=target_col,
        )
        transformed_history_df = _transform_training_frame(history_df, diff_context)
        adapter_inputs = _build_adapter_inputs(
            loaded,
            transformed_history_df,
            future_df,
            job,
            dt_col,
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
        actuals = future_df[target_col].reset_index(drop=True)
        cutoff = pd.to_datetime(train_df[dt_col].iloc[cutoff_idx])
        for row_idx, ds in enumerate(target_predictions["ds"]):
            y_hat_base = float(target_predictions[pred_col].iloc[row_idx])
            y = float(actuals.iloc[row_idx])
            row_source = future_df.iloc[row_idx]
            rows.append(
                {
                    "model_name": job.model,
                    "fold_idx": fold_idx,
                    "panel_split": "backcast_train",
                    "unique_id": target_col,
                    "cutoff": cutoff,
                    "train_end_ds": cutoff,
                    "ds": pd.to_datetime(ds),
                    "horizon_step": row_idx + 1,
                    "y": y,
                    "y_hat_base": y_hat_base,
                    "residual_target": 0.0,
                    **_residual_panel_feature_values(
                        loaded,
                        row_source,
                        static_source_df=history_df,
                    ),
                }
            )
    panel = pd.DataFrame(
        rows,
        columns=_canonical_panel_columns(
            feature_source_columns=feature_source_columns,
        ),
    ).reset_index(drop=True)
    panel["residual_target"] = build_residual_target(
        panel,
        loaded.config.residual.target,
    )
    return panel


def _resolve_runtime_active_feature_columns(
    loaded: LoadedConfig,
    fold_payloads: Sequence[dict[str, Any]],
) -> list[str]:
    for payload in fold_payloads:
        for panel_name in ("backcast_panel", "eval_panel"):
            panel = payload.get(panel_name)
            if isinstance(panel, pd.DataFrame):
                feature_frame = build_residual_feature_frame(
                    panel,
                    feature_config=loaded.config.residual.features,
                )
                return list(feature_frame.columns)
    return residual_active_feature_columns(loaded.config.residual.features)


def _apply_residual_plugin(
    loaded: LoadedConfig,
    job: JobConfig,
    run_root: Path,
    fold_payloads: list[dict[str, Any]],
    *,
    manifest_path: Path,
    residual_params_override: dict[str, Any] | None = None,
    residual_study_summary_override: dict[str, Any] | None = None,
) -> None:
    if job.model in BASELINE_MODEL_NAMES:
        return
    if not loaded.config.residual.enabled:
        return

    residual_root = run_root / "residual" / job.model
    residual_root.mkdir(parents=True, exist_ok=True)
    feature_policy = residual_feature_policy_payload(loaded.config.residual.features)
    active_feature_columns = _resolve_runtime_active_feature_columns(
        loaded, fold_payloads
    )
    corrected_groups: list[pd.DataFrame] = []
    checkpoint_metadata: dict[str, dict[str, object]] = {}
    total_backcast_rows = 0
    residual_params = {
        **RESIDUAL_DEFAULTS[loaded.config.residual.model],
        **loaded.config.residual.params,
    }
    if residual_params_override is not None:
        residual_params = {
            **RESIDUAL_DEFAULTS[loaded.config.residual.model],
            **residual_params_override,
        }
        best_params_path = residual_root / "best_params.json"
        best_params_path.write_text(
            json.dumps(residual_params, indent=2), encoding="utf-8"
        )
        residual_summary_path: Path | None = None
        if residual_study_summary_override is not None:
            residual_summary_path = residual_root / "optuna_study_summary.json"
            residual_summary_path.write_text(
                json.dumps(residual_study_summary_override, indent=2),
                encoding="utf-8",
            )
        _update_manifest_artifacts(
            manifest_path,
            job_name=job.model,
            residual_best_params_path=best_params_path,
            residual_study_summary_path=residual_summary_path,
        )
    elif loaded.config.residual.validated_mode == "residual_auto":
        selection = resolve_study_selection(loaded)
        residual_study_context = build_study_context(
            loaded,
            stage_root=residual_root,
            stage="residual-search",
            job_name=job.model,
            study_index=selection.canonical_projection_study_index,
            base_seed=optuna_seed(loaded.config.runtime.random_seed),
        )
        sampler = build_optuna_sampler(residual_study_context.sampler_seed)
        residual_trial_count = optuna_num_trials(loaded.config.runtime.opt_n_trial)

        def objective(trial: optuna.Trial) -> float:
            candidate_params = suggest_residual_params(
                loaded.config.residual.model,
                loaded.config.residual.selected_search_params,
                trial,
                param_specs=(
                    None
                    if loaded.search_space_payload is None
                    else loaded.search_space_payload["residual"][
                        loaded.config.residual.model
                    ]
                ),
            )
            trial.set_user_attr("best_params", candidate_params)
            trial_root = _trial_dir_from_context(residual_study_context, trial.number)
            prepared_payloads: list[dict[str, Any]] = []
            for payload in fold_payloads:
                prepared_payloads.append(
                    {
                        **payload,
                        "trial_dir": trial_root / f"fold-{int(payload['fold_idx']):03d}",
                    }
                )
            return _score_residual_params(
                loaded,
                job,
                candidate_params,
                prepared_payloads,
                trial=trial,
                study_context=residual_study_context,
            )

        study, study_metadata = _open_persistent_study(
            residual_study_context,
            sampler=sampler,
        )
        optimize_metadata = _optimize_study_with_resume(
            study,
            objective=objective,
            target_trial_count=residual_trial_count,
        )
        best_trial = _require_complete_best_trial(
            study, label=f"{job.model} residual Optuna study"
        )
        residual_params = {
            **RESIDUAL_DEFAULTS[loaded.config.residual.model],
            **best_trial.user_attrs["best_params"],
        }
        residual_study_context.best_params_path.parent.mkdir(parents=True, exist_ok=True)
        residual_study_context.best_params_path.write_text(
            json.dumps(residual_params, indent=2), encoding="utf-8"
        )
        residual_summary_payload = {
            **_trial_metrics_summary(study),
            **study_metadata,
            **optimize_metadata,
            "requested_mode": loaded.config.residual.requested_mode,
            "validated_mode": loaded.config.residual.validated_mode,
            "selected_search_params": list(
                loaded.config.residual.selected_search_params
            ),
            "best_params": residual_params,
            "objective_stage": "tuning_pre_replay_residual_corrected_predictions",
        }
        residual_study_context.summary_path.write_text(
            json.dumps(residual_summary_payload, indent=2),
            encoding="utf-8",
        )
        build_study_visualizations(residual_study_context, residual_summary_payload)
        _copy_projection_file(
            residual_study_context.best_params_path,
            residual_root / "best_params.json",
        )
        _copy_projection_file(
            residual_study_context.summary_path,
            residual_root / "optuna_study_summary.json",
        )
        _write_stage_study_catalog(
            residual_root,
            selection,
            summary_by_study={
                residual_study_context.study_index: residual_summary_payload,
            },
        )
        _update_manifest_artifacts(
            manifest_path,
            job_name=job.model,
            study_catalog_path=residual_root / "study_catalog.json",
            selected_study_index=selection.selected_study_index,
            canonical_projection_study_index=(
                selection.canonical_projection_study_index
            ),
            residual_best_params_path=residual_root / "best_params.json",
            residual_study_summary_path=residual_root / "optuna_study_summary.json",
        )


    for payload in fold_payloads:
        fold_idx = int(payload["fold_idx"])
        backcast_panel: pd.DataFrame = payload["backcast_panel"]
        eval_panel: pd.DataFrame = payload["eval_panel"]
        base_summary: dict[str, object] = payload["base_summary"]
        fold_root = _fold_artifact_dir(residual_root, fold_idx)
        base_checkpoint_dir = fold_root / "base_checkpoint"
        residual_checkpoint_dir = fold_root / "residual_checkpoint"
        base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        plugin = build_residual_plugin(_residual_plugin_config(loaded, residual_params))

        backcast_panel.to_csv(fold_root / "backcast_panel.csv", index=False)
        (base_checkpoint_dir / "fit_summary.json").write_text(
            json.dumps(base_summary, indent=2),
            encoding="utf-8",
        )
        plugin.fit(
            backcast_panel,
            _build_residual_context(
                loaded,
                job,
                residual_checkpoint_dir,
                model_name=job.model,
            ),
        )
        predicted = plugin.predict(eval_panel.copy())
        corrected = eval_panel.reset_index(drop=True).copy()
        corrected["residual_hat"] = predicted["residual_hat"].astype(float).values
        corrected["y_hat_corrected"] = reconstruct_corrected_forecast(
            corrected,
            loaded.config.residual.target,
        )
        corrected_groups.append(corrected)
        corrected.to_csv(fold_root / "corrected_eval.csv", index=False)
        base_metrics = _compute_metrics(corrected["y"], corrected["y_hat_base"])
        corrected_metrics = _compute_metrics(
            corrected["y"], corrected["y_hat_corrected"]
        )
        (fold_root / "metrics.json").write_text(
            json.dumps(
                {
                    "fold_idx": fold_idx,
                    "cutoff": str(corrected["cutoff"].iloc[0]),
                    "train_end_ds": str(corrected["train_end_ds"].iloc[0]),
                    "backcast_rows": int(len(backcast_panel)),
                    "base_metrics": base_metrics,
                    "corrected_metrics": corrected_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        checkpoint_metadata[str(fold_idx)] = {
            **plugin.metadata(),
            "feature_policy": feature_policy,
            "active_feature_columns": active_feature_columns,
        }
        total_backcast_rows += len(backcast_panel)

    corrected_folds = pd.concat(corrected_groups, ignore_index=True)
    corrected_folds.to_csv(residual_root / "corrected_folds.csv", index=False)
    (residual_root / "plugin_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2), encoding="utf-8"
    )
    diagnostics = {
        "model": job.model,
        "residual_model": loaded.config.residual.model,
        "residual.target": loaded.config.residual.target,
        "residual.feature_policy": feature_policy,
        "active_feature_columns": active_feature_columns,
        "fold_count": int(len(fold_payloads)),
        "backcast_rows_total": int(total_backcast_rows),
        "corrected_eval_rows": int(len(corrected_folds)),
        "corrected_eval_mode": "per_fold_backcast_runtime",
        "tscv_policy": {
            "n_windows": loaded.config.cv.n_windows,
            "horizon": loaded.config.cv.horizon,
            "step_size": loaded.config.cv.step_size,
            "gap": loaded.config.cv.gap,
            "max_train_size": loaded.config.cv.max_train_size,
        },
        "artifact_layout": "residual/<model>/folds/fold_{i:03d}/(backcast_panel.csv, corrected_eval.csv, base_checkpoint/, residual_checkpoint/model.ubj)",
    }
    (residual_root / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    _update_manifest_artifacts(
        manifest_path,
        job_name=job.model,
        residual_feature_policy=feature_policy,
        residual_active_feature_columns=active_feature_columns,
    )


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


def _artifact_model_name(path: Path, suffix: str) -> str:
    return path.name.removesuffix(suffix)


def _residual_summary_model_name(model_name: str) -> str:
    return f"{model_name}_res"


def _load_residual_metrics_for_summary(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    residual_dir = root / "residual"
    if not residual_dir.exists():
        return rows
    for model_root in sorted(path for path in residual_dir.iterdir() if path.is_dir()):
        folds_dir = model_root / "folds"
        if not folds_dir.exists():
            continue
        for metrics_path in sorted(folds_dir.glob("fold_*/metrics.json")):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            corrected_metrics = payload.get("corrected_metrics")
            if not corrected_metrics:
                continue
            rows.append(
                {
                    "model": _residual_summary_model_name(model_root.name),
                    "fold_idx": payload.get("fold_idx"),
                    "cutoff": payload.get("cutoff"),
                    **corrected_metrics,
                }
            )
    return rows


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
        rows.extend(_load_residual_metrics_for_summary(root))
    return pd.DataFrame(rows)


def _build_leaderboard(metrics_frame: pd.DataFrame) -> pd.DataFrame:
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


def _load_summary_config(run_root: Path) -> dict[str, Any]:
    config_path = run_root / "config" / "config.resolved.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_sample_structure() -> dict[str, str]:
    return dict(DEFAULT_SAMPLE_STRUCTURE)


def _display_target_name(target_col: str | None) -> str:
    if not target_col:
        return ""
    if target_col in TARGET_DISPLAY_NAMES:
        return TARGET_DISPLAY_NAMES[target_col]
    target = target_col.removeprefix("Com_").replace("_", " ").strip()
    return target


def _case_title(task_name: str | None, target_col: str | None, run_root: Path) -> str:
    raw_name = (task_name or run_root.name).strip()
    lowered = raw_name.lower()
    variant = ""
    if lowered.endswith("_hpt"):
        raw_name = raw_name[:-4]
        variant = " HPT"
    parts = [part for part in raw_name.split("_") if part]
    case_number = ""
    for part in parts:
        if part.lower().startswith("case") and part[4:].isdigit():
            case_number = part[4:]
            break
    target_name = _display_target_name(target_col) or raw_name
    if case_number:
        return f"Case {case_number}{variant} | {target_name}"
    return f"{raw_name}{variant} | {target_name}".strip(" |")


def _yaml_hist_exog_block(hist_exog_cols: Sequence[str] | None) -> str:
    lines = ["...", "hist_exog_cols:"]
    for col in hist_exog_cols or ():
        lines.append(f"  - {col}")
    lines.append("...")
    return "\n".join(lines)


def _round_half_up(value: float | int | None) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if not np.isfinite(numeric):
        return ""
    quantized = Decimal(str(numeric)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return format(quantized, "f")


def _format_report_metric(value: float | int | None, *, percentage: bool = False) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if not np.isfinite(numeric):
        return ""
    if percentage:
        return f"{_round_half_up(numeric * 100)}%"
    return _round_half_up(numeric)


def _markdown_table_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.splitlines()).replace("|", "\\|")


def _build_summary_markdown(run_root: Path, leaderboard: pd.DataFrame) -> Path:
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_path = summary_dir / SUMMARY_REPORT_FILENAME

    resolved = _load_summary_config(run_root)
    dataset = resolved.get("dataset", {})
    cv = resolved.get("cv", {})
    task = resolved.get("task", {})
    sample_structure = _load_sample_structure()

    case_title = _case_title(task.get("name"), dataset.get("target_col"), run_root)
    hist_exog_yaml = _yaml_hist_exog_block(dataset.get("hist_exog_cols"))
    target_name = _display_target_name(dataset.get("target_col"))
    horizon = cv.get("horizon")
    step_size = cv.get("step_size")
    n_windows = cv.get("n_windows")
    gap = cv.get("gap")
    overlap_eval_policy = cv.get("overlap_eval_policy")

    leaderboard_lines = [
        "| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if leaderboard.empty:
        leaderboard_lines.append("|  |  |  |  |  |  |")
    else:
        for row in leaderboard.to_dict(orient="records"):
            leaderboard_lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_table_cell(row.get("rank", "")) if row.get("rank") is not None else "",
                        _markdown_table_cell(row.get("model", "") or ""),
                        _markdown_table_cell(_format_report_metric(row.get("mean_fold_mape"), percentage=True)),
                        _markdown_table_cell(_format_report_metric(row.get("mean_fold_nrmse"))),
                        _markdown_table_cell(_format_report_metric(row.get("mean_fold_mae"))),
                        _markdown_table_cell(_format_report_metric(row.get("mean_fold_r2"))),
                    ]
                )
                + " |"
            )

    model_sections: list[str] = []
    if leaderboard.empty:
        model_sections.append("- \n\n    | Case | MAPE | nRMSE | MAE | R2 |\n    | --- | --- | --- | --- | --- |\n    |  |  |  |  |  |")
    else:
        for row in leaderboard.to_dict(orient="records"):
            model_sections.append(
                "\n".join(
                    [
                        f"- {row.get('model', '')}",
                        "",
                        "    | Case | MAPE | nRMSE | MAE | R2 |",
                        "    | --- | --- | --- | --- | --- |",
                        "    | "
                        + " | ".join(
                            [
                                _markdown_table_cell(case_title),
                                _markdown_table_cell(_format_report_metric(row.get("mean_fold_mape"), percentage=True)),
                                _markdown_table_cell(_format_report_metric(row.get("mean_fold_nrmse"))),
                                _markdown_table_cell(_format_report_metric(row.get("mean_fold_mae"))),
                                _markdown_table_cell(_format_report_metric(row.get("mean_fold_r2"))),
                            ]
                        )
                        + " |",
                    ]
                )
            )

    report = "\n".join(
        [
            sample_structure["section_setup"],
            "",
            "---",
            "",
            f"## **{case_title}**",
            "",
            sample_structure["setup_intro"],
            "",
            "```yaml",
            hist_exog_yaml,
            "```",
            "",
            sample_structure["section_design"],
            "",
            "---",
            "",
            f"- 타깃: {target_name}",
            "- 각 타깃을 독립적인 forecasting 문제로 학습/평가",
            (
                f"- 평가는 {n_windows if n_windows is not None else ''}개 rolling TSCV"
                f"(h={horizon if horizon is not None else ''}, step={step_size if step_size is not None else ''}, gap={gap if gap is not None else ''}) 구조로 설계했다."
            ),
            f"- overlap_eval_policy: {overlap_eval_policy or ''}",
            "",
            sample_structure["section_results"],
            "",
            sample_structure["section_results_detail"],
            "",
            "---",
            "",
            sample_structure["results_intro"],
            "",
            f"## **{case_title}**",
            "",
            *leaderboard_lines,
            "",
            sample_structure["section_model_tables"],
            "",
            *model_sections,
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _load_last_fold_forecasts(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in _summary_job_roots(run_root):
        for path in sorted((root / "cv").glob("*_forecasts.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            model_name = _artifact_model_name(path, "_forecasts.csv")
            if "model" not in frame.columns:
                frame["model"] = model_name
            frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="coerce")
            frames.append(frame)
        residual_dir = root / "residual"
        if not residual_dir.exists():
            continue
        for path in sorted(residual_dir.glob("*/corrected_folds.csv")):
            frame = pd.read_csv(path)
            if frame.empty or "y_hat_corrected" not in frame.columns:
                continue
            model_name = path.parent.name
            frame = frame.copy()
            frame["model"] = _residual_summary_model_name(model_name)
            frame["y_hat"] = pd.to_numeric(
                frame["y_hat_corrected"], errors="coerce"
            )
            frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="coerce")
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts = forecasts.dropna(subset=["fold_idx"]).copy()
    forecasts["fold_idx"] = forecasts["fold_idx"].astype(int)
    return forecasts


def _build_residual_comparison_plots(
    run_root: Path, last_fold_forecasts: pd.DataFrame
) -> dict[str, str]:
    residual_dir = run_root / "summary" / "residual"
    plot_paths: dict[str, str] = {}
    available_models = set(last_fold_forecasts["model"])
    residual_models = sorted(
        model.removesuffix("_res")
        for model in available_models
        if model.endswith("_res") and model.removesuffix("_res") in available_models
    )
    if not residual_models:
        return plot_paths
    residual_dir.mkdir(parents=True, exist_ok=True)
    for model_name in residual_models:
        plot_path = residual_dir / f"{model_name}.png"
        _plot_last_fold_overlay(
            last_fold_forecasts,
            [model_name, _residual_summary_model_name(model_name)],
            plot_path,
            title=f"Last fold predictions ({model_name}: base vs base+residual)",
        )
        plot_paths[f"residual_{model_name}"] = str(plot_path)
    return plot_paths


def _plot_last_fold_overlay(
    forecasts: pd.DataFrame,
    selected_models: list[str],
    plot_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    selected = forecasts[forecasts["model"].isin(selected_models)].copy()
    if selected.empty:
        return
    selected["ds"] = pd.Index([pd.Timestamp(value) for value in selected["ds"]])
    actual = (
        selected[["ds", "y"]]
        .drop_duplicates(subset=["ds"])
        .sort_values("ds")
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual["ds"], actual["y"], label="actual", linewidth=2.5, color="black")
    for model_name in selected_models:
        model_frame = (
            selected[selected["model"] == model_name]
            .sort_values("ds")
            .reset_index(drop=True)
        )
        if model_frame.empty:
            continue
        ax.plot(model_frame["ds"], model_frame["y_hat"], label=model_name, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.legend(loc="best")
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_series = (
        curve_frame[["global_step", "train_loss"]]
        .dropna(subset=["train_loss"])
        .reset_index(drop=True)
    )
    val_series = (
        curve_frame[["global_step", "val_loss"]]
        .dropna(subset=["val_loss"])
        .reset_index(drop=True)
    )
    return train_series, val_series


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
    curve_frame = _trajectory_frame(nf)
    if curve_frame.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fold_root = _learned_fold_artifact_dir(run_root, model_name, fold_idx)
    fold_root.mkdir(parents=True, exist_ok=True)
    figure_path = fold_root / LOSS_CURVE_PLOT_FILENAME
    sampled_curve_path = fold_root / LOSS_CURVE_SAMPLE_FILENAME
    _sample_loss_curve_frame(curve_frame).to_csv(sampled_curve_path, index=False)
    train_series, val_series = _loss_curve_series(curve_frame)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        train_series["global_step"],
        train_series["train_loss"],
        label="train_loss",
        linewidth=1.8,
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


def _build_summary_artifacts(run_root: Path) -> dict[str, str]:
    summary_dir = run_root / "summary"
    metrics = _load_metrics_for_summary(run_root)
    if metrics.empty:
        return {}
    leaderboard = _build_leaderboard(metrics)
    workbook_path = summary_dir / "leaderboard.csv"
    _write_leaderboard_workbook(leaderboard, workbook_path)
    markdown_path = _build_summary_markdown(run_root, leaderboard)

    forecasts = _load_last_fold_forecasts(run_root)
    plot_paths: dict[str, str] = {
        "leaderboard": str(workbook_path),
        "markdown": str(markdown_path),
    }
    loss_artifact_summary_path = _write_loss_artifact_summary(run_root)
    if loss_artifact_summary_path is not None:
        plot_paths["loss_artifacts"] = str(loss_artifact_summary_path)
    if forecasts.empty:
        return plot_paths
    last_fold = int(forecasts["fold_idx"].max())
    last_fold_forecasts = forecasts[forecasts["fold_idx"] == last_fold].copy()
    ordered_models = [
        model for model in leaderboard["model"].tolist() if model in set(last_fold_forecasts["model"])
    ]
    plot_specs = [
        ("all_models", ordered_models, "Last fold predictions (all models)"),
        ("top3", ordered_models[:3], "Last fold predictions (top 3)"),
        ("top5", ordered_models[:5], "Last fold predictions (top 5)"),
    ]
    for slug, models, title in plot_specs:
        if not models:
            continue
        plot_path = summary_dir / f"last_fold_{slug}.png"
        _plot_last_fold_overlay(last_fold_forecasts, models, plot_path, title=title)
        plot_paths[slug] = str(plot_path)
    plot_paths.update(_build_residual_comparison_plots(run_root, last_fold_forecasts))
    return plot_paths


def _load_loss_artifacts_for_summary(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in _summary_job_roots(run_root):
        for csv_path in sorted(
            root.glob(f"models/*/folds/fold_*/{LOSS_CURVE_SAMPLE_FILENAME}")
        ):
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
                    "sample_every_n_steps": LOSS_CURVE_SAMPLE_EVERY_N_STEPS,
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


def _write_loss_artifact_summary(run_root: Path) -> Path | None:
    loss_artifacts = _load_loss_artifacts_for_summary(run_root)
    if loss_artifacts.empty:
        return None
    summary_path = run_root / "summary" / SUMMARY_LOSS_ARTIFACTS_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    loss_artifacts.to_csv(summary_path, index=False)
    return summary_path


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
    fold_payloads: list[dict[str, Any]] = []
    effective_job = job
    effective_training_params: dict[str, Any] = {}
    effective_residual_params: dict[str, Any] | None = None
    residual_study_summary: dict[str, Any] | None = None
    projection_study_context: StudyContext | None = None
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
        projection_study_context = projection_context
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
        effective_job = replace(job, params=best_params)
        effective_training_params = best_training_params
        if loaded.config.residual.enabled:
            residual_params = study_summary.get("best_residual_params")
            if isinstance(residual_params, dict) and residual_params:
                effective_residual_params = dict(residual_params)
                if loaded.config.residual.validated_mode == "residual_auto":
                    residual_study_summary = {
                        "direction": study_summary.get("direction"),
                        "trial_count": study_summary.get("trial_count"),
                        "finished_trial_count": study_summary.get(
                            "finished_trial_count"
                        ),
                        "state_counts": study_summary.get("state_counts", {}),
                        "best_value": study_summary.get("best_value"),
                        "best_trial_number": study_summary.get("best_trial_number"),
                        "objective_metric": study_summary.get("objective_metric"),
                        "storage_backend": study_summary.get("storage_backend"),
                        "storage_path": study_summary.get("storage_path"),
                        "study_name": study_summary.get("study_name"),
                        "requested_trial_count": study_summary.get(
                            "requested_trial_count"
                        ),
                        "existing_trial_count_before_optimize": study_summary.get(
                            "existing_trial_count_before_optimize"
                        ),
                        "existing_finished_trial_count_before_optimize": study_summary.get(
                            "existing_finished_trial_count_before_optimize"
                        ),
                        "remaining_trial_count": study_summary.get(
                            "remaining_trial_count"
                        ),
                        "reserved_trial_slots": study_summary.get(
                            "reserved_trial_slots"
                        ),
                        "requested_mode": loaded.config.residual.requested_mode,
                        "validated_mode": loaded.config.residual.validated_mode,
                        "selected_search_params": list(
                            loaded.config.residual.selected_search_params
                        ),
                        "best_params": effective_residual_params,
                        "objective_stage": "tuning_pre_replay_joint_corrected_predictions",
                        "source_study_name": study_summary.get("study_name"),
                        "source_storage_path": study_summary.get("storage_path"),
                    }

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
                    "training_override": effective_training_params,
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
            loss_curve_path = _write_loss_curve_artifact(
                run_root,
                effective_job.model,
                fold_idx,
                nf=nf,
            )
            future_df = source_df.iloc[test_idx].reset_index(drop=True)
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
                    }
                )
            if loaded.config.residual.enabled:
                backcast_panel = _build_fold_backcast_panel(
                    loaded,
                    effective_job,
                    nf,
                    train_df,
                    dt_col,
                    target_col,
                    fold_idx,
                )
                eval_panel = _build_fold_eval_panel(
                    loaded,
                    effective_job,
                    fold_idx,
                    train_end_ds,
                    target_predictions,
                    target_actuals,
                    future_df,
                    train_df,
                )
                fold_payloads.append(
                    {
                        "fold_idx": fold_idx,
                        "backcast_panel": backcast_panel,
                        "eval_panel": eval_panel,
                        "trial_dir": (
                            (projection_study_context or build_study_context(
                                loaded,
                                stage_root=run_root / "residual" / effective_job.model,
                                stage="residual-search",
                                job_name=effective_job.model,
                                study_index=resolve_study_selection(loaded).canonical_projection_study_index,
                            )).trials_root
                            / "replay-selected-study"
                        ),
                        "base_summary": {
                            "model": effective_job.model,
                            "fold_idx": fold_idx,
                            "train_rows": int(len(train_df)),
                            "eval_rows": int(len(future_df)),
                            "train_end_ds": str(train_end_ds),
                            "loss": loaded.config.training.loss,
                            "loss_curve_path": (
                                str(loss_curve_path) if loss_curve_path is not None else None
                            ),
                        },
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
                    "mean_fold_mape_on_corrected_predictions"
                    if job.validated_mode == "learned_auto"
                    and loaded.config.residual.enabled
                    else "mean_fold_mape_on_direct_predictions"
                    if job.validated_mode == "learned_auto"
                    else None
                ),
                "fold_count": len(splits),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if effective_job.model not in BASELINE_MODEL_NAMES:
        _apply_residual_plugin(
            loaded,
            effective_job,
            run_root,
            fold_payloads,
            manifest_path=manifest_path,
            residual_params_override=effective_residual_params,
            residual_study_summary_override=residual_study_summary,
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
