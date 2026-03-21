from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import optuna
import pandas as pd
from neuralforecast import NeuralForecast
from optuna.trial import TrialState

from .adapters import build_multivariate_inputs, build_univariate_inputs
from .config import JobConfig, LoadedConfig, load_app_config
from .features import build_residual_feature_frame
from .manifest import (
    build_manifest,
    residual_active_feature_columns,
    residual_feature_policy_payload,
    write_manifest,
)
from .models import BASELINE_MODEL_NAMES, build_model, validate_job
from .optuna_spaces import (
    DEFAULT_OPTUNA_STUDY_DIRECTION,
    RESIDUAL_DEFAULTS,
    SUPPORTED_AUTO_MODEL_NAMES,
    SUPPORTED_RESIDUAL_MODELS,
    TRAINING_SELECTOR_TO_CONFIG_FIELD,
    build_optuna_sampler,
    optuna_num_trials,
    optuna_seed,
    suggest_model_params,
    suggest_residual_params,
    suggest_training_params,
)
from .plugins_base import ResidualContext
from .progress import (
    ConsoleProgressRenderer,
    ModelProgressState,
    emit_progress_event,
)
from .registry import build_residual_plugin
from .scheduler import build_launch_plan, run_parallel_jobs

ENTRYPOINT_VERSION = "neuralforecast-residual-v1"
SUMMARY_REPORT_FILENAME = "sample.md"
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
MIN_PRUNE_FOLD_IDX = 10


class _OptunaTrialFailure(RuntimeError):
    """Recoverable per-trial failure that should not abort the whole study."""


def _can_prune_at_fold(fold_idx: int) -> bool:
    return fold_idx >= MIN_PRUNE_FOLD_IDX


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Residual wrapper runtime for neuralforecast."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--config-toml", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--jobs", nargs="+", default=None)
    parser.add_argument("--output-root", default=None)
    return parser


def _selected_jobs(loaded: LoadedConfig, names: list[str] | None):
    if not names:
        return list(loaded.config.jobs)
    allowed = set(names)
    return [job for job in loaded.config.jobs if job.model in allowed]


def _default_output_root(repo_root: Path, loaded: LoadedConfig) -> Path:
    task_name = (loaded.config.task.name or "").strip()
    if not task_name:
        return repo_root / "runs" / "validation"
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-" for char in task_name
    ).strip(".-")
    return repo_root / "runs" / (safe_name or "validation")


def _build_resolved_artifacts(
    repo_root: Path, loaded: LoadedConfig, output_root: Path
) -> dict[str, Path]:
    run_root = output_root.resolve()
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
    for job in selected_jobs:
        caps = validate_job(job)
        payload[job.model] = {
            **caps.__dict__,
            "requested_mode": job.requested_mode,
            "validated_mode": job.validated_mode,
            "supports_auto": job.model in SUPPORTED_AUTO_MODEL_NAMES,
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
    capability_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_manifest_artifacts(
    manifest_path: Path,
    *,
    job_name: str,
    model_best_params_path: Path | None = None,
    model_study_summary_path: Path | None = None,
    training_best_params_path: Path | None = None,
    training_study_summary_path: Path | None = None,
    residual_best_params_path: Path | None = None,
    residual_study_summary_path: Path | None = None,
    residual_feature_policy: dict[str, Any] | None = None,
    residual_active_feature_columns: Sequence[str] | None = None,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for job in manifest.get("jobs", []):
        if job.get("model") == job_name:
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
    if training_best_params_path is not None:
        manifest.setdefault("training_search", {})["best_params_path"] = str(
            training_best_params_path
        )
    if training_study_summary_path is not None:
        manifest.setdefault("training_search", {})["optuna_study_summary_path"] = str(
            training_study_summary_path
        )
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
        TRAINING_SELECTOR_TO_CONFIG_FIELD.get(key, key): value
        for key, value in training_override.items()
    }
    return replace(
        loaded.config,
        training=replace(loaded.config.training, **normalized_override),
    )


def _should_use_multivariate(loaded: LoadedConfig, job: JobConfig) -> bool:
    caps = validate_job(job)
    configured_native_exog = (
        (bool(loaded.config.dataset.hist_exog_cols) and caps.supports_hist_exog)
        or (bool(loaded.config.dataset.futr_exog_cols) and caps.supports_futr_exog)
        or (bool(loaded.config.dataset.static_exog_cols) and caps.supports_stat_exog)
    )
    return bool(caps.multivariate and not configured_native_exog)


def _validate_adapters(loaded: LoadedConfig, selected_jobs) -> None:
    source_df = pd.read_csv(loaded.config.dataset.path)
    dt_col = loaded.config.dataset.dt_col
    for job in selected_jobs:
        if _should_use_multivariate(loaded, job):
            build_multivariate_inputs(
                source_df, job, dataset=loaded.config.dataset, dt_col=dt_col
            )
        else:
            build_univariate_inputs(
                source_df, job, dataset=loaded.config.dataset, dt_col=dt_col
            )


def _cutoff_train_end(
    total_rows: int, horizon: int, step_size: int, n_windows: int, fold_idx: int
) -> int:
    remaining = horizon + step_size * (n_windows - 1 - fold_idx)
    return total_rows - remaining


def _resolve_freq(loaded: LoadedConfig, source_df: pd.DataFrame) -> str:
    if loaded.config.dataset.freq:
        return loaded.config.dataset.freq
    inferred = pd.infer_freq(pd.to_datetime(source_df[loaded.config.dataset.dt_col]))
    if inferred is None:
        raise ValueError(
            "Could not infer frequency from dataset.dt_col; set dataset.freq explicitly"
        )
    return inferred


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
    source_df: pd.DataFrame,
    freq: str,
    train_idx: list[int],
    test_idx: list[int],
    params_override: dict[str, Any] | None = None,
    training_override: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, NeuralForecast]:
    effective_config = _effective_config(loaded, training_override)
    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col
    train_df = source_df.iloc[train_idx].reset_index(drop=True)
    future_df = source_df.iloc[test_idx].reset_index(drop=True)
    adapter_inputs = _build_adapter_inputs(loaded, train_df, future_df, job, dt_col)
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
    _prediction_column(predictions, job.model)
    target_predictions = predictions[
        predictions["unique_id"] == target_col
    ].reset_index(drop=True)
    target_actuals = future_df[target_col].reset_index(drop=True)
    train_end_ds = pd.to_datetime(train_df[dt_col].iloc[-1])
    return target_predictions, target_actuals, train_end_ds, train_df, nf


def _trial_metrics_summary(study: optuna.Study) -> dict[str, Any]:
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
        "objective_metric": "mean_fold_mse",
    }


def _build_study_name(
    loaded: LoadedConfig, *, stage: str, job_name: str, suffix: str = ""
) -> str:
    task_name = loaded.config.task.name or loaded.source_path.stem or "run"
    parts = [
        "neuralforecast",
        task_name,
        job_name,
        stage,
        loaded.input_hash[:12],
    ]
    if suffix:
        parts.append(suffix)
    return "::".join(parts)


def _open_persistent_study(
    study_root: Path,
    *,
    loaded: LoadedConfig,
    stage: str,
    job_name: str,
    sampler: optuna.samplers.BaseSampler,
) -> tuple[optuna.Study, dict[str, Any]]:
    storage_dir = study_root / ".optuna"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = storage_dir / f"{stage}.journal"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(storage_path))
    )
    study_name = _build_study_name(loaded, stage=stage, job_name=job_name)
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
        sampler=sampler,
        direction=DEFAULT_OPTUNA_STUDY_DIRECTION,
    )
    metadata = {
        "study_name": study_name,
        "storage_backend": "journal",
        "storage_path": str(storage_path.resolve()),
    }
    return study, metadata


def _finished_trial_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state.is_finished())


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


def _require_complete_best_trial(study: optuna.Study, *, label: str) -> optuna.FrozenTrial:
    if not any(trial.state == TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError(
            f"{label} finished without a successful Optuna trial; inspect study summary for failed/pruned states"
        )
    return study.best_trial


def _tune_main_job(
    loaded: LoadedConfig,
    job: JobConfig,
    models_dir: Path,
    *,
    source_df: pd.DataFrame,
    freq: str,
    splits: list[tuple[list[int], list[int]]],
    progress: _ProgressLogger | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sampler = build_optuna_sampler(optuna_seed(loaded.config.runtime.random_seed))
    trial_count = optuna_num_trials(loaded.config.runtime.opt_n_trial)

    def objective(trial: optuna.Trial) -> float:
        candidate_params = suggest_model_params(
            job.model, job.selected_search_params, trial
        )
        candidate_training_params = suggest_training_params(
            loaded.config.training_search.selected_search_params, trial
        )
        trial.set_user_attr("best_params", candidate_params)
        trial.set_user_attr("best_training_params", candidate_training_params)
        fold_mse: list[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            phase = f"tune-trial-{trial.number + 1}/{trial_count}"
            if progress is not None:
                progress.fold_started(
                    fold_idx, total_folds=len(splits), phase=phase
                )
            try:
                target_predictions, target_actuals, _, _, _ = _fit_and_predict_fold(
                    loaded,
                    job,
                    source_df=source_df,
                    freq=freq,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    params_override=candidate_params,
                    training_override=candidate_training_params,
                )
                mse = _compute_metrics(
                    target_actuals, target_predictions[job.model]
                )["MSE"]
                fold_mse.append(mse)
                interim_metric = float(sum(fold_mse) / len(fold_mse))
                trial.set_user_attr("fold_mse", fold_mse.copy())
                trial.report(interim_metric, step=fold_idx)
                if _can_prune_at_fold(fold_idx) and trial.should_prune():
                    trial.set_user_attr("pruned_after_fold", fold_idx)
                    if progress is not None:
                        progress.fold_completed(
                            fold_idx,
                            total_folds=len(splits),
                            phase=phase,
                            detail=f"pruned mean_mse={interim_metric:.4f}",
                        )
                    raise optuna.TrialPruned(
                        f"Pruned after fold {fold_idx} with mean_mse={interim_metric:.4f}"
                    )
                if progress is not None:
                    progress.fold_completed(
                        fold_idx,
                        total_folds=len(splits),
                        phase=phase,
                        detail=f"mse={mse:.4f}",
                    )
            except optuna.TrialPruned:
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
                raise _OptunaTrialFailure(
                    f"{job.model} tuning failed on fold {fold_idx}: {type(exc).__name__}: {exc}"
                ) from exc
        metric = float(sum(fold_mse) / len(fold_mse))
        trial.set_user_attr("fold_mse", fold_mse)
        return metric

    study, study_metadata = _open_persistent_study(
        models_dir,
        loaded=loaded,
        stage="main-search",
        job_name=job.model,
        sampler=sampler,
    )
    optimize_metadata = _optimize_study_with_resume(
        study,
        objective=objective,
        target_trial_count=trial_count,
    )
    best_trial = _require_complete_best_trial(
        study, label=f"{job.model} main Optuna study"
    )
    best_params = dict(best_trial.user_attrs["best_params"])
    best_training_params = dict(best_trial.user_attrs["best_training_params"])
    summary = {
        **_trial_metrics_summary(study),
        **study_metadata,
        **optimize_metadata,
        "requested_mode": job.requested_mode,
        "validated_mode": job.validated_mode,
        "selected_search_params": list(job.selected_search_params),
        "selected_training_search_params": list(
            loaded.config.training_search.selected_search_params
        ),
        "best_params": best_params,
        "best_training_params": best_training_params,
        "fold_mse": best_trial.user_attrs["fold_mse"],
        "objective_stage": "tuning_pre_replay_direct_predictions",
    }
    return best_params, best_training_params, summary


def _score_residual_params(
    loaded: LoadedConfig,
    job: JobConfig,
    params: dict[str, Any],
    fold_payloads: list[dict[str, Any]],
    *,
    trial: optuna.Trial,
) -> float:
    mse_scores: list[float] = []
    for step_idx, payload in enumerate(fold_payloads):
        fold_idx = int(payload["fold_idx"])
        try:
            plugin = build_residual_plugin(
                {"model": loaded.config.residual.model, "params": params}
            )
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
            mse_scores.append(
                _compute_metrics(corrected["y"], corrected["y_hat_corrected"])["MSE"]
            )
            interim_metric = float(sum(mse_scores) / len(mse_scores))
            trial.set_user_attr("fold_mse", mse_scores.copy())
            trial.report(interim_metric, step=step_idx)
            if _can_prune_at_fold(fold_idx) and trial.should_prune():
                trial.set_user_attr("pruned_after_fold", fold_idx)
                raise optuna.TrialPruned(
                    f"Pruned residual trial after fold {fold_idx} with mean_mse={interim_metric:.4f}"
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
    return float(sum(mse_scores) / len(mse_scores))


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
    for column in (
        *loaded.config.residual.features.exog_sources.hist,
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
        values[column] = row_source[column]
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
        adapter_inputs = _build_adapter_inputs(
            loaded,
            history_df,
            future_df,
            job,
            dt_col,
        )
        predictions = _predict_with_fitted_model(nf, adapter_inputs)
        pred_col = _prediction_column(predictions, job.model)
        target_predictions = predictions[
            predictions["unique_id"] == target_col
        ].reset_index(drop=True)
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
    residual_params = {**RESIDUAL_DEFAULTS[loaded.config.residual.model], **loaded.config.residual.params}
    if loaded.config.residual.validated_mode == "residual_auto":
        sampler = build_optuna_sampler(optuna_seed(loaded.config.runtime.random_seed))
        residual_trial_count = optuna_num_trials(loaded.config.runtime.opt_n_trial)

        def objective(trial: optuna.Trial) -> float:
            candidate_params = suggest_residual_params(
                loaded.config.residual.model,
                loaded.config.residual.selected_search_params,
                trial,
            )
            trial.set_user_attr("best_params", candidate_params)
            for payload in fold_payloads:
                payload["trial_dir"].mkdir(parents=True, exist_ok=True)
            score = _score_residual_params(
                loaded,
                job,
                candidate_params,
                fold_payloads,
                trial=trial,
            )
            return score

        study, study_metadata = _open_persistent_study(
            residual_root,
            loaded=loaded,
            stage="residual-search",
            job_name=job.model,
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
        (residual_root / "best_params.json").write_text(
            json.dumps(residual_params, indent=2), encoding="utf-8"
        )
        (residual_root / "optuna_study_summary.json").write_text(
            json.dumps(
                {
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
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _update_manifest_artifacts(
            manifest_path,
            job_name=job.model,
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
        plugin = build_residual_plugin(
            {"model": loaded.config.residual.model, "params": residual_params}
        )

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
        if model_name == "Naive":
            pred_values = pd.Series([float(history.iloc[-1])] * len(future_actual))
        elif model_name == "SeasonalNaive":
            season = min(len(history), len(future_actual))
            tail = history.iloc[-season:].reset_index(drop=True)
            pred_values = pd.Series(
                (list(tail) * ((len(future_actual) // len(tail)) + 1))[
                    : len(future_actual)
                ]
            )
        else:
            pred_values = pd.Series([float(history.mean())] * len(future_actual))
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
    sample_path = Path(__file__).resolve().parents[1] / "sample.md"
    if not sample_path.exists():
        return dict(DEFAULT_SAMPLE_STRUCTURE)

    lines = sample_path.read_text(encoding="utf-8").splitlines()

    def _find_line(prefix: str, default: str) -> str:
        for line in lines:
            if line.strip().startswith(prefix):
                return line.rstrip()
        return default

    def _find_contains(snippet: str, default: str) -> str:
        for line in lines:
            if snippet in line:
                return line.rstrip()
        return default

    return {
        "section_setup": _find_line("# 02.", DEFAULT_SAMPLE_STRUCTURE["section_setup"]),
        "setup_intro": _find_contains(
            "hist_exog_cols만 남기고", DEFAULT_SAMPLE_STRUCTURE["setup_intro"]
        ),
        "section_design": _find_line("# 03.", DEFAULT_SAMPLE_STRUCTURE["section_design"]),
        "section_results": _find_line("# 04.", DEFAULT_SAMPLE_STRUCTURE["section_results"]),
        "section_results_detail": _find_line(
            "### 04-01.", DEFAULT_SAMPLE_STRUCTURE["section_results_detail"]
        ),
        "results_intro": _find_contains(
            "leaderboard", DEFAULT_SAMPLE_STRUCTURE["results_intro"]
        ),
        "section_model_tables": _find_line(
            "### 각 모형별 Table", DEFAULT_SAMPLE_STRUCTURE["section_model_tables"]
        ),
    }


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
                        str(row.get("rank", "")) if row.get("rank") is not None else "",
                        str(row.get("model", "") or ""),
                        _format_report_metric(row.get("mean_fold_mape"), percentage=True),
                        _format_report_metric(row.get("mean_fold_nrmse")),
                        _format_report_metric(row.get("mean_fold_mae")),
                        _format_report_metric(row.get("mean_fold_r2")),
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
                                case_title,
                                _format_report_metric(row.get("mean_fold_mape"), percentage=True),
                                _format_report_metric(row.get("mean_fold_nrmse")),
                                _format_report_metric(row.get("mean_fold_mae")),
                                _format_report_metric(row.get("mean_fold_r2")),
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
    if not frames:
        return pd.DataFrame()
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts = forecasts.dropna(subset=["fold_idx"]).copy()
    forecasts["fold_idx"] = forecasts["fold_idx"].astype(int)
    return forecasts


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
    selected["ds"] = pd.to_datetime(selected["ds"])
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


def _trajectory_frame(nf: NeuralForecast) -> pd.DataFrame:
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
    return train_frame.merge(val_frame, on="global_step", how="outer").sort_values(
        "global_step", kind="stable"
    )


def _write_loss_curve_artifact(
    run_root: Path,
    model_name: str,
    fold_idx: int,
    *,
    nf: NeuralForecast,
) -> Path | None:
    curve_frame = _trajectory_frame(nf)
    if curve_frame.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fold_root = _learned_fold_artifact_dir(run_root, model_name, fold_idx)
    fold_root.mkdir(parents=True, exist_ok=True)
    figure_path = fold_root / "loss_curve.png"
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        curve_frame["global_step"],
        curve_frame["train_loss"],
        label="train_loss",
        linewidth=1.8,
    )
    axis.plot(
        curve_frame["global_step"],
        curve_frame["val_loss"],
        label="val_loss",
        linewidth=1.8,
    )
    axis.set_title(f"{model_name} fold {fold_idx:03d} loss curve")
    axis.set_xlabel("global_step")
    axis.set_ylabel("loss")
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
    return plot_paths


def _run_single_job(
    loaded: LoadedConfig, job: JobConfig, run_root: Path, *, manifest_path: Path
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
    models_dir = run_root / "models" / job.model
    models_dir.mkdir(parents=True, exist_ok=True)
    total_steps = len(splits)
    if job.validated_mode == "learned_auto":
        total_steps *= optuna_num_trials(loaded.config.runtime.opt_n_trial) + 1
    progress = _ProgressLogger(job.model, total_steps)
    progress.model_started(
        total_folds=len(splits),
        detail=f"mode={job.validated_mode} output_root={run_root}",
    )

    if job.validated_mode == "learned_auto":
        best_params, best_training_params, study_summary = _tune_main_job(
            loaded,
            job,
            models_dir,
            source_df=source_df,
            freq=freq,
            splits=splits,
            progress=progress,
        )
        (models_dir / "best_params.json").write_text(
            json.dumps(best_params, indent=2), encoding="utf-8"
        )
        (models_dir / "training_best_params.json").write_text(
            json.dumps(best_training_params, indent=2), encoding="utf-8"
        )
        (models_dir / "optuna_study_summary.json").write_text(
            json.dumps(study_summary, indent=2), encoding="utf-8"
        )
        (models_dir / "training_optuna_study_summary.json").write_text(
            json.dumps(study_summary, indent=2), encoding="utf-8"
        )
        _update_manifest_artifacts(
            manifest_path,
            job_name=job.model,
            model_best_params_path=models_dir / "best_params.json",
            model_study_summary_path=models_dir / "optuna_study_summary.json",
            training_best_params_path=models_dir / "training_best_params.json",
            training_study_summary_path=models_dir
            / "training_optuna_study_summary.json",
        )
        effective_job = replace(job, params=best_params)
        effective_training_params = best_training_params

    if effective_job.model in BASELINE_MODEL_NAMES:
        train_series = source_df[[dt_col, target_col]].copy()
        train_series.rename(columns={dt_col: "ds", target_col: "y"}, inplace=True)
        train_series["ds"] = pd.to_datetime(train_series["ds"])
        train_series.insert(0, "unique_id", target_col)
        baseline_metrics, baseline_forecasts = _baseline_cross_validation(
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
                target_predictions, target_actuals, train_end_ds, train_df, nf = (
                    _fit_and_predict_fold(
                        loaded,
                        effective_job,
                        source_df=source_df,
                        freq=freq,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        training_override=effective_training_params,
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
                        "trial_dir": run_root / "residual" / effective_job.model / "_optuna_trial",
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
                "devices": 1,
                "loss": loaded.config.training.loss,
                "evaluation_policy": "tscv_only",
                "tuning_objective_metric": "mean_fold_mse_on_direct_predictions"
                if job.validated_mode == "learned_auto"
                else None,
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
        )
    progress.model_finished(detail="run-complete")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or args.config_path
    loaded = load_app_config(
        repo_root, config_path=config_path, config_toml_path=args.config_toml
    )
    output_root = (
        Path(args.output_root) if args.output_root else _default_output_root(repo_root, loaded)
    )
    paths = _build_resolved_artifacts(repo_root, loaded, output_root)
    selected_jobs = _selected_jobs(loaded, args.jobs)
    _validate_jobs(loaded, selected_jobs, paths["capability_path"])
    _validate_adapters(loaded, selected_jobs)
    if args.validate_only:
        print(json.dumps({"ok": True, "jobs": [job.model for job in selected_jobs]}))
        return 0
    if len(selected_jobs) == 1:
        _run_single_job(
            loaded,
            selected_jobs[0],
            paths["run_root"],
            manifest_path=paths["manifest_path"],
        )
        summary_artifacts = (
            _build_summary_artifacts(paths["run_root"])
            if _should_build_summary_artifacts()
            else {}
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "executed_jobs": [selected_jobs[0].model],
                    "summary_artifacts": summary_artifacts,
                }
            )
        )
        return 0
    launches = build_launch_plan(loaded.config, selected_jobs)
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
    return 0
