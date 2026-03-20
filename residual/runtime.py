from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import optuna
import pandas as pd
from neuralforecast import NeuralForecast

from .adapters import build_multivariate_inputs, build_univariate_inputs
from .config import JobConfig, LoadedConfig, load_app_config
from .manifest import build_manifest, write_manifest
from .models import BASELINE_MODEL_NAMES, build_model, validate_job
from .optuna_spaces import (
    DEFAULT_OPTUNA_STUDY_DIRECTION,
    DEFAULT_RESIDUAL_PARAMS,
    SUPPORTED_AUTO_MODEL_NAMES,
    SUPPORTED_RESIDUAL_MODELS,
    build_optuna_sampler,
    optuna_num_trials,
    optuna_seed,
    suggest_model_params,
    suggest_residual_params,
    suggest_training_params,
)
from .plugins_base import ResidualContext
from .registry import build_residual_plugin
from .scheduler import build_launch_plan, run_parallel_jobs

ENTRYPOINT_VERSION = "neuralforecast-residual-v1"


def _progress_bar(completed: int, total: int, *, width: int = 18) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    pct = int(round(ratio * 100))
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total} {pct:3d}%"


class _ProgressLogger:
    def __init__(self, job_name: str, total_steps: int) -> None:
        self.job_name = job_name
        self.total_steps = max(total_steps, 1)
        self.completed_steps = 0

    def _emit(
        self,
        event: str,
        *,
        fold_idx: int | None = None,
        total_folds: int | None = None,
        phase: str | None = None,
        detail: str | None = None,
    ) -> None:
        parts = [f"[progress][{self.job_name}] {event}"]
        if phase:
            parts.append(f"phase={phase}")
        if fold_idx is not None:
            fold_text = f"fold={fold_idx + 1}"
            if total_folds is not None:
                fold_text += f"/{total_folds}"
            parts.append(fold_text)
        parts.append(_progress_bar(self.completed_steps, self.total_steps))
        if detail:
            parts.append(detail)
        print(" | ".join(parts), flush=True)

    def model_started(self, *, total_folds: int, detail: str | None = None) -> None:
        self._emit("model-start", total_folds=total_folds, detail=detail)

    def fold_started(
        self, fold_idx: int, *, total_folds: int, phase: str | None = None
    ) -> None:
        self._emit(
            "fold-start",
            fold_idx=fold_idx,
            total_folds=total_folds,
            phase=phase,
        )

    def fold_completed(
        self,
        fold_idx: int,
        *,
        total_folds: int,
        phase: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.completed_steps += 1
        self._emit(
            "fold-done",
            fold_idx=fold_idx,
            total_folds=total_folds,
            phase=phase,
            detail=detail,
        )

    def error(
        self,
        fold_idx: int | None,
        *,
        total_folds: int | None = None,
        phase: str | None = None,
        exc: BaseException,
    ) -> None:
        self._emit(
            "error",
            fold_idx=fold_idx,
            total_folds=total_folds,
            phase=phase,
            detail=f"{type(exc).__name__}: {exc}",
        )


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
        "requested_mode": loaded.config.residual.requested_mode,
        "validated_mode": loaded.config.residual.validated_mode,
        "supports_auto": loaded.config.residual.model in SUPPORTED_RESIDUAL_MODELS,
        "search_space_entry_found": bool(loaded.config.residual.selected_search_params),
        "selected_search_params": list(loaded.config.residual.selected_search_params),
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
    return replace(
        loaded.config,
        training=replace(loaded.config.training, **training_override),
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
    err = actual.reset_index(drop=True) - predicted.reset_index(drop=True)
    mae = float(err.abs().mean())
    mse = float((err**2).mean())
    rmse = mse**0.5
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


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
    return {
        "direction": study.direction.name.lower(),
        "trial_count": len(study.trials),
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "objective_metric": "mean_fold_mse",
    }


def _tune_main_job(
    loaded: LoadedConfig,
    job: JobConfig,
    *,
    source_df: pd.DataFrame,
    freq: str,
    splits: list[tuple[list[int], list[int]]],
    progress: _ProgressLogger | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sampler = build_optuna_sampler(optuna_seed(loaded.config.runtime.random_seed))
    trial_count = optuna_num_trials()

    def objective(trial: optuna.Trial) -> float:
        candidate_params = suggest_model_params(
            job.model, job.selected_search_params, trial
        )
        candidate_training_params = suggest_training_params(
            loaded.config.training_search.selected_search_params, trial
        )
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
                if progress is not None:
                    progress.fold_completed(
                        fold_idx,
                        total_folds=len(splits),
                        phase=phase,
                        detail=f"mse={mse:.4f}",
                    )
            except Exception as exc:
                if progress is not None:
                    progress.error(
                        fold_idx,
                        total_folds=len(splits),
                        phase=phase,
                        exc=exc,
                    )
                raise
        metric = float(sum(fold_mse) / len(fold_mse))
        trial.set_user_attr("best_params", candidate_params)
        trial.set_user_attr("best_training_params", candidate_training_params)
        trial.set_user_attr("fold_mse", fold_mse)
        return metric

    study = optuna.create_study(
        sampler=sampler, direction=DEFAULT_OPTUNA_STUDY_DIRECTION
    )
    study.optimize(objective, n_trials=trial_count, show_progress_bar=False)
    best_params = dict(study.best_trial.user_attrs["best_params"])
    best_training_params = dict(study.best_trial.user_attrs["best_training_params"])
    summary = {
        **_trial_metrics_summary(study),
        "requested_mode": job.requested_mode,
        "validated_mode": job.validated_mode,
        "selected_search_params": list(job.selected_search_params),
        "selected_training_search_params": list(
            loaded.config.training_search.selected_search_params
        ),
        "best_params": best_params,
        "best_training_params": best_training_params,
        "fold_mse": study.best_trial.user_attrs["fold_mse"],
        "objective_stage": "tuning_pre_replay_direct_predictions",
    }
    return best_params, best_training_params, summary


def _score_residual_params(
    loaded: LoadedConfig,
    job: JobConfig,
    params: dict[str, Any],
    fold_payloads: list[dict[str, Any]],
) -> float:
    mse_scores: list[float] = []
    for payload in fold_payloads:
        plugin = build_residual_plugin({"model": loaded.config.residual.model, "params": params})
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
        corrected["y_hat_corrected"] = (
            corrected["y_hat_base"] + corrected["residual_hat"]
        )
        mse_scores.append(
            _compute_metrics(corrected["y"], corrected["y_hat_corrected"])["MSE"]
        )
    return float(sum(mse_scores) / len(mse_scores))


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


def _canonical_panel_columns(include_target: bool = True) -> list[str]:
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
    return columns


def _fold_artifact_dir(residual_root: Path, fold_idx: int) -> Path:
    return residual_root / "folds" / f"fold_{fold_idx:03d}"


def _build_fold_eval_panel(
    job: JobConfig,
    fold_idx: int,
    train_end_ds: object,
    target_predictions: pd.DataFrame,
    actuals: pd.Series,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(train_end_ds)
    rows: list[dict[str, object]] = []
    for row_idx, ds in enumerate(target_predictions["ds"]):
        y_hat_base = float(target_predictions[job.model].iloc[row_idx])
        y = float(actuals.iloc[row_idx])
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
                "residual_target": y - y_hat_base,
            }
        )
    return pd.DataFrame(rows, columns=_canonical_panel_columns()).reset_index(drop=True)


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
                    "residual_target": y - y_hat_base,
                }
            )
    return pd.DataFrame(rows, columns=_canonical_panel_columns()).reset_index(drop=True)


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
    corrected_groups: list[pd.DataFrame] = []
    checkpoint_metadata: dict[str, dict[str, object]] = {}
    total_backcast_rows = 0
    residual_params = {**DEFAULT_RESIDUAL_PARAMS, **loaded.config.residual.params}
    if loaded.config.residual.validated_mode == "residual_auto":
        sampler = build_optuna_sampler(optuna_seed(loaded.config.runtime.random_seed))

        def objective(trial: optuna.Trial) -> float:
            candidate_params = suggest_residual_params(
                loaded.config.residual.model,
                loaded.config.residual.selected_search_params,
                trial,
            )
            for payload in fold_payloads:
                payload["trial_dir"].mkdir(parents=True, exist_ok=True)
            score = _score_residual_params(loaded, job, candidate_params, fold_payloads)
            trial.set_user_attr("best_params", candidate_params)
            return score

        study = optuna.create_study(
            sampler=sampler, direction=DEFAULT_OPTUNA_STUDY_DIRECTION
        )
        study.optimize(objective, n_trials=optuna_num_trials(), show_progress_bar=False)
        residual_params = {**DEFAULT_RESIDUAL_PARAMS, **study.best_trial.user_attrs["best_params"]}
        (residual_root / "best_params.json").write_text(
            json.dumps(residual_params, indent=2), encoding="utf-8"
        )
        (residual_root / "optuna_study_summary.json").write_text(
            json.dumps(
                {
                    **_trial_metrics_summary(study),
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
        corrected["y_hat_corrected"] = (
            corrected["y_hat_base"] + corrected["residual_hat"]
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
        checkpoint_metadata[str(fold_idx)] = plugin.metadata()
        total_backcast_rows += len(backcast_panel)

    corrected_folds = pd.concat(corrected_groups, ignore_index=True)
    corrected_folds.to_csv(residual_root / "corrected_folds.csv", index=False)
    (residual_root / "plugin_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2), encoding="utf-8"
    )
    diagnostics = {
        "model": job.model,
        "residual_model": loaded.config.residual.model,
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
        total_steps *= optuna_num_trials() + 1
    progress = _ProgressLogger(job.model, total_steps)
    progress.model_started(
        total_folds=len(splits),
        detail=f"mode={job.validated_mode} output_root={run_root}",
    )

    if job.validated_mode == "learned_auto":
        best_params, best_training_params, study_summary = _tune_main_job(
            loaded,
            job,
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
                    effective_job,
                    fold_idx,
                    train_end_ds,
                    target_predictions,
                    target_actuals,
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
        print(json.dumps({"ok": True, "executed_jobs": [selected_jobs[0].model]}))
        return 0
    launches = build_launch_plan(loaded.config, selected_jobs)
    scheduler_dir = paths["run_root"] / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    (scheduler_dir / "launch_plan.json").write_text(
        json.dumps([launch.__dict__ for launch in launches], indent=2), encoding="utf-8"
    )
    results = run_parallel_jobs(repo_root, loaded, launches, scheduler_dir)
    if any(int(result["returncode"]) != 0 for result in results):
        raise SystemExit(json.dumps({"ok": False, "worker_results": results}))
    print(
        json.dumps(
            {
                "ok": True,
                "scheduled_jobs": [launch.__dict__ for launch in launches],
                "worker_results": results,
            }
        )
    )
    return 0
