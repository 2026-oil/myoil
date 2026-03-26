from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import optuna
import pandas as pd
import yaml
from neuralforecast import NeuralForecast
import numpy as np

from residual.adapters import build_multivariate_inputs, build_univariate_inputs
from residual.config import JobConfig, LoadedConfig
from residual.manifest import build_manifest, write_manifest
from residual.forecast_models import (
    BASELINE_MODEL_NAMES,
    MODEL_CLASSES,
    build_model,
    resolved_devices,
    resolved_strategy_name,
    validate_job,
)
from residual.optuna_spaces import (
    optuna_num_trials,
    suggest_model_params,
    suggest_training_params,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stage_root(run_root: Path) -> Path:
    return run_root / "bs_preforcast"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _assigned_gpu_ids(loaded: LoadedConfig, devices: int | None) -> list[int]:
    explicit_ids = os.environ.get("NEURALFORECAST_ASSIGNED_GPU_IDS", "").strip()
    if explicit_ids:
        return [int(part.strip()) for part in explicit_ids.split(",") if part.strip()]
    visible_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_ids:
        return [int(part.strip()) for part in visible_ids.split(",") if part.strip()]
    if devices is None:
        return []
    return list(loaded.config.scheduler.gpu_ids[:devices])


def _write_inline_fit_summary(
    loaded: LoadedConfig,
    *,
    run_root: Path | None,
    model_name: str,
) -> None:
    if run_root is None:
        return
    devices = resolved_devices(loaded.config)
    payload = {
        "model": model_name,
        "devices": devices,
        "strategy": resolved_strategy_name(loaded.config, devices),
        "assigned_gpu_ids": _assigned_gpu_ids(loaded, devices),
    }
    _write_json(
        _stage_root(run_root) / "artifacts" / "inline_fit_summary.json",
        payload,
    )


def resolve_bs_preforcast_injection_mode(
    loaded: LoadedConfig,
    *,
    selected_jobs: Iterable[Any],
) -> str:
    if not loaded.config.bs_preforcast.enabled:
        return "disabled"
    if not loaded.config.bs_preforcast.using_futr_exog:
        return "lag_derived"
    jobs = list(selected_jobs)
    if jobs and all(validate_job(job).supports_futr_exog for job in jobs):
        return "futr_exog"
    return "lag_derived"


def _stage_capability_payload(stage_loaded: LoadedConfig) -> dict[str, Any]:
    def _caps_for_stage_job(job: JobConfig) -> dict[str, Any]:
        try:
            return validate_job(job).__dict__
        except Exception:
            return {
                "name": job.model,
                "multivariate": False,
                "supports_hist_exog": job.model in {"xgboost", "lightgbm"},
                "supports_futr_exog": False,
                "supports_stat_exog": False,
                "requires_n_series": False,
                "single_device_only": True,
            }

    payload: dict[str, Any] = {
        "bs_preforcast": {
            "enabled": True,
            "config_path": stage_loaded.normalized_payload["bs_preforcast"].get(
                "config_path"
            ),
            "using_futr_exog": bool(
                stage_loaded.normalized_payload["bs_preforcast"]["using_futr_exog"]
            ),
            "target_columns": list(
                stage_loaded.normalized_payload["bs_preforcast"]["target_columns"]
            ),
            "multivariable": bool(
                stage_loaded.normalized_payload["bs_preforcast"]["task"][
                    "multivariable"
                ]
            ),
            "selected_config_path": stage_loaded.normalized_payload["bs_preforcast"].get(
                "selected_config_path",
                stage_loaded.normalized_payload["bs_preforcast"].get("config_path"),
            ),
        }
    }
    for job in stage_loaded.config.jobs:
        caps = _caps_for_stage_job(job)
        payload[job.model] = {
            **caps,
            "requested_mode": job.requested_mode,
            "validated_mode": job.validated_mode,
            "supports_auto": bool(job.selected_search_params),
            "search_space_entry_found": bool(job.selected_search_params),
            "selected_search_params": list(job.selected_search_params),
            "unknown_search_params": [],
            "validation_error": None,
        }
    payload["training_search"] = {
        "requested_mode": stage_loaded.config.training_search.requested_mode,
        "validated_mode": stage_loaded.config.training_search.validated_mode,
        "supports_auto": bool(stage_loaded.config.training_search.selected_search_params),
        "search_space_entry_found": bool(
            stage_loaded.config.training_search.selected_search_params
        ),
        "selected_search_params": list(
            stage_loaded.config.training_search.selected_search_params
        ),
        "unknown_search_params": [],
        "validation_error": None,
    }
    return payload


def _write_stage_dashboard(
    stage_root: Path,
    *,
    target_columns: list[str],
    injection_mode: str,
    stage_run_roots: list[str] | None = None,
) -> Path:
    path = stage_root / "summary" / "dashboard.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# bs_preforcast dashboard",
        "",
        f"- injection_mode: {injection_mode}",
        f"- target_columns: {', '.join(target_columns)}",
    ]
    if stage_run_roots:
        lines.append("- stage_run_roots:")
        lines.extend(f"  - {item}" for item in stage_run_roots)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_stage_forecast_artifact(
    stage_root: Path,
    *,
    loaded: LoadedConfig,
    stage_run_roots: list[Path] | None = None,
) -> Path:
    frame_rows: list[dict[str, Any]] = []
    if stage_run_roots:
        for run_root in stage_run_roots:
            for forecast_path in sorted((run_root / "cv").glob("*_forecasts.csv")):
                frame = pd.read_csv(forecast_path)
                if frame.empty:
                    continue
                for _, row in frame.iterrows():
                    frame_rows.append(
                        {
                            "target_column": str(row.get("unique_id", run_root.name)),
                            "horizon_step": int(row.get("horizon_step", 0) or 0),
                            "forecast": float(row.get("y_hat", 0.0)),
                            "source_dt": str(row.get("ds", "")),
                            "stage_run_root": str(run_root),
                        }
                    )
    path = stage_root / "artifacts" / "bs_preforcast_forecasts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_rows:
        pd.DataFrame(
            columns=[
                "target_column",
                "horizon_step",
                "forecast",
                "source_dt",
                "stage_run_root",
            ]
        ).to_csv(path, index=False)
        return path
    pd.DataFrame(frame_rows).to_csv(path, index=False)
    return path


def _forecast_column_name(column: str) -> str:
    return f"bs_preforcast_futr__{column}"


def _last_value_forecast(
    train_df: pd.DataFrame,
    *,
    column: str,
    horizon: int,
) -> list[float]:
    if column not in train_df.columns:
        raise ValueError(f"bs_preforcast target column missing from dataset: {column}")
    history = train_df[column].dropna()
    if history.empty:
        raise ValueError(f"bs_preforcast target column has no non-null history: {column}")
    last_value = float(history.iloc[-1])
    return [last_value] * horizon


def _stage_prediction_column(predictions: pd.DataFrame, model_name: str) -> str:
    if model_name in predictions.columns:
        return model_name
    raise KeyError(f"Could not find bs_preforcast prediction column for {model_name}")


def _stage_execution_loaded(loaded: LoadedConfig) -> LoadedConfig:
    if loaded.bs_preforcast_stage1 is None:
        raise ValueError("bs_preforcast stage1 config is required when enabled")
    return LoadedConfig(
        config=loaded.bs_preforcast_stage1.config,
        source_path=loaded.bs_preforcast_stage1.source_path,
        source_type=loaded.bs_preforcast_stage1.source_type,
        normalized_payload=loaded.bs_preforcast_stage1.normalized_payload,
        input_hash=loaded.bs_preforcast_stage1.input_hash,
        resolved_hash=loaded.bs_preforcast_stage1.resolved_hash,
        search_space_path=loaded.bs_preforcast_stage1.search_space_path,
        search_space_hash=loaded.bs_preforcast_stage1.search_space_hash,
        search_space_payload=loaded.bs_preforcast_stage1.search_space_payload,
    )


def _single_stage_job(loaded: LoadedConfig) -> JobConfig:
    jobs = list(loaded.config.jobs)
    if len(jobs) != 1:
        raise ValueError("bs_preforcast stage currently requires exactly one configured job")
    job = jobs[0]
    if job.model in BASELINE_MODEL_NAMES:
        raise ValueError("bs_preforcast stage does not support baseline-only jobs")
    return job


def _stage_run_roots_from_loaded(loaded: LoadedConfig) -> list[Path]:
    payload = loaded.normalized_payload.get("bs_preforcast", {})
    roots = payload.get("stage1_run_roots", [])
    if not isinstance(roots, list):
        return []
    return [Path(str(item)) for item in roots]


def _load_stage_best_params(
    loaded: LoadedConfig,
    *,
    model_name: str,
    variant_slug: str,
) -> dict[str, Any]:
    for run_root in _stage_run_roots_from_loaded(loaded):
        if run_root.name != variant_slug:
            continue
        best_params_path = run_root / "models" / model_name / "best_params.json"
        if not best_params_path.exists():
            continue
        payload = _load_json(best_params_path)
        if "stage_season_length" in payload and "season_length" not in payload:
            payload["season_length"] = payload.pop("stage_season_length")
        return payload
    raise NotImplementedError(
        "bs_preforcast learned_auto stage job requires materialized best_params artifact before fold injection"
    )


def _resolved_stage_job(
    loaded: LoadedConfig,
    stage_loaded: LoadedConfig,
    *,
    variant_slug: str,
) -> JobConfig:
    job = _single_stage_job(stage_loaded)
    if job.validated_mode != "learned_auto":
        return job
    best_params = _load_stage_best_params(
        loaded,
        model_name=job.model,
        variant_slug=variant_slug,
    )
    return replace(
        job,
        params=best_params,
        requested_mode="learned_fixed",
        validated_mode="learned_fixed",
        selected_search_params=(),
    )


def _load_stage_training_best_params(
    loaded: LoadedConfig,
    *,
    model_name: str,
    variant_slug: str,
) -> dict[str, Any]:
    for run_root in _stage_run_roots_from_loaded(loaded):
        if run_root.name != variant_slug:
            continue
        best_params_path = run_root / "models" / model_name / "training_best_params.json"
        if not best_params_path.exists():
            continue
        return _load_json(best_params_path)
    return {}


def _stage_training_specs(
    stage_loaded: LoadedConfig,
    *,
    model_name: str,
) -> dict[str, Any]:
    payload = stage_loaded.search_space_payload or {}
    training_payload = payload.get("bs_preforcast_training", {})
    per_model = training_payload.get("per_model", {})
    if model_name in per_model:
        return dict(per_model[model_name])
    return dict(training_payload.get("global", {}))


def _stage_loaded_with_job_and_training(
    stage_loaded: LoadedConfig,
    *,
    job: JobConfig,
    training_overrides: dict[str, Any] | None = None,
) -> LoadedConfig:
    training = stage_loaded.config.training
    if training_overrides:
        training = replace(training, **training_overrides)
    return replace(
        stage_loaded,
        config=replace(stage_loaded.config, jobs=(job,), training=training),
    )


def _resolved_stage_loaded(
    loaded: LoadedConfig,
    stage_loaded: LoadedConfig,
    *,
    variant_slug: str,
) -> LoadedConfig:
    resolved_job = _resolved_stage_job(
        loaded,
        stage_loaded,
        variant_slug=variant_slug,
    )
    training_overrides = {}
    if stage_loaded.config.training_search.validated_mode == "training_auto":
        training_overrides = _load_stage_training_best_params(
            loaded,
            model_name=resolved_job.model,
            variant_slug=variant_slug,
        )
    return _stage_loaded_with_job_and_training(
        stage_loaded,
        job=resolved_job,
        training_overrides=training_overrides,
    )


def _mean_mape(actual: pd.Series, forecast: list[float]) -> float:
    actual_series = actual.reset_index(drop=True).astype(float)
    pred_series = pd.Series(forecast, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (actual_series.sub(pred_series).abs() / actual_series.abs()).mean()
    return float(value)


def _optuna_study_summary(
    study: optuna.Study,
    *,
    best_params: dict[str, Any],
    best_training_params: dict[str, Any],
) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    for trial in study.trials:
        state = trial.state.name.lower()
        state_counts[state] = state_counts.get(state, 0) + 1
    return {
        "direction": study.direction.name.lower(),
        "trial_count": len(study.trials),
        "finished_trial_count": sum(1 for trial in study.trials if trial.state.is_finished()),
        "state_counts": state_counts,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": best_params,
        "best_training_params": best_training_params,
    }


def _stage_fit_loaded(
    stage_loaded: LoadedConfig,
    *,
    horizon: int,
    train_length: int,
) -> LoadedConfig:
    adjusted_input_size = max(
        1,
        min(int(stage_loaded.config.training.input_size), max(train_length - 1, horizon)),
    )
    adjusted_training = replace(
        stage_loaded.config.training,
        input_size=adjusted_input_size,
        val_size=0,
    )
    adjusted_cv = replace(stage_loaded.config.cv, horizon=horizon)
    return replace(
        stage_loaded,
        config=replace(stage_loaded.config, training=adjusted_training, cv=adjusted_cv),
    )


def _predict_stage_univariate(
    loaded: LoadedConfig,
    stage_loaded: LoadedConfig,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    run_root: Path | None = None,
) -> list[float]:
    stage_loaded = _resolved_stage_loaded(
        loaded,
        stage_loaded,
        variant_slug=target_column,
    )
    job = _single_stage_job(stage_loaded)
    if job.model == "AutoARIMA":
        return _predict_stage_univariate_autoarima(
            stage_loaded,
            job,
            target_column=target_column,
            train_df=train_df,
            future_df=future_df,
        )
    if job.model == "ES":
        return _predict_stage_univariate_es(
            stage_loaded,
            job,
            target_column=target_column,
            train_df=train_df,
            future_df=future_df,
        )
    if job.model in {"xgboost", "lightgbm"}:
        return _predict_stage_univariate_tree(
            stage_loaded,
            job,
            target_column=target_column,
            train_df=train_df,
            future_df=future_df,
            model_name=job.model,
        )
    stage_loaded = _stage_fit_loaded(
        stage_loaded,
        horizon=len(future_df),
        train_length=len(train_df),
    )
    stage_dataset = replace(stage_loaded.config.dataset, target_col=target_column)
    stage_loaded = replace(stage_loaded, config=replace(stage_loaded.config, dataset=stage_dataset))
    adapter_inputs = build_univariate_inputs(
        train_df,
        job,
        dataset=stage_loaded.config.dataset,
        dt_col=stage_loaded.config.dataset.dt_col,
        future_df=future_df,
    )
    model = build_model(stage_loaded.config, job)
    _write_inline_fit_summary(stage_loaded, run_root=run_root, model_name=job.model)
    nf = NeuralForecast(models=[model], freq=stage_loaded.config.dataset.freq or "W")
    nf.fit(
        adapter_inputs.fit_df,
        static_df=adapter_inputs.static_df,
        val_size=stage_loaded.config.training.val_size,
    )
    predict_kwargs: dict[str, Any] = {}
    if adapter_inputs.futr_df is not None:
        predict_kwargs["futr_df"] = adapter_inputs.futr_df
    if adapter_inputs.static_df is not None:
        predict_kwargs["static_df"] = adapter_inputs.static_df
    predictions = nf.predict(**predict_kwargs)
    pred_col = _stage_prediction_column(predictions, job.model)
    selected = predictions[predictions["unique_id"] == target_column][pred_col].reset_index(drop=True)
    return [float(value) for value in selected.tolist()]


def _stage_infer_freq(stage_loaded: LoadedConfig, train_df: pd.DataFrame) -> str:
    if stage_loaded.config.dataset.freq:
        return stage_loaded.config.dataset.freq
    inferred = pd.infer_freq(pd.to_datetime(train_df[stage_loaded.config.dataset.dt_col]))
    return inferred or "W"


def _predict_stage_univariate_autoarima(
    stage_loaded: LoadedConfig,
    job: JobConfig,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> list[float]:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    dt_col = stage_loaded.config.dataset.dt_col
    fit_df = train_df[[dt_col, target_column]].copy()
    fit_df.rename(columns={dt_col: "ds", target_column: "y"}, inplace=True)
    fit_df["ds"] = pd.to_datetime(fit_df["ds"])
    fit_df.insert(0, "unique_id", target_column)
    season_length = max(
        1,
        int(
            job.params.get(
                "season_length",
                job.params.get(
                    "stage_season_length",
                    stage_loaded.config.training.season_length,
                ),
            )
        ),
    )
    model = AutoARIMA(season_length=season_length)
    sf = StatsForecast(models=[model], freq=_stage_infer_freq(stage_loaded, train_df))
    fitted = sf.fit(df=fit_df)
    predictions = fitted.predict(h=len(future_df))
    pred_col = next(
        column for column in predictions.columns if column not in {"unique_id", "ds"}
    )
    return [float(value) for value in predictions[pred_col].tolist()]


def _predict_stage_univariate_es(
    stage_loaded: LoadedConfig,
    job: JobConfig,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> list[float]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = train_df[target_column].astype(float)
    season_length = int(
        job.params.get(
            "season_length",
            job.params.get(
                "stage_season_length",
                stage_loaded.config.training.season_length,
            ),
        )
    )
    kwargs: dict[str, Any] = {
        "trend": None,
        "seasonal": None,
        "initialization_method": "estimated",
    }
    if season_length > 1 and len(series) >= season_length * 2:
        kwargs["seasonal"] = "add"
        kwargs["seasonal_periods"] = season_length
    fitted = ExponentialSmoothing(
        series,
        **kwargs,
    ).fit()
    forecast = fitted.forecast(len(future_df))
    return [float(value) for value in forecast.tolist()]


def _predict_stage_univariate_tree(
    stage_loaded: LoadedConfig,
    job: JobConfig,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    model_name: str,
) -> list[float]:
    series = train_df[target_column].astype(float).tolist()
    max_lag = max(1, min(int(stage_loaded.config.training.input_size), len(series) - 1))
    if len(series) <= max_lag:
        return _last_value_forecast(train_df, column=target_column, horizon=len(future_df))
    X: list[list[float]] = []
    y: list[float] = []
    for idx in range(max_lag, len(series)):
        X.append(series[idx - max_lag : idx])
        y.append(series[idx])
    params = dict(job.params)
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=int(params.pop("n_estimators", 32)),
            max_depth=int(params.pop("max_depth", 3)),
            learning_rate=float(params.pop("learning_rate", 0.1)),
            objective="reg:squarederror",
            n_jobs=1,
            verbosity=0,
            **params,
        )
    else:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=int(params.pop("n_estimators", 64)),
            max_depth=int(params.pop("max_depth", -1)),
            learning_rate=float(params.pop("learning_rate", 0.05)),
            verbosity=-1,
            **params,
        )
    model.fit(np.asarray(X), np.asarray(y))
    history = list(series)
    forecasts: list[float] = []
    for _ in range(len(future_df)):
        features = np.asarray([history[-max_lag:]], dtype=float)
        value = float(model.predict(features)[0])
        forecasts.append(value)
        history.append(value)
    return forecasts


def _predict_stage_multivariate(
    loaded: LoadedConfig,
    stage_loaded: LoadedConfig,
    *,
    target_columns: list[str],
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    run_root: Path | None = None,
) -> dict[str, list[float]]:
    stage_loaded = _resolved_stage_loaded(
        loaded,
        stage_loaded,
        variant_slug="multivariable",
    )
    job = _single_stage_job(stage_loaded)
    if job.model in {"AutoARIMA", "ES", "xgboost", "lightgbm"}:
        raise ValueError(
            f"bs_preforcast multivariable execution is not supported for stage model {job.model}"
        )
    stage_loaded = _stage_fit_loaded(
        stage_loaded,
        horizon=len(future_df),
        train_length=len(train_df),
    )
    adapter_dataset = replace(
        stage_loaded.config.dataset,
        target_col=target_columns[0],
        hist_exog_cols=tuple(target_columns[1:]),
    )
    model_dataset = replace(
        stage_loaded.config.dataset,
        target_col=target_columns[0],
        hist_exog_cols=(),
    )
    stage_loaded = replace(
        stage_loaded, config=replace(stage_loaded.config, dataset=model_dataset)
    )
    adapter_inputs = build_multivariate_inputs(
        train_df,
        job,
        dataset=adapter_dataset,
        dt_col=stage_loaded.config.dataset.dt_col,
        future_df=future_df,
    )
    model = build_model(
        stage_loaded.config,
        job,
        n_series=adapter_inputs.metadata.get("n_series"),
    )
    _write_inline_fit_summary(stage_loaded, run_root=run_root, model_name=job.model)
    nf = NeuralForecast(models=[model], freq=stage_loaded.config.dataset.freq or "W")
    nf.fit(
        adapter_inputs.fit_df,
        static_df=adapter_inputs.static_df,
        val_size=stage_loaded.config.training.val_size,
    )
    predict_kwargs: dict[str, Any] = {}
    if adapter_inputs.futr_df is not None:
        predict_kwargs["futr_df"] = adapter_inputs.futr_df
    if adapter_inputs.static_df is not None:
        predict_kwargs["static_df"] = adapter_inputs.static_df
    predictions = nf.predict(**predict_kwargs)
    pred_col = _stage_prediction_column(predictions, job.model)
    out: dict[str, list[float]] = {}
    for column in target_columns:
        selected = predictions[predictions["unique_id"] == column][pred_col].reset_index(drop=True)
        out[column] = [float(value) for value in selected.tolist()]
    return out


def compute_bs_preforcast_fold_forecasts(
    loaded: LoadedConfig,
    *,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    run_root: Path | None = None,
) -> dict[str, list[float]]:
    stage_loaded = _stage_execution_loaded(loaded)
    target_columns = list(loaded.config.bs_preforcast.target_columns)
    if loaded.config.bs_preforcast.task.multivariable:
        return _predict_stage_multivariate(
            loaded,
            stage_loaded,
            target_columns=target_columns,
            train_df=train_df,
            future_df=future_df,
            run_root=run_root,
        )
    return {
        column: _predict_stage_univariate(
            loaded,
            stage_loaded,
            target_column=column,
            train_df=train_df,
            future_df=future_df,
            run_root=run_root,
        )
        for column in target_columns
    }


def prepare_bs_preforcast_fold_inputs(
    loaded: LoadedConfig,
    job: JobConfig,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    run_root: Path | None = None,
) -> tuple[LoadedConfig, pd.DataFrame, pd.DataFrame, str]:
    injection_mode = resolve_bs_preforcast_injection_mode(
        loaded,
        selected_jobs=[job],
    )
    if not loaded.config.bs_preforcast.enabled:
        return loaded, train_df, future_df, injection_mode
    target_columns = list(loaded.config.bs_preforcast.target_columns)
    horizon = len(future_df)
    train_frame = train_df.copy()
    future_frame = future_df.copy()
    forecasts_by_column = compute_bs_preforcast_fold_forecasts(
        loaded,
        train_df=train_df,
        future_df=future_df,
        run_root=run_root,
    )
    futr_columns = list(loaded.config.dataset.futr_exog_cols)
    hist_columns = list(loaded.config.dataset.hist_exog_cols)

    for column in target_columns:
        forecast_values = forecasts_by_column.get(column)
        if not forecast_values:
            forecast_values = _last_value_forecast(
                train_df,
                column=column,
                horizon=horizon,
            )
        forecast_column = _forecast_column_name(column)
        if injection_mode == "futr_exog":
            train_frame[forecast_column] = train_frame[column].astype(float)
            future_frame[forecast_column] = forecast_values
            if forecast_column not in futr_columns:
                futr_columns.append(forecast_column)
            continue
        train_frame[forecast_column] = train_frame[column].astype(float)
        overwrite_count = min(len(train_frame), len(forecast_values))
        if overwrite_count:
            train_frame.loc[
                len(train_frame) - overwrite_count :,
                forecast_column,
            ] = forecast_values[:overwrite_count]
        future_frame[forecast_column] = forecast_values
        if forecast_column not in hist_columns:
            hist_columns.append(forecast_column)

    updated_dataset = replace(
        loaded.config.dataset,
        hist_exog_cols=tuple(hist_columns),
        futr_exog_cols=tuple(futr_columns),
    )
    updated_loaded = replace(
        loaded,
        config=replace(loaded.config, dataset=updated_dataset),
    )
    return updated_loaded, train_frame, future_frame, injection_mode


def _stage_variant_payloads(stage_loaded: LoadedConfig) -> list[tuple[str, dict[str, Any]]]:
    target_columns = list(
        stage_loaded.normalized_payload["bs_preforcast"]["target_columns"]
    )
    if not target_columns:
        return []
    base_payload = stage_loaded.config.to_dict()
    base_payload.pop("bs_preforcast", None)
    base_payload.pop("training_search", None)
    base_payload.pop("search_space_path", None)
    base_payload.pop("search_space_sha256", None)
    for job in base_payload.get("jobs", []):
        job.pop("requested_mode", None)
        job.pop("validated_mode", None)
        job.pop("selected_search_params", None)
    residual_payload = base_payload.get("residual")
    if isinstance(residual_payload, dict):
        residual_payload.pop("requested_mode", None)
        residual_payload.pop("validated_mode", None)
        residual_payload.pop("selected_search_params", None)
    if stage_loaded.config.bs_preforcast.task.multivariable:
        payload = json.loads(json.dumps(base_payload))
        payload.setdefault("dataset", {})
        payload["dataset"]["target_col"] = target_columns[0]
        payload["dataset"]["hist_exog_cols"] = list(target_columns[1:])
        payload["dataset"]["futr_exog_cols"] = []
        payload.setdefault("task", {})["name"] = (
            f"{payload.get('task', {}).get('name') or 'bs_preforcast'}_multivariable"
        )
        return [("multivariable", payload)]
    variants: list[tuple[str, dict[str, Any]]] = []
    for column in target_columns:
        payload = json.loads(json.dumps(base_payload))
        payload.setdefault("dataset", {})
        payload["dataset"]["target_col"] = column
        payload.setdefault("task", {})["name"] = (
            f"{payload.get('task', {}).get('name') or 'bs_preforcast'}_{column}"
        )
        variants.append((column, payload))
    return variants


def _load_stage_variant_frames(
    payload: dict[str, Any],
    *,
    config_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    dataset = payload.get("dataset", {})
    data_path = Path(str(dataset.get("path", "")))
    if not data_path.is_absolute():
        data_path = (config_path.parent / data_path).resolve()
    frame = pd.read_csv(data_path)
    dt_col = str(dataset.get("dt_col", "dt"))
    horizon = int(payload.get("cv", {}).get("horizon", 1))
    frame = frame.sort_values(dt_col).reset_index(drop=True)
    if len(frame) <= horizon:
        raise ValueError("bs_preforcast stage direct run needs more than horizon rows")
    train_df = frame.iloc[:-horizon].reset_index(drop=True)
    future_df = frame.iloc[-horizon:].reset_index(drop=True)
    return train_df, future_df, str(dataset.get("target_col"))


def _write_direct_stage_artifacts(
    stage_run_root: Path,
    *,
    model_name: str,
    target_column: str,
    forecasts: list[float],
    future_df: pd.DataFrame,
    dt_col: str,
    best_params: dict[str, Any] | None = None,
    best_training_params: dict[str, Any] | None = None,
    study_summary: dict[str, Any] | None = None,
) -> None:
    (stage_run_root / "summary").mkdir(parents=True, exist_ok=True)
    (stage_run_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (stage_run_root / "cv").mkdir(parents=True, exist_ok=True)
    models_dir = stage_run_root / "models" / model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "target_column": [target_column] * len(forecasts),
            "horizon_step": list(range(1, len(forecasts) + 1)),
            "forecast": forecasts,
        }
    ).to_csv(stage_run_root / "artifacts" / "forecasts.csv", index=False)
    pd.DataFrame(
        {
            "unique_id": [target_column] * len(forecasts),
            "ds": pd.to_datetime(future_df[dt_col]).astype(str).tolist(),
            "horizon_step": list(range(1, len(forecasts) + 1)),
            "y_hat": forecasts,
        }
    ).to_csv(stage_run_root / "cv" / f"{model_name}_forecasts.csv", index=False)
    pd.DataFrame(
        [{"model": model_name, "target_column": target_column}]
    ).to_csv(stage_run_root / "summary" / "leaderboard.csv", index=False)
    (stage_run_root / "summary" / "sample.md").write_text(
        f"# bs_preforcast stage run\n\n- model: {model_name}\n- target_column: {target_column}\n",
        encoding="utf-8",
    )
    if best_params is not None:
        (models_dir / "best_params.json").write_text(
            json.dumps(best_params, indent=2),
            encoding="utf-8",
        )
    if best_training_params is not None:
        (models_dir / "training_best_params.json").write_text(
            json.dumps(best_training_params, indent=2),
            encoding="utf-8",
        )
    if study_summary is not None:
        (models_dir / "optuna_study_summary.json").write_text(
            json.dumps(study_summary, indent=2),
            encoding="utf-8",
        )
        (models_dir / "training_optuna_study_summary.json").write_text(
            json.dumps(study_summary, indent=2),
            encoding="utf-8",
        )


def _run_direct_stage_variant(
    stage_loaded: LoadedConfig,
    *,
    variant_slug: str,
    payload: dict[str, Any],
    config_path: Path,
    stage_run_root: Path,
) -> None:
    if stage_loaded.config.bs_preforcast.task.multivariable:
        raise ValueError("bs_preforcast direct stage execution does not support multivariable stage-only models")
    stage_job = _single_stage_job(stage_loaded)
    train_df, future_df, target_column = _load_stage_variant_frames(
        payload,
        config_path=config_path,
    )
    dt_col = str(payload.get("dataset", {}).get("dt_col", "dt"))
    best_job = stage_job
    best_training_params: dict[str, Any] = {}
    best_forecasts: list[float] | None = None
    study_summary: dict[str, Any] | None = None

    if stage_job.validated_mode == "learned_auto":
        model_specs = (
            (stage_loaded.search_space_payload or {})
            .get("bs_preforcast_models", {})
            .get(stage_job.model, {})
        )
        training_specs = _stage_training_specs(
            stage_loaded,
            model_name=stage_job.model,
        )
        selected_training = tuple(stage_loaded.config.training_search.selected_search_params)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=stage_loaded.config.runtime.random_seed),
        )
        trial_total = optuna_num_trials(stage_loaded.config.runtime.opt_n_trial)
        best_score: float | None = None
        for _ in range(trial_total):
            trial = study.ask()
            candidate_params = suggest_model_params(
                stage_job.model,
                tuple(stage_job.selected_search_params),
                trial,
                param_specs=model_specs,
                name_prefix="stage_model_",
            )
            candidate_training = (
                suggest_training_params(
                    selected_training,
                    trial,
                    param_specs=training_specs,
                    name_prefix="stage_training_",
                )
                if selected_training
                else {}
            )
            candidate_job = replace(
                stage_job,
                params=candidate_params,
                requested_mode="learned_fixed",
                validated_mode="learned_fixed",
                selected_search_params=(),
            )
            candidate_loaded = _stage_loaded_with_job_and_training(
                stage_loaded,
                job=candidate_job,
                training_overrides=candidate_training,
            )
            forecasts = _predict_stage_univariate(
                candidate_loaded,
                candidate_loaded,
                target_column=target_column,
                train_df=train_df,
                future_df=future_df,
            )
            score = _mean_mape(future_df[target_column], forecasts)
            study.tell(trial, score)
            if best_score is None or score < best_score:
                best_score = score
                best_job = candidate_job
                best_training_params = candidate_training
                best_forecasts = forecasts
        study_summary = _optuna_study_summary(
            study,
            best_params=best_job.params,
            best_training_params=best_training_params,
        )
    if best_forecasts is None:
        direct_loaded = _stage_loaded_with_job_and_training(
            stage_loaded,
            job=best_job,
            training_overrides=best_training_params,
        )
        best_forecasts = _predict_stage_univariate(
            direct_loaded,
            direct_loaded,
            target_column=target_column,
            train_df=train_df,
            future_df=future_df,
        )
    _write_direct_stage_artifacts(
        stage_run_root,
        model_name=best_job.model,
        target_column=target_column,
        forecasts=best_forecasts,
        future_df=future_df,
        dt_col=dt_col,
        best_params=(best_job.params if stage_job.validated_mode == "learned_auto" else None),
        best_training_params=(
            best_training_params
            if stage_job.validated_mode == "learned_auto"
            else None
        ),
        study_summary=study_summary,
    )


def _run_stage_variants(stage_loaded: LoadedConfig, *, run_root: Path) -> list[Path]:
    repo_root = _repo_root()
    stage_root = _stage_root(run_root)
    temp_dir = stage_root / "temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    run_roots: list[Path] = []
    stage_job = _single_stage_job(stage_loaded)
    for slug, payload in _stage_variant_payloads(stage_loaded):
        config_path = temp_dir / f"{slug}.yaml"
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        stage_run_root = stage_root / "runs" / slug
        if stage_job.model in MODEL_CLASSES or stage_job.model in BASELINE_MODEL_NAMES:
            subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "main.py"),
                    "--config",
                    str(config_path),
                    "--output-root",
                    str(stage_run_root),
                ],
                cwd=repo_root,
                check=True,
            )
        else:
            _run_direct_stage_variant(
                stage_loaded,
                variant_slug=slug,
                payload=payload,
                config_path=config_path,
                stage_run_root=stage_run_root,
            )
        run_roots.append(stage_run_root)
    return run_roots


def materialize_bs_preforcast_stage(
    *,
    loaded: LoadedConfig,
    selected_jobs: Iterable[Any],
    run_root: Path,
    main_resolved_path: Path,
    main_capability_path: Path,
    main_manifest_path: Path,
    entrypoint_version: str,
    validate_only: bool,
) -> dict[str, Any]:
    if not loaded.config.bs_preforcast.enabled:
        return {}
    stage_loaded = load_bs_preforcast_stage_config(_repo_root(), loaded)
    if stage_loaded is None:
        return {}

    stage_root = _stage_root(run_root)
    resolved_path = stage_root / "config" / "config.resolved.json"
    capability_path = stage_root / "config" / "capability_report.json"
    manifest_path = stage_root / "manifest" / "run_manifest.json"
    _write_json(resolved_path, stage_loaded.normalized_payload)
    _write_json(capability_path, _stage_capability_payload(stage_loaded))
    write_manifest(
        manifest_path,
        build_manifest(
            stage_loaded,
            compat_mode="dual_read",
            entrypoint_version=entrypoint_version,
            resolved_config_path=resolved_path,
        ),
    )
    stage_run_roots: list[Path] = []
    if not validate_only:
        stage_run_roots = _run_stage_variants(stage_loaded, run_root=run_root)
    metadata = attach_bs_preforcast_stage_metadata(
        loaded=loaded,
        selected_jobs=selected_jobs,
        run_root=run_root,
        main_resolved_path=main_resolved_path,
        main_capability_path=main_capability_path,
        main_manifest_path=main_manifest_path,
        validate_only=validate_only,
        stage_loaded=stage_loaded,
        stage_run_roots=stage_run_roots,
    )
    metadata.update(
        {
            "stage1_resolved_config_path": str(resolved_path),
            "stage1_capability_report_path": str(capability_path),
            "stage1_manifest_path": str(manifest_path),
            "stage1_run_roots": [str(path) for path in stage_run_roots],
        }
    )
    loaded.normalized_payload.setdefault("bs_preforcast", {}).update(metadata)
    for path in (main_resolved_path, main_capability_path, main_manifest_path):
        payload = _load_json(path)
        payload.setdefault("bs_preforcast", {})
        payload["bs_preforcast"].update(metadata)
        _write_json(path, payload)
    return metadata


def load_bs_preforcast_stage_config(
    _repo_root: Path,
    loaded: LoadedConfig,
) -> LoadedConfig | None:
    if not loaded.config.bs_preforcast.enabled:
        return None
    if loaded.bs_preforcast_stage1 is None:
        return None
    return LoadedConfig(
        config=replace(
            loaded.bs_preforcast_stage1.config,
            bs_preforcast=loaded.config.bs_preforcast,
        ),
        source_path=loaded.bs_preforcast_stage1.source_path,
        source_type=loaded.bs_preforcast_stage1.source_type,
        normalized_payload=loaded.bs_preforcast_stage1.normalized_payload,
        input_hash=loaded.bs_preforcast_stage1.input_hash,
        resolved_hash=loaded.bs_preforcast_stage1.resolved_hash,
        search_space_path=loaded.bs_preforcast_stage1.search_space_path,
        search_space_hash=loaded.bs_preforcast_stage1.search_space_hash,
        search_space_payload=loaded.bs_preforcast_stage1.search_space_payload,
    )


def write_bs_preforcast_dashboard(
    stage_root: Path,
    *,
    target_columns: list[str],
    injection_mode: str,
    stage_run_roots: list[str] | None = None,
) -> Path:
    return _write_stage_dashboard(
        stage_root,
        target_columns=target_columns,
        injection_mode=injection_mode,
        stage_run_roots=stage_run_roots,
    )


def attach_bs_preforcast_stage_metadata(
    *,
    loaded: LoadedConfig,
    selected_jobs: Iterable[Any],
    run_root: Path,
    main_resolved_path: Path,
    main_capability_path: Path,
    main_manifest_path: Path,
    validate_only: bool,
    stage_loaded: LoadedConfig | None = None,
    stage_run_roots: list[Path] | None = None,
) -> dict[str, Any]:
    if not loaded.config.bs_preforcast.enabled:
        return {}
    stage_loaded = stage_loaded or load_bs_preforcast_stage_config(_repo_root(), loaded)
    if stage_loaded is None:
        return {}
    injection_mode = resolve_bs_preforcast_injection_mode(
        loaded,
        selected_jobs=selected_jobs,
    )
    stage_root = _stage_root(run_root)
    dashboard_path = write_bs_preforcast_dashboard(
        stage_root,
        target_columns=list(loaded.config.bs_preforcast.target_columns),
        injection_mode=injection_mode,
        stage_run_roots=[] if stage_run_roots is None else [str(path) for path in stage_run_roots],
    )
    forecast_path = _write_stage_forecast_artifact(
        stage_root,
        loaded=loaded,
        stage_run_roots=stage_run_roots,
    )
    metadata = {
        "selected_config_path": str(stage_loaded.source_path),
        "injection_mode": injection_mode,
        "stage1_dashboard_path": str(dashboard_path),
        "stage1_forecast_artifact_path": str(forecast_path),
        "stage1_run_roots": [] if stage_run_roots is None else [str(path) for path in stage_run_roots],
        "target_columns_used_for_injection": list(
            loaded.config.bs_preforcast.target_columns
        ),
        "validate_only": validate_only,
    }
    for path in (main_resolved_path, main_capability_path):
        payload = _load_json(path)
        payload.setdefault("bs_preforcast", {})
        payload["bs_preforcast"].update(metadata)
        _write_json(path, payload)
    manifest_payload = _load_json(main_manifest_path)
    manifest_payload.setdefault("bs_preforcast", {})
    manifest_payload["bs_preforcast"].update(
        {key: value for key, value in metadata.items() if key != "selected_config_path"}
    )
    _write_json(main_manifest_path, manifest_payload)
    return metadata


def run_bs_preforcast_stage(
    *,
    loaded: LoadedConfig,
    selected_jobs: Iterable[Any],
    run_root: Path,
    main_resolved_path: Path,
    main_capability_path: Path,
    main_manifest_path: Path,
    entrypoint_version: str,
    validate_only: bool,
) -> dict[str, Any]:
    return materialize_bs_preforcast_stage(
        loaded=loaded,
        selected_jobs=selected_jobs,
        run_root=run_root,
        main_resolved_path=main_resolved_path,
        main_capability_path=main_capability_path,
        main_manifest_path=main_manifest_path,
        entrypoint_version=entrypoint_version,
        validate_only=validate_only,
    )
