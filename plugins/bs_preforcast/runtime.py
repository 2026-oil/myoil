from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from neuralforecast import NeuralForecast

from residual.adapters import build_multivariate_inputs, build_univariate_inputs
from residual.config import JobConfig, LoadedConfig
from residual.manifest import build_manifest, write_manifest
from residual.forecast_models import (
    BASELINE_MODEL_NAMES,
    build_model,
    resolved_devices,
    resolved_strategy_name,
    validate_job,
)
from neuralforecast.models.bs_preforcast_catalog import is_direct_stage_model
from neuralforecast.models.bs_preforcast_direct import (
    normalized_direct_job_params,
    normalized_direct_stage_job,
    predict_univariate_arima,
    predict_univariate_direct,
    predict_univariate_es,
    predict_univariate_tree,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    if not loaded.config.stage_plugin_config.enabled:
        return "disabled"
    jobs = list(selected_jobs)
    if not jobs:
        raise ValueError("bs_preforcast injection mode requires at least one selected main job")
    unique_modes = {_derived_injection_mode_for_job(job) for job in jobs}
    if len(unique_modes) == 1:
        return next(iter(unique_modes))
    return "mixed"


def _derived_injection_mode_for_job(job: Any) -> str:
    return "futr_exog" if validate_job(job).supports_futr_exog else "lag_derived"


def _derived_job_injection_results(selected_jobs: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "model": str(job.model),
            "injection_mode": _derived_injection_mode_for_job(job),
            "supports_futr_exog": bool(validate_job(job).supports_futr_exog),
        }
        for job in selected_jobs
    ]


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
    job_injection_results: list[dict[str, Any]],
    stage_run_roots: list[str] | None = None,
) -> Path:
    path = stage_root / "summary" / "dashboard.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# bs_preforcast dashboard",
        "",
        f"- target_columns: {', '.join(target_columns)}",
    ]
    if job_injection_results:
        lines.append("- job_injection_results:")
        for item in job_injection_results:
            lines.append(
                f"  - {item['model']}: {item['injection_mode']} (supports_futr_exog={item['supports_futr_exog']})"
            )
    if stage_run_roots:
        lines.append("- stage_run_roots:")
        lines.extend(f"  - {item}" for item in stage_run_roots)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_stage_forecast_artifact(
    stage_root: Path,
    *,
    loaded: LoadedConfig,
    forecasts_by_column: dict[str, list[float]] | None = None,
    future_df: pd.DataFrame | None = None,
    fold_run_root: Path | None = None,
) -> Path:
    frame_rows: list[dict[str, Any]] = []
    path = stage_root / "artifacts" / "bs_preforcast_forecasts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if forecasts_by_column is not None and future_df is not None:
        dt_col = loaded.config.dataset.dt_col
        future_dt = (
            pd.to_datetime(future_df[dt_col]).astype(str).tolist()
            if dt_col in future_df.columns
            else [""] * max((len(values) for values in forecasts_by_column.values()), default=0)
        )
        for column, values in forecasts_by_column.items():
            for index, forecast in enumerate(values, start=1):
                source_dt = future_dt[index - 1] if index - 1 < len(future_dt) else ""
                frame_rows.append(
                    {
                        "target_column": str(column),
                        "horizon_step": index,
                        "forecast": float(forecast),
                        "source_dt": source_dt,
                        "stage_run_root": "" if fold_run_root is None else str(fold_run_root),
                    }
                )
    if not frame_rows:
        if not path.exists():
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
    new_rows = pd.DataFrame(frame_rows)
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty:
            new_rows = pd.concat([existing, new_rows], ignore_index=True)
            new_rows = new_rows.drop_duplicates(
                subset=["target_column", "horizon_step", "source_dt", "stage_run_root"],
                keep="last",
            )
    new_rows.to_csv(path, index=False)
    return path


def _rerouted_exog_columns(
    *,
    hist_columns: list[str],
    futr_columns: list[str],
    target_columns: list[str],
    injection_mode: str,
) -> tuple[list[str], list[str]]:
    target_set = set(target_columns)
    next_hist = [column for column in hist_columns if column not in target_set]
    next_futr = [column for column in futr_columns if column not in target_set]
    destination = next_futr if injection_mode == "futr_exog" else next_hist
    for column in target_columns:
        if column not in destination:
            destination.append(column)
    return next_hist, next_futr


def _require_forecast_values(
    forecasts_by_column: dict[str, list[float]],
    *,
    column: str,
    horizon: int,
) -> list[float]:
    if column not in forecasts_by_column:
        raise ValueError(
            f"bs_preforcast did not produce forecasts for target column: {column}"
        )
    forecast_values = forecasts_by_column[column]
    if len(forecast_values) != horizon:
        raise ValueError(
            "bs_preforcast produced an unexpected forecast horizon for "
            f"{column}: expected {horizon}, got {len(forecast_values)}"
        )
    if any(pd.isna(value) for value in forecast_values):
        raise ValueError(
            f"bs_preforcast produced missing forecast value(s) for target column: {column}"
        )
    return [float(value) for value in forecast_values]


def _stage_prediction_column(predictions: pd.DataFrame, model_name: str) -> str:
    if model_name in predictions.columns:
        return model_name
    raise KeyError(f"Could not find bs_preforcast prediction column for {model_name}")


def _stage_execution_loaded(loaded: LoadedConfig) -> LoadedConfig:
    if loaded.stage_plugin_loaded is None:
        raise ValueError("bs_preforcast stage1 config is required when enabled")
    return LoadedConfig(
        config=loaded.stage_plugin_loaded.config,
        source_path=loaded.stage_plugin_loaded.source_path,
        source_type=loaded.stage_plugin_loaded.source_type,
        normalized_payload=loaded.stage_plugin_loaded.normalized_payload,
        input_hash=loaded.stage_plugin_loaded.input_hash,
        resolved_hash=loaded.stage_plugin_loaded.resolved_hash,
        search_space_path=loaded.stage_plugin_loaded.search_space_path,
        search_space_hash=loaded.stage_plugin_loaded.search_space_hash,
        search_space_payload=loaded.stage_plugin_loaded.search_space_payload,
    )


def _single_stage_job(loaded: LoadedConfig) -> JobConfig:
    jobs = _stage_jobs(loaded)
    if len(jobs) != 1:
        raise ValueError("bs_preforcast stage currently requires exactly one resolved job")
    return jobs[0]


def _stage_jobs(loaded: LoadedConfig) -> list[JobConfig]:
    jobs = list(loaded.config.jobs)
    if not jobs:
        raise ValueError("bs_preforcast stage requires at least one configured job")
    for job in jobs:
        if job.model in BASELINE_MODEL_NAMES:
            raise ValueError("bs_preforcast stage does not support baseline-only jobs")
    return jobs


def _normalized_direct_job_params(
    model_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return normalized_direct_job_params(model_name, params)


def _normalized_direct_stage_job(job: JobConfig) -> JobConfig:
    return normalized_direct_stage_job(job)


def _resolved_stage_job(stage_loaded: LoadedConfig) -> JobConfig:
    jobs = _stage_jobs(stage_loaded)
    if len(jobs) != 1:
        raise ValueError("bs_preforcast plugin-only stage requires exactly one configured job")
    job = jobs[0]
    if job.validated_mode != "learned_fixed":
        raise ValueError(
            "bs_preforcast plugin-only stage requires a fixed learned job; auto selection is not supported at runtime"
        )
    if stage_loaded.config.training_search.validated_mode != "training_fixed":
        raise ValueError(
            "bs_preforcast plugin-only stage requires fixed training settings; training auto mode is not supported at runtime"
        )
    return _normalized_direct_stage_job(job)


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


def _resolved_stage_loaded(stage_loaded: LoadedConfig) -> LoadedConfig:
    resolved_job = _resolved_stage_job(stage_loaded)
    return _stage_loaded_with_job_and_training(stage_loaded, job=resolved_job)


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
    stage_loaded = _resolved_stage_loaded(stage_loaded)
    job = _single_stage_job(stage_loaded)
    if job.model == "AutoARIMA":
        raise ValueError(
            "bs_preforcast no longer supports AutoARIMA; use ARIMA instead"
        )
    if is_direct_stage_model(job.model):
        return predict_univariate_direct(
            stage_loaded,
            job,
            target_column=target_column,
            train_df=train_df,
            future_df=future_df,
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


def _predict_stage_univariate_arima(
    stage_loaded: LoadedConfig,
    job: JobConfig,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> list[float]:
    return predict_univariate_arima(
        stage_loaded,
        job,
        target_column=target_column,
        train_df=train_df,
        future_df=future_df,
    )


def _predict_stage_univariate_es(
    stage_loaded: LoadedConfig,
    job: JobConfig,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> list[float]:
    return predict_univariate_es(
        stage_loaded,
        job,
        target_column=target_column,
        train_df=train_df,
        future_df=future_df,
    )


def _predict_stage_univariate_tree(
    stage_loaded: LoadedConfig,
    job: JobConfig,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    model_name: str,
) -> list[float]:
    return predict_univariate_tree(
        stage_loaded,
        job,
        target_column=target_column,
        train_df=train_df,
        future_df=future_df,
        model_name=model_name,
    )


def _predict_stage_multivariate(
    loaded: LoadedConfig,
    stage_loaded: LoadedConfig,
    *,
    target_columns: list[str],
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    run_root: Path | None = None,
) -> dict[str, list[float]]:
    stage_loaded = _resolved_stage_loaded(stage_loaded)
    job = _single_stage_job(stage_loaded)
    if is_direct_stage_model(job.model):
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
    target_columns = list(loaded.config.stage_plugin_config.target_columns)
    if loaded.config.stage_plugin_config.task.multivariable:
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
    if not loaded.config.stage_plugin_config.enabled:
        return loaded, train_df, future_df, injection_mode
    target_columns = list(loaded.config.stage_plugin_config.target_columns)
    horizon = len(future_df)
    train_frame = train_df.copy()
    future_frame = future_df.copy()
    if injection_mode == "lag_derived" and len(train_frame) < horizon:
        raise ValueError(
            "bs_preforcast lag_derived injection requires at least as many training rows as horizon"
        )
    forecasts_by_column = compute_bs_preforcast_fold_forecasts(
        loaded,
        train_df=train_df,
        future_df=future_df,
        run_root=run_root,
    )
    if run_root is not None:
        _write_stage_forecast_artifact(
            _stage_root(run_root),
            loaded=loaded,
            forecasts_by_column=forecasts_by_column,
            future_df=future_df,
            fold_run_root=run_root,
        )
    hist_columns, futr_columns = _rerouted_exog_columns(
        hist_columns=list(loaded.config.dataset.hist_exog_cols),
        futr_columns=list(loaded.config.dataset.futr_exog_cols),
        target_columns=target_columns,
        injection_mode=injection_mode,
    )

    for column in target_columns:
        forecast_values = _require_forecast_values(
            forecasts_by_column,
            column=column,
            horizon=horizon,
        )
        train_frame[column] = train_frame[column].astype(float)
        future_frame[column] = future_frame[column].astype(float)
        if injection_mode == "futr_exog":
            future_frame[column] = forecast_values
            continue
        overwrite_count = min(len(train_frame), len(forecast_values))
        if overwrite_count:
            train_frame.loc[
                len(train_frame) - overwrite_count :,
                column,
            ] = forecast_values[:overwrite_count]
        future_frame[column] = forecast_values

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
    if not loaded.config.stage_plugin_config.enabled:
        return {}
    stage_loaded = load_bs_preforcast_stage_config(_repo_root(), loaded)
    if stage_loaded is None:
        return {}
    job_injection_results = _derived_job_injection_results(selected_jobs)

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
    metadata = attach_bs_preforcast_stage_metadata(
        loaded=loaded,
        selected_jobs=selected_jobs,
        run_root=run_root,
        main_resolved_path=main_resolved_path,
        main_capability_path=main_capability_path,
        main_manifest_path=main_manifest_path,
        validate_only=validate_only,
        stage_loaded=stage_loaded,
        job_injection_results=job_injection_results,
    )
    metadata.update(
        {
            "stage1_resolved_config_path": str(resolved_path),
            "stage1_capability_report_path": str(capability_path),
            "stage1_manifest_path": str(manifest_path),
            "stage1_run_roots": [],
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
    if not loaded.config.stage_plugin_config.enabled:
        return None
    if loaded.stage_plugin_loaded is None:
        return None
    return LoadedConfig(
        config=replace(
            loaded.stage_plugin_loaded.config,
            stage_plugin_config=loaded.config.stage_plugin_config,
        ),
        source_path=loaded.stage_plugin_loaded.source_path,
        source_type=loaded.stage_plugin_loaded.source_type,
        normalized_payload=loaded.stage_plugin_loaded.normalized_payload,
        input_hash=loaded.stage_plugin_loaded.input_hash,
        resolved_hash=loaded.stage_plugin_loaded.resolved_hash,
        search_space_path=loaded.stage_plugin_loaded.search_space_path,
        search_space_hash=loaded.stage_plugin_loaded.search_space_hash,
        search_space_payload=loaded.stage_plugin_loaded.search_space_payload,
    )


def write_bs_preforcast_dashboard(
    stage_root: Path,
    *,
    target_columns: list[str],
    job_injection_results: list[dict[str, Any]],
    stage_run_roots: list[str] | None = None,
) -> Path:
    return _write_stage_dashboard(
        stage_root,
        target_columns=target_columns,
        job_injection_results=job_injection_results,
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
    job_injection_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not loaded.config.stage_plugin_config.enabled:
        return {}
    stage_loaded = stage_loaded or load_bs_preforcast_stage_config(_repo_root(), loaded)
    if stage_loaded is None:
        return {}
    job_injection_results = job_injection_results or _derived_job_injection_results(
        selected_jobs
    )
    stage_root = _stage_root(run_root)
    dashboard_path = write_bs_preforcast_dashboard(
        stage_root,
        target_columns=list(loaded.config.stage_plugin_config.target_columns),
        job_injection_results=job_injection_results,
        stage_run_roots=[],
    )
    forecast_path = _write_stage_forecast_artifact(
        stage_root,
        loaded=loaded,
    )
    metadata = {
        "selected_config_path": str(stage_loaded.source_path),
        "job_injection_results": job_injection_results,
        "stage1_dashboard_path": str(dashboard_path),
        "stage1_forecast_artifact_path": str(forecast_path),
        "stage1_selected_jobs_path": None,
        "stage1_run_roots": [],
        "target_columns_used_for_injection": list(
            loaded.config.stage_plugin_config.target_columns
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
