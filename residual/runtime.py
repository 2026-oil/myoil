from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from neuralforecast import NeuralForecast
from sklearn.model_selection import TimeSeriesSplit

from .adapters import build_multivariate_inputs, build_univariate_inputs
from .config import JobConfig, LoadedConfig, load_app_config
from .manifest import build_manifest, write_manifest
from .models import BASELINE_MODEL_NAMES, build_model, validate_job
from .plugins_base import ResidualContext
from .registry import build_residual_plugin
from .scheduler import build_launch_plan, run_parallel_jobs

ENTRYPOINT_VERSION = "neuralforecast-residual-v1"


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
        payload[job.model] = caps.__dict__
    capability_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _should_use_multivariate(loaded: LoadedConfig, job: JobConfig) -> bool:
    caps = validate_job(job)
    has_dataset_exog = bool(
        loaded.config.dataset.hist_exog_cols
        or loaded.config.dataset.futr_exog_cols
        or loaded.config.dataset.static_exog_cols
    )
    return bool(caps.multivariate and has_dataset_exog)


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
    if cv_config.n_windows == 1:
        train_end = total_rows - cv_config.gap - cv_config.horizon
        if train_end <= 0:
            raise ValueError(
                "Dataset is too short for the configured single-split TSCV policy "
                f"(horizon={cv_config.horizon}, gap={cv_config.gap})"
            )
        train_start = 0
        if cv_config.max_train_size is not None:
            train_start = max(0, train_end - cv_config.max_train_size)
        test_start = train_end + cv_config.gap
        test_end = test_start + cv_config.horizon
        return [
            (list(range(train_start, train_end)), list(range(test_start, test_end)))
        ]

    splitter = TimeSeriesSplit(
        n_splits=cv_config.n_windows,
        test_size=cv_config.horizon,
        gap=cv_config.gap,
        max_train_size=cv_config.max_train_size,
    )
    try:
        return [
            (train_idx.tolist(), test_idx.tolist())
            for train_idx, test_idx in splitter.split(range(total_rows))
        ]
    except ValueError as exc:
        raise ValueError(
            "Dataset is too short for the configured TSCV policy "
            f"(n_windows={cv_config.n_windows}, horizon={cv_config.horizon}, "
            f"gap={cv_config.gap}, max_train_size={cv_config.max_train_size})"
        ) from exc


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

    for payload in fold_payloads:
        fold_idx = int(payload["fold_idx"])
        backcast_panel: pd.DataFrame = payload["backcast_panel"]
        eval_panel: pd.DataFrame = payload["eval_panel"]
        base_summary: dict[str, object] = payload["base_summary"]
        fold_root = _fold_artifact_dir(residual_root, fold_idx)
        base_checkpoint_dir = fold_root / "base_checkpoint"
        residual_checkpoint_dir = fold_root / "residual_checkpoint"
        base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        plugin = build_residual_plugin(loaded.config.residual)

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


def _run_single_job(loaded: LoadedConfig, job: JobConfig, run_root: Path) -> None:
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

    if job.model in BASELINE_MODEL_NAMES:
        train_series = source_df[[dt_col, target_col]].copy()
        train_series.rename(columns={dt_col: "ds", target_col: "y"}, inplace=True)
        train_series["ds"] = pd.to_datetime(train_series["ds"])
        train_series.insert(0, "unique_id", target_col)
        baseline_metrics, baseline_forecasts = _baseline_cross_validation(
            train_series,
            splits,
            job.model,
        )
        metrics_rows.extend(baseline_metrics)
        cv_rows.extend(baseline_forecasts)
    else:
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_df = source_df.iloc[train_idx].reset_index(drop=True)
            future_df = source_df.iloc[test_idx].reset_index(drop=True)
            adapter_inputs = _build_adapter_inputs(
                loaded, train_df, future_df, job, dt_col
            )
            model = build_model(
                loaded.config, job, n_series=adapter_inputs.metadata.get("n_series")
            )
            nf = NeuralForecast(models=[model], freq=freq)
            nf.fit(
                adapter_inputs.fit_df,
                static_df=adapter_inputs.static_df,
                val_size=loaded.config.training.val_size,
            )
            predictions = _predict_with_fitted_model(nf, adapter_inputs)
            pred_col = _prediction_column(predictions, job.model)
            target_predictions = predictions[
                predictions["unique_id"] == target_col
            ].reset_index(drop=True)
            target_actuals = future_df[target_col].reset_index(drop=True)
            train_end_ds = pd.to_datetime(train_df[dt_col].iloc[-1])
            metrics = _compute_metrics(target_actuals, target_predictions[pred_col])
            metrics_rows.append(
                {"fold_idx": fold_idx, "cutoff": str(train_end_ds), **metrics}
            )
            for row_idx, ds in enumerate(target_predictions["ds"]):
                cv_rows.append(
                    {
                        "model": job.model,
                        "fold_idx": fold_idx,
                        "cutoff": str(train_end_ds),
                        "train_end_ds": str(train_end_ds),
                        "unique_id": target_col,
                        "ds": str(ds),
                        "horizon_step": row_idx + 1,
                        "y": float(target_actuals.iloc[row_idx]),
                        "y_hat": float(target_predictions[pred_col].iloc[row_idx]),
                    }
                )
            if loaded.config.residual.enabled:
                backcast_panel = _build_fold_backcast_panel(
                    loaded,
                    job,
                    nf,
                    train_df,
                    dt_col,
                    target_col,
                    fold_idx,
                )
                eval_panel = _build_fold_eval_panel(
                    job,
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
                        "base_summary": {
                            "model": job.model,
                            "fold_idx": fold_idx,
                            "train_rows": int(len(train_df)),
                            "eval_rows": int(len(future_df)),
                            "train_end_ds": str(train_end_ds),
                            "loss": loaded.config.training.loss,
                        },
                    }
                )

    cv_dir = run_root / "cv"
    models_dir = run_root / "models" / job.model
    cv_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(cv_dir / f"{job.model}_forecasts.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(
        cv_dir / f"{job.model}_metrics_by_cutoff.csv", index=False
    )
    (models_dir / "fit_summary.json").write_text(
        json.dumps(
            {
                "model": job.model,
                "devices": 1,
                "loss": loaded.config.training.loss,
                "evaluation_policy": "tscv_only",
                "fold_count": len(splits),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if job.model not in BASELINE_MODEL_NAMES:
        _apply_residual_plugin(loaded, job, run_root, fold_payloads)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or args.config_path
    loaded = load_app_config(
        repo_root, config_path=config_path, config_toml_path=args.config_toml
    )
    output_root = (
        Path(args.output_root)
        if args.output_root
        else repo_root / "runs" / "validation"
    )
    paths = _build_resolved_artifacts(repo_root, loaded, output_root)
    selected_jobs = _selected_jobs(loaded, args.jobs)
    _validate_jobs(loaded, selected_jobs, paths["capability_path"])
    _validate_adapters(loaded, selected_jobs)
    if args.validate_only:
        print(json.dumps({"ok": True, "jobs": [job.model for job in selected_jobs]}))
        return 0
    if len(selected_jobs) == 1:
        _run_single_job(loaded, selected_jobs[0], paths["run_root"])
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
