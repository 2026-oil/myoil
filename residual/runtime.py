from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd
from neuralforecast import NeuralForecast

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


def _build_residual_training_panel(cv_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(cv_rows)
    if frame.empty:
        raise ValueError("No CV rows available for residual training")
    for column in ("cutoff", "train_end_ds", "ds"):
        frame[column] = pd.to_datetime(frame[column])
    frame["y_hat_base"] = frame.pop("y_hat").astype(float)
    frame["residual_target"] = frame["y"].astype(float) - frame["y_hat_base"]
    return frame[
        [
            "model",
            "unique_id",
            "fold_idx",
            "cutoff",
            "train_end_ds",
            "ds",
            "horizon_step",
            "y",
            "y_hat_base",
            "residual_target",
        ]
    ].sort_values(["fold_idx", "ds"]).reset_index(drop=True)


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


def _residual_prediction_input(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["fold_idx", "ds"]).reset_index(drop=True)
    columns = [
        "model",
        "unique_id",
        "fold_idx",
        "cutoff",
        "train_end_ds",
        "ds",
        "horizon_step",
        "y_hat_base",
    ]
    if "residual_target" in ordered.columns:
        columns.append("residual_target")
    if "y" in ordered.columns:
        columns.append("y")
    return ordered[columns].copy()


def _fold_checkpoint_dir(residual_root: Path, fold_idx: int) -> Path:
    return residual_root / "checkpoints" / f"fold_{fold_idx:03d}"


def _build_holdout_residual_panel(
    job: JobConfig,
    holdout_df: pd.DataFrame,
    target_holdout: pd.DataFrame,
    pred_col: str,
    dt_col: str,
    target_col: str,
    train_end_ds: object,
) -> pd.DataFrame:
    panel = target_holdout[["unique_id", "ds", pred_col]].rename(
        columns={pred_col: "y_hat_base"}
    )
    panel["model"] = job.model
    panel["fold_idx"] = -1
    panel["cutoff"] = pd.to_datetime(holdout_df[dt_col].iloc[0])
    panel["train_end_ds"] = pd.to_datetime(train_end_ds)
    panel["horizon_step"] = range(1, len(panel) + 1)
    panel["y"] = holdout_df[target_col].astype(float).values
    return panel[
        [
            "model",
            "unique_id",
            "fold_idx",
            "cutoff",
            "train_end_ds",
            "ds",
            "horizon_step",
            "y",
            "y_hat_base",
        ]
    ].copy()


def _apply_residual_plugin(
    loaded: LoadedConfig,
    job: JobConfig,
    run_root: Path,
    cv_rows,
    holdout_df: pd.DataFrame,
    target_holdout: pd.DataFrame,
    pred_col: str,
    train_end_ds: object,
) -> None:
    if job.model in BASELINE_MODEL_NAMES:
        return
    if not loaded.config.residual.enabled:
        return
    residual_root = run_root / "residual" / job.model
    residual_root.mkdir(parents=True, exist_ok=True)
    train_panel = _build_residual_training_panel(cv_rows)
    train_panel.to_csv(residual_root / "training_panel.csv", index=False)

    corrected_groups: list[pd.DataFrame] = []
    checkpoint_metadata: dict[str, dict[str, object]] = {}
    for fold_idx, fold_panel in train_panel.groupby("fold_idx", sort=True):
        plugin = build_residual_plugin(loaded.config.residual)
        fit_panel = train_panel[train_panel["fold_idx"] < fold_idx].reset_index(
            drop=True
        )
        checkpoint_dir = _fold_checkpoint_dir(residual_root, int(fold_idx))
        plugin.fit(
            fit_panel,
            _build_residual_context(loaded, job, checkpoint_dir, model_name=job.model),
        )
        predicted = plugin.predict(_residual_prediction_input(fold_panel))
        corrected = fold_panel.reset_index(drop=True).copy()
        corrected["residual_hat"] = predicted["residual_hat"].astype(float).values
        corrected["y_hat_corrected"] = (
            corrected["y_hat_base"] + corrected["residual_hat"]
        )
        corrected_groups.append(corrected)
        checkpoint_metadata[str(fold_idx)] = plugin.metadata()

    corrected_cv = pd.concat(corrected_groups, ignore_index=True)
    corrected_cv.to_csv(residual_root / "corrected_cv.csv", index=False)

    holdout_panel = _build_holdout_residual_panel(
        job,
        holdout_df,
        target_holdout,
        pred_col,
        loaded.config.dataset.dt_col,
        loaded.config.dataset.target_col,
        train_end_ds=train_end_ds,
    )
    holdout_plugin = build_residual_plugin(loaded.config.residual)
    holdout_checkpoint_dir = _fold_checkpoint_dir(residual_root, -1)
    holdout_plugin.fit(
        train_panel,
        _build_residual_context(
            loaded, job, holdout_checkpoint_dir, model_name=job.model
        ),
    )
    holdout_pred = holdout_plugin.predict(_residual_prediction_input(holdout_panel))
    holdout_corrected = holdout_panel.copy()
    holdout_corrected["residual_hat"] = (
        holdout_pred["residual_hat"].astype(float).values
    )
    holdout_corrected["y_hat_corrected"] = (
        holdout_corrected["y_hat_base"] + holdout_corrected["residual_hat"]
    )
    holdout_corrected.to_csv(residual_root / "corrected_holdout.csv", index=False)
    checkpoint_metadata["-1"] = holdout_plugin.metadata()
    (residual_root / "plugin_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2), encoding="utf-8"
    )
    diagnostics = {
        "model": job.model,
        "residual_model": loaded.config.residual.model,
        "training_panel_rows": int(len(train_panel)),
        "corrected_cv_rows": int(len(corrected_cv)),
        "corrected_holdout_rows": int(len(holdout_corrected)),
        "corrected_cv_mode": "per_fold_panel_runtime",
        "holdout_truth_included": False,
        "checkpoint_layout": "residual/<model>/checkpoints/fold_{fold_idx:03d}/model.ubj",
    }
    (residual_root / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )


def _baseline_cross_validation(
    train_df: pd.DataFrame,
    horizon: int,
    step_size: int,
    n_windows: int,
    model_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics_rows = []
    forecast_rows = []
    values = train_df["y"].reset_index(drop=True)
    dates = pd.to_datetime(train_df["ds"]).reset_index(drop=True)
    for fold_idx in range(n_windows):
        train_end = _cutoff_train_end(
            len(train_df), horizon, step_size, n_windows, fold_idx
        )
        history = values.iloc[:train_end]
        future_actual = values.iloc[train_end : train_end + horizon].reset_index(
            drop=True
        )
        future_dates = dates.iloc[train_end : train_end + horizon].reset_index(
            drop=True
        )
        if model_name == "Naive":
            pred_values = pd.Series([float(history.iloc[-1])] * len(future_actual))
        elif model_name == "SeasonalNaive":
            season = min(len(history), horizon)
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
            {"fold_idx": fold_idx, "cutoff": str(future_dates.iloc[0]), **metrics}
        )
        for idx, ds in enumerate(future_dates):
            forecast_rows.append(
                {
                    "model": model_name,
                    "fold_idx": fold_idx,
                    "cutoff": str(future_dates.iloc[0]),
                    "unique_id": train_df["unique_id"].iloc[0],
                    "ds": str(ds),
                    "y": float(future_actual.iloc[idx]),
                    "y_hat": float(pred_values.iloc[idx]),
                }
            )
    return metrics_rows, forecast_rows


def _baseline_holdout(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame, model_name: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    history = train_df["y"].reset_index(drop=True)
    actual = holdout_df["y"].reset_index(drop=True)
    if model_name == "Naive":
        pred_values = pd.Series([float(history.iloc[-1])] * len(actual))
    elif model_name == "SeasonalNaive":
        season = min(len(history), len(actual))
        tail = history.iloc[-season:].reset_index(drop=True)
        pred_values = pd.Series(
            (list(tail) * ((len(actual) // len(tail)) + 1))[: len(actual)]
        )
    else:
        pred_values = pd.Series([float(history.mean())] * len(actual))
    predictions = holdout_df[["unique_id", "ds"]].copy()
    predictions[model_name] = pred_values.values
    return predictions, _compute_metrics(actual, pred_values)


def _run_single_job(loaded: LoadedConfig, job: JobConfig, run_root: Path) -> None:
    source_df = pd.read_csv(loaded.config.dataset.path)
    source_df = source_df.sort_values(loaded.config.dataset.dt_col).reset_index(
        drop=True
    )
    freq = _resolve_freq(loaded, source_df)
    total_rows = len(source_df)
    holdout = loaded.config.cv.final_holdout
    horizon = loaded.config.cv.horizon
    step_size = loaded.config.cv.step_size
    n_windows = loaded.config.cv.n_windows
    if total_rows <= holdout + horizon:
        raise ValueError("Dataset is too short for configured holdout + horizon")
    pre_holdout = source_df.iloc[:-holdout].reset_index(drop=True)
    holdout_df_source = source_df.iloc[-holdout:].reset_index(drop=True)
    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col

    cv_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []

    if job.model in BASELINE_MODEL_NAMES:
        train_series = pre_holdout[[dt_col, target_col]].copy()
        train_series.rename(columns={dt_col: "ds", target_col: "y"}, inplace=True)
        train_series["ds"] = pd.to_datetime(train_series["ds"])
        train_series.insert(0, "unique_id", target_col)
        baseline_metrics, baseline_forecasts = _baseline_cross_validation(
            train_series, horizon, step_size, n_windows, job.model
        )
        metrics_rows.extend(baseline_metrics)
        cv_rows.extend(baseline_forecasts)
        holdout_series = holdout_df_source[[dt_col, target_col]].copy()
        holdout_series.rename(columns={dt_col: "ds", target_col: "y"}, inplace=True)
        holdout_series["ds"] = pd.to_datetime(holdout_series["ds"])
        holdout_series.insert(0, "unique_id", target_col)
        holdout_predictions, holdout_metrics = _baseline_holdout(
            train_series, holdout_series, job.model
        )
        pred_col = job.model
        target_holdout = holdout_predictions.rename(columns={pred_col: job.model})
        nf = None
    else:
        for fold_idx in range(n_windows):
            train_end = _cutoff_train_end(
                len(pre_holdout), horizon, step_size, n_windows, fold_idx
            )
            if train_end <= 0:
                raise ValueError("Configured CV window exceeds dataset length")
            train_df = pre_holdout.iloc[:train_end].reset_index(drop=True)
            future_df = pre_holdout.iloc[train_end : train_end + horizon].reset_index(
                drop=True
            )
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
            predictions = (
                nf.predict(
                    futr_df=adapter_inputs.futr_df, static_df=adapter_inputs.static_df
                )
                if adapter_inputs.futr_df is not None
                else nf.predict(static_df=adapter_inputs.static_df)
            )
            pred_col = _prediction_column(predictions, job.model)
            target_predictions = predictions[
                predictions["unique_id"] == target_col
            ].reset_index(drop=True)
            target_actuals = future_df[target_col].reset_index(drop=True)
            metrics = _compute_metrics(target_actuals, target_predictions[pred_col])
            metrics_rows.append(
                {
                    "fold_idx": fold_idx,
                    "cutoff": str(future_df[dt_col].iloc[0]),
                    "train_end_ds": str(train_df[dt_col].iloc[-1]),
                    **metrics,
                }
            )
            for row_idx, ds in enumerate(target_predictions["ds"]):
                cv_rows.append(
                    {
                        "model": job.model,
                        "fold_idx": fold_idx,
                        "cutoff": str(future_df[dt_col].iloc[0]),
                        "train_end_ds": str(train_df[dt_col].iloc[-1]),
                        "unique_id": target_col,
                        "ds": str(ds),
                        "horizon_step": row_idx + 1,
                        "y": float(target_actuals.iloc[row_idx]),
                        "y_hat": float(target_predictions[pred_col].iloc[row_idx]),
                    }
                )
        full_inputs = _build_adapter_inputs(
            loaded, pre_holdout, holdout_df_source, job, dt_col
        )
        model = build_model(
            loaded.config, job, n_series=full_inputs.metadata.get("n_series")
        )
        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(
            full_inputs.fit_df,
            static_df=full_inputs.static_df,
            val_size=loaded.config.training.val_size,
        )
        holdout_predictions = (
            nf.predict(futr_df=full_inputs.futr_df, static_df=full_inputs.static_df)
            if full_inputs.futr_df is not None
            else nf.predict(static_df=full_inputs.static_df)
        )
        pred_col = _prediction_column(holdout_predictions, job.model)
        target_holdout = holdout_predictions[
            holdout_predictions["unique_id"] == target_col
        ].reset_index(drop=True)
        holdout_metrics = _compute_metrics(
            holdout_df_source[target_col], target_holdout[pred_col]
        )

    cv_dir = run_root / "cv"
    holdout_dir = run_root / "holdout"
    models_dir = run_root / "models" / job.model
    cv_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(cv_dir / f"{job.model}_forecasts.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(
        cv_dir / f"{job.model}_metrics_by_cutoff.csv", index=False
    )
    pd.DataFrame([{"model": job.model, **holdout_metrics}]).to_csv(
        holdout_dir / f"{job.model}_metrics.csv", index=False
    )
    target_holdout.to_csv(holdout_dir / f"{job.model}_forecasts.csv", index=False)
    (models_dir / "fit_summary.json").write_text(
        json.dumps(
            {"model": job.model, "devices": 1, "loss": loaded.config.training.loss},
            indent=2,
        ),
        encoding="utf-8",
    )
    if job.model not in BASELINE_MODEL_NAMES:
        _apply_residual_plugin(
            loaded,
            job,
            run_root,
            cv_rows,
            holdout_df_source,
            target_holdout,
            pred_col,
            pre_holdout[dt_col].iloc[-1],
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
