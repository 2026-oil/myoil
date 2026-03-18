from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import json

import pandas as pd
from neuralforecast import NeuralForecast

from .adapters import build_multivariate_inputs, build_univariate_inputs
from .config import LoadedConfig, load_app_config
from .manifest import build_manifest, write_manifest
from .models import build_model, validate_job
from .scheduler import build_launch_plan, run_parallel_jobs

ENTRYPOINT_VERSION = 'neuralforecast-residual-v1'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Residual wrapper runtime for neuralforecast.')
    parser.add_argument('--config', default=None)
    parser.add_argument('--config-path', default=None)
    parser.add_argument('--config-toml', default=None)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--jobs', nargs='+', default=None)
    parser.add_argument('--output-root', default=None)
    return parser


def _selected_jobs(loaded: LoadedConfig, names: list[str] | None):
    if not names:
        return [job for job in loaded.config.jobs if job.enabled]
    allowed = set(names)
    return [job for job in loaded.config.jobs if job.name in allowed and job.enabled]


def _build_resolved_artifacts(repo_root: Path, loaded: LoadedConfig, output_root: Path) -> dict[str, Path]:
    run_root = output_root.resolve()
    resolved_path = run_root / 'config' / 'config.resolved.json'
    capability_path = run_root / 'config' / 'capability_report.json'
    manifest_path = run_root / 'manifest' / 'run_manifest.json'
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    capability_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(loaded.normalized_payload, indent=2), encoding='utf-8')
    manifest = build_manifest(
        loaded,
        compat_mode='dual_read',
        entrypoint_version=ENTRYPOINT_VERSION,
        resolved_config_path=resolved_path,
    )
    write_manifest(manifest_path, manifest)
    return {
        'run_root': run_root,
        'resolved_path': resolved_path,
        'capability_path': capability_path,
        'manifest_path': manifest_path,
    }


def _validate_jobs(loaded: LoadedConfig, selected_jobs, capability_path: Path) -> None:
    payload = {}
    for job in selected_jobs:
        caps = validate_job(job)
        payload[job.name] = caps.__dict__
    capability_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _validate_adapters(loaded: LoadedConfig, selected_jobs) -> None:
    source_df = pd.read_csv(loaded.config.dataset.path)
    dt_col = loaded.config.dataset.dt_col
    for job in selected_jobs:
        if job.job_type == 'univariate_with_exog':
            build_univariate_inputs(source_df, job, dt_col=dt_col)
        else:
            build_multivariate_inputs(source_df, job, dt_col=dt_col)


def _cutoff_train_end(total_rows: int, horizon: int, step_size: int, n_windows: int, fold_idx: int) -> int:
    remaining = horizon + step_size * (n_windows - 1 - fold_idx)
    return total_rows - remaining


def _compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    err = actual.reset_index(drop=True) - predicted.reset_index(drop=True)
    mae = float(err.abs().mean())
    mse = float((err ** 2).mean())
    rmse = mse ** 0.5
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


def _prediction_column(predictions: pd.DataFrame, job_name: str, fallback: str) -> str:
    for candidate in (job_name, fallback):
        if candidate in predictions.columns:
            return candidate
    raise KeyError(f'Could not find prediction column for {job_name}')


def _build_adapter_inputs(train_df: pd.DataFrame, future_df: pd.DataFrame | None, job, dt_col: str):
    if job.job_type == 'univariate_with_exog':
        return build_univariate_inputs(train_df, job, dt_col=dt_col, future_df=future_df)
    return build_multivariate_inputs(train_df, job, dt_col=dt_col, future_df=future_df)


def _run_single_job(loaded: LoadedConfig, job, run_root: Path) -> None:
    source_df = pd.read_csv(loaded.config.dataset.path)
    source_df = source_df.sort_values(loaded.config.dataset.dt_col).reset_index(drop=True)
    total_rows = len(source_df)
    holdout = loaded.config.cv.final_holdout
    horizon = loaded.config.cv.horizon
    step_size = loaded.config.cv.step_size
    n_windows = loaded.config.cv.n_windows
    if total_rows <= holdout + horizon:
        raise ValueError('Dataset is too short for configured holdout + horizon')
    pre_holdout = source_df.iloc[:-holdout].reset_index(drop=True)
    holdout_df = source_df.iloc[-holdout:].reset_index(drop=True)
    dt_col = loaded.config.dataset.dt_col

    cv_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    for fold_idx in range(n_windows):
        train_end = _cutoff_train_end(len(pre_holdout), horizon, step_size, n_windows, fold_idx)
        if train_end <= 0:
            raise ValueError('Configured CV window exceeds dataset length')
        train_df = pre_holdout.iloc[:train_end].reset_index(drop=True)
        future_df = pre_holdout.iloc[train_end: train_end + horizon].reset_index(drop=True)
        adapter_inputs = _build_adapter_inputs(train_df, future_df, job, dt_col)
        model = build_model(loaded.config, job, n_series=(adapter_inputs.metadata.get('n_series')))
        nf = NeuralForecast(models=[model], freq=loaded.config.dataset.freq)
        nf.fit(adapter_inputs.fit_df, static_df=adapter_inputs.static_df, val_size=loaded.config.training.val_size)
        if adapter_inputs.futr_df is not None:
            predictions = nf.predict(futr_df=adapter_inputs.futr_df, static_df=adapter_inputs.static_df)
        else:
            predictions = nf.predict(static_df=adapter_inputs.static_df)
        pred_col = _prediction_column(predictions, job.name, job.model)
        target_predictions = predictions[predictions['unique_id'] == job.target_col].reset_index(drop=True)
        target_actuals = future_df[job.target_col].reset_index(drop=True)
        metrics = _compute_metrics(target_actuals, target_predictions[pred_col])
        metrics_rows.append({'fold_idx': fold_idx, 'cutoff': str(future_df[dt_col].iloc[0]), **metrics})
        for row_idx, ds in enumerate(target_predictions['ds']):
            cv_rows.append({
                'job_name': job.name,
                'model': job.model,
                'fold_idx': fold_idx,
                'cutoff': str(future_df[dt_col].iloc[0]),
                'unique_id': job.target_col,
                'ds': str(ds),
                'y': float(target_actuals.iloc[row_idx]),
                'y_hat': float(target_predictions[pred_col].iloc[row_idx]),
            })

    full_inputs = _build_adapter_inputs(pre_holdout, holdout_df, job, dt_col)
    model = build_model(loaded.config, job, n_series=(full_inputs.metadata.get('n_series')))
    nf = NeuralForecast(models=[model], freq=loaded.config.dataset.freq)
    nf.fit(full_inputs.fit_df, static_df=full_inputs.static_df, val_size=loaded.config.training.val_size)
    if full_inputs.futr_df is not None:
        holdout_predictions = nf.predict(futr_df=full_inputs.futr_df, static_df=full_inputs.static_df)
    else:
        holdout_predictions = nf.predict(static_df=full_inputs.static_df)
    pred_col = _prediction_column(holdout_predictions, job.name, job.model)
    target_holdout = holdout_predictions[holdout_predictions['unique_id'] == job.target_col].reset_index(drop=True)
    holdout_metrics = _compute_metrics(holdout_df[job.target_col], target_holdout[pred_col])

    cv_dir = run_root / 'cv'
    holdout_dir = run_root / 'holdout'
    models_dir = run_root / 'models' / job.name
    cv_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(cv_dir / f'{job.name}_forecasts.csv', index=False)
    pd.DataFrame(metrics_rows).to_csv(cv_dir / f'{job.name}_metrics_by_cutoff.csv', index=False)
    pd.DataFrame([{
        'job_name': job.name,
        'model': job.model,
        **holdout_metrics,
    }]).to_csv(holdout_dir / f'{job.name}_metrics.csv', index=False)
    target_holdout.assign(job_name=job.name, model=job.model).to_csv(holdout_dir / f'{job.name}_forecasts.csv', index=False)
    (models_dir / 'fit_summary.json').write_text(json.dumps({
        'job_name': job.name,
        'model': job.model,
        'devices': 1,
        'loss': loaded.config.training.loss,
    }, indent=2), encoding='utf-8')


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or args.config_path
    loaded = load_app_config(
        repo_root,
        config_path=config_path,
        config_toml_path=args.config_toml,
    )
    output_root = Path(args.output_root) if args.output_root else repo_root / 'runs' / 'validation'
    paths = _build_resolved_artifacts(repo_root, loaded, output_root)
    selected_jobs = _selected_jobs(loaded, args.jobs)
    _validate_jobs(loaded, selected_jobs, paths['capability_path'])
    _validate_adapters(loaded, selected_jobs)
    if args.validate_only:
        print(json.dumps({'ok': True, 'jobs': [job.name for job in selected_jobs]}))
        return 0
    if len(selected_jobs) == 1:
        _run_single_job(loaded, selected_jobs[0], paths['run_root'])
        print(json.dumps({'ok': True, 'executed_jobs': [selected_jobs[0].name]}))
        return 0
    launches = build_launch_plan(loaded.config, selected_jobs)
    scheduler_dir = paths['run_root'] / 'scheduler'
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    (scheduler_dir / 'launch_plan.json').write_text(
        json.dumps([launch.__dict__ for launch in launches], indent=2),
        encoding='utf-8',
    )
    results = run_parallel_jobs(repo_root, loaded, launches, scheduler_dir)
    if any(int(result['returncode']) != 0 for result in results):
        raise SystemExit(json.dumps({'ok': False, 'worker_results': results}))
    print(json.dumps({'ok': True, 'scheduled_jobs': [launch.__dict__ for launch in launches], 'worker_results': results}))
    return 0
