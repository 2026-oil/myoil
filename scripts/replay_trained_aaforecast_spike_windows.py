from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from neuralforecast import NeuralForecast

from app_config import load_app_config
from plugins.aa_forecast.runtime import _aa_params_override, _extract_target_prediction_frame, _predict_with_adapter
from runtime_support.forecast_models import build_model
from runtime_support.runner import (
    _build_adapter_inputs,
    _build_fold_diff_context,
    _effective_config,
    _resolve_freq,
    _restore_target_predictions,
    _transform_training_frame,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Replay trained AAForecast model on historical spike windows without retraining.')
    p.add_argument('--config', default='yaml/experiment/feature_set_aaforecast/.tmp-aaforecast-informer-no-retrieval-commonpath6.yaml')
    p.add_argument('--run-root', default='runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_commonpath6')
    p.add_argument('--train-cutoff', default='2026-02-23')
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--min-future-ret', type=float, default=0.12)
    return p.parse_args()


def _select_spike_cutoffs(df: pd.DataFrame, *, target_col: str, input_size: int, horizon: int, top_k: int, min_future_ret: float, latest_cutoff: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for end_idx in range(input_size - 1, len(df) - horizon):
        cutoff = pd.Timestamp(df.iloc[end_idx]['dt'])
        if cutoff >= latest_cutoff:
            continue
        anchor = float(df.iloc[end_idx][target_col])
        future = df.iloc[end_idx + 1:end_idx + 1 + horizon][target_col].to_numpy(dtype=float)
        ret = float((future[-1] - anchor) / max(abs(anchor), 1e-8))
        rows.append({
            'cutoff': cutoff,
            'anchor': anchor,
            'future_h1': float(future[0]),
            'future_h2': float(future[1]),
            'future_h2cum_ret': ret,
        })
    frame = pd.DataFrame(rows)
    selected = frame[frame['future_h2cum_ret'] >= min_future_ret].sort_values('future_h2cum_ret', ascending=False)
    return selected.head(top_k).reset_index(drop=True)


def _series_payload_summary(payload: dict[str, torch.Tensor]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ['event_summary', 'event_trajectory', 'non_star_regime']:
        value = payload.get(key)
        if value is not None:
            summary[key] = value.detach().cpu().numpy().reshape(-1).tolist()
    for key in ['regime_intensity', 'regime_density', 'count_active_channels']:
        value = payload.get(key)
        if value is not None:
            arr = value.detach().cpu().numpy().reshape(-1)
            summary[f'{key}_tail_mean'] = float(arr[-8:].mean()) if arr.size else 0.0
            summary[f'{key}_tail_max'] = float(arr[-8:].max()) if arr.size else 0.0
    return summary


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    run_root = ROOT / args.run_root
    output_dir = run_root / 'aa_forecast' / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_app_config(ROOT, config_path=config_path)
    job = loaded.config.jobs[0]
    target_col = loaded.config.dataset.target_col
    dt_col = loaded.config.dataset.dt_col
    source_df = pd.read_csv(loaded.config.dataset.path)
    source_df[dt_col] = pd.to_datetime(source_df[dt_col])
    source_df = source_df.sort_values(dt_col).reset_index(drop=True)
    latest_cutoff = pd.Timestamp(args.train_cutoff)
    full_train_df = source_df[source_df[dt_col] <= latest_cutoff].reset_index(drop=True)
    if full_train_df.empty:
        raise ValueError('No training rows found up to train cutoff')

    latest_future_df = source_df[source_df[dt_col] > latest_cutoff].head(loaded.config.cv.horizon).reset_index(drop=True)
    effective_config = _effective_config(loaded, None)
    full_diff_context = _build_fold_diff_context(loaded, full_train_df)
    transformed_full_train_df = _transform_training_frame(full_train_df, full_diff_context)
    freq = _resolve_freq(loaded, pd.concat([full_train_df, latest_future_df], ignore_index=True))
    adapter_inputs = _build_adapter_inputs(loaded, transformed_full_train_df, latest_future_df, job, dt_col)
    params_override = _aa_params_override(effective_config)
    model = build_model(
        effective_config,
        job,
        n_series=adapter_inputs.metadata.get('n_series'),
        params_override=params_override,
    )
    if hasattr(model, 'configure_stochastic_inference'):
        model.configure_stochastic_inference(enabled=False, dropout_p=0.0)
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(adapter_inputs.fit_df, static_df=adapter_inputs.static_df, val_size=effective_config.training.val_size)

    spike_cutoffs = _select_spike_cutoffs(
        source_df,
        target_col=target_col,
        input_size=loaded.config.training.input_size,
        horizon=loaded.config.cv.horizon,
        top_k=args.top_k,
        min_future_ret=args.min_future_ret,
        latest_cutoff=latest_cutoff,
    )
    replay_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    for _, spike in spike_cutoffs.iterrows():
        cutoff = pd.Timestamp(spike['cutoff'])
        historical_train = source_df[source_df[dt_col] <= cutoff].reset_index(drop=True)
        future_df = source_df[source_df[dt_col] > cutoff].head(loaded.config.cv.horizon).reset_index(drop=True)
        diff_context = _build_fold_diff_context(loaded, historical_train)
        transformed_hist_train = _transform_training_frame(historical_train, diff_context)
        replay_inputs = _build_adapter_inputs(loaded, transformed_hist_train, future_df, job, dt_col)
        predictions = _predict_with_adapter(nf, replay_inputs)
        target_predictions = _extract_target_prediction_frame(
            predictions,
            target_col=target_col,
            model_name=job.model,
            diff_context=diff_context,
            restore_target_predictions=_restore_target_predictions,
        )
        y_hat = target_predictions[job.model].to_numpy(dtype=float)
        y = future_df[target_col].to_numpy(dtype=float)
        replay_rows.append({
            'cutoff': str(cutoff),
            'anchor': float(historical_train[target_col].iloc[-1]),
            'actual_h1': float(y[0]),
            'actual_h2': float(y[1]),
            'pred_h1': float(y_hat[0]),
            'pred_h2': float(y_hat[1]),
            'ape_h1': float(abs(y_hat[0] - y[0]) / max(abs(y[0]), 1e-8)),
            'ape_h2': float(abs(y_hat[1] - y[1]) / max(abs(y[1]), 1e-8)),
            'h2_gt_h1': bool(y_hat[1] > y_hat[0]),
            'future_h2cum_ret': float(spike['future_h2cum_ret']),
        })

        if hasattr(model, '_compute_star_outputs'):
            tail_window = transformed_hist_train.tail(model.input_size).reset_index(drop=True)
            insample_y = torch.as_tensor(tail_window[target_col].to_numpy(dtype=np.float32), dtype=torch.float32).reshape(1, -1, 1)
            hist_exog = None
            if getattr(model, 'hist_exog_list', ()):
                hist_exog = torch.as_tensor(
                    tail_window[list(model.hist_exog_list)].to_numpy(dtype=np.float32),
                    dtype=torch.float32,
                ).reshape(1, len(tail_window), -1)
            with torch.no_grad():
                payload = model._compute_star_outputs(insample_y, hist_exog)
            summary = _series_payload_summary(payload)
            summary['cutoff'] = str(cutoff)
            payload_rows.append(summary)

    replay_df = pd.DataFrame(replay_rows)
    payload_df = pd.DataFrame(payload_rows)
    replay_df.to_csv(output_dir / 'trained_model_spike_window_replay.csv', index=False)
    payload_df.to_csv(output_dir / 'trained_model_spike_window_payloads.csv', index=False)

    report_lines = [
        '# Trained model historical spike replay',
        '',
        f'- train_cutoff: {latest_cutoff.date()}',
        f'- replay_count: {len(replay_df)}',
        '',
        '## Replay summary',
    ]
    for _, row in replay_df.iterrows():
        report_lines.append(
            f"- cutoff={row['cutoff']} | actual=({row['actual_h1']:.4f}, {row['actual_h2']:.4f}) | pred=({row['pred_h1']:.4f}, {row['pred_h2']:.4f}) | ape=({row['ape_h1']:.4f}, {row['ape_h2']:.4f}) | h2>h1={row['h2_gt_h1']} | future_ret={row['future_h2cum_ret']:.4f}"
        )
    report_lines += ['', '## Aggregate', f"- mean_ape_h1={replay_df['ape_h1'].mean():.4f}", f"- mean_ape_h2={replay_df['ape_h2'].mean():.4f}", f"- h2_gt_h1_rate={replay_df['h2_gt_h1'].mean():.4f}"]
    (output_dir / 'trained_model_spike_window_replay.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    payload = {
        'train_cutoff': str(latest_cutoff.date()),
        'replay_count': int(len(replay_df)),
        'replay_rows': replay_df.to_dict(orient='records'),
    }
    (output_dir / 'trained_model_spike_window_replay.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
