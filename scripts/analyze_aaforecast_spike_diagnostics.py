from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.aa_forecast import STARFeatureExtractor
DEFAULT_VARIABLES = [
    'GPRD_THREAT',
    'BS_Core_Index_A',
    'GPRD',
    'GPRD_ACT',
    'BS_Core_Index_B',
    'BS_Core_Index_C',
    'Idx_OVX',
    'Com_LMEX',
    'Com_BloombergCommodity_BCOM',
    'Idx_DxyUSD',
]


@dataclass
class SpikeWindow:
    cutoff: pd.Timestamp
    anchor: float
    future_h1: float
    future_h2: float
    future_h2cum_ret: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyze AAForecast spike diagnostics using STAR decomposition + run artifacts.'
    )
    parser.add_argument(
        '--config',
        default='yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml',
        help='Experiment config path.',
    )
    parser.add_argument(
        '--run-root',
        default='runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer_no_retrieval_commonpath6',
        help='Completed run root to attach diagnostics to.',
    )
    parser.add_argument(
        '--cutoff',
        default='2026-02-23',
        help='Cutoff date to inspect as YYYY-MM-DD.',
    )
    parser.add_argument(
        '--input-size',
        type=int,
        default=64,
        help='Rolling input window size.',
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=2,
        help='Prediction horizon.',
    )
    parser.add_argument(
        '--recent-steps',
        type=int,
        default=8,
        help='Recent tail length for rolling diagnostics.',
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'YAML root must be a mapping: {path}')
    return payload


def _safe_std(values: np.ndarray) -> float:
    std = float(np.std(values))
    return std if std > 1e-8 else 1.0


def _tail_mode_for_variable(column: str) -> str:
    return 'upward' if column == 'GPRD_THREAT' else 'two_sided'


def _star_components(values: np.ndarray, *, tail_mode: str, season_length: int, lowess_frac: float, lowess_delta: float, thresh: float) -> dict[str, np.ndarray]:
    extractor = STARFeatureExtractor(
        season_length=season_length,
        lowess_frac=lowess_frac,
        lowess_delta=lowess_delta,
        thresh=thresh,
    )
    tensor = torch.as_tensor(values, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        out = extractor(tensor, tail_modes=(tail_mode,))
    return {k: v.detach().cpu().numpy().reshape(-1) for k, v in out.items()}


def _build_window_feature_rows(df: pd.DataFrame, *, variables: list[str], target_col: str, input_size: int, horizon: int, recent_steps: int, season_length: int, lowess_frac: float, lowess_delta: float, thresh: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for end_idx in range(input_size - 1, len(df) - horizon):
        cutoff = pd.Timestamp(df.iloc[end_idx]['dt'])
        window = df.iloc[end_idx - input_size + 1:end_idx + 1].reset_index(drop=True)
        future = df.iloc[end_idx + 1:end_idx + 1 + horizon][target_col].to_numpy(dtype=float)
        anchor = float(df.iloc[end_idx][target_col])
        base_row: dict[str, Any] = {
            'cutoff': cutoff,
            'anchor': anchor,
            'future_h1': float(future[0]),
            'future_h2': float(future[1]),
            'future_h2cum_ret': float((future[-1] - anchor) / max(abs(anchor), 1e-8)),
        }
        for column in [target_col, *variables]:
            tail_mode = _tail_mode_for_variable(column)
            values = window[column].to_numpy(dtype=float)
            comps = _star_components(
                values,
                tail_mode=tail_mode,
                season_length=season_length,
                lowess_frac=lowess_frac,
                lowess_delta=lowess_delta,
                thresh=thresh,
            )
            anomalies = comps['anomalies']
            residual = comps['residual']
            ranking_score = comps['ranking_score']
            critical_mask = comps['critical_mask'].astype(float)
            prev = values[:-recent_steps]
            tail = values[-recent_steps:]
            recent_anom = anomalies[-recent_steps:]
            row_prefix = 'target' if column == target_col else column
            base_row[f'{row_prefix}__last_z'] = (values[-1] - values.mean()) / _safe_std(values)
            base_row[f'{row_prefix}__tail_shift_z'] = (tail.mean() - prev.mean()) / _safe_std(prev)
            base_row[f'{row_prefix}__tail_slope'] = float(np.polyfit(np.arange(recent_steps), tail, 1)[0])
            base_row[f'{row_prefix}__star_anom_mean'] = float(np.mean(recent_anom))
            base_row[f'{row_prefix}__star_rank_mean'] = float(np.mean(ranking_score[-recent_steps:]))
            base_row[f'{row_prefix}__star_mask_density'] = float(np.mean(critical_mask[-recent_steps:]))
            summaries.append(
                {
                    'cutoff': cutoff,
                    'series': column,
                    'tail_mode': tail_mode,
                    'last_value': float(values[-1]),
                    'trend_last': float(comps['trend'][-1]),
                    'seasonal_last': float(comps['seasonal'][-1]),
                    'anomalies_last': float(comps['anomalies'][-1]),
                    'residual_last': float(residual[-1]),
                    'ranking_last': float(ranking_score[-1]),
                    'critical_last': float(critical_mask[-1]),
                    'recent_anomaly_mean': float(np.mean(recent_anom)),
                    'recent_ranking_mean': float(np.mean(ranking_score[-recent_steps:])),
                    'recent_mask_density': float(np.mean(critical_mask[-recent_steps:])),
                    'tail_shift_z': float((tail.mean() - prev.mean()) / _safe_std(prev)),
                    'tail_slope': float(np.polyfit(np.arange(recent_steps), tail, 1)[0]),
                }
            )
        rows.append(base_row)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _spearman_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = feature_df['future_h2cum_ret']
    for column in feature_df.columns:
        if column in {'cutoff', 'anchor', 'future_h1', 'future_h2', 'future_h2cum_ret'}:
            continue
        rho = target.corr(feature_df[column], method='spearman')
        rows.append({'feature': column, 'spearman_rho': float(rho) if pd.notna(rho) else 0.0})
    return pd.DataFrame(rows).sort_values('spearman_rho', ascending=False).reset_index(drop=True)


def _rf_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    X = feature_df.drop(columns=['cutoff', 'anchor', 'future_h1', 'future_h2', 'future_h2cum_ret'])
    y = feature_df['future_h2cum_ret']
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=1,
        max_depth=5,
        min_samples_leaf=4,
    )
    model.fit(X, y)
    return pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)


def _nearest_windows(feature_df: pd.DataFrame, *, cutoff: pd.Timestamp, rf_table: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    top_features = rf_table.head(8)['feature'].tolist()
    full = feature_df[top_features].to_numpy(dtype=float)
    current = feature_df.loc[feature_df['cutoff'] == cutoff, top_features].iloc[0].to_numpy(dtype=float)
    mean = full.mean(axis=0)
    std = full.std(axis=0)
    std[std < 1e-8] = 1.0
    distances = np.linalg.norm((full - mean) / std - (current - mean) / std, axis=1)
    out = feature_df[['cutoff', 'future_h1', 'future_h2', 'future_h2cum_ret']].copy()
    out['dist_top8'] = distances
    out = out[out['cutoff'] < cutoff].nsmallest(top_k, 'dist_top8')
    return out.reset_index(drop=True)


def _build_report(*, run_root: Path, cutoff: pd.Timestamp, result_df: pd.DataFrame, feature_df: pd.DataFrame, summary_df: pd.DataFrame, spearman_df: pd.DataFrame, rf_df: pd.DataFrame, nearest_df: pd.DataFrame) -> str:
    current = feature_df.loc[feature_df['cutoff'] == cutoff].iloc[0]
    current_summary = summary_df.loc[summary_df['cutoff'] == cutoff].copy()
    lines: list[str] = []
    lines.append('# AAForecast spike diagnostics')
    lines.append('')
    lines.append(f'- cutoff: {cutoff.date()}')
    lines.append(f'- run_root: `{run_root}`')
    lines.append('')
    lines.append('## Result snapshot')
    for _, row in result_df.iterrows():
        ape = abs(float(row['y_hat']) - float(row['y'])) / max(abs(float(row['y'])), 1e-8)
        lines.append(f"- h{int(row['horizon_step'])}: y={row['y']:.6f}, y_hat={row['y_hat']:.6f}, ape={ape:.4f}")
    lines.append('')
    lines.append('## Top Spearman features')
    for _, row in spearman_df.head(12).iterrows():
        value = current[row['feature']]
        percentile = float((feature_df[row['feature']] <= value).mean())
        lines.append(f"- {row['feature']}: rho={row['spearman_rho']:.4f}, current={value:.4f}, pct={percentile:.3f}")
    lines.append('')
    lines.append('## Top RF features')
    for _, row in rf_df.head(12).iterrows():
        value = current[row['feature']]
        percentile = float((feature_df[row['feature']] <= value).mean())
        lines.append(f"- {row['feature']}: importance={row['importance']:.4f}, current={value:.4f}, pct={percentile:.3f}")
    lines.append('')
    lines.append('## Per-series STAR summary at cutoff')
    for _, row in current_summary.sort_values(['series']).iterrows():
        lines.append(
            f"- {row['series']}: tail={row['tail_mode']}, trend_last={row['trend_last']:.4f}, seasonal_last={row['seasonal_last']:.4f}, anomalies_last={row['anomalies_last']:.4f}, residual_last={row['residual_last']:.4f}, ranking_last={row['ranking_last']:.4f}, recent_mask_density={row['recent_mask_density']:.4f}, tail_shift_z={row['tail_shift_z']:.4f}, tail_slope={row['tail_slope']:.4f}"
        )
    lines.append('')
    lines.append('## Nearest historical windows by top RF features')
    for _, row in nearest_df.iterrows():
        lines.append(
            f"- {pd.Timestamp(row['cutoff']).date()}: future_h1={row['future_h1']:.4f}, future_h2={row['future_h2']:.4f}, future_h2cum_ret={row['future_h2cum_ret']:.4f}, dist_top8={row['dist_top8']:.4f}"
        )
    lines.append('')
    lines.append('## Architecture implication')
    lines.append('- Joint exogenous burst coherence should be preserved as a common path signal before horizon decoding.')
    lines.append('- STAR-derived exogenous anomaly density/intensity belongs in shared latent transport, not a horizon-specific branch.')
    lines.append('- Decoder/uncertainty changes should prefer path-level amplification of common burst fields over horizon-targeted biasing.')
    return '\n'.join(lines) + '\n'


def _json_ready_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient='records'):
        normalized: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                normalized[key] = str(value)
            elif isinstance(value, np.generic):
                normalized[key] = value.item()
            else:
                normalized[key] = value
        records.append(normalized)
    return records


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    run_root = ROOT / args.run_root
    output_dir = run_root / 'aa_forecast' / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = _read_yaml(config_path)
    plugin = _read_yaml(ROOT / experiment['aa_forecast']['config_path'])['aa_forecast']
    data_path = ROOT / experiment['dataset']['path']
    df = pd.read_csv(data_path)
    df['dt'] = pd.to_datetime(df['dt'])
    target_col = experiment['dataset']['target_col']
    variables = [column for column in DEFAULT_VARIABLES if column in df.columns]

    feature_df, summary_df = _build_window_feature_rows(
        df,
        variables=variables,
        target_col=target_col,
        input_size=args.input_size,
        horizon=args.horizon,
        recent_steps=args.recent_steps,
        season_length=int(plugin['model_params']['season_length']),
        lowess_frac=float(plugin['lowess_frac']),
        lowess_delta=float(plugin['lowess_delta']),
        thresh=float(plugin['thresh']),
    )
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff not in set(feature_df['cutoff']):
        raise ValueError(f'cutoff {cutoff.date()} not found in rolling feature frame')

    spearman_df = _spearman_table(feature_df)
    rf_df = _rf_table(feature_df)
    nearest_df = _nearest_windows(feature_df, cutoff=cutoff, rf_table=rf_df)
    result_df = pd.read_csv(run_root / 'summary' / 'result.csv')

    feature_df.to_csv(output_dir / 'rolling_feature_frame.csv', index=False)
    summary_df.to_csv(output_dir / 'star_series_summary.csv', index=False)
    spearman_df.to_csv(output_dir / 'spearman_rankings.csv', index=False)
    rf_df.to_csv(output_dir / 'rf_feature_importance.csv', index=False)
    nearest_df.to_csv(output_dir / 'nearest_windows.csv', index=False)

    report = _build_report(
        run_root=run_root,
        cutoff=cutoff,
        result_df=result_df,
        feature_df=feature_df,
        summary_df=summary_df,
        spearman_df=spearman_df,
        rf_df=rf_df,
        nearest_df=nearest_df,
    )
    (output_dir / 'spike_diagnostic_report.md').write_text(report, encoding='utf-8')

    payload = {
        'run_root': str(run_root),
        'output_dir': str(output_dir),
        'cutoff': str(cutoff.date()),
        'top_spearman_features': _json_ready_records(spearman_df.head(12)),
        'top_rf_features': _json_ready_records(rf_df.head(12)),
        'nearest_windows': _json_ready_records(nearest_df),
    }
    (output_dir / 'spike_diagnostic_report.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
