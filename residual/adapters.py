from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import JobConfig


@dataclass(frozen=True)
class AdapterInputs:
    fit_df: pd.DataFrame
    futr_df: pd.DataFrame | None
    static_df: pd.DataFrame | None
    channel_map: dict[str, int] | None
    metadata: dict[str, Any]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def build_univariate_inputs(
    source_df: pd.DataFrame,
    job: JobConfig,
    *,
    dt_col: str,
    future_df: pd.DataFrame | None = None,
) -> AdapterInputs:
    required = [dt_col, job.target_col, *job.hist_exog_cols, *job.futr_exog_cols]
    _require_columns(source_df, required)
    fit_df = source_df[required].copy()
    fit_df.rename(columns={dt_col: 'ds', job.target_col: 'y'}, inplace=True)
    fit_df['ds'] = pd.to_datetime(fit_df['ds'])
    fit_df.insert(0, 'unique_id', job.target_col)

    futr_df = None
    if job.futr_exog_cols:
        if future_df is None:
            raise ValueError('future_df is required when futr_exog_cols are configured')
        futr_required = [dt_col, *job.futr_exog_cols]
        _require_columns(future_df, futr_required)
        futr_df = future_df[futr_required].copy()
        futr_df.rename(columns={dt_col: 'ds'}, inplace=True)
        futr_df['ds'] = pd.to_datetime(futr_df['ds'])
        futr_df['ds'] = pd.to_datetime(futr_df['ds'])
        futr_df.insert(0, 'unique_id', job.target_col)

    static_df = None
    if job.static_exog_cols:
        _require_columns(source_df, list(job.static_exog_cols))
        static_payload = {column: source_df[column].iloc[-1] for column in job.static_exog_cols}
        static_payload['unique_id'] = job.target_col
        static_df = pd.DataFrame([static_payload])

    return AdapterInputs(
        fit_df=fit_df,
        futr_df=futr_df,
        static_df=static_df,
        channel_map=None,
        metadata={'job_type': job.job_type, 'target_col': job.target_col},
    )


def build_multivariate_inputs(
    source_df: pd.DataFrame,
    job: JobConfig,
    *,
    dt_col: str,
    future_df: pd.DataFrame | None = None,
) -> AdapterInputs:
    channel_columns = [job.target_col, *job.channel_cols]
    _require_columns(source_df, [dt_col, *channel_columns])
    fit_df = source_df[[dt_col, *channel_columns]].melt(
        id_vars=[dt_col],
        value_vars=channel_columns,
        var_name='unique_id',
        value_name='y',
    )
    fit_df.rename(columns={dt_col: 'ds'}, inplace=True)
    fit_df['ds'] = pd.to_datetime(fit_df['ds'])

    futr_df = None
    if future_df is not None and job.futr_exog_cols:
        futr_required = [dt_col, *job.futr_exog_cols]
        _require_columns(future_df, futr_required)
        futr_df = future_df[futr_required].copy()
        futr_df.rename(columns={dt_col: 'ds'}, inplace=True)
        futr_df['ds'] = pd.to_datetime(futr_df['ds'])

    static_df = None
    if job.static_exog_cols:
        _require_columns(source_df, list(job.static_exog_cols))
        static_rows = []
        for unique_id in channel_columns:
            row = {'unique_id': unique_id}
            row.update({column: source_df[column].iloc[-1] for column in job.static_exog_cols})
            static_rows.append(row)
        static_df = pd.DataFrame(static_rows)

    channel_map = {column: index for index, column in enumerate(channel_columns)}
    return AdapterInputs(
        fit_df=fit_df,
        futr_df=futr_df,
        static_df=static_df,
        channel_map=channel_map,
        metadata={
            'job_type': job.job_type,
            'target_col': job.target_col,
            'n_series': len(channel_columns),
        },
    )
