from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .bs_preforcast_catalog import is_direct_stage_model


def _parse_direct_model_literal(
    value: Any,
    *,
    field_name: str,
) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"bs_preforcast direct model field '{field_name}' cannot be empty")
        raise ValueError(
            f"bs_preforcast direct model field '{field_name}' must use native YAML list values, not string literals"
        )
    return value


def _coerce_arima_triplet(
    value: Any,
    *,
    field_name: str,
    default: tuple[int, int, int],
) -> tuple[int, int, int]:
    candidate = default if value is None else _parse_direct_model_literal(
        value, field_name=field_name
    )
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
        raise ValueError(
            f"bs_preforcast ARIMA field '{field_name}' must be a 3-item list/tuple"
        )
    triplet = tuple(int(item) for item in candidate)
    if any(item < 0 for item in triplet):
        raise ValueError(
            f"bs_preforcast ARIMA field '{field_name}' cannot contain negative values"
        )
    return triplet


def _coerce_tree_lags(value: Any) -> int | list[int]:
    candidate = _parse_direct_model_literal(value, field_name="lags")
    if isinstance(candidate, np.ndarray):
        candidate = candidate.tolist()
    if isinstance(candidate, range):
        candidate = list(candidate)
    if isinstance(candidate, int):
        if candidate < 1:
            raise ValueError("bs_preforcast direct tree lags must be positive")
        return candidate
    if not isinstance(candidate, (list, tuple)):
        raise ValueError(
            "bs_preforcast direct tree lags must be an int or a list of ints"
        )
    lags = [int(item) for item in candidate]
    if not lags or any(item < 1 for item in lags):
        raise ValueError(
            "bs_preforcast direct tree lags must be a non-empty list of positive ints"
        )
    seen: set[int] = set()
    ordered_unique: list[int] = []
    for lag in lags:
        if lag not in seen:
            seen.add(lag)
            ordered_unique.append(lag)
    return ordered_unique


def _max_lag(lags: int | list[int]) -> int:
    return lags if isinstance(lags, int) else max(lags)


def normalized_direct_job_params(
    model_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(params)
    if model_name == "ARIMA":
        if "order" in normalized:
            normalized["order"] = _coerce_arima_triplet(
                normalized["order"],
                field_name="order",
                default=(1, 0, 0),
            )
        if "seasonal_order" in normalized:
            normalized["seasonal_order"] = _coerce_arima_triplet(
                normalized["seasonal_order"],
                field_name="seasonal_order",
                default=(0, 0, 0),
            )
        return normalized
    if model_name in {"xgboost", "lightgbm"} and "lags" in normalized:
        normalized["lags"] = _coerce_tree_lags(normalized["lags"])
    return normalized


def normalized_direct_stage_job(job: Any) -> Any:
    if not is_direct_stage_model(job.model):
        return job
    return replace(job, params=normalized_direct_job_params(job.model, job.params))


def predict_univariate_arima(
    stage_loaded: Any,
    job: Any,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> list[float]:
    from statsforecast import StatsForecast
    from statsforecast.models import ARIMA

    dt_col = stage_loaded.config.dataset.dt_col
    fit_df = train_df[[dt_col, target_column]].copy()
    fit_df.rename(columns={dt_col: "ds", target_column: "y"}, inplace=True)
    fit_df["ds"] = pd.to_datetime(fit_df["ds"])
    fit_df.insert(0, "unique_id", target_column)
    params = dict(job.params)
    params.pop("season_length", None)
    seasonal_order = _coerce_arima_triplet(
        params.pop("seasonal_order", (0, 0, 0)),
        field_name="seasonal_order",
        default=(0, 0, 0),
    )
    if seasonal_order != (0, 0, 0):
        raise ValueError(
            "bs_preforcast ARIMA no longer supports seasonal_order/season_length"
        )
    model = ARIMA(
        order=_coerce_arima_triplet(
            params.pop("order", (1, 0, 0)),
            field_name="order",
            default=(1, 0, 0),
        ),
        season_length=1,
        seasonal_order=seasonal_order,
        include_mean=bool(params.pop("include_mean", True)),
        include_drift=bool(params.pop("include_drift", False)),
        include_constant=params.pop("include_constant", None),
        blambda=params.pop("blambda", None),
        biasadj=bool(params.pop("biasadj", False)),
        method=str(params.pop("method", "CSS-ML")),
        **params,
    )
    freq = stage_loaded.config.dataset.freq
    if not freq:
        inferred = pd.infer_freq(pd.to_datetime(train_df[stage_loaded.config.dataset.dt_col]))
        freq = inferred or "W"
    sf = StatsForecast(models=[model], freq=freq)
    fitted = sf.fit(df=fit_df)
    predictions = fitted.predict(h=len(future_df))
    pred_col = next(
        column for column in predictions.columns if column not in {"unique_id", "ds"}
    )
    return [float(value) for value in predictions[pred_col].tolist()]


def predict_univariate_es(
    _stage_loaded: Any,
    job: Any,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> list[float]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = train_df[target_column].astype(float)
    params = dict(job.params)
    params.pop("season_length", None)
    kwargs: dict[str, Any] = {
        "trend": params.pop("trend", None),
        "seasonal": params.pop("seasonal", None),
        "damped_trend": bool(params.pop("damped_trend", False)),
        "initialization_method": str(
            params.pop("initialization_method", "estimated")
        ),
    }
    if kwargs["seasonal"] is not None:
        raise ValueError(
            "bs_preforcast ES no longer supports seasonal components"
        )
    fitted = ExponentialSmoothing(series, **kwargs).fit(**params)
    forecast = fitted.forecast(len(future_df))
    return [float(value) for value in forecast.tolist()]


def predict_univariate_tree(
    stage_loaded: Any,
    job: Any,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    model_name: str,
) -> list[float]:
    from skforecast.direct import ForecasterDirect

    raw_lags = job.params.get("lags", stage_loaded.config.training.input_size)
    lags = _coerce_tree_lags(raw_lags)
    max_lag = _max_lag(lags)
    if len(train_df) <= max_lag:
        raise ValueError(
            "bs_preforcast tree stage requires more history before forecasting "
            f"target column: {target_column}"
        )
    params = dict(job.params)
    params.pop("lags", None)
    if "learning_rate" in params:
        raise ValueError(
            "bs_preforcast tree-stage learning_rate is no longer configurable; scheduler and optimizer-rate defaults are internal-only"
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        regressor = XGBRegressor(
            n_estimators=int(params.pop("n_estimators", 32)),
            max_depth=int(params.pop("max_depth", 3)),
            learning_rate=0.1,
            objective="reg:squarederror",
            n_jobs=1,
            verbosity=0,
            **params,
        )
    else:
        from lightgbm import LGBMRegressor

        regressor = LGBMRegressor(
            n_estimators=int(params.pop("n_estimators", 64)),
            max_depth=int(params.pop("max_depth", -1)),
            learning_rate=0.05,
            verbosity=-1,
            **params,
        )
    forecaster = ForecasterDirect(
        regressor=regressor,
        steps=len(future_df),
        lags=lags,
    )
    forecaster.fit(
        y=train_df[target_column].astype(float).reset_index(drop=True),
        suppress_warnings=True,
    )
    predictions = forecaster.predict(steps=list(range(1, len(future_df) + 1)))
    return [float(value) for value in predictions.to_list()]


def predict_univariate_direct(
    stage_loaded: Any,
    job: Any,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> list[float]:
    if job.model == "ARIMA":
        return predict_univariate_arima(
            stage_loaded,
            job,
            target_column=target_column,
            train_df=train_df,
            future_df=future_df,
        )
    if job.model == "ES":
        return predict_univariate_es(
            stage_loaded,
            job,
            target_column=target_column,
            train_df=train_df,
            future_df=future_df,
        )
    if job.model in {"xgboost", "lightgbm"}:
        return predict_univariate_tree(
            stage_loaded,
            job,
            target_column=target_column,
            train_df=train_df,
            future_df=future_df,
            model_name=job.model,
        )
    raise ValueError(f"Unsupported bs_preforcast direct stage model: {job.model}")
