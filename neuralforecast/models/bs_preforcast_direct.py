from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from .bs_preforcast_catalog import is_direct_stage_model


@dataclass(frozen=True)
class DirectPredictionResult:
    predictions: list[float]
    curve_frame: pd.DataFrame | None = None


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


def _lag_list(lags: int | list[int]) -> list[int]:
    return [lags] if isinstance(lags, int) else list(lags)


def _build_hist_exog_feature_frame(
    source_df: pd.DataFrame,
    *,
    hist_exog_cols: tuple[str, ...],
    lags: int | list[int],
) -> pd.DataFrame | None:
    if not hist_exog_cols:
        return None
    missing = [column for column in hist_exog_cols if column not in source_df.columns]
    if missing:
        raise ValueError(
            "direct tree models require configured dataset.hist_exog_cols to exist in the fold frame: "
            + ", ".join(sorted(missing))
        )
    feature_columns: dict[str, pd.Series] = {}
    for column in hist_exog_cols:
        series = source_df[column].astype(float).reset_index(drop=True)
        for lag in _lag_list(lags):
            feature_columns[f"{column}_lag_{lag}"] = series.shift(lag)
    return pd.DataFrame(feature_columns)


def _build_hist_exog_feature_frames(
    *,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    hist_exog_cols: tuple[str, ...],
    lags: int | list[int],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not hist_exog_cols:
        return None, None
    train_exog = _build_hist_exog_feature_frame(
        train_df,
        hist_exog_cols=hist_exog_cols,
        lags=lags,
    )
    combined = pd.concat(
        [
            train_df.loc[:, list(hist_exog_cols)],
            future_df.loc[:, list(hist_exog_cols)],
        ],
        ignore_index=True,
    )
    combined_exog = _build_hist_exog_feature_frame(
        combined,
        hist_exog_cols=hist_exog_cols,
        lags=lags,
    )
    assert train_exog is not None
    assert combined_exog is not None
    future_exog = combined_exog.iloc[len(train_df) : len(train_df) + len(future_df)].copy()
    return train_exog, future_exog


def _build_tree_regressor(model_name: str, params: dict[str, Any]) -> Any:
    resolved_params = dict(params)
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=int(resolved_params.pop("n_estimators", 32)),
            max_depth=int(resolved_params.pop("max_depth", 3)),
            learning_rate=0.1,
            objective="reg:squarederror",
            n_jobs=1,
            verbosity=0,
            **resolved_params,
        )

    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=int(resolved_params.pop("n_estimators", 64)),
        max_depth=int(resolved_params.pop("max_depth", -1)),
        learning_rate=0.05,
        verbosity=-1,
        **resolved_params,
    )


def _resolve_tree_history_metric(
    model_name: str,
    training_loss: str,
) -> tuple[str, Callable[[float], float]]:
    normalized_loss = training_loss.lower()
    if normalized_loss != "mse":
        raise ValueError(
            "bs_preforcast tree loss curves currently support only training.loss=mse"
        )
    if model_name == "xgboost":
        return "rmse", lambda value: float(value) ** 2
    return "l2", float


def _extract_tree_history_frame(
    estimator: Any,
    *,
    model_name: str,
    training_loss: str,
) -> pd.DataFrame:
    metric_name, postprocess = _resolve_tree_history_metric(model_name, training_loss)
    raw_history: list[float]
    if model_name == "xgboost":
        evals_result = estimator.evals_result()
        dataset_history = next(iter(evals_result.values()), {})
        raw_history = list(dataset_history.get(metric_name, []))
    else:
        evals_result = getattr(estimator, "evals_result_", {})
        dataset_history = next(iter(evals_result.values()), {})
        raw_history = list(dataset_history.get(metric_name, []))
    if not raw_history:
        return pd.DataFrame(columns=["global_step", "train_loss", "val_loss"])
    return pd.DataFrame(
        {
            "global_step": np.arange(1, len(raw_history) + 1, dtype=int),
            "train_loss": [postprocess(value) for value in raw_history],
            "val_loss": [np.nan] * len(raw_history),
        }
    )


def _build_tree_curve_frame(
    forecaster: Any,
    series: pd.Series,
    *,
    model_name: str,
    training_loss: str,
    train_exog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from sklearn.base import clone

    X_train, y_train = forecaster.create_train_X_y(y=series, exog=train_exog)
    step_frames: list[pd.DataFrame] = []
    metric_name, _ = _resolve_tree_history_metric(model_name, training_loss)
    for step in forecaster.steps:
        X_step, y_step = forecaster.filter_train_X_y_for_step(
            step=int(step),
            X_train=X_train,
            y_train=y_train,
            remove_suffix=True,
        )
        estimator = clone(forecaster.estimator)
        fit_kwargs: dict[str, Any] = {"eval_set": [(X_step, y_step)]}
        if model_name == "xgboost":
            estimator.set_params(eval_metric=metric_name)
            fit_kwargs["verbose"] = False
        else:
            fit_kwargs["eval_metric"] = metric_name
            fit_kwargs["eval_names"] = ["training"]
        estimator.fit(X_step, y_step, **fit_kwargs)
        step_frame = _extract_tree_history_frame(
            estimator,
            model_name=model_name,
            training_loss=training_loss,
        )
        if step_frame.empty:
            continue
        step_frame["step"] = int(step)
        step_frames.append(step_frame)
    if not step_frames:
        return pd.DataFrame(columns=["global_step", "train_loss", "val_loss"])
    combined = pd.concat(step_frames, ignore_index=True)
    aggregated = (
        combined.groupby("global_step", as_index=False)
        .agg(train_loss=("train_loss", "mean"))
        .sort_values("global_step", kind="stable")
        .reset_index(drop=True)
    )
    aggregated["val_loss"] = np.nan
    return aggregated[["global_step", "train_loss", "val_loss"]]


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
) -> DirectPredictionResult:
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
    regressor = _build_tree_regressor(model_name, params)
    forecaster = ForecasterDirect(
        regressor=regressor,
        steps=len(future_df),
        lags=lags,
    )
    train_exog, future_exog = _build_hist_exog_feature_frames(
        train_df=train_df,
        future_df=future_df,
        hist_exog_cols=stage_loaded.config.dataset.hist_exog_cols,
        lags=lags,
    )
    series = train_df[target_column].astype(float).reset_index(drop=True)
    forecaster.fit(
        y=series,
        exog=train_exog,
        suppress_warnings=True,
    )
    predictions = forecaster.predict(
        steps=list(range(1, len(future_df) + 1)),
        exog=future_exog,
    )
    curve_frame = _build_tree_curve_frame(
        forecaster,
        series,
        model_name=model_name,
        training_loss=stage_loaded.config.training.loss,
        train_exog=train_exog,
    )
    return DirectPredictionResult(
        predictions=[float(value) for value in predictions.to_list()],
        curve_frame=curve_frame,
    )


def predict_univariate_direct(
    stage_loaded: Any,
    job: Any,
    *,
    target_column: str,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> list[float] | DirectPredictionResult:
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
