from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

_BASE_FEATURE_KEYS = (
    "include_base_prediction",
    "include_horizon_step",
    "include_date_features",
    "hist",
    "futr",
    "static",
    "lag",
    "lag_sources",
    "lag_steps",
)
_FORBIDDEN_LAG_SOURCES = {"y", "residual_target"}


def hist_exog_lag_feature_name(column: str) -> str:
    return f"{column}_lag_1"


@dataclass(frozen=True)
class ResidualFeatureConfig:
    include_base_prediction: bool = True
    include_horizon_step: bool = True
    include_date_features: bool = False
    hist: tuple[str, ...] = ()
    futr: tuple[str, ...] = ()
    static: tuple[str, ...] = ()
    lag_sources: tuple[str, ...] = ()
    lag_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResidualFeatureFrame:
    frame: pd.DataFrame
    columns: tuple[str, ...]
    resolved_config: ResidualFeatureConfig


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _coerce_name_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(item) for item in value if str(item))


def _coerce_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (int(value),)
    return tuple(int(item) for item in value)


def _mapping_from_dataclass(value: Any) -> dict[str, Any]:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if hasattr(value, field.name)
    }


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        return _mapping_from_dataclass(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _extract_feature_payload(source: Any) -> dict[str, Any]:
    payload = _coerce_mapping(source)
    if not payload:
        return {}
    if "residual" in payload:
        return _extract_feature_payload(payload.get("residual"))
    if "features" in payload and not any(key in payload for key in _BASE_FEATURE_KEYS):
        nested = payload.get("features")
        nested_payload = _coerce_mapping(nested)
        if nested_payload:
            return nested_payload
    if "features" in payload and any(
        key in payload for key in ("enabled", "model", "params", "target")
    ):
        nested_payload = _coerce_mapping(payload.get("features"))
        return nested_payload
    return payload


def resolve_residual_feature_config(source: Any = None) -> ResidualFeatureConfig:
    payload = _extract_feature_payload(source)
    lag_payload = _coerce_mapping(payload.get("lag_features"))
    if not lag_payload:
        lag_payload = _coerce_mapping(payload.get("lag"))

    exog_payload = _coerce_mapping(payload.get("exog_sources"))
    if not exog_payload:
        exog_payload = _coerce_mapping(payload.get("exog"))

    lag_sources = payload.get("lag_sources", lag_payload.get("sources"))
    lag_steps = payload.get("lag_steps", lag_payload.get("steps"))
    hist = payload.get("hist", exog_payload.get("hist"))
    futr = payload.get("futr", exog_payload.get("futr"))
    static = payload.get("static", exog_payload.get("static"))
    return ResidualFeatureConfig(
        include_base_prediction=_coerce_bool(
            payload.get("include_base_prediction"), default=True
        ),
        include_horizon_step=_coerce_bool(
            payload.get("include_horizon_step"), default=True
        ),
        include_date_features=_coerce_bool(
            payload.get("include_date_features"), default=False
        ),
        hist=_coerce_name_tuple(hist),
        futr=_coerce_name_tuple(futr),
        static=_coerce_name_tuple(static),
        lag_sources=_coerce_name_tuple(lag_sources),
        lag_steps=_coerce_int_tuple(lag_steps),
    )


def _prepare_panel(
    panel_df: pd.DataFrame,
    *,
    include_date_features: bool,
) -> pd.DataFrame:
    panel = panel_df.copy()
    if "horizon_step" in panel:
        panel["horizon_step"] = pd.to_numeric(panel["horizon_step"], errors="coerce")
    if "y_hat_base" in panel:
        panel["y_hat_base"] = pd.to_numeric(panel["y_hat_base"], errors="coerce")
    if include_date_features:
        for column in ("cutoff", "ds"):
            if column in panel:
                panel[column] = pd.to_datetime(panel[column])
        if "cutoff" in panel:
            panel["cutoff_day"] = panel["cutoff"].astype("int64") // 86_400_000_000_000
        if "ds" in panel:
            panel["ds_day"] = panel["ds"].astype("int64") // 86_400_000_000_000
    return panel


def _lag_group_columns(panel: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("model_name", "fold_idx", "panel_split", "unique_id", "cutoff")
        if column in panel.columns
    ]


def _sort_columns(panel: pd.DataFrame, group_columns: Sequence[str]) -> list[str]:
    sort_columns = list(group_columns)
    for column in ("ds", "horizon_step"):
        if column in panel.columns and column not in sort_columns:
            sort_columns.append(column)
    return sort_columns


def _build_lag_features(
    panel: pd.DataFrame,
    config: ResidualFeatureConfig,
) -> dict[str, pd.Series]:
    lagged: dict[str, pd.Series] = {}
    if not config.lag_sources or not config.lag_steps:
        return lagged
    group_columns = _lag_group_columns(panel)
    ordered = panel.copy()
    sort_columns = _sort_columns(ordered, group_columns)
    if sort_columns:
        ordered = ordered.sort_values(sort_columns).reset_index()
    else:
        ordered = ordered.reset_index()
    for source in config.lag_sources:
        if source in _FORBIDDEN_LAG_SOURCES:
            raise ValueError(f"Unsupported residual lag source: {source}")
        if source not in ordered.columns:
            continue
        values = pd.to_numeric(ordered[source], errors="coerce")
        for step in config.lag_steps:
            if step <= 0:
                raise ValueError(
                    f"Residual lag steps must be positive integers, got {step}"
                )
            feature_name = f"{source}_lag_{step}"
            if group_columns:
                lagged_values = values.groupby(
                    [ordered[column] for column in group_columns],
                    sort=False,
                ).shift(step)
            else:
                lagged_values = values.shift(step)
            lagged[feature_name] = (
                lagged_values
                .set_axis(ordered["index"])
                .reindex(panel.index)
                .astype(float)
            )
    return lagged


def build_residual_feature_frame(
    panel_df: pd.DataFrame,
    *,
    feature_config: Any = None,
    required_columns: Sequence[str] | None = None,
) -> ResidualFeatureFrame:
    config = resolve_residual_feature_config(feature_config)
    panel = _prepare_panel(panel_df, include_date_features=config.include_date_features)
    features = pd.DataFrame(index=panel.index)

    if config.include_horizon_step and "horizon_step" in panel:
        features["horizon_step"] = pd.to_numeric(panel["horizon_step"], errors="coerce")
    if config.include_base_prediction and "y_hat_base" in panel:
        features["y_hat_base"] = pd.to_numeric(panel["y_hat_base"], errors="coerce")
    if config.include_date_features:
        for column in ("cutoff_day", "ds_day"):
            if column in panel:
                features[column] = pd.to_numeric(panel[column], errors="coerce")

    for feature_name, feature_values in _build_lag_features(panel, config).items():
        features[feature_name] = feature_values

    for column in config.hist:
        feature_name = hist_exog_lag_feature_name(column)
        source_column = column if column in panel else feature_name
        if source_column in panel and feature_name not in features:
            features[feature_name] = pd.to_numeric(panel[source_column], errors="coerce")

    for column in (*config.futr, *config.static):
        if column in panel and column not in features:
            features[column] = pd.to_numeric(panel[column], errors="coerce")

    if required_columns is not None:
        features = features.reindex(columns=list(required_columns), fill_value=0.0)

    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return ResidualFeatureFrame(
        frame=features,
        columns=tuple(str(column) for column in features.columns),
        resolved_config=config,
    )
