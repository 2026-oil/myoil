from __future__ import annotations

from typing import Any, Callable

from neuralforecast.models.bs_preforcast_catalog import (
    BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY,
    SUPPORTED_BS_PREFORCAST_MODELS,
)

__all__ = [
    "BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY",
    "SUPPORTED_BS_PREFORCAST_MODELS",
    "normalize_bs_preforcast_sections",
    "rewrite_search_space_error",
    "stage_search_space_payload",
]


def stage_search_space_payload(
    search_space_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if search_space_payload is None:
        return None
    model_payload = search_space_payload.get("bs_preforcast_models", {})
    training_payload = search_space_payload.get(
        "bs_preforcast_training", {"global": {}, "per_model": {}}
    )
    return {
        "__scope__": "bs_preforcast",
        "models": model_payload,
        "training": training_payload,
        "bs_preforcast_models": model_payload,
        "bs_preforcast_training": training_payload,
    }


def rewrite_search_space_error(message: str) -> str:
    return (
        message.replace("search_space.models.", "search_space.bs_preforcast_models.")
        .replace("search_space.training.", "search_space.bs_preforcast_training.")
        .replace("search_space.models ", "search_space.bs_preforcast_models ")
        .replace("search_space.training ", "search_space.bs_preforcast_training ")
    )


def _is_native_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _validate_positive_int_sequence(
    value: Any,
    *,
    path: str,
    expected_len: int | None = None,
    min_value: int = 0,
) -> None:
    if isinstance(value, str) or not _is_native_sequence(value):
        raise ValueError(f"{path} choices must use native YAML list values, not string literals")
    values = list(value)
    if not values:
        raise ValueError(f"{path} choices must be non-empty lists")
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"{path} choices must contain exactly {expected_len} integers")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < min_value
        for item in values
    ):
        qualifier = "positive" if min_value > 0 else "non-negative"
        raise ValueError(f"{path} choices must contain only {qualifier} integers")


def _validate_positive_nested_int_sequence(value: Any, *, path: str) -> None:
    if isinstance(value, str) or not _is_native_sequence(value):
        raise ValueError(f"{path} choices must use native YAML nested-list values, not string literals")
    stacks = list(value)
    if not stacks:
        raise ValueError(f"{path} choices must be non-empty nested lists")
    for stack in stacks:
        if isinstance(stack, str) or not _is_native_sequence(stack):
            raise ValueError(f"{path} choices must be lists of integer lists")
        units = list(stack)
        if not units or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in units):
            raise ValueError(f"{path} choices must contain only positive integers")


def _validate_native_list_search_space_contract(
    bs_preforcast_models: dict[str, dict[str, dict[str, Any]]],
) -> None:
    targeted_specs = (
        (
            "ARIMA",
            "order",
            lambda value, path: _validate_positive_int_sequence(
                value, path=path, expected_len=3
            ),
        ),
        (
            "xgboost",
            "lags",
            lambda value, path: _validate_positive_int_sequence(
                value, path=path, min_value=1
            ),
        ),
        (
            "lightgbm",
            "lags",
            lambda value, path: _validate_positive_int_sequence(
                value, path=path, min_value=1
            ),
        ),
        ("NHITS", "mlp_units", _validate_positive_nested_int_sequence),
    )
    for model_name, selector_name, validator in targeted_specs:
        model_specs = bs_preforcast_models.get(model_name)
        if model_specs is None or selector_name not in model_specs:
            continue
        spec = model_specs[selector_name]
        path = f"search_space.bs_preforcast_models.{model_name}.{selector_name}"
        if spec.get("type") != "categorical":
            raise ValueError(f"{path} must declare type: categorical")
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{path}.choices must be a non-empty list")
        for choice in choices:
            validator(choice, path=path)


def normalize_bs_preforcast_sections(
    payload: dict[str, Any],
    *,
    normalize_model_section: Callable[..., dict[str, Any]],
    normalize_training_section: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bs_preforcast_models = normalize_model_section(
        payload.get("bs_preforcast_models"),
        section="bs_preforcast_models",
        allowed_models=SUPPORTED_BS_PREFORCAST_MODELS,
    )
    bs_preforcast_training = normalize_training_section(
        payload.get("bs_preforcast_training"),
        section="bs_preforcast_training",
        allowed_models=SUPPORTED_BS_PREFORCAST_MODELS,
    )
    _validate_native_list_search_space_contract(bs_preforcast_models)
    return bs_preforcast_models, bs_preforcast_training
