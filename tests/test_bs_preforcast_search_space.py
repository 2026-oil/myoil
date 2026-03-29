from __future__ import annotations

import pytest

from plugins.bs_preforcast.search_space import (
    BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY,
    SUPPORTED_BS_PREFORCAST_MODELS,
    normalize_bs_preforcast_sections,
)
from tuning.search_space import (
    BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY as LEGACY_STAGE_ONLY_PARAM_REGISTRY,
    SUPPORTED_BS_PREFORCAST_MODELS as LEGACY_SUPPORTED_BS_PREFORCAST_MODELS,
)


def test_bs_preforcast_search_space_authority_is_exported_from_top_level_package() -> None:
    assert LEGACY_SUPPORTED_BS_PREFORCAST_MODELS == SUPPORTED_BS_PREFORCAST_MODELS
    assert LEGACY_STAGE_ONLY_PARAM_REGISTRY == BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY


def test_bs_preforcast_stage_only_registry_uses_native_list_choices_for_direct_models() -> None:
    assert BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY["ARIMA"]["order"]["choices"] == [
        [1, 0, 0],
        [1, 1, 0],
        [2, 1, 0],
    ]
    assert BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY["xgboost"]["lags"]["choices"] == [
        [1, 2, 3],
        [1, 2, 3, 6, 12],
    ]
    assert BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY["lightgbm"]["lags"]["choices"] == [
        [1, 2, 3],
        [1, 2, 3, 6, 12],
    ]


def test_normalize_bs_preforcast_sections_uses_dedicated_section_names() -> None:
    payload = {
        "bs_preforcast_models": {"TFT": {"hidden_size": {"type": "categorical", "choices": [64]}}},
        "bs_preforcast_training": {"global": {}, "per_model": {}},
    }

    def fake_normalize_model_section(
        value, *, section: str, allowed_models
    ):  # type: ignore[no-untyped-def]
        assert section == "bs_preforcast_models"
        assert allowed_models == SUPPORTED_BS_PREFORCAST_MODELS
        return value

    def fake_normalize_training_section(
        value, *, section: str, allowed_models
    ):  # type: ignore[no-untyped-def]
        assert section == "bs_preforcast_training"
        assert allowed_models == SUPPORTED_BS_PREFORCAST_MODELS
        return value

    model_specs, training_specs = normalize_bs_preforcast_sections(
        payload,
        normalize_model_section=fake_normalize_model_section,
        normalize_training_section=fake_normalize_training_section,
    )

    assert model_specs == payload["bs_preforcast_models"]
    assert training_specs == payload["bs_preforcast_training"]


def test_normalize_bs_preforcast_sections_rejects_stringified_list_choices() -> None:
    payload = {
        "bs_preforcast_models": {
            "ARIMA": {"order": {"type": "categorical", "choices": ["[1, 1, 0]"]}},
            "NHITS": {"mlp_units": {"type": "categorical", "choices": ["[[32, 32], [32, 32], [32, 32]]"]}},
        },
        "bs_preforcast_training": {"global": {}, "per_model": {}},
    }

    with pytest.raises(ValueError, match="native YAML"):
        normalize_bs_preforcast_sections(
            payload,
            normalize_model_section=lambda value, **_kwargs: value,  # type: ignore[no-any-return]
            normalize_training_section=lambda value, **_kwargs: value,  # type: ignore[no-any-return]
        )


def test_normalize_bs_preforcast_sections_accepts_native_list_choices() -> None:
    payload = {
        "bs_preforcast_models": {
            "ARIMA": {"order": {"type": "categorical", "choices": [[1, 1, 0]]}},
            "xgboost": {"lags": {"type": "categorical", "choices": [[1, 2, 3, 6, 12]]}},
            "lightgbm": {"lags": {"type": "categorical", "choices": [[1, 2, 3, 6, 12]]}},
            "NHITS": {
                "mlp_units": {
                    "type": "categorical",
                    "choices": [
                        [[32, 32], [32, 32], [32, 32]],
                        [[64, 64], [64, 64], [64, 64]],
                    ],
                }
            },
        },
        "bs_preforcast_training": {"global": {}, "per_model": {}},
    }

    model_specs, _training_specs = normalize_bs_preforcast_sections(
        payload,
        normalize_model_section=lambda value, **_kwargs: value,  # type: ignore[no-any-return]
        normalize_training_section=lambda value, **_kwargs: value,  # type: ignore[no-any-return]
    )

    assert model_specs["ARIMA"]["order"]["choices"][0] == [1, 1, 0]
    assert model_specs["NHITS"]["mlp_units"]["choices"][1] == [
        [64, 64],
        [64, 64],
        [64, 64],
    ]
