from __future__ import annotations

from bs_preforcast.search_space import (
    BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY,
    SUPPORTED_BS_PREFORCAST_MODELS,
    normalize_bs_preforcast_sections,
)
from residual.optuna_spaces import (
    BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY as LEGACY_STAGE_ONLY_PARAM_REGISTRY,
    SUPPORTED_BS_PREFORCAST_MODELS as LEGACY_SUPPORTED_BS_PREFORCAST_MODELS,
)


def test_bs_preforcast_search_space_authority_is_exported_from_top_level_package() -> None:
    assert LEGACY_SUPPORTED_BS_PREFORCAST_MODELS == SUPPORTED_BS_PREFORCAST_MODELS
    assert LEGACY_STAGE_ONLY_PARAM_REGISTRY == BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY


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
