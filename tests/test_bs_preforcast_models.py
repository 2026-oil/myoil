from neuralforecast.models.bs_preforcast_catalog import (
    BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY,
    DIRECT_STAGE_MODEL_NAMES,
    SUPPORTED_BS_PREFORCAST_MODELS,
    is_direct_stage_model,
)
from plugins.bs_preforcast.models import (
    BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY as PLUGIN_STAGE_ONLY_PARAM_REGISTRY,
    DIRECT_STAGE_MODEL_NAMES as PLUGIN_DIRECT_STAGE_MODEL_NAMES,
    SUPPORTED_BS_PREFORCAST_MODELS as PLUGIN_SUPPORTED_BS_PREFORCAST_MODELS,
)


def test_bs_preforcast_models_package_owns_catalog_constants() -> None:
    assert DIRECT_STAGE_MODEL_NAMES == {"ARIMA", "ES", "xgboost", "lightgbm"}
    assert DIRECT_STAGE_MODEL_NAMES.issubset(SUPPORTED_BS_PREFORCAST_MODELS)
    assert "TimeXer" in SUPPORTED_BS_PREFORCAST_MODELS
    assert "TSMixerx" in SUPPORTED_BS_PREFORCAST_MODELS


def test_bs_preforcast_models_package_exposes_direct_stage_registry() -> None:
    assert set(BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY) == DIRECT_STAGE_MODEL_NAMES
    assert is_direct_stage_model("ARIMA") is True
    assert is_direct_stage_model("LSTM") is False


def test_bs_preforcast_plugin_models_are_wrapper_exports_only() -> None:
    assert PLUGIN_DIRECT_STAGE_MODEL_NAMES is DIRECT_STAGE_MODEL_NAMES
    assert PLUGIN_SUPPORTED_BS_PREFORCAST_MODELS is SUPPORTED_BS_PREFORCAST_MODELS
    assert PLUGIN_STAGE_ONLY_PARAM_REGISTRY is BS_PREFORCAST_STAGE_ONLY_PARAM_REGISTRY
