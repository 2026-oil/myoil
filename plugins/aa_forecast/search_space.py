from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

__all__ = [
    "AA_FORECAST_STAGE_ONLY_PARAM_REGISTRY",
    "SUPPORTED_AA_FORECAST_BACKBONES",
    "normalize_aa_forecast_sections",
    "rewrite_search_space_error",
    "stage_search_space_payload",
]

SUPPORTED_AA_FORECAST_BACKBONES = {
    "gru",
    "vanillatransformer",
    "informer",
    "itransformer",
    "patchtst",
    "timexer",
}


def _positive_int(*, low: int = 1, high: int = 4096) -> dict[str, Any]:
    return {"type": "int", "low": low, "high": high, "step": 1}


def _probability() -> dict[str, Any]:
    return {"type": "float", "low": 0.0, "high": 0.999}


_AA_COMMON_PARAM_REGISTRY = {
    "decoder_hidden_size": _positive_int(low=1, high=4096),
    "decoder_layers": _positive_int(low=1, high=16),
    "attention_hidden_size": {
        "type": "categorical",
        "choices": [None, 32, 64, 128, 256],
    },
    "season_length": _positive_int(low=1, high=1024),
    "trend_kernel_size": {"type": "categorical", "choices": [None, 3, 5, 7, 9, 11]},
    "start_padding_enabled": {"type": "categorical", "choices": [True, False]},
}

AA_FORECAST_STAGE_ONLY_PARAM_REGISTRY = {
    "gru": {
        **_AA_COMMON_PARAM_REGISTRY,
        "encoder_hidden_size": _positive_int(low=1, high=4096),
        "encoder_n_layers": _positive_int(low=1, high=16),
        "encoder_dropout": _probability(),
    },
    "vanillatransformer": {
        **_AA_COMMON_PARAM_REGISTRY,
        "hidden_size": _positive_int(low=1, high=4096),
        "n_head": _positive_int(low=1, high=128),
        "encoder_layers": _positive_int(low=1, high=16),
        "dropout": _probability(),
        "linear_hidden_size": _positive_int(low=1, high=8192),
    },
    "informer": {
        **_AA_COMMON_PARAM_REGISTRY,
        "hidden_size": _positive_int(low=1, high=4096),
        "n_head": _positive_int(low=1, high=128),
        "encoder_layers": _positive_int(low=1, high=16),
        "dropout": _probability(),
        "linear_hidden_size": _positive_int(low=1, high=8192),
        "factor": _positive_int(low=1, high=64),
        "semantic_negative_scale": {"type": "float", "low": 0.0, "high": 2.0},
    },
    "itransformer": {
        **_AA_COMMON_PARAM_REGISTRY,
        "hidden_size": _positive_int(low=1, high=4096),
        "n_heads": _positive_int(low=1, high=128),
        "e_layers": _positive_int(low=1, high=16),
        "dropout": _probability(),
        "d_ff": _positive_int(low=1, high=8192),
        "factor": _positive_int(low=1, high=64),
        "use_norm": {"type": "categorical", "choices": [True, False]},
    },
    "patchtst": {
        **_AA_COMMON_PARAM_REGISTRY,
        "hidden_size": _positive_int(low=1, high=4096),
        "n_heads": _positive_int(low=1, high=128),
        "encoder_layers": _positive_int(low=1, high=16),
        "dropout": _probability(),
        "linear_hidden_size": _positive_int(low=1, high=8192),
        "attn_dropout": _probability(),
        "patch_len": _positive_int(low=1, high=1024),
        "stride": _positive_int(low=1, high=1024),
    },
    "timexer": {
        **_AA_COMMON_PARAM_REGISTRY,
        "hidden_size": _positive_int(low=1, high=4096),
        "n_heads": _positive_int(low=1, high=128),
        "e_layers": _positive_int(low=1, high=16),
        "dropout": _probability(),
        "d_ff": _positive_int(low=1, high=8192),
        "patch_len": _positive_int(low=1, high=1024),
        "use_norm": {"type": "categorical", "choices": [True, False]},
    },
}


def stage_search_space_payload(
    search_space_payload: dict[str, Any] | None,
    *,
    backbone: str,
) -> dict[str, Any] | None:
    if search_space_payload is None:
        return None
    normalized_backbone = str(backbone).strip().lower()
    model_payload = deepcopy(search_space_payload.get("aa_forecast_models", {}))
    if normalized_backbone not in model_payload:
        raise ValueError(
            "search_space.aa_forecast_models is missing an entry for "
            f"aa_forecast.model={normalized_backbone!r}"
        )
    training_payload = deepcopy(
        search_space_payload.get(
            "aa_forecast_training",
            {"global": {}, "per_model": {}},
        )
    )
    global_training = deepcopy(training_payload.get("global", {}))
    per_backbone_training = deepcopy(training_payload.get("per_model", {}))
    selected_training = deepcopy(
        per_backbone_training.get(normalized_backbone, global_training)
    )
    return {
        "__scope__": "aa_forecast",
        "models": {"AAForecast": deepcopy(model_payload[normalized_backbone])},
        "training": {
            "global": global_training,
            "per_model": {"AAForecast": selected_training},
        },
        "aa_forecast_models": model_payload,
        "aa_forecast_training": training_payload,
    }


def rewrite_search_space_error(message: str) -> str:
    return (
        message.replace("search_space.models.", "search_space.aa_forecast_models.")
        .replace("search_space.training.", "search_space.aa_forecast_training.")
        .replace("search_space.models ", "search_space.aa_forecast_models ")
        .replace("search_space.training ", "search_space.aa_forecast_training ")
    )


def normalize_aa_forecast_sections(
    payload: dict[str, Any],
    *,
    normalize_model_section: Callable[..., dict[str, Any]],
    normalize_training_section: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aa_forecast_models = normalize_model_section(
        payload.get("aa_forecast_models"),
        section="aa_forecast_models",
        allowed_models=SUPPORTED_AA_FORECAST_BACKBONES,
    )
    aa_forecast_training = normalize_training_section(
        payload.get("aa_forecast_training"),
        section="aa_forecast_training",
        allowed_models=SUPPORTED_AA_FORECAST_BACKBONES,
    )
    return aa_forecast_models, aa_forecast_training
