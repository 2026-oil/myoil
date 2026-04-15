from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tuning.search_space import SearchSpaceContract

from .search_space import (
    AA_FORECAST_STAGE_ONLY_PARAM_REGISTRY,
    SUPPORTED_AA_FORECAST_BACKBONES,
    rewrite_search_space_error,
    stage_search_space_payload,
)


AA_FORECAST_MAIN_KEYS = {
    "enabled",
    "config_path",
    "model",
    "tune_training",
    "model_params",
    "thresh",
    "star_anomaly_tails",
    "lowess_frac",
    "lowess_delta",
    "uncertainty",
    "retrieval",
    "encoding_export",
}
AA_FORECAST_LINKED_KEYS = {
    "model",
    "tune_training",
    "model_params",
    "thresh",
    "star_anomaly_tails",
    "lowess_frac",
    "lowess_delta",
    "uncertainty",
    "retrieval",
    "encoding_export",
}
AA_FORECAST_UNCERTAINTY_KEYS = {"enabled", "sample_count", "dropout_candidates"}
AA_FORECAST_STAR_ANOMALY_TAIL_KEYS = {"upward", "two_sided"}
AA_FORECAST_RETRIEVAL_KEYS = {
    "enabled",
    "top_k",
    "recency_gap_steps",
    "event_score_threshold",
    "min_similarity",
    "blend_max",
    "use_uncertainty_gate",
    "use_shape_key",
    "use_event_key",
    "event_score_log_bonus_alpha",
    "event_score_log_bonus_cap",
}
AA_FORECAST_ENCODING_EXPORT_KEYS = {"enabled"}


@dataclass(frozen=True)
class AAForecastUncertaintyConfig:
    enabled: bool = False
    dropout_candidates: tuple[float, ...] = (
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    )
    sample_count: int = 5


@dataclass(frozen=True)
class AAForecastRetrievalConfig:
    enabled: bool = False
    top_k: int = 5
    recency_gap_steps: int = 8
    event_score_threshold: float = 1.0
    min_similarity: float = 0.55
    blend_max: float = 0.25
    use_uncertainty_gate: bool = True
    mode: str = "posthoc_blend"
    similarity: str = "cosine"
    temperature: float = 0.10
    use_shape_key: bool = True
    use_event_key: bool = True
    blend_floor: float = 0.0
    event_score_log_bonus_alpha: float = 0.0
    event_score_log_bonus_cap: float = 0.0


@dataclass(frozen=True)
class AAForecastEncodingExportConfig:
    enabled: bool = False


def _default_star_anomaly_tails() -> dict[str, tuple[str, ...]]:
    return {"upward": (), "two_sided": ()}


@dataclass(frozen=True)
class AAForecastPluginConfig:
    enabled: bool = False
    config_path: str | None = None
    model: str = "gru"
    tune_training: bool = False
    model_params: dict[str, Any] = field(default_factory=dict)
    thresh: float = 3.5
    star_hist_exog_cols_resolved: tuple[str, ...] = field(default_factory=tuple)
    non_star_hist_exog_cols_resolved: tuple[str, ...] = field(default_factory=tuple)
    star_anomaly_tails: dict[str, tuple[str, ...]] = field(
        default_factory=_default_star_anomaly_tails
    )
    star_anomaly_tails_resolved: dict[str, tuple[str, ...]] = field(
        default_factory=_default_star_anomaly_tails
    )
    star_anomaly_tail_modes_resolved: tuple[str, ...] = field(default_factory=tuple)
    lowess_frac: float = 0.6
    lowess_delta: float = 0.01
    uncertainty: AAForecastUncertaintyConfig = field(
        default_factory=AAForecastUncertaintyConfig
    )
    retrieval: AAForecastRetrievalConfig = field(
        default_factory=AAForecastRetrievalConfig
    )
    encoding_export: AAForecastEncodingExportConfig = field(
        default_factory=AAForecastEncodingExportConfig
    )


@dataclass(frozen=True)
class AAForecastStageLoadedConfig:
    config: AAForecastPluginConfig
    source_path: Path
    source_type: str
    normalized_payload: dict[str, Any]
    input_hash: str
    resolved_hash: str
    search_space_path: Path | None = None
    search_space_hash: str | None = None
    search_space_payload: dict[str, Any] | None = None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def aa_forecast_uncertainty_public_dict(
    uncertainty: AAForecastUncertaintyConfig,
) -> dict[str, Any]:
    return {
        "enabled": uncertainty.enabled,
        "dropout_candidates": list(uncertainty.dropout_candidates),
        "sample_count": uncertainty.sample_count,
    }


def aa_forecast_retrieval_public_dict(
    retrieval: AAForecastRetrievalConfig,
) -> dict[str, Any]:
    return {
        "enabled": retrieval.enabled,
        "top_k": retrieval.top_k,
        "recency_gap_steps": retrieval.recency_gap_steps,
        "event_score_threshold": retrieval.event_score_threshold,
        "min_similarity": retrieval.min_similarity,
        "blend_max": retrieval.blend_max,
        "use_uncertainty_gate": retrieval.use_uncertainty_gate,
        "mode": retrieval.mode,
        "similarity": retrieval.similarity,
        "temperature": retrieval.temperature,
        "use_shape_key": retrieval.use_shape_key,
        "use_event_key": retrieval.use_event_key,
        "blend_floor": retrieval.blend_floor,
        "event_score_log_bonus_alpha": retrieval.event_score_log_bonus_alpha,
        "event_score_log_bonus_cap": retrieval.event_score_log_bonus_cap,
    }


def aa_forecast_encoding_export_public_dict(
    encoding_export: AAForecastEncodingExportConfig,
) -> dict[str, Any]:
    return {"enabled": encoding_export.enabled}


def aa_forecast_plugin_tuning_public_dict(cfg: AAForecastPluginConfig) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "backbone": cfg.model,
        "thresh": cfg.thresh,
        "star_hist_exog_cols_resolved": list(cfg.star_hist_exog_cols_resolved),
        "non_star_hist_exog_cols_resolved": list(cfg.non_star_hist_exog_cols_resolved),
        "star_anomaly_tails": {
            "upward": list(cfg.star_anomaly_tails["upward"]),
            "two_sided": list(cfg.star_anomaly_tails["two_sided"]),
        },
        "star_anomaly_tails_resolved": {
            "upward": list(cfg.star_anomaly_tails_resolved["upward"]),
            "two_sided": list(cfg.star_anomaly_tails_resolved["two_sided"]),
        },
        "star_anomaly_tail_modes_resolved": list(cfg.star_anomaly_tail_modes_resolved),
        "lowess_frac": cfg.lowess_frac,
        "lowess_delta": cfg.lowess_delta,
        "uncertainty": aa_forecast_uncertainty_public_dict(cfg.uncertainty),
        "retrieval": aa_forecast_retrieval_public_dict(cfg.retrieval),
        "encoding_export": aa_forecast_encoding_export_public_dict(
            cfg.encoding_export
        ),
    }


def aa_forecast_resolved_selected_path(
    cfg: AAForecastPluginConfig,
    stage_loaded: AAForecastStageLoadedConfig | None,
) -> str | None:
    if stage_loaded is not None:
        return str(stage_loaded.source_path)
    return cfg.config_path


def aa_forecast_plugin_state_dict(
    cfg: AAForecastPluginConfig,
    *,
    selected_config_path: str | None,
) -> dict[str, Any]:
    return {
        "enabled": cfg.enabled,
        "config_path": cfg.config_path,
        "selected_config_path": selected_config_path,
        "tune_training": cfg.tune_training,
        "model_params": dict(cfg.model_params),
        **aa_forecast_plugin_tuning_public_dict(cfg),
    }


def aa_forecast_stage_document_type(path: Path) -> Literal["yaml", "toml"]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".toml":
        return "toml"
    raise ValueError(
        f"aa_forecast config_path must be .yaml, .yml, or .toml, got {path.suffix!r} ({path})"
    )


def _normalize_model_params(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _normalize_model_name(value: Any, *, field_name: str) -> str:
    if value is None:
        return "gru"
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    model = value.strip().lower()
    if not model:
        raise ValueError(f"{field_name} must not be empty")
    if model not in SUPPORTED_AA_FORECAST_BACKBONES:
        supported = ", ".join(sorted(SUPPORTED_AA_FORECAST_BACKBONES))
        raise ValueError(f"{field_name} must be one of: {supported}")
    return model


def _validate_param_value_against_spec(
    value: Any,
    *,
    field_name: str,
    spec: dict[str, Any],
) -> None:
    spec_type = spec["type"]
    if spec_type == "categorical":
        if value not in spec["choices"]:
            raise ValueError(
                f"{field_name} must be one of: {', '.join(repr(choice) for choice in spec['choices'])}"
            )
        return
    if spec_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        if value < low or value > high:
            raise ValueError(f"{field_name} must satisfy {low} <= value <= {high}")
        if step > 1 and (value - low) % step != 0:
            raise ValueError(f"{field_name} must increase in steps of {step}")
        return
    if spec_type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be numeric")
        parsed = float(value)
        low = float(spec["low"])
        high = float(spec["high"])
        if parsed < low or parsed > high:
            raise ValueError(f"{field_name} must satisfy {low} <= value <= {high}")
        return
    raise ValueError(f"{field_name} uses unsupported schema type: {spec_type}")


def _validate_model_params_for_backbone(
    backbone: str,
    model_params: dict[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    registry = AA_FORECAST_STAGE_ONLY_PARAM_REGISTRY[backbone]
    unknown = sorted(set(model_params).difference(registry))
    if unknown:
        raise ValueError(
            f"{field_name} contains unsupported key(s) for aa_forecast.model={backbone!r}: "
            + ", ".join(unknown)
        )
    normalized = dict(model_params)
    for key, value in normalized.items():
        _validate_param_value_against_spec(
            value,
            field_name=f"{field_name}.{key}",
            spec=registry[key],
        )
    return normalized


def _coerce_name_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a list of strings")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings")
        candidate = item.strip()
        if not candidate:
            raise ValueError(f"{field_name} must not contain empty strings")
        normalized.append(candidate)
    return tuple(normalized)


def _coerce_probability(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if parsed <= 0 or parsed >= 1:
        raise ValueError(f"{field_name} must satisfy 0 < value < 1")
    return parsed


def _coerce_probability_tuple(
    value: Any,
    *,
    field_name: str,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a list of probabilities")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of probabilities")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    normalized = tuple(
        _coerce_probability(item, field_name=f"{field_name}[{idx}]")
        for idx, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _normalize_uncertainty_config(
    value: Any,
    *,
    section: str,
    unknown_keys: Any,
    coerce_bool: Any,
) -> AAForecastUncertaintyConfig:
    if value is None:
        return AAForecastUncertaintyConfig()
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=AA_FORECAST_UNCERTAINTY_KEYS, section=section)
    enabled = coerce_bool(
        payload.get("enabled"),
        field_name=f"{section}.enabled",
        default=False,
    )
    default_uncertainty = AAForecastUncertaintyConfig()
    dropout_candidates = _coerce_probability_tuple(
        payload.get("dropout_candidates"),
        field_name=f"{section}.dropout_candidates",
        default=default_uncertainty.dropout_candidates,
    )
    sample_count = _coerce_positive_int(
        payload.get("sample_count", default_uncertainty.sample_count),
        field_name=f"{section}.sample_count",
    )
    return AAForecastUncertaintyConfig(
        enabled=enabled,
        dropout_candidates=dropout_candidates,
        sample_count=sample_count,
    )


def _normalize_retrieval_config(
    value: Any,
    *,
    section: str,
    unknown_keys: Any,
    coerce_bool: Any,
    uncertainty: AAForecastUncertaintyConfig,
) -> AAForecastRetrievalConfig:
    if value is None:
        return AAForecastRetrievalConfig()
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=AA_FORECAST_RETRIEVAL_KEYS, section=section)
    enabled = coerce_bool(
        payload.get("enabled"),
        field_name=f"{section}.enabled",
        default=False,
    )
    top_k = _coerce_positive_int(
        payload.get("top_k", AAForecastRetrievalConfig().top_k),
        field_name=f"{section}.top_k",
    )
    recency_gap_steps = _coerce_non_negative_float(
        payload.get(
            "recency_gap_steps", AAForecastRetrievalConfig().recency_gap_steps
        ),
        field_name=f"{section}.recency_gap_steps",
    )
    recency_gap_steps_int = int(recency_gap_steps)
    if recency_gap_steps != recency_gap_steps_int:
        raise ValueError(f"{section}.recency_gap_steps must be an integer")
    event_score_threshold = _coerce_non_negative_float(
        payload.get(
            "event_score_threshold",
            AAForecastRetrievalConfig().event_score_threshold,
        ),
        field_name=f"{section}.event_score_threshold",
    )
    min_similarity = _coerce_non_negative_float(
        payload.get("min_similarity", AAForecastRetrievalConfig().min_similarity),
        field_name=f"{section}.min_similarity",
    )
    if min_similarity > 1:
        raise ValueError(f"{section}.min_similarity must satisfy 0 <= value <= 1")
    blend_max = _coerce_non_negative_float(
        payload.get("blend_max", AAForecastRetrievalConfig().blend_max),
        field_name=f"{section}.blend_max",
    )
    if blend_max > 1:
        raise ValueError(f"{section}.blend_max must satisfy 0 <= value <= 1")
    use_uncertainty_gate = coerce_bool(
        payload.get(
            "use_uncertainty_gate",
            AAForecastRetrievalConfig().use_uncertainty_gate,
        ),
        field_name=f"{section}.use_uncertainty_gate",
        default=AAForecastRetrievalConfig().use_uncertainty_gate,
    )
    use_shape_key = coerce_bool(
        payload.get("use_shape_key", AAForecastRetrievalConfig().use_shape_key),
        field_name=f"{section}.use_shape_key",
        default=AAForecastRetrievalConfig().use_shape_key,
    )
    use_event_key = coerce_bool(
        payload.get("use_event_key", AAForecastRetrievalConfig().use_event_key),
        field_name=f"{section}.use_event_key",
        default=AAForecastRetrievalConfig().use_event_key,
    )
    event_score_log_bonus_alpha = _coerce_non_negative_float(
        payload.get(
            "event_score_log_bonus_alpha",
            AAForecastRetrievalConfig().event_score_log_bonus_alpha,
        ),
        field_name=f"{section}.event_score_log_bonus_alpha",
    )
    event_score_log_bonus_cap = _coerce_non_negative_float(
        payload.get(
            "event_score_log_bonus_cap",
            AAForecastRetrievalConfig().event_score_log_bonus_cap,
        ),
        field_name=f"{section}.event_score_log_bonus_cap",
    )
    if enabled and not uncertainty.enabled:
        raise ValueError(
            f"{section}.enabled=true requires aa_forecast.uncertainty.enabled=true"
        )
    if enabled and not (use_shape_key or use_event_key):
        raise ValueError(
            f"{section}.enabled=true requires at least one of "
            f"{section}.use_shape_key or {section}.use_event_key to be true"
        )
    return AAForecastRetrievalConfig(
        enabled=enabled,
        top_k=top_k,
        recency_gap_steps=recency_gap_steps_int,
        event_score_threshold=event_score_threshold,
        min_similarity=min_similarity,
        blend_max=blend_max,
        use_uncertainty_gate=use_uncertainty_gate,
        use_shape_key=use_shape_key,
        use_event_key=use_event_key,
        event_score_log_bonus_alpha=event_score_log_bonus_alpha,
        event_score_log_bonus_cap=event_score_log_bonus_cap,
    )


def _normalize_encoding_export_config(
    value: Any,
    *,
    section: str,
    unknown_keys: Any,
    coerce_bool: Any,
) -> AAForecastEncodingExportConfig:
    if value is None:
        return AAForecastEncodingExportConfig()
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=AA_FORECAST_ENCODING_EXPORT_KEYS, section=section)
    return AAForecastEncodingExportConfig(
        enabled=coerce_bool(
            payload.get("enabled"),
            field_name=f"{section}.enabled",
            default=False,
        )
    )


def _normalize_star_anomaly_tails(
    value: Any,
    *,
    section: str,
    unknown_keys: Any,
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return _default_star_anomaly_tails()
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=AA_FORECAST_STAR_ANOMALY_TAIL_KEYS, section=section)
    upward = _coerce_name_tuple(
        payload.get("upward"),
        field_name=f"{section}.upward",
    )
    two_sided = _coerce_name_tuple(
        payload.get("two_sided"),
        field_name=f"{section}.two_sided",
    )
    if not upward and not two_sided:
        return _default_star_anomaly_tails()
    if len(set(upward)) != len(upward):
        raise ValueError(f"{section}.upward must not contain duplicates")
    if len(set(two_sided)) != len(two_sided):
        raise ValueError(f"{section}.two_sided must not contain duplicates")
    overlap = sorted(set(upward).intersection(two_sided))
    if overlap:
        raise ValueError(
            f"{section} groups must be disjoint: {', '.join(overlap)}"
        )
    return {
        "upward": upward,
        "two_sided": two_sided,
    }


def _normalize_canonical_fields(
    payload: dict[str, Any],
    *,
    section: str,
    unknown_keys: Any,
    coerce_bool: Any,
) -> AAForecastPluginConfig:
    if "p_value" in payload:
        raise ValueError(
            f"{section}.p_value has been removed; use {section}.thresh"
        )
    model = _normalize_model_name(
        payload.get("model"),
        field_name=f"{section}.model",
    )
    model_params = _normalize_model_params(
        payload.get("model_params"),
        field_name=f"{section}.model_params",
    )
    if "anomaly_threshold" in model_params:
        raise ValueError(
            f"{section}.model_params.anomaly_threshold has been removed; use {section}.thresh and {section}.star_anomaly_tails"
        )
    if "p_value" in model_params:
        raise ValueError(
            f"{section}.model_params.p_value has been removed; use {section}.thresh"
        )
    if "top_k" in model_params:
        raise ValueError(
            f"{section}.model_params.top_k has been removed; use {section}.thresh"
        )
    thresh_in_params = model_params.pop("thresh", None)
    if thresh_in_params is not None and "thresh" in payload:
        raise ValueError(
            f"{section}.thresh cannot be set both top-level and inside {section}.model_params"
        )
    model_params = _validate_model_params_for_backbone(
        model,
        model_params,
        field_name=f"{section}.model_params",
    )
    tune_training = coerce_bool(
        payload.get("tune_training"),
        field_name=f"{section}.tune_training",
        default=False,
    )
    if "top_k" in payload:
        raise ValueError(f"{section}.top_k has been removed; use {section}.thresh")
    thresh = _coerce_non_negative_float(
        payload.get("thresh", thresh_in_params if thresh_in_params is not None else 3.5),
        field_name=f"{section}.thresh",
    )
    star_anomaly_tails = _normalize_star_anomaly_tails(
        payload.get("star_anomaly_tails"),
        section=f"{section}.star_anomaly_tails",
        unknown_keys=unknown_keys,
    )
    lowess_frac = _coerce_probability(
        payload.get("lowess_frac", 0.6),
        field_name=f"{section}.lowess_frac",
    )
    lowess_delta = _coerce_non_negative_float(
        payload.get("lowess_delta", 0.01),
        field_name=f"{section}.lowess_delta",
    )
    uncertainty = _normalize_uncertainty_config(
        payload.get("uncertainty"),
        section=f"{section}.uncertainty",
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
    )
    retrieval = _normalize_retrieval_config(
        payload.get("retrieval"),
        section=f"{section}.retrieval",
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
        uncertainty=uncertainty,
    )
    encoding_export = _normalize_encoding_export_config(
        payload.get("encoding_export"),
        section=f"{section}.encoding_export",
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
    )
    if encoding_export.enabled and model != "informer":
        raise ValueError(
            f"{section}.encoding_export.enabled=true currently supports only aa_forecast.model='informer'"
        )
    return AAForecastPluginConfig(
        enabled=True,
        model=model,
        tune_training=tune_training,
        model_params=model_params,
        thresh=thresh,
        star_anomaly_tails=star_anomaly_tails,
        lowess_frac=lowess_frac,
        lowess_delta=lowess_delta,
        uncertainty=uncertainty,
        retrieval=retrieval,
        encoding_export=encoding_export,
    )


def normalize_aa_forecast_config(
    value: Any,
    *,
    unknown_keys: Any,
    coerce_bool: Any,
    coerce_optional_path_string: Any,
) -> AAForecastPluginConfig:
    if value is None:
        return AAForecastPluginConfig()
    if not isinstance(value, dict):
        raise ValueError("aa_forecast must be a mapping")
    payload = dict(value)
    if "p_value" in payload:
        raise ValueError("aa_forecast.p_value has been removed; use aa_forecast.thresh")
    if "top_k" in payload:
        raise ValueError("aa_forecast.top_k has been removed; use aa_forecast.thresh")
    unknown_keys(payload, allowed=AA_FORECAST_MAIN_KEYS, section="aa_forecast")
    enabled = coerce_bool(
        payload.get("enabled"),
        field_name="aa_forecast.enabled",
        default=False,
    )
    config_path = coerce_optional_path_string(
        payload.get("config_path"),
        field_name="aa_forecast.config_path",
    )
    if not enabled:
        return AAForecastPluginConfig(
            enabled=False,
            config_path=config_path,
        )
    if config_path is not None:
        inline_keys = {
            "model",
            "tune_training",
            "model_params",
            "thresh",
            "star_anomaly_tails",
            "lowess_frac",
            "lowess_delta",
            "uncertainty",
            "retrieval",
            "encoding_export",
        }
        if any(key in payload for key in inline_keys):
            raise ValueError(
                "aa_forecast.config_path cannot be combined with inline canonical fields"
            )
        return AAForecastPluginConfig(enabled=True, config_path=config_path)
    if any(
        key in payload
        for key in {
            "model",
            "tune_training",
            "model_params",
            "thresh",
            "star_anomaly_tails",
            "lowess_frac",
            "lowess_delta",
            "uncertainty",
            "retrieval",
            "encoding_export",
        }
    ):
        raise ValueError(
            "aa_forecast.enabled=true requires aa_forecast.config_path; top-level inline canonical fields are no longer supported"
        )
    raise ValueError(
        "aa_forecast.enabled=true requires aa_forecast.config_path"
    )


def normalize_linked_aa_forecast_config(
    value: Any,
    *,
    unknown_keys: Any,
    coerce_bool: Any,
) -> AAForecastPluginConfig:
    if value is None:
        raise ValueError("aa_forecast routed YAML must define a top-level aa_forecast block")
    if not isinstance(value, dict):
        raise ValueError("aa_forecast routed YAML block must be a mapping")
    payload = dict(value)
    if "p_value" in payload:
        raise ValueError("aa_forecast.p_value has been removed; use aa_forecast.thresh")
    if "top_k" in payload:
        raise ValueError("aa_forecast.top_k has been removed; use aa_forecast.thresh")
    unknown_keys(payload, allowed=AA_FORECAST_LINKED_KEYS, section="aa_forecast")
    return _normalize_canonical_fields(
        payload,
        section="aa_forecast",
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
    )


def aa_forecast_to_dict(config: AAForecastPluginConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["backbone"] = config.model
    payload["model_params"] = dict(config.model_params)
    payload["uncertainty"]["dropout_candidates"] = list(
        config.uncertainty.dropout_candidates
    )
    return payload


def resolve_aa_forecast_hist_exog(
    config: AAForecastPluginConfig,
    *,
    hist_exog_cols: tuple[str, ...],
) -> AAForecastPluginConfig:
    if not config.enabled:
        return config
    normalized_hist = tuple(column.strip() for column in hist_exog_cols if column.strip())
    if len(set(normalized_hist)) != len(normalized_hist):
        raise ValueError("dataset.hist_exog_cols must not contain duplicates")
    star_hist_exog_cols = (
        config.star_anomaly_tails["upward"] + config.star_anomaly_tails["two_sided"]
    )
    if not normalized_hist:
        if star_hist_exog_cols:
            raise ValueError(
                "aa_forecast.star_anomaly_tails cannot declare variables when dataset.hist_exog_cols is empty"
            )
        return replace(
            config,
            star_hist_exog_cols_resolved=(),
            non_star_hist_exog_cols_resolved=(),
            star_anomaly_tails_resolved=_default_star_anomaly_tails(),
            star_anomaly_tail_modes_resolved=(),
        )
    if not star_hist_exog_cols:
        raise ValueError(
            "aa_forecast.star_anomaly_tails must assign at least one STAR variable"
        )
    unknown = sorted(set(star_hist_exog_cols).difference(normalized_hist))
    if unknown:
        raise ValueError(
            "aa_forecast.star_anomaly_tails contains unknown column(s): "
            + ", ".join(unknown)
        )
    resolved_star = tuple(column for column in normalized_hist if column in star_hist_exog_cols)
    resolved_non_star = tuple(
        column for column in normalized_hist if column not in star_hist_exog_cols
    )
    if not resolved_star:
        raise ValueError(
            "aa_forecast.star_anomaly_tails must select at least one dataset.hist_exog_cols entry"
        )
    resolved_upward = tuple(
        column for column in resolved_star if column in config.star_anomaly_tails["upward"]
    )
    resolved_two_sided = tuple(
        column
        for column in resolved_star
        if column in config.star_anomaly_tails["two_sided"]
    )
    tail_modes_resolved = tuple(
        "upward"
        if column in config.star_anomaly_tails["upward"]
        else "two_sided"
        for column in resolved_star
    )
    return replace(
        config,
        star_hist_exog_cols_resolved=resolved_star,
        non_star_hist_exog_cols_resolved=resolved_non_star,
        star_anomaly_tails_resolved={
            "upward": resolved_upward,
            "two_sided": resolved_two_sided,
        },
        star_anomaly_tail_modes_resolved=tail_modes_resolved,
    )


def aa_forecast_jobs_payload(config: AAForecastPluginConfig) -> list[dict[str, Any]]:
    params = dict(config.model_params) if config.model_params else {}
    return [{"model": "AAForecast", "params": params}]


def aa_forecast_training_search_payload(
    config: AAForecastPluginConfig,
) -> dict[str, bool]:
    return {"enabled": bool(config.tune_training)}


def aa_forecast_selected_model_search_params(
    stage_loaded: AAForecastStageLoadedConfig | None,
) -> tuple[str, ...]:
    if stage_loaded is None or stage_loaded.search_space_payload is None:
        return ()
    return tuple(stage_loaded.search_space_payload["models"]["AAForecast"])


def aa_forecast_selected_training_search_params(
    stage_loaded: AAForecastStageLoadedConfig | None,
) -> tuple[str, ...]:
    if stage_loaded is None or stage_loaded.search_space_payload is None:
        return ()
    return tuple(stage_loaded.search_space_payload["training"]["per_model"]["AAForecast"])


def load_aa_forecast_stage1(
    repo_root: Path,
    *,
    source_path: Path,
    source_type: str,
    aa_forecast: AAForecastPluginConfig,
    search_space_contract: SearchSpaceContract | None,
) -> AAForecastStageLoadedConfig:
    from app_config import (
        _coerce_bool,
        _load_document,
        _resolve_relative_config_reference,
        _unknown_keys,
    )

    if aa_forecast.config_path is None:
        raise ValueError("aa_forecast enabled route requires config_path")
    stage_source_path = _resolve_relative_config_reference(
        repo_root,
        source_path,
        aa_forecast.config_path,
    )
    if not stage_source_path.exists():
        raise FileNotFoundError(
            f"aa_forecast selected route does not exist: {stage_source_path}"
        )
    stage_source_type = aa_forecast_stage_document_type(stage_source_path)
    raw_text = stage_source_path.read_text(encoding="utf-8")
    payload = _load_document(stage_source_path, stage_source_type)
    if not isinstance(payload, dict):
        raise ValueError(
            "aa_forecast config_path must resolve to a mapping with a top-level aa_forecast block"
        )
    linked = normalize_linked_aa_forecast_config(
        payload.get("aa_forecast"),
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
    )
    config = replace(
        linked,
        config_path=aa_forecast.config_path,
    )
    normalized_payload = {
        "aa_forecast": {
            **aa_forecast_to_dict(config),
            "selected_config_path": str(stage_source_path),
        }
    }
    scoped_search_space = None
    if search_space_contract is not None:
        try:
            scoped_search_space = stage_search_space_payload(
                search_space_contract.payload,
                backbone=config.model,
            )
        except ValueError as exc:
            raise ValueError(rewrite_search_space_error(str(exc))) from exc
    return AAForecastStageLoadedConfig(
        config=config,
        source_path=stage_source_path,
        source_type=stage_source_type,
        normalized_payload=normalized_payload,
        input_hash=_hash_text(raw_text),
        resolved_hash=_hash_text(json.dumps(normalized_payload, sort_keys=True)),
        search_space_path=search_space_contract.path if search_space_contract else None,
        search_space_hash=search_space_contract.sha256 if search_space_contract else None,
        search_space_payload=scoped_search_space,
    )
