from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tuning.search_space import SearchSpaceContract


AA_FORECAST_MAIN_KEYS = {
    "enabled",
    "config_path",
    "mode",
    "tune_training",
    "model_params",
    "star_hist_exog_cols",
    "lowess_frac",
    "lowess_delta",
    "uncertainty",
    "compatibility_mode",
    "compatibility_source_path",
}
AA_FORECAST_LINKED_KEYS = {
    "mode",
    "tune_training",
    "model_params",
    "star_hist_exog_cols",
    "lowess_frac",
    "lowess_delta",
    "uncertainty",
}
AA_FORECAST_UNCERTAINTY_KEYS = {"enabled", "dropout_candidates", "sample_count"}


@dataclass(frozen=True)
class AAForecastUncertaintyConfig:
    enabled: bool = False
    dropout_candidates: tuple[float, ...] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )
    sample_count: int = 5


@dataclass(frozen=True)
class AAForecastPluginConfig:
    enabled: bool = False
    config_path: str | None = None
    mode: Literal["fixed", "learned_auto"] = "learned_auto"
    tune_training: bool = False
    model_params: dict[str, Any] = field(default_factory=dict)
    star_hist_exog_cols: tuple[str, ...] = field(default_factory=tuple)
    star_hist_exog_cols_resolved: tuple[str, ...] = field(default_factory=tuple)
    non_star_hist_exog_cols_resolved: tuple[str, ...] = field(default_factory=tuple)
    lowess_frac: float = 0.6
    lowess_delta: float = 0.01
    uncertainty: AAForecastUncertaintyConfig = field(
        default_factory=AAForecastUncertaintyConfig
    )
    compatibility_mode: str | None = None
    compatibility_source_path: str | None = None


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


def aa_forecast_plugin_tuning_public_dict(cfg: AAForecastPluginConfig) -> dict[str, Any]:
    return {
        "star_hist_exog_cols": list(cfg.star_hist_exog_cols),
        "star_hist_exog_cols_resolved": list(cfg.star_hist_exog_cols_resolved),
        "non_star_hist_exog_cols_resolved": list(cfg.non_star_hist_exog_cols_resolved),
        "lowess_frac": cfg.lowess_frac,
        "lowess_delta": cfg.lowess_delta,
        "uncertainty": aa_forecast_uncertainty_public_dict(cfg.uncertainty),
        "compatibility_mode": cfg.compatibility_mode,
        "compatibility_source_path": cfg.compatibility_source_path,
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
        "mode": cfg.mode,
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


def _coerce_optional_name_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    return normalized or None


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
    raw_candidates = payload.get("dropout_candidates")
    if raw_candidates is None:
        dropout_candidates = AAForecastUncertaintyConfig().dropout_candidates
    else:
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"{section}.dropout_candidates must be a non-empty list")
        normalized = tuple(
            _coerce_probability(
                candidate,
                field_name=f"{section}.dropout_candidates",
            )
            for candidate in raw_candidates
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{section}.dropout_candidates must not contain duplicates")
        dropout_candidates = normalized
    sample_count = _coerce_positive_int(
        payload.get("sample_count", AAForecastUncertaintyConfig().sample_count),
        field_name=f"{section}.sample_count",
    )
    return AAForecastUncertaintyConfig(
        enabled=enabled,
        dropout_candidates=dropout_candidates,
        sample_count=sample_count,
    )


def _normalize_canonical_fields(
    payload: dict[str, Any],
    *,
    section: str,
    unknown_keys: Any,
    coerce_bool: Any,
) -> AAForecastPluginConfig:
    mode = str(payload.get("mode", "learned_auto")).strip().lower()
    if mode not in {"fixed", "learned_auto"}:
        raise ValueError(f"{section}.mode must be one of: fixed, learned_auto")
    model_params = _normalize_model_params(
        payload.get("model_params"),
        field_name=f"{section}.model_params",
    )
    if mode == "fixed" and not model_params:
        raise ValueError(f"{section}.mode=fixed requires {section}.model_params")
    if mode == "learned_auto" and model_params:
        raise ValueError(
            f"{section}.mode=learned_auto must not set {section}.model_params"
        )
    tune_training = coerce_bool(
        payload.get("tune_training"),
        field_name=f"{section}.tune_training",
        default=False,
    )
    star_hist_exog_cols = _coerce_name_tuple(
        payload.get("star_hist_exog_cols"),
        field_name=f"{section}.star_hist_exog_cols",
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
    return AAForecastPluginConfig(
        enabled=True,
        mode=mode,  # type: ignore[arg-type]
        tune_training=tune_training,
        model_params=model_params,
        star_hist_exog_cols=star_hist_exog_cols,
        lowess_frac=lowess_frac,
        lowess_delta=lowess_delta,
        uncertainty=uncertainty,
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
    compatibility_mode = _coerce_optional_name_string(
        payload.get("compatibility_mode"),
        field_name="aa_forecast.compatibility_mode",
    )
    compatibility_source_path = _coerce_optional_name_string(
        payload.get("compatibility_source_path"),
        field_name="aa_forecast.compatibility_source_path",
    )
    if not enabled:
        return AAForecastPluginConfig(
            enabled=False,
            config_path=config_path,
            compatibility_mode=compatibility_mode,
            compatibility_source_path=compatibility_source_path,
        )
    canonical = _normalize_canonical_fields(
        payload,
        section="aa_forecast",
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
    )
    if config_path is not None:
        inline_keys = {
            "mode",
            "tune_training",
            "model_params",
            "star_hist_exog_cols",
            "lowess_frac",
            "lowess_delta",
            "uncertainty",
        }
        if any(key in payload for key in inline_keys):
            raise ValueError(
                "aa_forecast.config_path cannot be combined with inline canonical fields"
            )
        return replace(
            AAForecastPluginConfig(enabled=True, config_path=config_path),
            compatibility_mode=compatibility_mode,
            compatibility_source_path=compatibility_source_path,
        )
    return replace(
        canonical,
        config_path=None,
        compatibility_mode=compatibility_mode,
        compatibility_source_path=compatibility_source_path,
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
    unknown_keys(payload, allowed=AA_FORECAST_LINKED_KEYS, section="aa_forecast")
    return _normalize_canonical_fields(
        payload,
        section="aa_forecast",
        unknown_keys=unknown_keys,
        coerce_bool=coerce_bool,
    )


def aa_forecast_to_dict(config: AAForecastPluginConfig) -> dict[str, Any]:
    payload = asdict(config)
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
    if not normalized_hist:
        raise ValueError(
            "AAForecast requires dataset.hist_exog_cols to be non-empty"
        )
    if len(set(normalized_hist)) != len(normalized_hist):
        raise ValueError("dataset.hist_exog_cols must not contain duplicates")
    if not config.star_hist_exog_cols:
        raise ValueError("aa_forecast.star_hist_exog_cols is required and must be non-empty")
    if len(set(config.star_hist_exog_cols)) != len(config.star_hist_exog_cols):
        raise ValueError("aa_forecast.star_hist_exog_cols must not contain duplicates")
    unknown = sorted(set(config.star_hist_exog_cols).difference(normalized_hist))
    if unknown:
        raise ValueError(
            "aa_forecast.star_hist_exog_cols contains unknown column(s): "
            + ", ".join(unknown)
        )
    resolved_star = tuple(column for column in normalized_hist if column in config.star_hist_exog_cols)
    resolved_non_star = tuple(
        column for column in normalized_hist if column not in config.star_hist_exog_cols
    )
    if not resolved_star:
        raise ValueError("aa_forecast.star_hist_exog_cols must select at least one dataset.hist_exog_cols entry")
    return replace(
        config,
        star_hist_exog_cols_resolved=resolved_star,
        non_star_hist_exog_cols_resolved=resolved_non_star,
    )


def aa_forecast_jobs_payload(config: AAForecastPluginConfig) -> list[dict[str, Any]]:
    params = dict(config.model_params) if config.mode == "fixed" else {}
    return [{"model": "AAForecast", "params": params}]


def aa_forecast_training_search_payload(
    config: AAForecastPluginConfig,
) -> dict[str, bool]:
    return {"enabled": bool(config.tune_training)}


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
        normalized_payload = {"aa_forecast": aa_forecast_to_dict(aa_forecast)}
        inline_text = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False)
        return AAForecastStageLoadedConfig(
            config=aa_forecast,
            source_path=source_path,
            source_type=source_type,
            normalized_payload=normalized_payload,
            input_hash=_hash_text(inline_text),
            resolved_hash=_hash_text(inline_text),
            search_space_path=search_space_contract.path if search_space_contract else None,
            search_space_hash=search_space_contract.sha256 if search_space_contract else None,
            search_space_payload=search_space_contract.payload if search_space_contract else None,
        )
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
        compatibility_mode=aa_forecast.compatibility_mode,
        compatibility_source_path=aa_forecast.compatibility_source_path,
    )
    normalized_payload = {
        "aa_forecast": {
            **aa_forecast_to_dict(config),
            "selected_config_path": str(stage_source_path),
        }
    }
    return AAForecastStageLoadedConfig(
        config=config,
        source_path=stage_source_path,
        source_type=stage_source_type,
        normalized_payload=normalized_payload,
        input_hash=_hash_text(raw_text),
        resolved_hash=_hash_text(json.dumps(normalized_payload, sort_keys=True)),
        search_space_path=search_space_contract.path if search_space_contract else None,
        search_space_hash=search_space_contract.sha256 if search_space_contract else None,
        search_space_payload=search_space_contract.payload if search_space_contract else None,
    )
