"""Configuration dataclasses and YAML normalization for the standalone retrieval plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RETRIEVAL_PLUGIN_TOP_KEYS = {
    "enabled",
    "config_path",
}

RETRIEVAL_PLUGIN_DETAIL_KEYS = {
    "enabled",
    "star",
    "mode",
    "similarity",
    "temperature",
    "memory_value_mode",
    "top_k",
    "recency_gap_steps",
    "event_score_threshold",
    "trigger_quantile",
    "neighbor_min_event_ratio",
    "min_similarity",
    "blend_floor",
    "blend_max",
    "use_uncertainty_gate",
    "use_shape_key",
    "use_event_key",
    "event_score_log_bonus_alpha",
    "event_score_log_bonus_cap",
}

RETRIEVAL_PLUGIN_MAIN_KEYS = RETRIEVAL_PLUGIN_TOP_KEYS | RETRIEVAL_PLUGIN_DETAIL_KEYS

RETRIEVAL_STAR_KEYS = {
    "season_length",
    "lowess_frac",
    "lowess_delta",
    "thresh",
    "anomaly_tails",
}

RETRIEVAL_ANOMALY_TAIL_KEYS = {"upward", "two_sided"}


def _default_anomaly_tails() -> dict[str, tuple[str, ...]]:
    return {"upward": (), "two_sided": ()}


@dataclass(frozen=True)
class RetrievalStarConfig:
    season_length: int = 4
    lowess_frac: float = 0.35
    lowess_delta: float = 0.01
    thresh: float = 3.5
    anomaly_tails: dict[str, tuple[str, ...]] = field(
        default_factory=_default_anomaly_tails
    )


@dataclass(frozen=True)
class RetrievalConfig:
    enabled: bool = False
    top_k: int = 5
    recency_gap_steps: int = 8
    event_score_threshold: float = 1.0
    trigger_quantile: float | None = None
    neighbor_min_event_ratio: float = 0.0
    min_similarity: float = 0.55
    blend_max: float = 0.25
    use_uncertainty_gate: bool = False
    mode: str = "posthoc_blend"
    similarity: str = "cosine"
    temperature: float = 0.10
    memory_value_mode: str = "future_return"
    use_shape_key: bool = True
    use_event_key: bool = True
    blend_floor: float = 0.0
    event_score_log_bonus_alpha: float = 0.0
    event_score_log_bonus_cap: float = 0.0


@dataclass(frozen=True)
class RetrievalPluginConfig:
    enabled: bool = False
    config_path: str | None = None
    star: RetrievalStarConfig = field(default_factory=RetrievalStarConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# ---------------------------------------------------------------------------
# Coercion helpers (mirror app_config patterns)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# STAR config normalization
# ---------------------------------------------------------------------------


def _normalize_star_config(
    value: Any,
    *,
    section: str,
    unknown_keys: Any,
) -> RetrievalStarConfig:
    if value is None:
        return RetrievalStarConfig()
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=RETRIEVAL_STAR_KEYS, section=section)

    defaults = RetrievalStarConfig()
    season_length = _coerce_positive_int(
        payload.get("season_length", defaults.season_length),
        field_name=f"{section}.season_length",
    )
    lowess_frac = _coerce_non_negative_float(
        payload.get("lowess_frac", defaults.lowess_frac),
        field_name=f"{section}.lowess_frac",
    )
    if lowess_frac <= 0 or lowess_frac > 1:
        raise ValueError(f"{section}.lowess_frac must satisfy 0 < value <= 1")
    lowess_delta = _coerce_non_negative_float(
        payload.get("lowess_delta", defaults.lowess_delta),
        field_name=f"{section}.lowess_delta",
    )
    thresh = _coerce_non_negative_float(
        payload.get("thresh", defaults.thresh),
        field_name=f"{section}.thresh",
    )

    anomaly_tails = _normalize_anomaly_tails(
        payload.get("anomaly_tails"),
        section=f"{section}.anomaly_tails",
        unknown_keys=unknown_keys,
    )
    return RetrievalStarConfig(
        season_length=season_length,
        lowess_frac=lowess_frac,
        lowess_delta=lowess_delta,
        thresh=thresh,
        anomaly_tails=anomaly_tails,
    )


def _normalize_anomaly_tails(
    value: Any,
    *,
    section: str,
    unknown_keys: Any,
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return _default_anomaly_tails()
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a mapping")
    payload = dict(value)
    unknown_keys(payload, allowed=RETRIEVAL_ANOMALY_TAIL_KEYS, section=section)
    upward = _coerce_name_tuple(
        payload.get("upward"), field_name=f"{section}.upward"
    )
    two_sided = _coerce_name_tuple(
        payload.get("two_sided"), field_name=f"{section}.two_sided"
    )
    overlap = set(upward) & set(two_sided)
    if overlap:
        raise ValueError(
            f"{section}: columns must not appear in both upward and two_sided: "
            + ", ".join(sorted(overlap))
        )
    return {"upward": upward, "two_sided": two_sided}


# ---------------------------------------------------------------------------
# Retrieval config normalization
# ---------------------------------------------------------------------------


def _normalize_retrieval_config(
    payload: dict[str, Any],
    *,
    section: str,
    coerce_bool: Any,
) -> RetrievalConfig:
    defaults = RetrievalConfig()

    top_k = _coerce_positive_int(
        payload.get("top_k", defaults.top_k),
        field_name=f"{section}.top_k",
    )
    recency_gap_steps = _coerce_non_negative_float(
        payload.get("recency_gap_steps", defaults.recency_gap_steps),
        field_name=f"{section}.recency_gap_steps",
    )
    recency_gap_steps_int = int(recency_gap_steps)
    if recency_gap_steps != recency_gap_steps_int:
        raise ValueError(f"{section}.recency_gap_steps must be an integer")

    event_score_threshold = _coerce_non_negative_float(
        payload.get("event_score_threshold", defaults.event_score_threshold),
        field_name=f"{section}.event_score_threshold",
    )
    trigger_quantile_raw = payload.get("trigger_quantile", defaults.trigger_quantile)
    if trigger_quantile_raw is None:
        trigger_quantile = None
    else:
        trigger_quantile = float(trigger_quantile_raw)
        if not (0.0 < trigger_quantile < 1.0):
            raise ValueError(
                f"{section}.trigger_quantile must satisfy 0 < value < 1"
            )

    neighbor_min_event_ratio = _coerce_non_negative_float(
        payload.get("neighbor_min_event_ratio", defaults.neighbor_min_event_ratio),
        field_name=f"{section}.neighbor_min_event_ratio",
    )
    min_similarity = _coerce_non_negative_float(
        payload.get("min_similarity", defaults.min_similarity),
        field_name=f"{section}.min_similarity",
    )
    if min_similarity > 1:
        raise ValueError(f"{section}.min_similarity must satisfy 0 <= value <= 1")

    blend_floor = _coerce_non_negative_float(
        payload.get("blend_floor", defaults.blend_floor),
        field_name=f"{section}.blend_floor",
    )
    if blend_floor > 1:
        raise ValueError(f"{section}.blend_floor must satisfy 0 <= value <= 1")
    blend_max = _coerce_non_negative_float(
        payload.get("blend_max", defaults.blend_max),
        field_name=f"{section}.blend_max",
    )
    if blend_max > 1:
        raise ValueError(f"{section}.blend_max must satisfy 0 <= value <= 1")
    if blend_floor > blend_max:
        raise ValueError(f"{section}.blend_floor must be <= {section}.blend_max")

    mode = str(payload.get("mode", defaults.mode)).strip().lower()
    if mode != "posthoc_blend":
        raise ValueError(f"{section}.mode currently only supports 'posthoc_blend'")

    similarity = str(payload.get("similarity", defaults.similarity)).strip().lower()
    if similarity != "cosine":
        raise ValueError(f"{section}.similarity currently only supports 'cosine'")

    temperature = _coerce_non_negative_float(
        payload.get("temperature", defaults.temperature),
        field_name=f"{section}.temperature",
    )
    if temperature <= 0:
        raise ValueError(f"{section}.temperature must be > 0")

    memory_value_mode = (
        str(payload.get("memory_value_mode", defaults.memory_value_mode))
        .strip()
        .lower()
    )
    if memory_value_mode not in {"future_return", "future_level"}:
        raise ValueError(
            f"{section}.memory_value_mode must be one of: 'future_return', 'future_level'"
        )

    use_uncertainty_gate = coerce_bool(
        payload.get("use_uncertainty_gate", defaults.use_uncertainty_gate),
        field_name=f"{section}.use_uncertainty_gate",
        default=defaults.use_uncertainty_gate,
    )
    use_shape_key = coerce_bool(
        payload.get("use_shape_key", defaults.use_shape_key),
        field_name=f"{section}.use_shape_key",
        default=defaults.use_shape_key,
    )
    use_event_key = coerce_bool(
        payload.get("use_event_key", defaults.use_event_key),
        field_name=f"{section}.use_event_key",
        default=defaults.use_event_key,
    )
    if not (use_shape_key or use_event_key):
        raise ValueError(
            f"{section}: at least one of use_shape_key or use_event_key must be true"
        )

    event_score_log_bonus_alpha = _coerce_non_negative_float(
        payload.get("event_score_log_bonus_alpha", defaults.event_score_log_bonus_alpha),
        field_name=f"{section}.event_score_log_bonus_alpha",
    )
    event_score_log_bonus_cap = _coerce_non_negative_float(
        payload.get("event_score_log_bonus_cap", defaults.event_score_log_bonus_cap),
        field_name=f"{section}.event_score_log_bonus_cap",
    )

    return RetrievalConfig(
        enabled=True,
        top_k=top_k,
        recency_gap_steps=recency_gap_steps_int,
        event_score_threshold=event_score_threshold,
        trigger_quantile=trigger_quantile,
        neighbor_min_event_ratio=neighbor_min_event_ratio,
        min_similarity=min_similarity,
        blend_max=blend_max,
        blend_floor=blend_floor,
        mode=mode,
        similarity=similarity,
        temperature=temperature,
        memory_value_mode=memory_value_mode,
        use_uncertainty_gate=use_uncertainty_gate,
        use_shape_key=use_shape_key,
        use_event_key=use_event_key,
        event_score_log_bonus_alpha=event_score_log_bonus_alpha,
        event_score_log_bonus_cap=event_score_log_bonus_cap,
    )


# ---------------------------------------------------------------------------
# Top-level plugin config normalization
# ---------------------------------------------------------------------------


def normalize_retrieval_plugin_config(
    payload: Any,
    *,
    unknown_keys: Any,
    coerce_bool: Any,
    coerce_optional_path_string: Any,
) -> RetrievalPluginConfig:
    """Parse and validate the ``retrieval:`` YAML block.

    Two usage modes:

    1. **Link mode** (recommended)::

           retrieval:
             enabled: true
             config_path: yaml/plugins/retrieval/baseline_retrieval.yaml

       Only ``enabled`` and ``config_path`` in the experiment YAML; full
       settings are loaded from the external file during ``validate_route``.

    2. **Inline mode** — all settings written directly in the experiment YAML.
    """
    if payload is None:
        return RetrievalPluginConfig()
    if not isinstance(payload, dict):
        raise ValueError("retrieval must be a mapping")

    raw = dict(payload)

    enabled = coerce_bool(
        raw.get("enabled"),
        field_name="retrieval.enabled",
        default=False,
    )
    if not enabled:
        return RetrievalPluginConfig(enabled=False)

    config_path = coerce_optional_path_string(
        raw.get("config_path"),
        field_name="retrieval.config_path",
    )

    if config_path is not None:
        unknown_keys(raw, allowed=RETRIEVAL_PLUGIN_TOP_KEYS, section="retrieval")
        return RetrievalPluginConfig(enabled=True, config_path=config_path)

    unknown_keys(raw, allowed=RETRIEVAL_PLUGIN_DETAIL_KEYS, section="retrieval")

    star_cfg = _normalize_star_config(
        raw.get("star"),
        section="retrieval.star",
        unknown_keys=unknown_keys,
    )

    retrieval_cfg = _normalize_retrieval_config(
        raw,
        section="retrieval",
        coerce_bool=coerce_bool,
    )

    return RetrievalPluginConfig(
        enabled=True,
        star=star_cfg,
        retrieval=retrieval_cfg,
    )


def normalize_retrieval_detail_payload(
    payload: Any,
    *,
    unknown_keys: Any,
    coerce_bool: Any,
) -> RetrievalPluginConfig:
    """Parse the external plugin YAML (the file pointed to by ``config_path``).

    The external file is expected to have a top-level ``retrieval:`` key
    containing all detail settings (star, mode, top_k, etc.).
    """
    if not isinstance(payload, dict):
        raise ValueError("retrieval plugin config file must be a mapping")
    retrieval_section = payload.get("retrieval")
    if retrieval_section is None:
        raise ValueError(
            "retrieval plugin config file must contain a 'retrieval' key"
        )
    if not isinstance(retrieval_section, dict):
        raise ValueError("retrieval plugin config 'retrieval' must be a mapping")
    raw = dict(retrieval_section)
    unknown_keys(raw, allowed=RETRIEVAL_PLUGIN_DETAIL_KEYS, section="retrieval")

    star_cfg = _normalize_star_config(
        raw.get("star"),
        section="retrieval.star",
        unknown_keys=unknown_keys,
    )

    retrieval_cfg = _normalize_retrieval_config(
        raw,
        section="retrieval",
        coerce_bool=coerce_bool,
    )

    return RetrievalPluginConfig(
        enabled=True,
        star=star_cfg,
        retrieval=retrieval_cfg,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def retrieval_config_to_dict(cfg: RetrievalPluginConfig) -> dict[str, Any]:
    if not cfg.enabled:
        return {"enabled": False}
    if cfg.config_path is not None and cfg.retrieval == RetrievalConfig():
        return {"enabled": True, "config_path": cfg.config_path}
    r = cfg.retrieval
    result: dict[str, Any] = {"enabled": True}
    if cfg.config_path is not None:
        result["config_path"] = cfg.config_path
    result.update({
        "star": {
            "season_length": cfg.star.season_length,
            "lowess_frac": cfg.star.lowess_frac,
            "lowess_delta": cfg.star.lowess_delta,
            "thresh": cfg.star.thresh,
            "anomaly_tails": {
                "upward": list(cfg.star.anomaly_tails.get("upward", ())),
                "two_sided": list(cfg.star.anomaly_tails.get("two_sided", ())),
            },
        },
        "mode": r.mode,
        "similarity": r.similarity,
        "temperature": r.temperature,
        "memory_value_mode": r.memory_value_mode,
        "top_k": r.top_k,
        "recency_gap_steps": r.recency_gap_steps,
        "event_score_threshold": r.event_score_threshold,
        "trigger_quantile": r.trigger_quantile,
        "neighbor_min_event_ratio": r.neighbor_min_event_ratio,
        "min_similarity": r.min_similarity,
        "blend_floor": r.blend_floor,
        "blend_max": r.blend_max,
        "use_uncertainty_gate": r.use_uncertainty_gate,
        "use_shape_key": r.use_shape_key,
        "use_event_key": r.use_event_key,
        "event_score_log_bonus_alpha": r.event_score_log_bonus_alpha,
        "event_score_log_bonus_cap": r.event_score_log_bonus_cap,
    })
    return result
