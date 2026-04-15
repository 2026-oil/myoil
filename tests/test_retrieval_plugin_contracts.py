"""Contract tests for standalone retrieval plugin config and StagePlugin compliance."""

from __future__ import annotations

from pathlib import Path

import pytest

from plugin_contracts.stage_plugin import StagePlugin
from plugins.retrieval.config import (
    RetrievalConfig,
    RetrievalPluginConfig,
    RetrievalStarConfig,
    normalize_retrieval_plugin_config,
    retrieval_config_to_dict,
)
from plugins.retrieval.plugin import RetrievalStagePlugin


REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------
# Plugin protocol conformance
# ------------------------------------------------------------------


def test_retrieval_plugin_satisfies_stage_plugin_protocol() -> None:
    plugin = RetrievalStagePlugin()
    assert isinstance(plugin, StagePlugin)


def test_retrieval_plugin_config_key() -> None:
    plugin = RetrievalStagePlugin()
    assert plugin.config_key == "retrieval"


def test_retrieval_plugin_does_not_own_top_level_jobs() -> None:
    plugin = RetrievalStagePlugin()
    assert plugin.owns_top_level_job("GRU") is False
    assert plugin.owns_top_level_job("Informer") is False
    assert plugin.owns_top_level_job("AAForecast") is False


def test_retrieval_plugin_supported_models_is_empty() -> None:
    plugin = RetrievalStagePlugin()
    assert plugin.supported_models() == set()


def test_retrieval_plugin_has_post_predict_fold() -> None:
    plugin = RetrievalStagePlugin()
    post_predict = getattr(plugin, "post_predict_fold", None)
    assert callable(post_predict)


def test_retrieval_plugin_predict_fold_raises() -> None:
    plugin = RetrievalStagePlugin()
    with pytest.raises(NotImplementedError, match="does not own top-level jobs"):
        plugin.predict_fold(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            train_df=None,  # type: ignore[arg-type]
            future_df=None,  # type: ignore[arg-type]
            run_root=None,
        )


def test_retrieval_plugin_capabilities_for_raises() -> None:
    plugin = RetrievalStagePlugin()
    with pytest.raises(ValueError, match="does not own"):
        plugin.capabilities_for("GRU")


# ------------------------------------------------------------------
# Config normalization
# ------------------------------------------------------------------


def _unknown_keys(
    payload: dict, *, allowed: set[str], section: str
) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(
            f"{section} contains unsupported key(s): {', '.join(unknown)}"
        )


def _coerce_bool(value, *, field_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _coerce_optional_path_string(value, *, field_name: str) -> str | None:
    del field_name
    if value is None:
        return None
    return str(value)


def test_normalize_disabled_returns_default() -> None:
    cfg = normalize_retrieval_plugin_config(
        None,
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
        coerce_optional_path_string=_coerce_optional_path_string,
    )
    assert isinstance(cfg, RetrievalPluginConfig)
    assert cfg.enabled is False


def test_normalize_disabled_explicit() -> None:
    cfg = normalize_retrieval_plugin_config(
        {"enabled": False},
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
        coerce_optional_path_string=_coerce_optional_path_string,
    )
    assert cfg.enabled is False


def test_normalize_enabled_minimal() -> None:
    cfg = normalize_retrieval_plugin_config(
        {"enabled": True},
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
        coerce_optional_path_string=_coerce_optional_path_string,
    )
    assert cfg.enabled is True
    assert isinstance(cfg.star, RetrievalStarConfig)
    assert isinstance(cfg.retrieval, RetrievalConfig)
    assert cfg.retrieval.enabled is True


def test_normalize_full_config() -> None:
    cfg = normalize_retrieval_plugin_config(
        {
            "enabled": True,
            "star": {
                "season_length": 8,
                "lowess_frac": 0.4,
                "lowess_delta": 0.02,
                "thresh": 4.0,
                "anomaly_tails": {
                    "upward": ["ColA", "ColB"],
                    "two_sided": [],
                },
            },
            "mode": "posthoc_blend",
            "top_k": 3,
            "recency_gap_steps": 4,
            "min_similarity": 0.4,
            "blend_floor": 0.1,
            "blend_max": 0.5,
            "temperature": 0.2,
            "use_shape_key": True,
            "use_event_key": True,
        },
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
        coerce_optional_path_string=_coerce_optional_path_string,
    )
    assert cfg.enabled is True
    assert cfg.star.season_length == 8
    assert cfg.star.thresh == 4.0
    assert cfg.star.anomaly_tails["upward"] == ("ColA", "ColB")
    assert cfg.retrieval.top_k == 3
    assert cfg.retrieval.blend_floor == 0.1
    assert cfg.retrieval.blend_max == 0.5


def test_normalize_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ValueError, match="unsupported key"):
        normalize_retrieval_plugin_config(
            {"enabled": True, "bad_key": 42},
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_unknown_star_key() -> None:
    with pytest.raises(ValueError, match="unsupported key"):
        normalize_retrieval_plugin_config(
            {"enabled": True, "star": {"bad_key": 1}},
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_blend_floor_gt_blend_max() -> None:
    with pytest.raises(ValueError, match="blend_floor must be"):
        normalize_retrieval_plugin_config(
            {"enabled": True, "blend_floor": 0.5, "blend_max": 0.1},
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_both_keys_false() -> None:
    with pytest.raises(ValueError, match="at least one"):
        normalize_retrieval_plugin_config(
            {
                "enabled": True,
                "use_shape_key": False,
                "use_event_key": False,
            },
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="posthoc_blend"):
        normalize_retrieval_plugin_config(
            {"enabled": True, "mode": "attention"},
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_overlapping_tails() -> None:
    with pytest.raises(ValueError, match="must not appear in both"):
        normalize_retrieval_plugin_config(
            {
                "enabled": True,
                "star": {
                    "anomaly_tails": {
                        "upward": ["ColX"],
                        "two_sided": ["ColX"],
                    }
                },
            },
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


# ------------------------------------------------------------------
# Serialization round-trip
# ------------------------------------------------------------------


def test_config_to_dict_round_trip() -> None:
    cfg = normalize_retrieval_plugin_config(
        {
            "enabled": True,
            "star": {"thresh": 3.0, "anomaly_tails": {"upward": ["A"]}},
            "top_k": 2,
        },
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
        coerce_optional_path_string=_coerce_optional_path_string,
    )
    serialized = retrieval_config_to_dict(cfg)
    assert serialized["enabled"] is True
    assert serialized["star"]["thresh"] == 3.0
    assert serialized["top_k"] == 2


# ------------------------------------------------------------------
# is_enabled
# ------------------------------------------------------------------


def test_plugin_is_enabled() -> None:
    plugin = RetrievalStagePlugin()
    assert plugin.is_enabled(RetrievalPluginConfig(enabled=True)) is True
    assert plugin.is_enabled(RetrievalPluginConfig(enabled=False)) is False
    assert plugin.is_enabled(None) is False
    assert plugin.is_enabled("not_a_config") is False
