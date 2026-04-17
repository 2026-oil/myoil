"""Contract tests for standalone retrieval plugin config and StagePlugin compliance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
import plugins.retrieval.runtime as retrieval_runtime
from plugins.retrieval.signatures import _build_star_extractor, compute_star_signature


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
        {"enabled": True, "trigger_quantile": 0.8},
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
            "trigger_quantile": 0.85,
            "min_similarity": 0.4,
            "blend_floor": 0.1,
            "blend_max": 0.5,
            "temperature": 0.2,
            "insample_y_included": False,
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
    assert cfg.retrieval.trigger_quantile == pytest.approx(0.85)
    assert cfg.retrieval.blend_floor == 0.1
    assert cfg.retrieval.blend_max == 0.5
    assert cfg.retrieval.insample_y_included is False


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
            {
                "enabled": True,
                "trigger_quantile": 0.8,
                "blend_floor": 0.5,
                "blend_max": 0.1,
            },
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_use_event_key_false() -> None:
    with pytest.raises(ValueError, match="use_event_key must be true"):
        normalize_retrieval_plugin_config(
            {
                "enabled": True,
                "trigger_quantile": 0.8,
                "use_event_key": False,
            },
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="posthoc_blend"):
        normalize_retrieval_plugin_config(
            {"enabled": True, "trigger_quantile": 0.8, "mode": "attention"},
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_overlapping_tails() -> None:
    with pytest.raises(ValueError, match="must not appear in both"):
        normalize_retrieval_plugin_config(
            {
                "enabled": True,
                "trigger_quantile": 0.8,
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
            "trigger_quantile": 0.75,
        },
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
        coerce_optional_path_string=_coerce_optional_path_string,
    )
    serialized = retrieval_config_to_dict(cfg)
    assert serialized["enabled"] is True
    assert serialized["star"]["thresh"] == 3.0
    assert serialized["top_k"] == 2
    assert serialized["trigger_quantile"] == pytest.approx(0.75)
    assert serialized["insample_y_included"] is True


def test_compute_star_signature_supports_exog_only_mode() -> None:
    star = _build_star_extractor(RetrievalStarConfig())
    frame = pd.DataFrame(
        {
            "target": [1.0, 1.1, 1.2, 1.3],
            "event": [0.0, 0.0, 5.0, 0.0],
        }
    )

    with_target = compute_star_signature(
        star=star,
        window_df=frame,
        target_col="target",
        hist_exog_cols=("event",),
        hist_exog_tail_modes=("upward",),
        insample_y_included=True,
    )
    exog_only = compute_star_signature(
        star=star,
        window_df=frame,
        target_col="target",
        hist_exog_cols=("event",),
        hist_exog_tail_modes=("upward",),
        insample_y_included=False,
    )

    assert exog_only["event_vector"].shape[0] < with_target["event_vector"].shape[0]
    with_target_prefix = with_target["event_vector"][
        : exog_only["event_vector"].shape[0]
    ]
    assert not np.allclose(exog_only["event_vector"], with_target_prefix)


def test_compute_star_signature_exog_only_requires_hist_exog() -> None:
    star = _build_star_extractor(RetrievalStarConfig())
    frame = pd.DataFrame({"target": [1.0, 1.1, 1.2, 1.3]})

    with pytest.raises(ValueError, match="require at least one hist exog column"):
        compute_star_signature(
            star=star,
            window_df=frame,
            target_col="target",
            hist_exog_cols=(),
            hist_exog_tail_modes=(),
            insample_y_included=False,
        )


def test_normalize_requires_trigger_quantile_when_enabled() -> None:
    with pytest.raises(ValueError, match="trigger_quantile is required"):
        normalize_retrieval_plugin_config(
            {"enabled": True, "top_k": 2},
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_normalize_rejects_unknown_event_score_threshold_key() -> None:
    with pytest.raises(ValueError, match="unsupported key"):
        normalize_retrieval_plugin_config(
            {
                "enabled": True,
                "trigger_quantile": 0.9,
                "event_score_threshold": 10.0,
            },
            unknown_keys=_unknown_keys,
            coerce_bool=_coerce_bool,
            coerce_optional_path_string=_coerce_optional_path_string,
        )


def test_effective_event_threshold_respects_trigger_quantile() -> None:
    retrieval_cfg = RetrievalConfig(
        enabled=True,
        trigger_quantile=0.9,
    )
    bank = [{"event_score": 1.0}, {"event_score": 5.0}, {"event_score": 9.0}]

    threshold = retrieval_runtime._effective_event_threshold(
        bank=bank,
        retrieval_cfg=retrieval_cfg,
    )

    assert threshold == pytest.approx(8.2)


def test_effective_event_threshold_quantile_empty_bank_returns_zero() -> None:
    retrieval_cfg = RetrievalConfig(
        enabled=True,
        trigger_quantile=0.9,
    )
    threshold = retrieval_runtime._effective_event_threshold(
        bank=[],
        retrieval_cfg=retrieval_cfg,
    )
    assert threshold == pytest.approx(0.0)


def test_retrieve_neighbors_ignores_neighbor_min_event_ratio() -> None:
    retrieval_cfg = RetrievalConfig(
        enabled=True,
        top_k=1,
        trigger_quantile=0.9,
        min_similarity=0.0,
        temperature=0.1,
        use_event_key=True,
        neighbor_min_event_ratio=10.0,
    )
    query = {
        "event_score": 5.0,
        "event_vector": np.array([1.0, 0.0], dtype=float),
    }
    bank = [
        {
            "candidate_end_ds": "anchor-like",
            "candidate_future_end_ds": "anchor-like+1",
            "event_vector": np.array([1.0, 0.0], dtype=float),
            "event_score": 2.0,
            "anchor_target_value": 1.0,
            "future_returns": np.array([0.05], dtype=float),
        }
    ]

    result = retrieval_runtime._retrieve_neighbors(
        query=query,
        bank=bank,
        retrieval_cfg=retrieval_cfg,
        effective_event_threshold=1.0,
    )

    assert result["retrieval_applied"] is True
    assert result["top_neighbors"][0]["candidate_end_ds"] == "anchor-like"


def test_post_predict_retrieval_writes_similarity_artifacts_even_without_neighbors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = RetrievalPluginConfig(
        enabled=True,
        retrieval=RetrievalConfig(
            enabled=True,
            top_k=1,
            trigger_quantile=0.5,
            min_similarity=0.0,
            use_event_key=True,
        ),
    )
    train_df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=4, freq="D"),
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    transformed_train_df = train_df.copy()
    future_df = pd.DataFrame({"ds": pd.date_range("2024-01-05", periods=1, freq="D")})
    target_predictions = pd.DataFrame({"prediction": [10.0]})

    monkeypatch.setattr(
        retrieval_runtime,
        "_build_star_extractor",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        retrieval_runtime,
        "_build_memory_bank",
        lambda **_kwargs: ([], 0),
    )
    monkeypatch.setattr(
        retrieval_runtime,
        "_build_query",
        lambda **_kwargs: {
            "event_vector": np.asarray([1.0], dtype=float),
            "event_score": 0.1,
        },
    )
    monkeypatch.setattr(
        retrieval_runtime,
        "_retrieve_neighbors",
        lambda **_kwargs: {
            "retrieval_attempted": True,
            "retrieval_applied": False,
            "skip_reason": "below_event_threshold",
            "top_neighbors": [],
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
        },
    )

    output = retrieval_runtime.post_predict_retrieval(
        plugin_cfg=cfg,
        target_predictions=target_predictions,
        train_df=train_df,
        transformed_train_df=transformed_train_df,
        future_df=future_df,
        target_col="target",
        dt_col="ds",
        hist_exog_cols=(),
        prediction_col="prediction",
        input_size=4,
        horizon=1,
        run_root=tmp_path,
    )

    retrieval_root = tmp_path / "retrieval"
    summary_files = [
        path
        for path in retrieval_root.glob("retrieval_summary_*.json")
        if not path.name.endswith("_windows.json")
    ]
    windows_jsons = list(retrieval_root.glob("retrieval_summary_*_windows.json"))
    windows_csvs = list(retrieval_root.glob("retrieval_summary_*_windows_long.csv"))
    raw_pngs = list(retrieval_root.glob("retrieval_summary_*_similarity_raw_overlay.png"))
    transformed_pngs = list(
        retrieval_root.glob("retrieval_summary_*_similarity_transformed_overlay.png")
    )
    summary_pngs = list(retrieval_root.glob("retrieval_summary_*_similarity_summary.png"))

    assert len(summary_files) == 1
    assert len(windows_jsons) == 1
    assert len(windows_csvs) == 1
    assert len(raw_pngs) == 1
    assert len(transformed_pngs) == 1
    assert len(summary_pngs) == 1
    summary_payload = json.loads(summary_files[0].read_text(encoding="utf-8"))
    windows_payload = json.loads(windows_jsons[0].read_text(encoding="utf-8"))
    assert summary_payload["skip_reason"] == "below_event_threshold"
    assert windows_payload["neighbors"] == []
    assert output["retrieval_applied"].tolist() == [False]


# ------------------------------------------------------------------
# is_enabled
# ------------------------------------------------------------------


def test_plugin_is_enabled() -> None:
    plugin = RetrievalStagePlugin()
    assert plugin.is_enabled(RetrievalPluginConfig(enabled=True)) is True
    assert plugin.is_enabled(RetrievalPluginConfig(enabled=False)) is False
    assert plugin.is_enabled(None) is False
    assert plugin.is_enabled("not_a_config") is False
