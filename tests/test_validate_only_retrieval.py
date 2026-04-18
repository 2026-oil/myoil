"""Validate-only smoke tests for the standalone retrieval plugin."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from app_config import load_app_config
from plugins.retrieval.config import RetrievalPluginConfig
from plugins.retrieval.plugin import RetrievalStagePlugin


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = Path("tests/fixtures/retrieval_runtime_smoke.yaml")
SMOKE_LINKED_CONFIG = Path("tests/fixtures/retrieval_runtime_smoke_linked.yaml")


def test_load_retrieval_smoke_config() -> None:
    """Validate-only: the retrieval smoke config loads without errors."""
    loaded = load_app_config(REPO_ROOT, config_path=SMOKE_CONFIG)
    cfg = loaded.config
    assert isinstance(cfg.stage_plugin_config, RetrievalPluginConfig)
    assert cfg.stage_plugin_config.enabled is True
    assert cfg.stage_plugin_config.retrieval.top_k == 1
    assert cfg.stage_plugin_config.star.thresh == 3.0


def test_load_retrieval_config_serialized_payload() -> None:
    """The normalized payload should contain the 'retrieval' key."""
    loaded = load_app_config(REPO_ROOT, config_path=SMOKE_CONFIG)
    payload = loaded.normalized_payload
    assert "retrieval" in payload
    assert payload["retrieval"]["enabled"] is True


def test_retrieval_plugin_discovered_in_registry() -> None:
    """The retrieval plugin must be auto-discovered by the stage registry."""
    from plugin_contracts.stage_registry import get_stage_plugin

    plugin = get_stage_plugin("retrieval")
    assert plugin is not None
    assert isinstance(plugin, RetrievalStagePlugin)


def test_load_linked_config_path_resolves_detail() -> None:
    """config_path mode: detail settings are loaded from external YAML."""
    loaded = load_app_config(REPO_ROOT, config_path=SMOKE_LINKED_CONFIG)
    cfg = loaded.config
    assert isinstance(cfg.stage_plugin_config, RetrievalPluginConfig)
    assert cfg.stage_plugin_config.enabled is True
    assert cfg.stage_plugin_config.retrieval.top_k == 1
    assert cfg.stage_plugin_config.star.thresh == 3.0
    assert cfg.stage_plugin_config.retrieval.blend_floor == 0.2


def test_load_baseline_without_stage_plugin_uses_shared_scaler() -> None:
    """Baseline remains retrieval-free and inherits the shared robust scaler."""
    baseline_path = Path("yaml/experiment/feature_set_aaforecast/baseline.yaml")
    if not (REPO_ROOT / baseline_path).exists():
        pytest.skip("baseline.yaml not found")
    loaded = load_app_config(REPO_ROOT, config_path=baseline_path)
    cfg = loaded.config
    assert isinstance(cfg.stage_plugin_config, RetrievalPluginConfig)
    assert cfg.stage_plugin_config.enabled is False
    assert cfg.training.scaler_type == "robust"
    assert "Idx_DxyUSD" in cfg.dataset.hist_exog_cols


def test_load_baseline_ret_linked_config_path() -> None:
    baseline_ret_path = Path("yaml/experiment/feature_set_aaforecast/baseline-ret.yaml")
    if not (REPO_ROOT / baseline_ret_path).exists():
        pytest.skip("baseline-ret.yaml not found")
    loaded = load_app_config(REPO_ROOT, config_path=baseline_ret_path)
    cfg = loaded.config
    assert isinstance(cfg.stage_plugin_config, RetrievalPluginConfig)
    assert cfg.stage_plugin_config.enabled is True
    assert (
        cfg.stage_plugin_config.config_path
        == "yaml/plugins/retrieval/baseline_retrieval.yaml"
    )
    assert cfg.stage_plugin_config.retrieval.top_k == 1
    assert cfg.stage_plugin_config.retrieval.trigger_quantile == pytest.approx(0.8)
    assert cfg.stage_plugin_config.retrieval.min_similarity == pytest.approx(0.7)
    assert cfg.stage_plugin_config.retrieval.blend_floor == pytest.approx(0.0)
    assert cfg.stage_plugin_config.retrieval.blend_max == pytest.approx(1.0)
    assert cfg.stage_plugin_config.retrieval.use_uncertainty_gate is True
    assert cfg.stage_plugin_config.retrieval.insample_y_included is False
    assert cfg.stage_plugin_config.star.anomaly_tails["upward"] == ("GPRD",)
    assert cfg.stage_plugin_config.star.anomaly_tails["two_sided"] == ()


def test_repo_var_yaml_supplies_retrieval_anomaly_tails() -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=Path("yaml/experiment/feature_set_aaforecast_brent/baseline-ret.yaml"),
    )
    cfg = loaded.config
    assert isinstance(cfg.stage_plugin_config, RetrievalPluginConfig)
    assert cfg.stage_plugin_config.star.anomaly_tails == {
        "upward": ("GPRD_THREAT",),
        "two_sided": (),
    }


def test_repo_var_yaml_duplicate_retrieval_tail_fails_fast() -> None:
    plugin_dir = REPO_ROOT / "yaml/plugins/retrieval"
    config_path = plugin_dir / ".tmp-duplicate-var-retrieval.yaml"
    payload = yaml.safe_load(
        (REPO_ROOT / "yaml/plugins/retrieval/baseline_retrieval.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["retrieval"]["star"]["anomaly_tails"] = {"upward": ["GPRD"], "two_sided": []}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    try:
        with pytest.raises(
            ValueError, match=r"config repeats shared var path\(s\): retrieval.star.anomaly_tails"
        ):
            load_app_config(REPO_ROOT, config_path=config_path.relative_to(REPO_ROOT))
    finally:
        config_path.unlink(missing_ok=True)


def test_baseline_hist_exog_columns_exist_in_df() -> None:
    baseline_path = Path("yaml/experiment/feature_set_aaforecast/baseline.yaml")
    if not (REPO_ROOT / baseline_path).exists():
        pytest.skip("baseline.yaml not found")
    loaded = load_app_config(REPO_ROOT, config_path=baseline_path)
    dataset_path = REPO_ROOT / loaded.config.dataset.path
    columns = set(pd.read_csv(dataset_path, nrows=0).columns.tolist())
    missing = [
        column for column in loaded.config.dataset.hist_exog_cols if column not in columns
    ]
    assert missing == []
