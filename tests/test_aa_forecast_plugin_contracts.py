"""Contract tests for aa_forecast plugin fail-fast and config helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.aa_forecast.config import aa_forecast_stage_document_type
from plugins.aa_forecast.plugin import AAForecastStagePlugin


def test_aa_forecast_stage_document_type_accepts_yaml_and_toml() -> None:
    assert aa_forecast_stage_document_type(Path("stage.yaml")) == "yaml"
    assert aa_forecast_stage_document_type(Path("stage.yml")) == "yaml"
    assert aa_forecast_stage_document_type(Path("stage.YML")) == "yaml"
    assert aa_forecast_stage_document_type(Path("stage.toml")) == "toml"


def test_aa_forecast_stage_document_type_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError, match=r"\.json"):
        aa_forecast_stage_document_type(Path("stage.json"))


def test_fanout_filter_payload_requires_dict() -> None:
    plugin = AAForecastStagePlugin()
    with pytest.raises(TypeError, match="dict"):
        plugin.fanout_filter_payload("not-a-dict")  # type: ignore[arg-type]


def test_fanout_stage_payload_none_when_disabled_or_missing() -> None:
    plugin = AAForecastStagePlugin()
    loaded = SimpleNamespace(normalized_payload={})
    assert plugin.fanout_stage_payload(loaded) is None
    loaded2 = SimpleNamespace(
        normalized_payload={"aa_forecast": {"enabled": False}}
    )
    assert plugin.fanout_stage_payload(loaded2) is None


def test_fanout_stage_payload_raises_when_enabled_without_stage1() -> None:
    plugin = AAForecastStagePlugin()
    loaded = SimpleNamespace(
        normalized_payload={"aa_forecast": {"enabled": True}}
    )
    with pytest.raises(ValueError, match="stage1"):
        plugin.fanout_stage_payload(loaded)


def test_fanout_stage_payload_returns_stage1_when_present() -> None:
    plugin = AAForecastStagePlugin()
    stage1 = {"source_path": "/tmp/x.yaml", "source_type": "yaml"}
    loaded = SimpleNamespace(
        normalized_payload={"aa_forecast": {"enabled": True, "stage1": stage1}}
    )
    assert plugin.fanout_stage_payload(loaded) == {"aa_forecast": {"stage1": stage1}}
