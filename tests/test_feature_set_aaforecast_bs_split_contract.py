from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

YES_EXPERIMENT_ROOT = REPO_ROOT / "yaml/experiment/feature_set_aaforecast_YES_BS"
NO_EXPERIMENT_ROOT = REPO_ROOT / "yaml/experiment/feature_set_aaforecast_NO_BS"

YES_PLUGIN_ROOT = REPO_ROOT / "yaml/plugins/feature_set_aaforecast_YES_BS"
NO_PLUGIN_ROOT = REPO_ROOT / "yaml/plugins/feature_set_aaforecast_NO_BS"

BS_FEATURES = (
    "BS_Core_Index_A",
    "BS_Core_Index_B",
    "BS_Core_Index_C",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_feature_set_aaforecast_no_bs_experiment_has_no_bs_features() -> None:
    paths = sorted(NO_EXPERIMENT_ROOT.glob("*.yaml"))
    assert paths, f"Missing NO_BS configs under: {NO_EXPERIMENT_ROOT}"

    for path in paths:
        text = _read_text(path)
        assert "BS_Core_Index_" not in text, f"NO_BS config still references BS: {path}"


def test_feature_set_aaforecast_yes_bs_experiment_references_yes_plugin_paths() -> None:
    for name in (
        "aaforecast-informer.yaml",
        "aaforecast-informer-ret.yaml",
        "aaforecast-gru.yaml",
        "aaforecast-gru-ret.yaml",
        "aaforecast-patchtst.yaml",
        "aaforecast-patchtst-ret.yaml",
        "aaforecast-timexer.yaml",
        "aaforecast-timexer-ret.yaml",
    ):
        text = _read_text(YES_EXPERIMENT_ROOT / name)
        assert "config_path: yaml/plugins/feature_set_aaforecast_YES_BS/" in text


def test_feature_set_aaforecast_no_bs_experiment_references_no_plugin_paths() -> None:
    for name in (
        "aaforecast-informer.yaml",
        "aaforecast-informer-ret.yaml",
        "aaforecast-gru.yaml",
        "aaforecast-gru-ret.yaml",
        "aaforecast-patchtst.yaml",
        "aaforecast-patchtst-ret.yaml",
        "aaforecast-timexer.yaml",
        "aaforecast-timexer-ret.yaml",
    ):
        text = _read_text(NO_EXPERIMENT_ROOT / name)
        assert "config_path: yaml/plugins/feature_set_aaforecast_NO_BS/" in text
        assert "BS_Core_Index_" not in text


def test_feature_set_aaforecast_no_bs_plugins_have_no_bs_features() -> None:
    paths = sorted(NO_PLUGIN_ROOT.glob("aa_forecast_*.yaml"))
    assert paths, f"Missing NO_BS plugin yamls under: {NO_PLUGIN_ROOT}"

    for path in paths:
        text = _read_text(path)
        assert "BS_Core_Index_" not in text, f"NO_BS plugin still references BS: {path}"


def test_feature_set_aaforecast_yes_bs_plugins_include_all_bs_features() -> None:
    paths = sorted(YES_PLUGIN_ROOT.glob("aa_forecast_*.yaml"))
    assert paths, f"Missing YES_BS plugin yamls under: {YES_PLUGIN_ROOT}"

    for path in paths:
        text = _read_text(path)
        for feature in BS_FEATURES:
            assert feature in text, f"YES_BS plugin missing {feature}: {path}"

