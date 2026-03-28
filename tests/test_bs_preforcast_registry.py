from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from plugins.bs_preforcast.plugins import DefaultBsPreforcastPlugin
from plugins.bs_preforcast.registry import (
    PLUGIN_EXTENSION_RULES,
    available_plugins,
    get_bs_preforcast_plugin,
    plugin_registry,
)
from residual.models import BACKEND_PLUGIN_CATEGORY, available_plugins as residual_plugins


def test_bs_preforcast_registry_is_separate_from_residual_backend_registry() -> None:
    assert BACKEND_PLUGIN_CATEGORY == "backend"
    assert residual_plugins() == ("lightgbm", "randomforest", "xgboost")
    assert plugin_registry() == ("default",)
    assert available_plugins() == ("default",)
    assert "separate from residual.models.registry" in PLUGIN_EXTENSION_RULES[-1]


@pytest.mark.usefixtures("monkeypatch")
def test_default_bs_preforcast_plugin_delegates_authoritative_runtime_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = get_bs_preforcast_plugin()
    loaded = object()
    job = object()
    train_df = pd.DataFrame({"y": [1.0]})
    future_df = pd.DataFrame({"y": [2.0]})
    sentinel_loaded = object()
    calls: dict[str, Any] = {}

    def fake_resolve(*args: Any, **kwargs: Any) -> str:
        calls["resolve"] = (args, kwargs)
        return "futr_exog"

    def fake_prepare(*args: Any, **kwargs: Any):
        calls["prepare"] = (args, kwargs)
        return sentinel_loaded, train_df, future_df, "futr_exog"

    def fake_materialize(**kwargs: Any) -> None:
        calls["materialize"] = kwargs

    def fake_load_stage(repo_root: Path, cfg: Any) -> Any:
        calls["load_stage"] = (repo_root, cfg)
        return sentinel_loaded

    monkeypatch.setattr(
        "plugins.bs_preforcast.runtime.resolve_bs_preforcast_injection_mode",
        fake_resolve,
    )
    monkeypatch.setattr(
        "plugins.bs_preforcast.runtime.prepare_bs_preforcast_fold_inputs",
        fake_prepare,
    )
    monkeypatch.setattr(
        "plugins.bs_preforcast.runtime.materialize_bs_preforcast_stage",
        fake_materialize,
    )
    monkeypatch.setattr(
        "plugins.bs_preforcast.runtime.load_bs_preforcast_stage_config",
        fake_load_stage,
    )

    assert isinstance(plugin, DefaultBsPreforcastPlugin)
    assert plugin.resolve_injection_mode(loaded, selected_jobs=[job]) == "futr_exog"
    assert plugin.prepare_fold_inputs(loaded, job, train_df, future_df) == (
        sentinel_loaded,
        train_df,
        future_df,
        "futr_exog",
    )
    plugin.materialize_stage(
        loaded=loaded,
        selected_jobs=[job],
        run_root=Path("/tmp/run-root"),
        main_resolved_path=Path("/tmp/main-resolved.json"),
        main_capability_path=Path("/tmp/main-capability.json"),
        main_manifest_path=Path("/tmp/main-manifest.json"),
        entrypoint_version="test-version",
        validate_only=True,
    )
    assert plugin.load_stage_config(Path("/tmp/repo-root"), loaded) is sentinel_loaded

    assert calls["resolve"][0] == (loaded,)
    assert calls["resolve"][1] == {"selected_jobs": [job]}
    assert calls["prepare"][0] == (loaded, job, train_df, future_df)
    assert calls["prepare"][1] == {"run_root": None}
    assert calls["materialize"]["loaded"] is loaded
    assert calls["materialize"]["validate_only"] is True
    assert calls["load_stage"] == (Path("/tmp/repo-root"), loaded)
