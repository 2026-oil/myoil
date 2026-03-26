from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from residual.models import (
    BACKEND_PLUGIN_CATEGORY,
    BS_PREFORCAST_PLUGIN_CATEGORY,
    PLUGIN_EXTENSION_RULES,
    available_plugins,
    build_residual_plugin,
    get_bs_preforcast_plugin,
    plugin_registry,
)
from residual.models.bs_preforcast import DefaultBsPreforcastPlugin
from residual.registry import (
    BACKEND_PLUGIN_CATEGORY as LEGACY_BACKEND_PLUGIN_CATEGORY,
    BS_PREFORCAST_PLUGIN_CATEGORY as LEGACY_BS_PREFORCAST_PLUGIN_CATEGORY,
    available_plugins as legacy_available_plugins,
    get_bs_preforcast_plugin as legacy_get_bs_preforcast_plugin,
    plugin_registry as legacy_plugin_registry,
)


def test_plugin_registry_exposes_exact_two_supported_categories() -> None:
    registry = plugin_registry()

    assert BACKEND_PLUGIN_CATEGORY == 'backend'
    assert BS_PREFORCAST_PLUGIN_CATEGORY == 'bs_preforcast'
    assert registry.names(BACKEND_PLUGIN_CATEGORY) == (
        'lightgbm',
        'randomforest',
        'xgboost',
    )
    assert registry.names(BS_PREFORCAST_PLUGIN_CATEGORY) == ('default',)
    assert available_plugins(BACKEND_PLUGIN_CATEGORY) == registry.names(
        BACKEND_PLUGIN_CATEGORY
    )
    assert available_plugins(BS_PREFORCAST_PLUGIN_CATEGORY) == registry.names(
        BS_PREFORCAST_PLUGIN_CATEGORY
    )
    assert len(PLUGIN_EXTENSION_RULES) == 3
    assert 'backend and bs_preforcast' in PLUGIN_EXTENSION_RULES[-1]


def test_legacy_registry_exports_point_to_common_plugin_registry_surface() -> None:
    assert LEGACY_BACKEND_PLUGIN_CATEGORY == BACKEND_PLUGIN_CATEGORY
    assert LEGACY_BS_PREFORCAST_PLUGIN_CATEGORY == BS_PREFORCAST_PLUGIN_CATEGORY
    assert legacy_plugin_registry() is plugin_registry()
    assert legacy_available_plugins(BACKEND_PLUGIN_CATEGORY) == available_plugins(
        BACKEND_PLUGIN_CATEGORY
    )
    assert isinstance(legacy_get_bs_preforcast_plugin(), DefaultBsPreforcastPlugin)


@pytest.mark.usefixtures('monkeypatch')
def test_default_bs_preforcast_plugin_delegates_runtime_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = get_bs_preforcast_plugin()
    loaded = object()
    job = object()
    train_df = pd.DataFrame({'y': [1.0]})
    future_df = pd.DataFrame({'y': [2.0]})
    sentinel_loaded = object()
    sentinel_payload: dict[str, Any] = {'ok': True}
    calls: dict[str, Any] = {}

    def fake_resolve(*args: Any, **kwargs: Any) -> str:
        calls['resolve'] = (args, kwargs)
        return 'futr_exog'

    def fake_prepare(*args: Any, **kwargs: Any):
        calls['prepare'] = (args, kwargs)
        return sentinel_loaded, train_df, future_df, sentinel_payload

    def fake_materialize(**kwargs: Any) -> None:
        calls['materialize'] = kwargs

    def fake_load_stage(repo_root: Path, cfg: Any) -> Any:
        calls['load_stage'] = (repo_root, cfg)
        return sentinel_loaded

    monkeypatch.setattr(
        'residual.bs_preforcast_runtime.resolve_bs_preforcast_injection_mode',
        fake_resolve,
    )
    monkeypatch.setattr(
        'residual.bs_preforcast_runtime.prepare_bs_preforcast_fold_inputs',
        fake_prepare,
    )
    monkeypatch.setattr(
        'residual.bs_preforcast_runtime.materialize_bs_preforcast_stage',
        fake_materialize,
    )
    monkeypatch.setattr(
        'residual.bs_preforcast_runtime.load_bs_preforcast_stage_config',
        fake_load_stage,
    )

    assert plugin.resolve_injection_mode(loaded, selected_jobs=[job]) == 'futr_exog'
    assert plugin.prepare_fold_inputs(loaded, job, train_df, future_df) == (
        sentinel_loaded,
        train_df,
        future_df,
        sentinel_payload,
    )

    run_root = Path('/tmp/run-root')
    plugin.materialize_stage(
        loaded=loaded,
        selected_jobs=[job],
        run_root=run_root,
        main_resolved_path=Path('/tmp/main-resolved.json'),
        main_capability_path=Path('/tmp/main-capability.json'),
        main_manifest_path=Path('/tmp/main-manifest.json'),
        entrypoint_version='test-version',
        validate_only=True,
    )
    assert plugin.load_stage_config(Path('/tmp/repo-root'), loaded) is sentinel_loaded

    assert calls['resolve'][0] == (loaded,)
    assert calls['resolve'][1] == {'selected_jobs': [job]}
    assert calls['prepare'][0] == (loaded, job, train_df, future_df)
    assert calls['materialize']['loaded'] is loaded
    assert calls['materialize']['selected_jobs'] == [job]
    assert calls['materialize']['validate_only'] is True
    assert calls['load_stage'] == (Path('/tmp/repo-root'), loaded)


def test_backend_builder_uses_new_models_package() -> None:
    pytest.importorskip('xgboost')
    from residual.models.backends.xgboost import XGBoostResidualPlugin

    plugin = build_residual_plugin({'model': 'xgboost', 'params': {'n_estimators': 8}})

    assert isinstance(plugin, XGBoostResidualPlugin)
    assert plugin.metadata()['plugin'] == 'xgboost'
    assert plugin.metadata()['n_estimators'] == 8
