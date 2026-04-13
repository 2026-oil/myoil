from __future__ import annotations

import importlib
from pathlib import Path

import plugin_contracts.stage_registry as stage_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stage_registry_discovers_only_installed_plugin_packages(
    monkeypatch,
) -> None:
    monkeypatch.setattr(stage_registry, "_PLUGINS_DISCOVERED", False)
    imported_modules: list[str] = []
    real_import_module = importlib.import_module

    def _recording_import_module(name: str):
        imported_modules.append(name)
        return real_import_module(name)

    monkeypatch.setattr(stage_registry.importlib, "import_module", _recording_import_module)

    expected_modules = {
        f"plugins.{path.parent.name}.plugin"
        for path in (REPO_ROOT / "plugins").glob("*/plugin.py")
    }

    discovered = stage_registry.all_stage_plugins()

    assert set(imported_modules) == expected_modules
    assert set(discovered) == {module.split(".")[1] for module in expected_modules}
    assert stage_registry.get_stage_plugin("bs_preforcast") is None
