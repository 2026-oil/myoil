"""Global registry for :class:`~residual.stage_plugin.StagePlugin` instances.

Concrete stage plugins (e.g. ``bs_preforcast``) register themselves at import
time so that ``residual/`` never references them directly.
"""
from __future__ import annotations

from typing import Any

from plugin_contracts.stage_plugin import StagePlugin

_STAGE_PLUGINS: dict[str, StagePlugin] = {}


def register_stage_plugin(plugin: StagePlugin) -> None:
    key = plugin.config_key
    if key in _STAGE_PLUGINS:
        raise ValueError(f"Stage plugin already registered for key {key!r}")
    _STAGE_PLUGINS[key] = plugin


def get_stage_plugin(key: str) -> StagePlugin | None:
    return _STAGE_PLUGINS.get(key)


def all_stage_plugins() -> dict[str, StagePlugin]:
    return dict(_STAGE_PLUGINS)


def _ensure_plugins_loaded() -> None:
    """Trigger lazy discovery of known stage-plugin packages.

    Called once during config loading so that plugins registered via their
    package ``__init__`` are available without requiring an explicit import
    at the top of ``residual/``.
    """
    if _STAGE_PLUGINS:
        return
    try:
        import bs_preforcast.plugin  # noqa: F401  side-effect: registers plugin
    except ImportError:
        pass


def get_stage_plugin_for_payload(payload: dict[str, Any]) -> StagePlugin | None:
    """Return the first registered plugin whose ``config_key`` appears in *payload*."""
    _ensure_plugins_loaded()
    for key, plugin in _STAGE_PLUGINS.items():
        if key in payload:
            return plugin
    return None


def get_active_stage_plugin(config: Any) -> tuple[StagePlugin, Any] | None:
    """Return ``(plugin, plugin_config)`` if a stage plugin is active on *config*.

    *config* is expected to be an ``AppConfig`` whose ``stage_plugin_config``
    may be populated.
    """
    _ensure_plugins_loaded()
    stage_cfg = getattr(config, "stage_plugin_config", None)
    if stage_cfg is None:
        return None
    for plugin in _STAGE_PLUGINS.values():
        if plugin.is_enabled(stage_cfg):
            return plugin, stage_cfg
    return None
