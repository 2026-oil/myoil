from __future__ import annotations

from typing import Callable

from .plugins import DefaultBsPreforcastPlugin

PLUGIN_EXTENSION_RULES = (
    "Add bs_preforcast implementations under bs_preforcast/plugins/ and register them here.",
    "Keep this registry scoped to bs_preforcast stage plugins only.",
)

_PLUGIN_REGISTRY: dict[str, Callable[[], object]] = {
    "default": DefaultBsPreforcastPlugin,
}


def plugin_registry() -> tuple[str, ...]:
    return tuple(sorted(_PLUGIN_REGISTRY))


def available_plugins() -> tuple[str, ...]:
    return plugin_registry()


def get_bs_preforcast_plugin(name: str = "default") -> DefaultBsPreforcastPlugin:
    factory = _PLUGIN_REGISTRY[name.lower()]
    return factory()
