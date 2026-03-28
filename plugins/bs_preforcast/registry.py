from __future__ import annotations

from typing import Callable

from .plugins import DefaultBsPreforcastPlugin

PLUGIN_EXTENSION_RULES = (
    "Add bs_preforcast implementations under bs_preforcast/plugins/ and register them here.",
    "Keep this registry separate from residual.models.registry.",
)

_PLUGIN_REGISTRY: dict[str, Callable[[], object]] = {
    "default": DefaultBsPreforcastPlugin,
}


def plugin_registry() -> tuple[str, ...]:
    return tuple(sorted(_PLUGIN_REGISTRY))


def available_plugins() -> tuple[str, ...]:
    return plugin_registry()


def get_bs_preforcast_plugin(name: str = "default") -> DefaultBsPreforcastPlugin:
    try:
        factory = _PLUGIN_REGISTRY[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported bs_preforcast plugin: {name}") from exc
    plugin = factory()
    if not isinstance(plugin, DefaultBsPreforcastPlugin):  # pragma: no cover - defensive
        raise TypeError(
            f"Registered bs_preforcast plugin {name!r} did not produce a DefaultBsPreforcastPlugin-compatible instance"
        )
    return plugin
