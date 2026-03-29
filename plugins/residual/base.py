from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ResidualContext:
    job_name: str
    model_name: str
    output_dir: Path
    config: dict[str, Any]


class ResidualPlugin(ABC):
    name: str

    @abstractmethod
    def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class PluginDefinition:
    category: str
    name: str
    factory: Callable[..., Any]
    description: str = ""
    aliases: tuple[str, ...] = ()


class PluginRegistry:
    def __init__(self, definitions: Iterable[PluginDefinition] | None = None) -> None:
        self._definitions: dict[tuple[str, str], PluginDefinition] = {}
        for definition in definitions or ():
            self.register(definition)

    def register(self, definition: PluginDefinition) -> None:
        keys = ((definition.category, definition.name.lower()),) + tuple(
            (definition.category, alias.lower()) for alias in definition.aliases
        )
        for key in keys:
            self._definitions[key] = definition

    def definition(self, category: str, name: str) -> PluginDefinition:
        key = (category, name.lower())
        if key not in self._definitions:
            raise ValueError(f"Unsupported {category} plugin: {name}")
        return self._definitions[key]

    def create(self, category: str, name: str, *args: Any, **kwargs: Any) -> Any:
        definition = self.definition(category, name)
        return definition.factory(*args, **kwargs)

    def names(self, category: str | None = None) -> tuple[str, ...]:
        names = {
            definition.name
            for definition in self._definitions.values()
            if category is None or definition.category == category
        }
        return tuple(sorted(names))
