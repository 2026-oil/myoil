from __future__ import annotations

from abc import ABC, abstractmethod
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
    def fit(self, train_df: pd.DataFrame, context: ResidualContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def predict_future(self, future_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError
