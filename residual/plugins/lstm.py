from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import pandas as pd

from residual.plugins_base import ResidualContext, ResidualPlugin


class _LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


@dataclass(frozen=True)
class _LSTMConfig:
    lookback: int = 4
    hidden_size: int = 8
    num_layers: int = 1
    epochs: int = 20
    learning_rate: float = 0.01


class LSTMResidualPlugin(ResidualPlugin):
    name = "lstm"

    def __init__(
        self,
        *,
        lookback: int = 4,
        hidden_size: int = 8,
        num_layers: int = 1,
        epochs: int = 20,
        learning_rate: float = 0.01,
    ):
        self.config = _LSTMConfig(
            lookback=lookback,
            hidden_size=hidden_size,
            num_layers=num_layers,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        self.model = _LSTMRegressor(hidden_size=hidden_size, num_layers=num_layers)
        self.history: list[float] = []
        self._trained = False
        self._has_trained_weights = False
        self._fallback_value = 0.0

    def _tensorize(self, values: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        lookback = self.config.lookback
        xs: list[list[float]] = []
        ys: list[float] = []
        for idx in range(lookback, len(values)):
            xs.append(values[idx - lookback : idx])
            ys.append(values[idx])
        if not xs:
            raise ValueError(
                "Residual train series is too short for configured lookback"
            )
        x = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
        return x, y

    def fit(self, train_df: pd.DataFrame, context: ResidualContext) -> None:
        series = train_df.sort_values("ds")["residual_target"].astype(float).tolist()
        self.history = series
        self._fallback_value = float(series[-1]) if series else 0.0
        self._has_trained_weights = False
        if len(series) <= self.config.lookback:
            self._trained = True
            return
        x, y = self._tensorize(series)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.config.epochs):
            optimizer.zero_grad()
            pred = self.model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        self._trained = True
        self._has_trained_weights = True

    def _predict_next(self, history: list[float]) -> float:
        lookback = self.config.lookback
        if len(history) < lookback or not self._has_trained_weights:
            return self._fallback_value
        window = history[-lookback:]
        x = torch.tensor(window, dtype=torch.float32).view(1, lookback, 1)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x).item()
        return float(pred)

    def predict_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("Residual plugin is not trained")
        ordered = train_df.sort_values("ds").reset_index(drop=True).copy()
        preds: list[float] = []
        history = ordered["residual_target"].astype(float).tolist()
        for idx in range(len(history)):
            if idx == 0:
                preds.append(self._fallback_value)
            else:
                preds.append(self._predict_next(history[:idx]))
        ordered["residual_hat"] = preds
        return ordered.reset_index(drop=True)

    def predict_future(self, future_df: pd.DataFrame) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("Residual plugin is not trained")
        ordered = future_df.sort_values("ds").reset_index(drop=True).copy()
        history = list(self.history)
        preds: list[float] = []
        for _ in range(len(ordered)):
            pred = self._predict_next(history)
            preds.append(pred)
            history.append(pred)
        ordered["residual_hat"] = preds
        return ordered

    def metadata(self) -> dict[str, Any]:
        return {
            "plugin": self.name,
            "lookback": self.config.lookback,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
        }
