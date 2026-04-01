from __future__ import annotations

import torch
import torch.nn as nn


class NecEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            layer_dim,
            dropout=dropout if layer_dim > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        device = x.device
        h0 = torch.zeros(self.layer_dim, batch, self.hidden_dim, device=device)
        c0 = torch.zeros(self.layer_dim, batch, self.hidden_dim, device=device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]


class NecRegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(self.bn1(x))
        x = self.fc2(self.bn2(x))
        return self.fc3(self.bn3(x))


class NecClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc1(self.bn1(x)))
