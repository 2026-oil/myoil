from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from neuralforecast.losses.pytorch import BasePointLoss, _weighted_mean


class NecSelectivePointLoss(BasePointLoss):
    def __init__(
        self,
        *,
        epsilon: float,
        branch_name: Literal["normal", "extreme"],
        base_loss: Literal["mae", "mse"] = "mse",
        horizon_weight=None,
    ) -> None:
        super().__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=1,
            output_names=[""],
        )
        self.epsilon = float(epsilon)
        self.branch_name = branch_name
        self.base_loss = base_loss

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del y_insample
        selected = (
            torch.abs(y) <= self.epsilon
            if self.branch_name == "normal"
            else torch.abs(y) > self.epsilon
        )
        weights = self._compute_weights(y=y, mask=mask) * selected.to(dtype=y.dtype)
        if self.base_loss == "mae":
            losses = torch.abs(y - y_hat)
        else:
            losses = (y - y_hat) ** 2
        return _weighted_mean(losses=losses, weights=weights)


class NecClassifierLoss(BasePointLoss):
    def __init__(
        self,
        *,
        alpha: float = 2.0,
        beta: float = 0.5,
        horizon_weight=None,
    ) -> None:
        super().__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=1,
            output_names=[""],
        )
        if alpha <= 0:
            raise ValueError("NEC classifier alpha must be > 0")
        if not 0 <= beta <= 1:
            raise ValueError("NEC classifier beta must be between 0 and 1")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del y_insample
        weights = self._compute_weights(y=y, mask=mask)
        probs = torch.sigmoid(y_hat)
        powered_probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6) ** self.alpha
        bce_losses = F.binary_cross_entropy(powered_probs, y, reduction="none")
        bce_term = _weighted_mean(losses=bce_losses, weights=weights)
        mse_term = _weighted_mean(losses=(y - probs) ** 2, weights=weights)
        rmse_term = torch.sqrt(torch.clamp(mse_term, min=0.0) + 1e-12)
        return self.beta * bce_term + (1.0 - self.beta) * rmse_term
