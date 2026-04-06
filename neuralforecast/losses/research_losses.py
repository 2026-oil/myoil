__all__ = ["LateHorizonWeightedMAPE", "HuberLateMAPE", "QuantileLateMAPE"]

from typing import Optional, Union

import torch
import torch.nn.functional as F

from neuralforecast.losses.pytorch import BasePointLoss, _divide_no_nan, _weighted_mean


class LateHorizonWeightedMAPE(BasePointLoss):
    """MAPE loss with exponentially increasing weights for later horizons.

    Designed for oil price forecasting where H6-H8 (surge period) accuracy
    is more critical than H1-H5. Applies a ramp weight that increases
    from horizon step 1 to H, controlled by ramp_power.

    Args:
        horizon (int): Forecast horizon length (default: 8).
        ramp_power (float): Power for weight ramp. Higher = more emphasis on late horizons.
            ramp_power=1: linear ramp [1,2,3,...,H]
            ramp_power=2: quadratic ramp [1,4,9,...,H^2]
            ramp_power=0.5: sqrt ramp [1,1.4,1.7,...,sqrt(H)]
        base_weight (float): Minimum weight for early horizons (default: 1.0).
        late_multiplier (float): Additional multiplier applied to horizons >= late_start (default: 1.0).
        late_start (int): Horizon step from which late_multiplier applies (default: 6).
    """

    def __init__(
        self,
        horizon: int = 8,
        ramp_power: float = 1.0,
        base_weight: float = 1.0,
        late_multiplier: float = 1.0,
        late_start: int = 6,
    ):
        self.horizon = horizon
        self.ramp_power = ramp_power
        self.base_weight = base_weight
        self.late_multiplier = late_multiplier
        self.late_start = late_start

        weights = torch.arange(1, horizon + 1, dtype=torch.float32) ** ramp_power
        weights = weights * base_weight
        weights[late_start - 1 :] *= late_multiplier

        super().__init__(
            horizon_weight=weights,
            outputsize_multiplier=1,
            output_names=[""],
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        scale = _divide_no_nan(torch.ones_like(y, device=y.device), torch.abs(y))
        losses = torch.abs(y - y_hat) * scale
        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


class HuberLateMAPE(BasePointLoss):
    """Huber-MAPE hybrid loss with late-horizon emphasis.

    Uses Huber loss for robustness to outliers on early horizons,
    and MAPE with increased weight on late horizons (H6-H8).

    Args:
        horizon (int): Forecast horizon length.
        delta (float): Huber threshold. Errors > delta use MAPE-like penalty.
        late_weight (float): Weight multiplier for horizons >= late_start.
        late_start (int): Horizon step from which increased weight applies.
    """

    def __init__(
        self,
        horizon: int = 8,
        delta: float = 1.0,
        late_weight: float = 3.0,
        late_start: int = 6,
    ):
        self.horizon = horizon
        self.delta = delta
        self.late_weight = late_weight
        self.late_start = late_start

        weights = torch.ones(horizon, dtype=torch.float32)
        weights[late_start - 1 :] *= late_weight

        super().__init__(
            horizon_weight=weights,
            outputsize_multiplier=1,
            output_names=[""],
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        error = y - y_hat
        abs_error = torch.abs(error)

        huber = torch.where(
            abs_error <= self.delta,
            0.5 * error**2,
            self.delta * (abs_error - 0.5 * self.delta),
        )

        scale = _divide_no_nan(torch.ones_like(y, device=y.device), torch.abs(y))
        mape_component = abs_error * scale

        losses = torch.where(
            abs_error <= self.delta,
            huber,
            huber + mape_component * 0.1,
        )

        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


class QuantileLateMAPE(BasePointLoss):
    """Quantile loss with asymmetric penalties for late-horizon underestimation.

    Oil price surges are more costly when underestimated. This loss applies
    heavier penalties for underestimation (y_hat < y) on late horizons.

    Args:
        horizon (int): Forecast horizon length.
        q_under (float): Quantile for underestimation penalty (default: 0.7).
        q_over (float): Quantile for overestimation penalty (default: 0.3).
        late_start (int): Horizon step from which asymmetric penalties apply.
        late_factor (float): Multiplier for late-horizon penalties.
    """

    def __init__(
        self,
        horizon: int = 8,
        q_under: float = 0.7,
        q_over: float = 0.3,
        late_start: int = 6,
        late_factor: float = 2.0,
    ):
        self.horizon = horizon
        self.q_under = q_under
        self.q_over = q_over
        self.late_start = late_start
        self.late_factor = late_factor

        weights = torch.ones(horizon, dtype=torch.float32)
        weights[late_start - 1 :] *= late_factor

        super().__init__(
            horizon_weight=weights,
            outputsize_multiplier=1,
            output_names=[""],
        )

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        error = y_hat - y
        scale = _divide_no_nan(torch.ones_like(y, device=y.device), torch.abs(y))

        under = torch.max(error, torch.zeros_like(error))
        over = torch.max(-error, torch.zeros_like(error))

        losses = (self.q_over * under + self.q_under * over) * scale

        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)
