from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from statsmodels.nonparametric.smoothers_lowess import lowess


def _safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    stabilized = torch.where(
        denominator.abs() < eps,
        torch.full_like(denominator, eps),
        denominator,
    )
    return numerator / stabilized


class STARFeatureExtractor(nn.Module):
    """Paper-aligned STAR decomposition helper for AA-Forecast.

    The original AA-Forecast pipeline performs multiplicative decomposition
    using LOWESS trend estimation, periodic seasonal averages, and anomaly
    extraction from the residual component. The implementation here keeps that
    structure while remaining compatible with PyTorch/neuralforecast training.
    """

    def __init__(
        self,
        season_length: int = 12,
        lowess_frac: float = 0.6,
        lowess_delta: float = 0.01,
        anomaly_threshold: float = 3.5,
    ):
        super().__init__()
        self.season_length = max(2, int(season_length))
        self.lowess_frac = float(lowess_frac)
        self.lowess_delta = float(lowess_delta)
        self.anomaly_threshold = float(anomaly_threshold)
        self._trend_cache: dict[bytes, np.ndarray] = {}
        self._trend_cache_order: list[bytes] = []
        self._trend_cache_limit = 4096

    def _trend(self, x: torch.Tensor) -> torch.Tensor:
        series = x.detach().cpu().numpy()
        batch, seq_len, channels = series.shape
        positions = np.arange(seq_len, dtype=float)
        trend = np.empty_like(series, dtype=np.float32)
        lowess_delta = self.lowess_delta * max(seq_len, 1)
        for batch_idx in range(batch):
            for channel_idx in range(channels):
                observed = series[batch_idx, :, channel_idx].astype(float, copy=False)
                cache_key = observed.astype(np.float32, copy=False).tobytes()
                cached = self._trend_cache.get(cache_key)
                if cached is None:
                    if np.allclose(observed, observed[0]):
                        smoothed = observed.copy()
                    else:
                        smoothed = lowess(
                            observed,
                            positions,
                            frac=self.lowess_frac,
                            delta=lowess_delta,
                            return_sorted=False,
                        )
                    cached = smoothed.astype(np.float32)
                    self._trend_cache[cache_key] = cached
                    self._trend_cache_order.append(cache_key)
                    if len(self._trend_cache_order) > self._trend_cache_limit:
                        stale_key = self._trend_cache_order.pop(0)
                        self._trend_cache.pop(stale_key, None)
                trend[batch_idx, :, channel_idx] = cached
        return torch.as_tensor(trend, device=x.device, dtype=x.dtype)

    def _seasonal(self, detrended: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = detrended.shape
        seasonal = torch.ones_like(detrended)
        period = min(self.season_length, seq_len)
        for phase in range(period):
            phase_values = detrended[:, phase:seq_len:period, :]
            if phase_values.numel() == 0:
                continue
            phase_mean = phase_values.mean(dim=1, keepdim=True)
            seasonal[:, phase:seq_len:period, :] = phase_mean.expand_as(
                seasonal[:, phase:seq_len:period, :]
            )
        return seasonal

    def forward(self, insample_y: torch.Tensor) -> dict[str, torch.Tensor]:
        trend = self._trend(insample_y)
        detrended = _safe_divide(insample_y, trend)
        seasonal = self._seasonal(detrended)
        residual = _safe_divide(detrended, seasonal)

        residual_center = residual.median(dim=1, keepdim=True).values
        mad = (
            (residual - residual_center)
            .abs()
            .median(dim=1, keepdim=True)
            .values.clamp_min(1e-4)
        )
        robustness = 0.6745 * (residual - residual_center).abs() / mad
        anomaly_mask = robustness > self.anomaly_threshold

        anomalies = torch.where(anomaly_mask, residual, torch.ones_like(residual))
        cleaned_residual = torch.where(
            anomaly_mask,
            torch.ones_like(residual),
            residual,
        )
        return {
            "trend": trend,
            "seasonal": seasonal,
            "anomalies": anomalies,
            "residual": cleaned_residual,
            "critical_mask": anomaly_mask,
        }


class CriticalSparseAttention(nn.Module):
    """Attention over anomaly/event timesteps only."""

    def __init__(self, hidden_size: int, attention_hidden_size: int | None = None):
        super().__init__()
        attention_hidden_size = (
            hidden_size if attention_hidden_size is None else int(attention_hidden_size)
        )
        self.proj = nn.Linear(hidden_size, attention_hidden_size)
        self.score = nn.Linear(attention_hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        critical_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = critical_mask.squeeze(-1).bool()
        logits = self.score(torch.tanh(self.proj(hidden_states))).squeeze(-1)
        masked_logits = logits.masked_fill(~mask, -1e9)
        has_any = mask.any(dim=1, keepdim=True)
        weights = torch.softmax(masked_logits, dim=1)
        weights = torch.where(has_any, weights, torch.zeros_like(weights))

        context = torch.einsum("bl,blh->bh", weights, hidden_states)
        fallback = hidden_states[:, -1, :]
        context = torch.where(has_any, context, fallback)
        expanded_context = context.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        attended = torch.where(mask.unsqueeze(-1), expanded_context, hidden_states)
        return attended, weights
