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
        thresh: float = 3.5,
    ):
        super().__init__()
        self.season_length = max(2, int(season_length))
        self.lowess_frac = float(lowess_frac)
        self.lowess_delta = float(lowess_delta)
        self.thresh = float(thresh)
        if self.thresh < 0:
            raise ValueError("STARFeatureExtractor thresh must satisfy value >= 0")
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

    @staticmethod
    def _normalize_tail_modes(
        tail_modes: tuple[str, ...] | list[str] | None,
        *,
        channels: int,
    ) -> tuple[str, ...]:
        if tail_modes is None:
            return tuple("two_sided" for _ in range(channels))
        normalized = tuple(str(mode).strip().lower() for mode in tail_modes)
        if len(normalized) != channels:
            raise ValueError(
                f"STARFeatureExtractor tail_modes length must equal channels ({channels})"
            )
        invalid = sorted(set(normalized).difference({"two_sided", "upward"}))
        if invalid:
            raise ValueError(
                "STARFeatureExtractor tail_modes contain unsupported value(s): "
                + ", ".join(invalid)
            )
        return normalized

    @staticmethod
    def _robust_scores(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_center = residual.median(dim=1, keepdim=True).values
        mad = (
            (residual - residual_center)
            .abs()
            .median(dim=1, keepdim=True)
            .values.clamp_min(1e-4)
        )
        signed_score = 0.6745 * (residual - residual_center) / mad
        abs_score = signed_score.abs()
        return signed_score, abs_score

    def _anomaly_mask(
        self,
        signed_score: torch.Tensor,
        *,
        tail_modes: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        threshold = torch.full_like(signed_score[:, :1, :], self.thresh)
        anomaly_mask = signed_score.abs() > threshold
        for channel_idx, mode in enumerate(tail_modes):
            if mode == "upward":
                anomaly_mask[:, :, channel_idx] = (
                    signed_score[:, :, channel_idx] > self.thresh
                )
        return anomaly_mask, threshold

    def forward(
        self,
        insample_y: torch.Tensor,
        *,
        tail_modes: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        trend = self._trend(insample_y)
        detrended = _safe_divide(insample_y, trend)
        seasonal = self._seasonal(detrended)
        residual = _safe_divide(detrended, seasonal)
        normalized_tail_modes = self._normalize_tail_modes(
            tail_modes,
            channels=residual.size(2),
        )
        signed_score, abs_score = self._robust_scores(residual)
        anomaly_mask, cutoff = self._anomaly_mask(
            signed_score,
            tail_modes=normalized_tail_modes,
        )

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
            "robust_score_signed": signed_score,
            "robust_score_abs": abs_score,
            "ranking_score": torch.where(
                torch.tensor(
                    [mode == "upward" for mode in normalized_tail_modes],
                    device=signed_score.device,
                    dtype=torch.bool,
                ).view(1, 1, -1),
                signed_score.clamp_min(0.0),
                signed_score.abs(),
            ),
            "ranking_cutoff": cutoff,
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
        count_active_channels: torch.Tensor,
        channel_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = critical_mask.squeeze(-1).bool()
        if count_active_channels.ndim != 3:
            raise ValueError("CriticalSparseAttention count_active_channels must be rank-3")
        if channel_activity.ndim != 3:
            raise ValueError("CriticalSparseAttention channel_activity must be rank-3")
        count_signal = count_active_channels.squeeze(-1).to(dtype=hidden_states.dtype)
        activity = channel_activity.to(dtype=hidden_states.dtype).clamp_min(0.0)
        logits = self.score(torch.tanh(self.proj(hidden_states))).squeeze(-1)
        logits = logits + torch.log1p(count_signal.clamp_min(0.0))
        channel_mask = activity > 0
        channel_logits = logits.unsqueeze(1) + torch.log1p(activity.transpose(1, 2))
        channel_logits = channel_logits.masked_fill(~channel_mask.transpose(1, 2), -1e9)
        has_channel_any = channel_mask.any(dim=1)
        weights = torch.softmax(channel_logits, dim=-1)
        weights = torch.where(
            has_channel_any.unsqueeze(-1),
            weights,
            torch.zeros_like(weights),
        )

        channel_contexts = torch.einsum("bcl,blh->bch", weights, hidden_states)
        fallback = hidden_states[:, -1, :].unsqueeze(1).expand_as(channel_contexts)
        channel_contexts = torch.where(
            has_channel_any.unsqueeze(-1),
            channel_contexts,
            fallback,
        )
        mix_denom = activity.sum(dim=2, keepdim=True).clamp_min(1.0)
        mix = activity / mix_denom
        expanded_context = torch.einsum("blc,bch->blh", mix, channel_contexts)
        max_count = count_signal.amax(dim=1, keepdim=True).clamp_min(1.0)
        density_gate = 1.0 + (count_signal / max_count)
        attended = torch.where(
            mask.unsqueeze(-1),
            expanded_context * density_gate.unsqueeze(-1),
            hidden_states,
        )
        return attended, weights


class ITransformerTokenSparseAttention(nn.Module):
    """Sparse attention over iTransformer token states."""

    def __init__(self, hidden_size: int, attention_hidden_size: int | None = None):
        super().__init__()
        attention_hidden_size = (
            hidden_size if attention_hidden_size is None else int(attention_hidden_size)
        )
        self.proj = nn.Linear(hidden_size, attention_hidden_size)
        self.score = nn.Linear(attention_hidden_size, 1)

    def forward(
        self,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        token_count: torch.Tensor,
        token_activity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_states.ndim != 3:
            raise ValueError("ITransformerTokenSparseAttention token_states must be rank-3")
        if token_mask.ndim != 3:
            raise ValueError("ITransformerTokenSparseAttention token_mask must be rank-3")
        if token_count.ndim != 3:
            raise ValueError("ITransformerTokenSparseAttention token_count must be rank-3")
        if token_activity.ndim != 3:
            raise ValueError("ITransformerTokenSparseAttention token_activity must be rank-3")

        token_mask = token_mask.squeeze(-1).bool()
        token_count = token_count.squeeze(-1).to(dtype=token_states.dtype)
        token_activity = token_activity.squeeze(-1).to(dtype=token_states.dtype).clamp_min(0.0)

        logits = self.score(torch.tanh(self.proj(token_states))).squeeze(-1)
        logits = logits + torch.log1p(token_count) + torch.log1p(token_activity)
        logits = logits.masked_fill(~token_mask, -1e9)

        has_any = token_mask.any(dim=1, keepdim=True)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(has_any, weights, torch.zeros_like(weights))

        token_context = torch.einsum("bt,bth->bh", weights, token_states)
        fallback = token_states[:, 0, :]
        token_context = torch.where(has_any, token_context, fallback)

        max_count = token_count.amax(dim=1, keepdim=True).clamp_min(1.0)
        density_gate = 1.0 + (token_count / max_count)
        attended = torch.where(
            token_mask.unsqueeze(-1),
            token_context.unsqueeze(1) * density_gate.unsqueeze(-1),
            token_states,
        )
        return attended, weights


class TimeXerTokenSparseAttention(nn.Module):
    """Sparse attention over per-series TimeXer patch/global tokens."""

    def __init__(self, hidden_size: int, attention_hidden_size: int | None = None):
        super().__init__()
        attention_hidden_size = (
            hidden_size if attention_hidden_size is None else int(attention_hidden_size)
        )
        self.proj = nn.Linear(hidden_size, attention_hidden_size)
        self.score = nn.Linear(attention_hidden_size, 1)

    def forward(
        self,
        patch_states: torch.Tensor,
        global_states: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_count: torch.Tensor,
        patch_activity: torch.Tensor,
        global_mask: torch.Tensor,
        global_count: torch.Tensor,
        global_activity: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if patch_states.ndim != 4:
            raise ValueError("TimeXerTokenSparseAttention patch_states must be rank-4")
        if global_states.ndim != 4:
            raise ValueError("TimeXerTokenSparseAttention global_states must be rank-4")
        if patch_mask.ndim != 3 or global_mask.ndim != 3:
            raise ValueError("TimeXerTokenSparseAttention masks must be rank-3")
        if patch_count.ndim != 3 or global_count.ndim != 3:
            raise ValueError("TimeXerTokenSparseAttention counts must be rank-3")
        if patch_activity.ndim != 3 or global_activity.ndim != 3:
            raise ValueError("TimeXerTokenSparseAttention activity tensors must be rank-3")

        token_states = torch.cat([patch_states, global_states], dim=2)
        token_mask = torch.cat([patch_mask, global_mask], dim=1).squeeze(-1).bool()
        token_count = torch.cat([patch_count, global_count], dim=1).squeeze(-1)
        token_activity = torch.cat([patch_activity, global_activity], dim=1).squeeze(-1)

        token_count = token_count.to(dtype=token_states.dtype)
        token_activity = token_activity.to(dtype=token_states.dtype).clamp_min(0.0)

        logits = self.score(torch.tanh(self.proj(token_states))).squeeze(-1)
        logits = logits + torch.log1p(token_count).unsqueeze(1)
        logits = logits + torch.log1p(token_activity).unsqueeze(1)

        token_mask = token_mask.unsqueeze(1).expand(-1, token_states.shape[1], -1)
        logits = logits.masked_fill(~token_mask, -1e9)

        has_any = token_mask.any(dim=2, keepdim=True)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(has_any, weights, torch.zeros_like(weights))

        token_context = torch.einsum("bct,bcth->bch", weights, token_states)
        fallback = global_states.squeeze(2)
        token_context = torch.where(
            has_any.expand_as(token_context),
            token_context,
            fallback,
        )

        max_count = token_count.amax(dim=1, keepdim=True).clamp_min(1.0)
        density_gate = 1.0 + (token_count / max_count)
        expanded_context = token_context.unsqueeze(2).expand_as(token_states)
        attended = torch.where(
            token_mask.unsqueeze(-1),
            expanded_context * density_gate.unsqueeze(1).unsqueeze(-1),
            token_states,
        )

        patch_num = patch_states.shape[2]
        return (attended[:, :, :patch_num, :], attended[:, :, patch_num:, :]), weights
