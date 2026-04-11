"""Internal AA-Forecast building blocks.

This package is intentionally internal-only. The authoritative public model
surface is ``neuralforecast.models.AAForecast``.
"""

from .modules import (
    CriticalSparseAttention,
    ITransformerTokenSparseAttention,
    STARFeatureExtractor,
    TimeXerTokenSparseAttention,
)

__all__ = [
    "CriticalSparseAttention",
    "ITransformerTokenSparseAttention",
    "STARFeatureExtractor",
    "TimeXerTokenSparseAttention",
]
