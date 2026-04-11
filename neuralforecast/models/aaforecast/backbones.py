from __future__ import annotations

import torch.nn as nn

from .models import AA_BACKBONE_BUILDERS, AA_SUPPORTED_BACKBONES


def build_aaforecast_backbone(backbone: str, **kwargs: object) -> nn.Module:
    normalized = str(backbone).strip().lower()
    builder = AA_BACKBONE_BUILDERS.get(normalized)
    if builder is None:
        supported = ", ".join(sorted(AA_SUPPORTED_BACKBONES))
        raise ValueError(f"AAForecast backbone must be one of: {supported}")
    return builder(**kwargs)


__all__ = ["AA_SUPPORTED_BACKBONES", "build_aaforecast_backbone"]
