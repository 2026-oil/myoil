from __future__ import annotations

from .base import AABackboneAdapter, AABackboneEvidence
from .gru import build_gru_backbone
from .informer import build_informer_backbone
from .itransformer import build_itransformer_backbone
from .patchtst import build_patchtst_backbone
from .timexer import build_timexer_backbone
from .vanillatransformer import build_vanillatransformer_backbone

AA_BACKBONE_BUILDERS = {
    "gru": build_gru_backbone,
    "vanillatransformer": build_vanillatransformer_backbone,
    "informer": build_informer_backbone,
    "itransformer": build_itransformer_backbone,
    "patchtst": build_patchtst_backbone,
    "timexer": build_timexer_backbone,
}

AA_SUPPORTED_BACKBONES = frozenset(AA_BACKBONE_BUILDERS)

__all__ = [
    "AABackboneAdapter",
    "AABackboneEvidence",
    "AA_BACKBONE_BUILDERS",
    "AA_SUPPORTED_BACKBONES",
]
