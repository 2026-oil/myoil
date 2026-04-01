from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Constructor-level aliases that must stay owned by shared training config rather
# than NEC branch-local `model_params`.
CENTRALIZED_TRAINING_MODEL_PARAM_KEYS = frozenset(
    {
        "accelerator",
        "batch_size",
        "devices",
        "early_stop_patience_steps",
        "inference_windows_batch_size",
        "lr_scheduler",
        "lr_scheduler_kwargs",
        "max_lr",
        "max_steps",
        "min_steps_before_early_stop",
        "optimizer",
        "optimizer_kwargs",
        "precision",
        "scaler_type",
        "step_size",
        "strategy",
        "valid_batch_size",
        "val_check_steps",
        "windows_batch_size",
        "_lr_scheduler_cls",
        "_lr_scheduler_kwargs",
        "_max_lr",
    }
)


def duplicated_centralized_training_model_param_keys(
    params: Mapping[str, Any],
) -> set[str]:
    return CENTRALIZED_TRAINING_MODEL_PARAM_KEYS.intersection(params)
