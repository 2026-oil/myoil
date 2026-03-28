from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import inspect
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class OptimizerResolution:
    name: str
    optimizer_cls: type[torch.optim.Optimizer]
    default_kwargs: dict[str, Any]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class _OptimizerSpec:
    loader: Callable[[], type[torch.optim.Optimizer]]
    default_kwargs: dict[str, Any]


def _load_torch_adamw() -> type[torch.optim.Optimizer]:
    return torch.optim.AdamW


def _load_pytorch_optimizer(name: str) -> type[torch.optim.Optimizer]:
    try:
        import pytorch_optimizer
    except ImportError as exc:  # pragma: no cover - exercised in integration/runtime only
        raise ImportError(
            "pytorch-optimizer is required for training.optimizer "
            f"name '{name}'. Install project dependencies first."
        ) from exc
    optimizer_cls = getattr(pytorch_optimizer, name, None)
    if optimizer_cls is None:
        raise ImportError(
            f"pytorch-optimizer does not expose optimizer class '{name}'"
        )
    return optimizer_cls


_REGISTRY: dict[str, _OptimizerSpec] = {
    "adamw": _OptimizerSpec(loader=_load_torch_adamw, default_kwargs={}),
    "ademamix": _OptimizerSpec(
        loader=lambda: _load_pytorch_optimizer("AdEMAMix"), default_kwargs={}
    ),
    "mars": _OptimizerSpec(
        loader=lambda: _load_pytorch_optimizer("MARS"), default_kwargs={}
    ),
    "soap": _OptimizerSpec(
        loader=lambda: _load_pytorch_optimizer("SOAP"), default_kwargs={}
    ),
}


def available_optimizers() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def validate_optimizer_class(
    name: str,
    optimizer_cls: type[torch.optim.Optimizer],
    optimizer_kwargs: dict[str, Any] | None = None,
) -> type[torch.optim.Optimizer]:
    if not inspect.isclass(optimizer_cls):
        raise TypeError(f"training.optimizer '{name}' did not resolve to a class")
    if not issubclass(optimizer_cls, torch.optim.Optimizer):
        raise TypeError(
            f"training.optimizer '{name}' must resolve to a torch.optim.Optimizer subclass"
        )
    signature = inspect.signature(optimizer_cls.__init__)
    if "params" not in signature.parameters:
        raise TypeError(
            f"training.optimizer '{name}' must accept a 'params' constructor argument"
        )
    step_method = getattr(optimizer_cls, "step", None)
    if step_method is None or not callable(step_method):
        raise TypeError(
            f"training.optimizer '{name}' must provide a callable step() method"
        )
    probe = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    probe_kwargs = deepcopy(optimizer_kwargs or {})
    probe_kwargs.setdefault("lr", 1e-3)
    try:
        optimizer = optimizer_cls([probe], **probe_kwargs)
        probe.grad = torch.ones_like(probe)
        optimizer.step()
        optimizer.zero_grad()
    except Exception as exc:  # pragma: no cover - exercised by compatibility tests
        raise TypeError(
            f"training.optimizer '{name}' failed compatibility validation: {exc}"
        ) from exc
    return optimizer_cls


def resolve_optimizer(
    name: str, overrides: dict[str, Any] | None = None
) -> OptimizerResolution:
    normalized_name = name.strip().lower()
    spec = _REGISTRY.get(normalized_name)
    if spec is None:
        raise ValueError(
            "training.optimizer.name must be one of: " + ", ".join(available_optimizers())
        )
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("training.optimizer.kwargs must be a mapping")
    default_kwargs = deepcopy(spec.default_kwargs)
    merged_kwargs = {**default_kwargs, **deepcopy(overrides)}
    optimizer_cls = validate_optimizer_class(
        normalized_name, spec.loader(), merged_kwargs
    )
    return OptimizerResolution(
        name=normalized_name,
        optimizer_cls=optimizer_cls,
        default_kwargs=default_kwargs,
        kwargs=merged_kwargs,
    )
