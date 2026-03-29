from __future__ import annotations

import pytest
import torch

from plugins.optimizer import (
    available_optimizers,
    resolve_optimizer,
    validate_optimizer_class,
)


def test_available_optimizers_matches_expected_registry() -> None:
    assert available_optimizers() == (
        "adamw",
        "ademamix",
        "mars",
        "soap",
        "rmsprop",
        "radam",
    )


@pytest.mark.parametrize("name", available_optimizers())
def test_resolve_optimizer_returns_optimizer_subclass(name: str) -> None:
    resolved = resolve_optimizer(name, {})

    assert resolved.name == name
    assert issubclass(resolved.optimizer_cls, torch.optim.Optimizer)
    assert resolved.kwargs == {}


@pytest.mark.parametrize(
    ("name", "optimizer_cls"),
    [
        ("rmsprop", torch.optim.RMSprop),
        ("radam", torch.optim.RAdam),
    ],
)
def test_resolve_optimizer_returns_expected_native_torch_optimizer(
    name: str, optimizer_cls: type[torch.optim.Optimizer]
) -> None:
    resolved = resolve_optimizer(name, {})

    assert resolved.optimizer_cls is optimizer_cls
    assert resolved.default_kwargs == {}
    assert resolved.kwargs == {}


def test_resolve_optimizer_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="training.optimizer.name must be one of"):
        resolve_optimizer("madeup", {})


def test_validate_optimizer_class_rejects_non_optimizer_subclass() -> None:
    class NotOptimizer:
        pass

    with pytest.raises(TypeError, match="torch.optim.Optimizer subclass"):
        validate_optimizer_class("broken", NotOptimizer)  # type: ignore[arg-type]


def test_validate_optimizer_class_rejects_missing_params_arg() -> None:
    class MissingParamsOptimizer(torch.optim.Optimizer):
        def __init__(self) -> None:
            pass

        def step(self, closure=None):  # pragma: no cover - never executed
            return None

    with pytest.raises(TypeError, match="params"):
        validate_optimizer_class("broken", MissingParamsOptimizer)


def test_validate_optimizer_class_rejects_runtime_incompatible_optimizer() -> None:
    class RuntimeIncompatibleOptimizer(torch.optim.Optimizer):
        def __init__(self, params, lr: float = 1e-3) -> None:
            raise RuntimeError("bad optimizer runtime")

        def step(self, closure=None):  # pragma: no cover - never executed
            return None

    with pytest.raises(TypeError, match="failed compatibility validation"):
        validate_optimizer_class("broken", RuntimeIncompatibleOptimizer)
