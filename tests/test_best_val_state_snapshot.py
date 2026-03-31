import torch
from types import SimpleNamespace

from tests.dummy.dummy_models import DummyUnivariate


def _build_dummy_model() -> DummyUnivariate:
    return DummyUnivariate(
        h=2,
        input_size=4,
        max_steps=10,
        val_check_steps=1,
        min_steps_before_early_stop=500,
    )


def _set_global_step(model: DummyUnivariate, step: int) -> None:
    model._trainer = SimpleNamespace(global_step=step)


def test_best_val_state_snapshot_is_not_blocked_by_min_steps_gate():
    model = _build_dummy_model()
    _set_global_step(model, 10)
    model.w.data.fill_(1.5)

    model._update_best_val_state(2.0)

    assert model._best_val_metric == 2.0
    assert model._best_val_global_step == 10
    assert model._best_val_state_dict is not None
    assert torch.equal(model._best_val_state_dict["w"], torch.tensor([1.5]))


def test_restore_best_val_state_recovers_early_snapshot():
    model = _build_dummy_model()
    _set_global_step(model, 10)
    model.w.data.fill_(1.5)
    model._update_best_val_state(2.0)

    _set_global_step(model, 20)
    model.w.data.fill_(3.0)

    restored = model._restore_best_val_state()

    assert restored is True
    assert torch.equal(model.w.detach(), torch.tensor([1.5]))
