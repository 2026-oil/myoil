from __future__ import annotations

from types import SimpleNamespace

import runtime_support.runner as runner


def _loaded(metric: str | None):
    return SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(tuning_objective_metric=metric)
        )
    )


def test_resolved_tuning_objective_metric_defaults_to_mape() -> None:
    loaded = _loaded(None)

    assert (
        runner._resolved_tuning_objective_metric(loaded)
        == "mean_fold_mape_on_direct_predictions"
    )
    assert runner._tuning_objective_metric_key(loaded) == "MAPE"
    assert runner._tuning_objective_metric_label(loaded) == "mape"


def test_resolved_tuning_objective_metric_supports_mse() -> None:
    loaded = _loaded("mean_fold_mse_on_direct_predictions")

    assert (
        runner._resolved_tuning_objective_metric(loaded)
        == "mean_fold_mse_on_direct_predictions"
    )
    assert runner._tuning_objective_metric_key(loaded) == "MSE"
    assert runner._tuning_objective_metric_label(loaded) == "mse"
