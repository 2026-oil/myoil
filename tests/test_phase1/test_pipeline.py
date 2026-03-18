from __future__ import annotations

from pathlib import Path

import pandas as pd

from phase1.pipeline import (
    ALL_MODEL_NAMES,
    Phase1Config,
    _resolved_gpu_devices,
    _trainer_strategy,
    baseline_cross_validation,
    baseline_holdout_predictions,
    build_learned_model,
    compute_metrics,
    ensure_gpu_policy,
    infer_frequency,
    make_target_frame,
    rank_leaderboard,
    split_final_holdout,
)


def _sample_source_frame() -> pd.DataFrame:
    periods = 150
    ds = pd.date_range("2020-01-06", periods=periods, freq="W-MON")
    return pd.DataFrame(
        {
            "dt": ds,
            "Com_CrudeOil": range(periods),
            "Com_BrentCrudeOil": [value * 2 for value in range(periods)],
        }
    )


def test_infer_frequency_weekly_monday() -> None:
    df = _sample_source_frame()
    assert infer_frequency(df) == "W-MON"


def test_make_target_frame_and_holdout_split() -> None:
    source = _sample_source_frame()
    target_df = make_target_frame(source, "Com_CrudeOil")
    train_df, holdout_df = split_final_holdout(target_df, 12)
    assert list(target_df.columns) == ["unique_id", "ds", "y"]
    assert len(train_df) == 138
    assert len(holdout_df) == 12
    assert holdout_df["ds"].iloc[0] == source["dt"].iloc[-12]


def test_compute_metrics_expected_values() -> None:
    metrics = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert metrics["RMSE"] == (4.0 / 3.0) ** 0.5
    assert metrics["MAE"] == 2.0 / 3.0
    assert metrics["MAPE"] > 0
    assert metrics["NRMSE"] > 0


def test_baseline_cross_validation_and_holdout_shapes(tmp_path: Path) -> None:
    source = _sample_source_frame()
    target_df = make_target_frame(source, "Com_CrudeOil")
    train_df, holdout_df = split_final_holdout(target_df, 12)
    config = Phase1Config(
        data_path=tmp_path / "dummy.csv",
        output_root=tmp_path / "artifacts",
        targets=["Com_CrudeOil"],
        model_names=["Naive", "SeasonalNaive", "HistoricAverage"],
        input_size=24,
    )
    cv_metrics, cv_predictions = baseline_cross_validation(train_df, config, "Com_CrudeOil")
    holdout_metrics, holdout_predictions = baseline_holdout_predictions(
        train_df, holdout_df, config, "Com_CrudeOil"
    )
    assert set(cv_metrics["model"]) == {"Naive", "SeasonalNaive", "HistoricAverage"}
    assert len(cv_predictions) == config.n_windows * config.horizon * 3
    assert len(holdout_predictions) == config.horizon * 3
    assert len(holdout_metrics) == 3


def test_rank_leaderboard_orders_by_average_rank() -> None:
    leaderboard = rank_leaderboard(
        pd.DataFrame(
            [
                {"target": "Com_CrudeOil", "model": "a", "RMSE": 1.0, "MAE": 1.0, "MAPE": 1.0, "NRMSE": 1.0},
                {"target": "Com_CrudeOil", "model": "b", "RMSE": 2.0, "MAE": 2.0, "MAPE": 2.0, "NRMSE": 2.0},
            ]
        )
    )
    assert leaderboard.iloc[0]["model"] == "a"


def test_build_learned_model_supports_nhits(tmp_path: Path) -> None:
    config = Phase1Config(
        data_path=tmp_path / "dummy.csv",
        output_root=tmp_path / "artifacts",
        model_names=["NHITS"],
        input_size=24,
        max_steps=2,
        val_check_steps=1,
        early_stop_patience_steps=1,
    )

    model = build_learned_model("NHITS", gpu_devices=1, config=config)

    assert "NHITS" in ALL_MODEL_NAMES
    assert model.__class__.__name__ == "NHITS"
    assert "dropout" not in model.trainer_kwargs


def test_trainer_strategy_uses_explicit_backend_for_multi_gpu(monkeypatch) -> None:
    monkeypatch.setattr("phase1.pipeline.torch.cuda.is_available", lambda: True)
    strategy = _trainer_strategy(gpu_devices=2)
    assert strategy.__class__.__name__ == "DDPStrategy"
    assert getattr(strategy, "_process_group_backend", None) == "gloo"


def test_resolved_gpu_devices_downgrades_fedformer_on_gloo(monkeypatch) -> None:
    monkeypatch.setattr("phase1.pipeline.torch.cuda.is_available", lambda: True)
    assert _resolved_gpu_devices("FEDformer", gpu_devices=2) == 1
    assert _resolved_gpu_devices("TFT", gpu_devices=2) == 2


def test_ensure_gpu_policy_defaults_to_single_gpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("phase1.pipeline.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("phase1.pipeline.torch.cuda.device_count", lambda: 2)
    config = Phase1Config(
        data_path=tmp_path / "dummy.csv",
        output_root=tmp_path / "artifacts",
    )

    assert ensure_gpu_policy(config) == 1
