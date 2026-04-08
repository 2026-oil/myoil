from __future__ import annotations

from pathlib import Path

import pandas as pd

from app_config import load_app_config
from neuralforecast.tsdataset import TimeSeriesDataModule, TimeSeriesDataset
from plugins.aa_forecast.runtime import _aa_params_override
from runtime_support.forecast_models import build_model
from runtime_support.runner import (
    _build_adapter_inputs,
    _build_fold_diff_context,
    _build_tscv_splits,
    _effective_config,
    _transform_training_frame,
)


def test_aaforecast_single_batch_loader_repeats_by_available_windows() -> None:
    repo_root = Path.cwd()
    loaded = load_app_config(
        repo_root,
        config_path="yaml/experiment/feature_set_aaforecast/aa_forecast_brentoil.yaml",
    )
    source_df = pd.read_csv(loaded.config.dataset.path).sort_values(
        loaded.config.dataset.dt_col
    )
    source_df = source_df.reset_index(drop=True)
    train_idx, test_idx = _build_tscv_splits(len(source_df), loaded.config.cv)[0]
    train_df = source_df.iloc[train_idx].reset_index(drop=True)
    future_df = source_df.iloc[test_idx].reset_index(drop=True)
    effective = _effective_config(
        loaded,
        {
            "input_size": 64,
            "batch_size": 16,
            "scaler_type": "standard",
            "model_step_size": 8,
        },
    )
    diff_context = _build_fold_diff_context(loaded, train_df)
    transformed_train_df = _transform_training_frame(train_df, diff_context)
    adapter_inputs = _build_adapter_inputs(
        loaded,
        transformed_train_df,
        future_df,
        loaded.config.jobs[0],
        loaded.config.dataset.dt_col,
    )
    dataset, *_ = TimeSeriesDataset.from_df(
        adapter_inputs.fit_df, static_df=adapter_inputs.static_df
    )
    datamodule = TimeSeriesDataModule(
        dataset=dataset,
        batch_size=effective.training.batch_size,
        valid_batch_size=effective.training.valid_batch_size,
        drop_last=False,
        shuffle_train=True,
    )
    assert len(datamodule.train_dataloader()) == 1

    model = build_model(
        effective,
        loaded.config.jobs[0],
        params_override=_aa_params_override(effective),
    )
    assert model.trainer_kwargs["enable_model_summary"] is False
    assert model.trainer_kwargs["enable_progress_bar"] is False
    assert model.trainer_kwargs["logger"] is False
    batch = next(iter(datamodule.train_dataloader()))
    _, _, _, final_condition = model._create_windows(batch, step="train")
    expected_window_count = len(final_condition)
    train_batches_per_epoch = model._window_based_train_batches_per_epoch(datamodule)

    assert train_batches_per_epoch == expected_window_count
    assert train_batches_per_epoch > 1

    model._configure_window_based_train_dataloader(datamodule)
    loader = datamodule.train_dataloader()

    assert len(loader) == expected_window_count
    assert sum(1 for _ in loader) == expected_window_count
