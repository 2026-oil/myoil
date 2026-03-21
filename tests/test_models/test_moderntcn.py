import torch
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.common._model_checks import (
    FREQ,
    Y_TEST_DF_2,
    Y_TRAIN_DF_2,
)
from neuralforecast.models.moderntcn import ModernTCN


def test_moderntcn_model(suppress_warnings):
    full_df = pd.concat([Y_TRAIN_DF_2, Y_TEST_DF_2], ignore_index=True)
    model = ModernTCN(
        h=7,
        input_size=28,
        n_series=5,
        max_steps=1,
        val_check_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        patch_size=8,
        patch_stride=4,
        num_blocks=(1, 1, 1, 1),
        large_size=(5, 5, 3, 3),
        small_size=(3, 3, 3, 3),
        dims=(8, 8, 8, 8),
        batch_size=8,
        valid_batch_size=8,
        windows_batch_size=8,
        inference_windows_batch_size=8,
    )
    nf = NeuralForecast(models=[model], freq=FREQ)
    nf.fit(df=Y_TRAIN_DF_2, val_size=7)
    preds = nf.predict(futr_df=Y_TEST_DF_2)
    assert not preds.empty

    cv = nf.cross_validation(df=full_df, n_windows=1, step_size=7)
    assert not cv.empty


def test_moderntcn_forward_shape():
    model = ModernTCN(
        h=7,
        input_size=28,
        n_series=5,
        max_steps=1,
        val_check_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        patch_size=8,
        patch_stride=4,
        num_blocks=(1, 1, 1, 1),
        large_size=(5, 5, 3, 3),
        small_size=(3, 3, 3, 3),
        dims=(8, 8, 8, 8),
        batch_size=8,
        valid_batch_size=8,
        windows_batch_size=8,
        inference_windows_batch_size=8,
        decomposition=True,
        individual=True,
        kernel_size=3,
    )
    windows_batch = {"insample_y": torch.randn(2, 28, 5)}
    preds = model(windows_batch)
    assert preds.shape == (2, 7, 5)
