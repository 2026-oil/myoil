import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.common._model_checks import FREQ, Y_TEST_DF_1, Y_TRAIN_DF_1
from neuralforecast.models.mamba import Mamba


def test_mamba_model(suppress_warnings):
    full_df = pd.concat([Y_TRAIN_DF_1, Y_TEST_DF_1], ignore_index=True)
    model = Mamba(
        h=7,
        input_size=28,
        max_steps=1,
        val_check_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        batch_size=8,
        valid_batch_size=8,
        windows_batch_size=8,
        inference_windows_batch_size=8,
    )
    nf = NeuralForecast(models=[model], freq=FREQ)
    nf.fit(df=Y_TRAIN_DF_1, val_size=7)
    preds = nf.predict(futr_df=Y_TEST_DF_1)
    assert not preds.empty

    cv = nf.cross_validation(df=full_df, n_windows=1, step_size=7)
    assert not cv.empty
