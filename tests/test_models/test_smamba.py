import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.common._model_checks import FREQ, Y_TEST_DF_1, Y_TEST_DF_2, Y_TRAIN_DF_1, Y_TRAIN_DF_2
from neuralforecast.models.smamba import SMamba


def _run_smoke(n_series):
    train_df = Y_TRAIN_DF_1 if n_series == 1 else Y_TRAIN_DF_2
    test_df = Y_TEST_DF_1 if n_series == 1 else Y_TEST_DF_2
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    model = SMamba(
        h=7,
        input_size=28,
        n_series=n_series,
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
    nf.fit(df=train_df, val_size=7)
    preds = nf.predict(futr_df=test_df)
    assert not preds.empty

    cv = nf.cross_validation(df=full_df, n_windows=1, step_size=7)
    assert not cv.empty


def test_smamba_n_series_1(suppress_warnings):
    _run_smoke(n_series=1)


def test_smamba_n_series_5(suppress_warnings):
    _run_smoke(n_series=5)
