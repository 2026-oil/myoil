import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoSMamba
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import FREQ, Y_TEST_DF_1, Y_TEST_DF_2, Y_TRAIN_DF_1, Y_TRAIN_DF_2
from neuralforecast.models.smamba import SMamba

from .test_helpers import check_args


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


def test_autosmamba(setup_dataset):
    check_args(AutoSMamba, exclude_args=["cls_model"])

    my_config = AutoSMamba.get_default_config(h=12, n_series=1, backend="optuna")

    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update(
            {
                "max_steps": 1,
                "val_check_steps": 1,
                "input_size": 12,
                "hidden_size": 16,
                "n_block": 1,
                "accelerator": "cpu",
                "devices": 1,
            }
        )
        return config

    model = AutoSMamba(
        h=12,
        n_series=1,
        config=my_config_new,
        backend="optuna",
        num_samples=1,
        cpus=1,
        gpus=0,
    )
    assert model.config(MockTrial())["h"] == 12
    assert model.config(MockTrial())["n_series"] == 1
    model.fit(dataset=setup_dataset)

    my_config = AutoSMamba.get_default_config(h=12, n_series=1, backend="ray")
    my_config["max_steps"] = 1
    my_config["val_check_steps"] = 1
    my_config["input_size"] = 12
    my_config["hidden_size"] = 16
    my_config["n_block"] = 1
    my_config["accelerator"] = "cpu"
    my_config["devices"] = 1
    model = AutoSMamba(
        h=12,
        n_series=1,
        config=my_config,
        backend="ray",
        num_samples=1,
        cpus=1,
        gpus=0,
    )
    assert model.config["n_series"] == 1
