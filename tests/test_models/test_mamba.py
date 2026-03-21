import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoMamba
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import FREQ, Y_TEST_DF_1, Y_TRAIN_DF_1
from neuralforecast.models.mamba import Mamba

from .test_helpers import check_args


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


def test_automamba(setup_dataset):
    check_args(AutoMamba, exclude_args=["cls_model"])

    my_config = AutoMamba.get_default_config(h=12, backend="optuna")

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

    model = AutoMamba(
        h=12, config=my_config_new, backend="optuna", num_samples=1, cpus=1, gpus=0
    )
    assert model.config(MockTrial())["h"] == 12
    model.fit(dataset=setup_dataset)

    my_config = AutoMamba.get_default_config(h=12, backend="ray")
    my_config["max_steps"] = 1
    my_config["val_check_steps"] = 1
    my_config["input_size"] = 12
    my_config["hidden_size"] = 16
    my_config["n_block"] = 1
    my_config["accelerator"] = "cpu"
    my_config["devices"] = 1
    model = AutoMamba(
        h=12, config=my_config, backend="ray", num_samples=1, cpus=1, gpus=0
    )
    assert model.config["input_size"] == 12
