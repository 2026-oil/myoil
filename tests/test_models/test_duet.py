from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoDUET
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.models.duet import DUET
from neuralforecast.utils import AirPassengersPanel

from .test_helpers import check_args


def _run_duet_smoke(n_series):
    panel = AirPassengersPanel.copy()
    if n_series == 1:
        panel = panel[panel["unique_id"] == panel["unique_id"].iloc[0]].copy()

    train_df = panel[panel.ds < panel.ds.values[-12]]
    test_df = panel[panel.ds >= panel.ds.values[-12]].reset_index(drop=True)

    model = DUET(
        h=12,
        input_size=24,
        n_series=train_df["unique_id"].nunique(),
        hidden_size=16,
        ff_dim=32,
        n_block=1,
        max_steps=1,
        val_check_steps=1,
        batch_size=2,
        valid_batch_size=2,
        windows_batch_size=8,
        inference_windows_batch_size=8,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    nf = NeuralForecast(models=[model], freq="M")
    nf.fit(df=train_df, val_size=12)
    forecasts = nf.predict(futr_df=test_df)
    assert not forecasts.empty
    cv = nf.cross_validation(df=panel, n_windows=1, step_size=12)
    assert not cv.empty


def test_duet_multivariate_smoke_n_series_1():
    _run_duet_smoke(n_series=1)



def test_duet_multivariate_smoke_n_series_5():
    _run_duet_smoke(n_series=5)


def test_autoduet(setup_dataset):
    check_args(AutoDUET, exclude_args=["cls_model"])

    my_config = AutoDUET.get_default_config(h=12, n_series=1, backend="optuna")

    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update(
            {
                "max_steps": 1,
                "val_check_steps": 1,
                "input_size": 12,
                "hidden_size": 16,
                "ff_dim": 32,
                "n_block": 1,
                "accelerator": "cpu",
                "devices": 1,
            }
        )
        return config

    model = AutoDUET(
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

    my_config = AutoDUET.get_default_config(h=12, n_series=1, backend="ray")
    my_config["max_steps"] = 1
    my_config["val_check_steps"] = 1
    my_config["input_size"] = 12
    my_config["hidden_size"] = 16
    my_config["ff_dim"] = 32
    my_config["n_block"] = 1
    my_config["accelerator"] = "cpu"
    my_config["devices"] = 1
    model = AutoDUET(
        h=12,
        n_series=1,
        config=my_config,
        backend="ray",
        num_samples=1,
        cpus=1,
        gpus=0,
    )
    assert model.config["n_series"] == 1
