import torch

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoTimeXer, TimeXer
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import FREQ, Y_TEST_DF_1, Y_TRAIN_DF_1

from .test_helpers import check_args


def _make_model(**overrides):
    params = {
        "h": 2,
        "input_size": 4,
        "n_series": 1,
        "max_steps": 1,
        "learning_rate": 1e-3,
        "batch_size": 1,
        "valid_batch_size": 1,
        "windows_batch_size": 1,
        "inference_windows_batch_size": 1,
        "futr_exog_list": ["futr_a"],
        "patch_len": 2,
        "hidden_size": 4,
        "n_heads": 1,
        "e_layers": 1,
        "d_ff": 8,
    }
    params.update(overrides)
    return TimeXer(**params)


def test_timexer(suppress_warnings):
    model = TimeXer(
        h=7,
        input_size=28,
        n_series=1,
        patch_len=7,
        hidden_size=8,
        n_heads=1,
        e_layers=1,
        d_ff=16,
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


def test_autotimxer(setup_dataset):

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoTimeXer, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoTimeXer.get_default_config(h=12, n_series=1, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update(
            {
                'max_steps': 1,
                'val_check_steps': 1,
                'input_size': 12,
                'patch_len': 12,
                'accelerator': 'cpu',
                'devices': 1,
            }
        )
        return config

    model = AutoTimeXer(
        h=12,
        n_series=1,
        config=my_config_new,
        backend='optuna',
        num_samples=1,
        cpus=1,
        gpus=0,
    )
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoTimeXer.get_default_config(h=12, n_series=1, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['patch_len'] = 12
    my_config['accelerator'] = 'cpu'
    my_config['devices'] = 1
    model = AutoTimeXer(
        h=12,
        n_series=1,
        config=my_config,
        backend='ray',
        num_samples=1,
        cpus=1,
        gpus=0,
    )
    model.fit(dataset=setup_dataset)


def test_timexer_forward_adds_future_exog_tokens():
    model = _make_model()

    class ZeroEmbedding(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size

        def forward(self, x, x_mark):
            tokens = x.shape[2] + (0 if x_mark is None else x_mark.shape[2])
            return torch.zeros(x.shape[0], tokens, self.hidden_size)

    class RecordingEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_cross = None

        def forward(self, x, cross, **kwargs):
            self.last_cross = cross
            return x

    model.ex_embedding = ZeroEmbedding(model.hidden_size)
    recorder = RecordingEncoder()
    model.encoder = recorder
    with torch.no_grad():
        model.futr_exog_embedding.weight.fill_(1.0)
        model.futr_exog_embedding.bias.zero_()

    windows_batch = {
        "insample_y": torch.zeros(1, 4, 1),
        "hist_exog": torch.empty(1, 0, 4, 1),
        "futr_exog": torch.tensor([[[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]]]),
        "stat_exog": torch.empty(1, 0),
    }

    y_pred = model(windows_batch)

    assert y_pred.shape == (1, 2, 1)
    assert recorder.last_cross is not None
    assert recorder.last_cross.shape == (1, 2, model.hidden_size)
    assert torch.allclose(recorder.last_cross[:, 0, :], torch.zeros(1, model.hidden_size))
    assert torch.allclose(
        recorder.last_cross[:, 1, :],
        torch.full((1, model.hidden_size), 21.0),
    )


def test_timexer_smoke_with_future_exog(longer_horizon_test):
    model = TimeXer(
        h=longer_horizon_test.h,
        input_size=longer_horizon_test.input_size,
        n_series=longer_horizon_test.n_series,
        futr_exog_list=longer_horizon_test.futr_exog_list,
        patch_len=2,
        hidden_size=8,
        n_heads=1,
        e_layers=1,
        d_ff=16,
        max_steps=1,
        val_check_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        batch_size=1,
        valid_batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
    )
    nf = NeuralForecast(models=[model], freq="ME")
    nf.fit(df=longer_horizon_test.train_df)

    forecasts = nf.predict(futr_df=longer_horizon_test.test_df)

    assert not forecasts.empty
