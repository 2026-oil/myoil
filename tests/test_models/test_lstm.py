import torch
import torch.nn as nn

from neuralforecast.auto import AutoLSTM
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import (
    AirPassengersPanel,
    AirPassengersStatic,
)
from neuralforecast.core import NeuralForecast
from neuralforecast.models import LSTM

from .test_helpers import check_args


def test_lstm_model(suppress_warnings):
    h = 12
    train_df = AirPassengersPanel[
        AirPassengersPanel.ds < AirPassengersPanel["ds"].values[-h]
    ]
    test_df = AirPassengersPanel[
        AirPassengersPanel.ds >= AirPassengersPanel["ds"].values[-h]
    ].reset_index(drop=True)

    config = {
        "h": h,
        "input_size": 24,
        "max_steps": 2,
        "val_check_steps": 2,
        "batch_size": 8,
        "windows_batch_size": 8,
        "valid_batch_size": 8,
        "inference_windows_batch_size": 8,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "accelerator": "cpu",
        "devices": 1,
    }

    fcst = NeuralForecast(models=[LSTM(**config)], freq="M")
    fcst.fit(df=train_df, static_df=AirPassengersStatic)
    fcst.predict(futr_df=test_df)

    fcst = NeuralForecast(models=[LSTM(**config)], freq="M")
    fcst.cross_validation(
        df=AirPassengersPanel,
        static_df=AirPassengersStatic,
        n_windows=2,
        step_size=12,
    )


def test_autolstm_model(setup_dataset, monkeypatch):
    dataset = setup_dataset
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoLSTM, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoLSTM.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update(
            {
                'max_steps': 1,
                'val_check_steps': 1,
                'input_size': -1,
                'encoder_hidden_size': 8,
                'accelerator': 'cpu',
                'devices': 1,
            }
        )
        return config

    model = AutoLSTM(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoLSTM.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = -1
    my_config['encoder_hidden_size'] = 8
    my_config['accelerator'] = 'cpu'
    my_config['devices'] = 1
    model = AutoLSTM(h=12, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=dataset)


class _DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x, state=None):
        batch_size, seq_len = x.shape[:2]
        return torch.ones((batch_size, seq_len, self.hidden_size), device=x.device), None


def test_lstm_direct_head_uses_horizon_identity():
    model = LSTM(
        h=4,
        input_size=8,
        max_steps=1,
        learning_rate=1e-3,
        batch_size=1,
        valid_batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
        encoder_hidden_size=8,
        decoder_hidden_size=8,
        decoder_layers=2,
        recurrent=False,
    )

    model.hist_encoder = _DummyEncoder(model.encoder_hidden_size)
    model.mlp_decoder = nn.Linear(
        model.encoder_hidden_size + model.horizon_embedding_dim,
        1,
        bias=False,
    )

    with torch.no_grad():
        model.mlp_decoder.weight.zero_()
        model.mlp_decoder.weight[0, -model.horizon_embedding_dim :] = 1.0
        for horizon_idx in range(model.h):
            model.horizon_embeddings.weight[horizon_idx].fill_(float(horizon_idx))

    windows_batch = {
        "insample_y": torch.ones(2, 8, 1),
        "futr_exog": torch.zeros(2, 12, 0),
        "hist_exog": torch.zeros(2, 8, 0),
        "stat_exog": torch.zeros(2, 0),
    }

    y_hat = model(windows_batch).squeeze(-1)

    assert y_hat.shape == (2, 4)
    assert torch.unique(y_hat[0]).numel() == model.h


def test_autolstm_default_config_excludes_deprecated_context_size():
    optuna_config = AutoLSTM.get_default_config(h=12, backend='optuna')
    assert 'context_size' not in optuna_config(MockTrial())

    ray_config = AutoLSTM.get_default_config(h=12, backend='ray')
    assert 'context_size' not in ray_config
