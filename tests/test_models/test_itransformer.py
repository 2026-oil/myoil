import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoiTransformer, iTransformer
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import FREQ, Y_TEST_DF_1, Y_TRAIN_DF_1

from .test_helpers import check_args


def _make_model(**overrides):
    params = {
        "h": 2,
        "input_size": 4,
        "n_series": 1,
        "max_steps": 1,
        "max_lr": 1e-3,
        "batch_size": 1,
        "valid_batch_size": 1,
        "windows_batch_size": 1,
        "inference_windows_batch_size": 1,
        "hist_exog_list": ["hist_a"],
        "hidden_size": 4,
        "n_heads": 1,
        "e_layers": 1,
        "d_ff": 8,
    }
    params.update(overrides)
    return iTransformer(**params)


def test_itransformer_model(suppress_warnings):
    full_df = pd.concat([Y_TRAIN_DF_1, Y_TEST_DF_1], ignore_index=True)
    model = iTransformer(
        h=7,
        input_size=28,
        n_series=1,
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


def test_autoitransformer(setup_dataset):
    check_args(AutoiTransformer, exclude_args=["cls_model"])

    optuna_config = AutoiTransformer.get_default_config(
        h=12, n_series=1, backend="optuna"
    )

    def my_config_new(trial):
        config = {**optuna_config(trial)}
        config.update(
            {
                "max_steps": 1,
                "val_check_steps": 1,
                "input_size": 12,
                "hidden_size": 16,
                "accelerator": "cpu",
                "devices": 1,
            }
        )
        return config

    model = AutoiTransformer(
        h=12,
        n_series=1,
        config=my_config_new,
        backend="optuna",
        num_samples=1,
        cpus=1,
        gpus=0,
    )
    assert model.config(MockTrial())["h"] == 12
    model.fit(dataset=setup_dataset)

    ray_config = AutoiTransformer.get_default_config(h=12, n_series=1, backend="ray")
    ray_config["max_steps"] = 1
    ray_config["val_check_steps"] = 1
    ray_config["input_size"] = 12
    ray_config["hidden_size"] = 16
    ray_config["accelerator"] = "cpu"
    ray_config["devices"] = 1
    model = AutoiTransformer(
        h=12,
        n_series=1,
        config=ray_config,
        backend="ray",
        num_samples=1,
        cpus=1,
        gpus=0,
    )
    model.fit(dataset=setup_dataset)


def test_itransformer_forward_uses_hist_exog():
    model = _make_model()

    class RecorderEmbedding(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.last_x = None
            self.last_x_mark = None

        def forward(self, x, x_mark):
            self.last_x = x
            self.last_x_mark = x_mark
            tokens = x.shape[2] + (0 if x_mark is None else x_mark.shape[2])
            return torch.zeros(x.shape[0], tokens, self.hidden_size)

    class IdentityEncoder(torch.nn.Module):
        def forward(self, x, attn_mask=None):
            return x, None

    recorder = RecorderEmbedding(model.hidden_size)
    model.enc_embedding = recorder
    model.encoder = IdentityEncoder()

    windows_batch = {
        "insample_y": torch.zeros(1, 4, 1),
        "hist_exog": torch.ones(1, 1, 4, 1),
    }

    y_pred = model(windows_batch)

    assert y_pred.shape == (1, 2, 1)
    assert recorder.last_x_mark is not None
    assert recorder.last_x_mark.shape == (1, 4, 1)
    assert torch.isfinite(recorder.last_x_mark).all()
    assert torch.allclose(
        recorder.last_x_mark.mean(dim=1), torch.zeros(1, 1), atol=1e-5
    )


def test_itransformer_forward_normalizes_large_hist_exog_before_embedding():
    model = _make_model()

    class RecorderEmbedding(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.last_x_mark = None

        def forward(self, x, x_mark):
            self.last_x_mark = x_mark
            tokens = x.shape[2] + (0 if x_mark is None else x_mark.shape[2])
            return torch.zeros(x.shape[0], tokens, self.hidden_size)

    class IdentityEncoder(torch.nn.Module):
        def forward(self, x, attn_mask=None):
            return x, None

    recorder = RecorderEmbedding(model.hidden_size)
    model.enc_embedding = recorder
    model.encoder = IdentityEncoder()

    raw_hist_exog = torch.tensor([[[[10_000.0], [20_000.0], [30_000.0], [40_000.0]]]])
    windows_batch = {
        "insample_y": torch.zeros(1, 4, 1),
        "hist_exog": raw_hist_exog,
    }

    model(windows_batch)

    assert recorder.last_x_mark is not None
    assert torch.isfinite(recorder.last_x_mark).all()
    assert recorder.last_x_mark.abs().max() < raw_hist_exog.abs().max()
    assert torch.allclose(
        recorder.last_x_mark.mean(dim=1), torch.zeros(1, 1), atol=1e-5
    )


def test_itransformer_backward_stays_finite_with_large_hist_exog():
    model = _make_model()

    windows_batch = {
        "insample_y": torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        "hist_exog": torch.tensor(
            [[[[10_000.0], [20_000.0], [30_000.0], [40_000.0]]]]
        ),
    }

    output = model(windows_batch)
    loss = output.square().mean()
    loss.backward()

    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_itransformer_forward_raises_on_non_finite_hist_exog():
    model = _make_model()

    windows_batch = {
        "insample_y": torch.zeros(1, 4, 1),
        "hist_exog": torch.tensor([[[[1.0], [float("nan")], [1.0], [1.0]]]]),
    }

    with pytest.raises(ValueError, match="hist_exog"):
        model(windows_batch)


def test_itransformer_forward_raises_on_non_finite_forecast_output():
    model = _make_model()

    def forecast_with_nan(x_enc, x_mark_enc=None):
        return torch.full((x_enc.shape[0], model.h, model.n_series), float("nan"))

    model.forecast = forecast_with_nan
    windows_batch = {
        "insample_y": torch.zeros(1, 4, 1),
        "hist_exog": torch.ones(1, 1, 4, 1),
    }

    with pytest.raises(ValueError, match="y_pred"):
        model(windows_batch)
