from neuralforecast.auto import AutoTimeMixer, TimeMixer
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
import torch

from .test_helpers import check_args


def test_time_mixer_model(suppress_warnings):
    check_model(TimeMixer, ["airpassengers"])

def test_autotimemixer(setup_dataset):
    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoTimeMixer, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoTimeMixer.get_default_config(h=12, n_series=1, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12, 'd_model': 16})
        return config

    model = AutoTimeMixer(h=12, n_series=1, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoTimeMixer.get_default_config(h=12, n_series=1, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['d_model'] = 16
    model = AutoTimeMixer(h=12, n_series=1, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=setup_dataset)


def test_timemixer_forward_uses_hist_exog():
    model = TimeMixer(
        h=2,
        input_size=4,
        n_series=1,
        max_steps=1,
        max_lr=1e-3,
        batch_size=1,
        valid_batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
        hist_exog_list=["hist_a"],
        d_model=4,
        d_ff=8,
        e_layers=1,
        down_sampling_layers=1,
    )

    model.forecast = lambda x, x_mark_enc, x_mark_dec: torch.zeros(
        x.shape[0], model.h, model.n_series
    )
    model.hist_exog_projection.weight.data.fill_(1.0)
    model.hist_exog_projection.bias.data.zero_()

    windows_batch = {
        "insample_y": torch.zeros(1, 4, 1),
        "futr_exog": None,
        "hist_exog": torch.ones(1, 1, 4, 1),
    }

    y_pred = model(windows_batch)

    assert y_pred.shape == (1, 2, 1)
    assert torch.equal(y_pred, torch.full((1, 2, 1), 4.0))
