from neuralforecast.auto import AutoDLinear, DLinear
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
import torch

from .test_helpers import check_args


def test_dlinear(suppress_warnings):
    check_model(DLinear, ["airpassengers"])

def test_autodlinear(setup_dataset):
    dataset = setup_dataset

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoDLinear, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoDLinear.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 2, 'val_check_steps': 1, 'input_size': 12})
        return config

    model = AutoDLinear(h=12, config=my_config_new, backend='optuna', cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoDLinear.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 2
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    model = AutoDLinear(h=12, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=dataset)


def test_dlinear_forward_uses_hist_exog():
    model = DLinear(
        h=2,
        input_size=4,
        max_steps=1,
        max_lr=1e-3,
        batch_size=1,
        valid_batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
        hist_exog_list=["hist_a"],
        moving_avg_window=3,
    )

    model.linear_trend.weight.data.zero_()
    model.linear_trend.bias.data.zero_()
    model.linear_season.weight.data.zero_()
    model.linear_season.bias.data.zero_()
    model.hist_exog_projection.weight.data.fill_(1.0)
    model.hist_exog_projection.bias.data.zero_()

    windows_batch = {
        "insample_y": torch.zeros(1, 4, 1),
        "hist_exog": torch.ones(1, 4, 1),
    }

    y_pred = model(windows_batch)

    assert y_pred.shape == (1, 2, 1)
    assert torch.equal(y_pred, torch.full((1, 2, 1), 4.0))
