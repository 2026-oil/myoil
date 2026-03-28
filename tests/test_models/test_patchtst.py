import torch
from neuralforecast.auto import AutoPatchTST, PatchTST
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model

from .test_helpers import check_args


def test_patchtst_model(suppress_warnings):
    check_model(PatchTST, ["airpassengers"])

def test_autopatchtst(setup_dataset):
    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoPatchTST, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoPatchTST.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12, 'hidden_size': 16})
        return config

    model = AutoPatchTST(h=12, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoPatchTST.get_default_config(h=12, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['hidden_size'] = 16
    model = AutoPatchTST(h=12, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=setup_dataset)


def test_patchtst_forward_concatenates_hist_exog():
    model = PatchTST(
        h=2,
        input_size=4,
        max_steps=1,
        max_lr=1e-3,
        batch_size=1,
        valid_batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
        hist_exog_list=["hist_a"],
        patch_len=2,
        stride=1,
        hidden_size=4,
        linear_hidden_size=8,
        n_heads=1,
        encoder_layers=1,
    )

    class Recorder(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h
            self.last_input = None

        def forward(self, x):
            self.last_input = x
            return x[:, :, -1:].repeat(1, 1, self.h)

    recorder = Recorder(h=2)
    model.model = recorder
    windows_batch = {
        "insample_y": torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        "hist_exog": torch.tensor([[[10.0], [11.0], [12.0], [13.0]]]),
    }

    forecast = model(windows_batch)

    assert recorder.last_input.shape == (1, 2, 4)
    assert torch.equal(recorder.last_input[0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(recorder.last_input[0, 1], torch.tensor([10.0, 11.0, 12.0, 13.0]))
    assert forecast.shape == (1, 2, 1)
    assert torch.equal(forecast[:, :, 0], torch.tensor([[4.0, 4.0]]))
