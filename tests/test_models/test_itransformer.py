from neuralforecast.auto import AutoiTransformer, iTransformer
from neuralforecast.common._base_auto import MockTrial
from neuralforecast.common._model_checks import check_model
import torch

from .test_helpers import check_args


def test_itransformer_model(suppress_warnings):
    check_model(iTransformer, ["airpassengers"])

def test_autoitransformer(setup_dataset):

    # Unit test to test that Auto* model contains all required arguments from BaseAuto
    check_args(AutoiTransformer, exclude_args=['cls_model'])

    # Unit test for situation: Optuna with updated default config
    my_config = AutoiTransformer.get_default_config(h=12, n_series=1, backend='optuna')
    def my_config_new(trial):
        config = {**my_config(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': 12, 'hidden_size': 16})
        return config

    model = AutoiTransformer(h=12, n_series=1, config=my_config_new, backend='optuna', num_samples=1, cpus=1)
    assert model.config(MockTrial())['h'] == 12
    model.fit(dataset=setup_dataset)

    # Unit test for situation: Ray with updated default config
    my_config = AutoiTransformer.get_default_config(h=12, n_series=1, backend='ray')
    my_config['max_steps'] = 1
    my_config['val_check_steps'] = 1
    my_config['input_size'] = 12
    my_config['hidden_size'] = 16
    model = AutoiTransformer(h=12, n_series=1, config=my_config, backend='ray', num_samples=1, cpus=1)
    model.fit(dataset=setup_dataset)


def test_itransformer_forward_uses_hist_exog():
    model = iTransformer(
        h=2,
        input_size=4,
        n_series=1,
        max_steps=1,
        learning_rate=1e-3,
        batch_size=1,
        valid_batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
        hist_exog_list=["hist_a"],
        hidden_size=4,
        n_heads=1,
        e_layers=1,
        d_ff=8,
    )

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
