from torch import nn

from neuralforecast import NeuralForecast
from neuralforecast.models.xlstm_mixer import xLSTMMixer
from neuralforecast.utils import AirPassengersPanel


def test_xlstm_mixer_importerror_path(monkeypatch):
    monkeypatch.setattr("neuralforecast.models.xlstm_mixer.IS_XLSTM_INSTALLED", False)
    try:
        xLSTMMixer(h=12, input_size=24, n_series=2)
    except ImportError as exc:
        assert "Please install `xlstm`" in str(exc)
    else:
        raise AssertionError("xLSTMMixer should raise ImportError when xlstm is unavailable")


class _FakeConfig:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


class _FakeStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.kwargs["embedding_dim"])
        self.ffn = nn.Sequential(
            nn.Linear(config.kwargs["embedding_dim"], config.kwargs["embedding_dim"]),
            nn.GELU(),
            nn.Linear(config.kwargs["embedding_dim"], config.kwargs["embedding_dim"]),
        )

    def forward(self, x):
        return self.norm(x + self.ffn(x))


def test_xlstm_mixer_multivariate_smoke(monkeypatch):
    monkeypatch.setattr("neuralforecast.models.xlstm_mixer.IS_XLSTM_INSTALLED", True)
    monkeypatch.setattr("neuralforecast.models.xlstm_mixer.mLSTMBlockConfig", _FakeConfig)
    monkeypatch.setattr("neuralforecast.models.xlstm_mixer.sLSTMBlockConfig", _FakeConfig)
    monkeypatch.setattr("neuralforecast.models.xlstm_mixer.xLSTMBlockStackConfig", _FakeConfig)
    monkeypatch.setattr("neuralforecast.models.xlstm_mixer.xLSTMBlockStack", _FakeStack)

    train_df = AirPassengersPanel[AirPassengersPanel.ds < AirPassengersPanel.ds.values[-12]]
    test_df = AirPassengersPanel[AirPassengersPanel.ds >= AirPassengersPanel.ds.values[-12]].reset_index(drop=True)

    model = xLSTMMixer(
        h=12,
        input_size=24,
        n_series=train_df["unique_id"].nunique(),
        hidden_size=16,
        ff_dim=32,
        encoder_n_blocks=1,
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

    cv = nf.cross_validation(df=AirPassengersPanel, n_windows=1, step_size=12)
    assert not cv.empty
