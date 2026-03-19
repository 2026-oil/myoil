from neuralforecast import NeuralForecast
from neuralforecast.models.duet import DUET
from neuralforecast.utils import AirPassengersPanel


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
