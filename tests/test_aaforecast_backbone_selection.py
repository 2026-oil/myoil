from __future__ import annotations

import torch
import pytest

from neuralforecast.models import AAForecast


def _windows_batch(batch_size: int = 2, input_size: int = 4) -> dict[str, torch.Tensor]:
    insample_y = torch.arange(batch_size * input_size, dtype=torch.float32).reshape(
        batch_size, input_size, 1
    )
    hist_exog = torch.linspace(
        0.0, 1.0, steps=batch_size * input_size, dtype=torch.float32
    ).reshape(batch_size, input_size, 1)
    return {
        "insample_y": insample_y,
        "hist_exog": hist_exog,
        "futr_exog": torch.empty(batch_size, input_size, 0),
        "stat_exog": torch.empty(batch_size, 0),
    }


@pytest.mark.parametrize(
    ("backbone", "params"),
    [
        (
            "gru",
            {
                "encoder_hidden_size": 8,
                "encoder_n_layers": 2,
                "encoder_dropout": 0.1,
            },
        ),
        (
            "vanillatransformer",
            {
                "hidden_size": 8,
                "n_head": 2,
                "encoder_layers": 1,
                "dropout": 0.1,
                "linear_hidden_size": 16,
            },
        ),
        (
            "informer",
            {
                "hidden_size": 8,
                "n_head": 2,
                "encoder_layers": 1,
                "dropout": 0.1,
                "linear_hidden_size": 16,
                "factor": 1,
            },
        ),
        (
            "patchtst",
            {
                "hidden_size": 8,
                "n_heads": 2,
                "encoder_layers": 1,
                "dropout": 0.1,
                "linear_hidden_size": 16,
                "attn_dropout": 0.0,
                "patch_len": 2,
                "stride": 1,
            },
        ),
        (
            "timexer",
            {
                "hidden_size": 8,
                "n_heads": 2,
                "e_layers": 1,
                "dropout": 0.1,
                "d_ff": 16,
                "patch_len": 2,
                "use_norm": True,
            },
        ),
    ],
)
def test_aaforecast_supported_backbones_match_forward_contract(
    backbone: str,
    params: dict[str, object],
) -> None:
    model = AAForecast(
        h=2,
        input_size=4,
        backbone=backbone,
        hist_exog_list=["event"],
        star_hist_exog_list=["event"],
        non_star_hist_exog_list=[],
        star_hist_exog_tail_modes=["upward"],
        scaler_type="identity",
        max_steps=1,
        val_check_steps=1,
        batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
        **params,
    )
    model.eval()

    outputs = model(_windows_batch())

    assert outputs.shape == (2, 2, 1)
