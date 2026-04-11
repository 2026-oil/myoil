from __future__ import annotations

import pytest
import torch

from neuralforecast.models import AAForecast
from neuralforecast.models.informer import InformerEncoderOnly
from neuralforecast.models.itransformer import ITransformerTokenEncoderOnly
from neuralforecast.models.patchtst import PatchTSTEncoderOnly
from neuralforecast.models.timexer import TimeXerEncoderOnly
from neuralforecast.models.vanillatransformer import VanillaTransformerEncoderOnly


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


BACKBONE_CASES = [
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
        PatchTSTEncoderOnly,
        "patch-to-time bridge",
    ),
    (
        "timexer",
        {
            "hidden_size": 8,
            "n_heads": 2,
            "e_layers": 1,
            "dropout": 0.1,
            "d_ff": 16,
            "factor": 1,
            "patch_len": 2,
            "use_norm": True,
        },
        TimeXerEncoderOnly,
        "global token",
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
        VanillaTransformerEncoderOnly,
        "decoder/head",
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
        InformerEncoderOnly,
        "distillation",
    ),
    (
        "itransformer",
        {
            "hidden_size": 8,
            "n_heads": 2,
            "e_layers": 1,
            "dropout": 0.1,
            "d_ff": 16,
            "factor": 1,
            "use_norm": True,
        },
        ITransformerTokenEncoderOnly,
        "token-to-time projection",
    ),
]


@pytest.mark.parametrize(("backbone", "params", "helper_cls", "expected_fragment"), BACKBONE_CASES)
def test_aaforecast_backbone_modules_use_per_family_encoder_helpers(
    backbone: str,
    params: dict[str, object],
    helper_cls: type[torch.nn.Module],
    expected_fragment: str,
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

    assert model.encoder.__class__.__module__.startswith(
        "neuralforecast.models.aaforecast.models."
    )
    assert hasattr(model.encoder, "encoder_only")
    assert isinstance(model.encoder.encoder_only, helper_cls)

    evidence = model.encoder.faithfulness_evidence()
    joined = " ".join(evidence.aa_bridge_steps + evidence.unavoidable_divergences)
    assert expected_fragment in joined


@pytest.mark.parametrize(("backbone", "params", "_helper_cls", "_expected_fragment"), BACKBONE_CASES)
def test_aaforecast_non_gru_backbones_preserve_forward_contract_after_module_split(
    backbone: str,
    params: dict[str, object],
    _helper_cls: type[torch.nn.Module],
    _expected_fragment: str,
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
    outputs = model(_windows_batch())
    assert outputs.shape == (2, 2, 1)
