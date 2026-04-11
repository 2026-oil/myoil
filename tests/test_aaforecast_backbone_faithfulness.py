from __future__ import annotations

import pytest
import torch

from neuralforecast.models import AAForecast
from neuralforecast.models.aaforecast.gru import _build_encoder
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
        "late projection",
    ),
]


def _make_model(backbone: str, **params: object) -> AAForecast:
    return AAForecast(
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
    ).eval()


def _encoder_input(model: AAForecast, batch_size: int = 2) -> torch.Tensor:
    feature_size = (
        (0 if model.exclude_insample_y else 1)
        + len(model.non_star_hist_exog_list)
        + 4
        + 4 * len(model.star_hist_exog_list)
    )
    return torch.randn(batch_size, model.input_size, feature_size)


def _build_standalone_helper(model: AAForecast, *, feature_size: int) -> torch.nn.Module:
    encoder = model.encoder
    if model.backbone == "patchtst":
        return PatchTSTEncoderOnly(
            c_in=feature_size,
            input_size=model.input_size,
            patch_len=encoder.patch_len,
            stride=encoder.stride,
            n_layers=model.encoder_layers,
            hidden_size=model.hidden_size,
            n_heads=model.n_heads,
            linear_hidden_size=model.linear_hidden_size or model.hidden_size * 4,
            attn_dropout=model.attn_dropout,
            dropout=model.dropout,
            revin=False,
            padding_patch="end",
        ).eval()
    if model.backbone == "timexer":
        return TimeXerEncoderOnly(
            input_size=model.input_size,
            n_series=encoder.encoder_only.en_embedding.glb_token.shape[1],
            patch_len=encoder.patch_len,
            hidden_size=model.hidden_size,
            n_heads=model.n_heads,
            e_layers=model.e_layers,
            d_ff=model.d_ff,
            factor=model.factor,
            dropout=model.dropout,
            use_norm=model.use_norm,
        ).eval()
    if model.backbone == "vanillatransformer":
        return VanillaTransformerEncoderOnly(
            c_in=encoder.encoder_only.enc_embedding.value_embedding.tokenConv.in_channels,
            exog_input_size=0,
            hidden_size=model.hidden_size,
            n_head=model.n_head,
            conv_hidden_size=model.linear_hidden_size or model.hidden_size * 4,
            activation="gelu",
            encoder_layers=model.encoder_layers,
            dropout=model.dropout,
        ).eval()
    if model.backbone == "informer":
        return InformerEncoderOnly(
            c_in=encoder.encoder_only.enc_embedding.value_embedding.tokenConv.in_channels,
            exog_input_size=0,
            hidden_size=model.hidden_size,
            factor=model.factor,
            n_head=model.n_head,
            conv_hidden_size=model.linear_hidden_size or model.hidden_size * 4,
            activation="gelu",
            encoder_layers=model.encoder_layers,
            dropout=model.dropout,
            distil=False,
        ).eval()
    if model.backbone == "itransformer":
        return ITransformerTokenEncoderOnly(
            input_size=model.input_size,
            hidden_size=model.hidden_size,
            n_heads=model.n_heads,
            e_layers=model.e_layers,
            d_ff=model.d_ff,
            factor=model.factor,
            dropout=model.dropout,
            use_norm=model.use_norm,
        ).eval()
    raise ValueError(f"Unsupported backbone for standalone helper: {model.backbone}")


@pytest.mark.parametrize(("backbone", "params", "helper_cls", "expected_fragment"), BACKBONE_CASES)
def test_aaforecast_backbone_modules_use_per_family_encoder_helpers(
    backbone: str,
    params: dict[str, object],
    helper_cls: type[torch.nn.Module],
    expected_fragment: str,
) -> None:
    model = _make_model(backbone, **params)

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
    model = _make_model(backbone, **params)
    outputs = model(_windows_batch())
    assert outputs.shape == (2, 2, 1)


@pytest.mark.parametrize(("backbone", "params", "_helper_cls", "_expected_fragment"), BACKBONE_CASES)
def test_aaforecast_non_gru_backbones_match_standalone_encoder_only_pre_bridge(
    backbone: str,
    params: dict[str, object],
    _helper_cls: type[torch.nn.Module],
    _expected_fragment: str,
) -> None:
    torch.manual_seed(7)
    model = _make_model(backbone, **params)
    inputs = _encoder_input(model)
    standalone = _build_standalone_helper(model, feature_size=inputs.shape[-1])
    standalone.load_state_dict(model.encoder.encoder_only.state_dict())

    torch.manual_seed(11)
    adapter_raw = model.encoder.encoder_only(inputs)
    torch.manual_seed(11)
    standalone_raw = standalone(inputs)

    assert adapter_raw.shape == standalone_raw.shape
    assert torch.allclose(adapter_raw, standalone_raw, atol=1e-6, rtol=1e-5)


def test_aaforecast_gru_adapter_matches_shared_gru_encoder_hidden_states() -> None:
    torch.manual_seed(7)
    model = _make_model(
        "gru",
        encoder_hidden_size=8,
        encoder_n_layers=2,
        encoder_dropout=0.1,
    )
    inputs = _encoder_input(model)
    standalone = _build_encoder(
        feature_size=inputs.shape[-1],
        hidden_size=model.encoder_hidden_size,
        num_layers=model.encoder_n_layers,
        dropout=model.encoder_dropout,
    ).eval()
    standalone.load_state_dict(model.encoder.encoder.state_dict())

    adapter_hidden = model.encoder(inputs)
    standalone_hidden, _ = standalone(inputs)

    assert adapter_hidden.shape == standalone_hidden.shape
    assert torch.allclose(adapter_hidden, standalone_hidden, atol=1e-6, rtol=1e-5)


def test_patchtst_bridge_discards_channel_identity_after_reusing_encoder_core() -> None:
    model = _make_model(
        "patchtst",
        hidden_size=8,
        n_heads=2,
        encoder_layers=1,
        dropout=0.1,
        linear_hidden_size=16,
        attn_dropout=0.0,
        patch_len=2,
        stride=1,
    )
    inputs = _encoder_input(model)
    swapped = inputs.clone()
    swapped[:, :, [0, 1]] = swapped[:, :, [1, 0]]

    raw = model.encoder.encoder_only(inputs)
    raw_swapped = model.encoder.encoder_only(swapped)
    restored = raw_swapped.clone()
    restored[:, [0, 1]] = restored[:, [1, 0]]

    bridged = model.encoder(inputs)
    bridged_swapped = model.encoder(swapped)

    assert torch.allclose(raw, restored, atol=1e-6, rtol=1e-5)
    assert torch.allclose(bridged, bridged_swapped, atol=1e-6, rtol=1e-5)


def test_timexer_bridge_reduces_series_order_sensitivity_relative_to_raw_encoder() -> None:
    model = _make_model(
        "timexer",
        hidden_size=8,
        n_heads=2,
        e_layers=1,
        dropout=0.1,
        d_ff=16,
        factor=1,
        patch_len=2,
        use_norm=True,
    )
    inputs = _encoder_input(model)
    swapped = inputs.clone()
    swapped[:, :, [0, 1]] = swapped[:, :, [1, 0]]

    raw_diff = (model.encoder.encoder_only(inputs) - model.encoder.encoder_only(swapped)).abs().max()
    bridged_diff = (model.encoder(inputs) - model.encoder(swapped)).abs().max()

    assert raw_diff.item() > 1.0
    assert bridged_diff.item() < 0.1
    assert raw_diff.item() > bridged_diff.item()


def test_informer_distillation_changes_sequence_length_while_aa_path_preserves_time_alignment() -> None:
    aligned = InformerEncoderOnly(
        c_in=9,
        exog_input_size=0,
        hidden_size=8,
        factor=1,
        n_head=2,
        conv_hidden_size=16,
        activation="gelu",
        encoder_layers=2,
        dropout=0.1,
        distil=False,
    ).eval()
    distilled = InformerEncoderOnly(
        c_in=9,
        exog_input_size=0,
        hidden_size=8,
        factor=1,
        n_head=2,
        conv_hidden_size=16,
        activation="gelu",
        encoder_layers=2,
        dropout=0.1,
        distil=True,
    ).eval()
    inputs = torch.randn(2, 4, 9)

    aligned_out = aligned(inputs, None)
    distilled_out = distilled(inputs, None)

    assert aligned_out.shape == (2, 4, 8)
    assert distilled_out.shape[1] < aligned_out.shape[1]


def test_itransformer_adapter_preserves_raw_tokens_until_late_projection() -> None:
    model = _make_model(
        "itransformer",
        hidden_size=8,
        n_heads=2,
        e_layers=1,
        dropout=0.1,
        d_ff=16,
        factor=1,
        use_norm=True,
    )
    inputs = _encoder_input(model)

    raw_tokens = model.encoder.encoder_only(inputs)
    adapter_tokens = model.encoder(inputs)
    late_projected = model.encoder.project_to_time_states(adapter_tokens)
    manual_projection = model.encoder.late_token_projection(
        raw_tokens.transpose(1, 2)
    ).transpose(1, 2)

    assert raw_tokens.shape == (2, inputs.shape[-1], model.hidden_size)
    assert adapter_tokens.shape == raw_tokens.shape
    assert torch.allclose(adapter_tokens, raw_tokens, atol=1e-6, rtol=1e-5)
    assert late_projected.shape == (2, model.input_size, model.hidden_size)
    assert torch.allclose(late_projected, manual_projection, atol=1e-6, rtol=1e-5)
    assert not hasattr(model.encoder, "input_projection")
    assert not hasattr(model.encoder, "output_norm")
