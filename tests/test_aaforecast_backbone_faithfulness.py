from __future__ import annotations

import pytest
import torch

from neuralforecast.models import AAForecast
from neuralforecast.models.aaforecast.gru import _build_encoder
from neuralforecast.models.aaforecast.models.base import AATimeXerTokenStates
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
        "token space",
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
            c_in=1,
            exog_input_size=feature_size - 1,
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
    windows_batch = {
        "insample_y": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
        "hist_exog": torch.arange(32, dtype=torch.float32).reshape(2, 4, 4),
        "futr_exog": torch.empty(2, 4, 0),
        "stat_exog": torch.empty(2, 0),
    }
    outputs = model(windows_batch)
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
    if model.backbone == "informer":
        signal, exog = model.encoder._split_inputs(inputs)
        adapter_raw = model.encoder(inputs)
        torch.manual_seed(11)
        standalone_raw = standalone(signal, exog)
    else:
        adapter_raw = model.encoder.encoder_only(inputs)
        torch.manual_seed(11)
        standalone_raw = standalone(inputs)

    assert adapter_raw.shape == standalone_raw.shape
    assert torch.allclose(adapter_raw, standalone_raw, atol=1e-6, rtol=1e-5)


def test_informer_exposes_anomaly_projection_and_preserves_forward_contract() -> None:
    model = _make_model(
        "informer",
        hidden_size=8,
        n_head=2,
        encoder_layers=1,
        dropout=0.1,
        linear_hidden_size=16,
        factor=1,
    )

    assert model.transformer_anomaly_projection is not None
    weight = model.transformer_anomaly_projection.weight
    assert weight.shape == (model.encoder_hidden_size, 1)
    windows_batch = {
        "insample_y": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
        "hist_exog": torch.arange(32, dtype=torch.float32).reshape(2, 4, 4),
        "futr_exog": torch.empty(2, 4, 0),
        "stat_exog": torch.empty(2, 0),
    }
    outputs = model(windows_batch)
    assert outputs.shape == (2, 2, 1)


def test_informer_cross_bridge_selects_named_star_and_non_star_context() -> None:
    model = AAForecast(
        h=2,
        input_size=4,
        backbone="informer",
        hist_exog_list=[
            "GPRD_THREAT",
            "BS_Core_Index_A",
            "BS_Core_Index_C",
            "Idx_OVX",
        ],
        star_hist_exog_list=["GPRD_THREAT", "BS_Core_Index_A"],
        non_star_hist_exog_list=["BS_Core_Index_C", "Idx_OVX"],
        star_hist_exog_tail_modes=["upward", "upward"],
        informer_bridge_exog_list=[
            "GPRD_THREAT",
            "BS_Core_Index_A",
            "BS_Core_Index_C",
            "Idx_OVX",
        ],
        informer_bridge_hidden_size=8,
        informer_bridge_layers=1,
        scaler_type="identity",
        max_steps=1,
        val_check_steps=1,
        batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
        hidden_size=8,
        n_head=2,
        encoder_layers=1,
        dropout=0.1,
        linear_hidden_size=16,
        factor=1,
    ).eval()

    assert model.informer_bridge_sources == (
        ("star_activity", 0),
        ("star_activity", 1),
        ("non_star_raw", 0),
        ("non_star_raw", 1),
    )
    assert model.informer_cross_bridge is not None

    star_payload = {
        "star_hist_activity": torch.tensor(
            [[
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]]
        )
    }
    hist_exog = torch.tensor(
        [[
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]]
    )
    context = model._build_informer_bridge_context(
        hist_exog=hist_exog,
        star_payload=star_payload,
        template=torch.zeros(1, 4, 1),
    )

    assert context is not None
    expected = torch.tensor(
        [[
            [1.0, 2.0, 30.0, 40.0],
            [3.0, 4.0, 31.0, 41.0],
            [5.0, 6.0, 32.0, 42.0],
            [7.0, 8.0, 33.0, 43.0],
        ]]
    )
    assert torch.equal(context, expected)

    windows_batch = {
        "insample_y": torch.arange(8, dtype=torch.float32).reshape(2, 4, 1),
        "hist_exog": torch.arange(32, dtype=torch.float32).reshape(2, 4, 4),
        "futr_exog": torch.empty(2, 4, 0),
        "stat_exog": torch.empty(2, 0),
    }
    outputs = model(windows_batch)
    assert outputs.shape == (2, 2, 1)


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


def test_timexer_adapter_preserves_patch_and_global_token_structure() -> None:
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
    raw_tokens = model.encoder.encoder_only(inputs)
    adapter_states = model.encoder(inputs)

    assert isinstance(adapter_states, AATimeXerTokenStates)
    expected_patch = raw_tokens[..., :-1].permute(0, 1, 3, 2)
    expected_global = raw_tokens[..., -1:].permute(0, 1, 3, 2)
    assert adapter_states.patch_states.shape == (
        inputs.shape[0],
        inputs.shape[-1],
        model.encoder.patch_num,
        model.hidden_size,
    )
    assert adapter_states.global_states.shape == (
        inputs.shape[0],
        inputs.shape[-1],
        1,
        model.hidden_size,
    )
    assert torch.allclose(adapter_states.patch_states, expected_patch, atol=1e-6, rtol=1e-5)
    assert torch.allclose(adapter_states.global_states, expected_global, atol=1e-6, rtol=1e-5)


def test_timexer_time_signals_are_aggregated_to_patch_and_global_tokens() -> None:
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
    critical_mask = torch.tensor(
        [[
            [True],
            [False],
            [False],
            [True],
        ]],
        dtype=torch.bool,
    )
    count_active_channels = torch.tensor(
        [[
            [2.0],
            [0.0],
            [0.0],
            [3.0],
        ]]
    )
    channel_activity = torch.tensor(
        [[
            [1.0, 0.5],
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 1.0],
        ]]
    )

    aggregated = model._aggregate_timexer_attention_signals(
        critical_mask=critical_mask,
        count_active_channels=count_active_channels,
        channel_activity=channel_activity,
    )

    assert torch.equal(
        aggregated["patch_mask"],
        torch.tensor([[[True], [True]]], dtype=torch.bool),
    )
    assert torch.allclose(
        aggregated["patch_count"],
        torch.tensor([[[2.0], [3.0]]]),
    )
    assert torch.allclose(
        aggregated["patch_activity"],
        torch.tensor([[[1.5], [3.0]]]),
    )
    assert torch.equal(
        aggregated["global_mask"],
        torch.tensor([[[True]]], dtype=torch.bool),
    )
    assert torch.allclose(aggregated["global_count"], torch.tensor([[[5.0]]]))
    assert torch.allclose(aggregated["global_activity"], torch.tensor([[[4.5]]]))


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


def test_itransformer_adapter_keeps_sparse_attention_and_decode_in_token_space() -> None:
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
    signals = model._aggregate_itransformer_attention_signals(
        star_payload={
            "target_activity": torch.tensor(
                [[
                    [2.0],
                    [0.0],
                    [1.0],
                    [0.0],
                ]],
                dtype=inputs.dtype,
            ),
            "star_hist_activity": torch.tensor(
                [[
                    [0.0],
                    [0.0],
                    [3.0],
                    [0.0],
                ]],
                dtype=inputs.dtype,
            ),
        },
        template=inputs[:1],
    )
    attended_tokens, _ = model.attention(
        adapter_tokens[:1],
        signals["token_mask"],
        signals["token_count"].to(dtype=adapter_tokens.dtype),
        signals["token_activity"].to(dtype=adapter_tokens.dtype),
    )
    decoded = model._decode_itransformer_forecast(
        raw_tokens=adapter_tokens[:1],
        attended_tokens=attended_tokens,
    )

    assert raw_tokens.shape == (2, inputs.shape[-1], model.hidden_size)
    assert adapter_tokens.shape == raw_tokens.shape
    assert torch.allclose(adapter_tokens, raw_tokens, atol=1e-6, rtol=1e-5)
    assert signals["token_mask"].shape == (1, inputs.shape[-1], 1)
    assert signals["token_count"].shape == (1, inputs.shape[-1], 1)
    assert signals["token_activity"].shape == (1, inputs.shape[-1], 1)
    assert attended_tokens.shape == (1, inputs.shape[-1], model.hidden_size)
    assert decoded.shape == (1, model.h, 1)
    assert model.target_token_indices == (0, 1, 2, 3, 4)
    assert model.itransformer_decoder.in_features == 2 * model.hidden_size


def test_informer_event_projection_paths_keep_decoder_width_stable() -> None:
    model = _make_model(
        "informer",
        hidden_size=8,
        n_head=2,
        encoder_layers=1,
        dropout=0.1,
        linear_hidden_size=16,
        factor=1,
    )
    hidden_states = torch.randn(2, model.input_size, model.hidden_size)
    attended_states = torch.randn(2, model.input_size, model.hidden_size)
    event_summary = torch.tensor(
        [
            [0.1] * model.EVENT_SUMMARY_SIZE,
            [0.8] * model.EVENT_SUMMARY_SIZE,
        ],
        dtype=hidden_states.dtype,
    )
    event_trajectory = torch.tensor(
        [
            [0.1] * model.EVENT_TRAJECTORY_SIZE,
            [0.8] * model.EVENT_TRAJECTORY_SIZE,
        ],
        dtype=hidden_states.dtype,
    )

    hidden_aligned, attended_aligned = model._build_time_decoder_features(
        hidden_states=hidden_states,
        attended_states=attended_states,
    )
    event_latent = model._project_event_summary(event_summary)
    event_path = model._project_event_trajectory(event_trajectory)
    decoder_input = model._build_time_decoder_input(
        hidden_states=hidden_states,
        attended_states=attended_states,
    )

    assert hidden_aligned.shape == (2, model.h, model.hidden_size)
    assert attended_aligned.shape == hidden_aligned.shape
    assert event_latent.shape == (2, model.hidden_size)
    assert event_path.shape == (2, model.hidden_size)
    assert decoder_input.shape == (2, model.h, 2 * model.hidden_size)
    assert not torch.allclose(event_latent[0], event_latent[1])
    assert not torch.allclose(event_path[0], event_path[1])
