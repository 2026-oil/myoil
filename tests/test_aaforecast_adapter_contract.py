from __future__ import annotations

import torch

from neuralforecast.models.aaforecast.backbones import (
    AA_SUPPORTED_BACKBONES,
    build_aaforecast_backbone,
)
from neuralforecast.models.aaforecast.models.base import AATimeXerTokenStates
from plugins.aa_forecast.modules import TimeXerTokenSparseAttention


def test_adapter_contract_requires_b_time_hidden_output() -> None:
    backbone = build_aaforecast_backbone(
        "patchtst",
        feature_size=5,
        input_size=4,
        encoder_hidden_size=8,
        encoder_n_layers=2,
        encoder_dropout=0.1,
        hidden_size=8,
        n_head=2,
        n_heads=2,
        encoder_layers=1,
        dropout=0.1,
        linear_hidden_size=16,
        factor=1,
        attn_dropout=0.0,
        patch_len=2,
        stride=1,
        e_layers=1,
        d_ff=16,
        use_norm=True,
    )
    outputs = backbone(torch.randn(2, 4, 5))
    assert outputs.shape == (2, 4, 8)
    assert backbone.faithfulness_evidence().required_output == "[B, time, hidden]"


def test_adapter_contract_exposes_proof_metadata() -> None:
    backbone = build_aaforecast_backbone(
        "vanillatransformer",
        feature_size=5,
        input_size=4,
        encoder_hidden_size=8,
        encoder_n_layers=2,
        encoder_dropout=0.1,
        hidden_size=8,
        n_head=2,
        n_heads=2,
        encoder_layers=1,
        dropout=0.1,
        linear_hidden_size=16,
        factor=1,
        attn_dropout=0.0,
        patch_len=2,
        stride=1,
        e_layers=1,
        d_ff=16,
        use_norm=True,
    )
    evidence = backbone.faithfulness_evidence()
    assert evidence.backbone == "vanillatransformer"
    assert evidence.reused_components
    assert evidence.aa_bridge_steps
    assert evidence.unavoidable_divergences


def test_itransformer_adapter_exposes_token_space_internal_contract() -> None:
    backbone = build_aaforecast_backbone(
        "itransformer",
        feature_size=5,
        input_size=4,
        encoder_hidden_size=8,
        encoder_n_layers=2,
        encoder_dropout=0.1,
        hidden_size=8,
        n_head=2,
        n_heads=2,
        encoder_layers=1,
        dropout=0.1,
        linear_hidden_size=16,
        factor=1,
        attn_dropout=0.0,
        patch_len=2,
        stride=1,
        e_layers=1,
        d_ff=16,
        use_norm=True,
    )
    token_states = backbone(torch.randn(2, 4, 5))
    assert token_states.shape == (2, 5, 8)
    assert backbone.project_to_time_states(token_states).shape == (2, 4, 8)
    assert backbone.faithfulness_evidence().required_output == "[B, token, hidden]"


def test_timexer_adapter_exposes_patch_and_global_token_contract() -> None:
    backbone = build_aaforecast_backbone(
        "timexer",
        feature_size=5,
        input_size=4,
        encoder_hidden_size=8,
        encoder_n_layers=2,
        encoder_dropout=0.1,
        hidden_size=8,
        n_head=2,
        n_heads=2,
        encoder_layers=1,
        dropout=0.1,
        linear_hidden_size=16,
        factor=1,
        attn_dropout=0.0,
        patch_len=2,
        stride=1,
        e_layers=1,
        d_ff=16,
        use_norm=True,
    )
    token_states = backbone(torch.randn(2, 4, 5))
    assert isinstance(token_states, AATimeXerTokenStates)
    assert token_states.patch_states.shape == (2, 5, 2, 8)
    assert token_states.global_states.shape == (2, 5, 1, 8)
    assert (
        backbone.faithfulness_evidence().required_output
        == "{patch:[B, channel, patch, hidden], global:[B, channel, 1, hidden]}"
    )


def test_timexer_token_sparse_attention_preserves_hidden_states_when_no_token_is_active() -> None:
    attention = TimeXerTokenSparseAttention(hidden_size=4)
    patch_states = torch.tensor(
        [[
            [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
        ]]
    )
    global_states = torch.tensor(
        [[
            [[10.0, 1.0, 0.0, 0.0]],
            [[20.0, 1.0, 0.0, 0.0]],
        ]]
    )
    patch_mask = torch.zeros(1, 2, 1, dtype=torch.bool)
    patch_count = torch.zeros(1, 2, 1)
    patch_activity = torch.zeros(1, 2, 1)
    global_mask = torch.zeros(1, 1, 1, dtype=torch.bool)
    global_count = torch.zeros(1, 1, 1)
    global_activity = torch.zeros(1, 1, 1)

    (attended_patch, attended_global), weights = attention(
        patch_states,
        global_states,
        patch_mask,
        patch_count,
        patch_activity,
        global_mask,
        global_count,
        global_activity,
    )

    assert attended_patch.shape == patch_states.shape
    assert attended_global.shape == global_states.shape
    assert weights.shape == (1, 2, 3)
    assert torch.allclose(weights, torch.zeros_like(weights))
    assert torch.allclose(attended_patch, patch_states)
    assert torch.allclose(attended_global, global_states)


def test_supported_backbones_registry_remains_stable_after_router_split() -> None:
    assert AA_SUPPORTED_BACKBONES == {
        "gru",
        "vanillatransformer",
        "informer",
        "itransformer",
        "patchtst",
        "timexer",
    }
