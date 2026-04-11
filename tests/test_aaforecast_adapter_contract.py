from __future__ import annotations

import torch

from neuralforecast.models.aaforecast.backbones import (
    AA_SUPPORTED_BACKBONES,
    build_aaforecast_backbone,
)


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


def test_supported_backbones_registry_remains_stable_after_router_split() -> None:
    assert AA_SUPPORTED_BACKBONES == {
        "gru",
        "vanillatransformer",
        "informer",
        "itransformer",
        "patchtst",
        "timexer",
    }
