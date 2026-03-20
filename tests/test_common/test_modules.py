from __future__ import annotations

import torch

from neuralforecast.common._modules import FullAttention


def test_full_attention_falls_back_when_flash_attention_rejects_large_batch(
    monkeypatch,
):
    attention = FullAttention(mask_flag=False, output_attention=False)
    queries = torch.randn(2, 3, 1, 4)
    keys = torch.randn(2, 3, 1, 4)
    values = torch.randn(2, 3, 1, 4)

    def _raise_flash_limit(*_args, **_kwargs):
        raise RuntimeError(
            "Efficient attention cannot produce valid seed and offset outputs when the batch size exceeds (65535)."
        )

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", _raise_flash_limit
    )

    output, weights = attention(queries, keys, values, attn_mask=None)

    assert output.shape == queries.shape
    assert weights is None
