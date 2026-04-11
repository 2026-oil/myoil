from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AABackboneEvidence:
    backbone: str
    reused_components: tuple[str, ...]
    aa_bridge_steps: tuple[str, ...]
    unavoidable_divergences: tuple[str, ...]
    required_output: str = "[B, time, hidden]"


@dataclass(frozen=True)
class AATimeXerTokenStates:
    patch_states: torch.Tensor
    global_states: torch.Tensor

    def combined(self) -> torch.Tensor:
        return torch.cat([self.patch_states, self.global_states], dim=2)


class AABackboneAdapter(nn.Module):
    evidence: AABackboneEvidence

    def faithfulness_evidence(self) -> AABackboneEvidence:
        return self.evidence

    def project_to_time_states(self, states: torch.Tensor) -> torch.Tensor:
        return states


def validate_attention_heads(
    hidden_size: int,
    n_heads: int,
    *,
    field_name: str,
) -> None:
    if n_heads <= 0:
        raise ValueError(f"AAForecast {field_name} must be positive, got {n_heads}")
    if hidden_size % n_heads != 0:
        raise ValueError(
            f"AAForecast hidden_size={hidden_size} must be divisible by {field_name}={n_heads}"
        )


def scatter_patch_tokens_to_time_states(
    patch_tokens: torch.Tensor,
    *,
    seq_len: int,
    patch_len: int,
    stride: int,
    template: torch.Tensor,
    global_context: torch.Tensor | None = None,
) -> torch.Tensor:
    if patch_tokens.ndim != 3:
        raise ValueError("patch_tokens must be rank-3 [B, patch_num, hidden]")
    batch_size, patch_num, hidden_size = patch_tokens.shape
    hidden = template.new_zeros(batch_size, seq_len, hidden_size)
    counts = template.new_zeros(batch_size, seq_len, 1)
    for patch_idx in range(patch_num):
        start = patch_idx * stride
        stop = min(start + patch_len, seq_len)
        if stop <= start:
            continue
        patch_hidden = patch_tokens[:, patch_idx].unsqueeze(1)
        hidden[:, start:stop, :] += patch_hidden
        counts[:, start:stop, :] += 1
    hidden = hidden / counts.clamp_min(1.0)
    if global_context is not None:
        hidden = hidden + global_context.unsqueeze(1)
    return hidden
