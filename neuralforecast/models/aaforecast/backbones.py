from __future__ import annotations

import torch
import torch.nn as nn

from ...common._modules import (
    AttentionLayer,
    DataEmbedding_inverted,
    FullAttention,
    TransEncoder,
    TransEncoderLayer,
)
from ..informer import ProbAttention
from ..patchtst import TSTiEncoder, _patchtst_create_patches, _patchtst_patch_num

AA_SUPPORTED_BACKBONES = {
    "gru",
    "vanillatransformer",
    "informer",
    "itransformer",
    "patchtst",
    "timexer",
}


class GRUEncoderBackbone(nn.Module):
    def __init__(
        self,
        *,
        feature_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = feature_size
        self.encoder = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.encoder(inputs)
        return hidden_states


class VanillaTransformerEncoderBackbone(nn.Module):
    def __init__(
        self,
        *,
        feature_size: int,
        hidden_size: int,
        n_head: int,
        encoder_layers: int,
        dropout: float,
        linear_hidden_size: int | None,
    ) -> None:
        super().__init__()
        _validate_attention_heads(hidden_size, n_head, field_name="n_head")
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(feature_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_head,
                dim_feedforward=linear_hidden_size or hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=encoder_layers,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.input_projection(inputs))


class InformerEncoderBackbone(nn.Module):
    def __init__(
        self,
        *,
        feature_size: int,
        hidden_size: int,
        n_head: int,
        encoder_layers: int,
        dropout: float,
        linear_hidden_size: int | None,
        factor: int,
    ) -> None:
        super().__init__()
        _validate_attention_heads(hidden_size, n_head, field_name="n_head")
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(feature_size, hidden_size)
        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        hidden_size,
                        n_head,
                    ),
                    hidden_size,
                    linear_hidden_size or hidden_size * 4,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(encoder_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self.input_projection(inputs))
        return encoded


class PatchTSTEncoderBackbone(nn.Module):
    def __init__(
        self,
        *,
        feature_size: int,
        input_size: int,
        hidden_size: int,
        n_heads: int,
        encoder_layers: int,
        dropout: float,
        linear_hidden_size: int | None,
        attn_dropout: float,
        patch_len: int,
        stride: int,
    ) -> None:
        super().__init__()
        _validate_attention_heads(hidden_size, n_heads, field_name="n_heads")
        if patch_len <= 0:
            raise ValueError(f"PatchTST patch_len must be positive, got {patch_len}")
        if patch_len > input_size:
            raise ValueError(
                "PatchTST patch_len must not exceed input_size; "
                f"got patch_len={patch_len}, input_size={input_size}"
            )
        if stride <= 0:
            raise ValueError(f"PatchTST stride must be positive, got {stride}")
        self.hidden_size = hidden_size
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = "end"
        self.patch_num = _patchtst_patch_num(
            input_size=input_size,
            patch_len=patch_len,
            stride=stride,
            padding_patch=self.padding_patch,
        )
        self.encoder = TSTiEncoder(
            c_in=feature_size,
            patch_num=self.patch_num,
            patch_len=patch_len,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            linear_hidden_size=linear_hidden_size or hidden_size * 4,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.shape
        feature_major = inputs.transpose(1, 2)
        patches = _patchtst_create_patches(
            feature_major,
            patch_len=self.patch_len,
            stride=self.stride,
            padding_patch=self.padding_patch,
        )
        encoded = self.encoder(patches).permute(0, 1, 3, 2)
        patch_tokens = encoded.mean(dim=1)
        hidden = inputs.new_zeros(batch_size, seq_len, self.hidden_size)
        counts = inputs.new_zeros(batch_size, seq_len, 1)
        for idx, start in enumerate(range(0, self.patch_num * self.stride, self.stride)):
            stop = min(start + self.patch_len, seq_len)
            if stop <= start:
                continue
            patch_hidden = patch_tokens[:, idx].unsqueeze(1)
            hidden[:, start:stop, :] += patch_hidden
            counts[:, start:stop, :] += 1
        hidden = hidden / counts.clamp_min(1.0)
        return self.output_norm(hidden)


class ITransformerEncoderBackbone(nn.Module):
    def __init__(
        self,
        *,
        feature_size: int,
        input_size: int,
        hidden_size: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
        d_ff: int,
        factor: int,
        use_norm: bool,
    ) -> None:
        super().__init__()
        _validate_attention_heads(hidden_size, n_heads, field_name="n_heads")
        self.hidden_size = hidden_size
        self.embedding = DataEmbedding_inverted(input_size, hidden_size, dropout)
        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                        ),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size),
        )
        self.token_to_time = nn.Linear(feature_size, input_size)
        self.input_projection = nn.Linear(feature_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size) if use_norm else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        token_states = self.embedding(inputs, None)
        encoded_tokens, _ = self.encoder(token_states, attn_mask=None)
        time_states = self.token_to_time(encoded_tokens.transpose(1, 2)).transpose(1, 2)
        residual = self.input_projection(inputs)
        return self.output_norm(time_states + residual)


class TimeXerEncoderBackbone(nn.Module):
    def __init__(
        self,
        *,
        feature_size: int,
        input_size: int,
        hidden_size: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
        d_ff: int,
        factor: int,
        patch_len: int,
        use_norm: bool,
    ) -> None:
        del factor
        super().__init__()
        _validate_attention_heads(hidden_size, n_heads, field_name="n_heads")
        if patch_len <= 0:
            raise ValueError(f"TimeXer patch_len must be positive, got {patch_len}")
        if input_size % patch_len != 0:
            raise ValueError(
                "TimeXer requires patch_len to evenly divide input_size; "
                f"got input_size={input_size}, patch_len={patch_len}"
            )
        self.hidden_size = hidden_size
        self.patch_len = patch_len
        self.feature_patch_projection = nn.Linear(patch_len, hidden_size)
        self.feature_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=e_layers,
        )
        self.time_projection = nn.Linear(feature_size, hidden_size)
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=e_layers,
        )
        self.output_norm = nn.LayerNorm(hidden_size) if use_norm else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_major = inputs.transpose(1, 2)
        feature_patches = feature_major.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        patch_tokens = self.feature_patch_projection(feature_patches)
        feature_tokens = patch_tokens.mean(dim=2)
        encoded_features = self.feature_encoder(feature_tokens)
        feature_context = encoded_features.mean(dim=1, keepdim=True)
        time_tokens = self.time_projection(inputs) + feature_context
        return self.output_norm(self.time_encoder(time_tokens))


def build_aaforecast_backbone(
    backbone: str,
    *,
    feature_size: int,
    input_size: int,
    encoder_hidden_size: int,
    encoder_n_layers: int,
    encoder_dropout: float,
    hidden_size: int,
    n_head: int,
    n_heads: int,
    encoder_layers: int,
    dropout: float,
    linear_hidden_size: int | None,
    factor: int,
    attn_dropout: float,
    patch_len: int,
    stride: int,
    e_layers: int,
    d_ff: int,
    use_norm: bool,
) -> nn.Module:
    normalized = str(backbone).strip().lower()
    if normalized not in AA_SUPPORTED_BACKBONES:
        supported = ", ".join(sorted(AA_SUPPORTED_BACKBONES))
        raise ValueError(f"AAForecast backbone must be one of: {supported}")
    if normalized == "gru":
        return GRUEncoderBackbone(
            feature_size=feature_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_n_layers,
            dropout=encoder_dropout,
        )
    if normalized == "vanillatransformer":
        return VanillaTransformerEncoderBackbone(
            feature_size=feature_size,
            hidden_size=hidden_size,
            n_head=n_head,
            encoder_layers=encoder_layers,
            dropout=dropout,
            linear_hidden_size=linear_hidden_size,
        )
    if normalized == "informer":
        return InformerEncoderBackbone(
            feature_size=feature_size,
            hidden_size=hidden_size,
            n_head=n_head,
            encoder_layers=encoder_layers,
            dropout=dropout,
            linear_hidden_size=linear_hidden_size,
            factor=factor,
        )
    if normalized == "itransformer":
        return ITransformerEncoderBackbone(
            feature_size=feature_size,
            input_size=input_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
            d_ff=d_ff,
            factor=factor,
            use_norm=use_norm,
        )
    if normalized == "patchtst":
        return PatchTSTEncoderBackbone(
            feature_size=feature_size,
            input_size=input_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            dropout=dropout,
            linear_hidden_size=linear_hidden_size,
            attn_dropout=attn_dropout,
            patch_len=patch_len,
            stride=stride,
        )
    return TimeXerEncoderBackbone(
        feature_size=feature_size,
        input_size=input_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
        d_ff=d_ff,
        factor=factor,
        patch_len=patch_len,
        use_norm=use_norm,
    )


def _validate_attention_heads(
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
