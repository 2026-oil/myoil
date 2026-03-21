from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE

__all__ = ["ModernTCN"]


class RevIN(nn.Module):
    """Minimal reversible instance normalization.

    This is a self-contained subset of the official ModernTCN preprocessing.
    It normalizes `[batch, time, channels]` tensors and can undo that transform
    after forecasting. The implementation is intentionally small and local to
    keep the port self-contained.
    """

    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(mode)

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :]
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, channels]
        pad_len = max((self.kernel_size - 1) // 2, 0)
        front = x[:, :1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


def _conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    bias: bool,
) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def _conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int,
    *,
    dilation: int = 1,
    bias: bool = False,
) -> nn.Sequential:
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        _conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
        nn.BatchNorm1d(out_channels),
    )


class ReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: Optional[int],
        *,
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = _conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=False,
            )
            if small_kernel is not None:
                if small_kernel > kernel_size:
                    raise ValueError(
                        "small_kernel cannot be larger than kernel_size"
                    )
                self.small_conv = _conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bias=False,
                )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "lkb_reparam"):
            return self.lkb_reparam(inputs)
        out = self.lkb_origin(inputs)
        if hasattr(self, "small_conv"):
            out = out + self.small_conv(inputs)
        return out


class Block(nn.Module):
    def __init__(
        self,
        large_size: int,
        small_size: Optional[int],
        dmodel: int,
        dff: int,
        nvars: int,
        *,
        small_kernel_merged: bool = False,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
        )
        self.norm = nn.BatchNorm1d(dmodel)

        self.ffn1pw1 = nn.Conv1d(
            in_channels=nvars * dmodel,
            out_channels=nvars * dff,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=nvars,
        )
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(
            in_channels=nvars * dff,
            out_channels=nvars * dmodel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=nvars,
        )
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        self.ffn2pw1 = nn.Conv1d(
            in_channels=nvars * dmodel,
            out_channels=nvars * dff,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=dmodel,
        )
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(
            in_channels=nvars * dff,
            out_channels=nvars * dmodel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=dmodel,
        )
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, n_vars, d_model, seq_len = x.shape

        x = x.reshape(batch_size, n_vars * d_model, seq_len)
        x = self.dw(x)
        x = x.reshape(batch_size, n_vars, d_model, seq_len)
        x = x.reshape(batch_size * n_vars, d_model, seq_len)
        x = self.norm(x)
        x = x.reshape(batch_size, n_vars, d_model, seq_len)
        x = x.reshape(batch_size, n_vars * d_model, seq_len)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(batch_size, n_vars, d_model, seq_len)

        x = x.permute(0, 2, 1, 3).reshape(batch_size, d_model * n_vars, seq_len)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(batch_size, d_model, n_vars, seq_len).permute(0, 2, 1, 3)

        return residual + x


class Stage(nn.Module):
    def __init__(
        self,
        ffn_ratio: int,
        num_blocks: int,
        large_size: int,
        small_size: Optional[int],
        dmodel: int,
        nvars: int,
        *,
        small_kernel_merged: bool = False,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        dff = dmodel * ffn_ratio
        self.blocks = nn.ModuleList(
            [
                Block(
                    large_size=large_size,
                    small_size=small_size,
                    dmodel=dmodel,
                    dff=dff,
                    nvars=nvars,
                    small_kernel_merged=small_kernel_merged,
                    drop=drop,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _infer_feature_length(
    input_size: int,
    patch_size: int,
    patch_stride: int,
    downsample_ratio: int,
    num_stages: int,
) -> int:
    """Infer the last temporal length produced by the backbone."""
    length = max(input_size, patch_size)
    if patch_size != patch_stride:
        length += patch_size - patch_stride
    length = (length - patch_size) // patch_stride + 1
    for _ in range(max(num_stages - 1, 0)):
        length = _ceil_div(length, downsample_ratio)
    return length


class FlattenHead(nn.Module):
    def __init__(
        self,
        n_vars: int,
        head_nf: int,
        target_window: int,
        *,
        head_dropout: float = 0.0,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.individual = individual
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        if individual:
            self.linear = nn.ModuleList(
                [nn.Linear(head_nf, target_window) for _ in range(n_vars)]
            )
        else:
            self.linear = nn.Linear(head_nf, target_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, D, N] -> flatten the feature map and predict per series.
        if self.individual:
            outputs = [
                layer(self.flatten(x[:, idx, :, :]))
                for idx, layer in enumerate(self.linear)
            ]
            x = torch.stack(outputs, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
        return self.dropout(x)


class ModernTCNBackbone(nn.Module):
    """Self-contained subset of the official ModernTCN architecture.

    The original paper/repo includes a larger multi-scale head and optional time
    feature patching. This local port keeps the core stem -> stage residual block
    hierarchy and uses a lightweight flattening forecast head so the model remains
    CPU-safe and self-contained in this checkout.
    """

    def __init__(
        self,
        *,
        patch_size: int,
        patch_stride: int,
        downsample_ratio: int,
        ffn_ratio: int,
        num_blocks: Sequence[int],
        large_size: Sequence[int],
        small_size: Sequence[int],
        dims: Sequence[int],
        nvars: int,
        small_kernel_merged: bool = False,
        backbone_dropout: float = 0.1,
        head_dropout: float = 0.1,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        individual: bool = False,
        head_nf: int = 1,
        target_window: int = 96,
    ) -> None:
        super().__init__()
        if patch_size < patch_stride:
            raise ValueError("patch_size must be >= patch_stride")
        if len(dims) < 1:
            raise ValueError("dims must contain at least one stage")

        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(
                nvars, affine=affine, subtract_last=subtract_last
            )

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio
        self.num_stage = len(num_blocks)
        self.n_vars = nvars
        self.individual = individual
        self.dims = list(dims)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv1d(1, self.dims[0], kernel_size=patch_size, stride=patch_stride),
                nn.BatchNorm1d(self.dims[0]),
            )
        )
        for stage_idx in range(self.num_stage - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(self.dims[stage_idx]),
                    nn.Conv1d(
                        self.dims[stage_idx],
                        self.dims[stage_idx + 1],
                        kernel_size=downsample_ratio,
                        stride=downsample_ratio,
                    ),
                )
            )

        self.stages = nn.ModuleList(
            [
                Stage(
                    ffn_ratio=ffn_ratio,
                    num_blocks=num_blocks[idx],
                    large_size=large_size[idx],
                    small_size=small_size[idx],
                    dmodel=self.dims[idx],
                    nvars=nvars,
                    small_kernel_merged=small_kernel_merged,
                    drop=backbone_dropout,
                )
                for idx in range(self.num_stage)
            ]
        )

        self.head = FlattenHead(
            n_vars=nvars,
            head_nf=head_nf,
            target_window=target_window,
            head_dropout=head_dropout,
            individual=individual,
        )

    def _pad_to_length(self, x: torch.Tensor, min_length: int) -> torch.Tensor:
        current_length = x.shape[-1]
        if current_length >= min_length:
            return x
        pad_len = min_length - current_length
        return torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, L]
        x = x.unsqueeze(-2)  # [B, M, 1, L]
        for stage_idx in range(self.num_stage):
            batch_size, n_vars, channels, seq_len = x.shape
            x = x.reshape(batch_size * n_vars, channels, seq_len)
            if stage_idx == 0:
                x = self._pad_to_length(x, self.patch_size)
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad_len)], dim=-1)
            elif seq_len % self.downsample_ratio != 0:
                pad_len = self.downsample_ratio - (seq_len % self.downsample_ratio)
                x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[stage_idx](x)
            _, channels, seq_len = x.shape
            x = x.reshape(batch_size, n_vars, channels, seq_len)
            x = self.stages[stage_idx](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.revin_layer(x.permute(0, 2, 1), "norm").permute(0, 2, 1)
        x = self.forward_feature(x)
        x = self.head(x)
        if self.revin:
            x = self.revin_layer(x.permute(0, 2, 1), "denorm").permute(0, 2, 1)
        return x


def _as_tuple(value: int | Sequence[int], length: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return tuple([value] * length)
    value = tuple(value)
    if len(value) != length:
        raise ValueError(f"Expected length {length}, got {len(value)}")
    return value


class ModernTCN(BaseModel):
    """ModernTCN forecasting model.

    This is a self-contained NeuralForecast port of the official ModernTCN
    idea. It preserves the stem -> residual large-kernel block -> stage
    hierarchy, but intentionally uses a smaller pooled head and omits the
    upstream time-feature patching / fusion extras to keep the implementation
    CPU-safe and maintainable inside this repository.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int,
        n_series: int,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y: bool = False,
        patch_size: int = 16,
        patch_stride: int = 8,
        stem_ratio: int = 1,
        downsample_ratio: int = 2,
        ffn_ratio: int = 2,
        num_blocks: int | Sequence[int] = (1, 1, 1, 1),
        large_size: int | Sequence[int] = (7, 7, 5, 5),
        small_size: Optional[int | Sequence[int]] = (3, 3, 3, 3),
        dims: int | Sequence[int] = (16, 16, 16, 16),
        dw_dims: Optional[int | Sequence[int]] = None,
        small_kernel_merged: bool = False,
        backbone_dropout: float = 0.1,
        head_dropout: float = 0.1,
        use_multi_scale: bool = True,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        freq: Optional[str] = None,
        decomposition: bool = False,
        kernel_size: int = 25,
        individual: bool = False,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=32,
        inference_windows_batch_size=32,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ) -> None:
        super().__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        # Keep the original public knobs for compatibility, even when this
        # minimal port does not use every upstream helper layer.
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.stem_ratio = stem_ratio
        self.downsample_ratio = downsample_ratio
        self.ffn_ratio = ffn_ratio
        self.small_kernel_merged = small_kernel_merged
        self.backbone_dropout = backbone_dropout
        self.head_dropout = head_dropout
        self.use_multi_scale = use_multi_scale
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.freq = freq
        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.individual = individual
        self.nvars = n_series
        self.c_in = n_series
        self.target_window = h * self.loss.outputsize_multiplier

        n_stage = 4
        self.num_blocks = _as_tuple(num_blocks, n_stage)
        self.large_size = _as_tuple(large_size, n_stage)
        self.small_size = _as_tuple(small_size, n_stage) if small_size is not None else tuple([None] * n_stage)
        self.dims = _as_tuple(dims, n_stage)
        self.dw_dims = (
            _as_tuple(dw_dims, n_stage) if dw_dims is not None else self.dims
        )
        self.head_nf = self.dims[-1] * _infer_feature_length(
            input_size=self.input_size,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            downsample_ratio=self.downsample_ratio,
            num_stages=n_stage,
        )

        if self.decomposition and self.kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd when decomposition is enabled")

        self.model = ModernTCNBackbone(
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            downsample_ratio=self.downsample_ratio,
            ffn_ratio=self.ffn_ratio,
            num_blocks=self.num_blocks,
            large_size=self.large_size,
            small_size=self.small_size,
            dims=self.dims,
            nvars=self.nvars,
            small_kernel_merged=self.small_kernel_merged,
            backbone_dropout=self.backbone_dropout,
            head_dropout=self.head_dropout,
            revin=self.revin,
            affine=self.affine,
            subtract_last=self.subtract_last,
            individual=self.individual,
            head_nf=self.head_nf,
            target_window=self.target_window,
        )

        if self.decomposition:
            self.decomp_module = SeriesDecomp(self.kernel_size)
            self.model_res = ModernTCNBackbone(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.backbone_dropout,
                head_dropout=self.head_dropout,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                individual=self.individual,
                head_nf=self.head_nf,
                target_window=self.target_window,
            )
            self.model_trend = ModernTCNBackbone(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.backbone_dropout,
                head_dropout=self.head_dropout,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                individual=self.individual,
                head_nf=self.head_nf,
                target_window=self.target_window,
            )

    def forward(self, windows_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        insample_y = windows_batch["insample_y"]
        x = insample_y.permute(0, 2, 1)

        if self.decomposition:
            residual, trend = self.decomp_module(x)
            x = self.model_res(residual) + self.model_trend(trend)
        else:
            x = self.model(x)

        if self.loss.outputsize_multiplier > 1:
            # Keep the output contract compatible with the rest of the repo:
            # [B, n_series, h * out_mult] -> [B, h, n_series * out_mult]
            x = x.reshape(x.shape[0], x.shape[1], self.h, self.loss.outputsize_multiplier)
            x = x.permute(0, 2, 1, 3).reshape(x.shape[0], self.h, -1)
        else:
            x = x.permute(0, 2, 1)
            x = x.reshape(x.shape[0], self.h, -1)
        return x
