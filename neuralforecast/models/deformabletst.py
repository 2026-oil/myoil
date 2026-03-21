__all__ = ["DeformableTST"]


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common._base_model import BaseModel
from ..common._modules import RevIN
from ..losses.pytorch import MAE


def _as_list(value, length: int):
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) == length:
            return items
        if len(items) == 1:
            return items * length
        if len(items) < length:
            return items + [items[-1]] * (length - len(items))
        return items[:length]
    return [value] * length


def _conv_out_length(
    length: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    out = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return max(1, int(out))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        return x * self.weight.view(1, -1, 1)


class LayerNormProxy(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class FlattenHead(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TransformerMLP(nn.Module):
    def __init__(self, channels, expansion, drop, local_kernel_size=None):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.block(x)


class TransformerMLPWithConv(nn.Module):
    def __init__(self, channels, expansion, drop, local_kernel_size):
        super().__init__()
        hidden = channels * expansion
        local_kernel_size = max(1, int(local_kernel_size))
        self.linear1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.dwc = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=local_kernel_size,
            stride=1,
            padding=local_kernel_size // 2,
            groups=hidden,
        )
        self.linear2 = nn.Conv1d(hidden, channels, kernel_size=1)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


class DAttentionBaseline(nn.Module):
    def __init__(
        self,
        q_size,
        kv_size,
        n_heads,
        n_head_channels,
        n_groups,
        attn_drop,
        proj_drop,
        stride,
        offset_range_factor,
        use_pe,
        dwc_pe,
        no_off,
        fixed_pe,
        ksize,
        log_cpb,
    ):
        super().__init__()
        del q_size, kv_size, n_groups, offset_range_factor, dwc_pe, fixed_pe, ksize, log_cpb

        self.n_heads = n_heads
        self.n_head_channels = n_head_channels
        self.nc = n_heads * n_head_channels
        self.scale = n_head_channels**-0.5
        self.stride = max(1, int(stride))
        self.use_pe = bool(use_pe)
        self.no_off = bool(no_off)

        self.proj_q = nn.Conv1d(self.nc, self.nc, kernel_size=1)
        self.proj_k = nn.Conv1d(self.nc, self.nc, kernel_size=1)
        self.proj_v = nn.Conv1d(self.nc, self.nc, kernel_size=1)
        self.proj_out = nn.Conv1d(self.nc, self.nc, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        sample_stride = min(self.stride, seq_len)

        q = self.proj_q(x)
        if sample_stride > 1:
            x_sampled = F.avg_pool1d(x, kernel_size=sample_stride, stride=sample_stride)
        else:
            x_sampled = x
        k = self.proj_k(x_sampled)
        v = self.proj_v(x_sampled)

        q = q.reshape(batch_size, self.n_heads, self.n_head_channels, seq_len)
        q = q.permute(0, 1, 3, 2)
        kv_len = k.shape[-1]
        k = k.reshape(batch_size, self.n_heads, self.n_head_channels, kv_len)
        k = k.permute(0, 1, 3, 2)
        v = v.reshape(batch_size, self.n_heads, self.n_head_channels, kv_len)
        v = v.permute(0, 1, 3, 2)

        scores = torch.einsum("bhld,bhmd->bhlm", q, k) * self.scale
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhlm,bhmd->bhld", attn, v)
        out = out.permute(0, 1, 3, 2).reshape(batch_size, channels, seq_len)
        out = self.proj_drop(self.proj_out(out))

        pos = torch.linspace(-1.0, 1.0, kv_len, device=x.device, dtype=x.dtype)
        pos = pos.view(1, kv_len, 1).expand(batch_size, -1, -1)
        return out, pos, pos


class Stage(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_embed,
        depths,
        drop_path_rate,
        layer_scale_value,
        use_pe,
        use_lpu,
        local_kernel_size,
        expansion,
        drop,
        use_dwc_mlp,
        heads,
        attn_drop,
        proj_drop,
        stage_spec,
        window_size,
        nat_ksize,
        ksize,
        stride,
        n_groups,
        offset_range_factor,
        no_off,
        dwc_pe,
        fixed_pe,
        log_cpb,
    ):
        super().__init__()
        del window_size, nat_ksize
        self.depths = depths
        self.use_lpu = bool(use_lpu)
        self.stage_spec = stage_spec

        if dim_embed % heads != 0:
            raise ValueError(
                f"dim_embed={dim_embed} must be divisible by heads={heads}."
            )

        head_channels = dim_embed // heads

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv1d(
                    dim_embed,
                    dim_embed,
                    kernel_size=max(1, int(local_kernel_size)),
                    stride=1,
                    padding=max(1, int(local_kernel_size)) // 2,
                    groups=dim_embed,
                )
                if self.use_lpu
                else nn.Identity()
                for _ in range(depths)
            ]
        )

        self.attns = nn.ModuleList(
            [
                DAttentionBaseline(
                    fmap_size,
                    fmap_size,
                    heads,
                    head_channels,
                    n_groups,
                    attn_drop,
                    proj_drop,
                    stride,
                    offset_range_factor,
                    use_pe,
                    dwc_pe,
                    no_off,
                    fixed_pe,
                    ksize,
                    log_cpb,
                )
                for _ in range(depths)
            ]
        )
        self.drop_path = nn.ModuleList(
            [DropPath(rate) if rate > 0.0 else nn.Identity() for rate in drop_path_rate]
        )
        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP
        self.mlps = nn.ModuleList(
            [mlp_fn(dim_embed, expansion, drop, local_kernel_size) for _ in range(depths)]
        )
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value)
                if layer_scale_value > 0.0
                else nn.Identity()
                for _ in range(2 * depths)
            ]
        )

    def forward(self, x):
        for d in range(self.depths):
            if self.use_lpu:
                x = x + self.local_perception_units[d](x)

            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            del pos, ref
            x = self.layer_scales[2 * d](x)
            x = x0 + self.drop_path[d](x)

            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.layer_scales[2 * d + 1](x)
            x = x0 + self.drop_path[d](x)

        return x


class DeformableTSTBackbone(nn.Module):
    def __init__(
        self,
        n_vars,
        rev,
        revin_affine,
        revin_subtract_last,
        stem_ratio,
        down_ratio,
        fmap_size,
        dims,
        depths,
        drop_path_rate,
        layer_scale_value,
        use_pe,
        use_lpu,
        local_kernel_size,
        expansion,
        drop,
        use_dwc_mlp,
        heads,
        attn_drop,
        proj_drop,
        stage_spec,
        window_size,
        nat_ksize,
        ksize,
        stride,
        n_groups,
        offset_range_factor,
        no_off,
        dwc_pe,
        fixed_pe,
        log_cpb,
        seq_len,
        pred_len,
        c_out,
        head_dropout,
        head_type,
        use_head_norm,
    ):
        super().__init__()
        self.rev = rev
        if self.rev:
            self.revin = RevIN(
                n_vars, affine=revin_affine, subtract_last=revin_subtract_last
            )

        self.n_vars = n_vars
        self.num_stage = len(depths)
        self.head_type = head_type
        self.use_head_norm = use_head_norm
        self.seq_len = seq_len
        self.pred_len = pred_len

        dims = list(dims)
        if len(dims) == 1 and self.num_stage > 1:
            dims = dims * self.num_stage
        if len(dims) < self.num_stage:
            dims = dims + [dims[-1]] * (self.num_stage - len(dims))
        dims = dims[: self.num_stage]
        self.dims = dims

        depths = list(depths)
        total_depth = sum(depths)
        if isinstance(drop_path_rate, (list, tuple)):
            dpr = [float(x) for x in _as_list(drop_path_rate, total_depth)]
        else:
            dpr = [x.item() for x in torch.linspace(0, float(drop_path_rate), total_depth)]
        layer_scale_value = _as_list(layer_scale_value, self.num_stage)
        use_pe = _as_list(use_pe, self.num_stage)
        use_lpu = _as_list(use_lpu, self.num_stage)
        local_kernel_size = _as_list(local_kernel_size, self.num_stage)
        use_dwc_mlp = _as_list(use_dwc_mlp, self.num_stage)
        heads = _as_list(heads, self.num_stage)
        attn_drop = _as_list(attn_drop, self.num_stage)
        proj_drop = _as_list(proj_drop, self.num_stage)
        stage_spec = _as_list(stage_spec, self.num_stage)
        window_size = _as_list(window_size, self.num_stage)
        nat_ksize = _as_list(nat_ksize, self.num_stage)
        ksize = _as_list(ksize, self.num_stage)
        stride = _as_list(stride, self.num_stage)
        n_groups = _as_list(n_groups, self.num_stage)
        offset_range_factor = _as_list(offset_range_factor, self.num_stage)
        no_off = _as_list(no_off, self.num_stage)
        dwc_pe = _as_list(dwc_pe, self.num_stage)
        fixed_pe = _as_list(fixed_pe, self.num_stage)
        log_cpb = _as_list(log_cpb, self.num_stage)

        self.downsample_layers = nn.ModuleList()
        current_len = seq_len

        if stem_ratio > 1:
            stem_stride = max(1, stem_ratio // 2)
            stem_mid_channels = max(1, dims[0] // 2)
            stem = nn.Sequential(
                nn.Conv1d(
                    1, stem_mid_channels, kernel_size=3, stride=2, padding=1
                ),
                LayerNormProxy(stem_mid_channels),
                nn.GELU(),
                nn.Conv1d(
                    stem_mid_channels,
                    dims[0],
                    kernel_size=min(stem_stride, _conv_out_length(current_len, 3, 2, 1)),
                    stride=min(stem_stride, _conv_out_length(current_len, 3, 2, 1)),
                ),
                LayerNormProxy(dims[0]),
            )
            current_len = _conv_out_length(current_len, 3, 2, 1)
            second_kernel = min(stem_stride, current_len)
            current_len = _conv_out_length(current_len, second_kernel, second_kernel)
        else:
            stem = nn.Sequential(
                nn.Conv1d(1, dims[0], kernel_size=1, stride=1),
                LayerNormProxy(dims[0]),
            )
        self.downsample_layers.append(stem)

        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                down_stride = max(1, int(down_ratio))
                down_kernel = min(down_stride, current_len)
                downsample_layer = nn.Sequential(
                    nn.Conv1d(
                        dims[i],
                        dims[i + 1],
                        kernel_size=down_kernel,
                        stride=down_kernel,
                    ),
                    LayerNormProxy(dims[i + 1]),
                )
                self.downsample_layers.append(downsample_layer)
                current_len = _conv_out_length(current_len, down_kernel, down_kernel)

        self.final_length = current_len
        self.stages = nn.ModuleList()
        for i in range(self.num_stage):
            self.stages.append(
                Stage(
                    current_len,
                    dims[i],
                    depths[i],
                    dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    layer_scale_value[i],
                    use_pe[i],
                    use_lpu[i],
                    local_kernel_size[i],
                    expansion,
                    drop,
                    use_dwc_mlp[i],
                    heads[i],
                    attn_drop[i],
                    proj_drop[i],
                    stage_spec[i],
                    window_size[i],
                    nat_ksize[i],
                    ksize[i],
                    stride[i],
                    n_groups[i],
                    offset_range_factor[i],
                    no_off[i],
                    dwc_pe[i],
                    fixed_pe[i],
                    log_cpb[i],
                )
            )

        self.c_out = c_out
        self.head_norm = LayerNormProxy(dims[self.num_stage - 1]) if use_head_norm else None
        self.head_type = head_type
        self.head_dropout = head_dropout

        if self.head_type != "Flatten":
            raise NotImplementedError("Only head_type='Flatten' is supported.")

        self.head = FlattenHead(
            individual=False,
            n_vars=self.n_vars,
            nf=self.dims[self.num_stage - 1] * self.final_length,
            target_window=pred_len * c_out,
            head_dropout=self.head_dropout,
        )

    def forward(self, x):
        if self.rev:
            x = self.revin(x, "norm")

        batch_size, seq_len, n_vars = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * n_vars, 1, seq_len)
        x = self.downsample_layers[0](x)

        for i in range(self.num_stage):
            x = self.stages[i](x)
            if i < self.num_stage - 1:
                x = self.downsample_layers[i + 1](x)

        if self.use_head_norm and self.head_norm is not None:
            x = self.head_norm(x)

        _, hidden_size, hidden_len = x.shape
        x = x.reshape(batch_size, n_vars, hidden_size, hidden_len)
        x = self.head(x)
        x = x.permute(0, 2, 1).reshape(batch_size, self.pred_len, -1)

        if self.rev and self.c_out == 1:
            x = self.revin(x, "denorm")

        return x


class DeformableTST(BaseModel):
    """DeformableTST.

    Lightweight NeuralForecast port of the DeformableTST architecture. The model
    keeps the original multi-stage stem/downsampling backbone, uses shared weights
    across variables, and exposes a direct multivariate forecast head.

    References:
        - Donghao Luo and Xue Wang. "DeformableTST: Transformer for Time Series
          Forecasting without Over-reliance on Patching". NeurIPS 2024.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True
    RECURRENT = False

    def __init__(
        self,
        h,
        input_size,
        n_series,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y: bool = False,
        stem_ratio: int = 8,
        down_ratio: int = 2,
        fmap_size: int = 768,
        dims=(64, 128, 256, 512),
        depths=(1, 1, 3, 1),
        drop_path_rate: float = 0.3,
        layer_scale_value=(-1, -1, -1, -1),
        use_pe=(1, 1, 1, 1),
        use_lpu=(1, 1, 1, 1),
        local_kernel_size=(3, 3, 3, 3),
        expansion: int = 4,
        drop: float = 0.0,
        use_dwc_mlp=(1, 1, 1, 1),
        heads=(4, 8, 16, 32),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        stage_spec=(("D",), ("D",), ("D", "D", "D"), ("D",)),
        window_size=(3, 3, 3, 3),
        nat_ksize=(3, 3, 3, 3),
        ksize=(9, 7, 5, 3),
        stride=(8, 4, 2, 1),
        n_groups=(2, 4, 8, 16),
        offset_range_factor=(-1, -1, -1, -1),
        no_off=(0, 0, 0, 0),
        dwc_pe=(0, 0, 0, 0),
        fixed_pe=(0, 0, 0, 0),
        log_cpb=(0, 0, 0, 0),
        head_dropout: float = 0.1,
        head_type: str = "Flatten",
        use_head_norm: bool = True,
        rev: bool = True,
        revin_affine: bool = False,
        revin_subtract_last: bool = False,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size=None,
        windows_batch_size=32,
        inference_windows_batch_size=32,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        trainer_kwargs = dict(trainer_kwargs)
        trainer_kwargs.setdefault("accelerator", "cpu")
        trainer_kwargs.setdefault("devices", 1)
        trainer_kwargs.setdefault("logger", False)
        trainer_kwargs.setdefault("enable_checkpointing", False)
        trainer_kwargs.setdefault("enable_progress_bar", False)

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

        c_out = self.loss.outputsize_multiplier
        self.backbone = DeformableTSTBackbone(
            n_series,
            rev,
            revin_affine,
            revin_subtract_last,
            stem_ratio,
            down_ratio,
            fmap_size,
            dims,
            depths,
            drop_path_rate,
            layer_scale_value,
            use_pe,
            use_lpu,
            local_kernel_size,
            expansion,
            drop,
            use_dwc_mlp,
            heads,
            attn_drop,
            proj_drop,
            stage_spec,
            window_size,
            nat_ksize,
            ksize,
            stride,
            n_groups,
            offset_range_factor,
            no_off,
            dwc_pe,
            fixed_pe,
            log_cpb,
            input_size,
            h,
            c_out,
            head_dropout,
            head_type,
            use_head_norm,
        )

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        return self.backbone(insample_y)
