


__all__ = ['iTransformer']


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast.common._modules import (
    AttentionLayer,
    DataEmbedding_inverted,
    FullAttention,
    TransEncoder,
    TransEncoderLayer,
)

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class iTransformer(BaseModel):
    """iTransformer

    Args:
        h (int): Forecast horizon.
        input_size (int): autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].
        n_series (int): number of time-series.
        futr_exog_list (str list): future exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        stat_exog_list (str list): static exogenous columns.
        exclude_insample_y (bool): the model skips the autoregressive features y[t-input_size:t] if True.
        hidden_size (int): dimension of the model.
        n_heads (int): number of heads.
        e_layers (int): number of encoder layers.
        d_layers (int): number of decoder layers.
        d_ff (int): dimension of fully-connected layer.
        factor (int): attention factor.
        dropout (float): dropout rate.
        use_norm (bool): whether to normalize or not.
        loss (PyTorch module): instantiated train loss class from [losses collection](./losses.pytorch).
        valid_loss (PyTorch module): instantiated valid loss class from [losses collection](./losses.pytorch).
        max_steps (int): maximum number of training steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
        val_check_steps (int): Number of training steps between every validation loss check.
        batch_size (int): number of different series in each batch.
        valid_batch_size (int): number of different series in each validation and test batch, if None uses batch_size.
        windows_batch_size (int): number of windows to sample in each training batch, default uses all.
        inference_windows_batch_size (int): number of windows to sample in each inference batch, -1 uses all.
        start_padding_enabled (bool): if True, the model will pad the time series with zeros at the beginning, by input size.
        training_data_availability_threshold (Union[float, List[float]]): minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).
        step_size (int): step size between each window of temporal data.
        scaler_type (str): type of scaler for temporal inputs normalization see [temporal scalers](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_scalers.py).
        random_seed (int): random_seed for pytorch initializer and numpy generators.
        drop_last_loader (bool): if True `TimeSeriesDataLoader` drops last non-full batch.
        alias (str): optional,  Custom name of the model.
        optimizer (Subclass of 'torch.optim.Optimizer'): optional, user specified optimizer instead of the default choice (Adam).
        optimizer_kwargs (dict): optional, list of parameters used by the user specified `optimizer`.
        dataloader_kwargs (dict): optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`.
        **trainer_kwargs (int):  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

    References:
        - [Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"](https://arxiv.org/abs/2310.06625)
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
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
        exclude_insample_y=False,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        factor: int = 1,
        dropout: float = 0.1,
        use_norm: bool = True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
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
        dataloader_kwargs=None,
        **trainer_kwargs
    ):

        super(iTransformer, self).__init__(
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
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
        )

        self.enc_in = n_series
        self.dec_in = n_series
        self.c_out = n_series
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.use_norm = use_norm

        # Architecture
        self.enc_embedding = DataEmbedding_inverted(
            input_size, self.hidden_size, self.dropout
        )
        self.futr_exog_embedding = (
            nn.Linear(input_size + h, self.hidden_size)
            if self.futr_exog_size > 0
            else None
        )

        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, self.factor, attention_dropout=self.dropout
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    self.hidden_size,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=F.gelu,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_size),
        )

        self.projector = nn.Linear(
            self.hidden_size, h * self.loss.outputsize_multiplier, bias=True
        )

    @staticmethod
    def _check_finite(name: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            return
        finite_mask = torch.isfinite(tensor)
        if finite_mask.all():
            return

        bad_count = int((~finite_mask).sum().item())
        raise ValueError(
            f"iTransformer received non-finite values in {name}: "
            f"count={bad_count}, shape={tuple(tensor.shape)}"
        )

    @staticmethod
    def _normalize_hist_exog(hist_exog: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if hist_exog is None:
            return None
        means = hist_exog.mean(dim=2, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(hist_exog, dim=2, keepdim=True, unbiased=False) + 1e-5
        )
        return (hist_exog - means) / stdev

    def forecast(self, x_enc, x_mark_enc=None, futr_exog=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            variance = torch.var(x_enc, dim=1, keepdim=True, unbiased=False)
            stdev = torch.sqrt(variance + 1e-5)
            scale = torch.where(variance < 1e-5, torch.ones_like(stdev), stdev)
            x_enc /= scale

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;       E: hidden_size;
        # L: input_size;       S: horizon(h);
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(
            x_enc, x_mark_enc
        )  # covariates (e.g timestamp) can be also embedded as tokens

        if self.futr_exog_embedding is not None and futr_exog is not None:
            futr_exog_tokens = futr_exog.permute(0, 3, 1, 2).reshape(
                futr_exog.shape[0], -1, self.input_size + self.h
            )
            futr_exog_tokens = self.futr_exog_embedding(futr_exog_tokens)
            enc_out = torch.cat([enc_out, futr_exog_tokens], dim=1)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (
                scale[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.h * self.loss.outputsize_multiplier, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.h * self.loss.outputsize_multiplier, 1)
            )

        return dec_out

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        hist_exog = windows_batch["hist_exog"]
        futr_exog = windows_batch.get("futr_exog")

        self._check_finite("insample_y", insample_y)
        self._check_finite("hist_exog", hist_exog)
        self._check_finite("futr_exog", futr_exog)
        hist_exog = self._normalize_hist_exog(hist_exog)
        self._check_finite("hist_exog_normalized", hist_exog)

        x_mark_enc = None
        if self.hist_exog_size > 0:
            x_mark_enc = hist_exog.permute(0, 2, 1, 3).reshape(
                insample_y.shape[0], self.input_size, -1
            )
            self._check_finite("x_mark_enc", x_mark_enc)

        futr_exog_tokens = None
        if (
            self.futr_exog_embedding is not None
            and futr_exog is not None
            and futr_exog.shape[1] > 0
        ):
            futr_exog_tokens = futr_exog

        if futr_exog_tokens is None:
            y_pred = self.forecast(insample_y, x_mark_enc=x_mark_enc)
        else:
            y_pred = self.forecast(
                insample_y,
                x_mark_enc=x_mark_enc,
                futr_exog=futr_exog_tokens,
            )
        self._check_finite("y_pred", y_pred)
        if self.hist_exog_size > 0:
            y_pred = y_pred[:, :, : self.n_series]
        y_pred = y_pred.reshape(insample_y.shape[0], self.h, -1)

        return y_pred

# === AAForecast seam: encoder-only iTransformer helper ===
class ITransformerTokenEncoderOnly(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        factor: int,
        dropout: float,
        use_norm: bool,
    ) -> None:
        super().__init__()
        self.use_norm = use_norm
        self.enc_embedding = DataEmbedding_inverted(input_size, hidden_size, dropout)
        self.encoder = TransEncoder(
            [
                TransEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    d_ff,
                    dropout=dropout,
                    activation=F.gelu,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            centered = x - means
            variance = torch.var(centered, dim=1, keepdim=True, unbiased=False)
            scale = torch.where(variance < 1e-5, torch.ones_like(variance), variance.sqrt())
            x = centered / scale
        enc_out = self.enc_embedding(x, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return enc_out
