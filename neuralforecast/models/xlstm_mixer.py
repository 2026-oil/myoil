__all__ = ['xLSTMMixer']


from typing import Optional

import torch.nn as nn

from ..common._base_model import BaseModel
from ..common._modules import RevINMultivariate
from ..losses.pytorch import MAE
from .tsmixer import MixingLayer

try:
    from xlstm.blocks.mlstm.block import mLSTMBlockConfig
    from xlstm.blocks.slstm.block import sLSTMBlockConfig
    from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

    IS_XLSTM_INSTALLED = True
except ImportError:
    IS_XLSTM_INSTALLED = False


class xLSTMMixer(BaseModel):
    """xLSTMMixer

    xLSTMMixer combines an xLSTM temporal encoder with TSMixer-style
    latent refinements for multivariate forecasting. The model keeps the
    repo-native multivariate contract while reusing the optional ``xlstm``
    dependency gate already established by :class:`neuralforecast.models.xLSTM`.

    Args:
        h (int): Forecast horizon.
        input_size (int): Considered autoregressive inputs.
        n_series (int): Number of time series.
        hidden_size (int): Latent dimension used by the xLSTM encoder.
        n_block (int): Number of mixer refinement blocks.
        ff_dim (int): Hidden dimension of mixer feed-forward layers.
        dropout (float): Dropout rate.
        revin (bool): If True applies RevIN normalization.
        encoder_n_blocks (int): Number of xLSTM backbone blocks.
        encoder_bias (bool): Whether the xLSTM backbone uses biases.
        encoder_dropout (float): Dropout inside the xLSTM backbone.
        backbone (str): xLSTM backbone variant, either ``"sLSTM"`` or ``"mLSTM"``.
        loss (PyTorch module): Training loss from ``neuralforecast.losses.pytorch``.
        valid_loss (PyTorch module): Optional validation loss.

    References:
        - [Maximilian Beck et al. (2024). "xLSTM: Extended Long Short-Term Memory."](https://arxiv.org/abs/2405.04517)
        - [Chen et al. (2023). "TSMixer: An All-MLP Architecture for Time Series Forecasting."](http://arxiv.org/abs/2303.06053)
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
        exclude_insample_y=False,
        hidden_size: int = 128,
        n_block: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        revin: bool = True,
        encoder_n_blocks: int = 2,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.1,
        backbone: str = "mLSTM",
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
        **trainer_kwargs,
    ):
        if not IS_XLSTM_INSTALLED:
            raise ImportError(
                "Please install `xlstm`. You also need to install `mlstm_kernels` for backend='mLSTM' and `ninja` for backend='sLSTM'."
            )


        super(xLSTMMixer, self).__init__(
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
            **trainer_kwargs,
        )

        self.revin = revin
        if self.revin:
            self.norm = RevINMultivariate(num_features=n_series, affine=True)

        self.input_projection = nn.Linear(n_series, hidden_size)
        if backbone == "sLSTM":
            stack_config = xLSTMBlockStackConfig(
                slstm_block=sLSTMBlockConfig(),
                context_length=input_size,
                num_blocks=encoder_n_blocks,
                embedding_dim=hidden_size,
                bias=encoder_bias,
                dropout=encoder_dropout,
            )
        elif backbone == "mLSTM":
            stack_config = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(),
                context_length=input_size,
                num_blocks=encoder_n_blocks,
                embedding_dim=hidden_size,
                bias=encoder_bias,
                dropout=encoder_dropout,
            )
        else:
            raise ValueError("backbone must be either 'sLSTM' or 'mLSTM'.")

        self.hist_encoder = xLSTMBlockStack(stack_config)
        self.mixing_layers = nn.Sequential(
            *[
                MixingLayer(
                    n_series=hidden_size,
                    input_size=input_size,
                    dropout=dropout,
                    ff_dim=ff_dim,
                )
                for _ in range(n_block)
            ]
        )
        self.temporal_projection = nn.Linear(
            input_size, h * self.loss.outputsize_multiplier
        )
        self.output_projection = nn.Linear(hidden_size, n_series)

    def forward(self, windows_batch):
        x = windows_batch["insample_y"]
        batch_size = x.shape[0]

        if self.revin:
            x = self.norm(x, "norm")

        x = self.input_projection(x)
        x = self.hist_encoder(x)
        x = self.mixing_layers(x)
        x = x.permute(0, 2, 1)
        x = self.temporal_projection(x)
        x = x.permute(0, 2, 1)
        x = self.output_projection(x)

        if self.revin:
            x = self.norm(x, "denorm")

        x = x.reshape(
            batch_size, self.h, self.loss.outputsize_multiplier * self.n_series
        )
        return x
