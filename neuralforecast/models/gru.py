__all__ = ["GRU"]


import warnings
from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..common._modules import MLP
from ..losses.pytorch import MAE
from .aaforecast.gru import _align_horizon, _apply_stochastic_dropout, _build_encoder


class GRU(BaseModel):
    """GRU

    Public GRU baseline aligned to the AAForecast parity path.

    This model keeps the shared GRU encoder / horizon-alignment / MLP-decoder
    skeleton comparable to `AAForecast`, while intentionally omitting the
    AA-specific STAR feature augmentation and sparse-attention context path.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        inference_input_size: Optional[int] = None,
        h_train: int = 1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 200,
        encoder_activation: Optional[str] = None,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        context_size: Optional[int] = None,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y: bool = False,
        recurrent: bool = False,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 1000,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size=32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=128,
        inference_windows_batch_size=1024,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "robust",
        random_seed=1,
        drop_last_loader=False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        if recurrent:
            raise ValueError(
                "GRU recurrent=True is not supported in the AA-parity GRU path"
            )
        if futr_exog_list:
            raise ValueError(
                "GRU futr_exog_list is not supported in the AA-parity GRU path"
            )
        if stat_exog_list:
            raise ValueError(
                "GRU stat_exog_list is not supported in the AA-parity GRU path"
            )

        if encoder_activation is not None:
            warnings.warn(
                "The 'encoder_activation' argument is deprecated and will be removed in "
                "future versions. The activation function in GRU is frozen in PyTorch and "
                "it cannot be modified.",
                DeprecationWarning,
            )
        if context_size is not None:
            warnings.warn(
                "context_size is deprecated and will be removed in future versions."
            )

        super().__init__(
            h=h,
            input_size=input_size,
            inference_input_size=inference_input_size,
            h_train=h_train,
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

        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_bias = encoder_bias
        self.encoder_dropout = encoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers
        self.exclude_insample_y = exclude_insample_y
        self._stochastic_inference_enabled = False
        self._stochastic_dropout_p = float(encoder_dropout)
        self.horizon_embedding_dim = min(16, max(4, self.encoder_hidden_size // 16))

        feature_size = (0 if exclude_insample_y else 1) + self.hist_exog_size
        if feature_size <= 0:
            raise ValueError(
                "GRU parity path requires insample_y and/or hist_exog features"
            )

        self.encoder = _build_encoder(
            feature_size=feature_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_n_layers,
            dropout=encoder_dropout,
            bias=encoder_bias,
        )
        self.sequence_adapter = (
            torch.nn.Linear(self.input_size, self.h)
            if self.h > self.input_size
            else None
        )
        self.horizon_embeddings = nn.Embedding(
            num_embeddings=self.h,
            embedding_dim=self.horizon_embedding_dim,
        )
        self.horizon_context = nn.Linear(
            self.horizon_embedding_dim,
            self.encoder_hidden_size,
        )
        self.decoder = MLP(
            in_features=2 * encoder_hidden_size,
            out_features=self.loss.outputsize_multiplier,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            activation="ReLU",
            dropout=encoder_dropout,
        )

        # Compatibility aliases for existing downstream probes.
        self.hist_encoder = self.encoder
        self.upsample_sequence = self.sequence_adapter
        self.mlp_decoder = self.decoder

    def configure_stochastic_inference(
        self,
        *,
        enabled: bool,
        dropout_p: float | None = None,
    ) -> None:
        self._stochastic_inference_enabled = bool(enabled)
        if dropout_p is not None:
            self._stochastic_dropout_p = float(dropout_p)

    def _build_horizon_context(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        horizon_ids = torch.arange(self.h, device=device)
        horizon_embeddings = self.horizon_embeddings(horizon_ids)
        horizon_embeddings = horizon_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return self.horizon_context(horizon_embeddings)

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        hist_exog = windows_batch["hist_exog"]

        encoder_parts = []
        if not self.exclude_insample_y:
            encoder_parts.append(insample_y)
        if self.hist_exog_size > 0:
            encoder_parts.append(hist_exog)
        if not encoder_parts:
            raise ValueError(
                "GRU parity path requires insample_y and/or hist_exog features"
            )
        encoder_input = torch.cat(encoder_parts, dim=2)

        hidden_states, _ = self.encoder(encoder_input)
        hidden_aligned = _align_horizon(
            hidden_states,
            h=self.h,
            input_size=self.input_size,
            sequence_adapter=self.sequence_adapter,
        )
        context_aligned = self._build_horizon_context(
            batch_size=hidden_aligned.shape[0],
            device=hidden_aligned.device,
        )
        hidden_aligned = _apply_stochastic_dropout(
            hidden_aligned,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        context_aligned = _apply_stochastic_dropout(
            context_aligned,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        decoder_input = torch.cat([hidden_aligned, context_aligned], dim=-1)
        decoder_input = _apply_stochastic_dropout(
            decoder_input,
            training=self.training,
            stochastic_inference_enabled=self._stochastic_inference_enabled,
            train_dropout_p=self.encoder_dropout,
            inference_dropout_p=self._stochastic_dropout_p,
        )
        return self.decoder(decoder_input)[:, -self.h :]
