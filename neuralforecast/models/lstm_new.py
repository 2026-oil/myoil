__all__ = ["LSTM_new"]

import warnings
from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..common._modules import MLP
from ..losses.pytorch import MAE


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        attn_scores = self.attn_weights(lstm_output)
        attn_weights = self.softmax(attn_scores)
        context = torch.sum(lstm_output * attn_weights, dim=1)
        return context, attn_weights


class LSTM_new(BaseModel):
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True
    MULTIVARIATE = False
    RECURRENT = True

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        inference_input_size: Optional[int] = None,
        h_train: int = 1,
        encoder_n_layers: int = 2,
        encoder_hidden_size: int = 64,
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        decoder_hidden_size: int = 64,
        decoder_layers: int = 1,
        futr_exog_list=None,
        hist_exog_list=None,
        stat_exog_list=None,
        exclude_insample_y: bool = False,
        recurrent: bool = True,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 1,
        batch_size=32,
        valid_batch_size=32,
        windows_batch_size=1024,
        inference_windows_batch_size=1024,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "robust",
        random_seed: int = 1,
        drop_last_loader=False,
        optimizer=None,
        optimizer_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        self.RECURRENT = recurrent

        super().__init__(
            h=h,
            input_size=input_size,
            inference_input_size=inference_input_size,
            h_train=h_train,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            exclude_insample_y=exclude_insample_y,
            recurrent=recurrent,
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
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        self.save_hyperparameters()
        self.input_size = input_size
        self.inference_input_size = inference_input_size

        input_encoder = (
            1 + self.hist_exog_size + self.stat_exog_size + self.futr_exog_size
        )
        self.encoder = nn.LSTM(
            input_size=input_encoder,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_n_layers,
            bias=encoder_bias,
            dropout=encoder_dropout if encoder_n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = TemporalAttention(encoder_hidden_size)

        decoder_input_size = encoder_hidden_size + self.futr_exog_size
        self.decoder = MLP(
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            in_features=decoder_input_size,
            out_features=h,
            activation="ReLU",
            dropout=0.0,
        )

    def forward(self, insample_y, futr_exog, hist_exog, stat_exog):
        if self.exclude_insample_y:
            insample_y = torch.zeros_like(insample_y)

        insample_y = insample_y.unsqueeze(-1)

        inputs = [insample_y]
        if self.hist_exog_list:
            inputs.append(hist_exog)
        x = torch.cat(inputs, dim=-1)

        lstm_output, _ = self.encoder(x)

        context, _ = self.attention(lstm_output)

        decoder_inputs = [context]
        if self.futr_exog_list:
            futr_exog_mean = futr_exog.mean(dim=1)
            decoder_inputs.append(futr_exog_mean)

        decoder_input = torch.cat(decoder_inputs, dim=-1)
        y_hat = self.decoder(decoder_input)

        return y_hat
