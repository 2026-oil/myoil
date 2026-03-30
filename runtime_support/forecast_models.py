from __future__ import annotations

import inspect
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
from neuralforecast.losses.pytorch import ExLoss, MSE, _weighted_mean
from neuralforecast.models import (
    Autoformer,
    BiTCN,
    DLinear,
    DeepAR,
    DeepNPTS,
    DeformTime,
    DeformableTST,
    DilatedRNN,
    FEDformer,
    GRU,
    Informer,
    KAN,
    LSTM,
    MLP,
    MLPMultivariate,
    NBEATS,
    NBEATSx,
    NHITS,
    NLinear,
    NonstationaryTransformer,
    PatchTST,
    RMoK,
    RNN,
    SOFTS,
    StemGNN,
    TCN,
    TFT,
    TimeLLM,
    TSMixer,
    ModernTCN,
    TSMixerx,
    TiDE,
    TimeMixer,
    TimeXer,
    TimesNet,
    DUET,
    VanillaTransformer,
    XLinear,
    iTransformer,
    xLSTM,
)
from neuralforecast.models.cmamba import CMamba
from neuralforecast.models.bs_preforcast_catalog import DIRECT_STAGE_MODEL_NAMES
from neuralforecast.models.mamba import Mamba
from neuralforecast.models.smamba import SMamba
from neuralforecast.models.xlstm_mixer import xLSTMMixer
from plugins.optimizer import resolve_optimizer

from app_config import AppConfig, JobConfig, TrainingLossParams
from tuning.search_space import (
    BASELINE_MODEL_NAMES,
    SUPPORTED_MODEL_AUTO_MODEL_NAMES,
)


@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    multivariate: bool
    supports_hist_exog: bool
    supports_futr_exog: bool
    supports_stat_exog: bool
    requires_n_series: bool
    single_device_only: bool = True


class ForecastL1Loss(torch.nn.L1Loss):
    """NeuralForecast-compatible adapter around torch.nn.L1Loss.

    Uses torch.nn.L1Loss(reduction="none") for the elementwise absolute error,
    then applies the same mask / horizon weighting contract expected by the
    forecasting runtime.
    """

    def __init__(self, horizon_weight=None):
        super().__init__(reduction="none")
        if horizon_weight is not None:
            horizon_weight = torch.Tensor(horizon_weight.flatten())
        self.horizon_weight = horizon_weight
        self.outputsize_multiplier = 1
        self.output_names = [""]
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor) -> torch.Tensor:
        return y_hat

    def _compute_weights(self, y: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(y)

        if self.horizon_weight is None:
            weights = torch.ones_like(mask)
        else:
            assert mask.shape[1] == len(
                self.horizon_weight
            ), "horizon_weight must have same length as Y"
            weights = self.horizon_weight.clone()
            weights = weights[None, :, None].to(mask.device)
            weights = torch.ones_like(mask, device=mask.device) * weights

        return weights * mask

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: torch.Tensor | None = None,
        y_insample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del y_insample
        losses = super().forward(y_hat, y)
        weights = self._compute_weights(y=y, mask=mask)
        return _weighted_mean(losses=losses, weights=weights)


MODEL_CLASSES = {
    "RNN": RNN,
    "GRU": GRU,
    "TCN": TCN,
    "DeepAR": DeepAR,
    "DilatedRNN": DilatedRNN,
    "BiTCN": BiTCN,
    "xLSTM": xLSTM,
    "MLP": MLP,
    "NBEATS": NBEATS,
    "NBEATSx": NBEATSx,
    "DLinear": DLinear,
    "NLinear": NLinear,
    "TiDE": TiDE,
    "DeepNPTS": DeepNPTS,
    "DeformTime": DeformTime,
    "DeformableTST": DeformableTST,
    "KAN": KAN,
    "TFT": TFT,
    "VanillaTransformer": VanillaTransformer,
    "Informer": Informer,
    "Autoformer": Autoformer,
    "FEDformer": FEDformer,
    "PatchTST": PatchTST,
    "LSTM": LSTM,
    "NHITS": NHITS,
    "iTransformer": iTransformer,
    "TimeLLM": TimeLLM,
    "TimeXer": TimeXer,
    "ModernTCN": ModernTCN,
    "NonstationaryTransformer": NonstationaryTransformer,
    "TimesNet": TimesNet,
    "StemGNN": StemGNN,
    "TSMixer": TSMixer,
    "TSMixerx": TSMixerx,
    "MLPMultivariate": MLPMultivariate,
    "SOFTS": SOFTS,
    "TimeMixer": TimeMixer,
    "DUET": DUET,
    "Mamba": Mamba,
    "SMamba": SMamba,
    "CMamba": CMamba,
    "xLSTMMixer": xLSTMMixer,
    "RMoK": RMoK,
    "XLinear": XLinear,
}


def resolve_loss(
    name: str, *, loss_params: TrainingLossParams | None = None
) -> Any:
    normalized = name.lower()
    if normalized == "mae":
        return ForecastL1Loss()
    if normalized == "mse":
        return MSE()
    if normalized == "exloss":
        return ExLoss(**asdict(loss_params or TrainingLossParams()))
    raise ValueError(f"Unsupported common loss: {name}")


def capabilities_for(model_name: str) -> ModelCapabilities:
    if model_name in BASELINE_MODEL_NAMES:
        return ModelCapabilities(model_name, False, False, False, False, False)
    if model_name in DIRECT_STAGE_MODEL_NAMES:
        return ModelCapabilities(model_name, False, False, False, False, False)
    model_cls = MODEL_CLASSES[model_name]
    return ModelCapabilities(
        name=model_name,
        multivariate=bool(getattr(model_cls, "MULTIVARIATE", False)),
        supports_hist_exog=bool(getattr(model_cls, "EXOGENOUS_HIST", False)),
        supports_futr_exog=bool(getattr(model_cls, "EXOGENOUS_FUTR", False)),
        supports_stat_exog=bool(getattr(model_cls, "EXOGENOUS_STAT", False)),
        requires_n_series="n_series" in inspect.signature(model_cls.__init__).parameters,
    )


def validate_job(job: JobConfig) -> ModelCapabilities:
    if (
        job.model not in MODEL_CLASSES
        and job.model not in BASELINE_MODEL_NAMES
        and job.model not in DIRECT_STAGE_MODEL_NAMES
    ):
        raise ValueError(f"Unsupported model: {job.model}")
    return capabilities_for(job.model)


def resolved_devices(config: AppConfig) -> int | None:
    worker_devices = os.environ.get("NEURALFORECAST_WORKER_DEVICES")
    if config.training.devices is not None:
        if worker_devices:
            return min(int(config.training.devices), int(worker_devices))
        return int(config.training.devices)
    if worker_devices:
        return int(worker_devices)
    return int(config.scheduler.worker_devices)


def resolved_strategy_name(config: AppConfig, devices: int | None) -> str | None:
    if config.training.strategy is not None:
        return config.training.strategy
    if devices is None or devices <= 1:
        return None
    return "ddp-gloo-auto"


def resolved_strategy(config: AppConfig, devices: int | None) -> Any:
    strategy_name = resolved_strategy_name(config, devices)
    if strategy_name is None:
        return None
    if config.training.strategy is not None:
        return strategy_name
    from pytorch_lightning.strategies import DDPStrategy

    return DDPStrategy(process_group_backend="gloo")


def _resolved_dataloader_kwargs(config: AppConfig) -> dict[str, Any]:
    return dict(config.training.dataloader_kwargs)


def build_model(
    config: AppConfig,
    job: JobConfig,
    *,
    n_series: int | None = None,
    params_override: dict[str, Any] | None = None,
) -> Any:
    caps = validate_job(job)
    if job.model in BASELINE_MODEL_NAMES:
        raise ValueError(
            f"{job.model} is a baseline model and is not built via neuralforecast model constructors"
        )
    if job.model in DIRECT_STAGE_MODEL_NAMES:
        raise ValueError(
            f"{job.model} is a direct top-level model and must run via the dedicated direct runtime path"
        )
    model_cls = MODEL_CLASSES[job.model]

    def _configured_exog_list(
        supported: bool, values: tuple[str, ...]
    ) -> list[str] | None:
        if not supported or not values:
            return None
        return list(values)

    optimizer_resolution = resolve_optimizer(
        config.training.optimizer.name,
        config.training.optimizer.kwargs,
    )
    shared_kwargs: dict[str, Any] = {
        "h": config.cv.horizon,
        "input_size": config.training.input_size,
        "max_steps": config.training.max_steps,
        "scaler_type": config.training.scaler_type,
        "step_size": config.training.model_step_size,
        "val_check_steps": config.training.val_check_steps,
        "min_steps_before_early_stop": config.training.min_steps_before_early_stop,
        "early_stop_patience_steps": config.training.early_stop_patience_steps,
        "batch_size": config.training.batch_size,
        "valid_batch_size": config.training.valid_batch_size,
        "windows_batch_size": config.training.windows_batch_size,
        "inference_windows_batch_size": config.training.inference_windows_batch_size,
        "optimizer": optimizer_resolution.optimizer_cls,
        "optimizer_kwargs": optimizer_resolution.kwargs,
        "random_seed": config.runtime.random_seed,
        "alias": job.model,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "logger": False,
        "loss": resolve_loss(
            config.training.loss,
            loss_params=config.training.loss_params,
        ),
        "valid_loss": resolve_loss(
            config.training.loss,
            loss_params=config.training.loss_params,
        ),
        "dataloader_kwargs": _resolved_dataloader_kwargs(config),
        "hist_exog_list": _configured_exog_list(
            caps.supports_hist_exog, config.dataset.hist_exog_cols
        ),
        "futr_exog_list": _configured_exog_list(
            caps.supports_futr_exog, config.dataset.futr_exog_cols
        ),
        "stat_exog_list": _configured_exog_list(
            caps.supports_stat_exog, config.dataset.static_exog_cols
        ),
    }
    resolved_device_count = resolved_devices(config)
    resolved_strategy_value = resolved_strategy(config, resolved_device_count)
    if config.training.accelerator is not None:
        shared_kwargs["accelerator"] = config.training.accelerator
    if resolved_device_count is not None:
        shared_kwargs["devices"] = resolved_device_count
    if resolved_strategy_value is not None:
        shared_kwargs["strategy"] = resolved_strategy_value
    if config.training.precision is not None:
        shared_kwargs["precision"] = config.training.precision
    if caps.requires_n_series:
        shared_kwargs["n_series"] = 1 if n_series is None else n_series
    signature = inspect.signature(model_cls.__init__)
    scheduler_kwargs = {
        "max_lr": config.training.lr_scheduler.max_lr,
        "total_steps": config.training.max_steps,
        "pct_start": config.training.lr_scheduler.pct_start,
        "div_factor": config.training.lr_scheduler.div_factor,
        "final_div_factor": config.training.lr_scheduler.final_div_factor,
        "anneal_strategy": config.training.lr_scheduler.anneal_strategy,
        "three_phase": config.training.lr_scheduler.three_phase,
        "cycle_momentum": config.training.lr_scheduler.cycle_momentum,
    }
    if "max_lr" in signature.parameters:
        shared_kwargs["max_lr"] = config.training.lr_scheduler.max_lr
    else:
        shared_kwargs["_max_lr"] = config.training.lr_scheduler.max_lr
    if "lr_scheduler" in signature.parameters:
        shared_kwargs["lr_scheduler"] = torch.optim.lr_scheduler.OneCycleLR
    else:
        shared_kwargs["_lr_scheduler_cls"] = torch.optim.lr_scheduler.OneCycleLR
    if "lr_scheduler_kwargs" in signature.parameters:
        shared_kwargs["lr_scheduler_kwargs"] = scheduler_kwargs
    else:
        shared_kwargs["_lr_scheduler_kwargs"] = scheduler_kwargs
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    accepted = {}
    for key, value in {**shared_kwargs, **job.params, **(params_override or {})}.items():
        if key in signature.parameters or accepts_var_kwargs:
            accepted[key] = value
    model = model_cls(**accepted)
    if hasattr(model, "hparams"):
        model.hparams["max_lr"] = config.training.lr_scheduler.max_lr
    return model


def supports_auto_mode(model_name: str) -> bool:
    return model_name in SUPPORTED_MODEL_AUTO_MODEL_NAMES


def is_direct_top_level_model(model_name: str) -> bool:
    return model_name in DIRECT_STAGE_MODEL_NAMES
