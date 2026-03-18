from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import NHITS, TFT, TimeXer, VanillaTransformer, iTransformer

try:
    from tests.dummy.dummy_models import DummyMultivariate, DummyUnivariate
except Exception:  # pragma: no cover - test-only optional models
    DummyMultivariate = DummyUnivariate = None

from .config import AppConfig, JobConfig


@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    job_types_supported: tuple[str, ...]
    supports_hist_exog: bool
    supports_futr_exog: bool
    supports_stat_exog: bool
    supports_channel_map: bool
    requires_n_series: bool
    single_device_only: bool = True


MODEL_CLASSES = {
    'NHITS': NHITS,
    'TFT': TFT,
    'VanillaTransformer': VanillaTransformer,
    'iTransformer': iTransformer,
    'TimeXer': TimeXer,
}
if DummyUnivariate is not None:
    MODEL_CLASSES['DummyUnivariate'] = DummyUnivariate
if DummyMultivariate is not None:
    MODEL_CLASSES['DummyMultivariate'] = DummyMultivariate


def resolve_loss(name: str) -> Any:
    normalized = name.lower()
    if normalized != 'mse':
        raise ValueError(f'Unsupported common loss: {name}')
    return MSE()


def capabilities_for(model_name: str) -> ModelCapabilities:
    model_cls = MODEL_CLASSES[model_name]
    supports_hist = bool(getattr(model_cls, 'EXOGENOUS_HIST', False))
    supports_futr = bool(getattr(model_cls, 'EXOGENOUS_FUTR', False))
    supports_stat = bool(getattr(model_cls, 'EXOGENOUS_STAT', False))
    multivariate = bool(getattr(model_cls, 'MULTIVARIATE', False))
    job_types = ('multivariate_channels',) if multivariate else ('univariate_with_exog',)
    if multivariate and (supports_hist or supports_futr or supports_stat):
        job_types = ('multivariate_channels', 'multivariate_channels_exog')
    return ModelCapabilities(
        name=model_name,
        job_types_supported=job_types,
        supports_hist_exog=supports_hist,
        supports_futr_exog=supports_futr,
        supports_stat_exog=supports_stat,
        supports_channel_map=multivariate,
        requires_n_series='n_series' in inspect.signature(model_cls.__init__).parameters,
    )


def validate_job(job: JobConfig) -> ModelCapabilities:
    if job.model not in MODEL_CLASSES:
        raise ValueError(f'Unsupported model: {job.model}')
    caps = capabilities_for(job.model)
    if job.job_type not in caps.job_types_supported:
        raise ValueError(
            f'{job.model} does not support job_type={job.job_type}; '
            f'supported={caps.job_types_supported}'
        )
    if job.hist_exog_cols and not caps.supports_hist_exog:
        raise ValueError(f'{job.model} does not support historic exogenous columns')
    if job.futr_exog_cols and not caps.supports_futr_exog:
        raise ValueError(f'{job.model} does not support future exogenous columns')
    if job.static_exog_cols and not caps.supports_stat_exog:
        raise ValueError(f'{job.model} does not support static exogenous columns')
    return caps


def build_model(
    config: AppConfig,
    job: JobConfig,
    *,
    n_series: int | None = None,
) -> Any:
    caps = validate_job(job)
    model_cls = MODEL_CLASSES[job.model]
    shared_kwargs: dict[str, Any] = {
        'h': config.cv.horizon,
        'input_size': config.training.input_size,
        'max_steps': config.training.max_steps,
        'learning_rate': config.training.learning_rate,
        'batch_size': config.training.batch_size,
        'valid_batch_size': config.training.valid_batch_size,
        'windows_batch_size': config.training.windows_batch_size,
        'inference_windows_batch_size': config.training.inference_windows_batch_size,
        'random_seed': config.runtime.random_seed,
        'alias': job.name,
        'accelerator': 'gpu' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        'devices': 1,
        'enable_checkpointing': False,
        'loss': resolve_loss(config.training.loss),
        'valid_loss': resolve_loss(config.training.loss),
        'hist_exog_list': list(job.hist_exog_cols),
        'futr_exog_list': list(job.futr_exog_cols),
        'stat_exog_list': list(job.static_exog_cols),
    }
    if caps.requires_n_series:
        if n_series is None:
            raise ValueError(f'{job.model} requires n_series')
        shared_kwargs['n_series'] = n_series
    signature = inspect.signature(model_cls.__init__)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted = {}
    for key, value in {**shared_kwargs, **job.params}.items():
        if key in signature.parameters or accepts_var_kwargs:
            accepted[key] = value
    return model_cls(**accepted)
