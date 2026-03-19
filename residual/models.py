from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import Autoformer, FEDformer, Informer, LSTM, NHITS, PatchTST, TFT, VanillaTransformer, iTransformer

try:
    from tests.dummy.dummy_models import DummyMultivariate, DummyUnivariate
except Exception:  # pragma: no cover
    DummyMultivariate = DummyUnivariate = None

from .config import AppConfig, JobConfig


@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    multivariate: bool
    supports_hist_exog: bool
    supports_futr_exog: bool
    supports_stat_exog: bool
    requires_n_series: bool
    single_device_only: bool = True


MODEL_CLASSES = {
    'TFT': TFT,
    'VanillaTransformer': VanillaTransformer,
    'Informer': Informer,
    'Autoformer': Autoformer,
    'FEDformer': FEDformer,
    'PatchTST': PatchTST,
    'LSTM': LSTM,
    'NHITS': NHITS,
    'iTransformer': iTransformer,
}
if DummyUnivariate is not None:
    MODEL_CLASSES['DummyUnivariate'] = DummyUnivariate
if DummyMultivariate is not None:
    MODEL_CLASSES['DummyMultivariate'] = DummyMultivariate

BASELINE_MODEL_NAMES = {'Naive', 'SeasonalNaive', 'HistoricAverage'}


def resolve_loss(name: str) -> Any:
    normalized = name.lower()
    if normalized != 'mse':
        raise ValueError(f'Unsupported common loss: {name}')
    return MSE()


def capabilities_for(model_name: str) -> ModelCapabilities:
    if model_name in BASELINE_MODEL_NAMES:
        return ModelCapabilities(model_name, False, False, False, False, False)
    model_cls = MODEL_CLASSES[model_name]
    return ModelCapabilities(
        name=model_name,
        multivariate=bool(getattr(model_cls, 'MULTIVARIATE', False)),
        supports_hist_exog=bool(getattr(model_cls, 'EXOGENOUS_HIST', False)),
        supports_futr_exog=bool(getattr(model_cls, 'EXOGENOUS_FUTR', False)),
        supports_stat_exog=bool(getattr(model_cls, 'EXOGENOUS_STAT', False)),
        requires_n_series='n_series' in inspect.signature(model_cls.__init__).parameters,
    )


def validate_job(job: JobConfig) -> ModelCapabilities:
    if job.model not in MODEL_CLASSES and job.model not in BASELINE_MODEL_NAMES:
        raise ValueError(f'Unsupported model: {job.model}')
    return capabilities_for(job.model)


def build_model(config: AppConfig, job: JobConfig, *, n_series: int | None = None) -> Any:
    caps = validate_job(job)
    if job.model in BASELINE_MODEL_NAMES:
        raise ValueError(f'{job.model} is a baseline model and is not built via neuralforecast model constructors')
    model_cls = MODEL_CLASSES[job.model]
    shared_kwargs: dict[str, Any] = {
        'h': config.cv.horizon,
        'input_size': config.training.input_size,
        'max_steps': config.training.max_steps,
        'learning_rate': config.training.learning_rate,
        'val_check_steps': config.training.val_check_steps,
        'early_stop_patience_steps': config.training.early_stop_patience_steps,
        'batch_size': config.training.batch_size,
        'valid_batch_size': config.training.valid_batch_size,
        'windows_batch_size': config.training.windows_batch_size,
        'inference_windows_batch_size': config.training.inference_windows_batch_size,
        'random_seed': config.runtime.random_seed,
        'alias': job.model,
        'accelerator': 'gpu' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        'devices': 1,
        'enable_checkpointing': False,
        'loss': resolve_loss(config.training.loss),
        'valid_loss': resolve_loss(config.training.loss),
        'hist_exog_list': list(config.dataset.hist_exog_cols) if caps.supports_hist_exog and not caps.multivariate else [],
        'futr_exog_list': list(config.dataset.futr_exog_cols) if caps.supports_futr_exog and not caps.multivariate else [],
        'stat_exog_list': list(config.dataset.static_exog_cols) if caps.supports_stat_exog else [],
    }
    if caps.requires_n_series:
        shared_kwargs['n_series'] = 1 if n_series is None else n_series
    signature = inspect.signature(model_cls.__init__)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    accepted = {}
    for key, value in {**shared_kwargs, **job.params}.items():
        if key in signature.parameters or accepts_var_kwargs:
            accepted[key] = value
    return model_cls(**accepted)
