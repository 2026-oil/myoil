from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import hashlib
import json
import tomllib

import yaml

CONFIG_FILENAMES = ('config.yaml', 'config.yml', 'config.toml')
DEFAULT_MANIFEST_VERSION = '1'
DEFAULT_ARTIFACT_SCHEMA_VERSION = '1'
DEFAULT_EVALUATION_PROTOCOL_VERSION = '2'
SUPPORTED_LOSSES = {'mse'}
SUPPORTED_RESIDUAL_MODELS = {'lstm'}
SUPPORTED_RESIDUAL_SOURCES = {'insample_backcast', 'oof_cv'}


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    target_col: str
    dt_col: str = 'dt'
    freq: str | None = None
    hist_exog_cols: tuple[str, ...] = field(default_factory=tuple)
    futr_exog_cols: tuple[str, ...] = field(default_factory=tuple)
    static_exog_cols: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    input_size: int = 64
    season_length: int = 52
    batch_size: int = 32
    valid_batch_size: int = 32
    windows_batch_size: int = 1024
    inference_windows_batch_size: int = 1024
    learning_rate: float = 0.001
    max_steps: int = 100
    val_size: int = 0
    val_check_steps: int = 100
    early_stop_patience_steps: int = -1
    loss: str = 'mse'


@dataclass(frozen=True)
class CVConfig:
    horizon: int = 12
    step_size: int = 4
    n_windows: int = 24
    final_holdout: int = 12
    overlap_eval_policy: Literal['by_cutoff_mean'] = 'by_cutoff_mean'


@dataclass(frozen=True)
class SchedulerConfig:
    gpu_ids: tuple[int, ...] = (0, 1)
    max_concurrent_jobs: int = 2
    worker_devices: int = 1


@dataclass(frozen=True)
class ResidualConfig:
    enabled: bool = True
    train_source: Literal['insample_backcast', 'oof_cv'] = 'oof_cv'
    model: str = 'lstm'
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobConfig:
    model: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    runtime: RuntimeConfig
    training: TrainingConfig
    cv: CVConfig
    scheduler: SchedulerConfig
    residual: ResidualConfig
    jobs: tuple[JobConfig, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['dataset']['path'] = str(self.dataset.path)
        return payload


@dataclass(frozen=True)
class LoadedConfig:
    config: AppConfig
    source_path: Path
    source_type: str
    normalized_payload: dict[str, Any]
    input_hash: str
    resolved_hash: str


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(',') if part.strip())
    return tuple(str(item) for item in value)


def resolve_config_path(
    repo_root: Path,
    config_path: str | Path | None = None,
    config_toml_path: str | Path | None = None,
) -> tuple[Path, str]:
    if config_toml_path is not None:
        path = Path(config_toml_path)
        if not path.is_absolute():
            path = repo_root / path
        return path, 'toml'
    if config_path is not None:
        path = Path(config_path)
        if not path.is_absolute():
            path = repo_root / path
        suffix = path.suffix.lower()
        if suffix in {'.yaml', '.yml'}:
            return path, 'yaml'
        if suffix == '.toml':
            return path, 'toml'
        raise ValueError(f'Unsupported config extension: {path}')
    for name in CONFIG_FILENAMES:
        candidate = repo_root / name
        if candidate.exists():
            return candidate, 'yaml' if candidate.suffix in {'.yaml', '.yml'} else 'toml'
    raise FileNotFoundError('No config file found in repo root (config.yaml/yml/toml)')


def _load_document(path: Path, source_type: str) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    if source_type == 'toml':
        return tomllib.loads(text)
    payload = yaml.safe_load(text)
    return {} if payload is None else payload


def _normalize_job(job: dict[str, Any]) -> JobConfig:
    return JobConfig(
        model=str(job['model']),
        params=dict(job.get('params', {})),
    )


def _normalize_payload(payload: dict[str, Any], base_dir: Path) -> AppConfig:
    dataset = dict(payload.get('dataset', {}))
    runtime = dict(payload.get('runtime', {}))
    training = dict(payload.get('training', {}))
    cv = dict(payload.get('cv', {}))
    scheduler = dict(payload.get('scheduler', {}))
    residual = dict(payload.get('residual', {}))

    target_col = str(dataset.get('target_col', '')).strip()
    if not target_col:
        raise ValueError('dataset.target_col is required')

    dataset_path = Path(dataset.get('path', 'df.csv'))
    if not dataset_path.is_absolute():
        dataset_path = (base_dir / dataset_path).resolve()

    training.setdefault('loss', 'mse')
    loss = str(training['loss']).lower()
    if loss not in SUPPORTED_LOSSES:
        raise ValueError(f'Unsupported common loss: {loss}')
    training['loss'] = loss

    scheduler.setdefault('worker_devices', 1)
    scheduler['gpu_ids'] = tuple(int(item) for item in scheduler.get('gpu_ids', (0, 1)))
    if int(scheduler['worker_devices']) != 1:
        raise ValueError('worker_devices must remain 1 for scheduler-launched jobs')

    residual.setdefault('enabled', True)
    residual.setdefault('train_source', 'oof_cv')
    residual.setdefault('model', 'lstm')
    residual.setdefault('params', {})
    residual_model = str(residual['model']).lower()
    if residual_model not in SUPPORTED_RESIDUAL_MODELS:
        raise ValueError(f'Unsupported residual model: {residual_model}')
    if residual['train_source'] not in SUPPORTED_RESIDUAL_SOURCES:
        raise ValueError(f'Unsupported residual train_source: {residual["train_source"]}')
    residual['model'] = residual_model
    residual['params'] = dict(residual.get('params', {}))

    jobs = tuple(_normalize_job(job) for job in payload.get('jobs', []))
    if not jobs:
        raise ValueError('Config must define at least one job')
    models = [job.model for job in jobs]
    if len(models) != len(set(models)):
        raise ValueError('jobs.model values must be unique')

    return AppConfig(
        dataset=DatasetConfig(
            path=dataset_path,
            target_col=target_col,
            dt_col=str(dataset.get('dt_col', 'dt')),
            freq=dataset.get('freq'),
            hist_exog_cols=_as_tuple(dataset.get('hist_exog_cols')),
            futr_exog_cols=_as_tuple(dataset.get('futr_exog_cols')),
            static_exog_cols=_as_tuple(dataset.get('static_exog_cols')),
        ),
        runtime=RuntimeConfig(**runtime),
        training=TrainingConfig(**training),
        cv=CVConfig(**cv),
        scheduler=SchedulerConfig(**scheduler),
        residual=ResidualConfig(**residual),
        jobs=jobs,
    )


def load_app_config(
    repo_root: Path,
    *,
    config_path: str | Path | None = None,
    config_toml_path: str | Path | None = None,
) -> LoadedConfig:
    source_path, source_type = resolve_config_path(
        repo_root,
        config_path=config_path,
        config_toml_path=config_toml_path,
    )
    raw_text = source_path.read_text(encoding='utf-8')
    payload = _load_document(source_path, source_type)
    config = _normalize_payload(payload, source_path.parent)
    normalized_payload = config.to_dict()
    resolved_text = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False)
    return LoadedConfig(
        config=config,
        source_path=source_path,
        source_type=source_type,
        normalized_payload=normalized_payload,
        input_hash=_hash_text(raw_text),
        resolved_hash=_hash_text(resolved_text),
    )
