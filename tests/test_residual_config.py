from __future__ import annotations

import copy
import io
import inspect
import os
from pathlib import Path
import json
from types import SimpleNamespace
from typing import Any, TypedDict, cast

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
import yaml

import neuralforecast.auto as nf_auto
import neuralforecast.models as nf_models
import plugins.bs_preforcast.runtime as bs_runtime
from neuralforecast.core import MODEL_FILENAME_DICT
from runtime_support.adapters import build_multivariate_inputs, build_univariate_inputs
from plugins.bs_preforcast.runtime import prepare_bs_preforcast_fold_inputs
from app_config import (
    TrainingOptimizerConfig,
    TrainingLossParams,
    _effective_shared_settings_for_source,
    _load_shared_settings_for_yaml_app_config,
    _merge_shared_settings_into_payload,
    load_app_config,
)
from runtime_support.forecast_models import (
    MODEL_CLASSES,
    build_model,
    supports_auto_mode,
)
from neuralforecast.losses.pytorch import ExLoss
from plugins.bs_preforcast.search_space import SUPPORTED_BS_PREFORCAST_MODELS
from tuning.search_space import (
    DEFAULT_OPTUNA_NUM_TRIALS,
    EXCLUDED_AUTO_MODEL_NAMES,
    FIXED_TRAINING_KEYS,
    MODEL_PARAM_REGISTRY,
    RESIDUAL_PARAM_REGISTRY,
    SUPPORTED_AUTO_MODEL_NAMES,
    SUPPORTED_MODEL_AUTO_MODEL_NAMES,
    SUPPORTED_RESIDUAL_MODELS,
    SUPPORTED_TOP_LEVEL_DIRECT_MODEL_NAMES,
    TRAINING_PARAM_REGISTRY,
    TRAINING_PARAM_REGISTRY_BY_MODEL,
    suggest_training_params,
    suggest_model_params,
    optuna_num_trials,
    training_range_source_for_model,
)
from plugins.residual.base import ResidualContext, ResidualPlugin
from runtime_support.progress import PROGRESS_EVENT_PREFIX
from runtime_support.scheduler import (
    build_device_groups,
    build_launch_plan,
    build_tuning_launch_plan,
    run_parallel_jobs,
    worker_env,
)


_ORIGINAL_RESOLVED_STAGE_JOB = bs_runtime._resolved_stage_job


def _compat_resolved_stage_job(loaded, stage_loaded=None, *, variant_slug: str | None = None):
    jobs = list(getattr(getattr(loaded, "config", None), "jobs", ()))
    if stage_loaded is None:
        if len(jobs) != 1:
            raise ValueError("bs_preforcast stage currently requires exactly one resolved job")
        job = jobs[0]
        training_search = getattr(getattr(loaded, "config", None), "training_search", None)
        if (
            getattr(job, "validated_mode", None) != "learned_fixed"
            or getattr(training_search, "validated_mode", "training_fixed")
            != "training_fixed"
        ):
            raise ValueError("bs_preforcast stage requires a fixed learned job")
        stage_loaded = loaded
        variant_slug = variant_slug or job.model
    return _ORIGINAL_RESOLVED_STAGE_JOB(
        loaded,
        stage_loaded,
        variant_slug=variant_slug or (jobs[0].model if jobs else "stage"),
    )


if len(inspect.signature(bs_runtime._resolved_stage_job).parameters) == 3:
    bs_runtime._resolved_stage_job = _compat_resolved_stage_job


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SUPPORTED_RESIDUAL_MODELS = ("xgboost", "randomforest", "lightgbm")
XGBOOST_RESIDUAL_PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "subsample",
    "colsample_bytree",
)
RESIDUAL_AUTO_FIXTURE_FILES = [
    REPO_ROOT / "tests" / "fixtures" / "optuna_learned_auto_with_residual.yaml",
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "optuna_learned_auto_with_residual_randomforest.yaml",
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "optuna_learned_auto_with_residual_lightgbm.yaml",
]
RESIDUAL_RUNTIME_SMOKE_FIXTURE_FILES = [
    REPO_ROOT / "tests" / "fixtures" / "residual_runtime_smoke_xgboost.yaml",
    REPO_ROOT / "tests" / "fixtures" / "residual_runtime_smoke_randomforest.yaml",
    REPO_ROOT / "tests" / "fixtures" / "residual_runtime_smoke_lightgbm.yaml",
]
NEWLY_SUPPORTED_MODEL_ALIASES = {
    "NonstationaryTransformer": (
        "nonstationarytransformer",
        "autononstationarytransformer",
    ),
    "Mamba": ("mamba", "automamba"),
    "SMamba": ("smamba", "autosmamba"),
    "CMamba": ("cmamba", "autocmamba"),
    "xLSTMMixer": ("xlstmmixer", "autoxlstmmixer"),
    "DUET": ("duet", "autoduet"),
    "DeformTime": ("deformtime",),
    "DeformableTST": ("deformabletst",),
    "ModernTCN": ("moderntcn",),
}

def _onecycle_scheduler(max_lr: float = 0.001) -> dict[str, Any]:
    return {
        "name": "OneCycleLR",
        "max_lr": max_lr,
        "pct_start": 0.3,
        "div_factor": 25.0,
        "final_div_factor": 10000.0,
        "anneal_strategy": "cos",
        "three_phase": False,
        "cycle_momentum": False,
    }



def _write_search_space(
    root: Path,
    payload: dict[str, Any] | None = None,
    *,
    name: str = "yaml/HPO/search_space.yaml",
) -> Path:
    if payload is None:
        payload = {
            "models": {
                "TFT": ["hidden_size", "dropout", "n_head"],
                "iTransformer": ["hidden_size", "n_heads", "e_layers", "d_ff"],
            },
            "training": [],
            "residual": {
                "xgboost": ["n_estimators", "max_depth"],
                "randomforest": [
                    "n_estimators",
                    "max_depth",
                    "min_samples_leaf",
                    "max_features",
                ],
                "lightgbm": [
                    "n_estimators",
                    "max_depth",
                    "num_leaves",
                    "min_child_samples",
                    "feature_fraction",
                ],
            },
        }
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_config(tmp_path: Path, payload: dict, suffix: str) -> Path:
    path = tmp_path / f"config{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".toml":
        task_block = ""
        task_name = payload.get("task", {}).get("name")
        residual_target = payload.get("residual", {}).get("target", "level")
        runtime_transformations = payload.get("runtime", {}).get("transformations")
        runtime_transformations_target = payload.get("runtime", {}).get(
            "transformations_target"
        )
        runtime_transformations_exog = payload.get("runtime", {}).get(
            "transformations_exog"
        )
        if task_name:
            task_block = "\n[task]\nname = " + repr(task_name) + "\n"
        runtime_block = "\n[runtime]\nrandom_seed = 1\n"
        if runtime_transformations is not None:
            runtime_block += "transformations = " + repr(runtime_transformations) + "\n"
        if runtime_transformations_target is not None:
            runtime_block += (
                "transformations_target = "
                + repr(runtime_transformations_target)
                + "\n"
            )
        if runtime_transformations_exog is not None:
            runtime_block += (
                "transformations_exog = "
                + repr(runtime_transformations_exog)
                + "\n"
            )
        text = (
            task_block
            + runtime_block
            + """
[dataset]
path = 'data.csv'
target_col = 'target'
dt_col = 'dt'
hist_exog_cols = ['hist_a']
futr_exog_cols = []
static_exog_cols = []

[training]
input_size = 64
season_length = 52
batch_size = 32
valid_batch_size = 64
windows_batch_size = 1024
inference_windows_batch_size = 1024
max_steps = 50
loss = 'mse'

[training.lr_scheduler]
name = 'OneCycleLR'
max_lr = 0.001
pct_start = 0.3
div_factor = 25.0
final_div_factor = 10000.0
anneal_strategy = 'cos'
three_phase = false
cycle_momentum = false

[cv]
horizon = 12
step_size = 4
n_windows = 24
gap = 0
overlap_eval_policy = 'by_cutoff_mean'

[scheduler]
gpu_ids = [0, 1]
max_concurrent_jobs = 2
worker_devices = 1
parallelize_single_job_tuning = false

[residual]
enabled = true
model = 'xgboost'
target = '__RESIDUAL_TARGET__'

[residual.params]
n_estimators = 8
max_depth = 2

[[jobs]]
model = 'TFT'
params = { hidden_size = 32 }

[[jobs]]
model = 'iTransformer'
params = { hidden_size = 32, n_heads = 4, e_layers = 2, d_ff = 64 }
"""
        )
        text = text.replace("__RESIDUAL_TARGET__", residual_target)
        path.write_text(text, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_yaml_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path



def _payload() -> dict:
    return {
        "dataset": {
            "path": "data.csv",
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["hist_a"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "runtime": {"random_seed": 1},
        "training": {
            "input_size": 64,
            "season_length": 52,
            "batch_size": 32,
            "valid_batch_size": 64,
            "windows_batch_size": 1024,
            "inference_windows_batch_size": 1024,
            "max_steps": 50,
            "loss": "mse",
            "lr_scheduler": _onecycle_scheduler(0.001),
        },
        "cv": {
            "horizon": 12,
            "step_size": 4,
            "n_windows": 24,
            "gap": 0,
            "overlap_eval_policy": "by_cutoff_mean",
        },
        "scheduler": {
            "gpu_ids": [0, 1],
            "max_concurrent_jobs": 2,
            "worker_devices": 1,
            "parallelize_single_job_tuning": False,
        },
        "residual": {
            "enabled": True,
            "model": "xgboost",
            "params": {"n_estimators": 8, "max_depth": 2},
        },
        "jobs": [
            {"model": "TFT", "params": {"hidden_size": 32}},
            {
                "model": "iTransformer",
                "params": {"hidden_size": 32, "n_heads": 4, "e_layers": 2, "d_ff": 64},
            },
        ],
    }


def _repo_root_contract_payload() -> dict[str, Any]:
    payload = copy.deepcopy(_payload())
    payload["task"] = {"name": "semi_test"}
    payload["residual"]["params"] = {
        "n_estimators": 64,
        "max_depth": 3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    payload["jobs"] = [
        {"model": "TFT", "params": {"hidden_size": 128}},
        {
            "model": "VanillaTransformer",
            "params": {
                "hidden_size": 128,
                "dropout": 0.05,
                "scaler_type": "identity",
            },
        },
        {
            "model": "Informer",
            "params": {
                "hidden_size": 128,
                "factor": 3,
                "dropout": 0.05,
                "scaler_type": "identity",
            },
        },
        {
            "model": "Autoformer",
            "params": {
                "hidden_size": 128,
                "factor": 3,
                "dropout": 0.05,
                "scaler_type": "identity",
            },
        },
        {
            "model": "PatchTST",
            "params": {
                "hidden_size": 128,
                "n_heads": 16,
                "encoder_layers": 3,
                "patch_len": 16,
                "dropout": 0.2,
                "scaler_type": "identity",
            },
        },
        {
            "model": "LSTM",
            "params": {
                "encoder_hidden_size": 128,
                "decoder_hidden_size": 128,
            },
        },
        {
            "model": "NHITS",
            "params": {
                "mlp_units": [[64, 64], [64, 64], [64, 64]],
                "n_pool_kernel_size": [2, 2, 1],
                "n_freq_downsample": [4, 2, 1],
                "dropout_prob_theta": 0.0,
                "activation": "ReLU",
            },
        },
        {
            "model": "iTransformer",
            "params": {
                "hidden_size": 128,
                "n_heads": 8,
                "e_layers": 2,
                "d_ff": 256,
            },
        },
        {"model": "Naive", "params": {}},
    ]
    return payload


def _write_repo_root_contract_config(repo_root: Path) -> Path:
    (repo_root / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    _write_search_space(repo_root)
    return _write_yaml_payload(
        repo_root / "repo-root-contract.yaml",
        _repo_root_contract_payload(),
    )


def _main_bs_preforcast(
    *,
    enabled: bool = True,
    config_path: str | None = "yaml/plugins/bs_preforcast.yaml",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"enabled": enabled}
    if config_path is not None:
        payload["config_path"] = config_path
    return payload


def _linked_bs_preforcast(
    *,
    target_columns: tuple[str, ...] = ("bs_a",),
    multivariable: bool = False,
    legacy_using_futr_exog: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bs_preforcast": {
            "target_columns": list(target_columns),
            "task": {"multivariable": multivariable},
        }
    }
    if legacy_using_futr_exog is not None:
        payload["bs_preforcast"]["using_futr_exog"] = legacy_using_futr_exog
    return payload


def _thin_plugin_stage_payload(
    *,
    jobs: list[dict[str, Any]],
    target_columns: tuple[str, ...] = ("bs_a",),
    multivariable: bool = False,
    hist_columns: tuple[str, ...] = (),
    legacy_using_futr_exog: bool | None = None,
) -> dict[str, Any]:
    payload = _linked_bs_preforcast(
        target_columns=target_columns,
        multivariable=multivariable,
        legacy_using_futr_exog=legacy_using_futr_exog,
    )
    if hist_columns:
        payload["bs_preforcast"]["hist_columns"] = list(hist_columns)
    payload["jobs"] = jobs
    return payload


def _residual_defaults_map() -> dict[str, dict[str, Any]]:
    import tuning.search_space as residual_spaces

    for attr_name in (
        "DEFAULT_RESIDUAL_PARAMS_BY_MODEL",
        "RESIDUAL_DEFAULTS_MAP",
        "RESIDUAL_DEFAULTS_BY_MODEL",
        "RESIDUAL_DEFAULTS",
    ):
        value = getattr(residual_spaces, attr_name, None)
        if value is not None:
            return cast(dict[str, dict[str, Any]], value)
    pytest.fail("tuning.search_space must expose a per-model defaults map")
    raise AssertionError("unreachable")


def _import_build_residual_plugin():
    try:
        from plugins.residual import build_residual_plugin as builder
    except ModuleNotFoundError as exc:
        if exc.name in {"lightgbm", "xgboost", "sklearn"}:
            pytest.skip(f"optional dependency missing: {exc.name}")
        raise
    return builder


def _skip_missing_residual_backend(model_name: str) -> None:
    if model_name == "xgboost":
        pytest.importorskip("xgboost")
    elif model_name == "randomforest":
        pytest.importorskip("sklearn")
    elif model_name == "lightgbm":
        pytest.importorskip("lightgbm")


def _load_search_space_strict() -> dict[str, Any]:
    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> Any:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate YAML key: {key}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return cast(
        dict[str, Any],
        yaml.load((REPO_ROOT / "yaml/HPO/search_space.yaml").read_text(), Loader=UniqueKeySafeLoader),
    )


def test_load_app_config_preserves_runtime_opt_n_trial(tmp_path: Path):
    payload = _payload()
    payload["runtime"]["opt_n_trial"] = 9
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    assert loaded.config.runtime.opt_n_trial == 9
    assert loaded.normalized_payload["runtime"]["opt_n_trial"] == 9


def test_optuna_num_trials_prefers_yaml_then_env_then_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "7")
    assert optuna_num_trials(11) == 11
    assert optuna_num_trials(None) == 7
    monkeypatch.delenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", raising=False)
    assert optuna_num_trials(None) == DEFAULT_OPTUNA_NUM_TRIALS


def test_load_app_config_rejects_non_positive_runtime_opt_n_trial(tmp_path: Path):
    payload = _payload()
    payload["runtime"]["opt_n_trial"] = 0
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="runtime.opt_n_trial must be a positive integer"
    ):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_toml_and_yaml_normalize_to_same_typed_model(tmp_path: Path):
    payload = _payload()
    payload["task"] = {"name": "semi_test"}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    yaml_path = _write_config(tmp_path, payload, ".yaml")
    toml_path = _write_config(tmp_path, payload, ".toml")
    loaded_yaml = load_app_config(tmp_path, config_path=yaml_path)
    loaded_toml = load_app_config(tmp_path, config_toml_path=toml_path)
    assert loaded_yaml.config.to_dict() == loaded_toml.config.to_dict()


def test_task_name_is_loaded_into_normalized_config(tmp_path: Path):
    payload = _payload()
    payload["task"] = {"name": "semi_test"}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    assert loaded.config.task.name == "semi_test"
    assert loaded.normalized_payload["task"] == {"name": "semi_test"}


def test_load_app_config_preserves_residual_target_yaml_toml_parity(tmp_path: Path):
    payload = _payload()
    payload["residual"]["target"] = "delta"
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )
    yaml_path = _write_config(tmp_path, payload, ".yaml")
    toml_path = _write_config(tmp_path, payload, ".toml")

    loaded_yaml = load_app_config(tmp_path, config_path=yaml_path)
    loaded_toml = load_app_config(tmp_path, config_toml_path=toml_path)

    assert loaded_yaml.config.residual.target == "delta"
    assert loaded_yaml.config.to_dict() == loaded_toml.config.to_dict()


def test_runtime_transformations_omitted_when_unset_preserves_resolved_hash(
    tmp_path: Path,
):
    payload_default = _payload()
    payload_null = _payload()
    payload_null["runtime"]["transformations_target"] = None
    payload_null["runtime"]["transformations_exog"] = None
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )

    loaded_default = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload_default, ".yaml"),
    )
    loaded_null = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload_null, ".yaml"),
    )

    assert "transformations_target" not in loaded_default.normalized_payload["runtime"]
    assert "transformations_exog" not in loaded_default.normalized_payload["runtime"]
    assert "transformations_target" not in loaded_null.normalized_payload["runtime"]
    assert "transformations_exog" not in loaded_null.normalized_payload["runtime"]
    assert loaded_default.resolved_hash == loaded_null.resolved_hash


def test_runtime_transformations_yaml_toml_validation_parity(tmp_path: Path):
    payload = _payload()
    payload["runtime"]["transformations_target"] = "diff"
    payload["runtime"]["transformations_exog"] = "diff"
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n2020-01-08,2,3,4\n",
        encoding="utf-8",
    )
    yaml_path = _write_config(tmp_path, payload, ".yaml")
    toml_path = _write_config(tmp_path, payload, ".toml")

    loaded_yaml = load_app_config(tmp_path, config_path=yaml_path)
    loaded_toml = load_app_config(tmp_path, config_toml_path=toml_path)

    assert loaded_yaml.config.runtime.transformations_target == "diff"
    assert loaded_yaml.config.runtime.transformations_exog == "diff"
    assert loaded_yaml.config.to_dict() == loaded_toml.config.to_dict()
    assert loaded_yaml.normalized_payload["runtime"]["transformations_target"] == "diff"
    assert loaded_yaml.normalized_payload["runtime"]["transformations_exog"] == "diff"


@pytest.mark.parametrize("suffix", [".yaml", ".toml"])
def test_load_app_config_rejects_invalid_runtime_transformations(
    tmp_path: Path, suffix: str
):
    payload = _payload()
    payload["runtime"]["transformations_target"] = "log"
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n2020-01-08,2,3,4\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="runtime.transformations_target must be the string 'diff'"
    ):
        if suffix == ".toml":
            load_app_config(
                tmp_path, config_toml_path=_write_config(tmp_path, payload, suffix)
            )
        else:
            load_app_config(
                tmp_path, config_path=_write_config(tmp_path, payload, suffix)
            )


@pytest.mark.parametrize("suffix", [".yaml", ".toml"])
def test_load_app_config_rejects_legacy_runtime_transformations(
    tmp_path: Path, suffix: str
):
    payload = _payload()
    payload["runtime"]["transformations"] = "diff"
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n2020-01-08,2,3,4\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="runtime.transformations is no longer supported",
    ):
        if suffix == ".toml":
            load_app_config(
                tmp_path, config_toml_path=_write_config(tmp_path, payload, suffix)
            )
        else:
            load_app_config(
                tmp_path, config_path=_write_config(tmp_path, payload, suffix)
            )


def test_load_app_config_accepts_exloss_params(tmp_path: Path):
    payload = _payload()
    payload["training"].update(
        {
            "loss": "exloss",
            "loss_params": {
                "up_th": 0.9,
                "down_th": 0.1,
                "lamda_underestimate": 1.5,
                "lamda_overestimate": 1.0,
                "lamda": 0.7,
            },
        }
    )
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    assert loaded.config.training.loss == "exloss"
    assert loaded.config.training.loss_params == TrainingLossParams(
        up_th=0.9,
        down_th=0.1,
        lamda_underestimate=1.5,
        lamda_overestimate=1.0,
        lamda=0.7,
    )
    assert loaded.normalized_payload["training"]["loss_params"] == {
        "up_th": 0.9,
        "down_th": 0.1,
        "lamda_underestimate": 1.5,
        "lamda_overestimate": 1.0,
        "lamda": 0.7,
    }


@pytest.mark.parametrize(
    ("loss_params", "message"),
    [
        ({"extra": 1}, "training.loss_params contains unsupported key"),
        (
            {"up_th": 0.1, "down_th": 0.9},
            "thresholds must satisfy 0 < down_th < up_th < 1",
        ),
        ({"lamda_underestimate": -1}, "lamda_underestimate must be >= 0"),
        ({"lamda_overestimate": -1}, "lamda_overestimate must be >= 0"),
        ({"lamda": -1}, "training.loss_params.lamda must be >= 0"),
        (
            {"lamda_underestimate": "oops"},
            "training.loss_params.lamda_underestimate must be numeric",
        ),
        ({"up_th": float("nan")}, "training.loss_params.up_th must be finite"),
        ({"lamda": float("inf")}, "training.loss_params.lamda must be finite"),
    ],
)
def test_load_app_config_rejects_invalid_exloss_params(
    tmp_path: Path, loss_params: dict[str, Any], message: str
):
    payload = _payload()
    payload["training"]["loss"] = "exloss"
    payload["training"]["loss_params"] = loss_params
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_load_app_config_rejects_loss_params_for_mse(tmp_path: Path):
    payload = _payload()
    payload["training"]["loss_params"] = {"lamda": 0.7}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="training.loss_params is only supported when training.loss == exloss",
    ):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_load_app_config_preserves_default_mse_loss_payload_and_resolved_hash(
    tmp_path: Path,
):
    payload_default = _payload()
    payload_default["training"].pop("loss", None)
    payload_explicit = _payload()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )

    loaded_default = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload_default, ".yaml"),
    )
    loaded_explicit = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload_explicit, ".yaml"),
    )

    assert "loss_params" not in loaded_default.normalized_payload["training"]
    assert "loss_params" not in loaded_explicit.normalized_payload["training"]
    assert loaded_default.resolved_hash == loaded_explicit.resolved_hash


def test_load_app_config_applies_residual_feature_defaults(tmp_path: Path):
    payload = _payload()
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["dataset"]["static_exog_cols"] = ["static_a"]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,futr_a,static_a\n2020-01-01,1,2,3,4\n",
        encoding="utf-8",
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    features = loaded.config.residual.features
    assert features.include_base_prediction is True
    assert features.include_horizon_step is True
    assert features.include_date_features is False
    assert features.lag_features.enabled is False
    assert features.lag_features.sources == ()
    assert features.lag_features.steps == ()
    assert features.lag_features.transforms == ("raw",)
    assert features.exog_sources.hist == ("hist_a",)
    assert features.exog_sources.futr == ()
    assert features.exog_sources.static == ()
    assert loaded.normalized_payload["residual"]["features"] == {
        "include_base_prediction": True,
        "include_horizon_step": True,
        "include_date_features": False,
        "lag_features": {
            "enabled": False,
            "sources": [],
            "steps": [],
            "transforms": ["raw"],
        },
        "exog_sources": {"hist": ["hist_a"], "futr": [], "static": []},
    }


def test_load_app_config_accepts_residual_feature_opt_ins(tmp_path: Path):
    payload = _payload()
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["dataset"]["static_exog_cols"] = ["static_a"]
    payload["residual"]["features"] = {
        "include_date_features": True,
        "exog_sources": {
            "hist": ["hist_a"],
            "futr": ["futr_a"],
            "static": ["static_a"],
        },
        "lag_features": {
            "enabled": True,
            "sources": ["y_hat_base", "hist_a", "futr_a"],
            "steps": [1, 2],
            "transforms": ["raw"],
        },
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,futr_a,static_a\n2020-01-01,1,2,3,4\n",
        encoding="utf-8",
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    features = loaded.config.residual.features
    assert features.include_date_features is True
    assert features.exog_sources.hist == ("hist_a",)
    assert features.exog_sources.futr == ("futr_a",)
    assert features.exog_sources.static == ("static_a",)
    assert features.lag_features.enabled is True
    assert features.lag_features.sources == ("y_hat_base", "hist_a", "futr_a")
    assert features.lag_features.steps == (1, 2)
    assert loaded.normalized_payload["residual"]["features"]["lag_features"] == {
        "enabled": True,
        "sources": ["y_hat_base", "hist_a", "futr_a"],
        "steps": [1, 2],
        "transforms": ["raw"],
    }


def test_load_app_config_allows_explicit_empty_residual_hist_exog_override(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"]["features"] = {"exog_sources": {"hist": []}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n",
        encoding="utf-8",
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    features = loaded.config.residual.features
    assert features.exog_sources.hist == ()
    assert loaded.normalized_payload["residual"]["features"]["exog_sources"] == {
        "hist": [],
        "futr": [],
        "static": [],
    }


def test_manifest_and_capability_report_include_residual_feature_visibility(
    tmp_path: Path,
):
    import runtime_support.runner as runtime
    from runtime_support.manifest import build_manifest

    payload = _payload()
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["dataset"]["static_exog_cols"] = ["static_a"]
    payload["residual"]["features"] = {
        "include_date_features": True,
        "exog_sources": {
            "hist": ["hist_a"],
            "futr": ["futr_a"],
            "static": ["static_a"],
        },
        "lag_features": {
            "enabled": True,
            "sources": ["y_hat_base", "hist_a", "futr_a"],
            "steps": [1, 2],
            "transforms": ["raw"],
        },
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,futr_a,static_a\n2020-01-01,1,2,3,4\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    capability_path = tmp_path / "capability_report.json"
    runtime._validate_jobs(loaded, loaded.config.jobs, capability_path)
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    manifest = build_manifest(
        loaded,
        compat_mode="dual_read",
        entrypoint_version="test-entrypoint",
        resolved_config_path=tmp_path / "config.resolved.json",
    )

    expected_policy = {
        "include_base_prediction": True,
        "include_horizon_step": True,
        "include_date_features": True,
        "lag_features": {
            "enabled": True,
            "sources": ["y_hat_base", "hist_a", "futr_a"],
            "steps": [1, 2],
            "transforms": ["raw"],
        },
        "exog_sources": {
            "hist": ["hist_a"],
            "futr": ["futr_a"],
            "static": ["static_a"],
        },
    }
    expected_columns = [
        "horizon_step",
        "y_hat_base",
        "cutoff_day",
        "ds_day",
        "y_hat_base_lag_1",
        "y_hat_base_lag_2",
        "hist_a_lag_1",
        "hist_a_lag_2",
        "futr_a_lag_1",
        "futr_a_lag_2",
        "futr_a",
        "static_a",
    ]

    assert capability["residual"]["feature_policy"] == expected_policy
    assert capability["residual"]["active_feature_columns"] == expected_columns
    assert manifest["residual"]["target"] == "level"
    assert manifest["residual"]["feature_policy"] == expected_policy
    assert manifest["residual"]["active_feature_columns"] == expected_columns


def test_load_app_config_rejects_unknown_residual_feature_exog_selection(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"]["features"] = {
        "exog_sources": {"hist": ["missing_hist"]},
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="residual.features.exog_sources.hist must be selected from dataset.hist_exog_cols",
    ):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_load_app_config_rejects_forbidden_residual_lag_sources(tmp_path: Path):
    payload = _payload()
    payload["residual"]["features"] = {
        "lag_features": {
            "enabled": True,
            "sources": ["y"],
            "steps": [1],
        }
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="residual.features.lag_features.sources contains forbidden source",
    ):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_load_app_config_rejects_residual_lag_sources_without_matching_exog_opt_in(
    tmp_path: Path,
):
    payload = _payload()
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["residual"]["features"] = {
        "exog_sources": {"hist": []},
        "lag_features": {
            "enabled": True,
            "sources": ["hist_a"],
            "steps": [1],
        }
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,futr_a\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="residual.features.lag_features.sources must be y_hat_base or selected hist/futr exog columns",
    ):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_adapters_materialize_expected_frames(tmp_path: Path):
    payload = _payload()
    yaml_path = _write_config(tmp_path, payload, ".yaml")
    source_path = tmp_path / "data.csv"
    source_path.write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,10,2\n2020-01-08,2,11,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(tmp_path, config_path=yaml_path)
    import pandas as pd

    source_df = pd.read_csv(source_path)
    univariate = build_univariate_inputs(
        source_df, loaded.config.jobs[0], dataset=loaded.config.dataset, dt_col="dt"
    )
    multivariate = build_multivariate_inputs(
        source_df, loaded.config.jobs[1], dataset=loaded.config.dataset, dt_col="dt"
    )
    assert list(univariate.fit_df.columns) == ["unique_id", "ds", "y", "hist_a"]
    assert multivariate.channel_map == {"target": 0, "hist_a": 1}


def test_model_builder_applies_common_loss_and_multivariate_n_series(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, _payload(), ".yaml")
    )
    univariate_model = build_model(loaded.config, loaded.config.jobs[0])
    multivariate_model = build_model(loaded.config, loaded.config.jobs[1], n_series=2)
    assert type(univariate_model.loss).__name__ == "MSE"
    assert type(univariate_model.valid_loss).__name__ == "MSE"
    assert type(multivariate_model.loss).__name__ == "MSE"
    assert type(multivariate_model.valid_loss).__name__ == "MSE"
    assert getattr(multivariate_model, "n_series", 2) == 2


@pytest.mark.parametrize(
    ("optimizer_name", "kwargs"),
    [
        ("ademamix", {"weight_decay": 0.125}),
        ("rmsprop", {"alpha": 0.95}),
        ("radam", {"weight_decay": 0.125}),
    ],
)
def test_load_app_config_accepts_training_optimizer(
    tmp_path: Path, optimizer_name: str, kwargs: dict[str, Any]
):
    payload = _payload()
    payload["training"]["optimizer"] = {
        "name": optimizer_name,
        "kwargs": kwargs,
    }

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    assert loaded.config.training.optimizer == TrainingOptimizerConfig(
        name=optimizer_name,
        kwargs=kwargs,
    )
    assert loaded.normalized_payload["training"]["optimizer"] == {
        "name": optimizer_name,
        "kwargs": kwargs,
    }


@pytest.mark.parametrize(
    ("optimizer_payload", "message"),
    [
        ("adamw", "training.optimizer must be a mapping"),
        ({"name": "madeup"}, "training.optimizer.name must be one of"),
        (
            {"name": "adamw", "kwargs": []},
            "training.optimizer.kwargs must be a mapping",
        ),
        (
            {"name": "adamw", "kwargs": {"": 1}},
            "training.optimizer.kwargs keys must be non-empty strings",
        ),
    ],
)
def test_load_app_config_rejects_invalid_training_optimizer(
    tmp_path: Path, optimizer_payload: Any, message: str
):
    payload = _payload()
    payload["training"]["optimizer"] = optimizer_payload

    with pytest.raises(ValueError, match=message):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("optimizer", "adamw"),
        ("optimizer_kwargs", {"weight_decay": 0.01}),
    ],
)
def test_load_app_config_rejects_forbidden_job_param_optimizer_overrides(
    tmp_path: Path, key: str, value: Any
):
    payload = _payload()
    payload["jobs"][0]["params"][key] = value

    with pytest.raises(
        ValueError,
        match="Move optimizer selection under training.optimizer",
    ):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_model_builder_passes_hist_exog_to_patchtst(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [
        {
            "model": "PatchTST",
            "params": {
                "hidden_size": 16,
                "n_heads": 4,
                "encoder_layers": 2,
                "patch_len": 4,
            },
        }
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    model = build_model(loaded.config, loaded.config.jobs[0])

    assert model.hist_exog_list == ["hist_a"]
    assert model.hist_exog_size == 1


def test_model_builder_patchtst_forecasts_target_only_with_hist_exog(tmp_path: Path):
    payload = _payload()
    payload["training"]["input_size"] = 4
    payload["cv"]["horizon"] = 2
    payload["jobs"] = [
        {
            "model": "PatchTST",
            "params": {
                "hidden_size": 4,
                "n_heads": 1,
                "encoder_layers": 1,
                "patch_len": 2,
                "stride": 1,
            },
        }
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n"
        "2020-01-01,1,10\n"
        "2020-01-08,2,11\n"
        "2020-01-15,3,12\n"
        "2020-01-22,4,13\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    model = build_model(loaded.config, loaded.config.jobs[0])

    class Recorder(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h
            self.last_input = None

        def forward(self, x):
            self.last_input = x
            return x[:, :, -1:].repeat(1, 1, self.h)

    recorder = Recorder(h=loaded.config.cv.horizon)
    model.model = recorder
    windows_batch = {
        "insample_y": torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
        "hist_exog": torch.tensor([[[10.0], [11.0], [12.0], [13.0]]]),
        "futr_exog": None,
        "stat_exog": None,
        "insample_mask": torch.ones(1, 4, 1),
    }

    forecast = model(windows_batch)

    assert recorder.last_input.shape == (1, 2, 4)
    assert forecast.shape == (1, 2, 1)
    assert torch.equal(forecast[:, :, 0], torch.tensor([[4.0, 4.0]]))


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("DLinear", {"moving_avg_window": 3}),
        (
            "iTransformer",
            {"hidden_size": 16, "n_heads": 1, "e_layers": 1, "d_ff": 32},
        ),
        (
            "TimeMixer",
            {"d_model": 16, "d_ff": 32, "down_sampling_layers": 1, "top_k": 3},
        ),
    ],
)
def test_model_builder_passes_hist_exog_to_selected_models(
    tmp_path: Path, model_name: str, params: dict[str, Any]
):
    payload = _payload()
    payload["jobs"] = [{"model": model_name, "params": params}]
    _write_search_space(tmp_path)
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    model = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert model.hist_exog_list == ["hist_a"]
    assert model.hist_exog_size == 1


def test_model_builder_applies_exloss_and_multivariate_n_series(tmp_path: Path):
    payload = _payload()
    payload["training"].update(
        {
            "loss": "exloss",
            "loss_params": {
                "up_th": 0.9,
                "down_th": 0.1,
                "lamda_underestimate": 1.5,
                "lamda_overestimate": 1.0,
                "lamda": 0.7,
            },
        }
    )
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )
    univariate_model = build_model(loaded.config, loaded.config.jobs[0])
    multivariate_model = build_model(loaded.config, loaded.config.jobs[1], n_series=2)
    assert isinstance(univariate_model.loss, ExLoss)
    assert isinstance(univariate_model.valid_loss, ExLoss)
    assert isinstance(multivariate_model.loss, ExLoss)
    assert isinstance(multivariate_model.valid_loss, ExLoss)
    assert loaded.config.training.loss_params == TrainingLossParams(
        up_th=0.9,
        down_th=0.1,
        lamda_underestimate=1.5,
        lamda_overestimate=1.0,
        lamda=0.7,
    )
    assert getattr(multivariate_model, "n_series", 2) == 2


def test_model_builder_propagates_centralized_training_controls(tmp_path: Path):
    payload = _payload()
    payload["training"].update(
        {
            "max_steps": 17,
            "lr_scheduler": _onecycle_scheduler(0.123),
            "scaler_type": "standard",
            "model_step_size": 3,
            "val_check_steps": 7,
            "min_steps_before_early_stop": 13,
            "early_stop_patience_steps": 11,
        }
    )
    payload["jobs"] = [
        {"model": "TFT", "params": {"hidden_size": 32}},
        {
            "model": "LSTM",
            "params": {"encoder_hidden_size": 32, "decoder_hidden_size": 32},
        },
        {"model": "NHITS", "params": {"mlp_units": [[32, 32], [32, 32], [32, 32]]}},
        {
            "model": "iTransformer",
            "params": {"hidden_size": 32, "n_heads": 4, "e_layers": 2, "d_ff": 64},
        },
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    tft = build_model(loaded.config, loaded.config.jobs[0])
    lstm = build_model(loaded.config, loaded.config.jobs[1])
    nhits = build_model(loaded.config, loaded.config.jobs[2])
    itransformer = build_model(loaded.config, loaded.config.jobs[3], n_series=2)

    for model in (tft, lstm, nhits, itransformer):
        assert model.hparams.max_steps == 17
        assert model.hparams.max_lr == pytest.approx(0.123)
        assert model.hparams.scaler_type == "standard"
        assert model.hparams.step_size == 3
        assert model.hparams.val_check_steps == 7
        assert model.hparams.min_steps_before_early_stop == 13
        assert model.hparams.early_stop_patience_steps == 11


def test_load_app_config_drops_training_season_length_and_maps_model_step_size(
    tmp_path: Path,
):
    payload = _payload()
    payload["training"].update({"season_length": 52, "model_step_size": 6})
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    assert loaded.config.training.model_step_size == 6
    assert loaded.normalized_payload["training"]["model_step_size"] == 6
    assert "season_length" not in loaded.normalized_payload["training"]
    assert "step_size" not in loaded.normalized_payload["training"]


def test_load_app_config_migrates_legacy_shared_job_scaler_type_to_training(
    tmp_path: Path,
):
    payload = _payload()
    payload["jobs"] = [
        {"model": "VanillaTransformer", "params": {"hidden_size": 32, "scaler_type": "identity"}},
        {"model": "Informer", "params": {"hidden_size": 32, "factor": 3, "scaler_type": "identity"}},
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    assert loaded.config.training.scaler_type == "identity"
    assert loaded.config.jobs[0].params == {"hidden_size": 32}
    assert loaded.config.jobs[1].params == {"hidden_size": 32, "factor": 3}


def test_model_builder_disables_logger_and_keeps_timemixer_without_future_features(
    tmp_path: Path,
):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["dataset"]["futr_exog_cols"] = []
    payload["jobs"] = [
        {
            "model": "TimeMixer",
            "params": {
                "d_model": 16,
                "d_ff": 32,
                "down_sampling_layers": 1,
                "top_k": 3,
            },
        }
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    timemixer = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert timemixer.trainer_kwargs["logger"] is False
    assert timemixer.trainer_kwargs["enable_progress_bar"] is False
    assert timemixer.use_future_temporal_feature == 0


def test_build_model_supports_duet(tmp_path: Path):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["dataset"]["futr_exog_cols"] = []
    payload["jobs"] = [
        {
            "model": "DUET",
            "params": {
                "n_block": 2,
                "hidden_size": 32,
                "ff_dim": 64,
                "moving_avg_window": 5,
            },
        },
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    duet = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert getattr(duet.hparams, "n_series", 1) == 1
    assert duet.hparams.hidden_size == 32


def test_worker_env_supports_scalar_gpu_id():
    env = worker_env(0)

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["NEURALFORECAST_ASSIGNED_GPU_IDS"] == "0"
    assert env["NEURALFORECAST_WORKER_DEVICES"] == "1"
    assert env["NEURALFORECAST_PROGRESS_MODE"] == "structured"
    assert env["NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS"] == "1"


def test_scheduler_plan_and_worker_env_support_device_groups(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    payload = _payload()
    payload["scheduler"]["worker_devices"] = 2
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )
    launches = build_launch_plan(loaded.config, loaded.config.jobs)
    assert all(launch.devices == 2 for launch in launches)
    assert build_device_groups(loaded.config) == [(0, 1)]
    tuning_launches = build_tuning_launch_plan(loaded.config, job_name="TFT")
    assert len(tuning_launches) == 1
    env = worker_env((0, 1))
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["NEURALFORECAST_ASSIGNED_GPU_IDS"] == "0,1"
    assert env["NEURALFORECAST_WORKER_DEVICES"] == "2"
    assert env["NEURALFORECAST_PROGRESS_MODE"] == "structured"
    assert env["NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS"] == "1"


def test_build_device_groups_respects_assigned_gpu_ids_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    payload = _payload()
    payload["scheduler"]["gpu_ids"] = [0, 1]
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    monkeypatch.setenv("NEURALFORECAST_ASSIGNED_GPU_IDS", "1")

    assert build_device_groups(loaded.config) == [(1,)]
    assert len(build_tuning_launch_plan(loaded.config, job_name="TFT")) == 1


def test_load_app_config_rejects_non_divisible_scheduler_groups(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n",
        encoding="utf-8",
    )
    payload = _payload()
    payload["scheduler"]["gpu_ids"] = [0, 1, 2]
    payload["scheduler"]["worker_devices"] = 2
    path = _write_config(tmp_path, payload, ".yaml")

    with pytest.raises(
        ValueError,
        match="scheduler.gpu_ids must be evenly divisible by scheduler.worker_devices",
    ):
        load_app_config(tmp_path, config_path=path)


def test_scheduler_respects_max_concurrent_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    payload = _payload()
    payload["jobs"] = [
        {"model": "TFT", "params": {"hidden_size": 32}},
        {
            "model": "LSTM",
            "params": {"encoder_hidden_size": 16, "decoder_hidden_size": 16},
        },
        {"model": "NHITS", "params": {"mlp_units": [[8, 8], [8, 8], [8, 8]]}},
    ]
    payload["scheduler"]["max_concurrent_jobs"] = 2
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )
    launches = build_launch_plan(loaded.config, loaded.config.jobs)
    state = {"active": 0, "max_active": 0}

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            self._completed = False
            self.stdout = io.StringIO("")

        def poll(self):
            return 0 if not self._completed else 0

        def wait(self):
            self._completed = True
            state["active"] -= 1
            return 0

    monkeypatch.setattr("runtime_support.scheduler.subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "runtime_support.scheduler._worker_command",
        lambda *_args, **_kwargs: ["python", "fake_worker.py"],
    )

    results = run_parallel_jobs(tmp_path, loaded, launches, tmp_path / "scheduler")

    assert len(results) == 3
    assert state["max_active"] == 2


def test_build_model_surfaces_trainer_and_dataloader_settings(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n2020-01-08,2,3,4\n",
        encoding="utf-8",
    )
    payload = _payload()
    payload["training"].update(
        {
            "devices": 2,
            "strategy": "ddp",
            "precision": "16-mixed",
            "dataloader_kwargs": {
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
            },
        }
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    model = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert model.trainer_kwargs["devices"] == 2
    assert model.trainer_kwargs["strategy"] == "ddp"
    assert model.trainer_kwargs["precision"] == "16-mixed"
    assert model.dataloader_kwargs == {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def test_build_model_clamps_devices_to_visible_worker_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n2020-01-08,2,3,4\n",
        encoding="utf-8",
    )
    payload = _payload()
    payload["training"]["devices"] = 2
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )
    monkeypatch.setenv("NEURALFORECAST_WORKER_DEVICES", "1")

    model = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert model.trainer_kwargs["devices"] == 1


@pytest.mark.parametrize(
    ("optimizer_name", "kwargs"),
    [
        ("adamw", {"weight_decay": 0.02}),
        ("mars", {"mars_type": "adamw"}),
        ("rmsprop", {"alpha": 0.97}),
        ("radam", {"weight_decay": 0.01}),
    ],
)
def test_build_model_forwards_training_optimizer_configuration(
    tmp_path: Path, optimizer_name: str, kwargs: dict[str, Any]
):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n2020-01-08,2,3,4\n",
        encoding="utf-8",
    )
    payload = _payload()
    payload["training"]["optimizer"] = {"name": optimizer_name, "kwargs": kwargs}
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    model = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert issubclass(model.optimizer, torch.optim.Optimizer)
    assert model.optimizer_kwargs == kwargs


def test_scheduler_streams_worker_stdout_to_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, _payload(), ".yaml")
    )
    launch = build_launch_plan(loaded.config, loaded.config.jobs[:1])[0]

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self._completed = False
            self.stdout = io.StringIO(
                f"{PROGRESS_EVENT_PREFIX}"
                + json.dumps(
                    {
                        "job_name": "TFT",
                        "model_index": 1,
                        "total_models": 1,
                        "total_steps": 2,
                        "completed_steps": 1,
                        "total_folds": 2,
                        "current_fold": 0,
                        "phase": "replay",
                        "status": "running",
                        "detail": "mse=1.0000",
                        "event": "fold-done",
                        "progress_pct": 50,
                        "progress_text": "[#########---------] 1/2  50%",
                    }
                )
                + "\nworker note\n"
            )

        def poll(self):
            return 0

        def wait(self):
            self._completed = True
            return 0

    monkeypatch.setattr("runtime_support.scheduler.subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "runtime_support.scheduler._worker_command",
        lambda *_args, **_kwargs: ["python", "fake_worker.py"],
    )

    run_parallel_jobs(tmp_path, loaded, [launch], tmp_path / "scheduler")

    captured = capsys.readouterr()
    assert "[summary]" in captured.out
    assert "[model:TFT]" in captured.out
    assert "completed=1/1" in captured.out
    assert "[worker:TFT] worker note" in captured.out
    assert PROGRESS_EVENT_PREFIX not in captured.out


def test_scheduler_finalize_recreates_missing_worker_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,chan_b\n2020-01-01,1,2,3\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, _payload(), ".yaml")
    )
    launch = build_launch_plan(loaded.config, loaded.config.jobs[:1])[0]
    scheduler_root = tmp_path / "scheduler"

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.StringIO("")

        def poll(self):
            return 0

        def wait(self):
            worker_root = scheduler_root / "workers" / launch.job_name
            if worker_root.exists():
                for path in sorted(worker_root.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    else:
                        path.rmdir()
                worker_root.rmdir()
            return 7

    monkeypatch.setattr("runtime_support.scheduler.subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "runtime_support.scheduler._worker_command",
        lambda *_args, **_kwargs: ["python", "fake_worker.py"],
    )

    results = run_parallel_jobs(tmp_path, loaded, [launch], scheduler_root)

    assert results[0]["returncode"] == 7
    summary_path = scheduler_root / "workers" / launch.job_name / "summary.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["returncode"] == 7


def test_runtime_executes_single_job_with_dummy_model(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main

    output_root = tmp_path / "run"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    assert (output_root / "cv" / "DummyUnivariate_forecasts.csv").exists()
    assert not (output_root / "holdout").exists()
    capability = json.loads(
        (output_root / "config" / "capability_report.json").read_text()
    )
    assert capability["DummyUnivariate"]["supports_auto"] is False


def test_runtime_logs_model_and_fold_progress_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main

    output_root = tmp_path / "run_progress"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "[summary] completed=0/1" in captured.out
    assert "[summary] completed=1/1" in captured.out
    assert "[model:DummyUnivariate]" in captured.out
    assert "phase=replay" in captured.out
    assert "fold=2/2" in captured.out
    assert "100%" in captured.out


def test_runtime_logs_fold_errors_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    import runtime_support.runner as runtime

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _boom)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--jobs",
                "DummyUnivariate",
                "--output-root",
                str(tmp_path / "run_error"),
            ]
        )

    captured = capsys.readouterr()
    assert "[model:DummyUnivariate]" in captured.out
    assert "status=failed" in captured.out
    assert "synthetic failure" in captured.out


def test_summary_builder_writes_leaderboard_and_last_fold_plots(tmp_path: Path):
    import runtime_support.runner as runtime

    run_root = tmp_path / "summary_run"
    cv_dir = run_root / "cv"
    cv_dir.mkdir(parents=True)
    residual_model_dir = run_root / "residual" / "ModelA"
    (residual_model_dir / "folds" / "fold_000").mkdir(parents=True)
    (residual_model_dir / "folds" / "fold_001").mkdir(parents=True)
    config_dir = run_root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "config.resolved.json").write_text(
        json.dumps(
            {
                "task": {"name": "brentoil_case1"},
                "dataset": {
                    "target_col": "Com_BrentCrudeOil",
                    "hist_exog_cols": ["Com_Gasoline", "Idx_OVX"],
                },
                "cv": {
                    "horizon": 8,
                    "step_size": 8,
                    "n_windows": 12,
                    "gap": 0,
                    "overlap_eval_policy": "by_cutoff_mean",
                },
            }
        ),
        encoding="utf-8",
    )
    for fold_idx, corrected_metrics in {
        0: {
            "MAE": 0.4,
            "MSE": 0.16,
            "RMSE": 0.4,
            "MAPE": 0.04,
            "NRMSE": 0.08,
            "R2": 0.95,
        },
        1: {
            "MAE": 0.2,
            "MSE": 0.04,
            "RMSE": 0.2,
            "MAPE": 0.02,
            "NRMSE": 0.04,
            "R2": 0.98,
        },
    }.items():
        (
            residual_model_dir / "folds" / f"fold_{fold_idx:03d}" / "metrics.json"
        ).write_text(
            json.dumps(
                {
                    "fold_idx": fold_idx,
                    "cutoff": f"2020-01-{15 + (7 * fold_idx):02d}",
                    "base_metrics": {},
                    "corrected_metrics": corrected_metrics,
                }
            ),
            encoding="utf-8",
        )
    pd.DataFrame(
        [
            {
                "fold_idx": 0,
                "cutoff": "2020-01-15",
                "MAE": 1.0,
                "MSE": 1.0,
                "RMSE": 1.0,
                "MAPE": 0.1,
                "NRMSE": 0.25,
                "R2": 0.8,
            },
            {
                "fold_idx": 1,
                "cutoff": "2020-01-22",
                "MAE": 0.5,
                "MSE": 0.25,
                "RMSE": 0.5,
                "MAPE": 0.05,
                "NRMSE": 0.10,
                "R2": 0.9,
            },
        ]
    ).to_csv(cv_dir / "ModelA_metrics_by_cutoff.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_idx": 0,
                "cutoff": "2020-01-15",
                "MAE": 2.0,
                "MSE": 4.0,
                "RMSE": 2.0,
                "MAPE": 0.2,
                "NRMSE": 0.50,
                "R2": 0.4,
            },
            {
                "fold_idx": 1,
                "cutoff": "2020-01-22",
                "MAE": 1.5,
                "MSE": 2.25,
                "RMSE": 1.5,
                "MAPE": 0.15,
                "NRMSE": 0.40,
                "R2": 0.6,
            },
        ]
    ).to_csv(cv_dir / "ModelB_metrics_by_cutoff.csv", index=False)
    for model_name, values in {
        "ModelA": [10.0, 11.0],
        "ModelB": [9.5, 10.5],
    }.items():
        pd.DataFrame(
            [
                {
                    "model": model_name,
                    "fold_idx": 1,
                    "cutoff": "2020-01-22",
                    "train_end_ds": "2020-01-22",
                    "unique_id": "target",
                    "ds": "2020-01-29",
                    "horizon_step": 1,
                    "y": 10.0,
                    "y_hat": values[0],
                },
                {
                    "model": model_name,
                    "fold_idx": 1,
                    "cutoff": "2020-01-22",
                    "train_end_ds": "2020-01-22",
                    "unique_id": "target",
                    "ds": "2020-02-05",
                    "horizon_step": 2,
                    "y": 11.0,
                    "y_hat": values[1],
                },
            ]
        ).to_csv(cv_dir / f"{model_name}_forecasts.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold_idx": 1,
                "cutoff": "2020-01-22",
                "train_end_ds": "2020-01-22",
                "unique_id": "target",
                "ds": "2020-01-29",
                "horizon_step": 1,
                "y": 10.0,
                "y_hat_corrected": 10.1,
            },
            {
                "fold_idx": 1,
                "cutoff": "2020-01-22",
                "train_end_ds": "2020-01-22",
                "unique_id": "target",
                "ds": "2020-02-05",
                "horizon_step": 2,
                "y": 11.0,
                "y_hat_corrected": 10.9,
            },
        ]
    ).to_csv(residual_model_dir / "corrected_folds.csv", index=False)
    (run_root / "models" / "ModelA" / "folds" / "fold_000").mkdir(parents=True)
    (run_root / "models" / "ModelA" / "folds" / "fold_001").mkdir(parents=True)
    pd.DataFrame(
        [
            {"global_step": 10, "train_loss": 1.0, "val_loss": 1.2},
            {"global_step": 20, "train_loss": 0.8, "val_loss": 1.0},
        ]
    ).to_csv(
        run_root
        / "models"
        / "ModelA"
        / "folds"
        / "fold_000"
        / "loss_curve_every_10_global_steps.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"global_step": 10, "train_loss": 0.7, "val_loss": 0.9},
        ]
    ).to_csv(
        run_root
        / "models"
        / "ModelA"
        / "folds"
        / "fold_001"
        / "loss_curve_every_10_global_steps.csv",
        index=False,
    )

    artifacts = runtime._build_summary_artifacts(run_root)

    leaderboard_path = run_root / "summary" / "leaderboard.csv"
    loss_artifacts_path = run_root / "summary" / "loss_curve_artifacts.csv"
    markdown_path = run_root / "summary" / "sample.md"
    assert artifacts["leaderboard"] == str(leaderboard_path)
    assert artifacts["loss_artifacts"] == str(loss_artifacts_path)
    assert artifacts["markdown"] == str(markdown_path)
    assert leaderboard_path.exists()
    assert loss_artifacts_path.exists()
    assert markdown_path.exists()
    leaderboard = pd.read_csv(leaderboard_path)
    loss_artifacts = pd.read_csv(loss_artifacts_path)
    assert leaderboard.loc[0, "rank"] == 1
    assert leaderboard.loc[0, "model"] == "ModelA_res"
    assert leaderboard["model"].tolist() == ["ModelA_res", "ModelA", "ModelB"]
    assert "mean_fold_mape" in leaderboard.columns
    assert "mean_fold_nrmse" in leaderboard.columns
    assert "mean_fold_r2" in leaderboard.columns
    assert loss_artifacts.to_dict(orient="records") == [
        {
            "model": "ModelA",
            "fold_idx": 0,
            "sample_every_n_steps": 10,
            "sample_count": 2,
            "first_global_step": 10,
            "last_global_step": 20,
            "loss_csv_path": "models/ModelA/folds/fold_000/loss_curve_every_10_global_steps.csv",
        },
        {
            "model": "ModelA",
            "fold_idx": 1,
            "sample_every_n_steps": 10,
            "sample_count": 1,
            "first_global_step": 10,
            "last_global_step": 10,
            "loss_csv_path": "models/ModelA/folds/fold_001/loss_curve_every_10_global_steps.csv",
        },
    ]
    report = markdown_path.read_text(encoding="utf-8")
    assert "# 02. 데이터 및 모델 세팅" in report
    assert "## **Case 1 | BrentCrude**" in report
    assert "| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |" in report
    assert "| 1 | ModelA_res | 3.00% | 0.06 | 0.30 | 0.97 |" in report
    assert "| 2 | ModelA | 7.50% | 0.18 | 0.75 | 0.85 |" in report
    assert "hist_exog_cols:" in report
    assert (run_root / "summary" / "last_fold_all_models.png").exists()
    assert (run_root / "summary" / "last_fold_top3.png").exists()
    assert (run_root / "summary" / "last_fold_top5.png").exists()
    assert (run_root / "summary" / "residual" / "ModelA.png").exists()
    assert not (run_root / "summary" / "residual" / "ModelB.png").exists()


def test_summary_builder_leaves_missing_report_values_blank(tmp_path: Path):
    import runtime_support.runner as runtime

    run_root = tmp_path / "blank_summary"
    cv_dir = run_root / "cv"
    cv_dir.mkdir(parents=True)
    (run_root / "config").mkdir(parents=True)
    (run_root / "config" / "config.resolved.json").write_text(
        json.dumps(
            {
                "task": {"name": "wti_case9"},
                "dataset": {
                    "target_col": "Com_CrudeOil",
                    "hist_exog_cols": [],
                },
                "cv": {
                    "horizon": 4,
                    "step_size": 2,
                    "n_windows": 3,
                    "gap": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "fold_idx": 0,
                "cutoff": "2020-01-15",
                "MAE": 1.0,
                "MSE": 1.0,
                "RMSE": 1.0,
                "MAPE": 0.1,
                "NRMSE": float("nan"),
                "R2": float("nan"),
            }
        ]
    ).to_csv(cv_dir / "ModelA_metrics_by_cutoff.csv", index=False)

    artifacts = runtime._build_summary_artifacts(run_root)

    report = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    assert "## **Case 9 | WTI**" in report
    assert "| 1 | ModelA | 10.00% |  | 1.00 |  |" in report


def test_compute_metrics_includes_range_normalized_nrmse_and_r2():
    import runtime_support.runner as runtime

    metrics = runtime._compute_metrics(
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0, 2.0, 5.0]),
    )

    assert metrics["RMSE"] == pytest.approx((4.0 / 3.0) ** 0.5)
    assert metrics["NRMSE"] == pytest.approx(((4.0 / 3.0) ** 0.5) / 2.0)
    assert metrics["R2"] == pytest.approx(-1.0)


def test_runtime_skip_summary_env_suppresses_summary_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from runtime_support.runner import main as runtime_main

    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    output_root = tmp_path / "worker_like_run"

    monkeypatch.setenv("NEURALFORECAST_SKIP_SUMMARY_ARTIFACTS", "1")

    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    assert (output_root / "cv" / "DummyUnivariate_forecasts.csv").exists()
    assert not (output_root / "summary").exists()


def test_runtime_smoke_writes_summary_artifacts_for_dummy_model(tmp_path: Path):
    from runtime_support.runner import main as runtime_main

    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")
    output_root = tmp_path / "run_summary"

    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )

    leaderboard_path = output_root / "summary" / "leaderboard.csv"
    markdown_path = output_root / "summary" / "sample.md"
    assert code == 0
    assert leaderboard_path.exists()
    assert markdown_path.exists()
    assert (output_root / "summary" / "last_fold_all_models.png").exists()
    assert (output_root / "summary" / "last_fold_top3.png").exists()
    assert (output_root / "summary" / "last_fold_top5.png").exists()
    leaderboard = pd.read_csv(leaderboard_path)
    assert "rank" in leaderboard.columns
    assert (
        "| Rank (nRMSE) | Model | MAPE | nRMSE | MAE | R2 |"
        in markdown_path.read_text(encoding="utf-8")
    )


def test_trajectory_frame_contains_train_and_val_series():
    import runtime_support.runner as runtime

    nf = SimpleNamespace(
        models=[
            SimpleNamespace(
                train_trajectories=[(1, 0.9), (2, 0.7)],
                valid_trajectories=[(1, 1.1), (2, 0.8)],
            )
        ]
    )

    frame = runtime._trajectory_frame(nf)

    assert frame.columns.tolist() == ["global_step", "train_loss", "val_loss"]
    assert frame.to_dict(orient="records") == [
        {"global_step": 1, "train_loss": 0.9, "val_loss": 1.1},
        {"global_step": 2, "train_loss": 0.7, "val_loss": 0.8},
    ]


def test_loss_curve_series_drop_sparse_validation_gaps():
    import runtime_support.runner as runtime

    curve_frame = pd.DataFrame(
        [
            {"global_step": 1, "train_loss": 10.0, "val_loss": 12.0},
            {"global_step": 2, "train_loss": 8.0, "val_loss": np.nan},
            {"global_step": 3, "train_loss": 6.0, "val_loss": np.nan},
            {"global_step": 4, "train_loss": 4.0, "val_loss": 5.0},
        ]
    )

    train_series, val_series = runtime._loss_curve_series(curve_frame)

    assert train_series.to_dict(orient="records") == [
        {"global_step": 1, "train_loss": 10.0},
        {"global_step": 2, "train_loss": 8.0},
        {"global_step": 3, "train_loss": 6.0},
        {"global_step": 4, "train_loss": 4.0},
    ]
    assert val_series.to_dict(orient="records") == [
        {"global_step": 1, "val_loss": 12.0},
        {"global_step": 4, "val_loss": 5.0},
    ]


def test_sample_loss_curve_frame_keeps_every_10_global_steps():
    import runtime_support.runner as runtime

    curve_frame = pd.DataFrame(
        [
            {"global_step": step, "train_loss": float(step), "val_loss": float(step) + 0.5}
            for step in range(1, 26)
        ]
    )

    sampled = runtime._sample_loss_curve_frame(curve_frame)

    assert sampled.to_dict(orient="records") == [
        {"global_step": 10, "train_loss": 10.0, "val_loss": 10.5},
        {"global_step": 20, "train_loss": 20.0, "val_loss": 20.5},
    ]


def test_configure_loss_curve_axis_uses_log_base10_ticks():
    import runtime_support.runner as runtime

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    curve_frame = pd.DataFrame(
        [
            {"global_step": 1, "train_loss": 100.0, "val_loss": 120.0},
            {"global_step": 2, "train_loss": 10.0, "val_loss": np.nan},
            {"global_step": 3, "train_loss": 1.0, "val_loss": 5.0},
        ]
    )

    figure, axis = plt.subplots()
    runtime._configure_loss_curve_axis(axis, curve_frame)

    assert axis.get_yscale() == "log"
    assert isinstance(axis.yaxis.get_major_locator(), LogLocator)
    assert isinstance(axis.yaxis.get_major_formatter(), LogFormatterMathtext)
    plt.close(figure)


def test_build_tscv_splits_uses_configured_step_size(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 3, "step_size": 2, "n_windows": 3, "gap": 0})
    payload["dataset"]["hist_exog_cols"] = []
    (tmp_path / "data.csv").write_text(
        "dt,target\n"
        "2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
        "2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n"
        "2020-02-26,9\n2020-03-04,10\n2020-03-11,11\n2020-03-18,12\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import runtime_support.runner as runtime

    splits = runtime._build_tscv_splits(12, loaded.config.cv)

    assert [test_idx for _, test_idx in splits] == [
        [5, 6, 7],
        [7, 8, 9],
        [9, 10, 11],
    ]


def test_runtime_outer_cv_cutoffs_follow_step_size(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 2, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main

    output_root = tmp_path / "run_step_size_outer_cv"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )

    forecasts = pd.read_csv(output_root / "cv" / "DummyUnivariate_forecasts.csv")

    assert code == 0
    assert forecasts["cutoff"].tolist() == [
        "2020-01-15 00:00:00",
        "2020-01-29 00:00:00",
    ]


def test_runtime_uses_config_parent_and_task_name_for_default_run_directory(
    tmp_path: Path,
):
    payload = _payload()
    payload["task"] = {"name": "pytest_task_default_output"}
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main

    code = runtime_main(["--config", str(config_path), "--jobs", "DummyUnivariate"])

    output_root = REPO_ROOT / "runs" / f"{tmp_path.name}_pytest_task_default_output"
    try:
        assert code == 0
        assert (output_root / "cv" / "DummyUnivariate_forecasts.csv").exists()
        resolved_config = output_root / "config" / "config.resolved.json"
        assert resolved_config.exists()
        assert "pytest_task_default_output" in resolved_config.read_text(
            encoding="utf-8"
        )
    finally:
        if output_root.exists():
            import shutil

            shutil.rmtree(output_root)


def test_default_output_root_uses_repo_name_for_repo_root_config(tmp_path: Path):
    from runtime_support.runner import _default_output_root

    repo_root = tmp_path / "neuralforecast"
    repo_root.mkdir(parents=True)
    config_path = _write_repo_root_contract_config(repo_root)

    loaded = load_app_config(repo_root, config_path=config_path)

    assert _default_output_root(repo_root, loaded) == (
        repo_root / "runs" / f"{repo_root.name}_semi_test"
    )


def test_load_app_config_requires_explicit_path():
    with pytest.raises(
        ValueError,
        match="config path is required; pass --config/--config-path or --config-toml",
    ):
        load_app_config(REPO_ROOT)


@pytest.mark.parametrize(
    ("config_path", "expected_name"),
    [
        ("yaml/experiment/feature_set/wti-case3.yaml", "feature_set_wti_case3"),
        (
            "yaml/experiment/feature_set_HPT_n100_bs/brentoil-case3.yaml",
            "feature_set_HPT_n100_bs_brentoil_case3_HPO",
        ),
        ("yaml/experiment/feature_set_bs/wti-case3.yaml", "feature_set_bs_wti_case3_bs"),
        (
            "yaml/experiment/feature_set_bs_preforcast/wti-case3.yaml",
            "feature_set_bs_preforcast_wti_case3_bs_preforcast",
        ),
    ],
)
def test_default_output_root_uses_config_parent_for_nested_repo_configs(
    config_path: str,
    expected_name: str,
):
    from runtime_support.runner import _default_output_root

    loaded = load_app_config(REPO_ROOT, config_path=config_path)

    assert _default_output_root(REPO_ROOT, loaded) == (REPO_ROOT / "runs" / expected_name)


def test_resolve_run_roots_uses_default_output_root(tmp_path: Path):
    payload = _payload()
    payload["task"] = {"name": "pytest_task_run_roots"}
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_dir = tmp_path / "yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "case.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    loaded = load_app_config(tmp_path, config_path=config_path)

    import runtime_support.runner as runtime

    resolved = runtime._resolve_run_roots(tmp_path, loaded, output_root=None)
    expected_root = runtime._default_output_root(tmp_path, loaded)
    assert resolved["run_root"] == expected_root
    assert resolved["summary_root"] == expected_root


def test_resolve_run_roots_preserves_explicit_output_root(tmp_path: Path):
    payload = _payload()
    payload["task"] = {"name": "pytest_task_run_roots_explicit"}
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_dir = tmp_path / "yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "case.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    loaded = load_app_config(tmp_path, config_path=config_path)

    import runtime_support.runner as runtime

    explicit_root = tmp_path / "custom_output_root"
    resolved = runtime._resolve_run_roots(
        tmp_path, loaded, output_root=str(explicit_root)
    )
    assert resolved["run_root"] == explicit_root
    assert resolved["summary_root"] == explicit_root


def test_runtime_infers_freq_when_omitted(tmp_path: Path):
    payload = _payload()
    payload["dataset"].pop("freq", None)
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-06,1,9\n2020-01-13,2,8\n2020-01-20,3,7\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    import runtime_support.runner as runtime

    loaded = load_app_config(tmp_path, config_path=config_path)
    import pandas as pd

    source_df = pd.read_csv(tmp_path / "data.csv")
    assert runtime._resolve_freq(loaded, source_df) == "W-MON"


def test_jobs_use_unique_model_identifiers(tmp_path: Path):
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, _payload(), ".yaml")
    )
    assert [job.model for job in loaded.config.jobs] == ["TFT", "iTransformer"]


def test_load_app_config_rejects_duplicate_centralized_job_keys(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {"max_steps": 123}}]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    path = _write_config(tmp_path, payload, ".yaml")
    with pytest.raises(ValueError, match="repeats centralized training key"):
        load_app_config(tmp_path, config_path=path)


def test_load_app_config_rejects_removed_residual_train_source(tmp_path: Path):
    payload = _payload()
    payload["residual"]["train_source"] = "oof_cv"
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    path = _write_config(tmp_path, payload, ".yaml")
    with pytest.raises(ValueError, match="residual.train_source has been removed"):
        load_app_config(tmp_path, config_path=path)


def test_load_app_config_rejects_removed_cv_final_holdout(tmp_path: Path):
    payload = _payload()
    payload["cv"]["final_holdout"] = 12
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    path = _write_config(tmp_path, payload, ".yaml")
    with pytest.raises(ValueError, match="cv.final_holdout has been removed"):
        load_app_config(tmp_path, config_path=path)


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "disabled"])
def test_load_app_config_rejects_invalid_residual_target_values(
    tmp_path: Path, enabled: bool
):
    payload = _payload()
    payload["residual"] = {
        "enabled": enabled,
        "model": "xgboost",
        "target": "weird",
        "params": {"n_estimators": 8, "max_depth": 2},
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n",
        encoding="utf-8",
    )
    path = _write_config(tmp_path, payload, ".yaml")

    with pytest.raises(
        ValueError, match="residual.target must be one of: level, delta"
    ):
        load_app_config(tmp_path, config_path=path)


def test_residual_registry_builds_xgboost_plugin_with_custom_params():
    plugin = _import_build_residual_plugin()(
        {
            "model": "xgboost",
            "params": {"n_estimators": 8, "max_depth": 2},
        }
    )
    assert plugin.metadata()["plugin"] == "xgboost"
    assert plugin.metadata()["n_estimators"] == 8


def test_residual_registry_surfaces_cpu_thread_overrides():
    plugin = _import_build_residual_plugin()(
        {
            "model": "xgboost",
            "cpu_threads": 4,
            "params": {"n_estimators": 8, "max_depth": 2},
        }
    )

    assert plugin.metadata()["cpu_threads"] == 4


@pytest.mark.parametrize(
    "model_name",
    EXPECTED_SUPPORTED_RESIDUAL_MODELS,
    ids=EXPECTED_SUPPORTED_RESIDUAL_MODELS,
)
def test_residual_registry_builds_all_supported_plugins(model_name: str):
    _skip_missing_residual_backend(model_name)

    plugin = _import_build_residual_plugin()({"model": model_name, "params": {}})

    assert plugin.metadata()["plugin"] == model_name


def test_residual_registry_rejects_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported residual model: nope"):
        _import_build_residual_plugin()({"model": "nope", "params": {}})


def test_runtime_generates_per_fold_residual_artifacts_with_dummy_model(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 4, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2},
    }
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main

    output_root = tmp_path / "run_resid"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    residual_root = output_root / "residual" / "DummyUnivariate"
    assert not (residual_root / "training_panel.csv").exists()
    assert not (residual_root / "corrected_holdout.csv").exists()
    assert (residual_root / "corrected_folds.csv").exists()
    assert (residual_root / "diagnostics.json").exists()
    for fold_idx in range(4):
        fold_root = residual_root / "folds" / f"fold_{fold_idx:03d}"
        assert (fold_root / "backcast_panel.csv").exists()
        assert (fold_root / "corrected_eval.csv").exists()
        assert (fold_root / "residual_checkpoint" / "model.ubj").exists()
        assert (fold_root / "base_checkpoint" / "fit_summary.json").exists()
    corrected_folds = pd.read_csv(residual_root / "corrected_folds.csv")
    assert corrected_folds["fold_idx"].tolist() == [0, 1, 2, 3]
    assert "panel_split" in corrected_folds.columns
    assert set(corrected_folds["panel_split"]) == {"fold_eval"}
    diagnostics = pd.read_json(residual_root / "diagnostics.json", typ="series")
    assert diagnostics["corrected_eval_mode"] == "per_fold_backcast_runtime"
    assert diagnostics["fold_count"] == 4
    assert diagnostics["residual.target"] == "level"
    assert diagnostics["tscv_policy"]["gap"] == 0
    leaderboard = pd.read_csv(output_root / "summary" / "leaderboard.csv")
    assert "DummyUnivariate" in set(leaderboard["model"])
    assert "DummyUnivariate_res" in set(leaderboard["model"])
    report = (output_root / "summary" / "sample.md").read_text(encoding="utf-8")
    assert "DummyUnivariate_res" in report
    assert (output_root / "summary" / "residual" / "DummyUnivariate.png").exists()


def test_runtime_diff_preserves_raw_scale_for_baseline_learned_artifacts(
    tmp_path: Path,
):
    from runtime_support.runner import main as runtime_main

    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n"
    )
    cases = [
        ("Naive", {}, tmp_path / "run_diff_baseline"),
        (
            "DummyUnivariate",
            {"start_padding_enabled": True},
            tmp_path / "run_diff_learned",
        ),
    ]
    for model_name, params, output_root in cases:
        payload = _payload()
        payload["runtime"]["transformations_target"] = "diff"
        payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
        payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
        payload["dataset"]["hist_exog_cols"] = []
        payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
        payload["jobs"] = [{"model": model_name, "params": params}]
        (tmp_path / "data.csv").write_text(data, encoding="utf-8")
        config_path = _write_config(tmp_path, payload, ".yaml")

        code = runtime_main(
            [
                "--config",
                str(config_path),
                "--jobs",
                model_name,
                "--output-root",
                str(output_root),
            ]
        )

        forecasts = pd.read_csv(output_root / "cv" / f"{model_name}_forecasts.csv")
        assert code == 0
        assert forecasts["y"].tolist() == [6.0, 7.0]
        assert forecasts["y_hat"].tolist() == [6.0, 7.0]
        assert forecasts["y_hat"].tolist() != [1.0, 1.0]


def test_runtime_diff_residual_enabled_skips_short_backcast_history_without_crash(
    tmp_path: Path,
):
    from runtime_support.runner import main as runtime_main

    payload = _payload()
    payload["runtime"]["transformations_target"] = "diff"
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2},
    }
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    (tmp_path / "data.csv").write_text(
        (
            "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
            "2020-01-22,4\n2020-01-29,5\n"
        ),
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    output_root = tmp_path / "run_diff_residual"

    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    assert (
        output_root
        / "residual"
        / "DummyUnivariate"
        / "folds"
        / "fold_000"
        / "backcast_panel.csv"
    ).exists()


def test_runtime_writes_loss_curve_images_for_residual_disabled_learned_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main
    import runtime_support.runner as runtime

    class _FakeNF:
        def __init__(self, model_name: str, target_col: str):
            self.models = [
                SimpleNamespace(
                    train_trajectories=[(1, 1.0), (10, 0.6), (20, 0.3)],
                    valid_trajectories=[(1, 1.2), (10, 0.8), (20, 0.4)],
                )
            ]
            self._model_name = model_name
            self._target_col = target_col

        def predict(self, df, static_df=None, futr_df=None):
            source = futr_df if futr_df is not None else df.tail(1)
            ds = source["ds"].reset_index(drop=True)
            return pd.DataFrame(
                {
                    "unique_id": [self._target_col] * len(ds),
                    "ds": ds,
                    self._model_name: [1.0] * len(ds),
                }
            )

    def _fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
    ):
        nf = _FakeNF(job.model, loaded.config.dataset.target_col)
        predictions = pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.Series([source_df.iloc[test_idx[0]][loaded.config.dataset.dt_col]]),
                job.model: [float(source_df.iloc[test_idx[0]][loaded.config.dataset.target_col])],
            }
        )
        actuals = pd.Series([float(source_df.iloc[test_idx[0]][loaded.config.dataset.target_col])])
        return (
            predictions,
            actuals,
            source_df.iloc[train_idx][-1:][loaded.config.dataset.dt_col].iloc[0],
            source_df.iloc[train_idx].reset_index(drop=True),
            nf,
        )

    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit_and_predict_fold)

    output_root = tmp_path / "run_loss_curves"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    for fold_idx in range(2):
        assert (
            output_root
            / "models"
            / "DummyUnivariate"
            / "folds"
            / f"fold_{fold_idx:03d}"
            / "loss_curve.png"
        ).exists()
        assert (
            output_root
            / "models"
            / "DummyUnivariate"
            / "folds"
            / f"fold_{fold_idx:03d}"
            / "loss_curve_every_10_global_steps.csv"
        ).exists()
    assert (output_root / "summary" / "loss_curve_artifacts.csv").exists()


def test_runtime_writes_loss_curve_images_for_residual_enabled_learned_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2},
    }
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main
    import runtime_support.runner as runtime

    class _FakeNF:
        def __init__(self, model_name: str, target_col: str):
            self.models = [
                SimpleNamespace(
                    train_trajectories=[(1, 1.0), (10, 0.6), (20, 0.3)],
                    valid_trajectories=[(1, 1.2), (10, 0.8), (20, 0.4)],
                )
            ]
            self._model_name = model_name
            self._target_col = target_col

        def predict(self, df, static_df=None, futr_df=None):
            source = futr_df if futr_df is not None else df.tail(1)
            ds = source["ds"].reset_index(drop=True)
            return pd.DataFrame(
                {
                    "unique_id": [self._target_col] * len(ds),
                    "ds": ds,
                    self._model_name: [1.0] * len(ds),
                }
            )

    def _fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
    ):
        nf = _FakeNF(job.model, loaded.config.dataset.target_col)
        predictions = pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.Series([source_df.iloc[test_idx[0]][loaded.config.dataset.dt_col]]),
                job.model: [float(source_df.iloc[test_idx[0]][loaded.config.dataset.target_col])],
            }
        )
        actuals = pd.Series([float(source_df.iloc[test_idx[0]][loaded.config.dataset.target_col])])
        return (
            predictions,
            actuals,
            source_df.iloc[train_idx][-1:][loaded.config.dataset.dt_col].iloc[0],
            source_df.iloc[train_idx].reset_index(drop=True),
            nf,
        )

    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit_and_predict_fold)

    output_root = tmp_path / "run_loss_curves_residual"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "DummyUnivariate",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    for fold_idx in range(2):
        assert (
            output_root
            / "models"
            / "DummyUnivariate"
            / "folds"
            / f"fold_{fold_idx:03d}"
            / "loss_curve.png"
        ).exists()
        assert (
            output_root
            / "models"
            / "DummyUnivariate"
            / "folds"
            / f"fold_{fold_idx:03d}"
            / "loss_curve_every_10_global_steps.csv"
        ).exists()
        assert (
            output_root
            / "residual"
            / "DummyUnivariate"
            / "folds"
            / f"fold_{fold_idx:03d}"
            / "base_checkpoint"
            / "fit_summary.json"
        ).exists()
    assert (output_root / "summary" / "loss_curve_artifacts.csv").exists()


def test_baseline_models_do_not_write_loss_curve_images(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["jobs"] = [{"model": "Naive", "params": {}}]
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n"
        "2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main

    output_root = tmp_path / "run_baseline"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "Naive",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    assert not (output_root / "models" / "Naive" / "folds").exists()


class _RecordingResidualPlugin(ResidualPlugin):
    name = "recording"

    def __init__(self, plugin_log: list["_RecordingLog"]):
        self._plugin_log = plugin_log
        self._record: _RecordingLog = {
            "fit_lengths": [],
            "predict_panel_splits": [],
        }
        self._plugin_log.append(self._record)

    def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
        self._record["fit_lengths"].append(len(panel_df))

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        self._record["predict_panel_splits"].append(
            panel_df["panel_split"].unique().tolist()
        )
        return panel_df.copy().assign(residual_hat=0.0)

    def metadata(self) -> dict[str, object]:
        return {"plugin": self.name}


class _RecordingLog(TypedDict):
    fit_lengths: list[int]
    predict_panel_splits: list[list[str]]


class _CheckpointResidualPlugin(ResidualPlugin):
    def __init__(self, name: str):
        self.name = name

    def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
        context.output_dir.mkdir(parents=True, exist_ok=True)
        (context.output_dir / "model.ubj").write_text(self.name, encoding="utf-8")

    def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        return panel_df.copy().assign(residual_hat=0.0)

    def metadata(self) -> dict[str, object]:
        return {"plugin": self.name}


def test_apply_residual_plugin_uses_fold_local_backcasts_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    plugin_log: list[_RecordingLog] = []

    def _build_plugin(_config):
        return _RecordingResidualPlugin(plugin_log)

    monkeypatch.setattr(runtime, "build_residual_plugin", _build_plugin)
    config_path = _write_repo_root_contract_config(tmp_path)
    loaded = load_app_config(tmp_path, config_path=config_path)
    job = next(job for job in loaded.config.jobs if job.model == "TFT")
    run_root = tmp_path / "run"
    fold_payloads = []
    for fold_idx, fit_length in enumerate((1, 2, 3)):
        backcast_panel = pd.DataFrame(
            {
                "model_name": [job.model] * fit_length,
                "fold_idx": [fold_idx] * fit_length,
                "panel_split": ["backcast_train"] * fit_length,
                "unique_id": ["target"] * fit_length,
                "cutoff": pd.to_datetime(["2020-01-08"] * fit_length),
                "train_end_ds": pd.to_datetime(["2020-01-08"] * fit_length),
                "ds": pd.to_datetime(["2020-01-15"] * fit_length),
                "horizon_step": list(range(1, fit_length + 1)),
                "y_hat_base": [1.0] * fit_length,
                "y": [1.5] * fit_length,
                "residual_target": [0.5] * fit_length,
            }
        )
        eval_panel = pd.DataFrame(
            {
                "model_name": [job.model],
                "fold_idx": [fold_idx],
                "panel_split": ["fold_eval"],
                "unique_id": ["target"],
                "cutoff": pd.to_datetime(["2020-01-15"]),
                "train_end_ds": pd.to_datetime(["2020-01-15"]),
                "ds": pd.to_datetime(["2020-01-22"]),
                "horizon_step": [1],
                "y_hat_base": [2.0],
                "y": [2.5],
                "residual_target": [0.5],
            }
        )
        fold_payloads.append(
            {
                "fold_idx": fold_idx,
                "backcast_panel": backcast_panel,
                "eval_panel": eval_panel,
                "base_summary": {"fold_idx": fold_idx},
            }
        )

    manifest_path = run_root / "manifest" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": [{"model": job.model}], "residual": {}}, indent=2),
        encoding="utf-8",
    )

    runtime._apply_residual_plugin(
        loaded, job, run_root, fold_payloads, manifest_path=manifest_path
    )

    assert [record["fit_lengths"][0] for record in plugin_log] == [1, 2, 3]
    assert all(
        splits == ["fold_eval"]
        for record in plugin_log
        for splits in record["predict_panel_splits"]
    )
    residual_root = run_root / "residual" / job.model
    assert (residual_root / "corrected_folds.csv").exists()
    assert not (residual_root / "corrected_holdout.csv").exists()


def test_fold_panels_include_selected_residual_exog_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    payload = _payload()
    payload["cv"].update({"horizon": 2, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["dataset"]["static_exog_cols"] = ["static_a"]
    payload["residual"]["features"] = {
        "exog_sources": {
            "hist": ["hist_a"],
            "futr": ["futr_a"],
            "static": ["static_a"],
        }
    }
    (tmp_path / "data.csv").write_text(
        "\n".join(
            [
                "dt,target,hist_a,futr_a,static_a",
                "2020-01-01,1,10,100,1000",
                "2020-01-08,2,11,101,1001",
                "2020-01-15,3,12,102,1002",
                "2020-01-22,4,13,103,1003",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    job = loaded.config.jobs[0]
    train_df = pd.read_csv(tmp_path / "data.csv")
    future_df = train_df.iloc[2:4].reset_index(drop=True)
    target_predictions = pd.DataFrame(
        {
            "unique_id": ["target", "target"],
            "ds": pd.to_datetime(["2020-01-15", "2020-01-22"]),
            job.model: [9.5, 10.5],
        }
    )
    actuals = future_df["target"].reset_index(drop=True)

    monkeypatch.setattr(
        runtime,
        "_iter_backcast_cutoff_indices",
        lambda **_kwargs: [1],
    )
    monkeypatch.setattr(
        runtime,
        "_build_adapter_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(
            fit_df=pd.DataFrame(), static_df=None, futr_df=None, metadata={}
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_predict_with_fitted_model",
        lambda _nf, _adapter_inputs: target_predictions.rename(
            columns={job.model: "TFT"}
        ),
    )

    backcast_panel = runtime._build_fold_backcast_panel(
        loaded,
        job,
        nf=cast(Any, object()),
        train_df=train_df,
        dt_col="dt",
        target_col="target",
        fold_idx=0,
    )
    eval_panel = runtime._build_fold_eval_panel(
        loaded,
        job,
        fold_idx=0,
        train_end_ds=pd.Timestamp("2020-01-08"),
        target_predictions=target_predictions,
        actuals=actuals,
        future_df=future_df,
        train_df=train_df.iloc[:2].reset_index(drop=True),
    )

    assert "hist_a_lag_1" in backcast_panel.columns
    assert "hist_a_lag_1" in eval_panel.columns
    assert "hist_a" not in backcast_panel.columns
    assert "hist_a" not in eval_panel.columns
    assert backcast_panel[["hist_a_lag_1", "futr_a", "static_a"]].to_dict(
        orient="records"
    ) == [
        {"hist_a_lag_1": 11, "futr_a": 102, "static_a": 1001},
        {"hist_a_lag_1": 11, "futr_a": 103, "static_a": 1001},
    ]
    assert eval_panel[["hist_a_lag_1", "futr_a", "static_a"]].to_dict(
        orient="records"
    ) == [
        {"hist_a_lag_1": 11, "futr_a": 102, "static_a": 1001},
        {"hist_a_lag_1": 11, "futr_a": 103, "static_a": 1001},
    ]


@pytest.mark.parametrize(
    "path", RESIDUAL_RUNTIME_SMOKE_FIXTURE_FILES, ids=lambda p: p.stem
)
def test_runtime_residual_checkpoint_contract_extends_to_all_supported_models(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    payload = _load_case_yaml(path)
    loaded = load_app_config(REPO_ROOT, config_path=path)
    job = loaded.config.jobs[0]
    run_root = tmp_path / path.stem
    manifest_path = run_root / "manifest" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": [{"model": job.model}], "residual": {}}, indent=2),
        encoding="utf-8",
    )
    fold_payloads = [
        {
            "fold_idx": 0,
            "backcast_panel": _residual_target_panel(
                [
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "backcast_train",
                        "unique_id": "target",
                        "cutoff": "2020-01-08",
                        "train_end_ds": "2020-01-08",
                        "ds": "2020-01-15",
                        "horizon_step": 1,
                        "y_hat_base": 10.0,
                        "y": 11.0,
                        "residual_target": 1.0,
                    }
                ]
            ),
            "eval_panel": _residual_target_panel(
                [
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "fold_eval",
                        "unique_id": "target",
                        "cutoff": "2020-01-15",
                        "train_end_ds": "2020-01-15",
                        "ds": "2020-01-22",
                        "horizon_step": 1,
                        "y_hat_base": 12.0,
                        "y": 13.0,
                        "residual_target": 1.0,
                    }
                ]
            ),
            "base_summary": {"fold_idx": 0},
        }
    ]

    monkeypatch.setattr(
        runtime,
        "build_residual_plugin",
        lambda config: _CheckpointResidualPlugin(config["model"]),
    )

    runtime._apply_residual_plugin(
        loaded,
        job,
        run_root,
        fold_payloads,
        manifest_path=manifest_path,
    )

    residual_root = run_root / "residual" / job.model
    assert (residual_root / "corrected_folds.csv").exists()
    assert (
        residual_root / "folds" / "fold_000" / "residual_checkpoint" / "model.ubj"
    ).exists()
    assert payload["residual"]["model"] in (
        residual_root / "plugin_metadata.json"
    ).read_text(encoding="utf-8")


def test_build_residual_target_preserves_level_mode_parity():
    import runtime_support.runner as runtime

    panel = pd.DataFrame(
        {
            "fold_idx": [0, 0],
            "cutoff": pd.to_datetime(["2020-01-08", "2020-01-08"]),
            "horizon_step": [2, 1],
            "y_hat_base": [11.0, 10.0],
            "y": [12.0, 11.0],
        }
    )

    residual_target = runtime.build_residual_target(panel, "level")

    assert residual_target.tolist() == [1.0, 1.0]


def test_build_residual_target_delta_resets_per_cutoff_and_preserves_row_order():
    import runtime_support.runner as runtime

    panel = pd.DataFrame(
        {
            "fold_idx": [0, 0, 0, 0],
            "cutoff": pd.to_datetime(
                ["2020-01-08", "2020-01-15", "2020-01-08", "2020-01-15"]
            ),
            "horizon_step": [2, 1, 1, 2],
            "y_hat_base": [11.0, 20.0, 10.0, 22.0],
            "y": [12.0, 21.0, 11.0, 25.0],
        }
    )

    residual_target = runtime.build_residual_target(panel, "delta")

    assert residual_target.tolist() == [0.0, 1.0, 1.0, 2.0]
    assert not residual_target.isna().any()


def test_reconstruct_corrected_forecast_delta_uses_grouped_cumsum_and_preserves_row_order():
    import runtime_support.runner as runtime

    panel = pd.DataFrame(
        {
            "fold_idx": [0, 0, 0, 0],
            "cutoff": pd.to_datetime(
                ["2020-01-08", "2020-01-15", "2020-01-08", "2020-01-15"]
            ),
            "horizon_step": [2, 1, 1, 2],
            "y_hat_base": [11.0, 20.0, 10.0, 22.0],
            "residual_hat": [1.0, 1.0, 1.0, 2.0],
        }
    )

    corrected = runtime.reconstruct_corrected_forecast(panel, "delta")

    assert corrected.tolist() == [13.0, 21.0, 11.0, 25.0]


def test_reconstruct_corrected_forecast_level_mode_parity():
    import runtime_support.runner as runtime

    panel = pd.DataFrame(
        {
            "fold_idx": [0, 0],
            "cutoff": pd.to_datetime(["2020-01-08", "2020-01-08"]),
            "horizon_step": [2, 1],
            "y_hat_base": [11.0, 10.0],
            "residual_hat": [1.5, 0.5],
        }
    )

    corrected = runtime.reconstruct_corrected_forecast(panel, "level")

    assert corrected.tolist() == [12.5, 10.5]


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("xgboost", {"n_estimators": 8, "max_depth": 2}),
        (
            "randomforest",
            {
                "n_estimators": 8,
                "max_depth": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
        ),
        (
            "lightgbm",
            {
                "n_estimators": 8,
                "max_depth": 2,
                            "num_leaves": 7,
                "min_child_samples": 5,
                "feature_fraction": 1.0,
            },
        ),
    ],
)
def test_residual_plugins_predict_panel_and_write_checkpoint(
    tmp_path: Path, model_name: str, params: dict[str, Any]
):
    _skip_missing_residual_backend(model_name)
    plugin = cast(
        Any,
        _import_build_residual_plugin()(
            {
                "model": model_name,
                "params": params,
            }
        ),
    )
    train_df = pd.DataFrame(
        {
            "model_name": ["TFT", "TFT", "TFT"],
            "fold_idx": [0, 0, 0],
            "panel_split": ["backcast_train", "backcast_train", "backcast_train"],
            "unique_id": ["target", "target", "target"],
            "cutoff": pd.to_datetime(["2020-01-08", "2020-01-08", "2020-01-08"]),
            "train_end_ds": pd.to_datetime(["2020-01-08", "2020-01-08", "2020-01-08"]),
            "ds": pd.to_datetime(["2020-01-15", "2020-01-22", "2020-01-29"]),
            "horizon_step": [1, 2, 3],
            "y_hat_base": [1.5, 2.5, 3.5],
            "y": [2.0, 3.0, 4.0],
            "residual_target": [0.5, 0.5, 0.5],
        }
    )
    plugin.fit(
        train_df,
        ResidualContext(
            job_name="TFT",
            model_name="TFT",
            output_dir=tmp_path / "checkpoint",
            config={},
        ),
    )
    feature_frame = plugin._feature_frame(train_df)
    assert "fold_idx" not in feature_frame.columns
    assert list(feature_frame.columns) == [
        "horizon_step",
        "y_hat_base",
    ]
    predicted = plugin.predict(train_df.drop(columns=["residual_target"]))
    assert "residual_hat" in predicted.columns
    assert len(predicted) == 3
    assert plugin.metadata()["plugin"] == model_name
    assert (tmp_path / "checkpoint" / "model.ubj").exists()


def test_runtime_skips_residual_artifacts_for_baseline_models(tmp_path: Path):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2},
    }
    payload["jobs"] = [{"model": "Naive", "params": {}}]
    data = "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n"
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    from runtime_support.runner import main as runtime_main

    output_root = tmp_path / "run_baseline"
    code = runtime_main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "Naive",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    assert (output_root / "cv" / "Naive_forecasts.csv").exists()
    assert not (output_root / "holdout").exists()
    assert not (output_root / "residual" / "Naive").exists()

def test_repo_root_contract_keeps_only_naive_baseline_job(tmp_path: Path):
    config_path = _write_repo_root_contract_config(tmp_path)
    loaded = load_app_config(tmp_path, config_path=config_path)
    params_by_model = {job.model: job.params for job in loaded.config.jobs}
    assert params_by_model["Naive"] == {}
    assert "SeasonalNaive" not in params_by_model
    assert "HistoricAverage" not in params_by_model


def test_repo_root_contract_sets_explicit_itransformer_fairness_target(tmp_path: Path):
    config_path = _write_repo_root_contract_config(tmp_path)
    loaded = load_app_config(tmp_path, config_path=config_path)
    params_by_model = {job.model: job.params for job in loaded.config.jobs}
    assert params_by_model["iTransformer"] == {
        "hidden_size": 128,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 256,
    }


def test_repo_root_contract_conservative_fairness_matrix(tmp_path: Path):
    config_path = _write_repo_root_contract_config(tmp_path)
    loaded = load_app_config(tmp_path, config_path=config_path)
    params_by_model = {job.model: job.params for job in loaded.config.jobs}

    for model_name in (
        "TFT",
        "VanillaTransformer",
        "Informer",
        "Autoformer",
        "PatchTST",
        "iTransformer",
    ):
        assert params_by_model[model_name]["hidden_size"] == 128

    assert params_by_model["LSTM"]["encoder_hidden_size"] == 128
    assert params_by_model["LSTM"]["decoder_hidden_size"] == 128

    assert params_by_model["PatchTST"]["n_heads"] == 16
    assert params_by_model["PatchTST"]["patch_len"] == 16
    assert params_by_model["NHITS"] == {
        "mlp_units": [[64, 64], [64, 64], [64, 64]],
        "n_pool_kernel_size": [2, 2, 1],
        "n_freq_downsample": [4, 2, 1],
        "dropout_prob_theta": 0.0,
        "activation": "ReLU",
    }


def test_model_builder_does_not_alias_api_distinct_keys(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [
        {"model": "TFT", "params": {"hidden_size": 32, "encoder_hidden_size": 999}},
        {
            "model": "LSTM",
            "params": {
                "hidden_size": 999,
                "encoder_hidden_size": 32,
                "decoder_hidden_size": 48,
            },
        },
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n", encoding="utf-8"
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    tft = build_model(loaded.config, loaded.config.jobs[0])
    lstm = build_model(loaded.config, loaded.config.jobs[1])

    assert tft.hparams.hidden_size == 32
    assert getattr(tft.hparams, "encoder_hidden_size", 999) == 999

    assert lstm.hparams.encoder_hidden_size == 32
    assert lstm.hparams.decoder_hidden_size == 48
    assert getattr(lstm.hparams, "hidden_size", 999) == 999


def test_readme_documents_conservative_fairness_policy():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "baseline (`Naive`, `SeasonalNaive`, `HistoricAverage`)은 fairness normalization 대상이 아닙니다."
        in readme
    )
    assert (
        "`training:`에 있는 공통 key를 `jobs[*].params`에 다시 쓰면 안 됩니다."
        in readme
    )
    assert (
        "API가 다른 key들 사이에는 aliasing이나 canonicalization을 하지 않습니다."
        in readme
    )
    assert "PatchTST.n_heads" in readme
    assert "FEDformer.modes" in readme


def test_load_app_config_marks_auto_requested_and_validated_modes(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": True, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(tmp_path)

    loaded = load_app_config(tmp_path, config_path=config_path)

    job = loaded.config.jobs[0]
    assert job.requested_mode == "learned_auto_requested"
    assert job.validated_mode == "learned_auto"
    assert list(job.selected_search_params) == ["hidden_size", "dropout", "n_head"]
    assert loaded.config.residual.requested_mode == "residual_auto_requested"
    assert loaded.config.residual.validated_mode == "residual_auto"
    assert loaded.config.training_search.requested_mode == "training_fixed"
    assert loaded.config.training_search.validated_mode == "training_fixed"
    assert list(loaded.config.training_search.selected_search_params) == []
    assert loaded.normalized_payload["search_space_path"] == str(
        (tmp_path / "yaml/HPO/search_space.yaml").resolve()
    )
    assert loaded.normalized_payload["search_space_sha256"]


@pytest.mark.parametrize(
    ("model_name", "selectors"),
    (
        ("ARIMA", ["order", "include_mean", "include_drift"]),
        ("ES", ["trend", "damped_trend"]),
        ("xgboost", ["lags", "n_estimators", "max_depth"]),
        (
            "lightgbm",
            [
                "lags",
                "n_estimators",
                "max_depth",
                "num_leaves",
                "min_child_samples",
                "feature_fraction",
            ],
        ),
    ),
)
def test_load_app_config_direct_auto_models_keep_training_search_fixed(
    tmp_path: Path, model_name: str, selectors: list[str]
):
    payload = _payload()
    payload["jobs"] = [{"model": model_name, "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {model_name: selectors},
            "training": ["input_size", "model_step_size"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    loaded = load_app_config(tmp_path, config_path=config_path)

    job = loaded.config.jobs[0]
    assert job.requested_mode == "learned_auto_requested"
    assert job.validated_mode == "learned_auto"
    assert list(job.selected_search_params) == selectors
    assert loaded.config.training_search.requested_mode == "training_fixed"
    assert loaded.config.training_search.validated_mode == "training_fixed"
    assert list(loaded.config.training_search.selected_search_params) == []


def test_load_app_config_normalizes_bs_preforcast_selection(tmp_path: Path):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a,bs_b\n"
        "2020-01-01,1,2,10,20\n"
        "2020-01-08,2,3,11,21\n"
        "2020-01-15,3,4,12,22\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[{"model": "TFT", "params": {"hidden_size": 32}}],
                target_columns=("bs_a", "bs_b"),
                multivariable=True,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )

    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    assert loaded.config.stage_plugin_config.enabled is True
    assert loaded.config.stage_plugin_config.task.multivariable is True
    assert loaded.config.stage_plugin_config.target_columns == ("bs_a", "bs_b")
    assert loaded.config.stage_plugin_config.config_path == "yaml/plugins/bs_preforcast.yaml"
    assert loaded.normalized_payload["bs_preforcast"]["config_path"] == "yaml/plugins/bs_preforcast.yaml"


def test_load_app_config_defaults_bs_preforcast_config_path_when_enabled(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        """dt,target,hist_a,bs_a,bs_b
2020-01-01,1,2,10,20
2020-01-08,2,3,11,21
2020-01-15,3,4,12,22
""",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}],
                target_columns=("bs_a", "bs_b"),
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(tmp_path, {"models": {}, "training": [], "residual": {"xgboost": ["n_estimators"]}, "bs_preforcast_models": {}, "bs_preforcast_training": []})
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))
    assert loaded.config.stage_plugin_config.config_path == "yaml/plugins/bs_preforcast.yaml"
    assert loaded.normalized_payload["bs_preforcast"]["config_path"] == "yaml/plugins/bs_preforcast.yaml"
    assert loaded.stage_plugin_loaded is not None
    assert loaded.stage_plugin_loaded.source_path == (tmp_path / "yaml/plugins/bs_preforcast.yaml").resolve()
    assert loaded.stage_plugin_loaded.config.jobs[0].model == "DummyUnivariate"

def test_load_app_config_accepts_independent_bs_preforcast_config_path(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast(config_path="custom-bs.yaml")
    (tmp_path / "data.csv").write_text(
        """dt,target,hist_a,bs_a,bs_b
2020-01-01,1,2,10,20
2020-01-08,2,3,11,21
2020-01-15,3,4,12,22
""",
        encoding="utf-8",
    )
    custom_path = tmp_path / "custom-bs.yaml"
    custom_path.write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[{"model": "DummyMultivariate", "params": {"start_padding_enabled": True}}],
                target_columns=("bs_a", "bs_b"),
                multivariable=True,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(tmp_path, {"models": {}, "training": [], "residual": {"xgboost": ["n_estimators"]}, "bs_preforcast_models": {}, "bs_preforcast_training": []})
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))
    assert loaded.config.stage_plugin_config.config_path == "custom-bs.yaml"
    assert loaded.stage_plugin_loaded is not None
    assert loaded.stage_plugin_loaded.source_path == custom_path.resolve()
    assert loaded.stage_plugin_loaded.config.jobs[0].model == "DummyMultivariate"
    assert loaded.stage_plugin_loaded.config.dataset.hist_exog_cols == ("bs_b",)

def test_load_app_config_rejects_legacy_bs_preforcast_routing_keys(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = {
        "enabled": True,
        "config_path": "yaml/plugins/bs_preforcast.yaml",
        "routing": {
            "univariable_config": "yaml/legacy-univariable.yaml",
            "multivariable_config": "yaml/legacy-multivariable.yaml",
        },
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unsupported key\(s\): routing"):
        load_app_config(
            tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
        )


def test_load_app_config_rejects_inline_bs_preforcast_owner_fields(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = {
        "enabled": True,
        "config_path": "yaml/plugins/bs_preforcast.yaml",
        "using_futr_exog": True,
        "target_columns": ["bs_a"],
        "task": {"multivariable": False},
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"unsupported key\(s\): target_columns, task, using_futr_exog",
    ):
        load_app_config(
            tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
        )

def test_load_app_config_rejects_routed_bs_preforcast_without_target_columns(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(target_columns=()),
                "jobs": [
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="bs_preforcast.target_columns must be non-empty in routed"
    ):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))

def test_load_app_config_rejects_routed_bs_preforcast_without_owner_block(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            {
                "jobs": [
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )

    with pytest.raises(
        ValueError,
        match="must define a top-level bs_preforcast block",
    ):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_load_app_config_bs_preforcast_loads_stage1_route_and_authoritative_targets(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        """dt,target,hist_a,bs_a,bs_b
2020-01-01,1,2,10,20
2020-01-08,2,3,11,21
2020-01-15,3,4,12,22
""",
        encoding="utf-8",
    )
    stage_path = tmp_path / "stage_uni.yaml"
    stage_path.write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}],
                target_columns=("bs_a", "bs_b"),
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload["bs_preforcast"]["config_path"] = str(stage_path)
    _write_search_space(tmp_path, {"models": {}, "training": [], "residual": {"xgboost": ["n_estimators"]}, "bs_preforcast_models": {}, "bs_preforcast_training": []})
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))
    assert loaded.stage_plugin_loaded is not None
    assert loaded.stage_plugin_loaded.source_path == stage_path.resolve()
    assert loaded.stage_plugin_loaded.normalized_payload["bs_preforcast"]["target_columns"] == ["bs_a", "bs_b"]
    assert loaded.normalized_payload["bs_preforcast"]["stage1"]["source_path"] == str(stage_path.resolve())
    assert loaded.normalized_payload["bs_preforcast"]["stage1"]["config_resolved_sha256"]

def test_load_app_config_rejects_bs_preforcast_missing_config_file(tmp_path: Path):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast(config_path='missing-bs-preforcast.yaml')
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n2020-01-01,1,2,10\n2020-01-08,2,3,11\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="selected route does not exist"):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_load_app_config_bs_preforcast_stage1_empty_params_fail_before_search_space(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        """dt,target,hist_a,bs_a
2020-01-01,1,2,10
2020-01-08,2,3,11
2020-01-15,3,4,12
""",
        encoding="utf-8",
    )
    stage_path = tmp_path / "stage_auto_missing.yaml"
    stage_path.write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(jobs=[{"model": "TFT", "params": {}}], target_columns=("bs_a", "bs_b")),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload["bs_preforcast"]["config_path"] = str(stage_path)
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    with pytest.raises(ValueError, match="fixed-param job"):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))

def test_load_app_config_loads_bs_preforcast_stage1_with_dedicated_search_space(
    tmp_path: Path,
):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["jobs"] = [{"model": "TFT", "params": {"hidden_size": 32}}]
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        """dt,target,hist_a
2020-01-01,1,2
2020-01-08,2,3
2020-01-15,3,4
""",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(_thin_plugin_stage_payload(jobs=[{"model": "TFT", "params": {"hidden_size": 16, "n_head": 1, "attn_dropout": 0.0, "dropout": 0.0}}], target_columns=("bs_a", "bs_b")), sort_keys=False),
        encoding="utf-8",
    )
    _write_search_space(tmp_path, {"models": {}, "training": [], "residual": {}, "bs_preforcast_models": {"TFT": ["hidden_size"]}, "bs_preforcast_training": ["input_size"]})
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))
    assert loaded.stage_plugin_loaded is not None
    assert loaded.stage_plugin_loaded.config.jobs[0].model == "TFT"
    assert loaded.stage_plugin_loaded.config.jobs[0].validated_mode == "learned_fixed"
    assert loaded.stage_plugin_loaded.normalized_payload["bs_preforcast"]["target_columns"] == ["bs_a", "bs_b"]

def test_load_app_config_rejects_invalid_linked_bs_preforcast_owner_keys(tmp_path: Path):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n2020-01-15,3,4\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            {
                "bs_preforcast": {"enabled": True, "target_columns": ["bad"]},
                "jobs": [
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported key\\(s\\): enabled"):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_load_app_config_rejects_list_route_for_bs_preforcast_config_path(tmp_path: Path):
    payload = _payload()
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    payload["bs_preforcast"]["config_path"] = "yaml/jobs/bs_preforcast_jobs_uni.yaml"
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n2020-01-15,3,4\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/jobs/bs_preforcast_jobs_uni.yaml").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/jobs/bs_preforcast_jobs_uni.yaml").write_text(
        yaml.safe_dump(["uni/arima.yaml"], sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config_path must resolve to a mapping"):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_load_app_config_accepts_training_search_space_section(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size", "dropout", "n_head"]},
            "training": ["input_size", "model_step_size"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.config.training_search.requested_mode == "training_auto_requested"
    assert loaded.config.training_search.validated_mode == "training_auto"
    assert list(loaded.config.training_search.selected_search_params) == [
        "input_size",
        "model_step_size",
    ]


def test_load_app_config_rejects_training_search_space_learning_rate(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["learning_rate"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match=r"search_space\.training\.global contains unknown"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_training_num_lr_decays(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["training"]["num_lr_decays"] = -1
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match="legacy num_lr_decays training key has been removed"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_unknown_training_search_space_param(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["not_a_real_training_key"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match=r"search_space\.training\.global contains unknown"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_legacy_step_size_search_space_param(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["step_size"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match=r"search_space\.training\.global contains unknown"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_removed_early_stop_search_space_param(
    tmp_path: Path,
):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["early_stop_patience_steps"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match=r"search_space\.training\.global contains unknown"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_accepts_batch_training_search_space_param(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["batch_size"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.config.training_search.validated_mode == "training_auto"
    assert "batch_size" in loaded.config.training_search.selected_search_params


def test_repo_optuna_fixture_excludes_batch_window_training_selectors():
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / "tests" / "fixtures" / "optuna_learned_auto.yaml",
    )

    assert set(EXCLUDED_REPO_TRAINING_SELECTORS).isdisjoint(
        loaded.config.training_search.selected_search_params
    )


def test_suggest_training_params_supports_batch_size_selector():
    class _Trial:
        def suggest_categorical(self, _name, options):
            return options[0]

    suggested = suggest_training_params(("batch_size",), _Trial())
    contextual = suggest_training_params(("batch_size",), _Trial(), model_name="TFT")

    assert suggested["batch_size"] == 16
    assert contextual["batch_size"] == 16
    assert training_range_source_for_model(None) == "global_fallback"
    assert training_range_source_for_model("PatchTST") == "model_override:PatchTST"
    assert training_range_source_for_model("unknown-model") == "global_fallback"


def test_load_app_config_rejects_missing_model_search_space_entry(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"iTransformer": ["hidden_size"]},
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match="requires search_space.models.TFT"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_unknown_search_space_param(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["encoder_layers"]},
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match="unknown parameter"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_stale_unused_allowlisted_selector_entry(
    tmp_path: Path,
):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"], "PatchTST": ["not_a_real_key"]},
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    with pytest.raises(ValueError, match="PatchTST contains unknown parameter"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_uses_repo_root_search_space_not_config_parent(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_dir = tmp_path / "config_dir"
    repo_root.mkdir()
    config_dir.mkdir()
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (config_dir / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    payload["dataset"]["path"] = str((config_dir / "data.csv").resolve())
    config_path = _write_config(config_dir, payload, ".yaml")
    _write_search_space(repo_root)
    _write_search_space(
        config_dir,
        {"models": {"TFT": ["hidden_size"]}, "residual": {"xgboost": ["max_depth"]}},
    )

    loaded = load_app_config(repo_root, config_path=config_path)

    assert loaded.search_space_path == (repo_root / "yaml/HPO/search_space.yaml").resolve()
    assert list(loaded.config.jobs[0].selected_search_params) == [
        "hidden_size",
        "dropout",
        "n_head",
    ]


def test_load_app_config_resolves_relative_dataset_path_from_repo_root(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "yaml" / "blackswan"
    data_dir = repo_root / "data"
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (data_dir / "df.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    payload = _payload()
    payload["dataset"]["path"] = "data/df.csv"
    config_path = _write_config(config_dir, payload, ".yaml")

    loaded = load_app_config(repo_root, config_path=config_path)

    assert loaded.config.dataset.path == (repo_root / "data" / "df.csv").resolve()


def test_runtime_auto_mode_records_selector_provenance_and_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
        "2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(tmp_path)

    import runtime_support.runner as runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "1")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_auto"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    capability = json.loads(
        (output_root / "config" / "capability_report.json").read_text()
    )
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())

    assert resolved["search_space_path"] == str(
        (tmp_path / "yaml/HPO/search_space.yaml").resolve()
    )
    assert resolved["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert resolved["jobs"][0]["validated_mode"] == "learned_auto"
    assert resolved["training_search"]["requested_mode"] == "training_fixed"
    assert resolved["training_search"]["validated_mode"] == "training_fixed"
    assert capability["TFT"]["requested_mode"] == "learned_auto_requested"
    assert capability["TFT"]["validated_mode"] == "learned_auto"
    assert capability["training_search"]["validated_mode"] == "training_fixed"
    assert manifest["search_space_path"] == str(
        (tmp_path / "yaml/HPO/search_space.yaml").resolve()
    )
    assert manifest["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert manifest["jobs"][0]["validated_mode"] == "learned_auto"
    assert manifest["training_search"]["validated_mode"] == "training_fixed"
    assert manifest["jobs"][0]["model_best_params_path"]
    assert manifest["jobs"][0]["model_optuna_study_summary_path"]
    assert (output_root / "models" / "TFT" / "best_params.json").exists()
    assert (output_root / "models" / "TFT" / "optuna_study_summary.json").exists()


def test_runtime_validate_only_records_bs_preforcast_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["jobs"] = [{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    stage_payload = _thin_plugin_stage_payload(jobs=[{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}], target_columns=("bs_a", "bs_b"))
    (tmp_path / "data.csv").write_text(
        """dt,target,hist_a,bs_a,bs_b
2020-01-01,1,2,10,20
2020-01-08,2,3,11,21
2020-01-15,3,4,12,22
""",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(yaml.safe_dump(stage_payload, sort_keys=False), encoding="utf-8")
    _write_search_space(tmp_path, {"models": {}, "training": [], "residual": {"xgboost": ["n_estimators"]}, "bs_preforcast_models": {}, "bs_preforcast_training": []})
    config_path = _write_config(tmp_path, payload, ".yaml")
    import runtime_support.runner as runtime
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output_root = tmp_path / "run_bs_preforcast_validate"
    code = runtime.main(["--config", str(config_path), "--jobs", "DummyUnivariate", "--output-root", str(output_root), "--validate-only"])
    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    capability = json.loads((output_root / "config" / "capability_report.json").read_text())
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert resolved["bs_preforcast"]["enabled"] is True
    assert resolved["bs_preforcast"]["target_columns"] == ["bs_a", "bs_b"]
    assert resolved["bs_preforcast"]["config_path"] == "yaml/plugins/bs_preforcast.yaml"
    assert capability["bs_preforcast"]["enabled"] is True
    assert capability["bs_preforcast"]["multivariable"] is False
    assert capability["bs_preforcast"]["selected_config_path"] == str((tmp_path / "yaml/plugins/bs_preforcast.yaml").resolve())
    assert manifest["bs_preforcast"]["target_columns"] == ["bs_a", "bs_b"]
    assert manifest["bs_preforcast"]["config_path"] == "yaml/plugins/bs_preforcast.yaml"
    assert (output_root / "bs_preforcast" / "config" / "config.resolved.json").exists()
    assert (output_root / "bs_preforcast" / "config" / "capability_report.json").exists()
    assert (output_root / "bs_preforcast" / "manifest" / "run_manifest.json").exists()
    assert (output_root / "bs_preforcast" / "summary" / "dashboard.md").exists()
    assert (output_root / "bs_preforcast" / "artifacts" / "bs_preforcast_forecasts.csv").exists()
    assert resolved["bs_preforcast"]["job_injection_results"] == [{"model": "DummyUnivariate", "injection_mode": "futr_exog", "supports_futr_exog": True}]
    assert manifest["bs_preforcast"]["job_injection_results"] == [{"model": "DummyUnivariate", "injection_mode": "futr_exog", "supports_futr_exog": True}]

@pytest.mark.parametrize(
    ("model_name", "params"),
    (
        ("ARIMA", {"order": [1, 1, 0], "include_mean": True, "include_drift": False}),
        ("ES", {"trend": "add", "damped_trend": False}),
        ("xgboost", {"lags": [1, 2, 3], "n_estimators": 16, "max_depth": 2}),
        (
            "lightgbm",
            {
                "lags": [1, 2, 3],
                "n_estimators": 32,
                "max_depth": 4,
                "num_leaves": 15,
                "min_child_samples": 10,
            },
        ),
    ),
)
def test_runtime_validate_only_accepts_top_level_direct_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    params: dict[str, Any],
):
    payload = _payload()
    payload["jobs"] = [{"model": model_name, "params": params}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    import runtime_support.runner as runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output_root = tmp_path / f"run_validate_{model_name}"
    assert (
        runtime.main(
            [
                "--config",
                str(config_path),
                "--jobs",
                model_name,
                "--output-root",
                str(output_root),
                "--validate-only",
            ]
        )
        == 0
    )
    capability = json.loads(
        (output_root / "config" / "capability_report.json").read_text()
    )
    assert capability[model_name]["validation_error"] is None
    assert capability[model_name]["supports_auto"] is True


@pytest.mark.parametrize(
    ("model_name", "params"),
    (
        ("ARIMA", {"order": [1, 1, 0], "include_mean": True, "include_drift": False}),
        ("xgboost", {"lags": [1, 2, 3], "n_estimators": 16, "max_depth": 2}),
    ),
)
def test_runtime_executes_top_level_direct_models_without_loss_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    params: dict[str, Any],
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 2, "gap": 0})
    payload["training"].update({"input_size": 3, "max_steps": 1, "val_size": 1})
    payload["jobs"] = [{"model": model_name, "params": params}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n"
        "2020-01-01,1,2\n"
        "2020-01-08,2,3\n"
        "2020-01-15,3,4\n"
        "2020-01-22,4,5\n"
        "2020-01-29,5,6\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    import runtime_support.runner as runtime

    def _fake_predict(_stage_loaded, _job, *, target_column, train_df, future_df):
        return [float(train_df[target_column].iloc[-1])] * len(future_df)

    monkeypatch.setattr(runtime, "predict_univariate_direct", _fake_predict)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output_root = tmp_path / f"run_direct_{model_name}"
    assert (
        runtime.main(
            [
                "--config",
                str(config_path),
                "--jobs",
                model_name,
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    assert (output_root / "cv" / f"{model_name}_forecasts.csv").exists()
    assert (output_root / "cv" / f"{model_name}_metrics_by_cutoff.csv").exists()
    assert not any((output_root / "models" / model_name).rglob("loss_curve.png"))


def test_runtime_validate_only_rejects_residual_enabled_top_level_direct_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["jobs"] = [{"model": "ARIMA", "params": {"order": [1, 1, 0]}}]
    payload["residual"] = {
        "enabled": True,
        "model": "xgboost",
        "params": {"n_estimators": 8, "max_depth": 2},
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    import runtime_support.runner as runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output_root = tmp_path / "run_validate_direct_residual"
    with pytest.raises(
        ValueError,
        match="Top-level direct models do not yet support residual-enabled runs",
    ):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--jobs",
                "ARIMA",
                "--output-root",
                str(output_root),
                "--validate-only",
            ]
        )


def test_runtime_validate_only_records_mixed_job_injection_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}},
        {"model": "Naive", "params": {}},
    ]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a,bs_b\n"
        "2020-01-01,1,2,10,20\n"
        "2020-01-08,2,3,11,21\n"
        "2020-01-15,3,4,12,22\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}],
                target_columns=("bs_a", "bs_b"),
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    config_path = _write_config(tmp_path, payload, ".yaml")

    import runtime_support.runner as runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    output_root = tmp_path / "run_bs_preforcast_validate_mixed"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    assert resolved["bs_preforcast"]["job_injection_results"] == [
        {"model": "DummyUnivariate", "injection_mode": "futr_exog", "supports_futr_exog": True},
        {"model": "Naive", "injection_mode": "lag_derived", "supports_futr_exog": False},
    ]


def test_runtime_validate_only_bs_preforcast_fails_for_legacy_using_futr_exog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["jobs"] = [{"model": "Naive", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}],
                legacy_using_futr_exog=True,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    config_path = _write_config(tmp_path, payload, ".yaml")

    import runtime_support.runner as runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with pytest.raises(ValueError, match=r"unsupported key\(s\): using_futr_exog"):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "run_bs_preforcast_validate_lag"),
                "--validate-only",
            ]
        )


def test_prepare_bs_preforcast_fold_inputs_routes_same_name_target_to_futr_exog(
    tmp_path: Path,
):
    payload = _payload()
    payload["training"].update({"input_size": 2, "max_steps": 1, "val_size": 1})
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ]
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    job = loaded.config.jobs[0]
    source_df = pd.read_csv(tmp_path / "data.csv")
    train_df = source_df.iloc[:2].reset_index(drop=True)
    future_df = source_df.iloc[2:].reset_index(drop=True)

    effective_loaded, transformed_train, transformed_future, injection_mode = (
        prepare_bs_preforcast_fold_inputs(loaded, job, train_df, future_df)
    )

    assert injection_mode == "futr_exog"
    assert "bs_a" in effective_loaded.config.dataset.futr_exog_cols
    assert "bs_a" not in effective_loaded.config.dataset.hist_exog_cols
    assert not any(
        column.startswith("bs_preforcast_futr" "__")
        for column in transformed_train.columns
    )
    assert transformed_train["bs_a"].tolist() == [10, 11]
    assert transformed_future["bs_a"].tolist() == pytest.approx([11.0], abs=0.05)


def test_prepare_bs_preforcast_fold_inputs_moves_overlap_target_out_of_hist_for_futr_job(
    tmp_path: Path,
):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = ["hist_a", "bs_a"]
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["training"].update({"input_size": 2, "max_steps": 1, "val_size": 1})
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,futr_a,bs_a\n"
        "2020-01-01,1,2,7,10\n"
        "2020-01-08,2,3,8,11\n"
        "2020-01-15,3,4,9,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ]
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    source_df = pd.read_csv(tmp_path / "data.csv")

    effective_loaded, transformed_train, transformed_future, injection_mode = (
        prepare_bs_preforcast_fold_inputs(
            loaded,
            loaded.config.jobs[0],
            source_df.iloc[:2].reset_index(drop=True),
            source_df.iloc[2:].reset_index(drop=True),
        )
    )

    assert injection_mode == "futr_exog"
    assert effective_loaded.config.dataset.hist_exog_cols == ("hist_a",)
    assert effective_loaded.config.dataset.futr_exog_cols == ("futr_a", "bs_a")
    assert not any(
        column.startswith("bs_preforcast_futr" "__")
        for column in transformed_train.columns
    )
    assert transformed_train["bs_a"].tolist() == [10, 11]
    assert transformed_future["bs_a"].tolist() == pytest.approx([11.0], abs=0.05)


def test_prepare_bs_preforcast_fold_inputs_uses_same_name_lag_path_for_non_futr_job(
    tmp_path: Path,
):
    payload = _payload()
    payload["training"].update({"input_size": 2, "max_steps": 1, "val_size": 1})
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["jobs"] = [{"model": "Naive", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ]
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    source_df = pd.read_csv(tmp_path / "data.csv")

    effective_loaded, transformed_train, transformed_future, injection_mode = (
        prepare_bs_preforcast_fold_inputs(
            loaded,
            loaded.config.jobs[0],
            source_df.iloc[:2].reset_index(drop=True),
            source_df.iloc[2:].reset_index(drop=True),
        )
    )

    assert injection_mode == "lag_derived"
    assert "bs_a" not in effective_loaded.config.dataset.futr_exog_cols
    assert effective_loaded.config.dataset.hist_exog_cols == ("hist_a", "bs_a")
    assert not any(
        column.startswith("bs_preforcast_futr" "__")
        for column in transformed_train.columns
    )
    assert transformed_train["bs_a"].tolist() == pytest.approx([10.0, 11.0], abs=0.05)
    assert transformed_future["bs_a"].tolist() == pytest.approx([11.0], abs=0.05)


def test_prepare_bs_preforcast_fold_inputs_missing_forecasts_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import plugins.bs_preforcast.runtime as bs_runtime

    payload = _payload()
    payload["training"].update({"input_size": 2, "max_steps": 1, "val_size": 1})
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ]
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))
    source_df = pd.read_csv(tmp_path / "data.csv")

    monkeypatch.setattr(
        bs_runtime,
        "compute_bs_preforcast_fold_forecasts",
        lambda *args, **kwargs: {},
    )

    with pytest.raises(
        ValueError,
        match="did not produce forecasts for target column: bs_a",
    ):
        prepare_bs_preforcast_fold_inputs(
            loaded,
            loaded.config.jobs[0],
            source_df.iloc[:2].reset_index(drop=True),
            source_df.iloc[2:].reset_index(drop=True),
        )


def test_prepare_bs_preforcast_fold_inputs_uses_same_name_futr_path_for_timexer_native_future_exog(
    tmp_path: Path,
):
    payload = _payload()
    payload["training"].update({"input_size": 2, "max_steps": 1, "val_size": 1})
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["jobs"] = [
        {
            "model": "TimeXer",
            "params": {
                "hidden_size": 8,
                "n_heads": 1,
                "e_layers": 1,
                "d_ff": 16,
                "patch_len": 2,
                "factor": 1,
                "dropout": 0.0,
                "use_norm": True,
            },
        }
    ]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ]
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))
    source_df = pd.read_csv(tmp_path / "data.csv")

    effective_loaded, transformed_train, transformed_future, injection_mode = (
        prepare_bs_preforcast_fold_inputs(
            loaded,
            loaded.config.jobs[0],
            source_df.iloc[:2].reset_index(drop=True),
            source_df.iloc[2:].reset_index(drop=True),
        )
    )

    assert injection_mode == "futr_exog"
    assert "bs_a" in effective_loaded.config.dataset.futr_exog_cols
    assert "bs_a" not in effective_loaded.config.dataset.hist_exog_cols
    assert not any(
        column.startswith("bs_preforcast_futr" "__")
        for column in transformed_train.columns
    )
    assert transformed_train["bs_a"].tolist() == [10, 11]
    assert transformed_future["bs_a"].tolist() == pytest.approx([11.0], abs=0.05)


def test_prepare_bs_preforcast_fold_inputs_auto_adds_missing_target_to_futr_exog(
    tmp_path: Path,
):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = ["hist_a"]
    payload["dataset"]["futr_exog_cols"] = []
    payload["training"].update({"input_size": 2, "max_steps": 1, "val_size": 1})
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["jobs"] = [
        {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
    ]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ]
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    source_df = pd.read_csv(tmp_path / "data.csv")

    effective_loaded, transformed_train, transformed_future, injection_mode = (
        prepare_bs_preforcast_fold_inputs(
            loaded,
            loaded.config.jobs[0],
            source_df.iloc[:2].reset_index(drop=True),
            source_df.iloc[2:].reset_index(drop=True),
        )
    )

    assert injection_mode == "futr_exog"
    assert effective_loaded.config.dataset.hist_exog_cols == ("hist_a",)
    assert effective_loaded.config.dataset.futr_exog_cols == ("bs_a",)
    assert not any(
        column.startswith("bs_preforcast_futr" "__")
        for column in transformed_train.columns
    )
    assert transformed_train["bs_a"].tolist() == [10, 11]
    assert transformed_future["bs_a"].tolist() == pytest.approx([11.0], abs=0.05)


def test_bs_preforcast_inline_learned_stage_surfaces_multi_gpu_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import plugins.bs_preforcast.runtime as bs_runtime

    payload = _payload()
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(
                jobs=[
                    {
                        "model": "DummyUnivariate",
                        "params": {"start_padding_enabled": True},
                    }
                ]
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))
    source_df = pd.read_csv(tmp_path / "data.csv")
    run_root = tmp_path / "run_inline_multi_gpu"
    captured: dict[str, Any] = {}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("NEURALFORECAST_ASSIGNED_GPU_IDS", "0,1")
    monkeypatch.setenv("NEURALFORECAST_WORKER_DEVICES", "2")

    class FakeNeuralForecast:
        def __init__(self, *, models, freq):
            self.models = models
            self.freq = freq
            self._fit_unique_id: str | None = None

        def fit(self, fit_df, static_df=None, val_size=0):
            captured["fit_cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            captured["fit_val_size"] = val_size
            self._fit_unique_id = str(fit_df["unique_id"].iloc[0])

        def predict(self, **kwargs):
            captured["predict_cuda_visible_devices"] = os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            )
            futr_df = kwargs.get("futr_df")
            if futr_df is not None and self._fit_unique_id is not None:
                horizon = len(futr_df[futr_df["unique_id"] == self._fit_unique_id])
            else:
                horizon = 1
            return pd.DataFrame(
                {
                    "unique_id": [self._fit_unique_id or "bs_a"] * horizon,
                    "DummyUnivariate": [11.0] * horizon,
                }
            )

    monkeypatch.setattr(bs_runtime, "NeuralForecast", FakeNeuralForecast)

    prepare_bs_preforcast_fold_inputs(
        loaded,
        loaded.config.jobs[0],
        source_df.iloc[:2].reset_index(drop=True),
        source_df.iloc[2:].reset_index(drop=True),
        run_root=run_root,
    )

    inline_summary = json.loads(
        (
            run_root
            / "bs_preforcast"
            / "artifacts"
            / "inline_fit_summary.json"
        ).read_text()
    )
    assert inline_summary["devices"] == 2
    assert inline_summary["strategy"] in {"ddp-gloo-auto", "ddp"}
    assert inline_summary["assigned_gpu_ids"] == [0, 1]
    assert captured["fit_cuda_visible_devices"] == "0,1"
    assert captured["predict_cuda_visible_devices"] == "0,1"


def test_bs_preforcast_stage_baseline_job_stays_rejected(tmp_path: Path):
    payload = _payload()
    payload["bs_preforcast"] = _main_bs_preforcast()
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            _thin_plugin_stage_payload(jobs=[{"model": "Naive", "params": {}}]),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    with pytest.raises(ValueError, match="fixed-param job"):
        load_app_config(tmp_path, config_path=_write_config(tmp_path, payload, ".yaml"))


def test_runtime_auto_mode_prefers_yaml_opt_n_trial_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["runtime"]["opt_n_trial"] = 2
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
        "2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(tmp_path)

    import runtime_support.runner as runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "1")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_auto_yaml_trials"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    study_summary = json.loads(
        (output_root / "models" / "TFT" / "optuna_study_summary.json").read_text()
    )
    assert study_summary["trial_count"] == 2
    assert study_summary["objective_metric"] == "mean_fold_mape"


def test_runtime_main_parallelizes_single_auto_job_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["scheduler"]["parallelize_single_job_tuning"] = True
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(tmp_path, {"models": {"TFT": ["hidden_size"]}, "training": [], "residual": {}})

    import runtime_support.runner as runtime

    called: dict[str, Any] = {}

    def _fake_parallel(repo_root, loaded, job, run_root, *, manifest_path):
        called["repo_root"] = repo_root
        called["job_name"] = job.model
        called["run_root"] = run_root
        called["manifest_path"] = manifest_path
        return [{"job_name": job.model, "returncode": 0}]

    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )
    monkeypatch.setattr(runtime, "_run_single_job_with_parallel_tuning", _fake_parallel)
    monkeypatch.setattr(runtime, "_should_build_summary_artifacts", lambda: False)

    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(tmp_path / "run_parallel_auto"),
        ]
    )

    assert code == 0
    assert called["job_name"] == "TFT"


def test_runtime_main_does_not_recurse_parallel_tuning_inside_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["scheduler"]["parallelize_single_job_tuning"] = True
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {"models": {"TFT": ["hidden_size"]}, "training": [], "residual": {}},
    )

    import runtime_support.runner as runtime

    called = {"count": 0}

    def _fake_parallel(*_args, **_kwargs):
        called["count"] += 1
        return []

    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )
    monkeypatch.setattr(runtime, "_run_single_job_with_parallel_tuning", _fake_parallel)
    monkeypatch.setattr(runtime, "_run_single_job", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("NEURALFORECAST_WORKER_DEVICES", "1")

    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(tmp_path / "run_worker_no_recurse"),
        ]
    )

    assert code == 0
    assert called["count"] == 0


def test_runtime_main_reuses_scheduler_worker_root_and_parent_summary_for_auto_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {"models": {"TFT": ["hidden_size"]}, "training": [], "residual": {}},
    )

    import runtime_support.runner as runtime

    worker_root = tmp_path / "matched_run" / "scheduler" / "workers" / "TFT"
    summary_root = tmp_path / "matched_run"
    called: dict[str, Any] = {"summary_roots": [], "prunes": []}

    def _fake_summary(run_root: Path) -> dict[str, str]:
        called["summary_roots"].append(run_root)
        return {"leaderboard": str(run_root / "summary" / "leaderboard.csv")}

    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )
    monkeypatch.setattr(
        runtime,
        "_resolve_run_roots",
        lambda *_args, **_kwargs: {
            "run_root": worker_root,
            "summary_root": summary_root,
        },
    )
    monkeypatch.setattr(runtime, "_should_parallelize_single_job_tuning", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        runtime,
        "_prune_model_run_artifacts",
        lambda run_root, model_name: called["prunes"].append((run_root, model_name)),
    )
    monkeypatch.setattr(
        runtime,
        "_run_single_job",
        lambda loaded, job, run_root, *, manifest_path, main_stage="full": called.update(
            {
                "run_root": run_root,
                "manifest_path": manifest_path,
                "job_name": job.model,
                "main_stage": main_stage,
            }
        ),
    )
    monkeypatch.setattr(runtime, "_build_summary_artifacts", _fake_summary)

    code = runtime.main(["--config", str(config_path), "--jobs", "TFT"])

    assert code == 0
    assert called["run_root"] == worker_root
    assert called["manifest_path"] == worker_root / "manifest" / "run_manifest.json"
    assert called["job_name"] == "TFT"
    assert called["main_stage"] == "full"
    assert called["prunes"] == []
    assert called["summary_roots"] == [summary_root]


def test_runtime_auto_mode_can_prune_from_first_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["runtime"]["opt_n_trial"] = 2
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 11, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n2020-02-26,9\n2020-03-04,10\n2020-03-11,11\n2020-03-18,12\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    import runtime_support.runner as runtime

    should_prune_steps: list[int] = []

    def _fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
    ):
        hidden_size = (params_override or {}).get("hidden_size", 32)
        predicted = 1.0 if hidden_size == 32 else 2.0
        predictions = pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.Series(["2020-01-22"]),
                job.model: [predicted],
            }
        )
        actuals = pd.Series([1.0])
        return (
            predictions,
            actuals,
            pd.Timestamp("2020-01-15"),
            source_df.iloc[train_idx],
            object(),
        )

    def _should_prune_with_trace(self):
        should_prune_steps.append(len(self.user_attrs.get("fold_mape", [])))
        return self.number == 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(runtime.optuna.trial.Trial, "should_prune", _should_prune_with_trace)
    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit_and_predict_fold)
    monkeypatch.setattr(
        runtime,
        "suggest_model_params",
        lambda *_args, **_kwargs: {"hidden_size": 32 + _args[-1].number},
    )
    monkeypatch.setattr(
        runtime,
        "suggest_training_params",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_auto_prune"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )

    summary = json.loads(
        (output_root / "models" / "TFT" / "optuna_study_summary.json").read_text()
    )
    assert code == 0
    assert summary["trial_count"] >= 2
    assert summary["state_counts"]["complete"] == 1
    assert summary["state_counts"]["pruned"] == 1
    assert summary["objective_metric"] == "mean_fold_mape"
    assert should_prune_steps[0] == 1
    assert min(should_prune_steps) == 1


def test_residual_auto_mode_can_prune_from_first_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    payload = _payload()
    payload["residual"] = {"enabled": True, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n2020-01-15,3,4\n2020-01-22,4,5\n",
        encoding="utf-8",
    )
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )
    job = loaded.config.jobs[0]

    class _DummyResidualPlugin:
        def fit(self, *_args, **_kwargs):
            return None

        def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"residual_hat": [0.0] * len(panel)})

    class _FakeTrial:
        def __init__(self):
            self.user_attrs: dict[str, Any] = {}
            self.reported_steps: list[int] = []
            self.prune_checks: list[int] = []

        def set_user_attr(self, key: str, value: Any) -> None:
            self.user_attrs[key] = value

        def report(self, value: float, step: int) -> None:
            self.reported_steps.append(step)

        def should_prune(self) -> bool:
            self.prune_checks.append(len(self.user_attrs.get("fold_mape", [])))
            return True

    monkeypatch.setattr(
        runtime, "build_residual_plugin", lambda *_args, **_kwargs: _DummyResidualPlugin()
    )

    fold_payloads = []
    for fold_idx in range(11):
        panel = pd.DataFrame(
            {
                "y": [1.0],
                "y_hat_base": [1.0],
                "cutoff": [pd.Timestamp("2020-01-01")],
                "unique_id": ["target"],
                "panel_split": ["eval"],
                "fold_idx": [fold_idx],
                "residual_target": [0.0],
            }
        )
        fold_payloads.append(
            {
                "fold_idx": fold_idx,
                "backcast_panel": panel.copy(),
                "eval_panel": panel.copy(),
                "trial_dir": tmp_path / f"trial_{fold_idx}",
            }
        )

    trial = _FakeTrial()

    with pytest.raises(optuna.TrialPruned):
        runtime._score_residual_params(
            loaded,
            job,
            {},
            fold_payloads,
            trial=trial,
        )

    assert trial.reported_steps == [0]
    assert trial.prune_checks == [1]


def test_runtime_auto_mode_catches_recoverable_trial_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["runtime"]["opt_n_trial"] = 2
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    import runtime_support.runner as runtime

    def _fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
    ):
        hidden_size = (params_override or job.params).get("hidden_size", 32)
        if hidden_size == 32:
            raise ValueError("boom")
        predictions = pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.Series(["2020-01-22"]),
                job.model: [1.0],
            }
        )
        actuals = pd.Series([1.0])
        return (
            predictions,
            actuals,
            pd.Timestamp("2020-01-15"),
            source_df.iloc[train_idx],
            object(),
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit_and_predict_fold)
    monkeypatch.setattr(
        runtime,
        "suggest_model_params",
        lambda *_args, **_kwargs: {"hidden_size": 32 + _args[-1].number},
    )
    monkeypatch.setattr(
        runtime,
        "suggest_training_params",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_auto_failure_tolerant"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )

    summary = json.loads(
        (output_root / "models" / "TFT" / "optuna_study_summary.json").read_text()
    )
    assert code == 0
    assert summary["trial_count"] == 2
    assert summary["state_counts"]["complete"] == 1
    assert summary["state_counts"]["fail"] == 1
    assert summary["objective_metric"] == "mean_fold_mape"


def test_runtime_auto_mode_resumes_persistent_study_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["runtime"]["opt_n_trial"] = 2
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    import runtime_support.runner as runtime

    def _fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
    ):
        hidden_size = (params_override or {}).get("hidden_size", 32)
        predicted = 1.0 if hidden_size == 32 else 1.5
        predictions = pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.Series(["2020-01-22"]),
                job.model: [predicted],
            }
        )
        actuals = pd.Series([1.0])
        return (
            predictions,
            actuals,
            pd.Timestamp("2020-01-15"),
            source_df.iloc[train_idx],
            object(),
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit_and_predict_fold)
    monkeypatch.setattr(
        runtime,
        "suggest_model_params",
        lambda *_args, **_kwargs: {"hidden_size": 32 + _args[-1].number},
    )
    monkeypatch.setattr(
        runtime,
        "suggest_training_params",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_auto_resume"
    first_code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )
    first_summary = json.loads(
        (output_root / "models" / "TFT" / "optuna_study_summary.json").read_text()
    )

    second_code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )
    second_summary = json.loads(
        (output_root / "models" / "TFT" / "optuna_study_summary.json").read_text()
    )

    assert first_code == 0
    assert second_code == 0
    assert first_summary["requested_trial_count"] == 2
    assert first_summary["existing_finished_trial_count_before_optimize"] == 0
    assert first_summary["remaining_trial_count"] == 2
    assert second_summary["trial_count"] == 2
    assert second_summary["existing_finished_trial_count_before_optimize"] == 2
    assert second_summary["remaining_trial_count"] == 0
    assert second_summary["storage_backend"] == "journal"
    assert Path(second_summary["storage_path"]).exists()


def test_runtime_auto_mode_records_training_selector_provenance_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["batch_size"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    import runtime_support.runner as runtime

    calls: list[dict[str, Any]] = []

    def _fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
    ):
        calls.append(training_override or {})
        ds = pd.Series(["2020-01-22"])
        predictions = pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": ds,
                job.model: [1.0],
            }
        )
        actuals = pd.Series([1.0])
        return (
            predictions,
            actuals,
            pd.Timestamp("2020-01-15"),
            source_df.iloc[train_idx],
            object(),
        )

    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "1")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit_and_predict_fold)
    monkeypatch.setattr(
        runtime,
        "suggest_model_params",
        lambda *_args, **_kwargs: {"hidden_size": 32},
    )
    monkeypatch.setattr(
        runtime,
        "suggest_training_params",
        lambda *_args, **_kwargs: {"batch_size": 123},
    )
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_auto_training"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )

    manifest = json.loads(
        (output_root / "manifest" / "run_manifest.json").read_text(encoding="utf-8")
    )
    capability = json.loads(
        (output_root / "config" / "capability_report.json").read_text(encoding="utf-8")
    )
    training_summary = json.loads(
        (
            output_root
            / "models"
            / "TFT"
            / "optuna_study_summary.json"
        ).read_text(encoding="utf-8")
    )
    training_best = json.loads(
        (output_root / "models" / "TFT" / "training_best_params.json").read_text(
            encoding="utf-8"
        )
    )

    assert code == 0
    assert calls[-1]["batch_size"] == 123
    assert manifest["training_search"]["selected_search_params"] == ["batch_size"]
    assert manifest["training_search"]["training_range_source"] == "model_override:TFT"
    assert manifest["jobs"][0]["training_best_params_path"]
    assert manifest["jobs"][0]["training_optuna_study_summary_path"]
    assert capability["training_search"]["validated_mode"] == "training_auto"
    assert training_summary["training_range_source"] == "model_override:TFT"
    assert training_best == {"batch_size": 123}


def test_update_manifest_artifacts_tracks_training_range_source_per_job(
    tmp_path: Path,
):
    import runtime_support.runner as runtime

    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {"model": "PatchTST"},
                    {"model": "TFT"},
                ],
                "training_search": {},
            }
        ),
        encoding="utf-8",
    )

    runtime._update_manifest_artifacts(
        manifest_path,
        job_name="PatchTST",
        training_range_source="model_override:PatchTST",
    )
    runtime._update_manifest_artifacts(
        manifest_path,
        job_name="TFT",
        training_range_source="model_override:TFT",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = {job["model"]: job for job in manifest["jobs"]}

    assert "training_range_source" not in manifest["training_search"]
    assert manifest["training_search"]["training_range_source_by_job"] == {
        "PatchTST": "model_override:PatchTST",
        "TFT": "model_override:TFT",
    }
    assert jobs["PatchTST"]["training_range_source"] == "model_override:PatchTST"
    assert jobs["TFT"]["training_range_source"] == "model_override:TFT"


def test_effective_config_pins_val_size_to_horizon_for_training_auto(tmp_path: Path):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["batch_size"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    import runtime_support.runner as runtime

    loaded = load_app_config(tmp_path, config_path=config_path)
    effective = runtime._effective_config(loaded, {"batch_size": 123})

    assert effective.training.batch_size == 123
    assert effective.training.val_size == loaded.config.cv.horizon


def test_effective_config_maps_training_model_step_size_override_for_training_auto(
    tmp_path: Path,
):
    payload = _payload()
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": ["model_step_size"],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    import runtime_support.runner as runtime

    loaded = load_app_config(tmp_path, config_path=config_path)
    effective = runtime._effective_config(loaded, {"model_step_size": 5})

    assert effective.training.model_step_size == 5
    assert effective.training.val_size == loaded.config.cv.horizon


def test_supported_auto_model_matrix_matches_registry_and_yaml():
    search_space = _load_search_space_strict()
    learned_model_classes = {
        model_name for model_name in MODEL_CLASSES if not model_name.startswith("Dummy")
    }
    model_auto_registry_models = set(MODEL_PARAM_REGISTRY)
    expected_model_auto_models = set(EXPECTED_REPO_AUTO_MODELS)
    expected_training_auto_models = set(EXPECTED_REPO_TRAINING_AUTO_MODELS)
    assert "HINT" not in SUPPORTED_AUTO_MODEL_NAMES
    assert SUPPORTED_AUTO_MODEL_NAMES == learned_model_classes
    assert set(SUPPORTED_TOP_LEVEL_DIRECT_MODEL_NAMES) == set(EXPECTED_TOP_LEVEL_DIRECT_MODELS)
    assert set(SUPPORTED_MODEL_AUTO_MODEL_NAMES) == (
        learned_model_classes | set(EXPECTED_TOP_LEVEL_DIRECT_MODELS)
    )
    assert model_auto_registry_models == expected_model_auto_models
    assert set(search_space["models"]) == expected_model_auto_models
    assert set(TRAINING_PARAM_REGISTRY_BY_MODEL) == set(SUPPORTED_AUTO_MODEL_NAMES)
    assert set(search_space["training"]["per_model"]) == expected_training_auto_models
    assert all(
        set(TRAINING_PARAM_REGISTRY_BY_MODEL[model_name])
        == set(search_space["training"]["per_model"][model_name])
        == set(TRAINING_PARAM_REGISTRY)
        == set(search_space["training"]["global"])
        for model_name in EXPECTED_REPO_TRAINING_AUTO_MODELS
    )
    assert tuple(search_space["training"]["global"]) == tuple(TRAINING_PARAM_REGISTRY)
    assert set(search_space["residual"]) == {"xgboost", "randomforest", "lightgbm"}
    assert all(
        isinstance(search_space["models"][model], dict)
        for model in search_space["models"]
    )
    assert "PatchTST" not in search_space["models"]
    assert "TFT" not in search_space["models"]
    assert training_range_source_for_model("PatchTST") == "model_override:PatchTST"
    assert training_range_source_for_model("TFT") == "model_override:TFT"
    assert training_range_source_for_model(None) == "global_fallback"


def test_narrowed_model_param_ranges_are_explicit_for_selected_auto_models():
    class _RangeRecordingTrial:
        def __init__(self) -> None:
            self.categorical: dict[str, tuple[Any, ...]] = {}
            self.integer: dict[str, tuple[int, int, int]] = {}
            self.floating: dict[str, tuple[float, float, bool]] = {}

        def suggest_categorical(self, name, options):
            self.categorical[name] = tuple(options)
            return options[0]

        def suggest_int(self, name, low, high, step=1):
            self.integer[name] = (low, high, step)
            return low

        def suggest_float(self, name, low, high, log=False):
            self.floating[name] = (low, high, log)
            return low

    expectations = {
        "TSMixerx": {
            "categorical": {
                "n_block": (4, 8),
                "ff_dim": (256, 512, 1024),
                "dropout": (0.1, 0.2, 0.3),
                "revin": (True, False),
            },
        },
        "TimeXer": {
            "categorical": {
                "patch_len": (8, 16),
                "hidden_size": (256, 512, 768),
                "n_heads": (16, 32),
                "e_layers": (4, 8),
                "d_ff": (512, 1024),
                "factor": (4, 8),
                "dropout": (0.1, 0.2, 0.3),
                "use_norm": (True,),
            },
        },
        "iTransformer": {
            "categorical": {
                "hidden_size": (256, 512, 768),
                "n_heads": (16, 32),
                "e_layers": (4, 8),
                "d_ff": (1024, 2048),
                "d_layers": (4, 8),
                "factor": (4, 8),
                "dropout": (0.0, 0.1, 0.2),
                "use_norm": (True, False),
            },
        },
        "LSTM": {
            "categorical": {
                "encoder_hidden_size": (256, 512, 768),
                "encoder_n_layers": (4, 6, 8),
                "inference_input_size": (32, 64, 128),
                "encoder_dropout": (0.0, 0.1, 0.2, 0.3),
                "decoder_hidden_size": (256, 512, 768),
                "decoder_layers": (2, 4),
                "context_size": (16, 32, 64),
            },
        },
    }

    for model_name, expected in expectations.items():
        trial = _RangeRecordingTrial()
        suggested = suggest_model_params(
            model_name, tuple(MODEL_PARAM_REGISTRY[model_name]), trial
        )
        assert set(suggested) == set(MODEL_PARAM_REGISTRY[model_name])
        for name, options in expected.get("categorical", {}).items():
            assert trial.categorical[name] == options
        for name, bounds in expected.get("integer", {}).items():
            assert trial.integer[name] == bounds


def test_priority_models_have_narrowed_training_range_overrides():
    class _RangeRecordingTrial:
        def __init__(self) -> None:
            self.categorical: dict[str, tuple[Any, ...]] = {}
            self.floating: dict[str, tuple[float, float, bool]] = {}

        def suggest_categorical(self, name, options):
            self.categorical[name] = tuple(options)
            return options[0]

        def suggest_float(self, name, low, high, log=False):
            self.floating[name] = (low, high, log)
            return low

    cases = {
        "PatchTST": {
            "categorical": {
                "input_size": (48, 64, 96),
                "batch_size": (16, 32, 64, 128),
                "scaler_type": (None,),
                "model_step_size": (4, 8),
            },
            "floating": {},
        },
        "TSMixerx": {
            "categorical": {
                "input_size": (48, 64, 72),
                "batch_size": (16,),
                "scaler_type": (None,),
                "model_step_size": (4, 8),
            },
            "floating": {},
        },
        "iTransformer": {
            "categorical": {
                "input_size": (48, 64, 96),
                "batch_size": (16,),
                "scaler_type": (None,),
                "model_step_size": (4, 8),
            },
            "floating": {},
        },
        "LSTM": {
            "categorical": {
                "input_size": (24, 48, 96),
                "batch_size": (32, 64),
                "scaler_type": (None,),
                "model_step_size": (4, 8),
            },
            "floating": {},
        },
    }

    for model_name, expected in cases.items():
        trial = _RangeRecordingTrial()
        suggested = suggest_training_params(
            tuple(TRAINING_PARAM_REGISTRY),
            trial,
            model_name=model_name,
        )
        assert set(suggested) == set(TRAINING_PARAM_REGISTRY)
        for name, options in expected["categorical"].items():
            assert trial.categorical[name] == options
        for name, bounds in expected["floating"].items():
            assert trial.floating[name] == bounds


def test_supported_residual_model_matrix_matches_defaults_registry_and_yaml():
    search_space = _load_search_space_strict()
    residual_defaults_map = _residual_defaults_map()

    assert set(SUPPORTED_RESIDUAL_MODELS) == set(EXPECTED_SUPPORTED_RESIDUAL_MODELS)
    assert set(residual_defaults_map) == set(EXPECTED_SUPPORTED_RESIDUAL_MODELS)
    assert set(RESIDUAL_PARAM_REGISTRY) == set(EXPECTED_SUPPORTED_RESIDUAL_MODELS)
    assert set(search_space["residual"]) == set(EXPECTED_SUPPORTED_RESIDUAL_MODELS)

    for model_name in EXPECTED_SUPPORTED_RESIDUAL_MODELS:
        assert set(residual_defaults_map[model_name]) == set(
            RESIDUAL_PARAM_REGISTRY[model_name]
        )
        assert set(residual_defaults_map[model_name]) == set(
            search_space["residual"][model_name]
        )
        assert residual_defaults_map[model_name]

    assert tuple(search_space["residual"]["xgboost"]) == XGBOOST_RESIDUAL_PARAM_KEYS


def test_supported_auto_model_matrix_includes_v3_expansion():
    for model_name in (
        "RNN",
        "GRU",
        "TCN",
        "DeepAR",
        "DilatedRNN",
        "BiTCN",
        "xLSTM",
        "MLP",
        "NBEATS",
        "NBEATSx",
        "DLinear",
        "NLinear",
        "TiDE",
        "DeepNPTS",
        "DeformTime",
        "KAN",
        "TimeLLM",
        "TimeXer",
        "TimesNet",
        "StemGNN",
        "TSMixer",
        "TSMixerx",
        "MLPMultivariate",
        "SOFTS",
        "TimeMixer",
        "ModernTCN",
        "DUET",
        "Mamba",
        "SMamba",
        "CMamba",
        "xLSTMMixer",
        "RMoK",
        "XLinear",
    ):
        assert model_name in SUPPORTED_AUTO_MODEL_NAMES
    assert EXCLUDED_AUTO_MODEL_NAMES == {"HINT"}


def test_repo_search_space_bs_preforcast_sections_are_unique_and_include_stage_only_models():
    search_space = _load_search_space_strict()

    assert set(search_space["bs_preforcast_models"]) == {
        "ARIMA",
        "ES",
        "xgboost",
        "lightgbm",
        "NHITS",
        "LSTM",
        "TSMixerx",
        "TimeXer",
        "TFT",
    }
    assert "ARIMA" in SUPPORTED_BS_PREFORCAST_MODELS
    assert "ES" in SUPPORTED_BS_PREFORCAST_MODELS
    assert "xgboost" in SUPPORTED_BS_PREFORCAST_MODELS
    assert "lightgbm" in SUPPORTED_BS_PREFORCAST_MODELS
    assert tuple(search_space["bs_preforcast_models"]["ARIMA"]) == (
        "order",
        "include_mean",
        "include_drift",
    )
    assert tuple(search_space["bs_preforcast_models"]["ES"]) == (
        "trend",
        "damped_trend",
    )
    assert tuple(search_space["bs_preforcast_models"]["xgboost"]) == (
        "lags",
        "n_estimators",
        "max_depth",
    )
    assert tuple(search_space["bs_preforcast_models"]["lightgbm"]) == (
        "lags",
        "n_estimators",
        "max_depth",
        "num_leaves",
        "min_child_samples",
        "feature_fraction",
    )
    assert tuple(search_space["bs_preforcast_models"]["NHITS"]) == ("mlp_units",)
    assert search_space["bs_preforcast_models"]["ARIMA"]["order"]["choices"][1] == [
        1,
        1,
        0,
    ]
    assert search_space["bs_preforcast_models"]["xgboost"]["lags"]["choices"][1] == [
        1,
        2,
        3,
        6,
        12,
    ]
    assert search_space["bs_preforcast_models"]["lightgbm"]["lags"]["choices"][2] == [
        1,
        2,
        3,
        6,
        12,
        24,
    ]
    assert search_space["bs_preforcast_models"]["NHITS"]["mlp_units"]["choices"][0] == [
        [32, 32],
        [32, 32],
        [32, 32],
    ]


def test_package_exports_and_intentional_omissions_are_explicit():
    for model_name in NEWLY_SUPPORTED_MODEL_ALIASES:
        assert hasattr(nf_models, model_name)
    for auto_name in (
        "AutoNonstationaryTransformer",
        "AutoMamba",
        "AutoSMamba",
        "AutoCMamba",
        "AutoxLSTMMixer",
        "AutoDUET",
    ):
        assert hasattr(nf_auto, auto_name)
    assert hasattr(nf_models, "HINT")
    assert "HINT" not in SUPPORTED_AUTO_MODEL_NAMES
    assert not hasattr(nf_auto, "AutoDeformTime")
    assert not hasattr(nf_auto, "AutoModernTCN")
    assert not hasattr(nf_auto, "AutoDeepEDM")
    assert hasattr(nf_models, "DeformTime")
    assert not hasattr(nf_models, "DeepEDM")
    assert hasattr(nf_models, "NonstationaryTransformer")
    assert hasattr(nf_models, "DeformableTST")
    assert "DeformTime" in SUPPORTED_AUTO_MODEL_NAMES
    assert "DeformTime" in MODEL_CLASSES
    assert "DeepEDM" not in SUPPORTED_AUTO_MODEL_NAMES
    assert "DeepEDM" not in MODEL_CLASSES
    assert "NonstationaryTransformer" in SUPPORTED_AUTO_MODEL_NAMES
    assert "NonstationaryTransformer" in MODEL_CLASSES
    assert "deepedm" not in MODEL_FILENAME_DICT
    assert "autodeepedm" not in MODEL_FILENAME_DICT
    assert "nonstationarytransformer" in MODEL_FILENAME_DICT
    assert "autononstationarytransformer" in MODEL_FILENAME_DICT
    assert "DeformableTST" in SUPPORTED_AUTO_MODEL_NAMES
    assert "DeformableTST" in MODEL_CLASSES
    search_space = yaml.safe_load((REPO_ROOT / "yaml/HPO/search_space.yaml").read_text())
    assert "DeformTime" not in search_space["models"]
    assert "DeepEDM" not in search_space["models"]
    assert "NonstationaryTransformer" in search_space["models"]
    assert "DeformableTST" not in search_space["models"]


@pytest.mark.parametrize(
    ("model_name", "aliases"),
    NEWLY_SUPPORTED_MODEL_ALIASES.items(),
    ids=NEWLY_SUPPORTED_MODEL_ALIASES.keys(),
)
def test_model_filename_dict_includes_newly_supported_aliases(
    model_name: str, aliases: tuple[str, ...]
):
    expected_cls = getattr(nf_models, model_name)
    for alias in aliases:
        assert MODEL_FILENAME_DICT[alias] is expected_cls


def test_supports_auto_mode_expands_to_newly_added_models():
    for model_name in (
        "RNN",
        "DeepAR",
        "DeformTime",
        "DeformableTST",
        "NonstationaryTransformer",
        "TimeMixer",
        "ModernTCN",
        "DUET",
        "XLinear",
        "StemGNN",
        "TimeLLM",
        "Mamba",
        "SMamba",
        "CMamba",
        "xLSTMMixer",
        "ARIMA",
        "ES",
        "xgboost",
        "lightgbm",
    ):
        assert supports_auto_mode(model_name) is True
    assert supports_auto_mode("DeepEDM") is False
    assert supports_auto_mode("Naive") is False


def test_build_model_supports_new_official_model_ports(tmp_path: Path):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {
            "model": "DeformTime",
            "params": {"d_model": 16, "patch_len": 4},
        },
        {
            "model": "DeformableTST",
            "params": {
                "dims": [32, 64, 128, 256],
                "depths": [1, 1, 2, 1],
                "drop": 0.1,
                "heads": [2, 4, 8, 16],
            },
        },
        {
            "model": "ModernTCN",
            "params": {
                "patch_size": 8,
                "patch_stride": 4,
                "num_blocks": [1, 1, 1, 1],
                "large_size": [5, 5, 3, 3],
                "small_size": [3, 3, 3, 3],
                "dims": [8, 8, 8, 8],
            },
        },
        {
            "model": "NonstationaryTransformer",
            "params": {
                "hidden_size": 16,
                "conv_hidden_size": 16,
                "dropout": 0.1,
                "n_head": 4,
                "encoder_layers": 1,
                "decoder_layers": 1,
            },
        },
    ]
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    deform = build_model(loaded.config, loaded.config.jobs[0], n_series=1)
    deformabletst = build_model(loaded.config, loaded.config.jobs[1], n_series=1)
    modern = build_model(loaded.config, loaded.config.jobs[2], n_series=1)
    nonstationary = build_model(loaded.config, loaded.config.jobs[3], n_series=1)

    assert deform.__class__.__name__ == "DeformTime"
    assert deformabletst.__class__.__name__ == "DeformableTST"
    assert modern.__class__.__name__ == "ModernTCN"
    assert nonstationary.__class__.__name__ == "NonstationaryTransformer"


def test_should_use_multivariate_for_no_exog_multivariate_model(tmp_path: Path):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {"model": "DummyMultivariate", "params": {"start_padding_enabled": True}}
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import runtime_support.runner as runtime

    assert runtime._should_use_multivariate(loaded, loaded.config.jobs[0]) is True


def test_should_use_univariate_adapter_for_multivariate_model_with_native_exog(
    tmp_path: Path,
):
    payload = _payload()
    payload["jobs"] = [
        {
            "model": "TimeXer",
            "params": {
                "patch_len": 1,
                "hidden_size": 8,
                "n_heads": 1,
                "e_layers": 1,
                "d_ff": 16,
            },
        }
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,10\n2020-01-08,2,11\n2020-01-15,3,12\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import runtime_support.runner as runtime

    assert runtime._should_use_multivariate(loaded, loaded.config.jobs[0]) is False


def test_should_use_univariate_adapter_for_timexer_with_native_future_exog(
    tmp_path: Path,
):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["jobs"] = [
        {
            "model": "TimeXer",
            "params": {
                "patch_len": 1,
                "hidden_size": 8,
                "n_heads": 1,
                "e_layers": 1,
                "d_ff": 16,
            },
        }
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,futr_a\n2020-01-01,1,10\n2020-01-08,2,11\n2020-01-15,3,12\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import runtime_support.runner as runtime

    assert runtime._should_use_multivariate(loaded, loaded.config.jobs[0]) is False
    model = build_model(loaded.config, loaded.config.jobs[0], n_series=1)
    assert model.futr_exog_list == ["futr_a"]


def test_build_model_supports_representative_expanded_models(tmp_path: Path):
    payload = _payload()
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [
        {
            "model": "RNN",
            "params": {
                "encoder_hidden_size": 16,
                "encoder_n_layers": 1,
                "context_size": 5,
                "decoder_hidden_size": 16,
            },
        },
        {
            "model": "MLP",
            "params": {"hidden_size": 32, "num_layers": 2},
        },
        {
            "model": "TimeMixer",
            "params": {
                "d_model": 16,
                "d_ff": 32,
                "down_sampling_layers": 1,
                "top_k": 3,
            },
        },
        {
            "model": "MLPMultivariate",
            "params": {"hidden_size": 32, "num_layers": 2},
        },
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    rnn = build_model(loaded.config, loaded.config.jobs[0])
    mlp = build_model(loaded.config, loaded.config.jobs[1])
    timemixer = build_model(loaded.config, loaded.config.jobs[2], n_series=1)
    mlpmulti = build_model(loaded.config, loaded.config.jobs[3], n_series=1)

    assert rnn.hparams.encoder_hidden_size == 16
    assert mlp.hparams.hidden_size == 32
    assert timemixer.hparams.d_model == 16
    assert getattr(mlpmulti.hparams, "n_series", 1) == 1


def test_runtime_executes_multivariate_model_without_hist_exog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["dataset"]["futr_exog_cols"] = []
    payload["jobs"] = [{"model": "iTransformer", "params": {}}]
    payload["residual"] = {"enabled": False, "model": "xgboost", "params": {}}
    _write_search_space(tmp_path)
    data = (
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n"
        "2020-01-29,5\n2020-02-05,6\n2020-02-12,7\n2020-02-19,8\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    config_path = _write_config(tmp_path, payload, ".yaml")

    import runtime_support.runner as runtime

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "1")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )

    output_root = tmp_path / "run_multivariate"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "iTransformer",
            "--output-root",
            str(output_root),
        ]
    )
    assert code == 0
    assert (output_root / "cv" / "iTransformer_forecasts.csv").exists()


def test_runtime_single_job_rerun_prunes_stale_model_artifacts_and_rebuilds_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _payload()
    payload["task"] = {"name": "wti_case3_residual"}
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["jobs"] = [
        {
            "model": "iTransformer",
            "params": {"hidden_size": 32, "n_heads": 4, "e_layers": 2, "d_ff": 64},
        }
    ]
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,9\n2020-01-08,2,8\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(tmp_path)

    import runtime_support.runner as runtime

    def _write_metrics_and_forecasts(
        root: Path,
        model_name: str,
        *,
        mae: float,
        nrmse: float,
        y_hat: float,
    ) -> None:
        cv_dir = root / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "fold_idx": 0,
                    "cutoff": "2020-01-08",
                    "MAE": mae,
                    "MSE": mae**2,
                    "RMSE": mae,
                    "MAPE": mae / 10,
                    "NRMSE": nrmse,
                    "R2": 1 - mae,
                }
            ]
        ).to_csv(cv_dir / f"{model_name}_metrics_by_cutoff.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": model_name,
                    "fold_idx": 0,
                    "cutoff": "2020-01-08",
                    "train_end_ds": "2020-01-08",
                    "unique_id": "target",
                    "ds": "2020-01-15",
                    "horizon_step": 1,
                    "y": 1.0,
                    "y_hat": y_hat,
                }
            ]
        ).to_csv(cv_dir / f"{model_name}_forecasts.csv", index=False)

    def _write_residual_artifacts(
        root: Path,
        model_name: str,
        *,
        mae: float,
        nrmse: float,
        y_hat_corrected: float,
    ) -> None:
        residual_root = root / "residual" / model_name
        fold_root = residual_root / "folds" / "fold_000"
        fold_root.mkdir(parents=True, exist_ok=True)
        (fold_root / "metrics.json").write_text(
            json.dumps(
                {
                    "fold_idx": 0,
                    "cutoff": "2020-01-08",
                    "corrected_metrics": {
                        "MAE": mae,
                        "MSE": mae**2,
                        "RMSE": mae,
                        "MAPE": mae / 10,
                        "NRMSE": nrmse,
                        "R2": 1 - mae,
                    },
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "fold_idx": 0,
                    "cutoff": "2020-01-08",
                    "train_end_ds": "2020-01-08",
                    "unique_id": "target",
                    "ds": "2020-01-15",
                    "horizon_step": 1,
                    "y": 1.0,
                    "y_hat_corrected": y_hat_corrected,
                }
            ]
        ).to_csv(residual_root / "corrected_folds.csv", index=False)

    run_root = tmp_path / "rerun_root"
    sibling_files: dict[Path, bytes] = {}
    sibling_models = ["PatchTST", "TSMixerx", "Naive", "LSTM"]
    for index, model_name in enumerate(sibling_models, start=1):
        worker_root = run_root / "scheduler" / "workers" / model_name
        _write_metrics_and_forecasts(
            worker_root,
            model_name,
            mae=float(index),
            nrmse=0.1 * index,
            y_hat=1.0 + index,
        )
        for path in sorted((worker_root / "cv").glob("*")):
            sibling_files[path] = path.read_bytes()

    stale_worker_root = run_root / "scheduler" / "workers" / "iTransformer"
    _write_metrics_and_forecasts(
        stale_worker_root,
        "iTransformer",
        mae=77.0,
        nrmse=7.7,
        y_hat=77.0,
    )
    _write_residual_artifacts(
        stale_worker_root,
        "iTransformer",
        mae=66.0,
        nrmse=6.6,
        y_hat_corrected=66.0,
    )
    (stale_worker_root / "summary.json").write_text(
        json.dumps({"job_name": "iTransformer", "returncode": 1}),
        encoding="utf-8",
    )

    _write_metrics_and_forecasts(
        run_root,
        "iTransformer",
        mae=99.0,
        nrmse=9.9,
        y_hat=99.0,
    )
    _write_residual_artifacts(
        run_root,
        "iTransformer",
        mae=88.0,
        nrmse=8.8,
        y_hat_corrected=88.0,
    )
    stale_model_dir = run_root / "models" / "iTransformer"
    stale_model_dir.mkdir(parents=True, exist_ok=True)
    (stale_model_dir / "fit_summary.json").write_text(
        json.dumps({"stale": True}),
        encoding="utf-8",
    )

    def _fake_run_single_job(
        loaded,
        job,
        run_root_arg: Path,
        *,
        manifest_path: Path,
        main_stage: str = "full",
    ) -> None:
        assert job.model == "iTransformer"
        assert run_root_arg == run_root
        assert main_stage == "full"
        _write_metrics_and_forecasts(
            run_root_arg,
            "iTransformer",
            mae=0.7,
            nrmse=0.07,
            y_hat=1.7,
        )
        _write_residual_artifacts(
            run_root_arg,
            "iTransformer",
            mae=0.3,
            nrmse=0.03,
            y_hat_corrected=1.3,
        )
        model_dir = run_root_arg / "models" / "iTransformer"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "fit_summary.json").write_text(
            json.dumps({"stale": False, "main_stage": main_stage}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )
    monkeypatch.setattr(runtime, "_run_single_job", _fake_run_single_job)

    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "iTransformer",
            "--output-root",
            str(run_root),
        ]
    )

    assert code == 0
    for path, content in sibling_files.items():
        assert path.read_bytes() == content
    assert not stale_worker_root.exists()
    assert json.loads((run_root / "models" / "iTransformer" / "fit_summary.json").read_text()) == {
        "stale": False,
        "main_stage": "full",
    }

    leaderboard = pd.read_csv(run_root / "summary" / "leaderboard.csv")
    assert sorted(leaderboard["model"].tolist()) == sorted(
        [*sibling_models, "iTransformer", "iTransformer_res"]
    )
    itransformer_row = leaderboard.loc[leaderboard["model"] == "iTransformer"].iloc[0]
    residual_row = leaderboard.loc[leaderboard["model"] == "iTransformer_res"].iloc[0]
    assert itransformer_row["mean_fold_mae"] == pytest.approx(0.7)
    assert itransformer_row["mean_fold_nrmse"] == pytest.approx(0.07)
    assert residual_row["mean_fold_mae"] == pytest.approx(0.3)
    assert residual_row["mean_fold_nrmse"] == pytest.approx(0.03)
    assert (run_root / "summary" / "sample.md").exists()
    assert (run_root / "summary" / "last_fold_all_models.png").exists()


CASE_YAML_FILES = [
    REPO_ROOT / "yaml" / "experiment" / "feature_set" / "brentoil-case1.yaml",
    REPO_ROOT / "yaml" / "experiment" / "feature_set" / "brentoil-case2.yaml",
    REPO_ROOT / "yaml" / "experiment" / "feature_set" / "brentoil-case3.yaml",
    REPO_ROOT / "yaml" / "experiment" / "feature_set" / "brentoil-case4.yaml",
    REPO_ROOT / "yaml" / "experiment" / "feature_set" / "wti-case1.yaml",
    REPO_ROOT / "yaml" / "experiment" / "feature_set" / "wti-case2.yaml",
    REPO_ROOT / "yaml" / "experiment" / "feature_set" / "wti-case3.yaml",
    REPO_ROOT / "yaml" / "experiment" / "feature_set" / "wti-case4.yaml",
]

def _existing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


FEATURE_SET_RESIDUAL_YAML_FILES = _existing_paths(
    [
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "brentoil-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "brentoil-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "brentoil-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "brentoil-case4.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "wti-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "wti-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "wti-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "wti-case4.yaml",
    ]
)

NEW_FEATURE_SET_RESIDUAL_YAML_FILES = _existing_paths(
    [
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "brentoil-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "brentoil-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "brentoil-case4.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "wti-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "wti-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_residual" / "wti-case4.yaml",
    ]
)

HPT_CASE12_YAML_FILES = _existing_paths(
    [
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "brentoil-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "brentoil-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "wti-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "wti-case2.yaml",
    ]
)

HPT_YAML_FILES = _existing_paths(
    [
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "brentoil-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "brentoil-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "brentoil-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "brentoil-case4.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "wti-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "wti-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "wti-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "wti-case4.yaml",
    ]
)

HPT_N100_YAML_FILES = _existing_paths(
    [
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / "brentoil-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / "brentoil-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / "brentoil-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / "brentoil-case4.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / "wti-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / "wti-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / "wti-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / "wti-case4.yaml",
    ]
)

HPT_N100_RESIDUAL_YAML_FILES = _existing_paths(
    [
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual" / "brentoil-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual" / "brentoil-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual" / "brentoil-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual" / "brentoil-case4.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual" / "wti-case1.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual" / "wti-case2.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual" / "wti-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual" / "wti-case4.yaml",
    ]
)

HPT_CASE3_REP_YAML_FILES = _existing_paths(
    [
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "brentoil-case3.yaml",
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT" / "wti-case3.yaml",
    ]
)

OPTUNA_CONFIG_YAML_FILES = [
    *HPT_CASE12_YAML_FILES,
    REPO_ROOT / "tests" / "fixtures" / "optuna_learned_auto.yaml",
    *RESIDUAL_AUTO_FIXTURE_FILES,
    REPO_ROOT / "tests" / "fixtures" / "optuna_relocated_config.yaml",
    REPO_ROOT / "tests" / "fixtures" / "optuna_unsupported_learned_empty_params.yaml",
]

EXPECTED_CASE_TRAINING = {
    "input_size": 64,
    "batch_size": 32,
    "valid_batch_size": 64,
    "windows_batch_size": 1024,
    "inference_windows_batch_size": 1024,
    "optimizer": {"name": "adamw", "kwargs": {}},
    "lr_scheduler": _onecycle_scheduler(0.001),
    "model_step_size": 8,
    "max_steps": 1000,
    "val_size": 8,
    "val_check_steps": 20,
    "min_steps_before_early_stop": 500,
    "early_stop_patience_steps": 3,
    "loss": "mse",
}

EXPECTED_CASE_MODEL_PARAMS = {
    "LSTM": {
        "encoder_hidden_size": 64,
        "decoder_hidden_size": 64,
        "encoder_n_layers": 4,
        "context_size": 8,
    },
    "TimeXer": {
        "patch_len": 8,
        "hidden_size": 768,
        "n_heads": 16,
        "e_layers": 4,
        "d_ff": 1024,
        "factor": 4,
        "dropout": 0.15,
        "use_norm": True,
    },
    "iTransformer": {
        "hidden_size": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 192,
        "dropout": 0.0,
    },
    "TSMixerx": {
        "n_block": 3,
        "ff_dim": 64,
        "dropout": 0.05,
        "revin": True,
    },
    "Naive": {},
}

EXPECTED_SHARED_DEFAULT_MODEL_PARAMS = {
    "LSTM": {
        "encoder_hidden_size": 64,
        "decoder_hidden_size": 64,
        "encoder_n_layers": 4,
        "context_size": 10,
    },
    "TimeXer": {
        "patch_len": 16,
        "hidden_size": 768,
        "n_heads": 16,
        "e_layers": 4,
        "d_ff": 1024,
        "factor": 8,
        "dropout": 0.2,
        "use_norm": True,
    },
    "iTransformer": {
        "hidden_size": 64,
        "n_heads": 4,
        "e_layers": 2,
        "d_ff": 256,
        "dropout": 0.0,
    },
    "TSMixerx": {
        "n_block": 2,
        "ff_dim": 64,
        "dropout": 0.1,
        "revin": True,
    },
    "Naive": {},
}

EXPECTED_CASE_MODEL_LIST = [
    "TimeXer",
    "TSMixerx",
    "Naive",
    "iTransformer",
    "LSTM",
]

EXPECTED_HPT_CASE12_TRAINING = {
    "batch_size": 32,
    "valid_batch_size": 64,
    "windows_batch_size": 1024,
    "inference_windows_batch_size": 1024,
    "max_steps": 1000,
    "val_size": 8,
    "val_check_steps": 100,
    "min_steps_before_early_stop": 500,
    "early_stop_patience_steps": 5,
    "loss": "mse",
    "optimizer": {"name": "adamw", "kwargs": {}},
    "dataloader_kwargs": {
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    },
}

EXPECTED_HPT_CASE12_MODELS = [
    "PatchTST",
    "Naive",
    "iTransformer",
    "LSTM",
    "NHITS",
]

EXPECTED_HPT_N100_MODELS = [
    "TimeXer",
    "TSMixerx",
    "Naive",
    "iTransformer",
    "LSTM",
]

SEARCH_SPACE_PAYLOAD = yaml.safe_load(
    (REPO_ROOT / "yaml/HPO/search_space.yaml").read_text(encoding="utf-8")
)
SEARCH_SPACE_MODELS_RAW = SEARCH_SPACE_PAYLOAD["models"]
SEARCH_SPACE_MODELS = {
    model_name: list(specs)
    for model_name, specs in SEARCH_SPACE_MODELS_RAW.items()
}
SEARCH_SPACE_TRAINING_RAW = SEARCH_SPACE_PAYLOAD["training"]
SEARCH_SPACE_TRAINING = list(SEARCH_SPACE_TRAINING_RAW["global"])
SEARCH_SPACE_TRAINING_PER_MODEL = {
    model_name: list(specs)
    for model_name, specs in SEARCH_SPACE_TRAINING_RAW.get("per_model", {}).items()
}
SEARCH_SPACE_RESIDUAL_RAW = SEARCH_SPACE_PAYLOAD["residual"]
SEARCH_SPACE_RESIDUAL = {
    model_name: list(specs)
    for model_name, specs in SEARCH_SPACE_RESIDUAL_RAW.items()
}

EXPECTED_TOP_LEVEL_DIRECT_MODELS = [
    "ARIMA",
    "ES",
    "xgboost",
    "lightgbm",
]

EXPECTED_REPO_TRAINING_AUTO_MODELS = [
    "LSTM",
    "iTransformer",
    "TSMixerx",
    "TimeXer",
    "NonstationaryTransformer",
]

EXPECTED_REPO_AUTO_MODELS = [
    *EXPECTED_REPO_TRAINING_AUTO_MODELS,
    *EXPECTED_TOP_LEVEL_DIRECT_MODELS,
]

EXPECTED_REPO_AUTO_SELECTORS = {
    "iTransformer": [
        "hidden_size",
        "n_heads",
        "e_layers",
        "d_ff",
        "d_layers",
        "factor",
        "dropout",
        "use_norm",
    ],
    "TimeXer": [
        "patch_len",
        "hidden_size",
        "n_heads",
        "e_layers",
        "d_ff",
        "factor",
        "dropout",
        "use_norm",
    ],
    "TSMixerx": [
        "n_block",
        "ff_dim",
        "dropout",
        "revin",
    ],
    "LSTM": [
        "encoder_hidden_size",
        "encoder_n_layers",
        "inference_input_size",
        "encoder_dropout",
        "decoder_hidden_size",
        "decoder_layers",
        "context_size",
    ],
    "NonstationaryTransformer": [
        "hidden_size",
        "dropout",
        "n_head",
        "conv_hidden_size",
        "encoder_layers",
        "decoder_layers",
    ],
    "ARIMA": [
        "order",
        "include_mean",
        "include_drift",
    ],
    "ES": [
        "trend",
        "damped_trend",
    ],
    "xgboost": [
        "lags",
        "n_estimators",
        "max_depth",
    ],
    "lightgbm": [
        "lags",
        "n_estimators",
        "max_depth",
        "num_leaves",
        "min_child_samples",
        "feature_fraction",
    ],
}
EXPECTED_REPO_TRAINING_SELECTORS = [
    "input_size",
    "batch_size",
    "scaler_type",
    "model_step_size",
]

EXCLUDED_REPO_TRAINING_SELECTORS = [
    "valid_batch_size",
    "windows_batch_size",
    "inference_windows_batch_size",
]

FEATURE_SET_RESIDUAL_EXPECTED_FILENAMES = sorted(
    path.name for path in FEATURE_SET_RESIDUAL_YAML_FILES
)

EXPECTED_FEATURE_SET_RESIDUAL_PARAMS = {
    "n_estimators": 96,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

EXPECTED_FEATURE_SET_RESIDUAL_LAG_STEPS = [1, 2, 3, 4, 6, 12]

EXPECTED_CASE_METADATA = {
    "brentoil-case1.yaml": {
        "task_name": "brentoil_case1",
        "target_col": "Com_BrentCrudeOil",
        "hist_exog_cols": [
            "Com_Gasoline",
            "Com_Steel",
            "Bonds_US_Spread_10Y_1Y",
            "Bonds_CHN_Spread_30Y_5Y",
            "EX_USD_BRL",
            "Com_Cheese",
            "Bonds_BRZ_Spread_10Y_1Y",
            "Com_Cu_Gold_Ratio",
            "Idx_OVX",
            "Com_Oil_Spread",
            "Com_LME_Zn_Spread",
            "Idx_CSI300",
            "Bonds_CHN_Spread_5Y_1Y",
            "Com_LME_Cu_Spread",
            "Com_LME_Pb_Spread",
            "Com_LME_Al_Spread",
        ],
    },
    "brentoil-case2.yaml": {
        "task_name": "brentoil_case2",
        "target_col": "Com_BrentCrudeOil",
        "hist_exog_cols": [
            "Com_Gasoline",
            "Com_BloombergCommodity_BCOM",
            "Com_LME_Ni_Cash",
            "Com_Coal",
            "Com_Cotton",
            "Com_LME_Al_Cash",
            "Bonds_KOR_10Y",
            "Com_Barley",
            "Com_Canola",
            "Com_LMEX",
            "Com_LME_Ni_Inv",
            "Com_Corn",
            "Com_Wheat",
        ],
    },
    "brentoil-case3.yaml": {
        "task_name": "brentoil_case3",
        "target_col": "Com_BrentCrudeOil",
        "hist_exog_cols": [
            "Com_Gasoline",
            "Com_BloombergCommodity_BCOM",
            "Com_LME_Ni_Cash",
            "Com_Coal",
            "Com_LME_Al_Cash",
            "Bonds_KOR_10Y",
            "Com_LMEX",
            "Com_LME_Ni_Inv",
        ],
    },
    "brentoil-case4.yaml": {
        "task_name": "brentoil_case4",
        "target_col": "Com_BrentCrudeOil",
        "hist_exog_cols": [
            "Com_Gasoline",
            "Com_BloombergCommodity_BCOM",
            "Com_LME_Ni_Cash",
            "Com_Coal",
            "Com_Cotton",
            "Com_LME_Al_Cash",
            "Bonds_KOR_10Y",
            "Com_Barley",
            "Com_Canola",
            "Com_LMEX",
            "Com_LME_Ni_Inv",
            "Com_Corn",
            "Com_Wheat",
            "Com_NaturalGas",
            "Idx_OVX",
            "Com_Gold",
        ],
    },
    "wti-case1.yaml": {
        "task_name": "wti_case1",
        "target_col": "Com_CrudeOil",
        "hist_exog_cols": [
            "Com_Gasoline",
            "Com_LME_Zn_Inv",
            "Com_OrangeJuice",
            "Com_Cheese",
            "Bonds_BRZ_1Y",
            "Idx_OVX",
            "Com_Cu_Gold_Ratio",
            "Com_LME_Sn_Inv",
            "Idx_CSI300",
            "Com_LME_Zn_Spread",
            "Bonds_CHN_Spread_5Y_2Y",
            "Com_LME_Al_Spread",
            "Bonds_CHN_Spread_2Y_1Y",
            "Com_Oil_Spread",
            "Bonds_CHN_Spread_10Y_5Y",
        ],
    },
    "wti-case2.yaml": {
        "task_name": "wti_case2",
        "target_col": "Com_CrudeOil",
        "hist_exog_cols": [
            "Com_Gasoline",
            "Com_BloombergCommodity_BCOM",
            "Com_LME_Ni_Cash",
            "Com_Coal",
            "Com_Canola",
            "Com_Cotton",
            "Com_LME_Al_Cash",
            "Com_LMEX",
            "Bonds_KOR_10Y",
            "Com_PalmOil",
            "Com_Barley",
            "Com_Corn",
            "Com_Oat",
            "Com_Wheat",
            "Com_Soybeans",
            "Com_LME_Ni_Inv",
        ],
    },
    "wti-case3.yaml": {
        "task_name": "wti_case3",
        "target_col": "Com_CrudeOil",
        "hist_exog_cols": [
            "Com_Gasoline",
            "Com_BloombergCommodity_BCOM",
            "Com_LME_Ni_Cash",
            "Com_Coal",
            "Com_LME_Al_Cash",
            "Com_LMEX",
            "Bonds_KOR_10Y",
            "Com_LME_Ni_Inv",
        ],
    },
    "wti-case4.yaml": {
        "task_name": "wti_case4",
        "target_col": "Com_CrudeOil",
        "hist_exog_cols": [
            "Com_Gasoline",
            "Com_BloombergCommodity_BCOM",
            "Com_LME_Ni_Cash",
            "Com_Coal",
            "Com_Canola",
            "Com_Cotton",
            "Com_LME_Al_Cash",
            "Com_LMEX",
            "Bonds_KOR_10Y",
            "Com_PalmOil",
            "Com_Barley",
            "Com_Corn",
            "Com_Oat",
            "Com_Wheat",
            "Com_Soybeans",
            "Com_LME_Ni_Inv",
            "Com_NaturalGas",
            "Idx_OVX",
            "Com_Gold",
        ],
    },
}


def _load_case_yaml_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        pytest.skip(f"missing case file: {path}")
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def _resolve_case_jobs(path: Path, jobs: Any) -> list[dict[str, Any]]:
    if isinstance(jobs, list):
        if jobs and all(isinstance(item, str) for item in jobs):
            return _resolve_case_jobs(path, jobs[0])
        return jobs
    if isinstance(jobs, dict):
        resolved_jobs = jobs.get("jobs")
        if not isinstance(resolved_jobs, list):
            raise TypeError(f"Unsupported jobs mapping for {path}: {jobs!r}")
        return cast(list[dict[str, Any]], resolved_jobs)
    if isinstance(jobs, str):
        repo_candidate = (REPO_ROOT / jobs).resolve()
        local_candidate = (path.parent / jobs).resolve()
        jobs_path = local_candidate if local_candidate.exists() else repo_candidate
        loaded = yaml.safe_load(jobs_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            resolved_jobs = loaded
        else:
            resolved_jobs = loaded.get("jobs")
        return cast(list[dict[str, Any]], resolved_jobs)
    raise TypeError(f"Unsupported jobs payload for {path}: {type(jobs)!r}")


def _load_case_yaml(path: Path) -> dict[str, Any]:
    payload = _load_case_yaml_raw(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        shared_settings_payload, _, _ = _load_shared_settings_for_yaml_app_config(
            REPO_ROOT
        )
        if shared_settings_payload is not None:
            effective_shared_settings, effective_owned_paths = (
                _effective_shared_settings_for_source(
                    REPO_ROOT, path.resolve(), shared_settings_payload
                )
            )
            payload = _merge_shared_settings_into_payload(
                payload,
                effective_shared_settings,
                owned_paths=effective_owned_paths,
            )
    payload["jobs"] = _resolve_case_jobs(path, payload.get("jobs", []))
    return payload


def _case_jobs_by_model(path: Path) -> dict[str, dict[str, Any]]:
    return {job["model"]: job["params"] for job in _load_case_yaml(path)["jobs"]}


def _hpt_n100_source_path_for_residual(path: Path) -> Path:
    return REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100" / path.name


def _normalized_payload_without_task_and_residual(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    normalized["task"]["name"] = "__SOURCE_TASK__"
    normalized.pop("residual", None)
    return normalized


@pytest.mark.parametrize(
    "relative_path, expected_jobs_ref",
    [
        (
            "yaml/experiment/feature_set/brentoil-case1.yaml",
            [
                "yaml/jobs/main/jobs_1.yaml",
                "yaml/jobs/main/jobs_2.yaml",
                "yaml/jobs/main/jobs_3.yaml",
                "yaml/jobs/main/jobs_4.yaml",
            ],
        ),
        (
            "yaml/experiment/feature_set_bs/brentoil-case1.yaml",
            [
                "yaml/jobs/main/jobs_1.yaml",
                "yaml/jobs/main/jobs_2.yaml",
                "yaml/jobs/main/jobs_3.yaml",
                "yaml/jobs/main/jobs_4.yaml",
            ],
        ),
        ("yaml/experiment/feature_set_bs_diff/brentoil-case1.yaml", "yaml/jobs/main/jobs_default.yaml"),
        ("yaml/experiment/feature_set_bs_exloss/brentoil-case1.yaml", "yaml/jobs/main/jobs_default.yaml"),
        ("yaml/experiment/feature_set_residual/brentoil-case1.yaml", "yaml/jobs/main/jobs_default.yaml"),
        ("yaml/experiment/feature_set_residual_bs/brentoil-case1.yaml", "yaml/jobs/main/jobs_default.yaml"),
        ("yaml/experiment/feature_set_residual_params/brentoil-case3.yaml", "yaml/jobs/main/jobs_default.yaml"),
        ("yaml/experiment/feature_set_HPT_n100_bs/brentoil-case1.yaml", "yaml/jobs/main/jobs_tune.yaml"),
        ("yaml/experiment/feature_set_HPT_n100_residual/brentoil-case1.yaml", "yaml/jobs/main/jobs_tune.yaml"),
        ("yaml/experiment/feature_set_residual_bs_HPT/brentoil-case1.yaml", "yaml/jobs/main/jobs_tune.yaml"),
    ],
)
def test_case_yaml_files_reference_shared_jobs_paths(
    relative_path: str, expected_jobs_ref: object
) -> None:
    payload = _load_case_yaml_raw(REPO_ROOT / relative_path)
    assert payload["jobs"] == expected_jobs_ref


@pytest.mark.parametrize(
    ("relative_path", "expected_jobs_ref"),
    [
        (
            "yaml/plugins/bs_preforcast_uni.yaml",
            "yaml/jobs/bs_preforcast/bs_preforcast_jobs_uni.yaml",
        ),
        (
            "yaml/plugins/bs_preforcast_multi.yaml",
            "yaml/jobs/bs_preforcast/bs_preforcast_jobs_multi.yaml",
        ),
    ],
)
def test_bs_preforcast_plugin_yaml_routes_to_shared_jobs_files(
    relative_path: str, expected_jobs_ref: str
) -> None:
    path = REPO_ROOT / relative_path
    payload = _load_case_yaml_raw(path)

    assert payload["jobs"] == expected_jobs_ref


def test_bs_preforcast_uni_jobs_yaml_fanout_catalog_matches_requested_fixed_params() -> None:
    path = REPO_ROOT / "yaml" / "jobs" / "bs_preforcast" / "bs_preforcast_jobs_uni.yaml"
    jobs = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert [job["model"] for job in jobs] == [
        "ARIMA",
        "DLinear",
        "ES",
        "lightgbm",
        "LSTM",
        "NHITS",
        "PatchTST",
        "xgboost",
    ]
    params_by_model = {job["model"]: job["params"] for job in jobs}
    assert params_by_model == {
        "xgboost": {
            "lags": [1, 2, 3, 6, 12],
            "n_estimators": 64,
            "max_depth": 4,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "lightgbm": {
            "lags": [1, 2, 3, 6, 12],
            "n_estimators": 96,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 20,
        },
        "LSTM": {
            "encoder_hidden_size": 64,
            "decoder_hidden_size": 64,
            "encoder_n_layers": 4,
            "context_size": 8,
        },
        "PatchTST": {
            "hidden_size": 16,
            "n_heads": 4,
            "encoder_layers": 2,
            "patch_len": 4,
        },
        "DLinear": {
            "moving_avg_window": 3,
        },
        "NHITS": {
            "mlp_units": [[32, 32], [32, 32], [32, 32]],
        },
        "ES": {
            "trend": "add",
            "damped_trend": False,
        },
        "ARIMA": {
            "order": [1, 1, 0],
            "include_mean": True,
            "include_drift": False,
        },
    }


def test_bs_preforcast_multi_jobs_yaml_matches_requested_fixed_params() -> None:
    path = REPO_ROOT / "yaml" / "jobs" / "bs_preforcast" / "bs_preforcast_jobs_multi.yaml"
    jobs = yaml.safe_load(path.read_text(encoding="utf-8"))
    params_by_model = {job["model"]: job["params"] for job in jobs}

    assert [job["model"] for job in jobs] == [
        "iTransformer",
        "LSTM",
        "TimeXer",
        "TSMixerx",
    ]
    assert params_by_model == {
        "TimeXer": {
            "patch_len": 8,
            "hidden_size": 768,
            "n_heads": 16,
            "e_layers": 4,
            "d_ff": 1024,
            "factor": 4,
            "dropout": 0.15,
            "use_norm": True,
        },
        "iTransformer": {
            "hidden_size": 64,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 192,
            "dropout": 0.0,
        },
        "LSTM": {
            "encoder_hidden_size": 64,
            "decoder_hidden_size": 64,
            "encoder_n_layers": 4,
            "context_size": 8,
        },
        "TSMixerx": {
            "n_block": 3,
            "ff_dim": 64,
            "dropout": 0.05,
            "revin": True,
        },
    }


def test_shared_jobs_yaml_files_match_expected_contract() -> None:
    default_jobs = _resolve_case_jobs(
        REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_default.yaml",
        yaml.safe_load((REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_default.yaml").read_text(encoding="utf-8")),
    )
    tune_jobs = _resolve_case_jobs(
        REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_tune.yaml",
        yaml.safe_load((REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_tune.yaml").read_text(encoding="utf-8")),
    )

    assert [job["model"] for job in default_jobs] == EXPECTED_CASE_MODEL_LIST
    assert {job["model"]: job["params"] for job in default_jobs} == EXPECTED_SHARED_DEFAULT_MODEL_PARAMS
    assert [job["model"] for job in tune_jobs] == EXPECTED_HPT_N100_MODELS
    assert all(job["params"] == {} for job in tune_jobs)


def test_jobs_path_loading_keeps_inline_compatibility_and_supports_shared_files(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")

    inline_path = tmp_path / "inline.yaml"
    inline_path.write_text(
        yaml.safe_dump(
                {
                    "task": {"name": "inline_case"},
                    "dataset": {"path": str(dataset_path), "target_col": "y"},
                    "training": {"max_steps": 1},
                    "cv": {"n_windows": 1, "step_size": 1},
                    "residual": {"enabled": False},
                    "jobs": [{"model": "Naive", "params": {}}],
                },
                sort_keys=False,
            ),
        encoding="utf-8",
    )
    shared_path = tmp_path / "shared.yaml"
    shared_path.write_text(
        yaml.safe_dump(
                {
                    "task": {"name": "shared_case"},
                    "dataset": {"path": str(dataset_path), "target_col": "y"},
                    "training": {"max_steps": 1},
                    "cv": {"n_windows": 1, "step_size": 1},
                    "residual": {"enabled": False},
                    "jobs": "yaml/jobs/main/jobs_default.yaml",
                },
                sort_keys=False,
            ),
        encoding="utf-8",
    )

    inline_loaded = load_app_config(tmp_path, config_path=inline_path)
    shared_loaded = load_app_config(REPO_ROOT, config_path=shared_path)

    assert [job.model for job in inline_loaded.config.jobs] == ["Naive"]
    assert [job.model for job in shared_loaded.config.jobs] == EXPECTED_CASE_MODEL_LIST
    assert next(job for job in shared_loaded.config.jobs if job.model == "TimeXer").params == (
        EXPECTED_SHARED_DEFAULT_MODEL_PARAMS["TimeXer"]
    )


def test_jobs_path_loading_rejects_missing_or_malformed_shared_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    malformed_jobs = tmp_path / "bad_jobs.yaml"
    malformed_jobs.write_text("jobs: wrong\n", encoding="utf-8")

    for name, jobs_ref, expected_error in [
        ("missing.yaml", "yaml/does-not-exist.yaml", "jobs route does not exist"),
        ("malformed.yaml", str(malformed_jobs), "jobs route must resolve to a list"),
    ]:
        config_path = tmp_path / name
        config_path.write_text(
            yaml.safe_dump(
                {
                    "task": {"name": name.replace(".yaml", "")},
                    "dataset": {"path": str(dataset_path), "target_col": "y"},
                    "training": {"max_steps": 1},
                    "cv": {"n_windows": 1, "step_size": 1},
                    "residual": {"enabled": False},
                    "jobs": jobs_ref,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        with pytest.raises((FileNotFoundError, ValueError), match=expected_error):
            load_app_config(REPO_ROOT, config_path=config_path)


def test_jobs_path_list_loading_builds_fanout_specs(tmp_path: Path) -> None:
    from app_config import loaded_config_for_jobs_fanout

    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    jobs_one = tmp_path / "jobs_1.yaml"
    jobs_two = tmp_path / "jobs_2.yaml"
    jobs_one.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    jobs_two.write_text(
        yaml.safe_dump(
            {"jobs": [{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "fanout.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "fanout_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {"max_steps": 1},
                "cv": {"n_windows": 1, "step_size": 1},
                "residual": {"enabled": False},
                "jobs": [str(jobs_one), str(jobs_two)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert tuple(spec.route_slug for spec in loaded.jobs_fanout_specs) == (
        "jobs_1",
        "jobs_2",
    )
    first_variant = loaded_config_for_jobs_fanout(
        tmp_path, loaded, loaded.jobs_fanout_specs[0]
    )
    second_variant = loaded_config_for_jobs_fanout(
        tmp_path, loaded, loaded.jobs_fanout_specs[1]
    )

    assert [job.model for job in first_variant.config.jobs] == ["Naive"]
    assert [job.model for job in second_variant.config.jobs] == ["DummyUnivariate"]
    assert second_variant.active_jobs_route_slug == "jobs_2"


def test_jobs_path_list_loading_expands_nested_catalog_routes() -> None:
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT
        / "yaml"
        / "experiment"
        / "bs_forecast_uni"
        / "bs_forecast_uni.yaml",
    )

    assert loaded.jobs_fanout_specs == ()
    assert [job.model for job in loaded.config.jobs] == ["Naive"]
    assert [job.model for job in loaded.stage_plugin_loaded.config.jobs] == [
        "ARIMA",
        "DLinear",
        "ES",
        "lightgbm",
        "LSTM",
        "NHITS",
        "PatchTST",
        "xgboost",
    ]


def test_load_app_config_auto_loads_repo_shared_settings_for_repo_yaml(tmp_path: Path) -> None:
    (tmp_path / "yaml" / "setting").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml" / "setting" / "setting.yaml").write_text(
        yaml.safe_dump(
            {
                "runtime": {"random_seed": 11},
                "training": {
                    "input_size": 8,
                    "batch_size": 32,
                    "valid_batch_size": 64,
                    "windows_batch_size": 128,
                    "inference_windows_batch_size": 128,
                    "max_steps": 99,
                    "val_size": 7,
                    "val_check_steps": 5,
                    "model_step_size": 6,
                    "early_stop_patience_steps": 4,
                    "loss": "mse",
                },
                "cv": {
                    "gap": 0,
                    "horizon": 2,
                    "step_size": 2,
                    "n_windows": 3,
                    "max_train_size": None,
                    "overlap_eval_policy": "by_cutoff_mean",
                },
                "scheduler": {
                    "gpu_ids": [0, 1],
                    "max_concurrent_jobs": 2,
                    "worker_devices": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    config_path = tmp_path / "yaml" / "case.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "shared_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {},
                "cv": {},
                "scheduler": {},
                "residual": {"enabled": False},
                "jobs": [{"model": "Naive", "params": {}}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.config.runtime.random_seed == 11
    assert loaded.config.training.input_size == 8
    assert loaded.config.training.batch_size == 32
    assert loaded.config.training.loss == "mse"
    assert loaded.config.training.model_step_size == 6
    assert loaded.config.cv.n_windows == 3
    assert loaded.config.scheduler.gpu_ids == (0, 1)
    assert loaded.normalized_payload["shared_settings_path"].endswith("yaml/setting/setting.yaml")
    assert loaded.shared_settings_hash is not None


def test_load_app_config_rejects_duplicate_repo_shared_setting_paths(tmp_path: Path) -> None:
    (tmp_path / "yaml" / "setting").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml" / "setting" / "setting.yaml").write_text(
        yaml.safe_dump({"training": {"batch_size": 32}}, sort_keys=False),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    config_path = tmp_path / "yaml" / "case.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {"batch_size": 16},
                "cv": {"horizon": 1, "step_size": 1, "n_windows": 1},
                "scheduler": {"gpu_ids": [0]},
                "residual": {"enabled": False},
                "jobs": [{"model": "Naive", "params": {}}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.batch_size"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_duplicate_repo_shared_optimizer_setting_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "yaml" / "setting").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml" / "setting" / "setting.yaml").write_text(
        yaml.safe_dump(
            {"training": {"optimizer": {"name": "adamw", "kwargs": {}}}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    config_path = tmp_path / "yaml" / "case.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {"optimizer": {"name": "mars", "kwargs": {}}},
                "cv": {"horizon": 1, "step_size": 1, "n_windows": 1},
                "scheduler": {"gpu_ids": [0]},
                "residual": {"enabled": False},
                "jobs": [{"model": "Naive", "params": {}}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.optimizer"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_explicit_shared_settings_path_overrides_repo_default(
    tmp_path: Path,
) -> None:
    (tmp_path / "yaml" / "setting").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml" / "setting" / "setting.yaml").write_text(
        yaml.safe_dump({"runtime": {"random_seed": 11}}, sort_keys=False),
        encoding="utf-8",
    )
    explicit_setting_path = tmp_path / "custom-setting.yaml"
    explicit_setting_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"random_seed": 22},
                "training": {"batch_size": 48},
                "cv": {"horizon": 3, "step_size": 3, "n_windows": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    config_path = tmp_path / "yaml" / "case.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "explicit_shared_setting_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {},
                "cv": {},
                "scheduler": {"gpu_ids": [0]},
                "residual": {"enabled": False},
                "jobs": [{"model": "Naive", "params": {}}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_app_config(
        tmp_path,
        config_path=config_path,
        shared_settings_path="custom-setting.yaml",
    )

    assert loaded.config.runtime.random_seed == 22
    assert loaded.config.training.batch_size == 48
    assert loaded.config.cv.horizon == 3
    assert loaded.normalized_payload["shared_settings_path"].endswith(
        "custom-setting.yaml"
    )
    assert loaded.shared_settings_hash is not None


def test_load_app_config_rejects_missing_explicit_shared_settings_path(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    config_path = tmp_path / "yaml" / "case.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "cv": {"horizon": 1, "step_size": 1, "n_windows": 1},
                "scheduler": {"gpu_ids": [0]},
                "residual": {"enabled": False},
                "jobs": [{"model": "Naive", "params": {}}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing-setting.yaml"):
        load_app_config(
            tmp_path,
            config_path=config_path,
            shared_settings_path="missing-setting.yaml",
        )


def test_jobs_path_list_rejects_duplicate_route_stems(tmp_path: Path) -> None:
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    jobs_dir_a = tmp_path / "a"
    jobs_dir_b = tmp_path / "b"
    jobs_dir_a.mkdir()
    jobs_dir_b.mkdir()
    first_jobs = jobs_dir_a / "jobs.yaml"
    second_jobs = jobs_dir_b / "jobs.yaml"
    first_jobs.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    second_jobs.write_text(
        yaml.safe_dump(
            {"jobs": [{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "fanout.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "fanout_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {"max_steps": 1},
                "cv": {"n_windows": 1, "step_size": 1},
                "residual": {"enabled": False},
                "jobs": [str(first_jobs), str(second_jobs)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate stem: jobs"):
        load_app_config(tmp_path, config_path=config_path)


def test_default_output_root_appends_jobs_fanout_slug(tmp_path: Path) -> None:
    from app_config import loaded_config_for_jobs_fanout
    from runtime_support.runner import _default_output_root

    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    jobs_one = tmp_path / "jobs_1.yaml"
    jobs_two = tmp_path / "jobs_2.yaml"
    jobs_one.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    jobs_two.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    config_path = tmp_path / "fanout.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "fanout_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {"max_steps": 1},
                "cv": {"n_windows": 1, "step_size": 1},
                "residual": {"enabled": False},
                "jobs": [str(jobs_one), str(jobs_two)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_app_config(tmp_path, config_path=config_path)
    variant = loaded_config_for_jobs_fanout(tmp_path, loaded, loaded.jobs_fanout_specs[1])

    assert _default_output_root(tmp_path, variant) == (
        tmp_path / "runs" / f"{tmp_path.name}_fanout_case_jobs_2"
    )


def test_jobs_fanout_variant_preserves_repo_relative_dataset_resolution(
    tmp_path: Path,
) -> None:
    from app_config import loaded_config_for_jobs_fanout

    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "df.csv").write_text(
        "unique_id,dt,y\nA,2024-01-01,1\n",
        encoding="utf-8",
    )
    config_dir = tmp_path / "yaml"
    config_dir.mkdir(parents=True, exist_ok=True)
    jobs_one = tmp_path / "jobs_1.yaml"
    jobs_two = tmp_path / "jobs_2.yaml"
    jobs_one.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    jobs_two.write_text(
        yaml.safe_dump(
            {"jobs": [{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = config_dir / "fanout.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "fanout_case"},
                "dataset": {"path": "data/df.csv", "target_col": "y"},
                "training": {"max_steps": 1},
                "cv": {"n_windows": 1, "step_size": 1},
                "residual": {"enabled": False},
                "jobs": [str(jobs_one), str(jobs_two)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_app_config(tmp_path, config_path=config_path)
    variant = loaded_config_for_jobs_fanout(tmp_path, loaded, loaded.jobs_fanout_specs[1])

    assert variant.config.dataset.path == dataset_dir / "df.csv"


def test_jobs_fanout_variant_preserves_shared_settings_metadata(tmp_path: Path) -> None:
    from app_config import loaded_config_for_jobs_fanout

    (tmp_path / "yaml" / "setting").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml" / "setting" / "setting.yaml").write_text(
        yaml.safe_dump(
            {
                "runtime": {"random_seed": 13},
                "training": {
                    "input_size": 8,
                    "batch_size": 32,
                    "valid_batch_size": 64,
                    "val_check_steps": 5,
                    "early_stop_patience_steps": 5,
                    "loss": "mse",
                },
                "cv": {
                    "gap": 0,
                    "horizon": 1,
                    "step_size": 1,
                    "n_windows": 1,
                    "max_train_size": None,
                    "overlap_eval_policy": "by_cutoff_mean",
                },
                "scheduler": {
                    "gpu_ids": [0],
                    "max_concurrent_jobs": 2,
                    "worker_devices": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    jobs_one = tmp_path / "jobs_1.yaml"
    jobs_two = tmp_path / "jobs_2.yaml"
    jobs_one.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    jobs_two.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    config_path = tmp_path / "yaml" / "fanout.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "fanout_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {"max_steps": 10},
                "cv": {},
                "scheduler": {},
                "residual": {"enabled": False},
                "jobs": [str(jobs_one), str(jobs_two)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_app_config(tmp_path, config_path=config_path)
    variant = loaded_config_for_jobs_fanout(
        tmp_path, loaded, loaded.jobs_fanout_specs[1]
    )

    assert variant.shared_settings_path == loaded.shared_settings_path
    assert variant.shared_settings_hash == loaded.shared_settings_hash
    assert variant.normalized_payload["shared_settings_sha256"] == loaded.shared_settings_hash


def test_runtime_validate_only_fans_out_jobs_path_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import runtime_support.runner as runtime

    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    jobs_one = tmp_path / "jobs_1.yaml"
    jobs_two = tmp_path / "jobs_2.yaml"
    jobs_one.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    jobs_two.write_text(
        yaml.safe_dump(
            {"jobs": [{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "fanout.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "fanout_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {"max_steps": 1},
                "cv": {"n_windows": 1, "step_size": 1},
                "residual": {"enabled": False},
                "jobs": [str(jobs_one), str(jobs_two)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "residual" / "runtime.py"))

    assert runtime.main(["--config", str(config_path), "--validate-only"]) == 0

    result = json.loads(capsys.readouterr().out)
    run_roots = [Path(item["run_root"]) for item in result["fanout_runs"]]
    assert [item["jobs_route"] for item in result["fanout_runs"]] == ["jobs_1", "jobs_2"]
    assert [item["jobs"] for item in result["fanout_runs"]] == [
        ["Naive"],
        ["DummyUnivariate"],
    ]
    assert run_roots == [
        tmp_path / "runs" / f"{tmp_path.name}_fanout_case_jobs_1",
        tmp_path / "runs" / f"{tmp_path.name}_fanout_case_jobs_2",
    ]
    assert (run_roots[0] / "config" / "config.resolved.json").exists()
    assert (run_roots[1] / "config" / "config.resolved.json").exists()


def test_runtime_rejects_output_root_for_jobs_path_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import runtime_support.runner as runtime

    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text("unique_id,dt,y\nA,2024-01-01,1\n", encoding="utf-8")
    jobs_one = tmp_path / "jobs_1.yaml"
    jobs_two = tmp_path / "jobs_2.yaml"
    jobs_one.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    jobs_two.write_text(
        yaml.safe_dump(
            {"jobs": [{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "fanout.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "fanout_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {"max_steps": 1},
                "cv": {"n_windows": 1, "step_size": 1},
                "residual": {"enabled": False},
                "jobs": [str(jobs_one), str(jobs_two)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "residual" / "runtime.py"))

    with pytest.raises(SystemExit):
        runtime.main(
            [
                "--config",
                str(config_path),
                "--validate-only",
                "--output-root",
                str(tmp_path / "custom-root"),
            ]
        )

    assert (
        "--output-root is not supported when jobs is a list of job-file paths"
        in capsys.readouterr().err
    )

def test_runtime_execution_fans_out_jobs_path_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import runtime_support.runner as runtime

    dataset_path = tmp_path / "df.csv"
    dataset_path.write_text(
        "\n".join(
            ["unique_id,dt,y"]
            + [
                f"A,2024-01-{day:02d},{day}"
                for day in range(1, 21)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    jobs_one = tmp_path / "jobs_1.yaml"
    jobs_two = tmp_path / "jobs_2.yaml"
    jobs_one.write_text(
        yaml.safe_dump({"jobs": [{"model": "Naive", "params": {}}]}, sort_keys=False),
        encoding="utf-8",
    )
    jobs_two.write_text(
        yaml.safe_dump(
            {"jobs": [{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "fanout_exec.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task": {"name": "fanout_exec_case"},
                "dataset": {"path": str(dataset_path), "target_col": "y"},
                "training": {
                    "input_size": 2,
                    "season_length": 1,
                    "batch_size": 1,
                    "valid_batch_size": 1,
                    "windows_batch_size": 8,
                    "inference_windows_batch_size": 8,
                    "max_steps": 1,
                                },
                "cv": {"horizon": 1, "n_windows": 1, "step_size": 1},
                "residual": {"enabled": False},
                "jobs": [str(jobs_one), str(jobs_two)],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime, "__file__", str(tmp_path / "residual" / "runtime.py"))

    assert runtime.main(["--config", str(config_path)]) == 0

    captured = capsys.readouterr().out.strip().splitlines()
    results = [json.loads(line) for line in captured if line.lstrip().startswith("{")]
    assert len(results) == 2
    run_roots = [
        tmp_path / "runs" / f"{tmp_path.name}_fanout_exec_case_jobs_1",
        tmp_path / "runs" / f"{tmp_path.name}_fanout_exec_case_jobs_2",
    ]
    assert (run_roots[0] / "cv" / "Naive_forecasts.csv").exists()
    assert (run_roots[1] / "cv" / "DummyUnivariate_forecasts.csv").exists()


def test_case_yaml_training_mapping_matches_expected_across_all_files():
    for path in CASE_YAML_FILES:
        payload = _load_case_yaml(path)
        assert payload["training"] == EXPECTED_CASE_TRAINING


def test_feature_set_raw_yaml_defers_requested_training_cv_scheduler_keys_to_setting():
    raw_payload = _load_case_yaml_raw(REPO_ROOT / "yaml" / "experiment" / "feature_set" / "brentoil-case1.yaml")

    assert "training" not in raw_payload
    assert "cv" not in raw_payload
    assert "scheduler" not in raw_payload


def test_feature_set_case3_effective_scheduler_gpu_ids_comes_from_global_setting():
    payload = _load_case_yaml(REPO_ROOT / "yaml" / "experiment" / "feature_set" / "brentoil-case3.yaml")

    assert payload["scheduler"]["gpu_ids"] == [0, 1]


def test_hpt_n100_bs_raw_yaml_defers_requested_training_cv_scheduler_keys_to_setting():
    raw_payload = _load_case_yaml_raw(
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_bs" / "brentoil-case1.yaml"
    )

    assert "input_size" not in raw_payload["training"]
    assert "lr_scheduler" not in raw_payload["training"]
    assert "val_check_steps" not in raw_payload["training"]
    assert "early_stop_patience_steps" not in raw_payload["training"]
    assert "loss" not in raw_payload["training"]
    assert "cv" not in raw_payload
    assert "gpu_ids" not in raw_payload["scheduler"]


def test_hpt_n100_bs_effective_training_and_scheduler_come_from_global_setting():
    payload = _load_case_yaml(
        REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_bs" / "brentoil-case1.yaml"
    )

    assert payload["training"]["input_size"] == 64
    assert payload["training"]["lr_scheduler"]["max_lr"] == pytest.approx(0.001)
    assert payload["training"]["val_check_steps"] == 20
    assert payload["training"]["min_steps_before_early_stop"] == 500
    assert payload["training"]["early_stop_patience_steps"] == 3
    assert payload["training"]["loss"] == "mse"
    assert payload["cv"]["n_windows"] == 6
    assert payload["scheduler"]["gpu_ids"] == [0, 1]


@pytest.mark.parametrize("path", CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_jobs_keep_only_requested_models(path: Path):
    payload = _load_case_yaml(path)
    assert [job["model"] for job in payload["jobs"]] == EXPECTED_CASE_MODEL_LIST


@pytest.mark.parametrize("path", CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_learned_model_params_match_expected(path: Path):
    jobs = _case_jobs_by_model(path)

    for model_name, expected_params in EXPECTED_CASE_MODEL_PARAMS.items():
        if model_name == "Naive":
            continue
        assert jobs[model_name] == expected_params
        assert jobs[model_name]


@pytest.mark.parametrize("path", CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_naive_params_remain_empty(path: Path):
    jobs = _case_jobs_by_model(path)
    assert jobs["Naive"] == {}


@pytest.mark.parametrize("path", CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_feature_lists_and_targets_do_not_drift(path: Path):
    payload = _load_case_yaml(path)
    expected = EXPECTED_CASE_METADATA[path.name]

    assert payload["task"]["name"] == expected["task_name"]
    assert payload["dataset"]["target_col"] == expected["target_col"]
    assert payload["dataset"]["hist_exog_cols"] == expected["hist_exog_cols"]


@pytest.mark.parametrize("path", CASE_YAML_FILES, ids=lambda p: p.name)
def test_case_yaml_normalizes_to_fixed_modes_without_auto(path: Path):
    loaded = load_app_config(REPO_ROOT, config_path=path)

    for job in loaded.config.jobs:
        if job.model == "Naive":
            assert job.requested_mode == "baseline_fixed"
            assert job.validated_mode == "baseline_fixed"
        else:
            assert job.params == EXPECTED_CASE_MODEL_PARAMS[job.model]
            assert job.requested_mode == "learned_fixed"
            assert job.validated_mode == "learned_fixed"

    assert all(job.validated_mode != "learned_auto" for job in loaded.config.jobs)


def test_feature_set_residual_directory_contains_expected_case_files():
    actual = sorted(path.name for path in (REPO_ROOT / "yaml" / "experiment" / "feature_set_residual").glob("*.yaml"))

    assert actual == FEATURE_SET_RESIDUAL_EXPECTED_FILENAMES


@pytest.mark.parametrize("path", FEATURE_SET_RESIDUAL_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_residual_yaml_tracks_base_case_metadata(path: Path):
    payload = _load_case_yaml(path)
    base_payload = _load_case_yaml(REPO_ROOT / "yaml" / "experiment" / "feature_set" / path.name)

    assert payload["task"]["name"] == f'{base_payload["task"]["name"]}_residual'
    assert payload["dataset"]["target_col"] == base_payload["dataset"]["target_col"]
    assert payload["dataset"]["hist_exog_cols"] == base_payload["dataset"]["hist_exog_cols"]


@pytest.mark.parametrize("path", FEATURE_SET_RESIDUAL_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_residual_yaml_residual_block_matches_template(path: Path):
    payload = _load_case_yaml(path)
    residual = payload["residual"]

    assert residual["enabled"] is True
    assert residual["model"] == "xgboost"
    assert residual["target"] == "level"
    assert residual["params"] == EXPECTED_FEATURE_SET_RESIDUAL_PARAMS
    assert residual["features"]["include_base_prediction"] is True
    assert residual["features"]["include_horizon_step"] is True
    assert residual["features"]["include_date_features"] is False
    assert residual["features"]["lag_features"]["enabled"] is True
    assert residual["features"]["lag_features"]["sources"] == ["y_hat_base"]
    assert residual["features"]["lag_features"]["steps"] == EXPECTED_FEATURE_SET_RESIDUAL_LAG_STEPS
    assert residual["features"]["lag_features"]["transforms"] == ["raw"]
    assert residual["features"]["exog_sources"]["futr"] == []
    assert residual["features"]["exog_sources"]["static"] == []


@pytest.mark.parametrize("path", NEW_FEATURE_SET_RESIDUAL_YAML_FILES, ids=lambda p: p.name)
def test_new_feature_set_residual_yaml_exog_sources_hist_matches_base_dataset(path: Path):
    payload = _load_case_yaml(path)
    base_payload = _load_case_yaml(REPO_ROOT / "yaml" / "experiment" / "feature_set" / path.name)

    assert payload["residual"]["features"]["exog_sources"]["hist"] == base_payload["dataset"]["hist_exog_cols"]


@pytest.mark.parametrize("path", FEATURE_SET_RESIDUAL_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_residual_yaml_default_output_root_uses_parent_dir(path: Path):
    from runtime_support.runner import _default_output_root

    loaded = load_app_config(REPO_ROOT, config_path=path)
    base_payload = _load_case_yaml(REPO_ROOT / "yaml" / "experiment" / "feature_set" / path.name)
    expected_task_name = f'{base_payload["task"]["name"]}_residual'

    assert _default_output_root(REPO_ROOT, loaded) == (
        REPO_ROOT / "runs" / f"feature_set_residual_{expected_task_name}"
    )


def test_case_yaml_build_model_preserves_expected_fixed_params():
    loaded = load_app_config(
        REPO_ROOT,
        config_path=REPO_ROOT / "yaml" / "experiment" / "feature_set" / "brentoil-case1.yaml",
    )

    for job in loaded.config.jobs:
        if job.model == "Naive":
            continue
        model = build_model(
            loaded.config,
            job,
            n_series=1 if job.model == "iTransformer" else None,
        )
        for key, expected_value in EXPECTED_CASE_MODEL_PARAMS[job.model].items():
            actual_value = getattr(model, key, None)
            if actual_value is None and hasattr(model, "hparams"):
                actual_value = getattr(model.hparams, key, None)
            assert actual_value == expected_value


@pytest.mark.parametrize("path", HPT_CASE12_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_case12_training_keeps_only_fixed_controls(path: Path):
    payload = _load_case_yaml(path)
    assert payload["training"] == EXPECTED_HPT_CASE12_TRAINING


def test_feature_set_hpt_directory_contains_expected_case12_files():
    actual = sorted(
        path.name for path in (REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT").glob("*.yaml")
    )
    for expected in HPT_CASE12_YAML_FILES:
        assert expected.name in actual


@pytest.mark.parametrize("path", OPTUNA_CONFIG_YAML_FILES, ids=lambda p: p.name)
def test_optuna_config_yamls_pin_runtime_opt_n_trial(path: Path):
    payload = _load_case_yaml(path)
    assert payload["runtime"]["opt_n_trial"] == 20


@pytest.mark.parametrize("path", HPT_CASE12_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_case12_cv_and_jobs_follow_optuna_scope(path: Path):
    payload = _load_case_yaml(path)

    assert payload["cv"]["n_windows"] == 5
    assert [job["model"] for job in payload["jobs"]] == EXPECTED_HPT_CASE12_MODELS
    assert all(job["params"] == {} for job in payload["jobs"])


@pytest.mark.parametrize("path", HPT_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_jobs_keep_only_requested_models(path: Path):
    payload = _load_case_yaml(path)
    assert [job["model"] for job in payload["jobs"]] == EXPECTED_HPT_CASE12_MODELS
    assert all(job["params"] == {} for job in payload["jobs"])


@pytest.mark.parametrize("path", HPT_CASE12_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_case12_preserves_metadata(path: Path):
    payload = _load_case_yaml(path)
    expected = EXPECTED_CASE_METADATA[path.name]
    expected_task_name = path.stem.replace("-", "_") + "_HPT"

    assert payload["task"]["name"] == expected_task_name
    assert payload["dataset"]["target_col"] == expected["target_col"]
    assert payload["dataset"]["hist_exog_cols"] == expected["hist_exog_cols"]


def test_feature_set_hpt_n100_residual_directory_contains_expected_files():
    actual = sorted(
        path.name
        for path in (REPO_ROOT / "yaml" / "experiment" / "feature_set_HPT_n100_residual").glob("*.yaml")
    )
    assert actual == sorted(path.name for path in HPT_N100_YAML_FILES)


@pytest.mark.parametrize("path", HPT_N100_RESIDUAL_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_n100_residual_files_preserve_non_residual_sections(path: Path):
    payload = _load_case_yaml(path)
    source_payload = _load_case_yaml(_hpt_n100_source_path_for_residual(path))

    assert (
        _normalized_payload_without_task_and_residual(payload)
        == _normalized_payload_without_task_and_residual(source_payload)
    )
    assert payload["task"]["name"] == f'{source_payload["task"]["name"]}_residual'


@pytest.mark.parametrize("path", HPT_N100_RESIDUAL_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_n100_residual_files_apply_requested_residual_policy(path: Path):
    payload = _load_case_yaml(path)
    residual = payload["residual"]

    assert residual["enabled"] is True
    assert residual["model"] == "xgboost"
    assert residual["target"] == "level"
    assert residual["params"] == {}
    assert residual["features"]["include_base_prediction"] is True
    assert residual["features"]["include_horizon_step"] is True
    assert residual["features"]["include_date_features"] is False
    assert residual["features"]["lag_features"]["enabled"] is True
    assert residual["features"]["lag_features"]["sources"] == ["y_hat_base"]
    assert residual["features"]["lag_features"]["steps"] == [1, 2, 3, 4, 6, 12]
    assert residual["features"]["lag_features"]["transforms"] == ["raw"]
    assert residual["features"]["exog_sources"]["hist"] == payload["dataset"]["hist_exog_cols"]
    assert residual["features"]["exog_sources"]["futr"] == []
    assert residual["features"]["exog_sources"]["static"] == []


@pytest.mark.parametrize("path", HPT_N100_RESIDUAL_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_n100_residual_files_preserve_n100_knobs(path: Path):
    payload = _load_case_yaml(path)
    source_payload = _load_case_yaml(_hpt_n100_source_path_for_residual(path))

    assert payload["runtime"]["opt_n_trial"] == 100
    assert payload["runtime"]["opt_n_trial"] == source_payload["runtime"]["opt_n_trial"]
    assert payload["scheduler"]["parallelize_single_job_tuning"] is True
    assert (
        payload["scheduler"]["parallelize_single_job_tuning"]
        == source_payload["scheduler"]["parallelize_single_job_tuning"]
    )
    assert [job["model"] for job in payload["jobs"]] == EXPECTED_HPT_N100_MODELS
    assert payload["jobs"] == source_payload["jobs"]


@pytest.mark.parametrize("path", HPT_N100_RESIDUAL_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_n100_residual_files_normalize_to_auto_modes(path: Path):
    loaded = load_app_config(REPO_ROOT, config_path=path)

    assert loaded.config.training_search.requested_mode == "training_auto_requested"
    assert loaded.config.training_search.validated_mode == "training_auto"
    assert list(loaded.config.training_search.selected_search_params) == list(
        TRAINING_PARAM_REGISTRY
    )
    assert loaded.config.residual.model == "xgboost"
    assert loaded.config.residual.requested_mode == "residual_auto_requested"
    assert loaded.config.residual.validated_mode == "residual_auto"
    assert list(loaded.config.residual.selected_search_params) == list(
        SEARCH_SPACE_RESIDUAL["xgboost"]
    )

    for job in loaded.config.jobs:
        if job.model == "Naive":
            assert job.requested_mode == "baseline_fixed"
            assert job.validated_mode == "baseline_fixed"
            assert job.selected_search_params == ()
        else:
            assert job.params == {}
            assert job.requested_mode == "learned_auto_requested"
            assert job.validated_mode == "learned_auto"
            assert list(job.selected_search_params) == list(
                SEARCH_SPACE_MODELS[job.model]
            )


@pytest.mark.parametrize("path", RESIDUAL_AUTO_FIXTURE_FILES, ids=lambda p: p.name)
def test_optuna_residual_auto_fixtures_cover_all_supported_models(path: Path):
    payload = _load_case_yaml(path)
    loaded = load_app_config(REPO_ROOT, config_path=path)
    model_name = payload["residual"]["model"]

    assert model_name in EXPECTED_SUPPORTED_RESIDUAL_MODELS
    assert loaded.config.residual.model == model_name
    assert loaded.config.residual.requested_mode == "residual_auto_requested"
    assert loaded.config.residual.validated_mode == "residual_auto"
    assert list(loaded.config.residual.selected_search_params) == list(
        SEARCH_SPACE_RESIDUAL[model_name]
    )


@pytest.mark.parametrize("path", HPT_CASE12_YAML_FILES, ids=lambda p: p.name)
def test_feature_set_hpt_case12_normalizes_to_auto_training_and_learned_jobs(
    path: Path,
):
    loaded = load_app_config(REPO_ROOT, config_path=path)

    assert loaded.config.training_search.requested_mode == "training_auto_requested"
    assert loaded.config.training_search.validated_mode == "training_auto"
    assert list(loaded.config.training_search.selected_search_params) == list(
        TRAINING_PARAM_REGISTRY
    )
    assert loaded.config.residual.requested_mode == "residual_disabled"
    assert loaded.config.residual.validated_mode == "residual_disabled"
    assert loaded.config.residual.selected_search_params == ()

    for job in loaded.config.jobs:
        if job.model == "Naive":
            assert job.requested_mode == "baseline_fixed"
            assert job.validated_mode == "baseline_fixed"
            assert job.selected_search_params == ()
        else:
            assert job.params == {}
            assert job.requested_mode == "learned_auto_requested"
            assert job.validated_mode == "learned_auto"
            assert list(job.selected_search_params) == list(
                SEARCH_SPACE_MODELS[job.model]
            )


def test_repo_search_space_updates_requested_auto_selectors_only():
    assert list(SEARCH_SPACE_MODELS) == EXPECTED_REPO_AUTO_MODELS
    for model_name, expected in EXPECTED_REPO_AUTO_SELECTORS.items():
        assert SEARCH_SPACE_MODELS[model_name] == expected

    assert SEARCH_SPACE_TRAINING == EXPECTED_REPO_TRAINING_SELECTORS
    assert set(EXCLUDED_REPO_TRAINING_SELECTORS).isdisjoint(SEARCH_SPACE_TRAINING)
    assert set(SEARCH_SPACE_TRAINING_PER_MODEL) == set(EXPECTED_REPO_TRAINING_AUTO_MODELS)
    assert set(FIXED_TRAINING_KEYS).isdisjoint(SEARCH_SPACE_TRAINING)
    assert "step_size" not in SEARCH_SPACE_TRAINING
    assert "early_stop_patience_steps" not in SEARCH_SPACE_TRAINING
    assert "context_size" in SEARCH_SPACE_MODELS["LSTM"]
    assert "GRU" not in SEARCH_SPACE_MODELS
    assert "TFT" not in SEARCH_SPACE_MODELS


def _residual_target_panel(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows).assign(
        cutoff=lambda frame: pd.to_datetime(frame["cutoff"]),
        train_end_ds=lambda frame: pd.to_datetime(frame["train_end_ds"]),
        ds=lambda frame: pd.to_datetime(frame["ds"]),
    )


def test_build_residual_target_delta_matches_horizon_differences():
    from runtime_support.runner import build_residual_target

    panel = _residual_target_panel(
        [
            {
                "fold_idx": 0,
                "cutoff": "2020-01-08",
                "train_end_ds": "2020-01-08",
                "ds": "2020-01-15",
                "horizon_step": 1,
                "y": 11.0,
                "y_hat_base": 10.0,
            },
            {
                "fold_idx": 0,
                "cutoff": "2020-01-08",
                "train_end_ds": "2020-01-08",
                "ds": "2020-01-22",
                "horizon_step": 2,
                "y": 16.0,
                "y_hat_base": 12.0,
            },
            {
                "fold_idx": 0,
                "cutoff": "2020-01-08",
                "train_end_ds": "2020-01-08",
                "ds": "2020-01-29",
                "horizon_step": 3,
                "y": 22.0,
                "y_hat_base": 15.0,
            },
        ]
    )

    residual_target = build_residual_target(panel, "delta")

    assert residual_target.tolist() == [1.0, 3.0, 3.0]


def test_build_residual_target_delta_does_not_leak_across_fold_and_cutoff_groups():
    from runtime_support.runner import build_residual_target

    panel = _residual_target_panel(
        [
            {
                "fold_idx": 0,
                "cutoff": "2020-01-15",
                "train_end_ds": "2020-01-15",
                "ds": "2020-01-29",
                "horizon_step": 2,
                "y": 17.0,
                "y_hat_base": 14.0,
            },
            {
                "fold_idx": 0,
                "cutoff": "2020-01-08",
                "train_end_ds": "2020-01-08",
                "ds": "2020-01-15",
                "horizon_step": 1,
                "y": 11.0,
                "y_hat_base": 10.0,
            },
            {
                "fold_idx": 0,
                "cutoff": "2020-01-15",
                "train_end_ds": "2020-01-15",
                "ds": "2020-01-22",
                "horizon_step": 1,
                "y": 14.0,
                "y_hat_base": 12.0,
            },
            {
                "fold_idx": 0,
                "cutoff": "2020-01-08",
                "train_end_ds": "2020-01-08",
                "ds": "2020-01-22",
                "horizon_step": 2,
                "y": 16.0,
                "y_hat_base": 12.0,
            },
            {
                "fold_idx": 1,
                "cutoff": "2020-01-08",
                "train_end_ds": "2020-01-08",
                "ds": "2020-01-15",
                "horizon_step": 1,
                "y": 20.0,
                "y_hat_base": 18.0,
            },
            {
                "fold_idx": 1,
                "cutoff": "2020-01-08",
                "train_end_ds": "2020-01-08",
                "ds": "2020-01-22",
                "horizon_step": 2,
                "y": 25.0,
                "y_hat_base": 22.0,
            },
        ]
    )

    residual_target = build_residual_target(panel, "delta")

    assert residual_target.tolist() == [1.0, 1.0, 2.0, 3.0, 2.0, 1.0]


def test_runtime_diff_inverse_reconstruction_uses_anchor_and_cumsum():
    from runtime_support.runner import _FoldDiffContext, _restore_prediction_series

    restored = _restore_prediction_series(
        pd.Series([1.0, 2.0, 3.0]),
        _FoldDiffContext(target_col="target", anchor=10.0),
    )

    assert restored.tolist() == [11.0, 13.0, 16.0]


def test_runtime_target_diff_only_transforms_target_channel_in_multivariate_inputs(
    tmp_path: Path,
):
    payload = _payload()
    payload["runtime"]["transformations_target"] = "diff"
    payload["dataset"]["hist_exog_cols"] = ["hist_a"]
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["jobs"] = [
        {"model": "DummyMultivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target,hist_a,futr_a\n"
        "2020-01-01,1,10,100\n"
        "2020-01-08,2,11,101\n"
        "2020-01-15,3,12,102\n"
        "2020-01-22,4,13,103\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import runtime_support.runner as runtime

    train_df = pd.read_csv(tmp_path / "data.csv").iloc[:3].reset_index(drop=True)
    diff_context = runtime._build_fold_diff_context(loaded, train_df)
    transformed_train_df = runtime._transform_training_frame(train_df, diff_context)
    adapter_inputs = build_multivariate_inputs(
        transformed_train_df,
        loaded.config.jobs[0],
        dataset=loaded.config.dataset,
        dt_col=loaded.config.dataset.dt_col,
    )

    fit_frame = adapter_inputs.fit_df.sort_values(["ds", "unique_id"]).reset_index(
        drop=True
    )
    assert fit_frame["ds"].dt.strftime("%Y-%m-%d").unique().tolist() == [
        "2020-01-08",
        "2020-01-15",
    ]
    assert fit_frame.loc[fit_frame["unique_id"] == "target", "y"].tolist() == [1.0, 1.0]
    assert fit_frame.loc[fit_frame["unique_id"] == "hist_a", "y"].tolist() == [11, 12]
    assert fit_frame.loc[fit_frame["unique_id"] == "futr_a", "y"].tolist() == [101, 102]


def test_runtime_exog_diff_only_transforms_hist_exog_channel_in_multivariate_inputs(
    tmp_path: Path,
):
    payload = _payload()
    payload["runtime"]["transformations_exog"] = "diff"
    payload["dataset"]["hist_exog_cols"] = ["hist_a"]
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["jobs"] = [
        {"model": "DummyMultivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target,hist_a,futr_a\n"
        "2020-01-01,1,10,100\n"
        "2020-01-08,2,11,101\n"
        "2020-01-15,3,12,102\n"
        "2020-01-22,4,13,103\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import runtime_support.runner as runtime

    train_df = pd.read_csv(tmp_path / "data.csv").iloc[:3].reset_index(drop=True)
    diff_context = runtime._build_fold_diff_context(loaded, train_df)
    transformed_train_df = runtime._transform_training_frame(train_df, diff_context)
    adapter_inputs = build_multivariate_inputs(
        transformed_train_df,
        loaded.config.jobs[0],
        dataset=loaded.config.dataset,
        dt_col=loaded.config.dataset.dt_col,
    )

    fit_frame = adapter_inputs.fit_df.sort_values(["ds", "unique_id"]).reset_index(
        drop=True
    )
    assert fit_frame["ds"].dt.strftime("%Y-%m-%d").unique().tolist() == [
        "2020-01-08",
        "2020-01-15",
    ]
    assert fit_frame.loc[fit_frame["unique_id"] == "target", "y"].tolist() == [2, 3]
    assert fit_frame.loc[fit_frame["unique_id"] == "hist_a", "y"].tolist() == [1.0, 1.0]
    assert fit_frame.loc[fit_frame["unique_id"] == "futr_a", "y"].tolist() == [101, 102]


def test_runtime_target_and_exog_diff_transform_both_channels_in_multivariate_inputs(
    tmp_path: Path,
):
    payload = _payload()
    payload["runtime"]["transformations_target"] = "diff"
    payload["runtime"]["transformations_exog"] = "diff"
    payload["dataset"]["hist_exog_cols"] = ["hist_a"]
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["jobs"] = [
        {"model": "DummyMultivariate", "params": {"start_padding_enabled": True}}
    ]
    data = (
        "dt,target,hist_a,futr_a\n"
        "2020-01-01,1,10,100\n"
        "2020-01-08,2,11,101\n"
        "2020-01-15,3,12,102\n"
        "2020-01-22,4,13,103\n"
    )
    (tmp_path / "data.csv").write_text(data, encoding="utf-8")
    loaded = load_app_config(
        tmp_path, config_path=_write_config(tmp_path, payload, ".yaml")
    )

    import runtime_support.runner as runtime

    train_df = pd.read_csv(tmp_path / "data.csv").iloc[:3].reset_index(drop=True)
    diff_context = runtime._build_fold_diff_context(loaded, train_df)
    transformed_train_df = runtime._transform_training_frame(train_df, diff_context)
    adapter_inputs = build_multivariate_inputs(
        transformed_train_df,
        loaded.config.jobs[0],
        dataset=loaded.config.dataset,
        dt_col=loaded.config.dataset.dt_col,
    )

    fit_frame = adapter_inputs.fit_df.sort_values(["ds", "unique_id"]).reset_index(
        drop=True
    )
    assert fit_frame.loc[fit_frame["unique_id"] == "target", "y"].tolist() == [1.0, 1.0]
    assert fit_frame.loc[fit_frame["unique_id"] == "hist_a", "y"].tolist() == [1.0, 1.0]
    assert fit_frame.loc[fit_frame["unique_id"] == "futr_a", "y"].tolist() == [101, 102]


def test_apply_residual_plugin_writes_residual_target_to_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    class _ZeroResidualPlugin(ResidualPlugin):
        name = "zero"

        def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
            return None

        def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
            return panel_df.copy().assign(residual_hat=0.0)

        def metadata(self) -> dict[str, object]:
            return {"plugin": self.name}

    monkeypatch.setattr(
        runtime,
        "build_residual_plugin",
        lambda _config: _ZeroResidualPlugin(),
    )
    payload = _payload()
    payload["residual"]["target"] = "delta"
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    job = loaded.config.jobs[0]
    run_root = tmp_path / "run"
    manifest_path = run_root / "manifest" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": [{"model": job.model}], "residual": {}}, indent=2),
        encoding="utf-8",
    )
    fold_payloads = [
        {
            "fold_idx": 0,
            "backcast_panel": _residual_target_panel(
                [
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "backcast_train",
                        "unique_id": "target",
                        "cutoff": "2020-01-08",
                        "train_end_ds": "2020-01-08",
                        "ds": "2020-01-15",
                        "horizon_step": 1,
                        "y_hat_base": 10.0,
                        "y": 11.0,
                        "residual_target": 1.0,
                    }
                ]
            ),
            "eval_panel": _residual_target_panel(
                [
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "fold_eval",
                        "unique_id": "target",
                        "cutoff": "2020-01-15",
                        "train_end_ds": "2020-01-15",
                        "ds": "2020-01-22",
                        "horizon_step": 1,
                        "y_hat_base": 12.0,
                        "y": 13.0,
                        "residual_target": 1.0,
                    }
                ]
            ),
            "base_summary": {"fold_idx": 0},
        }
    ]

    runtime._apply_residual_plugin(
        loaded,
        job,
        run_root,
        fold_payloads,
        manifest_path=manifest_path,
    )

    diagnostics = pd.read_json(
        run_root / "residual" / job.model / "diagnostics.json",
        typ="series",
    )

    assert diagnostics["residual.target"] == "delta"


def test_runtime_main_joint_auto_mode_records_residual_best_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    payload = _payload()
    payload["runtime"]["opt_n_trial"] = 2
    payload["cv"].update({"horizon": 1, "step_size": 1, "n_windows": 1, "gap": 0})
    payload["training"].update({"input_size": 1, "max_steps": 1, "val_size": 1})
    payload["dataset"]["hist_exog_cols"] = []
    payload["jobs"] = [{"model": "TFT", "params": {}}]
    payload["residual"] = {"enabled": True, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target\n2020-01-01,1\n2020-01-08,2\n2020-01-15,3\n2020-01-22,4\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, payload, ".yaml")
    _write_search_space(
        tmp_path,
        {
            "models": {"TFT": ["hidden_size"]},
            "training": [],
            "residual": {"xgboost": ["n_estimators"]},
        },
    )

    captured: dict[str, Any] = {}

    def _fake_fit_and_predict_fold(
        loaded,
        job,
        *,
        source_df,
        freq,
        train_idx,
        test_idx,
        params_override=None,
        training_override=None,
    ):
        predictions = pd.DataFrame(
            {
                "unique_id": [loaded.config.dataset.target_col],
                "ds": pd.Series(["2020-01-22"]),
                job.model: [1.0],
            }
        )
        actuals = pd.Series([1.0])
        return (
            predictions,
            actuals,
            pd.Timestamp("2020-01-15"),
            source_df.iloc[train_idx].reset_index(drop=True),
            SimpleNamespace(),
        )

    def _fake_score_main_trial_with_residual(
        loaded,
        job,
        *,
        residual_params,
        **_kwargs,
    ):
        return float(residual_params["n_estimators"])

    def _fake_suggest_prefixed_residual_params(loaded, trial):
        return {"n_estimators": 8 + trial.number}

    def _fake_build_fold_panel(*_args, **_kwargs):
        return _residual_target_panel(
            [
                {
                    "model_name": "TFT",
                    "fold_idx": 0,
                    "panel_split": "fold_eval",
                    "unique_id": "target",
                    "cutoff": "2020-01-15",
                    "train_end_ds": "2020-01-15",
                    "ds": "2020-01-22",
                    "horizon_step": 1,
                    "y_hat_base": 1.0,
                    "y": 1.0,
                    "residual_target": 0.0,
                }
            ]
        )

    def _fake_apply_residual_plugin(*args, **kwargs):
        captured["residual_params_override"] = kwargs.get("residual_params_override")
        captured["residual_study_summary_override"] = kwargs.get(
            "residual_study_summary_override"
        )
        return None

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")
    monkeypatch.setattr(
        runtime,
        "load_app_config",
        lambda _repo_root, **kwargs: load_app_config(tmp_path, **kwargs),
    )
    monkeypatch.setattr(runtime, "_fit_and_predict_fold", _fake_fit_and_predict_fold)
    monkeypatch.setattr(
        runtime,
        "_score_main_trial_with_residual",
        _fake_score_main_trial_with_residual,
    )
    monkeypatch.setattr(
        runtime,
        "_suggest_prefixed_residual_params",
        _fake_suggest_prefixed_residual_params,
    )
    monkeypatch.setattr(runtime, "_build_fold_backcast_panel", _fake_build_fold_panel)
    monkeypatch.setattr(runtime, "_build_fold_eval_panel", _fake_build_fold_panel)
    monkeypatch.setattr(
        runtime,
        "suggest_model_params",
        lambda *_args, **_kwargs: {"hidden_size": 32},
    )
    monkeypatch.setattr(
        runtime,
        "suggest_training_params",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(runtime, "_apply_residual_plugin", _fake_apply_residual_plugin)
    monkeypatch.setattr(runtime, "_should_build_summary_artifacts", lambda: False)

    output_root = tmp_path / "run_joint_auto"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--jobs",
            "TFT",
            "--output-root",
            str(output_root),
        ]
    )

    summary = json.loads(
        (output_root / "models" / "TFT" / "optuna_study_summary.json").read_text()
    )
    fit_summary = json.loads(
        (output_root / "models" / "TFT" / "fit_summary.json").read_text()
    )

    assert code == 0
    assert summary["objective_stage"] == "tuning_pre_replay_corrected_predictions"
    assert summary["objective_metric"] == "mean_fold_mape"
    assert summary["best_residual_params"] == {"n_estimators": 8}
    assert fit_summary["tuning_objective_metric"] == (
        "mean_fold_mape_on_corrected_predictions"
    )
    assert captured["residual_params_override"] == {"n_estimators": 8}
    assert (
        captured["residual_study_summary_override"]["objective_stage"]
        == "tuning_pre_replay_joint_corrected_predictions"
    )


def test_apply_residual_plugin_uses_override_params_without_running_residual_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    class _ZeroResidualPlugin(ResidualPlugin):
        name = "zero"

        def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
            return None

        def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
            return panel_df.copy().assign(residual_hat=0.0)

        def metadata(self) -> dict[str, object]:
            return {"plugin": self.name}

    monkeypatch.setattr(
        runtime,
        "build_residual_plugin",
        lambda _config: _ZeroResidualPlugin(),
    )
    monkeypatch.setattr(
        runtime,
        "_score_residual_params",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("serial residual study should be skipped")
        ),
    )

    payload = _payload()
    payload["residual"] = {"enabled": True, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n",
        encoding="utf-8",
    )
    _write_search_space(tmp_path)
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    job = loaded.config.jobs[0]
    run_root = tmp_path / "run_override"
    manifest_path = run_root / "manifest" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": [{"model": job.model}], "residual": {}}, indent=2),
        encoding="utf-8",
    )
    panel = _residual_target_panel(
        [
            {
                "model_name": job.model,
                "fold_idx": 0,
                "panel_split": "fold_eval",
                "unique_id": "target",
                "cutoff": "2020-01-15",
                "train_end_ds": "2020-01-15",
                "ds": "2020-01-22",
                "horizon_step": 1,
                "y_hat_base": 12.0,
                "y": 13.0,
                "residual_target": 1.0,
            }
        ]
    )
    fold_payloads = [
        {
            "fold_idx": 0,
            "trial_dir": run_root / "residual" / job.model / "_optuna_trial",
            "backcast_panel": panel.copy(),
            "eval_panel": panel.copy(),
            "base_summary": {"fold_idx": 0},
        }
    ]

    runtime._apply_residual_plugin(
        loaded,
        job,
        run_root,
        fold_payloads,
        manifest_path=manifest_path,
        residual_params_override={"n_estimators": 17},
        residual_study_summary_override={
            "objective_stage": "tuning_pre_replay_joint_corrected_predictions"
        },
    )

    assert json.loads(
        (run_root / "residual" / job.model / "best_params.json").read_text()
    ) == {
        "n_estimators": 17,
        "max_depth": 3,
            "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
    assert json.loads(
        (run_root / "residual" / job.model / "optuna_study_summary.json").read_text()
    )["objective_stage"] == "tuning_pre_replay_joint_corrected_predictions"


def test_apply_residual_plugin_prefers_yaml_opt_n_trial_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    class _ZeroResidualPlugin(ResidualPlugin):
        name = "zero"

        def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
            return None

        def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
            return panel_df.copy().assign(residual_hat=0.0)

        def metadata(self) -> dict[str, object]:
            return {"plugin": self.name}

    monkeypatch.setattr(
        runtime,
        "build_residual_plugin",
        lambda _config: _ZeroResidualPlugin(),
    )
    monkeypatch.setattr(
        runtime,
        "_score_residual_params",
        lambda *_args, **_kwargs: 1.0,
    )
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_NUM_TRIALS", "1")
    monkeypatch.setenv("NEURALFORECAST_OPTUNA_SEED", "7")

    payload = _payload()
    payload["runtime"]["opt_n_trial"] = 2
    payload["residual"] = {"enabled": True, "model": "xgboost", "params": {}}
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a\n2020-01-01,1,2\n",
        encoding="utf-8",
    )
    _write_search_space(tmp_path)
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    job = loaded.config.jobs[0]
    run_root = tmp_path / "run"
    manifest_path = run_root / "manifest" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": [{"model": job.model}], "residual": {}}, indent=2),
        encoding="utf-8",
    )
    fold_payloads = [
        {
            "fold_idx": 0,
            "trial_dir": run_root / "residual" / job.model / "_optuna_trial",
            "backcast_panel": _residual_target_panel(
                [
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "backcast_train",
                        "unique_id": "target",
                        "cutoff": "2020-01-15",
                        "train_end_ds": "2020-01-15",
                        "ds": "2020-01-22",
                        "horizon_step": 1,
                        "y_hat_base": 12.0,
                        "y": 13.0,
                        "residual_target": 1.0,
                    }
                ]
            ),
            "eval_panel": _residual_target_panel(
                [
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "fold_eval",
                        "unique_id": "target",
                        "cutoff": "2020-01-15",
                        "train_end_ds": "2020-01-15",
                        "ds": "2020-01-22",
                        "horizon_step": 1,
                        "y_hat_base": 12.0,
                        "y": 13.0,
                        "residual_target": 1.0,
                    }
                ]
            ),
            "base_summary": {"fold_idx": 0},
        }
    ]

    runtime._apply_residual_plugin(
        loaded,
        job,
        run_root,
        fold_payloads,
        manifest_path=manifest_path,
    )

    first_summary = json.loads(
        (run_root / "residual" / job.model / "optuna_study_summary.json").read_text()
    )
    runtime._apply_residual_plugin(
        loaded,
        job,
        run_root,
        fold_payloads,
        manifest_path=manifest_path,
    )

    summary = json.loads(
        (run_root / "residual" / job.model / "optuna_study_summary.json").read_text()
    )
    assert first_summary["trial_count"] == 2
    assert first_summary["remaining_trial_count"] == 2
    assert summary["trial_count"] == 2
    assert summary["existing_finished_trial_count_before_optimize"] == 2
    assert summary["remaining_trial_count"] == 0


def test_apply_residual_plugin_writes_feature_visibility_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import runtime_support.runner as runtime

    class _ZeroResidualPlugin(ResidualPlugin):
        name = "zero"

        def fit(self, panel_df: pd.DataFrame, context: ResidualContext) -> None:
            return None

        def predict(self, panel_df: pd.DataFrame) -> pd.DataFrame:
            return panel_df.copy().assign(residual_hat=0.0)

        def metadata(self) -> dict[str, object]:
            return {"plugin": self.name}

    monkeypatch.setattr(
        runtime,
        "build_residual_plugin",
        lambda _config: _ZeroResidualPlugin(),
    )

    payload = _payload()
    payload["dataset"]["futr_exog_cols"] = ["futr_a"]
    payload["dataset"]["static_exog_cols"] = ["static_a"]
    payload["residual"]["features"] = {
        "include_date_features": True,
        "exog_sources": {
            "hist": ["hist_a"],
            "futr": ["futr_a"],
            "static": ["static_a"],
        },
        "lag_features": {
            "enabled": True,
            "sources": ["y_hat_base", "hist_a", "futr_a"],
            "steps": [1],
            "transforms": ["raw"],
        },
    }
    (tmp_path / "data.csv").write_text(
        "dt,target,hist_a,futr_a,static_a\n"
        "2020-01-01,1,2,20,200\n"
        "2020-01-08,2,3,30,300\n",
        encoding="utf-8",
    )
    loaded = load_app_config(
        tmp_path,
        config_path=_write_config(tmp_path, payload, ".yaml"),
    )
    job = loaded.config.jobs[0]
    run_root = tmp_path / "run"
    manifest_path = run_root / "manifest" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"jobs": [{"model": job.model}], "residual": {}}, indent=2),
        encoding="utf-8",
    )
    fold_payloads = [
        {
            "fold_idx": 0,
            "backcast_panel": _residual_target_panel(
                [
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "backcast_train",
                        "unique_id": "target",
                        "cutoff": "2020-01-08",
                        "train_end_ds": "2020-01-08",
                        "ds": "2020-01-15",
                        "horizon_step": 1,
                        "y_hat_base": 10.0,
                        "y": 11.0,
                        "residual_target": 1.0,
                        "hist_a_lag_1": 2.0,
                        "futr_a": 20.0,
                        "static_a": 200.0,
                    },
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "backcast_train",
                        "unique_id": "target",
                        "cutoff": "2020-01-08",
                        "train_end_ds": "2020-01-08",
                        "ds": "2020-01-22",
                        "horizon_step": 2,
                        "y_hat_base": 12.0,
                        "y": 13.0,
                        "residual_target": 1.0,
                        "hist_a_lag_1": 3.0,
                        "futr_a": 30.0,
                        "static_a": 200.0,
                    },
                ]
            ),
            "eval_panel": _residual_target_panel(
                [
                    {
                        "model_name": job.model,
                        "fold_idx": 0,
                        "panel_split": "fold_eval",
                        "unique_id": "target",
                        "cutoff": "2020-01-22",
                        "train_end_ds": "2020-01-22",
                        "ds": "2020-01-29",
                        "horizon_step": 1,
                        "y_hat_base": 14.0,
                        "y": 15.0,
                        "residual_target": 1.0,
                        "hist_a_lag_1": 4.0,
                        "futr_a": 40.0,
                        "static_a": 200.0,
                    }
                ]
            ),
            "base_summary": {"fold_idx": 0},
        }
    ]

    runtime._apply_residual_plugin(
        loaded,
        job,
        run_root,
        fold_payloads,
        manifest_path=manifest_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    written_backcast = pd.read_csv(
        run_root / "residual" / job.model / "folds" / "fold_000" / "backcast_panel.csv"
    )
    written_corrected = pd.read_csv(
        run_root / "residual" / job.model / "folds" / "fold_000" / "corrected_eval.csv"
    )
    plugin_metadata = json.loads(
        (run_root / "residual" / job.model / "plugin_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = json.loads(
        (run_root / "residual" / job.model / "diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    expected_columns = [
        "horizon_step",
        "y_hat_base",
        "cutoff_day",
        "ds_day",
        "y_hat_base_lag_1",
        "futr_a_lag_1",
        "hist_a_lag_1",
        "futr_a",
        "static_a",
    ]

    assert manifest["residual"]["feature_policy"]["include_date_features"] is True
    assert manifest["residual"]["active_feature_columns"] == expected_columns
    assert "hist_a_lag_1" in written_backcast.columns
    assert "hist_a_lag_1" in written_corrected.columns
    assert "hist_a" not in written_backcast.columns
    assert "hist_a" not in written_corrected.columns
    assert plugin_metadata["0"]["active_feature_columns"] == expected_columns
    assert plugin_metadata["0"]["feature_policy"]["lag_features"]["steps"] == [1]
    assert diagnostics["active_feature_columns"] == expected_columns
    assert diagnostics["residual.feature_policy"]["exog_sources"]["static"] == [
        "static_a"
    ]
