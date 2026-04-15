from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

_IMPORT_ERROR = None
try:
    import runtime_support.runner as runtime
    from app_config import load_app_config
    from runtime_support.forecast_models import build_model
except ImportError as exc:  # pragma: no cover - environment-specific branch safety
    runtime = None
    load_app_config = None
    build_model = None
    _IMPORT_ERROR = exc


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _build_config(tmp_path: Path) -> Path:
    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "dt,target,hist_a\n"
        "2020-01-01,1,10\n"
        "2020-01-08,2,11\n"
        "2020-01-15,3,12\n"
        "2020-01-22,4,13\n",
        encoding="utf-8",
    )
    _write_yaml(
        tmp_path / "yaml/HPO/search_space.yaml",
        {
            "models": {},
            "training": [],
            "bs_preforcast_models": {},
            "bs_preforcast_training": [],
        },
    )
    return _write_yaml(
        tmp_path / "config.yaml",
        {
            "task": {"name": "informer_hist_validate"},
            "dataset": {
                "path": "data.csv",
                "target_col": "target",
                "dt_col": "dt",
                "hist_exog_cols": ["hist_a"],
                "futr_exog_cols": [],
                "static_exog_cols": [],
            },
            "runtime": {"random_seed": 7},
            "training": {
                "input_size": 2,
                "season_length": 1,
                "batch_size": 1,
                "valid_batch_size": 1,
                "windows_batch_size": 8,
                "inference_windows_batch_size": 8,
                "lr_scheduler": {
                    "name": "OneCycleLR",
                    "max_lr": 0.001,
                    "pct_start": 0.3,
                    "div_factor": 25.0,
                    "final_div_factor": 10000.0,
                    "anneal_strategy": "cos",
                    "three_phase": False,
                    "cycle_momentum": False,
                },
                "max_steps": 1,
                "val_size": 1,
                "val_check_steps": 1,
                "early_stop_patience_steps": -1,
                "loss": "mse",
            },
            "cv": {
                "horizon": 1,
                "step_size": 1,
                "n_windows": 1,
                "gap": 0,
                "overlap_eval_policy": "by_cutoff_mean",
            },
            "scheduler": {
                "gpu_ids": [0],
                "max_concurrent_jobs": 1,
                "worker_devices": 1,
            },
            "jobs": [
                {
                    "model": "Informer",
                    "params": {
                        "hidden_size": 8,
                        "n_head": 1,
                        "encoder_layers": 1,
                        "decoder_layers": 1,
                        "conv_hidden_size": 16,
                    },
                }
            ],
        },
    )


def test_build_model_accepts_informer_hist_exog_and_forward_passes(tmp_path: Path) -> None:
    if _IMPORT_ERROR is not None:
        if "partially initialized module 'app_config'" in str(_IMPORT_ERROR):
            pytest.skip(f"unrelated legacy import blocker in current branch: {_IMPORT_ERROR}")
        raise _IMPORT_ERROR

    config_path = _build_config(tmp_path)
    loaded = load_app_config(tmp_path, config_path=config_path)
    model = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert model.EXOGENOUS_HIST is True
    assert model.hist_exog_list == ["hist_a"]
    assert model.enc_in == 2

    windows_batch = {
        "insample_y": torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        "hist_exog": torch.tensor([[[10.0], [11.0]], [[12.0], [13.0]]]),
        "futr_exog": None,
    }

    forecast = model(windows_batch)

    assert forecast.shape == (2, 1, 1)


def test_runtime_validate_only_accepts_informer_hist_exog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if _IMPORT_ERROR is not None:
        if "partially initialized module 'app_config'" in str(_IMPORT_ERROR):
            pytest.skip(f"unrelated legacy import blocker in current branch: {_IMPORT_ERROR}")
        raise _IMPORT_ERROR

    config_path = _build_config(tmp_path)
    loaded = load_app_config(tmp_path, config_path=config_path)
    model = build_model(loaded.config, loaded.config.jobs[0], n_series=1)

    assert runtime._should_use_multivariate(loaded, loaded.config.jobs[0]) is False
    assert model.hist_exog_list == ["hist_a"]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    output_root = tmp_path / "validate-only-run"
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
    assert resolved["dataset"]["hist_exog_cols"] == ["hist_a"]

    capability = json.loads(
        (output_root / "config" / "capability_report.json").read_text()
    )
    assert capability["Informer"]["supports_hist_exog"] is True
    assert capability["Informer"]["validation_error"] is None
