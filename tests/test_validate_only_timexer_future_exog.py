from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import runtime_support.runner as runtime


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_runtime_validate_only_accepts_timexer_future_exog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "dt,target,futr_a\n"
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
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        {
            "dataset": {
                "path": "data.csv",
                "target_col": "target",
                "dt_col": "dt",
                "hist_exog_cols": [],
                "futr_exog_cols": ["futr_a"],
                "static_exog_cols": [],
            },
            "runtime": {"random_seed": 7},
            "training": {
                "input_size": 1,
                "season_length": 1,
                "batch_size": 1,
                "valid_batch_size": 1,
                "windows_batch_size": 8,
                "inference_windows_batch_size": 8,
                "lr_scheduler": {"name": "OneCycleLR", "max_lr": 0.001, "pct_start": 0.3, "div_factor": 25.0, "final_div_factor": 10000.0, "anneal_strategy": "cos", "three_phase": False, "cycle_momentum": False},
                "max_steps": 1,
                "val_size": 1,
                "val_check_steps": 1,
                "early_stop_patience_steps": -1,
                "loss": "mse",
            },
            "cv": {
                "horizon": 1,
                "step_size": 1,
                "n_windows": 2,
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
                    "model": "TimeXer",
                    "params": {
                        "patch_len": 1,
                        "hidden_size": 8,
                        "n_heads": 1,
                        "e_layers": 1,
                        "d_ff": 16,
                    },
                }
            ],
        },
    )

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
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert manifest["jobs"] == [
        {
            "model": "TimeXer",
            "requested_mode": "learned_fixed",
            "validated_mode": "learned_fixed",
            "selected_search_params": [],
        }
    ]
