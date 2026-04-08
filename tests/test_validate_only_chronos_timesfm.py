from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import app_config
import plugin_contracts.stage_registry as stage_registry
import runtime_support.runner as runtime


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("Chronos2", {"model_id": "amazon/chronos-2"}),
        (
            "TimesFM2_5",
            {"model_id": "google/timesfm-2.5-200m-transformers"},
        ),
    ],
)
def test_runtime_validate_only_accepts_chronos_timesfm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    params: dict,
):
    (tmp_path / "data.csv").write_text(
        "dt,target\n"
        "2020-01-01,1\n"
        "2020-01-08,2\n"
        "2020-01-15,3\n"
        "2020-01-22,4\n"
        "2020-01-29,5\n"
        "2020-02-05,6\n"
        "2020-02-12,7\n"
        "2020-02-19,8\n"
        "2020-02-26,9\n"
        "2020-03-04,10\n",
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
            "task": {"name": f"validate_only_{model_name.lower()}"},
            "dataset": {
                "path": "data.csv",
                "target_col": "target",
                "dt_col": "dt",
                "hist_exog_cols": [],
                "futr_exog_cols": [],
                "static_exog_cols": [],
            },
            "runtime": {"random_seed": 7},
            "training": {
                "input_size": 8,
                "batch_size": 1,
                "valid_batch_size": 1,
                "windows_batch_size": 8,
                "inference_windows_batch_size": 8,
                "max_steps": 1,
                "val_size": 2,
                "val_check_steps": 1,
                "early_stop_patience_steps": -1,
                "loss": "mse",
                "accelerator": "cpu",
            },
            "cv": {
                "horizon": 2,
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
            "jobs": [{"model": model_name, "params": params}],
        },
    )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(app_config, "_ensure_plugins_loaded", lambda: None)
    monkeypatch.setattr(stage_registry, "_ensure_plugins_loaded", lambda: None)

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
            "model": model_name,
            "requested_mode": "learned_fixed",
            "validated_mode": "learned_fixed",
            "selected_search_params": [],
        }
    ]
