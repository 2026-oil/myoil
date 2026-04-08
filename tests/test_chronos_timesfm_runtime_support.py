from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import app_config
import plugin_contracts.stage_registry as stage_registry
from app_config import JobConfig, load_app_config
from runtime_support.forecast_models import MODEL_CLASSES, build_model, validate_job


def _config_payload() -> dict:
    return {
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
        },
        "cv": {"horizon": 2, "step_size": 1, "n_windows": 1, "gap": 0},
        "scheduler": {
            "gpu_ids": [0],
            "max_concurrent_jobs": 1,
            "worker_devices": 1,
        },
        "jobs": [],
    }


@pytest.mark.parametrize(
    ("model_name", "params", "class_name"),
    [
        ("Chronos2", {"model_id": "amazon/chronos-2"}, "Chronos2"),
        (
            "TimesFM2_5",
            {"model_id": "google/timesfm-2.5-200m-transformers"},
            "TimesFM2_5",
        ),
    ],
)
def test_chronos_timesfm_models_are_runtime_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    params: dict,
    class_name: str,
) -> None:
    assert model_name in MODEL_CLASSES
    assert MODEL_CLASSES[model_name].__name__ == class_name

    payload = _config_payload()
    payload["jobs"] = [{"model": model_name, "params": params}]
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
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(app_config, "_ensure_plugins_loaded", lambda: None)
    monkeypatch.setattr(stage_registry, "_ensure_plugins_loaded", lambda: None)
    loaded = load_app_config(tmp_path, config_path=config_path)
    job = loaded.config.jobs[0]

    caps = validate_job(job)
    assert caps.name == model_name
    assert caps.supports_hist_exog is False
    assert caps.supports_futr_exog is False
    assert caps.supports_stat_exog is False

    model = build_model(loaded.config, job)
    assert model.__class__.__name__ == class_name
    assert repr(model) == model_name
    assert model.loss.outputsize_multiplier == 1


def test_chronos_timesfm_validate_job_accepts_fixed_mode_contract() -> None:
    for model_name in ("Chronos2", "TimesFM2_5"):
        job = JobConfig(
            model=model_name,
            params={"model_id": "dummy"},
            requested_mode="learned_fixed",
            validated_mode="learned_fixed",
            selected_search_params=(),
        )
        caps = validate_job(job)
        assert caps.name == model_name
