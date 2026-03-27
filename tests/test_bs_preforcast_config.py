from __future__ import annotations

from pathlib import Path

import yaml

import residual.config as residual_config
from bs_preforcast.config import BsPreforcastConfig, BsPreforcastStageLoadedConfig
from residual.config import load_app_config


def _base_payload(data_path: Path) -> dict[str, object]:
    return {
        "dataset": {
            "path": str(data_path),
            "target_col": "target",
            "dt_col": "dt",
            "hist_exog_cols": ["hist_a"],
            "futr_exog_cols": [],
            "static_exog_cols": [],
        },
        "runtime": {"random_seed": 1},
        "training": {
            "input_size": 2,
            "season_length": 52,
            "batch_size": 16,
            "valid_batch_size": 16,
            "windows_batch_size": 16,
            "inference_windows_batch_size": 16,
            "learning_rate": 0.001,
            "max_steps": 1,
            "val_size": 1,
            "val_check_steps": 1,
            "early_stop_patience_steps": -1,
            "num_lr_decays": -1,
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
            "parallelize_single_job_tuning": False,
        },
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "jobs": [{"model": "Naive", "params": {}}],
    }


def _main_bs_preforcast(*, config_path: str) -> dict[str, object]:
    return {"enabled": True, "config_path": config_path}


def _with_linked_bs_preforcast(
    payload: dict[str, object],
    *,
    using_futr_exog: bool = False,
    target_columns: tuple[str, ...] = ("bs_a",),
    multivariable: bool = False,
) -> dict[str, object]:
    linked = dict(payload)
    linked["bs_preforcast"] = {
        "using_futr_exog": using_futr_exog,
        "target_columns": list(target_columns),
        "task": {"multivariable": multivariable},
    }
    return linked


def _linked_bs_preforcast(
    *,
    using_futr_exog: bool = False,
    target_columns: tuple[str, ...] = ("bs_a",),
    multivariable: bool = False,
) -> dict[str, object]:
    return {
        "bs_preforcast": {
            "using_futr_exog": using_futr_exog,
            "target_columns": list(target_columns),
            "task": {"multivariable": multivariable},
        }
    }


def test_residual_config_uses_bs_preforcast_authoritative_types() -> None:
    assert residual_config.BsPreforcastConfig is BsPreforcastConfig
    assert residual_config.BsPreforcastStageLoadedConfig is BsPreforcastStageLoadedConfig


def test_load_app_config_materializes_bs_preforcast_stage_with_top_level_config_types(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    stage_path = tmp_path / "bs_preforcast.yaml"
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(),
                "common": _base_payload(data_path),
                "univariable": {
                    "dataset": {"target_col": "bs_a", "hist_exog_cols": []},
                    "jobs": [{"model": "Naive", "params": {}}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = {
        "enabled": True,
        "config_path": str(stage_path),
    }
    (tmp_path / "search_space.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {},
                "training": [],
                "residual": {"xgboost": ["n_estimators"]},
                "bs_preforcast_models": {},
                "bs_preforcast_training": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert isinstance(loaded.config.bs_preforcast, BsPreforcastConfig)
    assert isinstance(loaded.bs_preforcast_stage1, BsPreforcastStageLoadedConfig)
    assert loaded.bs_preforcast_stage1 is not None
    assert loaded.bs_preforcast_stage1.source_path == stage_path.resolve()
    assert loaded.config.bs_preforcast.using_futr_exog is False
    assert loaded.config.bs_preforcast.target_columns == ("bs_a",)


def test_repo_bs_preforcast_yaml_uses_shared_jobs_default_path() -> None:
    payload = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "bs_preforcast.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert payload["univariable"]["jobs"] == "yaml/bs_preforcast_jobs_default.yaml"


def test_repo_bs_preforcast_jobs_default_yaml_contains_direct_models() -> None:
    payload = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "yaml"
            / "bs_preforcast_jobs_default.yaml"
        ).read_text(encoding="utf-8")
    )
    jobs = payload if isinstance(payload, list) else payload["jobs"]

    assert [job["model"] for job in jobs] == ["ARIMA", "ES", "xgboost", "lightgbm"]
    assert jobs[0]["params"]["season_length"] == 12
    assert jobs[2]["params"]["lags"] == [1, 2, 3, 6, 12]
