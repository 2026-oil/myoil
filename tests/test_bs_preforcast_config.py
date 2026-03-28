from __future__ import annotations

from pathlib import Path

import pytest
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


def _linked_bs_preforcast(
    *,
    target_columns: tuple[str, ...] = ("bs_a",),
    multivariable: bool = False,
    hist_columns: tuple[str, ...] = (),
    legacy_using_futr_exog: bool | None = None,
) -> dict[str, object]:
    payload = {
        "bs_preforcast": {
            "target_columns": list(target_columns),
            "task": {"multivariable": multivariable},
            "hist_columns": list(hist_columns),
        }
    }
    if legacy_using_futr_exog is not None:
        payload["bs_preforcast"]["using_futr_exog"] = legacy_using_futr_exog
    return payload


def _write_search_space(tmp_path: Path, payload: dict[str, object] | None = None) -> None:
    base = {
        "models": {},
        "training": [],
        "residual": {"xgboost": ["n_estimators"]},
        "bs_preforcast_models": {},
        "bs_preforcast_training": [],
    }
    if payload is not None:
        base.update(payload)
    (tmp_path / "yaml/HPO").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/HPO/search_space.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False),
        encoding="utf-8",
    )


def test_bs_preforcast_types_not_leaked_into_residual_config() -> None:
    assert not hasattr(residual_config, "BsPreforcastConfig")
    assert not hasattr(residual_config, "BsPreforcastStageLoadedConfig")


def test_load_app_config_materializes_bs_preforcast_stage_with_top_level_config_types(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "dt,target,hist_a,aux_a,bs_a\n"
        "2020-01-01,1,2,5,10\n"
        "2020-01-08,2,3,6,11\n"
        "2020-01-15,3,4,7,12\n",
        encoding="utf-8",
    )
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(hist_columns=("aux_a",)),
                "jobs": [
                    {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = _main_bs_preforcast(config_path=str(stage_path))
    _write_search_space(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert isinstance(loaded.config.stage_plugin_config, BsPreforcastConfig)
    assert isinstance(loaded.stage_plugin_loaded, BsPreforcastStageLoadedConfig)
    assert loaded.stage_plugin_loaded is not None
    assert loaded.stage_plugin_loaded.source_path == stage_path.resolve()
    assert loaded.config.stage_plugin_config.target_columns == ("bs_a",)
    assert loaded.config.stage_plugin_config.hist_columns == ("aux_a",)
    assert loaded.stage_plugin_loaded.config.dataset.path == data_path
    assert loaded.stage_plugin_loaded.config.dataset.target_col == "bs_a"
    assert loaded.stage_plugin_loaded.config.dataset.hist_exog_cols == ("aux_a",)
    assert loaded.stage_plugin_loaded.normalized_payload["bs_preforcast"][
        "selected_config_path"
    ] == str(stage_path.resolve())


def test_load_app_config_accepts_bs_preforcast_plugin_with_multiple_jobs(
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
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(),
                "jobs": [
                    {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}},
                    {"model": "WindowAverage", "params": {"window_size": 2}},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = _main_bs_preforcast(config_path=str(stage_path))
    _write_search_space(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.stage_plugin_loaded is not None
    assert [job.model for job in loaded.stage_plugin_loaded.config.jobs] == [
        "DummyUnivariate",
        "WindowAverage",
    ]


def test_load_app_config_rejects_bs_preforcast_hist_columns_overlap(
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
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(hist_columns=("bs_a",)),
                "jobs": [
                    {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = _main_bs_preforcast(config_path=str(stage_path))
    _write_search_space(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"hist_columns.*overlap"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_legacy_plugin_owned_linked_yaml_sections(
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
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(),
                "dataset": {"path": str(data_path), "target_col": "bs_a"},
                "runtime": {"random_seed": 7},
                "cv": {"horizon": 1},
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
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = _main_bs_preforcast(config_path=str(stage_path))
    _write_search_space(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"plugin YAML contains unsupported key\(s\): cv, dataset, runtime",
    ):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_legacy_using_futr_exog_in_linked_stage_yaml(
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
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(legacy_using_futr_exog=True),
                "jobs": [
                    {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = _main_bs_preforcast(config_path=str(stage_path))
    _write_search_space(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported key\(s\): using_futr_exog"):
        load_app_config(tmp_path, config_path=config_path)


def test_repo_bs_preforcast_yaml_defaults_to_integrated_univariable_plugin() -> None:
    payload = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "yaml/plugins/bs_preforcast.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert payload["bs_preforcast"]["target_columns"] == ["BS_Core_Index_Integrated"]
    assert payload["bs_preforcast"]["task"]["multivariable"] is False
    assert payload["bs_preforcast"]["hist_columns"] == []
    assert payload["jobs"] == [
        {
            "model": "TimeXer",
            "params": {
                "patch_len": 16,
                "hidden_size": 768,
                "n_heads": 16,
                "e_layers": 4,
                "d_ff": 1024,
                "factor": 8,
                "dropout": 0.2,
                "use_norm": True,
            },
        }
    ]


def test_repo_bs_preforcast_jobs_default_yaml_contains_direct_models() -> None:
    payload = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "yaml"
            / "jobs"
            / "bs_preforcast_jobs_default.yaml"
        ).read_text(encoding="utf-8")
    )
    jobs = payload if isinstance(payload, list) else payload["jobs"]

    assert [job["model"] for job in jobs] == ["ARIMA", "ES", "xgboost", "lightgbm"]
    assert "season_length" not in jobs[0]["params"]
    assert "seasonal_order" not in jobs[0]["params"]
    assert "seasonal" not in jobs[1]["params"]
    assert jobs[2]["params"]["lags"] == [1, 2, 3, 6, 12]


def test_repo_default_bs_preforcast_path_is_loadable_with_defaults_yaml(
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
    defaults_dir = tmp_path / "yaml" / "jobs"
    defaults_dir.mkdir(parents=True, exist_ok=True)
    defaults_payload = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "yaml"
            / "jobs"
            / "bs_preforcast_jobs_default.yaml"
        ).read_text(encoding="utf-8")
    )
    (defaults_dir / "bs_preforcast_jobs_default.yaml").write_text(
        yaml.safe_dump(defaults_payload, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "yaml/plugins").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/plugins/bs_preforcast.yaml").write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(),
                "jobs": "yaml/jobs/bs_preforcast_jobs_default.yaml",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = {"enabled": True}
    (tmp_path / "yaml/HPO").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/HPO/search_space.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {},
                "training": [],
                "residual": {"xgboost": ["n_estimators"]},
                "bs_preforcast_models": {
                    "ARIMA": {
                        "order": {
                            "type": "categorical",
                            "choices": [[1, 0, 0], [1, 1, 0]],
                        },
                        "include_mean": {"type": "categorical", "choices": [True, False]},
                        "include_drift": {"type": "categorical", "choices": [False, True]},
                    },
                    "ES": ["trend", "damped_trend"],
                    "xgboost": {
                        "lags": {
                            "type": "categorical",
                            "choices": [[1, 2, 3], [1, 2, 3, 6, 12]],
                        },
                        "n_estimators": {"type": "categorical", "choices": [16, 32, 64]},
                        "max_depth": {"type": "int", "low": 2, "high": 6, "step": 1},
                    },
                    "lightgbm": {
                        "lags": {
                            "type": "categorical",
                            "choices": [[1, 2, 3], [1, 2, 3, 6, 12]],
                        },
                        "n_estimators": {"type": "categorical", "choices": [32, 64, 96]},
                        "max_depth": {"type": "categorical", "choices": [4, 6, -1]},
                        "num_leaves": {"type": "categorical", "choices": [15, 31, 63]},
                        "min_child_samples": {"type": "categorical", "choices": [10, 20, 40]},
                    },
                },
                "bs_preforcast_training": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.stage_plugin_loaded is not None
    assert [job.model for job in loaded.stage_plugin_loaded.config.jobs] == [
        "ARIMA",
        "ES",
        "xgboost",
        "lightgbm",
    ]


def test_load_app_config_preserves_native_list_bs_preforcast_search_space_contract(
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
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(),
                "jobs": [
                    {
                        "model": "xgboost",
                        "params": {
                            "lags": [1, 2, 3, 6, 12],
                            "n_estimators": 16,
                            "max_depth": 3,
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = _main_bs_preforcast(config_path=str(stage_path))
    _write_search_space(
        tmp_path,
        {
            "bs_preforcast_models": {
                "ARIMA": {
                    "order": {
                        "type": "categorical",
                        "choices": [[1, 0, 0], [1, 1, 0]],
                    }
                },
                "xgboost": {
                    "lags": {
                        "type": "categorical",
                        "choices": [[1, 2, 3], [1, 2, 3, 6, 12]],
                    }
                },
                "NHITS": {
                    "mlp_units": {
                        "type": "categorical",
                        "choices": [
                            [[32, 32], [32, 32], [32, 32]],
                            [[64, 64], [64, 64], [64, 64]],
                        ],
                    }
                },
            }
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    loaded = load_app_config(tmp_path, config_path=config_path)

    assert loaded.stage_plugin_loaded is not None
    assert loaded.stage_plugin_loaded.config.jobs[0].params["lags"] == [1, 2, 3, 6, 12]
    assert loaded.stage_plugin_loaded.search_space_payload is not None
    assert loaded.stage_plugin_loaded.search_space_payload["models"]["xgboost"]["lags"]["choices"][1] == [
        1,
        2,
        3,
        6,
        12,
    ]
    assert loaded.stage_plugin_loaded.search_space_payload["models"]["NHITS"]["mlp_units"]["choices"][0] == [
        [32, 32],
        [32, 32],
        [32, 32],
    ]


def test_load_app_config_rejects_stringified_bs_preforcast_list_search_space_contract(
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
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(),
                "jobs": [
                    {"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = _main_bs_preforcast(config_path=str(stage_path))
    _write_search_space(
        tmp_path,
        {
            "bs_preforcast_models": {
                "ARIMA": {
                    "order": {
                        "type": "categorical",
                        "choices": ["[1, 1, 0]"],
                    }
                }
            }
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="native YAML list values"):
        load_app_config(tmp_path, config_path=config_path)


def test_load_app_config_rejects_bs_preforcast_autoarima_stage_job(
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
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(
        yaml.safe_dump(
            {
                **_linked_bs_preforcast(),
                "jobs": [{"model": "AutoARIMA", "params": {}}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["bs_preforcast"] = _main_bs_preforcast(config_path=str(stage_path))
    (tmp_path / "yaml/HPO").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/HPO/search_space.yaml").write_text(
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

    with pytest.raises(
        ValueError, match="bs_preforcast stage no longer supports AutoARIMA; use ARIMA instead"
    ):
        load_app_config(tmp_path, config_path=config_path)
