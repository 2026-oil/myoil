from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from residual.config import load_app_config
from residual.bs_preforcast_runtime import prepare_bs_preforcast_fold_inputs


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
            "num_lr_decays": -1,
            "loss": "mse",
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
            "parallelize_single_job_tuning": False,
        },
        "residual": {"enabled": False, "model": "xgboost", "params": {}},
        "jobs": [{"model": "Naive", "params": {}}],
        "bs_preforcast": {"enabled": True, "config_path": "yaml/plugins/bs_preforcast.yaml"},
    }


def _plugin_payload(
    *,
    jobs: list[dict[str, object]],
    target_columns: tuple[str, ...] = ("bs_a",),
    multivariable: bool = False,
    exog_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "bs_preforcast": {
            "target_columns": list(target_columns),
            "task": {"multivariable": multivariable},
            "exog_columns": list(exog_columns),
        },
        "jobs": jobs,
    }


def _write_search_space(tmp_path: Path, payload: dict[str, object] | None = None) -> None:
    base = {
        "models": {},
        "training": [],
        "residual": {"xgboost": ["n_estimators"]},
        "bs_preforcast_models": {},
        "bs_preforcast_training": [],
    }
    if payload:
        base.update(payload)
    (tmp_path / "yaml/HPO").mkdir(parents=True, exist_ok=True)
    (tmp_path / "yaml/HPO/search_space.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False),
        encoding="utf-8",
    )


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_plugin_only_yaml_inherits_main_dataset_and_exog_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "dt,target,hist_a,aux_a,bs_a\n"
        "2020-01-01,1,2,5,10\n"
        "2020-01-08,2,3,6,11\n"
        "2020-01-15,3,4,7,12\n",
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    _write_config(
        stage_path,
        _plugin_payload(
            jobs=[{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}],
            exog_columns=("aux_a",),
        ),
    )
    _write_search_space(tmp_path)

    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path / "config.yaml", main_payload))

    assert loaded.bs_preforcast_stage1 is not None
    assert loaded.bs_preforcast_stage1.config.dataset.path == data_path
    assert loaded.bs_preforcast_stage1.config.dataset.target_col == "bs_a"
    assert loaded.bs_preforcast_stage1.config.dataset.hist_exog_cols == ("aux_a",)
    assert loaded.bs_preforcast_stage1.config.bs_preforcast.exog_columns == ("aux_a",)


def test_plugin_only_yaml_rejects_dataset_block(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("dt,target,hist_a\n2020-01-01,1,2\n2020-01-08,2,3\n", encoding="utf-8")
    main_payload = _base_payload(data_path)
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    _write_config(
        stage_path,
        {
            **_plugin_payload(jobs=[{"model": "ARIMA", "params": {"season_length": 4}}]),
            "dataset": {"path": "forbidden.csv"},
        },
    )
    _write_search_space(tmp_path)

    with pytest.raises(ValueError, match=r"plugin YAML contains unsupported key\(s\): dataset"):
        load_app_config(tmp_path, config_path=_write_config(tmp_path / "config.yaml", main_payload))


def test_plugin_only_yaml_rejects_learned_auto_and_training_auto(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "dt,target,hist_a,bs_a\n"
        "2020-01-01,1,2,10\n"
        "2020-01-08,2,3,11\n"
        "2020-01-15,3,4,12\n",
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    _write_config(stage_path, _plugin_payload(jobs=[{"model": "TFT", "params": {}}]))
    _write_search_space(
        tmp_path,
        {
            "bs_preforcast_models": {"TFT": ["hidden_size"]},
            "bs_preforcast_training": ["batch_size"],
        },
    )

    with pytest.raises(ValueError, match=r"fixed-param jobs"):
        load_app_config(tmp_path, config_path=_write_config(tmp_path / "config.yaml", main_payload))


def test_metadata_shell_uses_empty_run_roots_and_null_selected_jobs_path(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/bs_preforcast_runtime_smoke.yaml")
    output_root = tmp_path / "validate"
    from residual import runtime

    code = runtime.main(["--config", str(fixture_path), "--validate-only", "--output-root", str(output_root)])
    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())

    assert resolved["bs_preforcast"]["stage1_run_roots"] == []
    assert resolved["bs_preforcast"]["stage1_selected_jobs_path"] is None
    assert manifest["bs_preforcast"]["stage1_run_roots"] == []
    assert manifest["bs_preforcast"]["stage1_selected_jobs_path"] is None


def test_lag_derived_injection_fails_when_train_shorter_than_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import residual.bs_preforcast_runtime as bs_runtime

    data_path = tmp_path / "data.csv"
    data_path.write_text(
        "dt,target,bs_a\n"
        "2020-01-01,1,10\n"
        "2020-01-08,2,11\n"
        "2020-01-15,3,12\n",
        encoding="utf-8",
    )
    main_payload = _base_payload(data_path)
    main_payload["cv"]["horizon"] = 2
    main_payload["jobs"] = [{"model": "Naive", "params": {}}]
    stage_path = tmp_path / "yaml/plugins/bs_preforcast.yaml"
    _write_config(
        stage_path,
        _plugin_payload(
            jobs=[{"model": "DummyUnivariate", "params": {"start_padding_enabled": True}}],
        ),
    )
    _write_search_space(tmp_path)
    loaded = load_app_config(tmp_path, config_path=_write_config(tmp_path / "config.yaml", main_payload))
    source_df = pd.read_csv(data_path)
    train_df = source_df.iloc[:1].reset_index(drop=True)
    future_df = source_df.iloc[1:].reset_index(drop=True)
    monkeypatch.setattr(
        bs_runtime,
        "compute_bs_preforcast_fold_forecasts",
        lambda *args, **kwargs: {"bs_a": [10.5, 10.75]},
    )

    with pytest.raises(ValueError, match="requires at least as many training rows as horizon"):
        prepare_bs_preforcast_fold_inputs(loaded, loaded.config.jobs[0], train_df, future_df)
