from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import feature_set_aaforecast_postprocess as postprocess


def test_iter_passed_run_entries_expands_fanout_and_groups_configs(
    monkeypatch,
) -> None:
    baseline_loaded = SimpleNamespace(
        jobs_fanout_specs=(SimpleNamespace(route_slug="gru_informer"),),
        active_jobs_route_slug=None,
    )
    aa_loaded = SimpleNamespace(
        jobs_fanout_specs=(),
        active_jobs_route_slug=None,
    )

    def fake_load_app_config(_repo_root, *, config_path):
        if config_path.endswith("baseline-ret.yaml"):
            return baseline_loaded
        if config_path.endswith("aaforecast-gru.yaml"):
            return aa_loaded
        raise AssertionError(config_path)

    monkeypatch.setattr(postprocess, "load_app_config", fake_load_app_config)
    monkeypatch.setattr(
        postprocess,
        "loaded_config_for_jobs_fanout",
        lambda _repo_root, _loaded, spec: f"variant:{spec.route_slug}",
    )
    monkeypatch.setattr(
        postprocess.runtime,
        "_default_output_root",
        lambda _repo_root, loaded: Path("/tmp")
        / (
            "feature_set_aaforecast_brentoil_baseline-ret_gru_informer"
            if loaded == "variant:gru_informer"
            else "feature_set_aaforecast_aaforecast_gru"
        ),
    )

    entries = postprocess._iter_passed_run_entries(
        Path("/repo"),
        {
            "results": [
                {
                    "config": "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml",
                    "status": "passed",
                },
                {
                    "config": "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml",
                    "status": "passed",
                },
                {
                    "config": "yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml",
                    "status": "failed",
                },
            ]
        },
    )

    assert entries == [
        {
            "config": "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml",
            "group": "ret",
            "jobs_route": "gru_informer",
            "canonical_run_root": "/tmp/feature_set_aaforecast_brentoil_baseline-ret_gru_informer",
            "run_name": "feature_set_aaforecast_brentoil_baseline-ret_gru_informer",
            "derived": False,
        },
        {
            "config": "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml",
            "group": "nonret",
            "jobs_route": None,
            "canonical_run_root": "/tmp/feature_set_aaforecast_aaforecast_gru",
            "run_name": "feature_set_aaforecast_aaforecast_gru",
            "derived": False,
        },
    ]


def test_resolve_existing_run_root_falls_back_to_runs_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_root = tmp_path / "runs" / "feature_set_aaforecast_wti_aaforecast_timexer-ret"
    actual_root = tmp_path / "runs_1" / "final_wti" / missing_root.name
    _write_run_fixture(
        run_root=actual_root,
        resolved_payload={"dataset": {"target_col": "y"}, "training": {}, "cv": {}},
        leaderboard_rows=[{"rank": 1, "model": "AAForecast"}],
        metrics_rows=[{"model": "AAForecast", "fold_idx": 0, "cutoff": "2026-02-23"}],
        prediction_rows=[
            {
                "model": "AAForecast",
                "fold_idx": 0,
                "cutoff": "2026-02-23",
                "horizon_step": 1,
                "ds": "2026-03-02",
                "y": 1.0,
                "y_hat": 1.0,
            }
        ],
    )

    monkeypatch.setattr(postprocess, "REPO_ROOT", tmp_path)

    resolved = postprocess._resolve_existing_run_root(missing_root)

    assert resolved == actual_root.resolve()


def test_normalize_combined_fold_idx_by_train_end_ds_realigns_misaligned_runs() -> None:
    combined = pd.DataFrame(
        {
            "run_id": ["baseline", "baseline", "aa", "aa"],
            "run_root": ["/tmp/baseline", "/tmp/baseline", "/tmp/aa", "/tmp/aa"],
            "model": ["GRU", "GRU", "AAForecast", "AAForecast"],
            "fold_idx": [0, 1, 0, 1],
            "train_end_ds": [
                "2025-12-29 00:00:00",
                "2026-02-23 00:00:00",
                "2026-02-23 00:00:00",
                "2026-03-02 00:00:00",
            ],
            "ds": [
                "2026-01-05 00:00:00",
                "2026-03-02 00:00:00",
                "2026-03-02 00:00:00",
                "2026-03-09 00:00:00",
            ],
            "y_hat": [1.0, 2.0, 3.0, 4.0],
        }
    )

    normalized = postprocess._normalize_combined_fold_idx_by_train_end_ds(combined)

    assert normalized["fold_idx"].tolist() == [0, 1, 1, 2]


def test_write_group_plot_returns_continuous_plot_when_fold_overlay_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = {
        "retrieval": True,
        "run_root": tmp_path / "run-a",
        "predictions": pd.DataFrame(),
        "config": "yaml/experiment/feature_set_aaforecast_wti/aaforecast-gru-ret.yaml",
        "derived": False,
    }

    monkeypatch.setattr(
        postprocess,
        "_build_plot_frame_from_report",
        lambda _report: pd.DataFrame(
            {
                "fold_idx": [0],
                "train_end_ds": ["2026-02-23 00:00:00"],
                "ds": ["2026-03-02 00:00:00"],
                "y_hat": [1.0],
            }
        ),
    )

    def fake_plot_continuous_series(*args, **kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("continuous", encoding="utf-8")
        return output_path

    monkeypatch.setattr(postprocess.overlay, "plot_continuous_series", fake_plot_continuous_series)
    monkeypatch.setattr(
        postprocess.overlay,
        "plot_folds",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("fold overlay failed")),
    )

    plot_path, folds_dir = postprocess._write_group_plot(
        raw_batch_root=tmp_path / "raw",
        run_reports=[report],
        group="ret",
        x_start=None,
        x_end=None,
    )

    assert plot_path is not None
    assert plot_path.exists()
    assert folds_dir is None


def test_main_links_runs_writes_manifest_and_calls_both_group_plots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_dir = tmp_path / "logs-source"
    log_dir.mkdir()
    ret_root = tmp_path / "canonical-ret"
    ret_root.mkdir()
    nonret_root = tmp_path / "canonical-nonret"
    nonret_root.mkdir()
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "log_dir": str(log_dir),
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "config": "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml",
            "group": "ret",
            "jobs_route": "gru_informer",
            "canonical_run_root": str(ret_root),
            "run_name": ret_root.name,
        },
        {
            "config": "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml",
            "group": "nonret",
            "jobs_route": None,
            "canonical_run_root": str(nonret_root),
            "run_name": nonret_root.name,
        },
    ]
    plot_calls: list[tuple[str, str | None, str | None]] = []

    monkeypatch.setattr(
        postprocess,
        "_iter_passed_run_entries",
        lambda _repo_root, _summary_payload: entries,
    )

    def fake_write_group_plot(*, raw_batch_root, run_reports, group, x_start, x_end):
        plot_calls.append((group, x_start, x_end))
        assert len(run_reports) == 2
        plot_path = raw_batch_root / "plots" / group / "all_folds_continuous_overlay.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_text(group, encoding="utf-8")
        folds_dir = raw_batch_root / "plots" / group / "folds"
        (folds_dir / "regular").mkdir(parents=True, exist_ok=True)
        (folds_dir / "regular" / "fold_000_predictions_overlay.png").write_text(
            "fold", encoding="utf-8"
        )
        return plot_path, folds_dir

    monkeypatch.setattr(postprocess, "_write_group_plot", fake_write_group_plot)

    def fake_write_result_markdown(
        *,
        raw_batch_root,
        run_reports,
        ret_plot,
        ret_folds_dir,
        nonret_plot,
        nonret_folds_dir,
    ):
        assert len(run_reports) == 2
        output = raw_batch_root / "result.md"
        output.write_text("ok", encoding="utf-8")
        return output

    monkeypatch.setattr(postprocess, "_write_result_markdown", fake_write_result_markdown)
    monkeypatch.setattr(
        postprocess,
        "_read_run_report",
        lambda entry: {"run_name": entry["run_name"], "resolved": {}, "leaderboard": pd.DataFrame()},
    )
    monkeypatch.setattr(
        postprocess,
        "_augment_with_derived_nonret_reports",
        lambda entries, run_reports: (entries, run_reports),
    )

    raw_batch_root = tmp_path / "raw_feature_set_aaforecast" / "20260417T000000Z"
    assert (
        postprocess.main(
            [
                "--summary-json",
                str(summary_json),
                "--raw-batch-root",
                str(raw_batch_root),
                "--x-start",
                "2025-08-15",
                "--x-end",
                "2026-03-09",
            ]
        )
        == 0
    )

    assert (raw_batch_root / "runs" / ret_root.name).is_symlink()
    assert (raw_batch_root / "runs" / nonret_root.name).is_symlink()
    assert (raw_batch_root / "logs").is_symlink()
    manifest = json.loads((raw_batch_root / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["log_dir"] == str(log_dir.resolve())
    assert len(manifest["entries"]) == 2
    assert plot_calls == [
        ("ret", "2025-08-15", "2026-03-09"),
        ("nonret", "2025-08-15", "2026-03-09"),
    ]
    assert (raw_batch_root / "result.md").read_text(encoding="utf-8") == "ok"


def _write_run_fixture(
    *,
    run_root: Path,
    resolved_payload: dict[str, object],
    leaderboard_rows: list[dict[str, object]],
    metrics_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    stage_config: dict[str, object] | None = None,
    retrieval_payloads: list[tuple[str, dict[str, object]]] | None = None,
) -> None:
    (run_root / "config").mkdir(parents=True, exist_ok=True)
    (run_root / "summary" / "folds" / "fold_000").mkdir(parents=True, exist_ok=True)
    (run_root / "summary").mkdir(parents=True, exist_ok=True)
    (run_root / "aa_forecast" / "config").mkdir(parents=True, exist_ok=True)

    (run_root / "config" / "config.resolved.json").write_text(
        json.dumps(resolved_payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(leaderboard_rows).to_csv(run_root / "summary" / "leaderboard.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_root / "summary" / "result.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(
        run_root / "summary" / "folds" / "fold_000" / "metrics.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        run_root / "summary" / "folds" / "fold_000" / "predictions.csv", index=False
    )
    if stage_config is not None:
        (run_root / "aa_forecast" / "config" / "stage_config.json").write_text(
            json.dumps(stage_config, indent=2), encoding="utf-8"
        )
    if retrieval_payloads:
        for relative, payload in retrieval_payloads:
            target = run_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_write_result_markdown_uses_actual_run_artifacts(tmp_path: Path) -> None:
    raw_batch_root = tmp_path / "raw" / "20260417T000000Z"
    raw_batch_root.mkdir(parents=True)

    resolved_payload = {
        "dataset": {
            "target_col": "Com_CrudeOil",
            "hist_exog_cols": ["GPRD_THREAT", "GPRD"],
            "transformations_target": "diff",
            "transformations_exog": "diff",
        },
        "training": {
            "input_size": 64,
            "max_steps": 400,
            "val_size": 4,
            "loss": "mse",
        },
        "cv": {
            "horizon": 2,
            "step_size": 1,
            "n_windows": 1,
            "gap": 0,
            "overlap_eval_policy": "by_cutoff_mean",
        },
    }
    stage_config = {
        "star_hist_exog_cols_resolved": ["GPRD_THREAT"],
        "non_star_hist_exog_cols_resolved": ["GPRD"],
        "lowess_frac": 0.35,
        "lowess_delta": 0.01,
        "uncertainty": {"enabled": True, "sample_count": 50},
    }

    baseline_off = tmp_path / "baseline-off"
    _write_run_fixture(
        run_root=baseline_off,
        resolved_payload=resolved_payload,
        leaderboard_rows=[
            {
                "rank": 1,
                "model": "GRU",
                "mean_fold_mae": 5.8,
                "mean_fold_mse": 40.0,
                "mean_fold_rmse": 6.3,
                "fold_count": 1,
                "mean_fold_mape": 0.10,
                "mean_fold_nrmse": 1.10,
                "mean_fold_r2": -1.0,
            },
            {
                "rank": 2,
                "model": "Informer",
                "mean_fold_mae": 5.0,
                "mean_fold_mse": 38.0,
                "mean_fold_rmse": 6.1,
                "fold_count": 1,
                "mean_fold_mape": 0.09,
                "mean_fold_nrmse": 1.00,
                "mean_fold_r2": -0.8,
            },
        ],
        metrics_rows=[
            {
                "model": "GRU",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 5.8,
                "MSE": 40.0,
                "RMSE": 6.3,
                "MAPE": 0.10,
                "NRMSE": 1.10,
                "R2": -1.0,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            },
            {
                "model": "Informer",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 5.0,
                "MSE": 38.0,
                "RMSE": 6.1,
                "MAPE": 0.09,
                "NRMSE": 1.00,
                "R2": -0.8,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            },
        ],
        prediction_rows=[
            {
                "model": "GRU",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 75.0,
            },
            {
                "model": "GRU",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 78.0,
            },
            {
                "model": "Informer",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 76.0,
            },
            {
                "model": "Informer",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 79.0,
            },
        ],
    )

    baseline_on = tmp_path / "baseline-on"
    _write_run_fixture(
        run_root=baseline_on,
        resolved_payload=resolved_payload,
        leaderboard_rows=[
            {
                "rank": 1,
                "model": "GRU",
                "mean_fold_mae": 5.5,
                "mean_fold_mse": 39.0,
                "mean_fold_rmse": 6.2,
                "fold_count": 1,
                "mean_fold_mape": 0.08,
                "mean_fold_nrmse": 1.05,
                "mean_fold_r2": -0.9,
            },
            {
                "rank": 2,
                "model": "Informer",
                "mean_fold_mae": 4.7,
                "mean_fold_mse": 37.0,
                "mean_fold_rmse": 6.0,
                "fold_count": 1,
                "mean_fold_mape": 0.07,
                "mean_fold_nrmse": 0.95,
                "mean_fold_r2": -0.7,
            },
        ],
        metrics_rows=[
            {
                "model": "GRU",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 5.5,
                "MSE": 39.0,
                "RMSE": 6.2,
                "MAPE": 0.08,
                "NRMSE": 1.05,
                "R2": -0.9,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            },
            {
                "model": "Informer",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 4.7,
                "MSE": 37.0,
                "RMSE": 6.0,
                "MAPE": 0.07,
                "NRMSE": 0.95,
                "R2": -0.7,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            },
        ],
        prediction_rows=[
            {
                "model": "GRU",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 77.0,
            },
            {
                "model": "GRU",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 81.0,
            },
            {
                "model": "Informer",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 78.0,
            },
            {
                "model": "Informer",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 82.0,
            },
        ],
    )

    aa_off = tmp_path / "aa-off"
    _write_run_fixture(
        run_root=aa_off,
        resolved_payload=resolved_payload,
        leaderboard_rows=[
            {
                "rank": 1,
                "model": "AAForecast",
                "mean_fold_mae": 4.9,
                "mean_fold_mse": 35.0,
                "mean_fold_rmse": 5.9,
                "fold_count": 1,
                "mean_fold_mape": 0.06,
                "mean_fold_nrmse": 0.90,
                "mean_fold_r2": -0.5,
            }
        ],
        metrics_rows=[
            {
                "model": "AAForecast",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 4.9,
                "MSE": 35.0,
                "RMSE": 5.9,
                "MAPE": 0.06,
                "NRMSE": 0.90,
                "R2": -0.5,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            }
        ],
        prediction_rows=[
            {
                "model": "AAForecast",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 79.0,
                "aaforecast_retrieval_enabled": False,
                "aaforecast_retrieval_applied": False,
            },
            {
                "model": "AAForecast",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 84.0,
                "aaforecast_retrieval_enabled": False,
                "aaforecast_retrieval_applied": False,
            },
        ],
        stage_config=stage_config,
    )

    aa_on = tmp_path / "aa-on"
    _write_run_fixture(
        run_root=aa_on,
        resolved_payload=resolved_payload,
        leaderboard_rows=[
            {
                "rank": 1,
                "model": "AAForecast",
                "mean_fold_mae": 4.5,
                "mean_fold_mse": 34.0,
                "mean_fold_rmse": 5.8,
                "fold_count": 1,
                "mean_fold_mape": 0.05,
                "mean_fold_nrmse": 0.85,
                "mean_fold_r2": -0.4,
            }
        ],
        metrics_rows=[
            {
                "model": "AAForecast",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 4.5,
                "MSE": 34.0,
                "RMSE": 5.8,
                "MAPE": 0.05,
                "NRMSE": 0.85,
                "R2": -0.4,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            }
        ],
        prediction_rows=[
            {
                "model": "AAForecast",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 80.0,
                "aaforecast_retrieval_enabled": True,
                "aaforecast_retrieval_applied": True,
                "aaforecast_retrieval_artifact": "aa_forecast/retrieval/fold_000/20260223T000000.json",
            },
            {
                "model": "AAForecast",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 86.0,
                "aaforecast_retrieval_enabled": True,
                "aaforecast_retrieval_applied": True,
                "aaforecast_retrieval_artifact": "aa_forecast/retrieval/fold_000/20260223T000000.json",
            },
        ],
        stage_config=stage_config,
        retrieval_payloads=[
            (
                "aa_forecast/retrieval/fold_000/20260223T000000.json",
                {"retrieval_applied": True, "top_k_used": 1},
            )
        ],
    )

    entries = [
        {
            "config": "yaml/experiment/feature_set_aaforecast_wti/baseline.yaml",
            "group": "nonret",
            "jobs_route": "gru_informer",
            "canonical_run_root": str(baseline_off),
            "run_name": baseline_off.name,
        },
        {
            "config": "yaml/experiment/feature_set_aaforecast_wti/baseline-ret.yaml",
            "group": "ret",
            "jobs_route": "gru_informer",
            "canonical_run_root": str(baseline_on),
            "run_name": baseline_on.name,
        },
        {
            "config": "yaml/experiment/feature_set_aaforecast_wti/aaforecast-gru.yaml",
            "group": "nonret",
            "jobs_route": None,
            "canonical_run_root": str(aa_off),
            "run_name": aa_off.name,
        },
        {
            "config": "yaml/experiment/feature_set_aaforecast_wti/aaforecast-gru-ret.yaml",
            "group": "ret",
            "jobs_route": None,
            "canonical_run_root": str(aa_on),
            "run_name": aa_on.name,
        },
    ]
    run_reports = [postprocess._read_run_report(entry) for entry in entries]

    ret_plot = raw_batch_root / "plots" / "ret" / "all_folds_continuous_overlay.png"
    ret_plot.parent.mkdir(parents=True, exist_ok=True)
    ret_plot.write_text("ret", encoding="utf-8")
    ret_folds_dir = raw_batch_root / "plots" / "ret" / "folds"
    (ret_folds_dir / "regular").mkdir(parents=True, exist_ok=True)
    (ret_folds_dir / "regular" / "fold_000_predictions_overlay.png").write_text(
        "ret-fold", encoding="utf-8"
    )

    nonret_plot = raw_batch_root / "plots" / "nonret" / "all_folds_continuous_overlay.png"
    nonret_plot.parent.mkdir(parents=True, exist_ok=True)
    nonret_plot.write_text("nonret", encoding="utf-8")
    nonret_folds_dir = raw_batch_root / "plots" / "nonret" / "folds"
    (nonret_folds_dir / "regular").mkdir(parents=True, exist_ok=True)
    (nonret_folds_dir / "regular" / "fold_000_predictions_overlay.png").write_text(
        "nonret-fold", encoding="utf-8"
    )

    output_path = postprocess._write_result_markdown(
        raw_batch_root=raw_batch_root,
        run_reports=run_reports,
        ret_plot=ret_plot,
        ret_folds_dir=ret_folds_dir,
        nonret_plot=nonret_plot,
        nonret_folds_dir=nonret_folds_dir,
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert "# 01. 핵심쟁점" in markdown
    assert "- **예측 타깃:** Com_CrudeOil" in markdown
    assert "- **예측 단위:** 주간 예측" in markdown
    assert "- **max_steps:** `400`" in markdown
    assert "AAForecast retrieval on run은 총 1개 fold artifact" in markdown
    assert "baseline과 직접 비교 가능한 2개 조건 중 **2개 조건에서 AAForecast가 평균 MAPE를 개선**했다." in markdown
    assert "retrieval on은 직접 비교 가능한 3개 backbone 조건 중 **3개 조건에서 평균 MAPE를 낮췄다**." in markdown
    assert "| Rank | 실험군" in markdown
    assert "![ret continuous overlay](plots/ret/all_folds_continuous_overlay.png)" in markdown
    assert "![nonret fold overlay](plots/nonret/folds/regular/fold_000_predictions_overlay.png)" in markdown


def test_augment_with_derived_nonret_reports_reuses_base_prediction_for_aaforecast(
    tmp_path: Path,
) -> None:
    resolved_payload = {
        "dataset": {
            "target_col": "Com_CrudeOil",
            "hist_exog_cols": ["GPRD_THREAT"],
            "transformations_target": "diff",
            "transformations_exog": "diff",
        },
        "training": {
            "input_size": 64,
            "max_steps": 400,
            "val_size": 4,
            "loss": "mse",
        },
        "cv": {
            "horizon": 2,
            "step_size": 1,
            "n_windows": 1,
            "gap": 0,
            "overlap_eval_policy": "by_cutoff_mean",
        },
    }
    stage_config = {
        "backbone": "GRU",
        "star_hist_exog_cols_resolved": ["GPRD_THREAT"],
        "non_star_hist_exog_cols_resolved": [],
        "lowess_frac": 0.35,
        "lowess_delta": 0.01,
        "uncertainty": {"enabled": True, "sample_count": 50},
    }
    aa_on = tmp_path / "aa-on"
    _write_run_fixture(
        run_root=aa_on,
        resolved_payload=resolved_payload,
        leaderboard_rows=[
            {
                "rank": 1,
                "model": "AAForecast",
                "mean_fold_mae": 4.5,
                "mean_fold_mse": 34.0,
                "mean_fold_rmse": 5.8,
                "fold_count": 1,
                "mean_fold_mape": 0.05,
                "mean_fold_nrmse": 0.85,
                "mean_fold_r2": -0.4,
            }
        ],
        metrics_rows=[
            {
                "model": "AAForecast",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 4.5,
                "MSE": 34.0,
                "RMSE": 5.8,
                "MAPE": 0.05,
                "NRMSE": 0.85,
                "R2": -0.4,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            }
        ],
        prediction_rows=[
            {
                "model": "AAForecast",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 79.0,
                "aaforecast_retrieval_enabled": True,
                "aaforecast_retrieval_applied": True,
                "aaforecast_retrieval_artifact": "aa_forecast/retrieval/fold_000/20260223T000000.json",
            },
            {
                "model": "AAForecast",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 86.0,
                "aaforecast_retrieval_enabled": True,
                "aaforecast_retrieval_applied": True,
                "aaforecast_retrieval_artifact": "aa_forecast/retrieval/fold_000/20260223T000000.json",
            },
        ],
        stage_config=stage_config,
        retrieval_payloads=[
            (
                "aa_forecast/retrieval/fold_000/20260223T000000.json",
                {
                    "cutoff": "2026-02-23 00:00:00",
                    "base_prediction": [70.0, 74.0],
                    "final_prediction": [79.0, 86.0],
                    "memory_prediction": [82.0, 88.0],
                    "retrieval_applied": True,
                    "top_k_used": 1,
                },
            )
        ],
    )
    entries = [
        {
            "config": "yaml/experiment/feature_set_aaforecast_wti/aaforecast-gru-ret.yaml",
            "group": "ret",
            "jobs_route": None,
            "canonical_run_root": str(aa_on),
            "run_name": aa_on.name,
            "derived": False,
        }
    ]
    run_reports = [postprocess._read_run_report(entry) for entry in entries]

    augmented_entries, augmented_reports = postprocess._augment_with_derived_nonret_reports(
        entries, run_reports
    )

    assert len(augmented_entries) == 2
    derived_entry = augmented_entries[1]
    assert derived_entry["derived"] is True
    assert derived_entry["run_name"] == "aa-on_nonret"

    assert len(augmented_reports) == 2
    derived_report = augmented_reports[1]
    assert derived_report["retrieval"] is False
    assert derived_report["derived"] is True
    assert derived_report["display_experiment"] == "AAForecast (derived non-ret)"
    assert derived_report["predictions"]["y_hat"].tolist() == [70.0, 74.0]
    assert derived_report["leaderboard"]["mean_fold_mape"].iloc[0] == 0.16032608695652173
    assert derived_report["metrics"]["MAE"].iloc[0] == 14.0


def test_augment_with_derived_nonret_reports_reuses_base_prediction_for_baseline(
    tmp_path: Path,
) -> None:
    resolved_payload = {
        "dataset": {
            "target_col": "Com_CrudeOil",
            "hist_exog_cols": ["GPRD_THREAT"],
            "transformations_target": "diff",
            "transformations_exog": "diff",
        },
        "training": {
            "input_size": 64,
            "max_steps": 400,
            "val_size": 4,
            "loss": "mse",
        },
        "cv": {
            "horizon": 2,
            "step_size": 1,
            "n_windows": 1,
            "gap": 0,
            "overlap_eval_policy": "by_cutoff_mean",
        },
    }
    baseline_on = tmp_path / "baseline-on"
    _write_run_fixture(
        run_root=baseline_on,
        resolved_payload=resolved_payload,
        leaderboard_rows=[
            {
                "rank": 1,
                "model": "GRU",
                "mean_fold_mae": 4.5,
                "mean_fold_mse": 34.0,
                "mean_fold_rmse": 5.8,
                "fold_count": 1,
                "mean_fold_mape": 0.05,
                "mean_fold_nrmse": 0.85,
                "mean_fold_r2": -0.4,
            }
        ],
        metrics_rows=[
            {
                "model": "GRU",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 4.5,
                "MSE": 34.0,
                "RMSE": 5.8,
                "MAPE": 0.05,
                "NRMSE": 0.85,
                "R2": -0.4,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            }
        ],
        prediction_rows=[
            {
                "model": "GRU",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 79.0,
                "retrieval_enabled": True,
                "retrieval_applied": True,
                "retrieval_artifact": "retrieval/retrieval_summary_2026-02-23_000000.json",
            },
            {
                "model": "GRU",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 86.0,
                "retrieval_enabled": True,
                "retrieval_applied": True,
                "retrieval_artifact": "retrieval/retrieval_summary_2026-02-23_000000.json",
            },
        ],
        retrieval_payloads=[
            (
                "retrieval/retrieval_summary_2026-02-23_000000.json",
                {
                    "cutoff": "2026-02-23 00:00:00",
                    "base_prediction": [70.0, 74.0],
                    "final_prediction": [79.0, 86.0],
                    "memory_prediction": [82.0, 88.0],
                    "retrieval_applied": True,
                    "top_k_used": 1,
                },
            )
        ],
    )
    entries = [
        {
            "config": "yaml/experiment/feature_set_aaforecast_wti/baseline-ret.yaml",
            "group": "ret",
            "jobs_route": "gru_informer",
            "canonical_run_root": str(baseline_on),
            "run_name": baseline_on.name,
            "derived": False,
        }
    ]
    run_reports = [postprocess._read_run_report(entry) for entry in entries]

    augmented_entries, augmented_reports = postprocess._augment_with_derived_nonret_reports(
        entries, run_reports
    )

    assert len(augmented_entries) == 2
    derived_entry = augmented_entries[1]
    assert derived_entry["derived"] is True
    assert derived_entry["run_name"] == "baseline-on_nonret"
    assert derived_entry["config"] == "yaml/experiment/feature_set_aaforecast_wti/baseline.yaml"

    assert len(augmented_reports) == 2
    derived_report = augmented_reports[1]
    assert derived_report["experiment"] == "baseline"
    assert derived_report["retrieval"] is False
    assert derived_report["derived"] is True
    assert derived_report["display_experiment"] == "baseline (derived non-ret)"
    assert derived_report["predictions"]["y_hat"].tolist() == [70.0, 74.0]
    assert derived_report["predictions"]["retrieval_enabled"].tolist() == [False, False]
    assert derived_report["predictions"]["retrieval_applied"].tolist() == [False, False]
    assert derived_report["leaderboard"]["mean_fold_mape"].iloc[0] == 0.16032608695652173
    assert derived_report["metrics"]["MAE"].iloc[0] == 14.0


def test_augment_with_derived_nonret_reports_uses_worker_retrieval_payloads_by_model(
    tmp_path: Path,
) -> None:
    resolved_payload = {
        "dataset": {
            "target_col": "Com_CrudeOil",
            "hist_exog_cols": ["GPRD_THREAT"],
            "transformations_target": "diff",
            "transformations_exog": "diff",
        },
        "training": {
            "input_size": 64,
            "max_steps": 400,
            "val_size": 4,
            "loss": "mse",
        },
        "cv": {
            "horizon": 2,
            "step_size": 1,
            "n_windows": 1,
            "gap": 0,
            "overlap_eval_policy": "by_cutoff_mean",
        },
    }
    baseline_on = tmp_path / "baseline-on-worker-payloads"
    _write_run_fixture(
        run_root=baseline_on,
        resolved_payload=resolved_payload,
        leaderboard_rows=[
            {
                "rank": 1,
                "model": "GRU",
                "mean_fold_mae": 4.5,
                "mean_fold_mse": 34.0,
                "mean_fold_rmse": 5.8,
                "fold_count": 1,
                "mean_fold_mape": 0.05,
                "mean_fold_nrmse": 0.85,
                "mean_fold_r2": -0.4,
            },
            {
                "rank": 2,
                "model": "Informer",
                "mean_fold_mae": 5.5,
                "mean_fold_mse": 44.0,
                "mean_fold_rmse": 6.8,
                "fold_count": 1,
                "mean_fold_mape": 0.07,
                "mean_fold_nrmse": 0.95,
                "mean_fold_r2": -0.6,
            },
        ],
        metrics_rows=[
            {
                "model": "GRU",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 4.5,
                "MSE": 34.0,
                "RMSE": 5.8,
                "MAPE": 0.05,
                "NRMSE": 0.85,
                "R2": -0.4,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            },
            {
                "model": "Informer",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "MAE": 5.5,
                "MSE": 44.0,
                "RMSE": 6.8,
                "MAPE": 0.07,
                "NRMSE": 0.95,
                "R2": -0.6,
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
            },
        ],
        prediction_rows=[
            {
                "model": "GRU",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 79.0,
            },
            {
                "model": "GRU",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 86.0,
            },
            {
                "model": "Informer",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-02 00:00:00",
                "horizon_step": 1,
                "y": 80.0,
                "y_hat": 78.0,
            },
            {
                "model": "Informer",
                "requested_mode": "learned_fixed",
                "validated_mode": "learned_fixed",
                "fold_idx": 0,
                "cutoff": "2026-02-23 00:00:00",
                "train_end_ds": "2026-02-23 00:00:00",
                "unique_id": "Com_CrudeOil",
                "ds": "2026-03-09 00:00:00",
                "horizon_step": 2,
                "y": 92.0,
                "y_hat": 85.0,
            },
        ],
        retrieval_payloads=[
            (
                "scheduler/workers/GRU/retrieval/retrieval_summary_2026-02-23_000000.json",
                {
                    "cutoff": "2026-02-23 00:00:00",
                    "base_prediction": [70.0, 74.0],
                    "final_prediction": [79.0, 86.0],
                    "retrieval_applied": True,
                    "top_k_used": 1,
                },
            ),
            (
                "scheduler/workers/Informer/retrieval/retrieval_summary_2026-02-23_000000.json",
                {
                    "cutoff": "2026-02-23 00:00:00",
                    "base_prediction": [60.0, 64.0],
                    "final_prediction": [78.0, 85.0],
                    "retrieval_applied": True,
                    "top_k_used": 1,
                },
            ),
        ],
    )
    entries = [
        {
            "config": "yaml/experiment/feature_set_aaforecast_wti/baseline-ret.yaml",
            "group": "ret",
            "jobs_route": "gru_informer",
            "canonical_run_root": str(baseline_on),
            "run_name": baseline_on.name,
            "derived": False,
        }
    ]
    run_reports = [postprocess._read_run_report(entry) for entry in entries]

    _, augmented_reports = postprocess._augment_with_derived_nonret_reports(
        entries, run_reports
    )

    derived_report = augmented_reports[1]
    derived_predictions = derived_report["predictions"].sort_values(
        ["model", "horizon_step"]
    )
    assert derived_predictions["y_hat"].tolist() == [70.0, 74.0, 60.0, 64.0]


def test_write_result_markdown_mentions_derived_nonret_reports(tmp_path: Path) -> None:
    raw_batch_root = tmp_path / "raw" / "20260417T000000Z"
    raw_batch_root.mkdir(parents=True)
    run_reports = [
        {
            "experiment": "baseline",
            "display_experiment": "baseline (derived non-ret)",
            "retrieval": False,
            "backbone": "GRU",
            "run_name": "baseline_nonret",
            "run_root": tmp_path / "baseline",
            "config": "yaml/experiment/feature_set_aaforecast_wti/baseline.yaml",
            "leaderboard": pd.DataFrame(
                [
                    {
                        "model": "GRU",
                        "mean_fold_mape": 0.20,
                        "mean_fold_nrmse": 1.0,
                        "mean_fold_mae": 10.0,
                        "mean_fold_r2": -1.0,
                        "fold_count": 1,
                    }
                ]
            ),
            "metrics": pd.DataFrame(
                [{"model": "GRU", "fold_idx": 0, "cutoff": "2026-02-23", "MAPE": 0.20, "NRMSE": 1.0, "MAE": 10.0, "R2": -1.0}]
            ),
            "predictions": pd.DataFrame(
                [
                    {
                        "model": "GRU",
                        "fold_idx": 0,
                        "cutoff": "2026-02-23",
                        "horizon_step": 1,
                        "ds": "2026-03-02",
                        "y": 80.0,
                        "y_hat": 70.0,
                    },
                    {
                        "model": "GRU",
                        "fold_idx": 0,
                        "cutoff": "2026-02-23",
                        "horizon_step": 2,
                        "ds": "2026-03-09",
                        "y": 92.0,
                        "y_hat": 74.0,
                    },
                ]
            ),
            "resolved": {
                "dataset": {
                    "target_col": "Com_CrudeOil",
                    "hist_exog_cols": ["GPRD_THREAT"],
                    "transformations_target": "diff",
                    "transformations_exog": "diff",
                },
                "training": {"input_size": 64, "max_steps": 400, "val_size": 4, "loss": "mse"},
                "cv": {"horizon": 2, "step_size": 1, "n_windows": 1, "gap": 0, "overlap_eval_policy": "by_cutoff_mean"},
            },
            "stage_config": None,
            "retrieval_payloads": [],
            "derived": True,
        },
        {
            "experiment": "AAForecast",
            "display_experiment": "AAForecast (derived non-ret)",
            "retrieval": False,
            "backbone": "GRU",
            "run_name": "aaforecast_gru",
            "run_root": tmp_path / "aa",
            "config": "yaml/experiment/feature_set_aaforecast_wti/aaforecast-gru.yaml",
            "leaderboard": pd.DataFrame(
                [
                    {
                        "model": "AAForecast",
                        "mean_fold_mape": 0.16,
                        "mean_fold_nrmse": 0.9,
                        "mean_fold_mae": 14.0,
                        "mean_fold_r2": -0.5,
                        "fold_count": 1,
                    }
                ]
            ),
            "metrics": pd.DataFrame(
                [{"model": "AAForecast", "fold_idx": 0, "cutoff": "2026-02-23", "MAPE": 0.16, "NRMSE": 0.9, "MAE": 14.0, "R2": -0.5}]
            ),
            "predictions": pd.DataFrame(
                [
                    {
                        "model": "AAForecast",
                        "fold_idx": 0,
                        "cutoff": "2026-02-23",
                        "horizon_step": 1,
                        "ds": "2026-03-02",
                        "y": 80.0,
                        "y_hat": 70.0,
                    },
                    {
                        "model": "AAForecast",
                        "fold_idx": 0,
                        "cutoff": "2026-02-23",
                        "horizon_step": 2,
                        "ds": "2026-03-09",
                        "y": 92.0,
                        "y_hat": 74.0,
                    },
                ]
            ),
            "resolved": {
                "dataset": {
                    "target_col": "Com_CrudeOil",
                    "hist_exog_cols": ["GPRD_THREAT"],
                    "transformations_target": "diff",
                    "transformations_exog": "diff",
                },
                "training": {"input_size": 64, "max_steps": 400, "val_size": 4, "loss": "mse"},
                "cv": {"horizon": 2, "step_size": 1, "n_windows": 1, "gap": 0, "overlap_eval_policy": "by_cutoff_mean"},
            },
            "stage_config": {
                "star_hist_exog_cols_resolved": ["GPRD_THREAT"],
                "non_star_hist_exog_cols_resolved": [],
                "lowess_frac": 0.35,
                "lowess_delta": 0.01,
                "uncertainty": {"enabled": True, "sample_count": 50},
            },
            "retrieval_payloads": [],
            "derived": True,
        },
    ]

    output_path = postprocess._write_result_markdown(
        raw_batch_root=raw_batch_root,
        run_reports=run_reports,
        ret_plot=None,
        ret_folds_dir=None,
        nonret_plot=None,
        nonret_folds_dir=None,
    )
    markdown = output_path.read_text(encoding="utf-8")
    assert "baseline retrieval off 중 1개는 **`-ret` run의 base_prediction으로 재구성한 derived 결과**다." in markdown
    assert "AAForecast retrieval off 중 1개는 **`-ret` run의 base_prediction으로 재구성한 derived 결과**다." in markdown
    assert "적용후 (`AAForecast (derived non-ret)`)" in markdown
