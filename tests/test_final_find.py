from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from app_config import load_app_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_final_find_script():
    path = REPO_ROOT / "scripts" / "final_find.py"
    spec = importlib.util.spec_from_file_location("_final_find_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_aaforecast_exog_only_fixture(tmp_path: Path, *, insample_y_included: bool) -> Path:
    dataset_path = REPO_ROOT / "tests" / "fixtures" / "aa_forecast_runtime_smoke.csv"
    retrieval_path = tmp_path / "finding_aaforecast_smoke_retrieval.yaml"
    retrieval_path.write_text(
        "\n".join(
            [
                "retrieval:",
                "  star:",
                "    season_length: 4",
                "    lowess_frac: 0.6",
                "    lowess_delta: 0.01",
                "    thresh: 3.5",
                "    anomaly_tails:",
                "      upward:",
                "      - event",
                "      two_sided: []",
                "  mode: posthoc_blend",
                "  top_k: 2",
                "  recency_gap_steps: 0",
                "  trigger_quantile: 0.5",
                "  min_similarity: 0.0",
                "  similarity: cosine",
                "  temperature: 0.5",
                "  blend_floor: 0.0",
                "  blend_max: 0.5",
                "  use_uncertainty_gate: true",
                f"  insample_y_included: {'true' if insample_y_included else 'false'}",
                "  use_event_key: true",
                "  event_score_log_bonus_alpha: 0.0",
                "  event_score_log_bonus_cap: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    plugin_path = tmp_path / "finding_aaforecast_smoke_plugin.yaml"
    plugin_path.write_text(
        "\n".join(
            [
                "aa_forecast:",
                "  model: gru",
                "  tune_training: false",
                "  lowess_frac: 0.6",
                "  lowess_delta: 0.01",
                "  uncertainty:",
                "    enabled: true",
                "    sample_count: 2",
                "  retrieval:",
                "    enabled: true",
                f"    config_path: {retrieval_path.name}",
                "  model_params:",
                "    encoder_hidden_size: 16",
                "    encoder_n_layers: 1",
                "    encoder_dropout: 0.1",
                "    decoder_hidden_size: 16",
                "    decoder_layers: 1",
                "    season_length: 4",
                "    trend_kernel_size: 3",
                "    start_padding_enabled: true",
                "  star_anomaly_tails:",
                "    upward:",
                "    - event",
                "    two_sided: []",
                "  thresh: 3.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    main_path = tmp_path / "finding_aaforecast_smoke_main.yaml"
    main_path.write_text(
        "\n".join(
            [
                "task:",
                "  name: final_find_smoke",
                "dataset:",
                f"  path: {dataset_path}",
                "  target_col: target",
                "  dt_col: dt",
                "  hist_exog_cols:",
                "  - event",
                "  futr_exog_cols: []",
                "  static_exog_cols: []",
                "runtime:",
                "  random_seed: 7",
                "  transformations_target: diff",
                "  transformations_exog: diff",
                "training:",
                "  input_size: 4",
                "  batch_size: 1",
                "  valid_batch_size: 1",
                "  windows_batch_size: 8",
                "  inference_windows_batch_size: 8",
                "  max_steps: 1",
                "  val_size: 2",
                "  val_check_steps: 1",
                "  early_stop_patience_steps: -1",
                "  loss: mse",
                "  accelerator: cpu",
                "cv:",
                "  horizon: 2",
                "  step_size: 1",
                "  n_windows: 1",
                "  gap: 0",
                "  overlap_eval_policy: by_cutoff_mean",
                "scheduler:",
                "  gpu_ids:",
                "  - 0",
                "  max_concurrent_jobs: 1",
                "  worker_devices: 1",
                "aa_forecast:",
                "  enabled: true",
                f"  config_path: {plugin_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return main_path


def test_load_targets_json_accepts_strings_and_objects(tmp_path: Path) -> None:
    final_find = _load_final_find_script()
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(
        json.dumps(
            [
                "2024-01-15 00:00:00",
                {"label": "second", "candidate_end_ds": "2024-01-22"},
            ]
        ),
        encoding="utf-8",
    )
    targets = final_find._load_targets_json(targets_path)
    assert [target.label for target in targets] == ["2024-01-15 00:00:00", "second"]
    assert [target.normalized_candidate_end_ds for target in targets] == [
        "2024-01-15",
        "2024-01-22",
    ]


def test_final_find_cli_writes_csv_and_summary(tmp_path: Path) -> None:
    final_find = _load_final_find_script()
    main_path = _write_aaforecast_exog_only_fixture(
        tmp_path, insample_y_included=False
    )
    loaded = load_app_config(REPO_ROOT, config_path=main_path)
    source_df = pd.read_csv(REPO_ROOT / "tests" / "fixtures" / "aa_forecast_runtime_smoke.csv")
    source_df["dt"] = pd.to_datetime(source_df["dt"])
    train_df, future_df = final_find.resolve_train_future_frames(
        loaded, source_df, eval_slice="last_cv_fold"
    )
    stats = final_find.evaluate_exact_combo(
        loaded=loaded,
        train_df=train_df,
        future_df=future_df,
        input_size=4,
        hist_exog_cols=("event",),
        upward_cols=("event",),
        top_k=2,
    )
    assert not stats["error"]
    assert stats["neighbors"]
    target_path = tmp_path / "targets.json"
    target_path.write_text(
        json.dumps(
            [
                {
                    "label": "known-top-neighbor",
                    "candidate_end_ds": stats["neighbors"][0]["candidate_end_ds"],
                }
            ]
        ),
        encoding="utf-8",
    )
    out_csv = tmp_path / "final_find.csv"
    summary_json = tmp_path / "summary.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "final_find.py"),
            "--config",
            str(main_path),
            "--targets-json",
            str(target_path),
            "--input-sizes",
            "4",
            "--top-k-values",
            "2",
            "--output-csv",
            str(out_csv),
            "--summary-json",
            str(summary_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_csv.is_file()
    assert summary_json.is_file()
    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert any(row["match_found"] == "True" for row in rows)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["targets"][0]["best_match"] is not None
    assert summary["targets"][0]["best_match"]["matched_rank"] == 1


def test_final_find_rejects_non_exog_only_config(tmp_path: Path) -> None:
    main_path = _write_aaforecast_exog_only_fixture(tmp_path, insample_y_included=True)
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(["2024-01-15"]), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "final_find.py"),
            "--config",
            str(main_path),
            "--targets-json",
            str(target_path),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "insample_y_included=false" in proc.stderr


def test_final_find_dry_run_accepts_inline_pools_and_topk_values(tmp_path: Path) -> None:
    main_path = _write_aaforecast_exog_only_fixture(tmp_path, insample_y_included=False)
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(["2024-01-15"]), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "final_find.py"),
            "--config",
            str(main_path),
            "--targets-json",
            str(target_path),
            "--min-input-size",
            "3",
            "--max-input-size",
            "4",
            "--top-k-values",
            "1,2",
            "--exog-grid",
            "event",
            "--upward-grid",
            "event",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "input_sizes=[3, 4]" in proc.stdout
    assert "top_k_values=[1, 2]" in proc.stdout


def test_final_find_reports_missing_inline_grid_columns(tmp_path: Path) -> None:
    main_path = _write_aaforecast_exog_only_fixture(tmp_path, insample_y_included=False)
    target_path = tmp_path / "targets.json"
    target_path.write_text(json.dumps(["2024-01-15"]), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "final_find.py"),
            "--config",
            str(main_path),
            "--targets-json",
            str(target_path),
            "--exog-grid",
            "event,missing_col",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "--exog-grid contains column(s) missing from dataset" in proc.stderr


def test_final_find_optuna_mode_writes_summary(tmp_path: Path) -> None:
    main_path = _write_aaforecast_exog_only_fixture(tmp_path, insample_y_included=False)
    target_path = tmp_path / "targets.json"
    target_path.write_text(
        json.dumps([{"label": "target", "candidate_end_ds": "2024-01-15 00:00:00"}]),
        encoding="utf-8",
    )
    out_csv = tmp_path / "optuna_results.csv"
    summary_json = tmp_path / "optuna_summary.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "final_find.py"),
            "--config",
            str(main_path),
            "--targets-json",
            str(target_path),
            "--search-mode",
            "optuna",
            "--input-sizes",
            "3,4",
            "--top-k-values",
            "1,2",
            "--exog-grid",
            "event",
            "--upward-grid",
            "event",
            "--optuna-n-trials",
            "4",
            "--optuna-seed",
            "7",
            "--output-csv",
            str(out_csv),
            "--summary-json",
            str(summary_json),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["search_mode"] == "optuna"
    assert summary["study_summary"]["n_trials"] == 4


def test_final_find_optuna_resume_accumulates_trials(tmp_path: Path) -> None:
    main_path = _write_aaforecast_exog_only_fixture(tmp_path, insample_y_included=False)
    target_path = tmp_path / "targets.json"
    target_path.write_text(
        json.dumps([{"label": "target", "candidate_end_ds": "2024-01-15 00:00:00"}]),
        encoding="utf-8",
    )
    storage_path = tmp_path / "final_find_resume.sqlite3"
    summary_json = tmp_path / "resume_summary.json"
    out_csv = tmp_path / "resume_trials.csv"
    base_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "final_find.py"),
        "--config",
        str(main_path),
        "--targets-json",
        str(target_path),
        "--search-mode",
        "optuna",
        "--input-sizes",
        "3,4",
        "--top-k-values",
        "1,2",
        "--exog-grid",
        "event",
        "--upward-grid",
        "event",
        "--optuna-seed",
        "7",
        "--optuna-storage-path",
        str(storage_path),
        "--study-name",
        "resume-test",
        "--output-csv",
        str(out_csv),
        "--summary-json",
        str(summary_json),
    ]
    proc1 = subprocess.run(
        [*base_cmd, "--optuna-n-trials", "2"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc1.returncode == 0, proc1.stdout + proc1.stderr
    proc2 = subprocess.run(
        [*base_cmd, "--optuna-n-trials", "3", "--resume"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["search_mode"] == "optuna"
    assert summary["study_summary"]["n_trials"] == 5
    assert summary["study_summary"]["study_name"] == "resume-test"
