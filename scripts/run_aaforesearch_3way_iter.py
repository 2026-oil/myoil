from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import yaml

from app_config import load_app_config, loaded_config_for_jobs_fanout
from runtime_support.runner import _default_output_root, _summary_job_roots

ALLOWED_EXOG = [
    "GPRD_THREAT",
    "BS_Core_Index_A",
    "GPRD",
    "GPRD_ACT",
    "BS_Core_Index_B",
    "BS_Core_Index_C",
    "Idx_OVX",
    "Com_LMEX",
    "Com_BloombergCommodity_BCOM",
    "Idx_DxyUSD",
]


@dataclass(frozen=True)
class CaseSpec:
    key: str
    config_path: Path
    gpu_id: int


EXPECTED_INPUT_SIZE = 64
RUNTIME_LIMIT_SECONDS = 1200.0


CASE_SPECS = [
    CaseSpec("baseline", REPO_ROOT / "yaml/experiment/feature_set_aaforecast/baseline.yaml", 0),
    CaseSpec("aa_gru", REPO_ROOT / "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml", 1),
    CaseSpec("aa_informer", REPO_ROOT / "yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml", 0),
]


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("iter_%Y%m%d_%H%M%S")


def _bundle_root(run_tag: str) -> Path:
    return REPO_ROOT / "runs" / run_tag


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _expected_run_root(config_path: Path) -> Path:
    loaded = load_app_config(REPO_ROOT, config_path=str(config_path))
    if loaded.jobs_fanout_specs:
        if len(loaded.jobs_fanout_specs) != 1:
            raise ValueError(f"{config_path} expected exactly one jobs fanout route")
        loaded = loaded_config_for_jobs_fanout(REPO_ROOT, loaded, loaded.jobs_fanout_specs[0])
    return _default_output_root(REPO_ROOT, loaded)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_controls_from_yaml(config_path: Path) -> dict[str, Any]:
    raw = _load_yaml(config_path)
    dataset = raw.get("dataset", {})
    return {
        "target_ok": dataset.get("target_col") == "Com_BrentCrudeOil",
        "hist_exog_exact": dataset.get("hist_exog_cols", []) == ALLOWED_EXOG,
        "hist_exog_count": len(dataset.get("hist_exog_cols", [])),
        "hist_exog_cols": dataset.get("hist_exog_cols", []),
    }


def _resolved_controls(run_root: Path) -> dict[str, Any]:
    resolved = _read_json(run_root / "config" / "config.resolved.json")
    dataset = resolved.get("dataset", {})
    runtime = resolved.get("runtime", {})
    training = resolved.get("training", {})
    cv = resolved.get("cv", {})
    return {
        "target_col": dataset.get("target_col"),
        "hist_exog_cols": dataset.get("hist_exog_cols", []),
        "target_ok": dataset.get("target_col") == "Com_BrentCrudeOil",
        "hist_exog_exact": dataset.get("hist_exog_cols", []) == ALLOWED_EXOG,
        "transformations_target": runtime.get("transformations_target"),
        "transformations_exog": runtime.get("transformations_exog"),
        "input_size": training.get("input_size"),
        "horizon": cv.get("horizon"),
        "n_windows": cv.get("n_windows"),
        "settings_ok": cv.get("horizon") == 2 and cv.get("n_windows") == 1 and training.get("input_size") == EXPECTED_INPUT_SIZE,
    }


def _load_forecasts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"forecast file is empty: {path}")
    return frame


def _fold_eval(frame: pd.DataFrame, fold_idx: int) -> dict[str, Any]:
    fold = frame.loc[frame["fold_idx"] == fold_idx].sort_values("horizon_step").copy()
    first_two = fold.loc[fold["horizon_step"].isin([1, 2])].copy()
    first_two["abs_pct_err"] = (first_two["y_hat"] - first_two["y"]).abs() / first_two["y"].abs()
    within15 = bool((first_two["abs_pct_err"] <= 0.15).all())
    return {
        "fold_idx": int(fold_idx),
        "cutoff": str(fold["cutoff"].iloc[0]),
        "horizons": [
            {
                "h": int(row["horizon_step"]),
                "y": float(row["y"]),
                "y_hat": float(row["y_hat"]),
                "abs_pct_err": float(row["abs_pct_err"]) if "abs_pct_err" in row else float(abs(row["y_hat"] - row["y"]) / abs(row["y"])),
                "within15": bool((float(abs(row["y_hat"] - row["y"]) / abs(row["y"]))) <= 0.15),
            }
            for _, row in first_two.iterrows()
        ],
        "y_hat2_gt_y_hat1": bool(first_two.iloc[1]["y_hat"] > first_two.iloc[0]["y_hat"]) if len(first_two) == 2 else False,
        "within15": within15,
        "mean_abs_pct_err_h1h2": float(first_two["abs_pct_err"].mean()) if not first_two.empty else float("inf"),
    }


def _fold_eval_from_summary_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "horizon_step" not in frame.columns:
        return None
    fold = frame.sort_values("horizon_step").copy()
    first_two = fold.loc[fold["horizon_step"].isin([1, 2])].copy()
    if len(first_two) < 2:
        return None
    first_two["abs_pct_err"] = (first_two["y_hat"] - first_two["y"]).abs() / first_two["y"].abs()
    within15 = bool((first_two["abs_pct_err"] <= 0.15).all())
    return {
        "cutoff": str(first_two["cutoff"].iloc[0]) if "cutoff" in first_two.columns else None,
        "horizons": [
            {
                "h": int(row["horizon_step"]),
                "y": float(row["y"]),
                "y_hat": float(row["y_hat"]),
                "abs_pct_err": float(row["abs_pct_err"]),
                "within15": bool(float(row["abs_pct_err"]) <= 0.15),
            }
            for _, row in first_two.iterrows()
        ],
        "y_hat2_gt_y_hat1": bool(first_two.iloc[1]["y_hat"] > first_two.iloc[0]["y_hat"]),
        "within15": within15,
        "mean_abs_pct_err_h1h2": float(first_two["abs_pct_err"].mean()),
    }


def _model_summary(run_root: Path, model_name: str, forecast_path: Path) -> dict[str, Any]:
    summary = pd.read_csv(run_root / "summary" / "leaderboard.csv")
    summary_row = summary.loc[summary["model"] == model_name].iloc[0].to_dict()
    test2_path = run_root / "summary" / "test_2" / "leaderboard.csv"
    test2_row = None
    if test2_path.exists():
        test2 = pd.read_csv(test2_path)
        match = test2.loc[test2["model"] == model_name]
        if not match.empty:
            test2_row = match.iloc[0].to_dict()
    forecasts = _load_forecasts(forecast_path)
    latest_fold = int(forecasts["fold_idx"].max())
    latest_eval = _fold_eval(forecasts, latest_fold)
    test2_eval = _fold_eval_from_summary_result(run_root / "summary" / "test_2" / "result.csv")
    strict_pass = latest_eval["y_hat2_gt_y_hat1"] and latest_eval["within15"]
    fallback_pass = bool(test2_eval and test2_eval["within15"])
    rank_score = (
        (650 if strict_pass else 450 if fallback_pass else 0)
        + (40 if latest_eval["y_hat2_gt_y_hat1"] else 0)
        - 120.0 * latest_eval["mean_abs_pct_err_h1h2"]
        - 12.0 * float(summary_row["mean_fold_mape"])
        - float(summary_row["mean_fold_nrmse"])
    )
    return {
        "run_root": str(run_root),
        "model": model_name,
        "summary": {
            "mape": float(summary_row["mean_fold_mape"]),
            "nrmse": float(summary_row["mean_fold_nrmse"]),
            "mae": float(summary_row["mean_fold_mae"]),
            "rmse": float(summary_row["mean_fold_rmse"]),
        },
        "summary_test_2": None
        if test2_row is None
        else {
            "mape": float(test2_row["mean_fold_mape"]),
            "nrmse": float(test2_row["mean_fold_nrmse"]),
            "mae": float(test2_row["mean_fold_mae"]),
            "rmse": float(test2_row["mean_fold_rmse"]),
        },
        "latest_fold": latest_eval,
        "test_2_fold": test2_eval,
        "strict_pass": strict_pass,
        "fallback_pass": fallback_pass,
        "rank_score": rank_score,
    }


def _discover_models(run_root: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for job_root in _summary_job_roots(run_root):
        for forecast_path in sorted((job_root / "cv").glob("*_forecasts.csv")):
            frame = _load_forecasts(forecast_path)
            model_name = str(frame["model"].iloc[0])
            results[model_name] = _model_summary(run_root, model_name, forecast_path)
    if not results:
        raise FileNotFoundError(f"no forecast files found under {run_root / 'cv'}")
    return results


def _run_case(case: CaseSpec, logs_dir: Path) -> tuple[subprocess.Popen[bytes], Path, Path, float, Any]:
    expected_root = _expected_run_root(case.config_path)
    if expected_root.exists():
        shutil.rmtree(expected_root)
    log_path = logs_dir / f"{case.key}.log"
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env["CUDA_VISIBLE_DEVICES"] = str(case.gpu_id)
    command = [
        "uv",
        "run",
        "python",
        "main.py",
        "--config",
        str(case.config_path.relative_to(REPO_ROOT)),
    ]
    handle = log_path.open("wb")
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return proc, expected_root, log_path, started, handle


def _move_result(expected_root: Path, bundle_dir: Path) -> Path:
    if not expected_root.exists():
        raise FileNotFoundError(f"expected run root missing after run: {expected_root}")
    destination = bundle_dir / expected_root.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(expected_root), str(destination))
    return destination


def _format_case_summary(case_name: str, result: dict[str, Any]) -> str:
    latest = result["latest_fold"]
    hbits = ", ".join(
        f"h{item['h']} err={item['abs_pct_err'] * 100:.2f}% y_hat={item['y_hat']:.4f}"
        for item in latest["horizons"]
    )
    return (
        f"{case_name}: mape={result['summary']['mape']:.4f}, "
        f"nrmse={result['summary']['nrmse']:.4f}, "
        f"strict={'PASS' if result['strict_pass'] else 'FAIL'}, "
        f"fallback={'PASS' if result['fallback_pass'] else 'FAIL'}, "
        f"uptrend={'PASS' if latest['y_hat2_gt_y_hat1'] else 'FAIL'}, {hbits}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and evaluate the Brent 3-way AAForecast bundle.")
    parser.add_argument("--iter-tag", default=None, help="Bundle folder tag under runs/. Defaults to UTC timestamp.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned roots and guards without executing.")
    args = parser.parse_args()

    run_tag = args.iter_tag or _now_tag()
    bundle_dir = _bundle_root(run_tag)
    logs_dir = bundle_dir / "logs"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        case.key: {
            "config_path": str(case.config_path.relative_to(REPO_ROOT)),
            "gpu_id": case.gpu_id,
            "expected_run_root": str(_expected_run_root(case.config_path)),
            "yaml_controls": _case_controls_from_yaml(case.config_path),
        }
        for case in CASE_SPECS
    }
    if args.dry_run:
        print(json.dumps({"run_tag": run_tag, "bundle_dir": str(bundle_dir), "plan": plan}, indent=2))
        return 0

    completed: dict[str, dict[str, Any]] = {}
    batch1 = [CASE_SPECS[0], CASE_SPECS[1]]
    batch2 = [CASE_SPECS[2]]

    for batch in (batch1, batch2):
        running = []
        for case in batch:
            proc, expected_root, log_path, started, handle = _run_case(case, logs_dir)
            running.append((case, proc, expected_root, log_path, started, handle))
        for case, proc, expected_root, log_path, started, handle in running:
            code = proc.wait()
            handle.close()
            elapsed = time.perf_counter() - started
            if code != 0:
                raise SystemExit(
                    json.dumps(
                        {
                            "status": "run_failed",
                            "case": case.key,
                            "config_path": str(case.config_path.relative_to(REPO_ROOT)),
                            "exit_code": code,
                            "log_path": str(log_path),
                            "bundle_dir": str(bundle_dir),
                        }
                    )
                )
            moved_root = _move_result(expected_root, bundle_dir)
            completed[case.key] = {
                "case": case.key,
                "config_path": str(case.config_path.relative_to(REPO_ROOT)),
                "log_path": str(log_path),
                "elapsed_seconds": elapsed,
                "run_root": str(moved_root),
                "resolved_controls": _resolved_controls(moved_root),
                "models": _discover_models(moved_root),
            }

    baseline_models = completed["baseline"]["models"]
    if "GRU" not in baseline_models or "Informer" not in baseline_models:
        raise SystemExit(json.dumps({"status": "baseline_models_missing", "found": sorted(baseline_models)}))
    baseline_candidates = {
        "plain_gru": baseline_models["GRU"],
        "plain_informer": baseline_models["Informer"],
    }
    baseline_name, baseline_case = max(
        baseline_candidates.items(),
        key=lambda item: item[1]["rank_score"],
    )

    cases = {
        "baseline": baseline_case,
        "aa_gru": completed["aa_gru"]["models"]["AAForecast"],
        "aa_informer": completed["aa_informer"]["models"]["AAForecast"],
    }
    runtime_ok = all(item["elapsed_seconds"] < RUNTIME_LIMIT_SECONDS for item in completed.values())
    config_ok = all(
        item["resolved_controls"]["target_ok"]
        and item["resolved_controls"]["hist_exog_exact"]
        and item["resolved_controls"]["settings_ok"]
        for item in completed.values()
    )
    informer_primary_success = cases["aa_informer"]["strict_pass"] or cases["aa_informer"]["fallback_pass"]
    aa_gru_beats_baseline = cases["aa_gru"]["rank_score"] > baseline_case["rank_score"]
    order_ok = cases["aa_informer"]["rank_score"] > cases["aa_gru"]["rank_score"] > baseline_case["rank_score"]
    baseline_gap = cases["aa_gru"]["rank_score"] - baseline_case["rank_score"]
    aa_informer_gap = cases["aa_informer"]["rank_score"] - cases["aa_gru"]["rank_score"]
    success = config_ok and runtime_ok and informer_primary_success and order_ok
    metric = 1000.0 if success else (
        cases["aa_informer"]["rank_score"]
        + (220.0 if cases["aa_informer"]["strict_pass"] else 140.0 if cases["aa_informer"]["fallback_pass"] else 0.0)
        + max(0.0, cases["aa_informer"]["rank_score"] - baseline_case["rank_score"])
        + 0.5 * max(0.0, cases["aa_gru"]["rank_score"] - baseline_case["rank_score"])
        + (80.0 if order_ok else 0.0)
        - (0.0 if config_ok else 250.0)
        - (0.0 if runtime_ok else 250.0)
    )
    comparison_order = sorted(cases.items(), key=lambda item: item[1]["rank_score"], reverse=True)

    result = {
        "status": "ok",
        "run_tag": run_tag,
        "bundle_dir": str(bundle_dir),
        "metric": metric,
        "success": success,
        "config_ok": config_ok,
        "runtime_ok": runtime_ok,
        "informer_primary_success": informer_primary_success,
        "aa_gru_beats_baseline": aa_gru_beats_baseline,
        "baseline_model": baseline_name,
        "order_ok": order_ok,
        "baseline_gap": baseline_gap,
        "aa_informer_gap": aa_informer_gap,
        "comparison_order": [name for name, _ in comparison_order],
        "plan": plan,
        "completed": completed,
        "baseline_models": baseline_candidates,
        "cases": cases,
        "summaries": {
            name: _format_case_summary(name, data) for name, data in cases.items()
        },
    }
    report_path = bundle_dir / "iteration_report.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
