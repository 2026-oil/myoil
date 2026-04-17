from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_config import load_app_config, loaded_config_for_jobs_fanout
import runtime_support.runner as runtime
from scripts import plot_fold_prediction_overlay as overlay


def _load_summary(summary_json: Path) -> dict[str, Any]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError("summary_json must contain a results list")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_for_config(config_path: str) -> str:
    return "ret" if "-ret" in Path(config_path).stem else "nonret"


def _iter_passed_run_entries(
    repo_root: Path, summary_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in summary_payload["results"]:
        if row.get("status") != "passed":
            continue
        config_path = str(row["config"])
        loaded = load_app_config(repo_root, config_path=config_path)
        group = _group_for_config(config_path)
        if loaded.jobs_fanout_specs:
            for spec in loaded.jobs_fanout_specs:
                variant = loaded_config_for_jobs_fanout(repo_root, loaded, spec)
                run_root = runtime._default_output_root(repo_root, variant).resolve()
                entries.append(
                    {
                        "config": config_path,
                        "group": group,
                        "jobs_route": spec.route_slug,
                        "canonical_run_root": str(run_root),
                        "run_name": run_root.name,
                        "derived": False,
                    }
                )
            continue
        run_root = runtime._default_output_root(repo_root, loaded).resolve()
        entries.append(
            {
                "config": config_path,
                "group": group,
                "jobs_route": loaded.active_jobs_route_slug,
                "canonical_run_root": str(run_root),
                "run_name": run_root.name,
                "derived": False,
            }
        )
    return entries


def _replace_symlink(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            raise ValueError(f"Refusing to replace non-symlink directory: {path}")
        path.unlink()
    path.symlink_to(target)


def _link_batch_artifacts(
    *,
    raw_batch_root: Path,
    summary_payload: dict[str, Any],
    entries: list[dict[str, Any]],
) -> None:
    runs_dir = raw_batch_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(str(summary_payload["log_dir"])).resolve()
    _replace_symlink(raw_batch_root / "logs", log_dir)
    for entry in entries:
        if bool(entry.get("derived")):
            continue
        target = Path(str(entry["canonical_run_root"])).resolve()
        if not target.exists():
            continue
        _replace_symlink(runs_dir / str(entry["run_name"]), target)


def _build_manifest(
    *,
    raw_batch_root: Path,
    summary_json: Path,
    summary_payload: dict[str, Any],
    entries: list[dict[str, Any]],
) -> Path:
    manifest_entries: list[dict[str, Any]] = []
    for entry in entries:
        canonical_root = Path(str(entry["canonical_run_root"])).resolve()
        linked_path = None if bool(entry.get("derived")) else raw_batch_root / "runs" / str(entry["run_name"])
        manifest_entries.append(
            {
                **entry,
                "canonical_run_root": str(canonical_root),
                "exists": canonical_root.exists(),
                "linked_run_path": None if linked_path is None else str(linked_path),
            }
        )
    payload = {
        "summary_json": str(summary_json.resolve()),
        "repo_root": str(REPO_ROOT),
        "log_dir": str(Path(str(summary_payload["log_dir"])).resolve()),
        "entries": manifest_entries,
    }
    manifest_path = raw_batch_root / "batch_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _format_metric(value: Any, *, percent: bool = False, digits: int = 2) -> str:
    number = _safe_float(value)
    if number is None:
        return "N/A"
    if percent:
        return f"{number * 100:.{digits}f}%"
    return f"{number:.{digits}f}"


def _format_delta(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "N/A"
    sign = "+" if number > 0 else ""
    return f"{sign}{number * 100:.2f}%p"


def _render_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered_rows = [["" if value is None else str(value) for value in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rendered_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _render_row(values: list[str]) -> str:
        padded = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    header_line = _render_row(headers)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [_render_row(row) for row in rendered_rows]
    return "\n".join([header_line, separator, *body])


def _relpath(path: Path, *, start: Path) -> str:
    try:
        return str(path.resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path.resolve())


def _infer_prediction_unit(run_reports: list[dict[str, Any]]) -> str:
    for report in run_reports:
        predictions = report.get("predictions")
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            continue
        ds_values = (
            pd.to_datetime(predictions["ds"], errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        if len(ds_values) < 2:
            continue
        day_delta = int((ds_values.iloc[1] - ds_values.iloc[0]).days)
        if day_delta == 7:
            return "주간 예측"
        if day_delta == 1:
            return "일간 예측"
        return f"{day_delta}일 단위 예측"
    return "N/A"


def _baseline_label(experiment: str) -> str:
    return "baseline" if experiment == "baseline" else "AAForecast"


def _experiment_sort_key(experiment: str) -> tuple[int, str]:
    return (0, experiment) if experiment == "baseline" else (1, experiment)


def _derive_nonret_run_name(run_name: str) -> str:
    if "-ret" in run_name:
        return run_name.replace("-ret", "", 1)
    return f"{run_name}_nonret"


def _is_aaforecast_entry(entry: dict[str, Any]) -> bool:
    return "aaforecast" in str(entry.get("config", "")).lower()


def _is_aaforecast_ret_entry(entry: dict[str, Any]) -> bool:
    return _is_aaforecast_entry(entry) and str(entry.get("group")) == "ret"


def _recompute_metrics_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    ordered = predictions.sort_values(["model", "fold_idx", "cutoff", "horizon_step", "ds"]).reset_index(drop=True)
    for (model, fold_idx, cutoff), group in ordered.groupby(["model", "fold_idx", "cutoff"], sort=False):
        actual = pd.to_numeric(group["y"], errors="coerce")
        predicted = pd.to_numeric(group["y_hat"], errors="coerce")
        valid = actual.notna() & predicted.notna()
        if not valid.any():
            continue
        metrics = runtime._compute_metrics(
            actual.loc[valid].reset_index(drop=True),
            predicted.loc[valid].reset_index(drop=True),
        )
        first = group.iloc[0]
        rows.append(
            {
                "model": str(model),
                "fold_idx": int(fold_idx),
                "cutoff": str(cutoff),
                **metrics,
                "requested_mode": first.get("requested_mode"),
                "validated_mode": first.get("validated_mode"),
            }
        )
    return pd.DataFrame(rows)


def _retrieval_payload_by_cutoff(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for payload in report.get("retrieval_payloads", []):
        cutoff = str(payload.get("cutoff", "")).strip()
        if cutoff:
            payloads[cutoff] = payload
    return payloads


def _build_derived_nonret_predictions(report: dict[str, Any]) -> pd.DataFrame:
    predictions = report["predictions"].copy()
    if predictions.empty:
        raise ValueError(f"Cannot derive non-ret report from empty predictions: {report['run_name']}")
    if "cutoff" not in predictions.columns:
        raise ValueError(f"Cannot derive non-ret report without cutoff column: {report['run_name']}")
    payload_by_cutoff = _retrieval_payload_by_cutoff(report)
    if not payload_by_cutoff:
        raise ValueError(
            f"Cannot derive non-ret AAForecast report without retrieval payloads: {report['run_name']}"
        )

    derived = predictions.copy()
    for cutoff, indices in derived.groupby("cutoff", sort=False).groups.items():
        payload = payload_by_cutoff.get(str(cutoff))
        if payload is None:
            raise ValueError(
                f"Missing retrieval payload for derived non-ret cutoff={cutoff!r} run={report['run_name']}"
            )
        base_prediction = payload.get("base_prediction")
        if not isinstance(base_prediction, list):
            raise ValueError(
                f"Retrieval payload missing base_prediction list for run={report['run_name']} cutoff={cutoff!r}"
            )
        group = derived.loc[list(indices)].sort_values(["horizon_step", "ds"])
        if len(base_prediction) != len(group):
            raise ValueError(
                "Derived non-ret base_prediction length mismatch "
                f"for run={report['run_name']} cutoff={cutoff!r}: "
                f"payload={len(base_prediction)} rows={len(group)}"
            )
        derived.loc[group.index, "y_hat"] = [float(value) for value in base_prediction]
        if "aaforecast_retrieval_enabled" in derived.columns:
            derived.loc[group.index, "aaforecast_retrieval_enabled"] = False
        if "aaforecast_retrieval_applied" in derived.columns:
            derived.loc[group.index, "aaforecast_retrieval_applied"] = False
        if "aaforecast_retrieval_skip_reason" in derived.columns:
            derived.loc[group.index, "aaforecast_retrieval_skip_reason"] = "derived_nonret_from_ret"
        if "aaforecast_retrieval_artifact" in derived.columns:
            derived.loc[group.index, "aaforecast_retrieval_artifact"] = ""
    return derived


def _derive_aaforecast_nonret_report(report: dict[str, Any]) -> dict[str, Any]:
    derived_predictions = _build_derived_nonret_predictions(report)
    derived_metrics = _recompute_metrics_frame(derived_predictions)
    derived_leaderboard = runtime._build_leaderboard(derived_metrics)
    run_name = _derive_nonret_run_name(str(report["run_name"]))
    entry = {
        "config": str(report["config"]).replace("-ret.yaml", ".yaml"),
        "group": "nonret",
        "jobs_route": report.get("jobs_route"),
        "canonical_run_root": str(report["run_root"]),
        "run_name": run_name,
        "derived": True,
        "derived_from_run_name": report["run_name"],
        "derived_from_group": report["group"],
    }
    return {
        **report,
        **entry,
        "entry": entry,
        "leaderboard": derived_leaderboard,
        "predictions": derived_predictions,
        "metrics": derived_metrics,
        "retrieval": False,
        "retrieval_payloads": [],
        "derived": True,
        "derived_from_run_name": report["run_name"],
        "display_experiment": "AAForecast (derived non-ret)",
        "plot_run_id": run_name,
    }


def _augment_with_derived_aaforecast_nonret_reports(
    entries: list[dict[str, Any]],
    run_reports: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_nonret = {
        (report["experiment"], str(report.get("backbone") or ""))
        for report in run_reports
        if report["experiment"] == "AAForecast" and not report["retrieval"]
    }
    derived_entries: list[dict[str, Any]] = []
    derived_reports: list[dict[str, Any]] = []
    for report in run_reports:
        if report["experiment"] != "AAForecast" or not report["retrieval"]:
            continue
        backbone = str(report.get("backbone") or "")
        if ("AAForecast", backbone) in existing_nonret:
            continue
        derived = _derive_aaforecast_nonret_report(report)
        derived_entries.append(derived["entry"])
        derived_reports.append(derived)
    return entries + derived_entries, run_reports + derived_reports


def _read_run_report(entry: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(str(entry["canonical_run_root"])).resolve()
    config_path = run_root / "config" / "config.resolved.json"
    leaderboard_path = run_root / "summary" / "leaderboard.csv"
    result_path = run_root / "summary" / "result.csv"
    folds_root = run_root / "summary" / "folds"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.resolved.json for feature-set report: {config_path}")
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing leaderboard.csv for feature-set report: {leaderboard_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result.csv for feature-set report: {result_path}")

    resolved = _load_json(config_path)
    leaderboard = pd.read_csv(leaderboard_path)
    predictions = pd.read_csv(result_path)

    metrics_frames: list[pd.DataFrame] = []
    if folds_root.exists():
        for metrics_path in sorted(folds_root.glob("fold_*/metrics.csv")):
            frame = pd.read_csv(metrics_path)
            if not frame.empty:
                metrics_frames.append(frame)
    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()

    stage_config_path = run_root / "aa_forecast" / "config" / "stage_config.json"
    stage_config = _load_json(stage_config_path) if stage_config_path.exists() else None

    retrieval_payloads: list[dict[str, Any]] = []
    retrieval_root = run_root / "aa_forecast" / "retrieval"
    if retrieval_root.exists():
        for payload_path in sorted(retrieval_root.glob("fold_*/*.json")):
            retrieval_payloads.append(_load_json(payload_path))

    experiment = "baseline" if "baseline" in str(entry["config"]) else "AAForecast"
    retrieval = entry["group"] == "ret"
    backbone = None
    if experiment == "AAForecast":
        if stage_config is not None and stage_config.get("backbone"):
            backbone = str(stage_config["backbone"])
        else:
            stem = Path(str(entry["config"])).stem.replace("aaforecast-", "")
            backbone = stem.replace("-ret", "").upper()
    return {
        **entry,
        "entry": entry,
        "run_root": run_root,
        "resolved": resolved,
        "leaderboard": leaderboard,
        "predictions": predictions,
        "metrics": metrics,
        "stage_config": stage_config,
        "retrieval_payloads": retrieval_payloads,
        "experiment": experiment,
        "retrieval": retrieval,
        "backbone": backbone,
        "derived": bool(entry.get("derived", False)),
        "display_experiment": experiment,
        "plot_run_id": entry["run_name"],
    }


def _collect_experiment_rows(run_reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report in run_reports:
        leaderboard = report["leaderboard"].copy()
        if leaderboard.empty:
            continue
        for _, metric_row in leaderboard.iterrows():
            rows.append(
                {
                    "experiment": report["experiment"],
                    "display_experiment": report.get("display_experiment", report["experiment"]),
                    "retrieval": "on" if report["retrieval"] else "off",
                    "run_name": report["run_name"],
                    "run_root": str(report["run_root"]),
                    "config": report["config"],
                    "model": str(metric_row["model"]),
                    "backbone": report["backbone"] if report["experiment"] == "AAForecast" else str(metric_row["model"]),
                    "mean_mape": _safe_float(metric_row.get("mean_fold_mape")),
                    "mean_nrmse": _safe_float(metric_row.get("mean_fold_nrmse")),
                    "mean_mae": _safe_float(metric_row.get("mean_fold_mae")),
                    "mean_r2": _safe_float(metric_row.get("mean_fold_r2")),
                    "fold_count": int(metric_row.get("fold_count", 0) or 0),
                    "derived": bool(report.get("derived", False)),
                }
            )
    return pd.DataFrame(rows)


def _collect_fold_rows(run_reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report in run_reports:
        metrics = report["metrics"]
        if metrics.empty:
            continue
        for _, metric_row in metrics.iterrows():
            rows.append(
                {
                    "experiment": report["experiment"],
                    "display_experiment": report.get("display_experiment", report["experiment"]),
                    "retrieval": "on" if report["retrieval"] else "off",
                    "model": str(metric_row["model"]),
                    "backbone": report["backbone"] if report["experiment"] == "AAForecast" else str(metric_row["model"]),
                    "fold_idx": int(metric_row["fold_idx"]),
                    "cutoff": str(metric_row["cutoff"]),
                    "mape": _safe_float(metric_row.get("MAPE")),
                    "nrmse": _safe_float(metric_row.get("NRMSE")),
                    "mae": _safe_float(metric_row.get("MAE")),
                    "r2": _safe_float(metric_row.get("R2")),
                }
            )
    return pd.DataFrame(rows)


def _collect_last_fold_rows(run_reports: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for report in run_reports:
        predictions = report["predictions"]
        if predictions.empty:
            continue
        frame = predictions.copy()
        frame["fold_idx"] = pd.to_numeric(frame["fold_idx"], errors="coerce")
        max_fold = frame["fold_idx"].dropna().max()
        if pd.isna(max_fold):
            continue
        last_fold = frame[frame["fold_idx"] == max_fold].copy()
        if last_fold.empty:
            continue
        last_fold["abs_pct_error"] = (
            (pd.to_numeric(last_fold["y_hat"], errors="coerce") - pd.to_numeric(last_fold["y"], errors="coerce")).abs()
            / pd.to_numeric(last_fold["y"], errors="coerce")
        )
        for _, pred_row in last_fold.iterrows():
            rows.append(
                {
                    "experiment": report["experiment"],
                    "display_experiment": report.get("display_experiment", report["experiment"]),
                    "retrieval": "on" if report["retrieval"] else "off",
                    "model": str(pred_row["model"]),
                    "backbone": report["backbone"] if report["experiment"] == "AAForecast" else str(pred_row["model"]),
                    "fold_idx": int(pred_row["fold_idx"]),
                    "cutoff": str(pred_row["cutoff"]),
                    "horizon_step": int(pred_row["horizon_step"]),
                    "actual": _safe_float(pred_row.get("y")),
                    "y_hat": _safe_float(pred_row.get("y_hat")),
                    "abs_pct_error": _safe_float(pred_row.get("abs_pct_error")),
                }
            )
    return pd.DataFrame(rows)


def _report_core_settings(run_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_reports:
        raise ValueError("Cannot build feature-set report without run reports")
    resolved = run_reports[0]["resolved"]
    dataset = resolved["dataset"]
    training = resolved["training"]
    cv = resolved["cv"]
    hist_exog = list(dataset.get("hist_exog_cols") or [])
    return {
        "target_col": dataset.get("target_col", "N/A"),
        "prediction_unit": _infer_prediction_unit(run_reports),
        "horizon": cv.get("horizon", "N/A"),
        "step_size": cv.get("step_size", "N/A"),
        "n_windows": cv.get("n_windows", "N/A"),
        "gap": cv.get("gap", "N/A"),
        "overlap_eval_policy": cv.get("overlap_eval_policy", "N/A"),
        "loss": training.get("loss", "N/A"),
        "input_size": training.get("input_size", "N/A"),
        "val_size": training.get("val_size", "N/A"),
        "max_steps": training.get("max_steps", "N/A"),
        "transformations_target": dataset.get("transformations_target", "N/A"),
        "transformations_exog": dataset.get("transformations_exog", "N/A"),
        "hist_exog_cols": hist_exog,
    }


def _collect_model_rosters(experiment_rows: pd.DataFrame) -> dict[str, list[str]]:
    roster: dict[str, list[str]] = {}
    for experiment in ["baseline", "AAForecast"]:
        for retrieval in ["off", "on"]:
            key = f"{experiment}:{retrieval}"
            frame = experiment_rows[
                (experiment_rows["experiment"] == experiment)
                & (experiment_rows["retrieval"] == retrieval)
            ]
            roster[key] = sorted(frame["backbone"].dropna().astype(str).unique().tolist())
    return roster


def _build_plot_frame_from_report(report: dict[str, Any]) -> pd.DataFrame:
    frame = report["predictions"].copy()
    if frame.empty:
        return frame
    run_id = str(report.get("plot_run_id", report["run_name"]))
    frame["run_id"] = run_id
    frame["run_root"] = str(report["run_root"])
    return overlay._annotate_series_identity(frame, run_id=run_id)


def _render_result_markdown(
    *,
    raw_batch_root: Path,
    run_reports: list[dict[str, Any]],
    ret_plot: Path | None,
    ret_folds_dir: Path | None,
    nonret_plot: Path | None,
    nonret_folds_dir: Path | None,
) -> str:
    experiment_rows = _collect_experiment_rows(run_reports)
    if experiment_rows.empty:
        raise ValueError("Cannot build feature-set report because no experiment rows were found")
    fold_rows = _collect_fold_rows(run_reports)
    last_fold_rows = _collect_last_fold_rows(run_reports)
    settings = _report_core_settings(run_reports)
    roster = _collect_model_rosters(experiment_rows)

    stage_config = next(
        (report["stage_config"] for report in run_reports if report["stage_config"] is not None),
        None,
    )
    retrieval_payloads = [
        payload for report in run_reports for payload in report.get("retrieval_payloads", [])
    ]

    overall_best = experiment_rows.sort_values("mean_mape", ascending=True).iloc[0]
    retrieval_off_best = experiment_rows[experiment_rows["retrieval"] == "off"].sort_values(
        "mean_mape", ascending=True
    )
    baseline_best = experiment_rows[experiment_rows["experiment"] == "baseline"].sort_values(
        "mean_mape", ascending=True
    )

    direct_rows: list[list[Any]] = []
    direct_improvements = 0
    direct_total = 0
    for model in sorted(set(experiment_rows["backbone"].tolist())):
        for retrieval in ["off", "on"]:
            baseline_frame = experiment_rows[
                (experiment_rows["experiment"] == "baseline")
                & (experiment_rows["retrieval"] == retrieval)
                & (experiment_rows["backbone"] == model)
            ]
            aa_frame = experiment_rows[
                (experiment_rows["experiment"] == "AAForecast")
                & (experiment_rows["retrieval"] == retrieval)
                & (experiment_rows["backbone"] == model)
            ]
            if baseline_frame.empty or aa_frame.empty:
                continue
            baseline_row = baseline_frame.iloc[0]
            aa_row = aa_frame.iloc[0]
            delta = aa_row["mean_mape"] - baseline_row["mean_mape"]
            if delta < 0:
                direct_improvements += 1
            direct_total += 1
            direct_rows.extend(
                [
                    [
                        model,
                        retrieval,
                        "적용전 (`baseline`)",
                        _format_metric(baseline_row["mean_mape"], percent=True),
                        "",
                        _format_metric(baseline_row["mean_nrmse"]),
                        _format_metric(baseline_row["mean_mae"]),
                        _format_metric(baseline_row["mean_r2"]),
                    ],
                    [
                        "",
                        "",
                        f"적용후 (`{aa_row['display_experiment']}`)",
                        _format_metric(aa_row["mean_mape"], percent=True),
                        _format_delta(delta),
                        _format_metric(aa_row["mean_nrmse"]),
                        _format_metric(aa_row["mean_mae"]),
                        _format_metric(aa_row["mean_r2"]),
                    ],
                ]
            )

    integrated_rows = []
    ranked = experiment_rows.sort_values(
        ["mean_mape", "mean_nrmse", "mean_mae", "model"], ascending=[True, True, True, True]
    ).reset_index(drop=True)
    for idx, row in ranked.iterrows():
        integrated_rows.append(
            [
                idx + 1,
                row["display_experiment"],
                row["backbone"],
                row["retrieval"],
                _format_metric(row["mean_mape"], percent=True),
                _format_metric(row["mean_nrmse"]),
                _format_metric(row["mean_mae"]),
                _format_metric(row["mean_r2"]),
            ]
        )

    retrieval_delta_rows: list[list[Any]] = []
    retrieval_improved = 0
    retrieval_total = 0
    for experiment in ["baseline", "AAForecast"]:
        for model in sorted(
            experiment_rows[experiment_rows["experiment"] == experiment]["backbone"].dropna().astype(str).unique()
        ):
            off_frame = experiment_rows[
                (experiment_rows["experiment"] == experiment)
                & (experiment_rows["retrieval"] == "off")
                & (experiment_rows["backbone"] == model)
            ]
            on_frame = experiment_rows[
                (experiment_rows["experiment"] == experiment)
                & (experiment_rows["retrieval"] == "on")
                & (experiment_rows["backbone"] == model)
            ]
            if off_frame.empty or on_frame.empty:
                continue
            off_row = off_frame.iloc[0]
            on_row = on_frame.iloc[0]
            delta = on_row["mean_mape"] - off_row["mean_mape"]
            if delta < 0:
                retrieval_improved += 1
            retrieval_total += 1
            retrieval_delta_rows.append(
                [
                    experiment,
                    model,
                    _format_metric(off_row["mean_mape"], percent=True),
                    _format_metric(on_row["mean_mape"], percent=True),
                    _format_delta(delta),
                    _format_metric(off_row["mean_nrmse"]),
                    _format_metric(on_row["mean_nrmse"]),
                    _format_metric(off_row["mean_mae"]),
                    _format_metric(on_row["mean_mae"]),
                ]
            )

    fold_summary_rows: list[list[Any]] = []
    fold_note = "이번 실행에는 fold metrics가 없어 fold별 요약을 생략했다."
    if not fold_rows.empty:
        fold_note = None
        pivot_columns = sorted(fold_rows["fold_idx"].unique().tolist())
        per_combo = fold_rows.sort_values(
            ["mape", "nrmse", "mae", "model"], ascending=[True, True, True, True]
        )
        for _, group in per_combo.groupby(["experiment", "backbone", "retrieval"], sort=False):
            row = group.iloc[0]
            display = [row["experiment"], row["backbone"], row["retrieval"]]
            mape_by_fold = {
                int(item["fold_idx"]): _format_metric(item["mape"], percent=True)
                for _, item in group.sort_values("fold_idx").iterrows()
            }
            for fold_idx in pivot_columns:
                display.append(mape_by_fold.get(fold_idx, "N/A"))
            display.append(_format_metric(group["mape"].mean(), percent=True))
            fold_summary_rows.append(display)

    last_fold_comparison_rows: list[list[Any]] = []
    last_fold_note = "이번 실행에는 마지막 fold 예측 비교용 데이터가 없어 섹션을 축약했다."
    best_h2 = None
    if not last_fold_rows.empty:
        last_fold_note = None
        pivot = last_fold_rows.pivot_table(
            index=["experiment", "display_experiment", "backbone", "retrieval", "fold_idx", "cutoff"],
            columns="horizon_step",
            values=["actual", "y_hat", "abs_pct_error"],
            aggfunc="first",
        ).reset_index()
        pivot.columns = [
            f"{left}_{right}" if right else str(left)
            for left, right in pivot.columns.to_flat_index()
        ]
        if "abs_pct_error_2" in pivot.columns:
            pivot = pivot.sort_values(["abs_pct_error_2", "abs_pct_error_1"], ascending=[True, True])
            best_h2 = pivot.iloc[0]
        for idx, row in pivot.reset_index(drop=True).iterrows():
            last_fold_comparison_rows.append(
                [
                    idx + 1,
                    row["display_experiment"],
                    row["backbone"],
                    row["retrieval"],
                    _format_metric(row.get("actual_1")),
                    _format_metric(row.get("y_hat_1")),
                    _format_metric(row.get("abs_pct_error_1"), percent=True),
                    _format_metric(row.get("actual_2")),
                    _format_metric(row.get("y_hat_2")),
                    _format_metric(row.get("abs_pct_error_2"), percent=True),
                ]
            )

    retrieval_applied_count = sum(1 for payload in retrieval_payloads if payload.get("retrieval_applied"))
    retrieval_topk_all_one = bool(retrieval_payloads) and all(
        int(payload.get("top_k_used", 0) or 0) == 1 for payload in retrieval_payloads
    )

    lines: list[str] = []
    lines.append("# 01. 핵심쟁점")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**baseline 대비 AAForecast 적용 효과**와 **retrieval on/off 차이**를 backbone별로 정리")
    lines.append("")
    lines.append("# 02. 데이터 및 모델 세팅")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"- **예측 타깃:** {settings['target_col']}")
    lines.append(f"- **예측 단위:** {settings['prediction_unit']}")
    lines.append(
        f"- **평가 구조:** {settings['n_windows']}개 rolling TSCV (`h={settings['horizon']}`, `step={settings['step_size']}`, `gap={settings['gap']}`)"
    )
    lines.append(f"- **overlap_eval_policy:** `{settings['overlap_eval_policy']}`")
    lines.append(f"- **loss:** `{settings['loss']}`")
    lines.append(f"- **input_size:** `{settings['input_size']}`")
    lines.append(f"- **val_size:** `{settings['val_size']}`")
    lines.append(f"- **max_steps:** `{settings['max_steps']}`")
    lines.append(
        f"- **transformations:** `target={settings['transformations_target']}`, `exog={settings['transformations_exog']}`"
    )
    lines.append("- **비교축:** baseline vs AAForecast / backbone / retrieval off vs on")
    lines.append("")
    lines.append("## 02-01. 공통 hist_exog_cols")
    lines.append("")
    for column in settings["hist_exog_cols"]:
        lines.append(f"- `{column}`")
    lines.append("")
    lines.append("## 02-02. AAForecast 내부 분리 기준")
    lines.append("")
    if stage_config is None:
        lines.append("- 이번 실행에서는 AAForecast stage config artifact를 찾지 못했다.")
    else:
        star_cols = stage_config.get("star_hist_exog_cols_resolved") or []
        non_star_cols = stage_config.get("non_star_hist_exog_cols_resolved") or []
        lines.append(f"- **STAR 적용 exog:** `{', '.join(star_cols) if star_cols else 'N/A'}`")
        lines.append(f"- **non-STAR exog:** `{', '.join(non_star_cols) if non_star_cols else 'N/A'}`")
        lines.append(
            f"- **LOWESS:** `lowess_frac={stage_config.get('lowess_frac', 'N/A')}`, `lowess_delta={stage_config.get('lowess_delta', 'N/A')}`"
        )
        uncertainty = stage_config.get("uncertainty") or {}
        enabled = bool(uncertainty.get("enabled"))
        sample_count = uncertainty.get("sample_count", "N/A")
        lines.append(
            f"- **uncertainty:** {'활성화' if enabled else '비활성화'} (`sample_count={sample_count}`)"
        )
    lines.append("")
    lines.append("## 02-03. 실험군 구성")
    lines.append("")
    lines.append("- **baseline**")
    lines.append(
        f"    - retrieval off: `{', '.join(roster['baseline:off']) if roster['baseline:off'] else '없음'}`"
    )
    lines.append(
        f"    - retrieval on: `{', '.join(roster['baseline:on']) if roster['baseline:on'] else '없음'}`"
    )
    lines.append("- **AAForecast**")
    lines.append(
        f"    - retrieval off: `{', '.join(roster['AAForecast:off']) if roster['AAForecast:off'] else '없음'}`"
    )
    lines.append(
        f"    - retrieval on: `{', '.join(roster['AAForecast:on']) if roster['AAForecast:on'] else '없음'}`"
    )
    lines.append("")
    lines.append("# 03. 실험 설계 및 적용")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("- 모든 run은 동일 타깃, 동일 데이터, 동일 CV 설정 위에서 비교했다.")
    lines.append("- baseline은 **순수 backbone 성능 비교용**이고, retrieval on 조건에서는 retrieval 설정이 활성화됐다.")
    lines.append("- AAForecast는 **STAR → anomaly-aware forecasting → uncertainty optimization** 경로를 포함한다.")
    derived_nonret_count = sum(
        1
        for report in run_reports
        if report["experiment"] == "AAForecast" and not report["retrieval"] and bool(report.get("derived"))
    )
    if derived_nonret_count:
        lines.append(
            f"- AAForecast retrieval off 중 {derived_nonret_count}개는 **`-ret` run의 base_prediction으로 재구성한 derived 결과**다."
        )
    if retrieval_payloads:
        lines.append(
            f"- AAForecast retrieval on run은 총 {len(retrieval_payloads)}개 fold artifact를 남겼고, **retrieval_applied=true {retrieval_applied_count}건**이었다."
        )
        if retrieval_topk_all_one:
            lines.append("- 확인된 retrieval artifact 기준으로 **모든 fold에서 `top_k_used=1`**이었다.")
    else:
        lines.append("- 이번 실행에서는 AAForecast retrieval artifact가 없어 retrieval 상세 추적은 생략했다.")
    lines.append("- 본문의 평균 성능은 각 run의 `summary/leaderboard.csv` 기준 fold 평균값을 사용했다.")
    lines.append("")
    lines.append("# 04. 실험(모델링) 결과")
    lines.append("")
    lines.append("### 인사이트")
    lines.append("")
    lines.append(
        f"- 전체 {len(experiment_rows)}개 비교행 기준 최저 평균 MAPE는 **{overall_best['experiment']} + {overall_best['backbone']} + retrieval {overall_best['retrieval']} ({_format_metric(overall_best['mean_mape'], percent=True)})**였다."
    )
    if not retrieval_off_best.empty:
        row = retrieval_off_best.iloc[0]
        lines.append(
            f"- retrieval off만 놓고 보면 최저 평균 MAPE는 **{row['experiment']} + {row['backbone']} ({_format_metric(row['mean_mape'], percent=True)})**였다."
        )
    if not baseline_best.empty:
        row = baseline_best.iloc[0]
        lines.append(
            f"- baseline만 놓고 보면 최저 평균 MAPE는 **{row['backbone']} + retrieval {row['retrieval']} ({_format_metric(row['mean_mape'], percent=True)})**였다."
        )
    lines.append(
        f"- baseline과 직접 비교 가능한 {direct_total}개 조건 중 **{direct_improvements}개 조건에서 AAForecast가 평균 MAPE를 개선**했다."
    )
    lines.append(
        f"- retrieval on은 직접 비교 가능한 {retrieval_total}개 backbone 조건 중 **{retrieval_improved}개 조건에서 평균 MAPE를 낮췄다**."
    )
    if best_h2 is not None:
        lines.append(
            f"- 마지막 available fold 기준 h2 최저 오차는 **{best_h2['experiment']} + {best_h2['backbone']} + retrieval {best_h2['retrieval']} ({_format_metric(best_h2.get('abs_pct_error_2'), percent=True)})**였다."
        )
    lines.append("")
    lines.append("## 04-01. Backbone별 평균 성능 비교")
    lines.append("")
    if direct_rows:
        lines.append(
            _render_markdown_table(
                ["Backbone", "Retrieval", "구분", "Mean MAPE", "△ MAPE", "Mean nRMSE", "Mean MAE", "Mean R2"],
                direct_rows,
            )
        )
    else:
        lines.append("직접 비교 가능한 baseline/AAForecast 짝이 없어 본 표는 비었다.")
    lines.append("")
    missing_baseline_models = sorted(set(roster["AAForecast:off"] + roster["AAForecast:on"]) - set(roster["baseline:off"] + roster["baseline:on"]))
    if missing_baseline_models:
        lines.append(
            f"- `{', '.join(missing_baseline_models)}`는 이번 raw 묶음에서 baseline 대응 run이 없어 통합 순위표에서만 직접 비교했다."
        )
        lines.append("")
    lines.append("## 04-02. 실험군별 통합 Table")
    lines.append("")
    lines.append(
        _render_markdown_table(
            ["Rank", "실험군", "Backbone", "Retrieval", "Mean MAPE", "Mean nRMSE", "Mean MAE", "Mean R2"],
            integrated_rows,
        )
    )
    lines.append("")
    lines.append("## 04-03. Retrieval on/off 변화")
    lines.append("")
    if retrieval_delta_rows:
        lines.append(
            _render_markdown_table(
                [
                    "실험군",
                    "Backbone",
                    "off Mean MAPE",
                    "on Mean MAPE",
                    "△ MAPE (on-off)",
                    "off Mean nRMSE",
                    "on Mean nRMSE",
                    "off Mean MAE",
                    "on Mean MAE",
                ],
                retrieval_delta_rows,
            )
        )
    else:
        lines.append("retrieval on/off 직접 비교 가능한 run 쌍이 없어 본 표는 비었다.")
    lines.append("")
    lines.append("## 04-04. Fold별 MAPE 요약")
    lines.append("")
    if fold_note is not None:
        lines.append(fold_note)
    else:
        if nonret_folds_dir is not None:
            nonret_regular = nonret_folds_dir / "regular" / "fold_000_predictions_overlay.png"
            if nonret_regular.exists():
                lines.append("Retrieval off fold overlay")
                lines.append("")
                lines.append(f"![nonret fold overlay]({_relpath(nonret_regular, start=raw_batch_root)})")
                lines.append("")
        if ret_folds_dir is not None:
            ret_regular = ret_folds_dir / "regular" / "fold_000_predictions_overlay.png"
            if ret_regular.exists():
                lines.append("Retrieval on fold overlay")
                lines.append("")
                lines.append(f"![ret fold overlay]({_relpath(ret_regular, start=raw_batch_root)})")
                lines.append("")
        if nonret_plot is not None and nonret_plot.exists():
            lines.append("- retrieval off concat")
            lines.append("")
            lines.append(f"![nonret continuous overlay]({_relpath(nonret_plot, start=raw_batch_root)})")
            lines.append("")
        if ret_plot is not None and ret_plot.exists():
            lines.append("- retrieval on concat")
            lines.append("")
            lines.append(f"![ret continuous overlay]({_relpath(ret_plot, start=raw_batch_root)})")
            lines.append("")
        headers = ["실험군", "Backbone", "Retrieval"] + [
            f"Fold {fold_idx}" for fold_idx in sorted(fold_rows['fold_idx'].unique().tolist())
        ] + ["Mean MAPE"]
        lines.append(_render_markdown_table(headers, fold_summary_rows))
        lines.append("")
        worst_fold = fold_rows.groupby("fold_idx")["mape"].mean().sort_values(ascending=False).head(1)
        if not worst_fold.empty:
            lines.append(
                f"- 평균 MAPE 기준으로는 **Fold {int(worst_fold.index[0])}**가 가장 어려운 구간이었다."
            )
            lines.append("")
    lines.append("## 04-05. 마지막 fold spike 구간 비교")
    lines.append("")
    if last_fold_note is not None:
        lines.append(last_fold_note)
    else:
        lines.append(
            _render_markdown_table(
                [
                    "Rank (h2 오차)",
                    "실험군",
                    "Backbone",
                    "Retrieval",
                    "h1 actual",
                    "h1 y_hat",
                    "h1 오차율",
                    "h2 actual",
                    "h2 y_hat",
                    "h2 오차율",
                ],
                last_fold_comparison_rows,
            )
        )
        lines.append("")
    lines.append("---")
    return "\n".join(lines).rstrip() + "\n"


def _write_result_markdown(
    *,
    raw_batch_root: Path,
    run_reports: list[dict[str, Any]],
    ret_plot: Path | None,
    ret_folds_dir: Path | None,
    nonret_plot: Path | None,
    nonret_folds_dir: Path | None,
) -> Path:
    output_path = raw_batch_root / "result.md"
    markdown = _render_result_markdown(
        raw_batch_root=raw_batch_root,
        run_reports=run_reports,
        ret_plot=ret_plot,
        ret_folds_dir=ret_folds_dir,
        nonret_plot=nonret_plot,
        nonret_folds_dir=nonret_folds_dir,
    )
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def _build_combined_frame(run_roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_root in run_roots:
        raw = overlay.load_forecasts_from_run(run_root)
        if raw.empty:
            continue
        frame = raw.copy()
        frame["run_id"] = run_root.name
        frame["run_root"] = str(run_root.resolve())
        frame = overlay._annotate_series_identity(frame, run_id=run_root.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    required = {"fold_idx", "ds", "y_hat", "train_end_ds"}
    missing = required.difference(combined.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Combined forecasts missing required columns: {missing_text}")
    return combined


def _reports_for_group(run_reports: list[dict[str, Any]], *, group: str) -> list[dict[str, Any]]:
    selected = [report for report in run_reports if ("ret" if report["retrieval"] else "nonret") == group]
    if group != "nonret":
        return selected
    baseline_reports = [report for report in selected if report["experiment"] == "baseline"]
    aa_reports = [report for report in selected if report["experiment"] == "AAForecast"]
    actual_backbones = {
        str(report.get("backbone") or "")
        for report in aa_reports
        if not bool(report.get("derived"))
    }
    filtered_aa = [
        report
        for report in aa_reports
        if (not bool(report.get("derived"))) or str(report.get("backbone") or "") not in actual_backbones
    ]
    return baseline_reports + filtered_aa


def _write_group_plot(
    *,
    raw_batch_root: Path,
    run_reports: list[dict[str, Any]],
    group: str,
    x_start: str | None,
    x_end: str | None,
) -> tuple[Path | None, Path | None]:
    selected_reports = _reports_for_group(run_reports, group=group)
    if not selected_reports:
        return None, None
    frames = [_build_plot_frame_from_report(report) for report in selected_reports]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return None, None
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return None, None
    run_roots = [Path(str(report["run_root"])).resolve() for report in selected_reports]

    # 1. Continuous overlay
    plot_dir = raw_batch_root / "plots" / group
    output_path = plot_dir / "all_folds_continuous_overlay.png"
    overlay.plot_continuous_series(
        combined,
        run_roots=run_roots,
        output_path=output_path,
        show_mean_band=False,
        alpha_per_run=0.9,
        x_start=x_start,
        x_end=x_end,
    )

    # 2. Fold overlay plots (regular + window_16)
    fold_dir = plot_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    regular_dir = fold_dir / "regular"
    window_dir = fold_dir / "window_16"
    regular_dir.mkdir(exist_ok=True)
    window_dir.mkdir(exist_ok=True)

    fold_paths = overlay.plot_folds(
        combined,
        fold_dir,
        show_mean_band=False,
        alpha_per_run=0.9,
        window_history_steps=16,
    )

    # Separate window_16 and regular plots
    for fp in fold_paths:
        if "window_16" in fp.name:
            fp.rename(window_dir / fp.name)
        else:
            fp.rename(regular_dir / fp.name)

    # 3. Create GIFs
    _create_gif(regular_dir, fold_dir / "regular.gif")
    _create_gif(window_dir, fold_dir / "window_16.gif")

    # 4. Create combined MP4
    _create_combined_mp4(regular_dir, window_dir, fold_dir / "regular_window16.mp4")

    return output_path, fold_dir


def _create_gif(source_dir: Path, output_path: Path) -> None:
    """Create GIF from PNG files in directory."""
    try:
        from PIL import Image
    except ImportError:
        return

    images = []
    png_files = sorted(source_dir.glob("fold_*.png"))
    for fp in png_files:
        images.append(Image.open(fp))

    if images:
        images[0].save(
            output_path, save_all=True, append_images=images[1:], duration=500, loop=0
        )


def _create_combined_mp4(
    regular_dir: Path, window_dir: Path, output_path: Path
) -> None:
    """Create MP4 with regular and window_16 plots side by side."""
    try:
        from PIL import Image
    except ImportError:
        return

    import subprocess
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Get list of fold indices
        regular_files = sorted(regular_dir.glob("fold_*.png"))
        if not regular_files:
            return

        for i, reg_fp in enumerate(regular_files):
            # Extract fold index
            stem = reg_fp.stem  # fold_000_predictions_overlay
            # Find corresponding window_16 file
            win_fp = window_dir / f"{stem}_window_16.png"

            if not win_fp.exists():
                continue

            # Open images
            regular_img = Image.open(reg_fp)
            window_img = Image.open(win_fp)

            # Get dimensions
            w1, h1 = regular_img.size
            w2, h2 = window_img.size

            # Resize to same height
            target_height = max(h1, h2)
            scale1 = target_height / h1
            scale2 = target_height / h2

            regular_resized = regular_img.resize((int(w1 * scale1), target_height))
            window_resized = window_img.resize((int(w2 * scale2), target_height))

            # Concatenate horizontally
            combined = Image.new("RGB", (w1 + w2, target_height))
            combined.paste(regular_resized, (0, 0))
            combined.paste(window_resized, (w1, 0))

            combined.save(temp_dir / f"frame_{i:03d}.png")

        # Check if frames were created
        frames = sorted(temp_dir.glob("frame_*.png"))
        if not frames:
            return

        # Create MP4 with ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "2",
                "-i",
                str(temp_dir / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    except Exception:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect feature_set_aaforecast batch outputs into a raw batch root and render graphs."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--raw-batch-root", type=Path, required=True)
    parser.add_argument("--x-start", default=None)
    parser.add_argument("--x-end", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary_json = args.summary_json.resolve()
    raw_batch_root = args.raw_batch_root.resolve()
    summary_payload = _load_summary(summary_json)
    entries = _iter_passed_run_entries(REPO_ROOT, summary_payload)

    raw_batch_root.mkdir(parents=True, exist_ok=True)
    _link_batch_artifacts(
        raw_batch_root=raw_batch_root,
        summary_payload=summary_payload,
        entries=entries,
    )
    run_reports = [_read_run_report(entry) for entry in entries]
    entries, run_reports = _augment_with_derived_aaforecast_nonret_reports(entries, run_reports)
    manifest_path = _build_manifest(
        raw_batch_root=raw_batch_root,
        summary_json=summary_json,
        summary_payload=summary_payload,
        entries=entries,
    )
    ret_plot, ret_folds_dir = _write_group_plot(
        raw_batch_root=raw_batch_root,
        run_reports=run_reports,
        group="ret",
        x_start=args.x_start,
        x_end=args.x_end,
    )
    nonret_plot, nonret_folds_dir = _write_group_plot(
        raw_batch_root=raw_batch_root,
        run_reports=run_reports,
        group="nonret",
        x_start=args.x_start,
        x_end=args.x_end,
    )
    result_md = _write_result_markdown(
        raw_batch_root=raw_batch_root,
        run_reports=run_reports,
        ret_plot=ret_plot,
        ret_folds_dir=ret_folds_dir,
        nonret_plot=nonret_plot,
        nonret_folds_dir=nonret_folds_dir,
    )

    print(f"batch_manifest={manifest_path}")
    print(f"ret_continuous_plot={ret_plot}")
    print(f"ret_folds_dir={ret_folds_dir}")
    print(f"nonret_continuous_plot={nonret_plot}")
    print(f"nonret_folds_dir={nonret_folds_dir}")
    print(f"result_md={result_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
