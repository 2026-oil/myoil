"""Overlay y_hat from many completed runs.

Reads per-run forecasts from either:
  - ``<run_root>/summary/folds/fold_*/predictions.csv`` (preferred), or
  - ``<run_root>/cv/*_forecasts.csv`` and ``<run_root>/scheduler/workers/*/cv/*_forecasts.csv``
    (rolling-origin files excluded), matching where the runner writes artifacts.

Use after a batch of runs (e.g. retrieval grid sweep) to see whether predictions
cluster together per fold. Actual ``y`` is drawn once; each run contributes a
semi-transparent ``y_hat`` line. Mean ± std across runs is optional.

Examples:
  uv run python scripts/plot_fold_prediction_overlay.py \\
    --runs-glob 'runs/sweep_retrieval_sweep_ret_*' \\
    --output-dir runs/_fold_overlay_plots

  uv run python scripts/plot_fold_prediction_overlay.py \\
    --run-dir runs/some_task_run_1 \\
    --run-dir runs/some_task_run_2 \\
    --output-dir runs/_compare_two

  uv run python scripts/plot_fold_prediction_overlay.py \\
    --continuous \\
    --runs-glob 'runs/raw/feature_set_aaforecast_aaforecast*-ret' \\
    --output-dir runs/raw/_ret_continuous_overlay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import runtime_support.runner as runtime  # noqa: E402


_DEFAULT_OUTPUT_DIR = Path("runs/_fold_overlay_plots")
_HPO_STUDY_COLORS = {
    "study-01": "#1f77b4",
    "study-02": "#ff7f0e",
    "study-03": "#2ca02c",
    "study-04": "#d62728",
    "study-05": "#9467bd",
    "study-06": "#8c564b",
}


def _forecast_job_roots(run_root: Path) -> list[Path]:
    """Match ``runtime_support.runner._summary_job_roots``: cv may live under worker dirs."""
    roots: list[Path] = []
    run_root = run_root.resolve()
    cv_top = run_root / "cv"
    if cv_top.is_dir():
        roots.append(run_root)
    workers_root = run_root / "scheduler" / "workers"
    if workers_root.is_dir():
        for path in sorted(workers_root.iterdir()):
            if (path / "cv").is_dir():
                roots.append(path)
    return roots


def _rglob_cv_forecasts(run_root: Path) -> list[Path]:
    """Any ``.../cv/*_forecasts.csv`` under run_root (handles unexpected nesting)."""
    out: list[Path] = []
    for path in sorted(run_root.rglob("*_forecasts.csv")):
        if "rolling_origin" in path.name:
            continue
        if path.parent.name != "cv":
            continue
        out.append(path)
    return out


def _diagnose_empty_run(run_root: Path) -> str:
    run_root = run_root.resolve()
    lines = [f"run_root={run_root}"]
    if not run_root.is_dir():
        lines.append("  (directory does not exist)")
        return "\n".join(lines)
    top = sorted(p.name for p in run_root.iterdir())
    lines.append(f"  top-level entries ({len(top)}): {', '.join(top[:40])}{' ...' if len(top) > 40 else ''}")
    fc = _rglob_cv_forecasts(run_root)
    lines.append(f"  rglob **/cv/*_forecasts.csv (excl. rolling): {len(fc)} file(s)")
    for p in fc[:5]:
        lines.append(f"    - {p.relative_to(run_root)}")
    if len(fc) > 5:
        lines.append("    - ...")
    summ = list(run_root.glob("summary/folds/fold_*/predictions.csv"))
    lines.append(f"  summary fold predictions.csv: {len(summ)} file(s)")
    return "\n".join(lines)


def _parse_fold_idx_from_path(path: Path) -> int:
    # .../folds/fold_003/predictions.csv -> 3
    name = path.parent.name
    if not name.startswith("fold_"):
        raise ValueError(f"Unexpected fold directory name: {name!r} ({path})")
    return int(name.removeprefix("fold_"))


def load_forecasts_from_run(run_root: Path) -> pd.DataFrame:
    """Load all fold rows for one run into a single frame."""
    run_root = run_root.resolve()
    summary_predictions = sorted(
        run_root.glob("summary/folds/fold_*/predictions.csv")
    )
    frames: list[pd.DataFrame] = []
    if summary_predictions:
        for path in summary_predictions:
            fold_idx = _parse_fold_idx_from_path(path)
            frame = pd.read_csv(path)
            if "fold_idx" not in frame.columns:
                frame = frame.copy()
                frame["fold_idx"] = fold_idx
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    seen_paths: set[str] = set()
    for job_root in _forecast_job_roots(run_root):
        cv_dir = job_root / "cv"
        for path in sorted(cv_dir.glob("*_forecasts.csv")):
            if "rolling_origin" in path.name:
                continue
            key = str(path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            frames.append(pd.read_csv(path))

    if not frames:
        for path in _rglob_cv_forecasts(run_root):
            key = str(path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            frames.append(pd.read_csv(path))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_study_index(study_root: Path) -> int:
    name = study_root.name
    if not name.startswith("study-"):
        raise ValueError(f"Unexpected study directory name: {name!r} ({study_root})")
    return int(name.removeprefix("study-"))


def _parse_trial_number(trial_root: Path) -> int:
    name = trial_root.name
    if not name.startswith("trial-"):
        raise ValueError(f"Unexpected trial directory name: {name!r} ({trial_root})")
    return int(name.removeprefix("trial-"))


def _trial_result_payload(trial_root: Path) -> dict[str, object]:
    result_path = trial_root / "trial_result.json"
    if not result_path.is_file():
        return {}
    return json.loads(result_path.read_text(encoding="utf-8"))


def _load_trial_fold_predictions(
    trial_root: Path,
    *,
    actual_run_root: Path,
    study_label: str,
    study_index: int,
    status: str,
    model_filter: str | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    trial_number = _parse_trial_number(trial_root)
    run_id = f"{study_label}/trial-{trial_number:04d}"
    for path in sorted(trial_root.glob("folds/fold_*/predictions.csv")):
        fold_idx = _parse_fold_idx_from_path(path)
        frame = pd.read_csv(path)
        frame = _filter_model(frame, model_filter)
        if frame.empty:
            continue
        normalized = frame.copy()
        if "fold_idx" not in normalized.columns:
            normalized["fold_idx"] = fold_idx
        normalized["run_id"] = run_id
        normalized["run_root"] = str(actual_run_root.resolve())
        normalized["trial_root"] = str(trial_root.resolve())
        normalized["study_label"] = study_label
        normalized["study_index"] = study_index
        normalized["trial_number"] = trial_number
        normalized["trial_status"] = status
        normalized["series_id"] = run_id
        normalized["display_label"] = run_id
        frames.append(normalized)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_hpo_trial_forecasts(
    hpo_run_root: Path,
    *,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    hpo_run_root = hpo_run_root.resolve()
    studies_root = hpo_run_root / "models" / model_name / "studies"
    study_roots = sorted(path for path in studies_root.glob("study-*") if path.is_dir())
    if not study_roots:
        raise ValueError(f"No study directories found under {studies_root}")

    combined_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    all_fold_indices: set[int] = set()
    status_counts: dict[str, int] = {}
    plotted_trial_count = 0

    for study_root in study_roots:
        study_index = _parse_study_index(study_root)
        study_label = study_root.name
        trial_roots = sorted(
            path for path in (study_root / "trials").glob("trial-*") if path.is_dir()
        )
        for trial_root in trial_roots:
            payload = _trial_result_payload(trial_root)
            status = str(payload.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            available_fold_indices = sorted(
                _parse_fold_idx_from_path(path)
                for path in trial_root.glob("folds/fold_*/predictions.csv")
            )
            all_fold_indices.update(available_fold_indices)
            coverage_rows.append(
                {
                    "study_label": study_label,
                    "study_index": study_index,
                    "trial_number": _parse_trial_number(trial_root),
                    "trial_id": trial_root.name,
                    "status": status,
                    "available_fold_count": len(available_fold_indices),
                    "trial_root": str(trial_root.resolve()),
                    "objective_value": payload.get("objective_value"),
                }
            )
            frame = _load_trial_fold_predictions(
                trial_root,
                actual_run_root=hpo_run_root,
                study_label=study_label,
                study_index=study_index,
                status=status,
                model_filter=model_name,
            )
            if frame.empty:
                continue
            plotted_trial_count += 1
            combined_frames.append(frame)

    if not coverage_rows:
        raise ValueError(f"No trials found under {studies_root}")

    sorted_fold_indices = sorted(all_fold_indices)
    coverage_frame = pd.DataFrame(coverage_rows)
    for fold_idx in sorted_fold_indices:
        column = f"has_fold_{fold_idx:03d}"
        trial_roots = coverage_frame["trial_root"].map(Path)
        coverage_frame[column] = trial_roots.map(
            lambda root, idx=fold_idx: (root / "folds" / f"fold_{idx:03d}" / "predictions.csv").is_file()
        )

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    summary_payload: dict[str, object] = {
        "hpo_run_root": str(hpo_run_root),
        "model_name": model_name,
        "study_count": len(study_roots),
        "trial_count": int(len(coverage_frame)),
        "plotted_trial_count": plotted_trial_count,
        "fold_indices": sorted_fold_indices,
        "status_counts": status_counts,
        "folds": {},
    }
    for fold_idx in sorted_fold_indices:
        plotted_count = int(coverage_frame[f"has_fold_{fold_idx:03d}"].sum())
        summary_payload["folds"][f"fold_{fold_idx:03d}"] = {
            "plotted_trial_count": plotted_count,
            "skipped_trial_count": int(len(coverage_frame) - plotted_count),
        }
    return combined, coverage_frame, summary_payload


def _write_hpo_coverage_artifacts(
    *,
    coverage_frame: pd.DataFrame,
    summary_payload: Mapping[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "trial_fold_coverage.csv"
    summary_path = output_dir / "trial_fold_summary.json"
    coverage_frame.sort_values(
        ["study_index", "trial_number"], kind="stable"
    ).to_csv(coverage_path, index=False)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return coverage_path, summary_path


def discover_run_dirs(repo_root: Path, pattern: str) -> list[Path]:
    base = repo_root / pattern if not Path(pattern).is_absolute() else Path(pattern)
    if base.exists() and base.is_dir() and "*" not in pattern and "?" not in pattern:
        return [base]
    if any(ch in pattern for ch in "*?["):
        return sorted(repo_root.glob(pattern))
    return sorted(repo_root.glob(pattern))


def collect_run_roots(
    repo_root: Path,
    *,
    run_dirs: Iterable[Path],
    runs_glob: str | None,
    runs_list_file: Path | None,
) -> list[Path]:
    roots: list[Path] = []
    for p in run_dirs:
        roots.append((repo_root / p).resolve() if not p.is_absolute() else p)
    if runs_glob:
        roots.extend(discover_run_dirs(repo_root, runs_glob))
    if runs_list_file is not None:
        text = runs_list_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            roots.append(
                candidate.resolve()
                if candidate.is_absolute()
                else (repo_root / candidate).resolve()
            )
    # de-duplicate preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for r in roots:
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def _filter_model(frame: pd.DataFrame, model: str | None) -> pd.DataFrame:
    if model is None or "model" not in frame.columns:
        return frame
    return frame[frame["model"] == model].copy()


def _display_label_from_run_id(run_id: str) -> str:
    prefix = "feature_set_aaforecast_aaforecast_"
    normalized = run_id
    if normalized.startswith(prefix):
        normalized = normalized.removeprefix(prefix)
    pieces = normalized.split("-")
    backbone = pieces[0].upper()
    suffix = " RET" if "ret" in pieces[1:] else ""
    return f"{backbone}{suffix}"


def _annotate_series_identity(frame: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    annotated = frame.copy()
    normalized_run_id = run_id.lower()
    if normalized_run_id.startswith("feature_set_aaforecast_aaforecast_"):
        annotated["series_id"] = run_id
        annotated["display_label"] = f"AAForecast {_display_label_from_run_id(run_id)}"
        return annotated

    model_values = (
        annotated["model"].astype(str)
        if "model" in annotated.columns
        else pd.Series(["prediction"] * len(annotated), index=annotated.index)
    )
    family_label = "Baseline" if "baseline" in normalized_run_id else run_id
    suffix = " RET" if "-ret" in normalized_run_id else ""
    annotated["series_id"] = run_id + "::" + model_values
    annotated["display_label"] = family_label + " " + model_values + suffix
    return annotated


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.Series(pd.to_datetime(series, errors="coerce")).map(
        runtime._normalize_summary_timestamp
    )


def _connected_plot_frame(
    anchor_frame: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    value_col: str,
) -> pd.DataFrame:
    plot_frame = frame[["ds", value_col]].copy()
    plot_frame["ds"] = pd.to_datetime(plot_frame["ds"], errors="coerce")
    plot_frame[value_col] = pd.to_numeric(plot_frame[value_col], errors="coerce")
    plot_frame = plot_frame.dropna(subset=["ds", value_col]).reset_index(drop=True)
    if anchor_frame.empty or plot_frame.empty:
        return plot_frame
    anchor = anchor_frame.rename(columns={"y": value_col})
    return pd.concat([anchor, plot_frame], ignore_index=True)


def _frame_signature(frame: pd.DataFrame) -> tuple[tuple[str, float | None], ...]:
    normalized = frame.copy()
    normalized["ds"] = pd.to_datetime(normalized["ds"], errors="coerce")
    normalized["y"] = pd.to_numeric(normalized["y"], errors="coerce")
    normalized = normalized.dropna(subset=["ds"]).reset_index(drop=True)
    return tuple(
        (
            pd.Timestamp(ds).isoformat(),
            None if pd.isna(y) else float(y),
        )
        for ds, y in zip(normalized["ds"], normalized["y"], strict=True)
    )


def _validate_fold_alignment(fold_frame: pd.DataFrame, fold_idx: int) -> None:
    train_end_values = (
        pd.Series(fold_frame.get("train_end_ds"))
        .dropna()
        .pipe(_normalize_timestamp_series)
        .dropna()
        .drop_duplicates()
    )
    if len(train_end_values) > 1:
        values = ", ".join(pd.Timestamp(value).date().isoformat() for value in train_end_values)
        raise ValueError(
            f"Fold {fold_idx:03d} has inconsistent train_end_ds values across runs: {values}"
        )


def _resolve_actual_frames_for_fold(
    fold_frame: pd.DataFrame,
    *,
    fold_idx: int,
    history_steps_override: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_fold_alignment(fold_frame, fold_idx)
    reference_input = pd.DataFrame()
    reference_output = pd.DataFrame()
    resolved = False
    for run_root_text in fold_frame["run_root"].drop_duplicates().tolist():
        run_root = Path(str(run_root_text))
        run_frame = fold_frame[fold_frame["run_root"] == run_root_text].copy()
        input_actual_frame, output_actual_frame = runtime._summary_overlay_actual_frames(
            run_root,
            run_frame,
            history_steps_override=history_steps_override,
        )
        if input_actual_frame.empty and output_actual_frame.empty:
            continue
        if not resolved:
            reference_input = input_actual_frame.reset_index(drop=True)
            reference_output = output_actual_frame.reset_index(drop=True)
            resolved = True
            continue
        if _frame_signature(reference_input) != _frame_signature(input_actual_frame):
            raise ValueError(
                f"Fold {fold_idx:03d} has inconsistent input actual series across runs"
            )
        if _frame_signature(reference_output) != _frame_signature(output_actual_frame):
            raise ValueError(
                f"Fold {fold_idx:03d} has inconsistent output actual series across runs"
            )
    if not resolved:
        raise ValueError(
            f"Fold {fold_idx:03d} could not restore input/output actual frames from the selected runs"
        )
    return reference_input, reference_output


def _load_shared_actual_series(run_roots: Iterable[Path]) -> pd.DataFrame:
    dataset_signature: tuple[str, str, str] | None = None
    actual_series = pd.DataFrame()
    for run_root in run_roots:
        loaded = runtime._load_summary_loaded_config(run_root)
        candidate_signature = (
            str(Path(loaded.config.dataset.path).resolve()),
            loaded.config.dataset.dt_col,
            loaded.config.dataset.target_col,
        )
        if dataset_signature is None:
            dataset_signature = candidate_signature
            dataset_path, dt_col, target_col = candidate_signature
            source_df = pd.read_csv(dataset_path)
            if source_df.empty:
                raise ValueError(f"Shared dataset is empty: {dataset_path}")
            actual_series = (
                source_df[[dt_col, target_col]]
                .rename(columns={dt_col: "ds", target_col: "y"})
                .copy()
            )
            actual_series["ds"] = pd.to_datetime(actual_series["ds"], errors="coerce")
            actual_series["y"] = pd.to_numeric(actual_series["y"], errors="coerce")
            actual_series = (
                actual_series.dropna(subset=["ds", "y"])
                .drop_duplicates(subset=["ds"])
                .sort_values("ds", kind="stable")
                .reset_index(drop=True)
            )
            continue
        if candidate_signature != dataset_signature:
            raise ValueError(
                "Continuous overlay requires all runs to share one dataset path, dt_col, and target_col"
            )
    if dataset_signature is None or actual_series.empty:
        raise ValueError("Continuous overlay could not resolve a shared actual series")
    return actual_series


def _collect_fold_boundaries(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_idx in sorted(combined["fold_idx"].dropna().unique()):
        fold_frame = combined[combined["fold_idx"] == fold_idx].copy()
        _validate_fold_alignment(fold_frame, int(fold_idx))
        boundary_values = (
            pd.Series(fold_frame["train_end_ds"])
            .dropna()
            .pipe(_normalize_timestamp_series)
            .dropna()
            .drop_duplicates()
        )
        if len(boundary_values) != 1:
            raise ValueError(
                f"Continuous overlay requires a single train_end_ds per fold; fold={int(fold_idx):03d}"
            )
        rows.append(
            {
                "fold_idx": int(fold_idx),
                "train_end_ds": pd.Timestamp(boundary_values.iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_idx", kind="stable").reset_index(drop=True)


def _plot_single_fold_overlay(
    fold_frame: pd.DataFrame,
    *,
    input_actual_frame: pd.DataFrame,
    output_actual_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    show_mean_band: bool,
    alpha_per_run: float,
    color_by_col: str | None = None,
    color_map: Mapping[str, str] | None = None,
    show_series_legend: bool = True,
    group_legend_entries: Sequence[tuple[str, str]] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 6))
    actual_anchor_frame = (
        input_actual_frame.tail(1)[["ds", "y"]].copy()
        if not input_actual_frame.empty
        else pd.DataFrame(columns=["ds", "y"])
    )

    if not input_actual_frame.empty:
        ax.plot(
            input_actual_frame["ds"],
            input_actual_frame["y"],
            label="actual (input)",
            linewidth=2.0,
            color="black",
        )
    if not output_actual_frame.empty and output_actual_frame["y"].notna().any():
        observed_output = _connected_plot_frame(
            actual_anchor_frame,
            output_actual_frame,
            value_col="y",
        )
        ax.plot(
            observed_output["ds"],
            observed_output["y"],
            label="actual (output)",
            linewidth=1.8,
            linestyle="--",
            color="dimgray",
        )

    plot_frame = fold_frame.copy()
    plot_frame["ds"] = pd.to_datetime(plot_frame["ds"], errors="coerce")
    if "series_id" not in plot_frame.columns:
        plot_frame["series_id"] = plot_frame["run_id"]
    if "display_label" not in plot_frame.columns:
        plot_frame["display_label"] = plot_frame["run_id"]
    series_ids = plot_frame["series_id"].drop_duplicates().tolist()
    for series_id in series_ids:
        part = (
            plot_frame[plot_frame["series_id"] == series_id]
            .sort_values(["ds", "horizon_step"], kind="stable")
            .reset_index(drop=True)
        )
        if "y" in part.columns:
            part = part[part["y"].notna()].reset_index(drop=True)
        if part.empty:
            continue
        connected_model_frame = _connected_plot_frame(
            actual_anchor_frame,
            part,
            value_col="y_hat",
        )
        if connected_model_frame.empty:
            continue
        prediction_point_indices = list(range(1, len(connected_model_frame)))
        line_color = None
        if color_by_col is not None and color_map is not None and color_by_col in part.columns:
            color_value = str(part[color_by_col].iloc[0])
            line_color = color_map.get(color_value)
        ax.plot(
            connected_model_frame["ds"],
            connected_model_frame["y_hat"],
            label=str(part["display_label"].iloc[0]) if show_series_legend else "_nolegend_",
            linewidth=1.6,
            alpha=alpha_per_run,
            marker="o",
            markersize=4,
            markevery=prediction_point_indices if prediction_point_indices else None,
            color=line_color,
        )

    if show_mean_band and len(series_ids) > 1:
        group_keys = ["ds"]
        if "horizon_step" in plot_frame.columns:
            group_keys.append("horizon_step")
        stats = (
            plot_frame.groupby(group_keys, sort=False)["y_hat"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(group_keys, kind="stable")
        )
        stats["ds"] = pd.to_datetime(stats["ds"], errors="coerce")
        mean_hat = pd.to_numeric(stats["mean"], errors="coerce")
        std_hat = pd.to_numeric(stats["std"], errors="coerce").fillna(0.0)
        ax.plot(
            stats["ds"],
            mean_hat,
            color="darkred",
            linewidth=2.0,
            label="mean y_hat",
            zorder=4,
        )
        ax.fill_between(
            stats["ds"],
            mean_hat - std_hat,
            mean_hat + std_hat,
            color="darkred",
            alpha=0.12,
            label="±1 std (y_hat)",
        )

    ax.set_title(title)
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    if group_legend_entries:
        handles = [
            Line2D([0], [0], color="black", linewidth=2.0, label="actual (input)"),
            Line2D(
                [0],
                [0],
                color="dimgray",
                linewidth=1.8,
                linestyle="--",
                label="actual (output)",
            ),
        ]
        if show_mean_band and len(series_ids) > 1:
            handles.append(
                Line2D([0], [0], color="darkred", linewidth=2.0, label="mean y_hat")
            )
        for label, color in group_legend_entries:
            handles.append(Line2D([0], [0], color=color, linewidth=1.6, label=label))
        ax.legend(handles=handles, loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_continuous_overlay(
    combined: pd.DataFrame,
    *,
    actual_series: pd.DataFrame,
    fold_boundaries: pd.DataFrame,
    output_path: Path,
    title: str,
    show_mean_band: bool,
    alpha_per_run: float,
    x_start: str | None = None,
    x_end: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        actual_series["ds"],
        actual_series["y"],
        label="actual",
        linewidth=2.0,
        color="black",
        zorder=1,
    )

    plot_frame = combined.copy()
    plot_frame["ds"] = pd.to_datetime(plot_frame["ds"], errors="coerce")
    plot_frame["train_end_ds"] = _normalize_timestamp_series(plot_frame["train_end_ds"])
    plot_frame = plot_frame.dropna(subset=["ds", "y_hat", "train_end_ds"]).copy()
    if "series_id" not in plot_frame.columns:
        plot_frame["series_id"] = plot_frame["run_id"]
    if "display_label" not in plot_frame.columns:
        plot_frame["display_label"] = plot_frame["run_id"]

    for series_id in plot_frame["series_id"].drop_duplicates().tolist():
        run_frame = plot_frame[plot_frame["series_id"] == series_id].copy()
        label = str(run_frame["display_label"].iloc[0])
        first_segment = True
        for fold_idx in sorted(run_frame["fold_idx"].dropna().unique()):
            segment = (
                run_frame[run_frame["fold_idx"] == fold_idx]
                .sort_values(["ds", "horizon_step"], kind="stable")
                .reset_index(drop=True)
            )
            if segment.empty:
                continue
            ax.plot(
                segment["ds"],
                pd.to_numeric(segment["y_hat"], errors="coerce"),
                label=label if first_segment else "_nolegend_",
                linewidth=1.8,
                alpha=alpha_per_run,
                marker="o",
                markersize=4,
                zorder=3,
            )
            first_segment = False

    if show_mean_band and plot_frame["series_id"].nunique() > 1:
        group_keys = ["ds"]
        if "horizon_step" in plot_frame.columns:
            group_keys.append("horizon_step")
        stats = (
            plot_frame.groupby(group_keys, sort=False)["y_hat"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(group_keys, kind="stable")
        )
        stats["ds"] = pd.to_datetime(stats["ds"], errors="coerce")
        mean_hat = pd.to_numeric(stats["mean"], errors="coerce")
        std_hat = pd.to_numeric(stats["std"], errors="coerce").fillna(0.0)
        ax.plot(
            stats["ds"],
            mean_hat,
            color="darkred",
            linewidth=2.0,
            label="mean y_hat",
            zorder=4,
        )
        ax.fill_between(
            stats["ds"],
            mean_hat - std_hat,
            mean_hat + std_hat,
            color="darkred",
            alpha=0.12,
            label="±1 std (y_hat)",
        )

    for boundary in fold_boundaries["train_end_ds"].tolist():
        ax.axvline(
            pd.Timestamp(boundary),
            color="grey",
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )

    xlim_kwargs: dict[str, pd.Timestamp] = {}
    if x_start is not None:
        requested_start = pd.Timestamp(x_start)
        candidate_dates = list(
            actual_series.loc[actual_series["ds"] >= requested_start, "ds"].tolist()
        )
        candidate_dates.extend(
            plot_frame.loc[plot_frame["ds"] >= requested_start, "ds"].tolist()
        )
        effective_start = (
            min(pd.Timestamp(value) for value in candidate_dates)
            if candidate_dates
            else requested_start
        )
        xlim_kwargs["left"] = effective_start
    if x_end is not None:
        xlim_kwargs["right"] = pd.Timestamp(x_end)
    if xlim_kwargs:
        ax.set_xlim(**xlim_kwargs)

    ax.set_title(title)
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_folds(
    combined: pd.DataFrame,
    output_dir: Path,
    *,
    show_mean_band: bool,
    alpha_per_run: float,
    window_history_steps: int | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fold_idx in sorted(combined["fold_idx"].unique()):
        sub = combined[combined["fold_idx"] == fold_idx].copy()
        if sub.empty:
            continue
        sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
        if "horizon_step" not in sub.columns:
            sub["horizon_step"] = 0
        sub = sub.dropna(subset=["ds", "y_hat"])
        if sub.empty:
            continue
        run_ids = sorted(sub["run_id"].unique())
        n_runs = len(run_ids)
        input_actual_frame, output_actual_frame = _resolve_actual_frames_for_fold(
            sub,
            fold_idx=int(fold_idx),
            history_steps_override=None,
        )
        out_path = _plot_single_fold_overlay(
            sub,
            input_actual_frame=input_actual_frame,
            output_actual_frame=output_actual_frame,
            output_path=output_dir / f"fold_{int(fold_idx):03d}_predictions_overlay.png",
            title=f"Fold {int(fold_idx):03d} — {n_runs} runs",
            show_mean_band=show_mean_band,
            alpha_per_run=alpha_per_run,
        )
        written.append(out_path)

        if window_history_steps is not None and int(window_history_steps) > 0:
            input_window_frame, output_window_frame = _resolve_actual_frames_for_fold(
                sub,
                fold_idx=int(fold_idx),
                history_steps_override=int(window_history_steps),
            )
            window_path = _plot_single_fold_overlay(
                sub,
                input_actual_frame=input_window_frame,
                output_actual_frame=output_window_frame,
                output_path=output_dir
                / f"fold_{int(fold_idx):03d}_predictions_overlay_window_{int(window_history_steps)}.png",
                title=f"Fold {int(fold_idx):03d} — {n_runs} runs (input last {int(window_history_steps)})",
                show_mean_band=show_mean_band,
                alpha_per_run=alpha_per_run,
            )
            written.append(window_path)
    return written


def plot_hpo_trial_folds(
    hpo_run_root: Path,
    output_dir: Path,
    *,
    model_name: str,
    show_mean_band: bool,
    alpha_per_run: float,
) -> tuple[list[Path], Path, Path]:
    combined, coverage_frame, summary_payload = _collect_hpo_trial_forecasts(
        hpo_run_root,
        model_name=model_name,
    )
    if combined.empty:
        raise ValueError(
            f"No fold prediction artifacts found under {Path(hpo_run_root).resolve()}"
        )

    output_dir = output_dir.resolve()
    coverage_path, summary_path = _write_hpo_coverage_artifacts(
        coverage_frame=coverage_frame,
        summary_payload=summary_payload,
        output_dir=output_dir,
    )
    written: list[Path] = []
    legend_entries = [
        (study_label, _HPO_STUDY_COLORS[study_label])
        for study_label in sorted(combined["study_label"].dropna().astype(str).unique())
        if study_label in _HPO_STUDY_COLORS
    ]
    trial_count = int(summary_payload["trial_count"])
    study_count = int(summary_payload["study_count"])
    for fold_idx in sorted(combined["fold_idx"].dropna().astype(int).unique()):
        fold_key = f"fold_{int(fold_idx):03d}"
        sub = combined[combined["fold_idx"] == fold_idx].copy()
        if sub.empty:
            continue
        input_actual_frame, output_actual_frame = _resolve_actual_frames_for_fold(
            sub,
            fold_idx=int(fold_idx),
            history_steps_override=None,
        )
        plotted_count = int(summary_payload["folds"][fold_key]["plotted_trial_count"])
        written.append(
            _plot_single_fold_overlay(
                sub,
                input_actual_frame=input_actual_frame,
                output_actual_frame=output_actual_frame,
                output_path=output_dir / f"{fold_key}_all_trials_overlay.png",
                title=(
                    f"Fold {int(fold_idx):03d} — plotted {plotted_count}/{trial_count} trials "
                    f"across {study_count} studies"
                ),
                show_mean_band=show_mean_band,
                alpha_per_run=alpha_per_run,
                color_by_col="study_label",
                color_map=_HPO_STUDY_COLORS,
                show_series_legend=False,
                group_legend_entries=legend_entries,
            )
        )
    return written, coverage_path, summary_path


def plot_continuous_series(
    combined: pd.DataFrame,
    *,
    run_roots: Iterable[Path],
    output_path: Path,
    show_mean_band: bool,
    alpha_per_run: float,
    x_start: str | None = None,
    x_end: str | None = None,
) -> Path:
    actual_series = _load_shared_actual_series(run_roots)
    fold_boundaries = _collect_fold_boundaries(combined)
    return _plot_continuous_overlay(
        combined,
        actual_series=actual_series,
        fold_boundaries=fold_boundaries,
        output_path=output_path,
        title=f"Continuous forecast overlay across {combined['run_id'].nunique()} runs",
        show_mean_band=show_mean_band,
        alpha_per_run=alpha_per_run,
        x_start=x_start,
        x_end=x_end,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all runs' y_hat overlaid per CV fold (similarity check)."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        metavar="PATH",
        help="Run root directory (repeatable). Relative paths are under repo root.",
    )
    parser.add_argument(
        "--runs-glob",
        default=None,
        help="Glob under repo root, e.g. runs/sweep_retrieval_sweep_ret_*",
    )
    parser.add_argument(
        "--runs-list",
        type=Path,
        default=None,
        help="Text file: one run root path per line (relative to repo root ok).",
    )
    parser.add_argument(
        "--hpo-run-root",
        type=Path,
        default=None,
        help="HPO run root whose study-*/trial-* fold predictions should be overlaid.",
    )
    parser.add_argument(
        "--model",
        default="AAForecast",
        help="Filter to this model name when column exists (default: AAForecast).",
    )
    parser.add_argument(
        "--no-model-filter",
        action="store_true",
        help="Do not filter by model column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for fold_###_predictions_overlay.png files.",
    )
    parser.add_argument(
        "--mean-band",
        action="store_true",
        help="Draw mean and ±1 std of y_hat across runs at each ds.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Line alpha for each run's y_hat (default: 0.9).",
    )
    parser.add_argument(
        "--window-history-steps",
        type=int,
        default=16,
        help="Also render a second variant with this many input history steps (default: 16). Use 0 to disable.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Write one real-date continuous overlay across all folds instead of per-fold figures.",
    )
    parser.add_argument(
        "--x-start",
        default=None,
        help="Optional left x-axis bound for continuous plots, e.g. 2025-08-15.",
    )
    parser.add_argument(
        "--x-end",
        default=None,
        help="Optional right x-axis bound for continuous plots, e.g. 2026-03-09.",
    )
    args = parser.parse_args()

    if args.hpo_run_root is not None:
        if args.continuous:
            raise SystemExit("--continuous is not supported together with --hpo-run-root")
        hpo_run_root = (
            (REPO_ROOT / args.hpo_run_root).resolve()
            if not args.hpo_run_root.is_absolute()
            else args.hpo_run_root.resolve()
        )
        default_hpo_dir = (
            hpo_run_root / "models" / args.model / "visualizations" / "trial_fold_overlays"
        )
        out_dir = (
            default_hpo_dir
            if args.output_dir == _DEFAULT_OUTPUT_DIR
            else (
                args.output_dir
                if args.output_dir.is_absolute()
                else (REPO_ROOT / args.output_dir).resolve()
            )
        )
        paths, coverage_path, summary_path = plot_hpo_trial_folds(
            hpo_run_root,
            out_dir,
            model_name=args.model,
            show_mean_band=args.mean_band,
            alpha_per_run=args.alpha,
        )
        print(f"Wrote {len(paths)} figure(s) under {out_dir}")
        for path in paths:
            print(f"  {path}")
        print(f"coverage_csv={coverage_path}")
        print(f"summary_json={summary_path}")
        return

    run_roots = collect_run_roots(
        REPO_ROOT,
        run_dirs=[Path(p) for p in args.run_dir],
        runs_glob=args.runs_glob,
        runs_list_file=args.runs_list,
    )
    if not run_roots:
        print("No run directories provided.", file=sys.stderr)
        sys.exit(2)

    model_filter = None if args.no_model_filter else args.model
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for run_root in run_roots:
        if not run_root.is_dir():
            skipped.append(f"missing: {run_root}")
            continue
        raw = load_forecasts_from_run(run_root)
        if raw.empty:
            skipped.append(
                f"no forecasts: {run_root}\n{_diagnose_empty_run(run_root)}"
            )
            continue
        part = _filter_model(raw, model_filter)
        if part.empty:
            skipped.append(f"no rows after model filter: {run_root}")
            continue
        part = part.copy()
        part["run_id"] = run_root.name
        part["run_root"] = str(run_root.resolve())
        part = _annotate_series_identity(part, run_id=run_root.name)
        frames.append(part)

    if not frames:
        print("No forecast data loaded.", file=sys.stderr)
        for line in skipped:
            print(line, file=sys.stderr)
        print(
            "Hints: run must have finished writing cv (see run_root/cv/ or "
            "run_root/scheduler/workers/*/cv/), or summary/folds/*/predictions.csv; "
            "use --run-dir for a single path; if rows exist but vanish, try --no-model-filter.",
            file=sys.stderr,
        )
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    required = {"fold_idx", "ds", "y_hat"}
    missing = required.difference(combined.columns)
    if missing:
        raise SystemExit(f"Combined forecasts missing columns: {sorted(missing)}")

    out_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (REPO_ROOT / args.output_dir).resolve()
    )
    if args.continuous:
        output_path = out_dir / "all_folds_continuous_overlay.png"
        path = plot_continuous_series(
            combined,
            run_roots=run_roots,
            output_path=output_path,
            show_mean_band=args.mean_band,
            alpha_per_run=args.alpha,
            x_start=args.x_start,
            x_end=args.x_end,
        )
        print(f"Wrote 1 figure under {out_dir}")
        print(f"  {path}")
    else:
        paths = plot_folds(
            combined,
            out_dir,
            show_mean_band=args.mean_band,
            alpha_per_run=args.alpha,
            window_history_steps=(
                None if args.window_history_steps <= 0 else args.window_history_steps
            ),
        )
        print(f"Wrote {len(paths)} figure(s) under {out_dir}")
        for p in paths[:12]:
            print(f"  {p}")
        if len(paths) > 12:
            print(f"  ... and {len(paths) - 12} more")
    if skipped:
        print("Skipped:", file=sys.stderr)
        for line in skipped:
            print(f"  {line}", file=sys.stderr)


if __name__ == "__main__":
    main()
