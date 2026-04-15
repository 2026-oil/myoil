"""Overlay y_hat from many completed runs, one PNG per CV fold.

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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def plot_folds(
    combined: pd.DataFrame,
    output_dir: Path,
    *,
    show_mean_band: bool,
    alpha_per_run: float,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fold_idx in sorted(combined["fold_idx"].unique()):
        sub = combined[combined["fold_idx"] == fold_idx].copy()
        if sub.empty:
            continue
        sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
        sub = sub.dropna(subset=["ds", "y_hat"])
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        run_ids = sorted(sub["run_id"].unique())
        n_runs = len(run_ids)

        # Actual y: should match across runs for same (ds, [horizon_step]); take first non-null
        y_keys = ["ds"]
        if "horizon_step" in sub.columns:
            y_keys.append("horizon_step")
        if "y" in sub.columns:
            y_ref = (
                sub.groupby(y_keys, sort=False)["y"]
                .first()
                .reset_index()
                .sort_values(y_keys, kind="stable")
            )
            y_ref["ds"] = pd.to_datetime(y_ref["ds"], errors="coerce")
        else:
            y_ref = pd.DataFrame(columns=y_keys + ["y"])
        if "y" in y_ref.columns and not y_ref["y"].isna().all():
            ax.plot(
                y_ref["ds"],
                pd.to_numeric(y_ref["y"], errors="coerce"),
                color="black",
                linewidth=2.2,
                label="y (actual)",
                zorder=5,
            )

        for run_id in run_ids:
            part = sub[sub["run_id"] == run_id].sort_values("ds", kind="stable")
            ax.plot(
                part["ds"],
                pd.to_numeric(part["y_hat"], errors="coerce"),
                color="C0",
                alpha=alpha_per_run,
                linewidth=0.9,
                zorder=1,
            )

        if show_mean_band and n_runs > 1:
            group_keys = ["ds"]
            if "horizon_step" in sub.columns:
                group_keys.append("horizon_step")
            g = sub.groupby(group_keys, sort=False)["y_hat"].agg(["mean", "std"]).reset_index()
            g["ds"] = pd.to_datetime(g["ds"])
            g = g.sort_values(group_keys, kind="stable")
            mean_hat = pd.to_numeric(g["mean"], errors="coerce")
            std_hat = pd.to_numeric(g["std"], errors="coerce").fillna(0.0)
            ax.plot(
                g["ds"],
                mean_hat,
                color="darkred",
                linewidth=2.0,
                label="mean y_hat",
                zorder=4,
            )
            ax.fill_between(
                g["ds"],
                mean_hat - std_hat,
                mean_hat + std_hat,
                color="darkred",
                alpha=0.12,
                label="±1 std (y_hat)",
            )

        ax.set_title(
            f"Fold {int(fold_idx):03d} — {n_runs} runs, y_hat overlay (alpha={alpha_per_run})"
        )
        ax.set_xlabel("ds")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        has_y_plot = "y" in y_ref.columns and not y_ref["y"].isna().all()
        if show_mean_band or has_y_plot:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out_path = output_dir / f"fold_{int(fold_idx):03d}_predictions_overlay.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written


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
        default=Path("runs/_fold_overlay_plots"),
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
        default=0.08,
        help="Line alpha for each run's y_hat (default: 0.08).",
    )
    args = parser.parse_args()

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
    paths = plot_folds(
        combined,
        out_dir,
        show_mean_band=args.mean_band,
        alpha_per_run=args.alpha,
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
