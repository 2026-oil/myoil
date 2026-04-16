from __future__ import annotations

import argparse
import json
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
        linked_path = raw_batch_root / "runs" / str(entry["run_name"])
        manifest_entries.append(
            {
                **entry,
                "canonical_run_root": str(canonical_root),
                "exists": canonical_root.exists(),
                "linked_run_path": str(linked_path),
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


def _unique_existing_run_roots(
    entries: list[dict[str, Any]], *, group: str
) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for entry in entries:
        if entry["group"] != group:
            continue
        candidate = Path(str(entry["canonical_run_root"])).resolve()
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def _write_group_plot(
    *,
    raw_batch_root: Path,
    entries: list[dict[str, Any]],
    group: str,
    x_start: str | None,
    x_end: str | None,
) -> tuple[Path | None, Path | None]:
    run_roots = _unique_existing_run_roots(entries, group=group)
    if not run_roots:
        return None, None
    combined = _build_combined_frame(run_roots)
    if combined.empty:
        return None, None

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
    manifest_path = _build_manifest(
        raw_batch_root=raw_batch_root,
        summary_json=summary_json,
        summary_payload=summary_payload,
        entries=entries,
    )
    ret_plot, ret_folds_dir = _write_group_plot(
        raw_batch_root=raw_batch_root,
        entries=entries,
        group="ret",
        x_start=args.x_start,
        x_end=args.x_end,
    )
    nonret_plot, nonret_folds_dir = _write_group_plot(
        raw_batch_root=raw_batch_root,
        entries=entries,
        group="nonret",
        x_start=args.x_start,
        x_end=args.x_end,
    )

    print(f"batch_manifest={manifest_path}")
    print(f"ret_continuous_plot={ret_plot}")
    print(f"ret_folds_dir={ret_folds_dir}")
    print(f"nonret_continuous_plot={nonret_plot}")
    print(f"nonret_folds_dir={nonret_folds_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
