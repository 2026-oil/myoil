from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = REPO_ROOT / "yaml" / "experiment" / "sweep_retrieval"
PLUGIN_DIR = REPO_ROOT / "yaml" / "plugins" / "aa_forecast" / "sweep"
CONFIGS_TXT = EXPERIMENT_DIR / "configs.txt"
OUTPUT_DIR = REPO_ROOT / "runs" / "sweep_cv_results"
DEFAULT_SETTING_PATH = REPO_ROOT / "yaml" / "setting" / "setting.yaml"


def _safe_output_root_part(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value
    ).strip(".-")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def _resolve_run_root_for_config(config_path: Path) -> Path:
    """Mirror runtime_support.runner._default_output_root for sweep configs."""
    doc = _load_yaml(config_path)
    task = doc.get("task") or {}
    task_name = str(task.get("name") or "").strip()
    config_parent = config_path.parent
    parent_name = (
        REPO_ROOT.name
        if config_parent.resolve() == REPO_ROOT.resolve()
        else config_parent.name
    )
    safe_parts = [
        part
        for part in (
            _safe_output_root_part(parent_name),
            _safe_output_root_part(task_name),
        )
        if part
    ]
    safe_name = "_".join(safe_parts)
    return REPO_ROOT / "runs" / (safe_name or "validation")


def _build_setting_override(
    *,
    base_setting_path: Path,
    n_windows: int | None,
    max_steps: int | None,
) -> Path:
    if n_windows is None and max_steps is None:
        return base_setting_path
    doc = _load_yaml(base_setting_path)
    cv = doc.get("cv") or {}
    if not isinstance(cv, dict):
        raise ValueError(f"{base_setting_path}: cv must be a mapping")
    if n_windows is not None:
        cv["n_windows"] = int(n_windows)
    doc["cv"] = cv
    if max_steps is not None:
        training = doc.get("training") or {}
        if not isinstance(training, dict):
            raise ValueError(f"{base_setting_path}: training must be a mapping")
        training["max_steps"] = int(max_steps)
        doc["training"] = training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name_parts = []
    if n_windows is not None:
        name_parts.append(f"nwindows_{int(n_windows):03d}")
    if max_steps is not None:
        name_parts.append(f"maxsteps_{int(max_steps):04d}")
    suffix = "_".join(name_parts) or "override"
    out_path = OUTPUT_DIR / f"setting_{suffix}.yaml"
    out_path.write_text(
        yaml.dump(doc, default_flow_style=False, sort_keys=False, allow_unicode=True)
        + "\n",
        encoding="utf-8",
    )
    return out_path


def _read_mean_mape_from_summary(
    run_root: Path,
    *,
    last_folds: int | None,
) -> float | None:
    """Compute mean fold MAPE from summary/folds/*/metrics.csv.

    If last_folds is provided, only use the most recent N folds (largest fold_idx).
    """
    folds_root = run_root / "summary" / "folds"
    if not folds_root.exists():
        return None
    metrics_paths = sorted(folds_root.glob("fold_*/metrics.csv"))
    if not metrics_paths:
        return None
    rows: list[dict[str, Any]] = []
    for path in metrics_paths:
        try:
            import pandas as pd

            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        rows.append(row)
    if not rows:
        return None

    def _fold_key(item: dict[str, Any]) -> int:
        raw = item.get("fold_idx")
        try:
            return int(raw)
        except Exception:
            return -1

    rows.sort(key=_fold_key)
    if last_folds is not None and last_folds > 0:
        rows = rows[-int(last_folds) :]

    def _mape(item: dict[str, Any]) -> float | None:
        raw = item.get("MAPE")
        try:
            return float(raw)
        except Exception:
            return None

    mapes = [value for value in (_mape(row) for row in rows) if value is not None]
    if not mapes:
        return None
    return float(sum(mapes) / len(mapes))


def _extract_retrieval_params(config_path: Path) -> dict[str, Any]:
    doc = _load_yaml(config_path)
    aa = doc.get("aa_forecast") or {}
    cfg_path = aa.get("config_path")
    if not cfg_path:
        return {}
    plugin_path = (REPO_ROOT / str(cfg_path)).resolve()
    if not plugin_path.exists():
        return {"plugin_config_path": str(cfg_path), "plugin_missing": True}
    plugin_doc = _load_yaml(plugin_path)
    aaf = plugin_doc.get("aa_forecast") or {}
    retrieval = aaf.get("retrieval") or {}
    tails = (aaf.get("star_anomaly_tails") or {}).get("upward")
    out: dict[str, Any] = {"plugin_config_path": str(cfg_path)}
    if tails is not None:
        out["star_anomaly_tails_upward"] = tails
    if isinstance(retrieval, dict):
        for key in ("top_k", "trigger_quantile", "min_similarity"):
            if key in retrieval:
                out[key] = retrieval[key]
    return out


def load_config_paths() -> list[Path]:
    if not CONFIGS_TXT.exists():
        raise FileNotFoundError(f"configs.txt not found: {CONFIGS_TXT}")
    paths = CONFIGS_TXT.read_text().strip().split("\n")
    return [REPO_ROOT / p for p in paths if p.strip()]


def run_single_cv(
    config_path: Path,
    idx: int,
    *,
    setting_path: Path,
    last_folds: int | None,
    timeout_s: int,
) -> dict[str, Any]:
    cmd = [
        "uv",
        "run",
        "python",
        "main.py",
        "--config",
        str(config_path),
        "--setting",
        str(setting_path),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    run_root = _resolve_run_root_for_config(config_path)
    output = {
        "idx": idx,
        "config": str(config_path.name),
        "run_root": str(run_root.relative_to(REPO_ROOT)),
        "returncode": result.returncode,
        "stdout": result.stdout[-500:] if result.stdout else "",
        "stderr": result.stderr[-500:] if result.stderr else "",
        **_extract_retrieval_params(config_path),
    }
    if result.returncode == 0:
        output["mean_mape"] = _read_mean_mape_from_summary(
            run_root, last_folds=None
        )
        output["mean_mape_recent"] = _read_mean_mape_from_summary(
            run_root, last_folds=last_folds
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run sweep configs with full CV")
    parser.add_argument("--dry-run", action="store_true", help="Show config count")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit configs")
    parser.add_argument(
        "--setting",
        default=str(DEFAULT_SETTING_PATH.relative_to(REPO_ROOT)),
        help="Base shared setting YAML (relative to repo root).",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=None,
        help="Override cv.n_windows in the shared setting file (writes an override under runs/sweep_cv_results/).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps in the shared setting file (writes an override under runs/sweep_cv_results/).",
    )
    parser.add_argument(
        "--last-folds",
        type=int,
        default=None,
        help="Compute mean MAPE over the most recent N folds (by fold_idx).",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=1800,
        help="Per-config timeout in seconds (default: 1800).",
    )
    args = parser.parse_args()

    try:
        config_paths = load_config_paths()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.limit:
        config_paths = config_paths[: args.limit]

    total = len(config_paths)
    setting_path = (REPO_ROOT / args.setting).resolve()
    if not setting_path.exists():
        print(f"Setting file not found: {setting_path}", file=sys.stderr)
        return 1
    setting_override = _build_setting_override(
        base_setting_path=setting_path,
        n_windows=args.n_windows,
        max_steps=args.max_steps,
    )

    msg = f"Found {total} configs to run"
    if args.n_windows is not None:
        msg += f" (cv.n_windows={args.n_windows})"
    if args.max_steps is not None:
        msg += f" (training.max_steps={args.max_steps})"
    if args.last_folds is not None:
        msg += f" | scoring recent_folds={args.last_folds}"
    print(msg)

    if args.dry_run:
        for i, p in enumerate(config_paths[:5]):
            print(f"  {i + 1}: {p.name}")
        if total > 5:
            print(f"  ... and {total - 5} more")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_output = OUTPUT_DIR / "results.json"
    results_csv_output = OUTPUT_DIR / "results.csv"

    results: list[dict[str, Any]] = []
    success = 0
    failed = 0

    def run_at_index(idx_and_cfg):
        idx, cfg = idx_and_cfg
        return run_single_cv(
            cfg,
            idx,
            setting_path=setting_override,
            last_folds=args.last_folds,
            timeout_s=int(args.timeout_s),
        )

    if args.parallel == 1:
        for i, cfg in enumerate(config_paths, 1):
            print(f"[{i}/{total}] Running {cfg.name}...", flush=True)
            res = run_at_index((i, cfg))
            score = res.get("mean_mape_recent") or res.get("mean_mape")
            if score is not None:
                print(f"  -> MAPE: {float(score):.6f}")
                success += 1
            else:
                print(f"  -> FAILED")
                failed += 1
            results.append(res)
    else:
        print(f"Running {total} configs with {args.parallel} parallel workers...")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_at_index, (i, cfg)): (i, cfg)
                for i, cfg in enumerate(config_paths, 1)
            }
            for future in as_completed(futures):
                i, cfg = futures[future]
                try:
                    res = future.result()
                    score = res.get("mean_mape_recent") or res.get("mean_mape")
                    if score is not None:
                        print(f"[{i}/{total}] {cfg.name}: MAPE={float(score):.6f}")
                        success += 1
                    else:
                        print(f"[{i}/{total}] {cfg.name}: FAILED")
                        failed += 1
                    results.append(res)
                except Exception as e:
                    print(f"[{i}/{total}] {cfg.name}: ERROR ({e})")
                    failed += 1

    def _mape_key(x: dict[str, Any]) -> float:
        value = x.get("mean_mape_recent")
        if value is None:
            value = x.get("mean_mape")
        try:
            return float(value)
        except Exception:
            return 999.0

    results.sort(key=_mape_key)
    results_output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {results_output}")
    try:
        import pandas as pd

        pd.DataFrame(results).to_csv(results_csv_output, index=False)
        print(f"Saved results CSV to {results_csv_output}")
    except Exception:
        pass

    best = results[0] if results else {}
    best_score = best.get("mean_mape_recent") or best.get("mean_mape")
    if best_score is not None:
        print(f"Best MAPE: {float(best_score):.6f} with {best.get('config')}")
    else:
        print(f"Best MAPE: N/A (no successful runs)")
    print(f"Total: {success} OK, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
