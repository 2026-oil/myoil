"""Batch runner for sweep_retrieval_grid configs with finding.py

Usage:
    uv run python scripts/run_sweep_finding.py --dry-run   # show what would run
    uv run python scripts/run_sweep_finding.py        # actually run all configs
    uv run python scripts/run_sweep_finding.py --parallel 8  # run 8 parallel jobs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice

REPO_ROOT = Path(__file__).resolve().parents[1]

# Same args as your original command, except --config and --output-csv vary per sweep
FINDING_BASE_ARGS = [
    "uv",
    "run",
    "python",
    "scripts/finding.py",
    "--input-sizes",
    "64",
    "--eval-slice",
    "last_cv_fold",
]

EXPERIMENT_DIR = REPO_ROOT / "yaml" / "experiment" / "sweep_retrieval"
CONFIGS_TXT = EXPERIMENT_DIR / "configs.txt"
OUTPUT_DIR = REPO_ROOT / "runs" / "sweep_results"


def batched(iterable, size):
    """Yield successive batches from iterable."""
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


def load_config_paths() -> list[Path]:
    """Load experiment config paths from configs.txt."""
    if not CONFIGS_TXT.exists():
        raise FileNotFoundError(
            f"configs.txt not found: {CONFIGS_TXT}\n"
            "Run: uv run python scripts/sweep_retrieval_grid.py"
        )
    paths = CONFIGS_TXT.read_text().strip().split("\n")
    return [REPO_ROOT / p for p in paths if p.strip()]


def run_single_finding(config_path: Path, output_csv: Path) -> tuple[int, str, str]:
    """Run finding.py for single config. Returns (returncode, stdout, stderr)."""
    cmd = [
        *FINDING_BASE_ARGS,
        "--config",
        str(config_path),
        "--output-csv",
        str(output_csv),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run finding.py for sweep configs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of configs to run",
    )
    args = parser.parse_args()

    # Load configs
    try:
        config_paths = load_config_paths()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.limit:
        config_paths = config_paths[: args.limit]

    total = len(config_paths)
    print(f"Found {total} sweep configs to run")

    # Prepare output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build commands
    commands: list[tuple[Path, Path, list[str]]] = []
    for config_path in config_paths:
        idx = config_path.stem.replace("sweep-", "")
        output_csv = OUTPUT_DIR / f"results_{idx}.csv"
        cmd = [
            *FINDING_BASE_ARGS,
            "--config",
            str(config_path),
            "--output-csv",
            str(output_csv),
        ]
        commands.append((config_path, output_csv, cmd))

    if args.dry_run:
        print(f"Would run {len(commands)} configs:")
        for config_path, output_csv, cmd in commands[:5]:
            print(f"  {config_path.name} -> {output_csv.name}")
        if len(commands) > 5:
            print(f"  ... and {len(commands) - 5} more")
        return 0

    # Run commands
    success = 0
    failed = 0

    if args.parallel == 1:
        # Sequential
        for i, (config_path, output_csv, cmd) in enumerate(commands, 1):
            print(f"[{i}/{total}] Running {config_path.name}...", end=" ", flush=True)
            rc, stdout, stderr = run_single_finding(config_path, output_csv)
            if rc == 0:
                print("OK")
                success += 1
            else:
                print(f"FAILED (rc={rc})")
                if stderr:
                    print(f"  stderr: {stderr[:200]}")
                failed += 1
    else:
        # Parallel
        print(f"Running {total} configs with {args.parallel} parallel workers...")

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_single_finding, cfg, out): (i, cfg, out)
                for i, (cfg, out, cmd) in enumerate(commands, 1)
            }

            for future in as_completed(futures):
                i, config_path, output_csv = futures[future]
                try:
                    rc, stdout, stderr = future.result()
                    if rc == 0:
                        print(f"[{i}/{total}] {config_path.name}: OK")
                        success += 1
                    else:
                        print(f"[{i}/{total}] {config_path.name}: FAILED (rc={rc})")
                        failed += 1
                except Exception as e:
                    print(f"[{i}/{total}] {config_path.name}: ERROR ({e})")
                    failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
