from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from phase1.config import DEFAULT_CONFIG_PATH, Phase1Config, load_runtime_toml
from phase1.pipeline import ALL_MODEL_NAMES, TARGET_COLUMNS, run_phase1

DEFAULT_DATA_PATH = WORKSPACE_ROOT / "df.csv"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "phase1" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the crude-oil/brent phase1 workflow.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--freq", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--final-holdout", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--season-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--val-check-steps", type=int, default=None)
    parser.add_argument("--early-stop-patience-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--valid-batch-size", type=int, default=None)
    parser.add_argument("--windows-batch-size", type=int, default=None)
    parser.add_argument("--inference-windows-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--require-gpu-count", type=int, default=None)
    parser.add_argument("--allow-fewer-gpus", action="store_true", default=None)
    return parser.parse_args()


def _pick(args: argparse.Namespace, shared: dict[str, Any], field: str, fallback: Any) -> Any:
    value = getattr(args, field)
    if value is not None:
        return value
    return shared.get(field, fallback)


def build_config(args: argparse.Namespace) -> Phase1Config:
    runtime = load_runtime_toml(args.config_path)
    shared = runtime["shared"]
    run_id = os.environ.get("PHASE1_RUN_ID")
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.environ["PHASE1_RUN_ID"] = run_id

    output_root = _pick(args, shared, "output_root", DEFAULT_OUTPUT_ROOT)
    data_path = _pick(args, shared, "data_path", DEFAULT_DATA_PATH)
    selected_models = list(_pick(args, shared, "models", list(ALL_MODEL_NAMES)))
    selected_targets = list(_pick(args, shared, "targets", list(TARGET_COLUMNS)))

    return Phase1Config(
        data_path=data_path,
        output_root=output_root / run_id,
        targets=selected_targets,
        model_names=selected_models,
        freq=_pick(args, shared, "freq", None),
        horizon=_pick(args, shared, "horizon", 12),
        step_size=_pick(args, shared, "step_size", 4),
        n_windows=_pick(args, shared, "n_windows", 24),
        final_holdout=_pick(args, shared, "final_holdout", 12),
        input_size=_pick(args, shared, "input_size", 48),
        season_length=_pick(args, shared, "season_length", 52),
        max_steps=_pick(args, shared, "max_steps", 200),
        val_size=_pick(args, shared, "val_size", 12),
        val_check_steps=_pick(args, shared, "val_check_steps", 20),
        early_stop_patience_steps=_pick(args, shared, "early_stop_patience_steps", 5),
        batch_size=_pick(args, shared, "batch_size", 32),
        valid_batch_size=_pick(args, shared, "valid_batch_size", 32),
        windows_batch_size=_pick(args, shared, "windows_batch_size", 128),
        inference_windows_batch_size=_pick(args, shared, "inference_windows_batch_size", 128),
        learning_rate=_pick(args, shared, "learning_rate", 1e-3),
        random_seed=_pick(args, shared, "random_seed", 1),
        require_gpu_count=_pick(args, shared, "require_gpu_count", 1),
        allow_fewer_gpus=_pick(args, shared, "allow_fewer_gpus", False),
        model_overrides={name: dict(runtime["models"].get(name, {})) for name in selected_models if name in runtime["models"]},
        config_path=str(runtime["path"]),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    summary = run_phase1(config)
    if os.environ.get("LOCAL_RANK") in (None, "0"):
        print(f"Phase1 complete. Artifacts: {summary['artifacts']['run_dir']}")


if __name__ == "__main__":
    main()
