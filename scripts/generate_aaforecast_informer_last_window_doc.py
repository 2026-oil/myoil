from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_support.aaforecast_last_window_doc import generate_last_window_doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aa_inf.md for the final AAForecast Informer prediction window using archived artifacts plus current-code replay tensors."
    )
    parser.add_argument(
        "--config",
        default="yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml",
        help="Experiment config path.",
    )
    parser.add_argument(
        "--run-root",
        default="runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer",
        help="Completed run root to document.",
    )
    parser.add_argument(
        "--cutoff",
        default="2026-02-23",
        help="Final cutoff date to document.",
    )
    parser.add_argument(
        "--output-md",
        default="aa_inf.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--output-json",
        default="runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/aa_forecast/docs/20260223T000000.trace.json",
        help="Machine-readable trace output path.",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer/aa_forecast/docs/20260223T000000.current_replay.ckpt",
        help="Checkpoint path for the current replay fitted state used for tensor extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = ROOT
    payload = generate_last_window_doc(
        repo_root=repo_root,
        config_path=(repo_root / args.config).resolve(),
        run_root=(repo_root / args.run_root).resolve(),
        output_md=(repo_root / args.output_md).resolve(),
        output_json=(repo_root / args.output_json).resolve(),
        output_checkpoint=(repo_root / args.output_checkpoint).resolve(),
        cutoff=args.cutoff,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
