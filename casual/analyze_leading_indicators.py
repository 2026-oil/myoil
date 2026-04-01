from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casual.leading_indicators import PipelineConfig, parse_methods, parse_targets, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the casual df.csv leading-indicator method-comparison pipeline."
    )
    parser.add_argument("--csv", default="data/df.csv", help="Input CSV path.")
    parser.add_argument("--output-root", default=None, help="Optional output root.")
    parser.add_argument("--max-lag", type=int, default=8, help="Maximum lag to evaluate.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k threshold for synthesis support counts.")
    parser.add_argument(
        "--heavy-predictor-limit",
        type=int,
        default=20,
        help="Maximum predictor count passed to heavier optional methods.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method list. Default: all supported methods.",
    )
    parser.add_argument(
        "--targets",
        default="Com_CrudeOil,Com_BrentCrudeOil",
        help="Comma-separated target list.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = PipelineConfig(
        csv_path=Path(args.csv),
        output_root=Path(args.output_root) if args.output_root else None,
        max_lag=args.max_lag,
        top_k=args.top_k,
        methods=parse_methods(args.methods),
        targets=parse_targets(args.targets),
        heavy_predictor_limit=args.heavy_predictor_limit,
    )
    result = run_pipeline(config)
    print(f"PASS output_root={result['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
