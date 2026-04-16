"""CLI for memory-bank ``event_score`` distribution plots (see module docstring).

Examples:
  uv run python scripts/plot_retrieval_event_score_distribution.py \\
    --json runs/some_task/aa_forecast/retrieval/20250101T000000.json

  uv run python scripts/plot_retrieval_event_score_distribution.py \\
    --json path/to/summary.json --out path/to/plot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plugins.retrieval.event_score_distribution_plot import (
    load_payload,
    write_event_score_distribution_plot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot event_score histogram from retrieval summary JSON."
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to retrieval summary .json (aa_forecast/retrieval/*.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <json_stem>_event_score_dist.png next to JSON).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150).")
    args = parser.parse_args(argv)

    json_path = args.json.resolve()
    if not json_path.is_file():
        print(f"error: JSON file not found: {json_path}", file=sys.stderr)
        return 1

    out_path = args.out
    if out_path is None:
        out_path = json_path.with_name(f"{json_path.stem}_event_score_dist.png")
    else:
        out_path = out_path.resolve()

    try:
        payload = load_payload(json_path)
        write_event_score_distribution_plot(payload, out_path=out_path, dpi=args.dpi)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
