"""CLI for retrieval raw/transformed similarity inspection PNGs.

Examples:
  uv run python scripts/plot_retrieval_similarity_windows.py \
    --json runs/some_task/aa_forecast/retrieval/fold_000/20250101T000000.json

  uv run python scripts/plot_retrieval_similarity_windows.py \
    --json path/to/summary.json \
    --windows-json path/to/summary_windows.json \
    --out-dir path/to/output_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.retrieval.similarity_window_plot import (
    load_payload,
    write_similarity_plot_set,
)


def _default_windows_json(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}_windows.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot retrieval query/neighbor similarity windows from JSON artifacts."
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to retrieval summary JSON (aa_forecast/retrieval/*.json).",
    )
    parser.add_argument(
        "--windows-json",
        type=Path,
        default=None,
        help="Path to retrieval windows JSON (default: <json_stem>_windows.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for generated PNG files (default: next to JSON).",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Filename stem for generated PNGs (default: summary JSON stem).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150).")
    args = parser.parse_args(argv)

    json_path = args.json.resolve()
    if not json_path.is_file():
        print(f"error: JSON file not found: {json_path}", file=sys.stderr)
        return 1

    windows_json = (
        args.windows_json.resolve()
        if args.windows_json is not None
        else _default_windows_json(json_path)
    )
    if not windows_json.is_file():
        print(f"error: windows JSON file not found: {windows_json}", file=sys.stderr)
        return 1

    out_dir = args.out_dir.resolve() if args.out_dir is not None else json_path.parent
    stem = args.stem or json_path.stem

    try:
        summary_payload = load_payload(json_path)
        windows_payload = load_payload(windows_json)
        outputs = write_similarity_plot_set(
            summary_payload,
            windows_payload,
            out_dir=out_dir,
            stem=stem,
            dpi=args.dpi,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for path in outputs.values():
        print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
