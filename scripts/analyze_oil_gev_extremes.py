from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.oil_gev_extremes_lib import (
    TARGET_COLUMNS,
    analyze_target,
    build_audit,
    build_manifest,
    build_report,
    build_summary,
    default_output_dir,
    load_frame,
    make_context,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze WTI/Brent oil extremes in data/df.csv with a GEV framing."
    )
    parser.add_argument(
        "--input",
        default="data/df.csv",
        help="Input CSV path (default: data/df.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to runs/dfcsv-oil-gev-<timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    context = make_context(input_path, output_dir)

    frame = load_frame(context.input_path)
    audit = build_audit(frame)
    write_json(context.contract_dir / "dataset_audit.json", audit)

    analyses = []
    for target in TARGET_COLUMNS:
        analysis, tail_tables = analyze_target(frame, target)
        analyses.append(analysis)
        for direction, table in tail_tables.items():
            csv_name = analysis.upper_tail.csv_name if direction == "upper" else analysis.lower_tail.csv_name
            table.to_csv(context.tables_dir / csv_name, index=False)

    manifest = build_manifest(context.input_path, analyses)
    write_json(context.contract_dir / "analysis_manifest.json", manifest)

    summary = build_summary(context, analyses)
    write_json(context.output_dir / "summary.json", summary)

    report = build_report(context, analyses, audit)
    (context.output_dir / "report.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(context.output_dir),
                "contract_dir": str(context.contract_dir),
                "tables_dir": str(context.tables_dir),
                "summary_path": str(context.output_dir / "summary.json"),
                "report_path": str(context.output_dir / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
