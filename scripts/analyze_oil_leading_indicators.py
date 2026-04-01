from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

from scripts.oil_leading_indicators_lib import (
    build_audit,
    build_manifest,
    build_report,
    build_summary,
    ccf_family_rows,
    cycle_representation,
    default_output_dir,
    frequency_family,
    load_frame,
    make_context,
    numeric_frame,
    oos_family,
    plot_family_strength,
    plot_synthesis_counts,
    predictive_family,
    primary_target,
    sensitivity_target,
    stationary_representation,
    synthesis_table,
    turning_point_summary,
    write_family_csv,
    write_json,
    write_synthesis_csv,
    dfm_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze df.csv for oil leading/coincident/lagging indicators."
    )
    parser.add_argument("--input", default="../df.csv", help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to runs/dfcsv-leading-indicators-oil-<timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    context = make_context(Path(args.input), output_dir)

    frame = load_frame(context.input_path)
    numeric = numeric_frame(frame)
    audit = build_audit(frame, numeric)
    write_json(context.contract_dir / "dataset_audit.json", audit)

    primary = primary_target(numeric)
    pca_appendix = sensitivity_target(numeric)
    manifest = build_manifest(context.input_path, audit)
    write_json(context.contract_dir / "analysis_manifest.json", manifest)
    pd.DataFrame(
        {
            "dt": frame["dt"],
            "oil_target_primary": primary,
            "oil_target_pca1": pca_appendix,
        }
    ).to_csv(context.contract_dir / "oil_targets.csv", index=False)

    variables = [column for column in numeric.columns if column not in {"Com_CrudeOil", "Com_BrentCrudeOil"}]
    candidate_cycles = {variable: cycle_representation(numeric[variable]) for variable in variables}
    target_cycle = cycle_representation(primary)
    candidate_stationary: dict[str, pd.Series] = {}
    for variable in variables:
        stationary, _ = stationary_representation(numeric[variable])
        candidate_stationary[variable] = stationary
    target_stationary, _ = stationary_representation(primary)

    turning_rows = [turning_point_summary(candidate_cycles[variable], target_cycle, variable) for variable in variables]
    family_turning = pd.DataFrame(turning_rows)
    family_ccf = ccf_family_rows(variables, candidate_cycles, candidate_stationary, target_cycle, target_stationary)
    family_predictive = predictive_family(variables, numeric, primary, target_stationary)
    family_dfm = dfm_family(variables, numeric, target_stationary)
    family_oos_dm = oos_family(variables, primary, candidate_stationary)
    family_frequency = frequency_family(variables, candidate_stationary, target_stationary)

    family_frames = {
        "turning_point": family_turning,
        "ccf": family_ccf,
        "predictive": family_predictive,
        "dfm": family_dfm,
        "oos_dm": family_oos_dm,
        "frequency": family_frequency,
    }

    for family, frame_out in family_frames.items():
        write_family_csv(frame_out, family, context.tables_dir)
        plot_family_strength(frame_out, family, context.figures_dir / f"{family}_top_support.png")

    synthesis = synthesis_table(family_frames, variables)
    write_synthesis_csv(synthesis, context.tables_dir)
    plot_synthesis_counts(synthesis, context.figures_dir / "synthesis_counts.png")
    summary = build_summary(context, synthesis, family_frames)
    write_json(context.output_dir / "summary.json", summary)
    build_report(context, audit, synthesis, family_frames)

    print(
        pd.Series(
            {
                "run_dir": str(context.output_dir),
                "contract_dir": str(context.contract_dir),
                "tables_dir": str(context.tables_dir),
                "figures_dir": str(context.figures_dir),
                "summary_path": str(context.output_dir / "summary.json"),
                "report_path": str(context.output_dir / "report.md"),
            }
        ).to_json(indent=2)
    )


if __name__ == "__main__":
    main()
