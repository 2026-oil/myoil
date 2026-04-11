from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

OUTPUT_COLUMNS = [
    "dt",
    "Com_CrudeOil",
    "Com_BrentCrudeOil",
    "Com_Oil_Spread",
    "GPRD",
    "GPRD_ACT",
    "GPRD_THREAT",
]
WEEKLY_FREQ = "W-MON"
WEEKLY_LABEL = "left"
WEEKLY_CLOSED = "left"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/df_new.csv from oil.xls and data_gpr_daily_recent.xls."
    )
    parser.add_argument(
        "--oil-input",
        default="data/oil.xls",
        help="Path to the oil XLS file (default: data/oil.xls).",
    )
    parser.add_argument(
        "--gpr-input",
        default="data/data_gpr_daily_recent.xls",
        help="Path to the GPR XLS file (default: data/data_gpr_daily_recent.xls).",
    )
    parser.add_argument(
        "--output",
        default="data/df_new.csv",
        help="Output CSV path (default: data/df_new.csv).",
    )
    return parser.parse_args()


def _ensure_xls_support() -> None:
    try:
        importlib.import_module("xlrd")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading legacy .xls inputs requires xlrd. Run this script with "
            "`uv run --with xlrd python scripts/build_weekly_oil_gpr_df_new.py`."
        ) from exc


def load_oil_daily_frame(path: Path) -> pd.DataFrame:
    _ensure_xls_support()
    frame = pd.read_excel(path, sheet_name="Data 1").iloc[2:, :3].copy()
    frame.columns = ["dt", "Com_CrudeOil", "Com_BrentCrudeOil"]
    frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce")
    for column in ["Com_CrudeOil", "Com_BrentCrudeOil"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return frame


def load_gpr_daily_frame(path: Path) -> pd.DataFrame:
    _ensure_xls_support()
    frame = pd.read_excel(path, sheet_name="Sheet1")[
        ["date", "GPRD", "GPRD_ACT", "GPRD_THREAT"]
    ].copy()
    frame = frame.rename(columns={"date": "dt"})
    frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce")
    for column in ["GPRD", "GPRD_ACT", "GPRD_THREAT"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return frame


def weekly_average(frame: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        frame.set_index("dt")
        .sort_index()
        .resample(WEEKLY_FREQ, label=WEEKLY_LABEL, closed=WEEKLY_CLOSED)
        .mean(numeric_only=True)
        .reset_index()
    )
    weekly["dt"] = pd.to_datetime(weekly["dt"], errors="coerce")
    return weekly


def build_weekly_frame(oil_daily: pd.DataFrame, gpr_daily: pd.DataFrame) -> pd.DataFrame:
    oil_weekly = weekly_average(oil_daily)
    gpr_weekly = weekly_average(gpr_daily)
    merged = oil_weekly.merge(gpr_weekly, on="dt", how="inner")
    merged["Com_Oil_Spread"] = merged["Com_BrentCrudeOil"] - merged["Com_CrudeOil"]
    merged = merged.dropna(subset=OUTPUT_COLUMNS[1:])
    merged = merged.sort_values("dt").drop_duplicates(subset=["dt"]).reset_index(drop=True)
    return merged.loc[:, OUTPUT_COLUMNS]


def write_weekly_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["dt"] = pd.to_datetime(output["dt"], errors="coerce").dt.strftime("%Y-%m-%d")
    output.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_summary(frame: pd.DataFrame, oil_input: Path, gpr_input: Path, output: Path) -> dict[str, object]:
    return {
        "oil_input": str(oil_input),
        "gpr_input": str(gpr_input),
        "output": str(output),
        "rows": int(len(frame)),
        "columns": OUTPUT_COLUMNS,
        "dt_min": frame["dt"].min().strftime("%Y-%m-%d") if not frame.empty else None,
        "dt_max": frame["dt"].max().strftime("%Y-%m-%d") if not frame.empty else None,
        "weekly_freq": WEEKLY_FREQ,
        "weekly_label": WEEKLY_LABEL,
        "weekly_closed": WEEKLY_CLOSED,
    }


def main() -> None:
    args = parse_args()
    oil_input = Path(args.oil_input)
    gpr_input = Path(args.gpr_input)
    output = Path(args.output)

    oil_daily = load_oil_daily_frame(oil_input)
    gpr_daily = load_gpr_daily_frame(gpr_input)
    weekly = build_weekly_frame(oil_daily, gpr_daily)
    write_weekly_frame(weekly, output)

    print(json.dumps(build_summary(weekly, oil_input, gpr_input, output), indent=2))


if __name__ == "__main__":
    main()
