from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_TARGETS = ["Com_CrudeOil", "Com_BrentCrudeOil"]
DISPLAY_ROUND = 3
BAR_PLOT_LIMIT = 30


@dataclass
class CorrelationResult:
    target: str
    level: str
    display_target: str
    n_obs_min: int
    n_obs_max: int
    table: pd.DataFrame
    display_table: pd.DataFrame
    raw_csv: str
    display_csv: str
    markdown_table: str
    bar_plot: str
    table_image: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze raw/differenced WTI-Brent correlations for data/df.csv."
    )
    parser.add_argument(
        "--input",
        default="data/df.csv",
        help="Input CSV path (default: data/df.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to runs/df-csv-corr-analysis-<timestamp>.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"df-csv-corr-analysis-{timestamp}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        df = df.sort_values("dt").reset_index(drop=True)
    return df


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric columns found in input data.")
    return numeric


def diff_frame(numeric: pd.DataFrame) -> pd.DataFrame:
    return numeric.diff().iloc[1:].reset_index(drop=True)


def build_level_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    numeric = numeric_frame(df)
    return {
        "raw": numeric,
        "diff": diff_frame(numeric),
    }


def level_display_target(level: str, target: str) -> str:
    if level == "raw":
        return target
    return f"diff({target})"


def compute_correlations(frame: pd.DataFrame, target: str, level: str) -> CorrelationResult:
    if target not in frame.columns:
        raise KeyError(f"Target {target} is not present in frame for level {level}.")

    target_series = frame[target]
    records: list[dict[str, Any]] = []
    for variable in frame.columns:
        if variable == target:
            continue
        pair = pd.concat(
            [target_series.rename(target), frame[variable].rename(variable)], axis=1
        ).dropna()
        corr = float(pair[target].corr(pair[variable]))
        records.append(
            {
                "target": target,
                "display_target": level_display_target(level, target),
                "level": level,
                "variable": variable,
                "pearson_corr": corr,
                "abs_corr": float(abs(corr)),
                "sign": "positive" if corr >= 0 else "negative",
                "n_obs": int(len(pair)),
            }
        )

    table = (
        pd.DataFrame(records)
        .sort_values(["abs_corr", "variable"], ascending=[False, True])
        .reset_index(drop=True)
    )
    display_table = table.copy()
    display_table["pearson_corr"] = display_table["pearson_corr"].round(DISPLAY_ROUND)
    display_table["abs_corr"] = display_table["abs_corr"].round(DISPLAY_ROUND)

    prefix = f"{level}_{target}"
    return CorrelationResult(
        target=target,
        level=level,
        display_target=level_display_target(level, target),
        n_obs_min=int(table["n_obs"].min()),
        n_obs_max=int(table["n_obs"].max()),
        table=table,
        display_table=display_table,
        raw_csv=f"{prefix}_correlations.csv",
        display_csv=f"{prefix}_correlations_display.csv",
        markdown_table=f"{prefix}_correlations.md",
        bar_plot=f"{prefix}_correlations_bar.png",
        table_image=f"{prefix}_correlations_table.png",
    )


def render_bar_plot(result: CorrelationResult, figure_path: Path) -> None:
    plot_frame = (
        result.display_table.head(BAR_PLOT_LIMIT)
        .sort_values("pearson_corr", ascending=True)
        .copy()
    )
    colors = np.where(plot_frame["pearson_corr"] >= 0, "#4c78a8", "#e45756")
    fig_height = max(6, 0.28 * len(plot_frame))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(plot_frame["variable"], plot_frame["pearson_corr"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"Top {BAR_PLOT_LIMIT} correlations | {result.display_target}")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def render_table_image(result: CorrelationResult, figure_path: Path) -> None:
    image_frame = result.display_table[
        ["variable", "pearson_corr", "abs_corr", "sign", "n_obs"]
    ].copy()
    fig_height = max(6, 0.28 * (len(image_frame) + 2))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=image_frame.values.tolist(),
        colLabels=list(image_frame.columns),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.05)
    ax.set_title(
        f"Full correlation table | {result.display_target}",
        fontsize=12,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def to_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_tables(result: CorrelationResult, tables_dir: Path, figures_dir: Path) -> None:
    result.table.to_csv(tables_dir / result.raw_csv, index=False)
    result.display_table.to_csv(tables_dir / result.display_csv, index=False)
    markdown = to_markdown_table(result.display_table[
        ["variable", "pearson_corr", "abs_corr", "sign", "n_obs"]
    ])
    (tables_dir / result.markdown_table).write_text(markdown + "\n", encoding="utf-8")
    render_bar_plot(result, figures_dir / result.bar_plot)
    render_table_image(result, figures_dir / result.table_image)


def top_lines(result: CorrelationResult, count: int = 5) -> tuple[str, str]:
    positives = result.display_table[result.display_table["pearson_corr"] >= 0].head(count)
    negatives = result.display_table[result.display_table["pearson_corr"] < 0].head(count)
    pos_line = ", ".join(
        f"`{row.variable}` ({row.pearson_corr:.3f})"
        for row in positives.itertuples(index=False)
    )
    neg_line = ", ".join(
        f"`{row.variable}` ({row.pearson_corr:.3f})"
        for row in negatives.itertuples(index=False)
    )
    return pos_line or "none", neg_line or "none"


def write_report(output_dir: Path, results: list[CorrelationResult]) -> None:
    lines: list[str] = []
    lines.append("# WTI / Brent Correlation Analysis for `data/df.csv`")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Raw targets: `Com_CrudeOil`, `Com_BrentCrudeOil`")
    lines.append("- Differenced targets: `diff(Com_CrudeOil)`, `diff(Com_BrentCrudeOil)`")
    lines.append("- Method: Pearson correlation over the numeric frame; differenced targets use first-differenced numeric values.")
    lines.append("- Presentation outputs round coefficients to 3 decimals.")
    lines.append("")
    lines.append("## Artifact Inventory")
    lines.append("- Tables: `tables/`")
    lines.append("- Figures: `figures/`")
    lines.append("- Summary: `summary.json`")
    lines.append("")
    for result in results:
        positives, negatives = top_lines(result)
        lines.append(f"## Target: `{result.display_target}`")
        lines.append(f"- rows: {len(result.table)}")
        lines.append(f"- n_obs range: {result.n_obs_min}..{result.n_obs_max}")
        lines.append(f"- raw CSV: `tables/{result.raw_csv}`")
        lines.append(f"- display CSV: `tables/{result.display_csv}`")
        lines.append(f"- markdown table: `tables/{result.markdown_table}`")
        lines.append(f"- bar image: `figures/{result.bar_plot}`")
        lines.append(f"- full table image: `figures/{result.table_image}`")
        lines.append(f"- strongest positives: {positives}")
        lines.append(f"- strongest negatives: {negatives}")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(output_dir: Path, results: list[CorrelationResult]) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    per_target: dict[str, Any] = {}
    for result in results:
        key = result.display_target
        per_target[key] = {
            "level": result.level,
            "base_target": result.target,
            "row_count": int(len(result.table)),
            "n_obs_min": result.n_obs_min,
            "n_obs_max": result.n_obs_max,
            "top_positive": result.display_table.iloc[0][
                ["variable", "pearson_corr"]
            ].to_dict(),
            "top_negative": (
                result.display_table[result.display_table["pearson_corr"] < 0]
                .head(1)[["variable", "pearson_corr"]]
                .to_dict(orient="records")
            ),
        }
        artifacts[f"{result.level}_{result.target}_raw_csv"] = str(
            output_dir / "tables" / result.raw_csv
        )
        artifacts[f"{result.level}_{result.target}_display_csv"] = str(
            output_dir / "tables" / result.display_csv
        )
        artifacts[f"{result.level}_{result.target}_markdown"] = str(
            output_dir / "tables" / result.markdown_table
        )
        artifacts[f"{result.level}_{result.target}_bar"] = str(
            output_dir / "figures" / result.bar_plot
        )
        artifacts[f"{result.level}_{result.target}_table_image"] = str(
            output_dir / "figures" / result.table_image
        )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "display_round": DISPLAY_ROUND,
        "targets": per_target,
        "artifacts": artifacts,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")

    frame = load_frame(input_path)
    level_frames = build_level_frames(frame)

    results: list[CorrelationResult] = []
    for level in ["raw", "diff"]:
        for target in RAW_TARGETS:
            result = compute_correlations(level_frames[level], target, level)
            write_tables(result, tables_dir, figures_dir)
            results.append(result)

    summary = build_summary(output_dir, results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir, results)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "tables_dir": str(tables_dir),
                "figures_dir": str(figures_dir),
                "summary_path": str(output_dir / "summary.json"),
                "report_path": str(output_dir / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
