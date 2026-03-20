from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


TARGETS = ["Com_CrudeOil", "Com_BrentCrudeOil"]
VIF_HIGH = 5.0
VIF_SEVERE = 10.0
CORR_FLAG = 0.8
CONDITION_WARNING = 30.0
EIGENVALUE_RATIO_WARNING = 1e-6
TOP_VIF_PLOT_COUNT = 20


@dataclass
class LevelTargetResult:
    target: str
    level: str
    n_obs: int
    n_predictors: int
    dropped_zero_variance: list[str]
    vif_table: pd.DataFrame
    flagged_pairwise: pd.DataFrame
    group_summary: pd.DataFrame
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    eigenvalue_ratio: float
    near_singularity: bool
    top_vif_plot: str
    flagged_heatmap: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze target-specific multicollinearity for data/df.csv."
    )
    parser.add_argument(
        "--input",
        default="data/df.csv",
        help="Input CSV path (default: data/df.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to runs/df-csv-multicollinearity-<timestamp>.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"df-csv-multicollinearity-{timestamp}"


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
        "diff1": diff_frame(numeric),
    }


def classify_vif(vif: float) -> str:
    if np.isinf(vif) or vif > VIF_SEVERE:
        return "severe"
    if vif > VIF_HIGH:
        return "high"
    return "ok"


def sanitize_design_matrix(frame: pd.DataFrame, target: str) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    if target not in frame.columns:
        raise KeyError(f"Target {target} not present in level frame.")
    y = frame[target]
    X = frame.drop(columns=[target])
    combined = pd.concat([y.rename(target), X], axis=1).dropna().reset_index(drop=True)
    y = combined[target]
    X = combined.drop(columns=[target])
    variance = X.var(axis=0, ddof=0)
    dropped = variance[variance <= 0].index.tolist()
    if dropped:
        X = X.drop(columns=dropped)
    return y, X, dropped


def compute_vif_table(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    exog = sm.add_constant(X, has_constant="add")
    vif_rows: list[dict[str, Any]] = []
    target_corr = X.corrwith(y)
    for index, column in enumerate(X.columns, start=1):
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                vif = float(variance_inflation_factor(exog.values, index))
        except Exception:
            vif = float("inf")
        vif_rows.append(
            {
                "variable": column,
                "vif": vif,
                "severity": classify_vif(vif),
                "abs_target_corr": float(abs(target_corr[column])),
                "target_corr": float(target_corr[column]),
            }
        )
    vif_table = pd.DataFrame(vif_rows)
    return vif_table.sort_values(["vif", "abs_target_corr"], ascending=[False, False]).reset_index(drop=True)


def fit_condition_diagnostics(X: pd.DataFrame, y: pd.Series) -> dict[str, float | bool]:
    centered = X - X.mean(axis=0)
    scales = X.std(axis=0, ddof=0).replace(0, 1.0)
    zX = centered / scales
    exog = sm.add_constant(zX, has_constant="add")
    results = sm.OLS(y, exog).fit()
    eigenvals = np.asarray(results.eigenvals, dtype=float)
    positive = eigenvals[eigenvals > 0]
    min_eigenvalue = float(positive.min()) if positive.size else 0.0
    max_eigenvalue = float(positive.max()) if positive.size else 0.0
    ratio = min_eigenvalue / max_eigenvalue if max_eigenvalue > 0 else 0.0
    near_singularity = bool(min_eigenvalue <= 1e-8 or ratio <= EIGENVALUE_RATIO_WARNING)
    return {
        "condition_number": float(results.condition_number),
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": max_eigenvalue,
        "eigenvalue_ratio": float(ratio),
        "near_singularity": near_singularity,
    }


def find_flagged_pairwise(X: pd.DataFrame, vif_table: pd.DataFrame) -> pd.DataFrame:
    corr = X.corr()
    high_vif_vars = set(vif_table.loc[vif_table["vif"] > VIF_HIGH, "variable"])
    rows: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) < CORR_FLAG:
                continue
            if left not in high_vif_vars and right not in high_vif_vars:
                continue
            rows.append(
                {
                    "var_a": left,
                    "var_b": right,
                    "corr": value,
                    "abs_corr": abs(value),
                    "var_a_vif": float(vif_table.loc[vif_table["variable"] == left, "vif"].iloc[0]),
                    "var_b_vif": float(vif_table.loc[vif_table["variable"] == right, "vif"].iloc[0]),
                }
            )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise
    return pairwise.sort_values(["abs_corr", "var_a", "var_b"], ascending=[False, True, True]).reset_index(drop=True)


def connected_components(edges: pd.DataFrame) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        adjacency[row.var_a].add(row.var_b)
        adjacency[row.var_b].add(row.var_a)
    seen: set[str] = set()
    components: list[set[str]] = []
    for node in adjacency:
        if node in seen:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        if len(component) > 1:
            components.append(component)
    return components


def choose_representative(component: set[str], vif_table: pd.DataFrame) -> tuple[str, str]:
    subset = vif_table[vif_table["variable"].isin(component)].copy()
    subset["severity_rank"] = subset["severity"].map({"ok": 0, "high": 1, "severe": 2}).fillna(3)
    best = subset.sort_values(
        ["severity_rank", "vif", "abs_target_corr"],
        ascending=[True, True, False],
    ).iloc[0]
    rationale = (
        f"lowest-risk within flagged group (severity={best['severity']}, "
        f"vif={best['vif']:.2f}, abs_target_corr={best['abs_target_corr']:.3f})"
    )
    return str(best["variable"]), rationale


def summarize_groups(pairwise: pd.DataFrame, vif_table: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame(
            columns=[
                "group_id",
                "members",
                "representative",
                "representative_reason",
                "group_max_vif",
                "group_max_abs_corr",
                "severe_count",
                "high_or_severe_count",
                "proposal",
            ]
        )
    components = connected_components(pairwise)
    rows: list[dict[str, Any]] = []
    for idx, component in enumerate(sorted(components, key=lambda c: sorted(c)), start=1):
        subset = vif_table[vif_table["variable"].isin(component)]
        rep, rationale = choose_representative(component, vif_table)
        component_pairs = pairwise[
            pairwise["var_a"].isin(component) & pairwise["var_b"].isin(component)
        ]
        severe_count = int((subset["vif"] > VIF_SEVERE).sum())
        flagged_count = int((subset["vif"] > VIF_HIGH).sum())
        proposal = (
            f"Prefer {rep} as the first representative feature for this group; "
            "treat the remaining members as redundant candidates unless domain coverage requires grouped retention."
        )
        rows.append(
            {
                "group_id": f"group_{idx}",
                "members": ", ".join(sorted(component)),
                "representative": rep,
                "representative_reason": rationale,
                "group_max_vif": float(subset["vif"].max()),
                "group_max_abs_corr": float(component_pairs["abs_corr"].max()),
                "severe_count": severe_count,
                "high_or_severe_count": flagged_count,
                "proposal": proposal,
            }
        )
    return pd.DataFrame(rows).sort_values(["group_max_vif", "group_max_abs_corr"], ascending=[False, False]).reset_index(drop=True)


def render_vif_plot(vif_table: pd.DataFrame, target: str, level: str, figure_path: Path) -> None:
    plot_frame = vif_table.head(TOP_VIF_PLOT_COUNT).sort_values("vif", ascending=True).copy()
    finite_vif = plot_frame.loc[np.isfinite(plot_frame["vif"]), "vif"]
    cap = float(finite_vif.max() * 1.1) if not finite_vif.empty else VIF_SEVERE * 1.5
    plot_frame["vif_plot"] = plot_frame["vif"].replace([np.inf, -np.inf], cap).clip(upper=cap)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(plot_frame))))
    colors = plot_frame["severity"].map({"ok": "#4c78a8", "high": "#f58518", "severe": "#e45756"})
    ax.barh(plot_frame["variable"], plot_frame["vif_plot"], color=colors)
    ax.axvline(VIF_HIGH, color="#f58518", linestyle="--", linewidth=1, label="VIF > 5")
    ax.axvline(VIF_SEVERE, color="#e45756", linestyle=":", linewidth=1, label="VIF > 10")
    ax.set_title(f"Top VIF values | {target} | {level}")
    ax.set_xlabel("VIF")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def render_heatmap(X: pd.DataFrame, pairwise: pd.DataFrame, target: str, level: str, figure_path: Path) -> str | None:
    if pairwise.empty:
        return None
    variables = sorted(set(pairwise["var_a"]).union(pairwise["var_b"]))
    if len(variables) < 2:
        return None
    corr = X[variables].corr()
    fig, ax = plt.subplots(figsize=(0.45 * len(variables) + 4, 0.45 * len(variables) + 3))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(variables)))
    ax.set_yticks(np.arange(len(variables)))
    ax.set_xticklabels(variables, rotation=90)
    ax.set_yticklabels(variables)
    ax.set_title(f"Flagged-group pairwise correlations | {target} | {level}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path.name


def analyze_level_target(
    frame: pd.DataFrame,
    target: str,
    level: str,
    tables_dir: Path,
    figures_dir: Path,
) -> LevelTargetResult:
    y, X, dropped_zero_variance = sanitize_design_matrix(frame, target)
    vif_table = compute_vif_table(X, y)
    flagged_pairwise = find_flagged_pairwise(X, vif_table)
    group_summary = summarize_groups(flagged_pairwise, vif_table)
    condition = fit_condition_diagnostics(X, y)

    prefix = f"{target.lower()}_{level}".replace("-", "_")
    vif_table.to_csv(tables_dir / f"vif_{prefix}.csv", index=False)
    flagged_pairwise.to_csv(tables_dir / f"flagged_pairwise_{prefix}.csv", index=False)
    group_summary.to_csv(tables_dir / f"group_summary_{prefix}.csv", index=False)

    vif_plot_name = f"top_vif_{prefix}.png"
    render_vif_plot(vif_table, target, level, figures_dir / vif_plot_name)
    heatmap_name = render_heatmap(X, flagged_pairwise, target, level, figures_dir / f"flagged_corr_heatmap_{prefix}.png")

    return LevelTargetResult(
        target=target,
        level=level,
        n_obs=int(len(y)),
        n_predictors=int(X.shape[1]),
        dropped_zero_variance=dropped_zero_variance,
        vif_table=vif_table,
        flagged_pairwise=flagged_pairwise,
        group_summary=group_summary,
        condition_number=float(condition["condition_number"]),
        min_eigenvalue=float(condition["min_eigenvalue"]),
        max_eigenvalue=float(condition["max_eigenvalue"]),
        eigenvalue_ratio=float(condition["eigenvalue_ratio"]),
        near_singularity=bool(condition["near_singularity"]),
        top_vif_plot=vif_plot_name,
        flagged_heatmap=heatmap_name,
    )


def comparison_summary(raw: LevelTargetResult, diff1: LevelTargetResult) -> dict[str, Any]:
    raw_high = set(raw.vif_table.loc[raw.vif_table["vif"] > VIF_HIGH, "variable"])
    diff_high = set(diff1.vif_table.loc[diff1.vif_table["vif"] > VIF_HIGH, "variable"])
    raw_severe = set(raw.vif_table.loc[raw.vif_table["vif"] > VIF_SEVERE, "variable"])
    diff_severe = set(diff1.vif_table.loc[diff1.vif_table["vif"] > VIF_SEVERE, "variable"])
    return {
        "raw_high_count": len(raw_high),
        "diff_high_count": len(diff_high),
        "raw_severe_count": len(raw_severe),
        "diff_severe_count": len(diff_severe),
        "persisting_high": sorted(raw_high & diff_high),
        "raw_only_high": sorted(raw_high - diff_high),
        "diff_only_high": sorted(diff_high - raw_high),
        "persisting_severe": sorted(raw_severe & diff_severe),
    }


def render_target_comparison_plot(raw: LevelTargetResult, diff1: LevelTargetResult, figure_path: Path) -> None:
    labels = ["high (>5)", "severe (>10)"]
    raw_counts = [int((raw.vif_table["vif"] > VIF_HIGH).sum()), int((raw.vif_table["vif"] > VIF_SEVERE).sum())]
    diff_counts = [int((diff1.vif_table["vif"] > VIF_HIGH).sum()), int((diff1.vif_table["vif"] > VIF_SEVERE).sum())]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, raw_counts, width, label="raw", color="#4c78a8")
    ax.bar(x + width / 2, diff_counts, width, label="diff1", color="#72b7b2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("predictor count")
    ax.set_title(f"VIF severity counts | {raw.target}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def recommendation_lines(result: LevelTargetResult) -> list[str]:
    lines: list[str] = []
    if result.group_summary.empty:
        lines.append(
            "- No high-VIF group crossed the pairwise support threshold; keep the current predictor pool but monitor the top-ranked VIF variables before modeling."
        )
        return lines
    for row in result.group_summary.itertuples(index=False):
        lines.append(
            f"- {row.group_id}: keep `{row.representative}` as the first representative candidate; members [{row.members}] form a redundant block (max VIF={row.group_max_vif:.2f}, max |corr|={row.group_max_abs_corr:.3f})."
        )
    return lines


def write_report(
    output_dir: Path,
    results: dict[tuple[str, str], LevelTargetResult],
    comparisons: dict[str, dict[str, Any]],
    summary_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# df.csv Multicollinearity Analysis")
    lines.append("")
    lines.append("## Method")
    lines.append("- Analysis unit: target-specific design matrix")
    lines.append("- Targets: `Com_CrudeOil`, `Com_BrentCrudeOil`")
    lines.append("- Levels: raw and first difference")
    lines.append("- Statsmodels anchor: `variance_inflation_factor`, `OLSResults.condition_number`, `OLSResults.eigenvals`")
    lines.append("- Condition/eigen diagnostics use z-scored predictors plus intercept for scale comparability across mixed-unit variables.")
    lines.append("- For readability, VIF figures cap infinite values at 110 percent of the largest finite plotted VIF; CSV tables keep the raw infinite values.")
    lines.append("")
    lines.append("## Fixed Rule Set")
    lines.append("- `VIF > 5`: high multicollinearity")
    lines.append("- `VIF > 10`: severe multicollinearity")
    lines.append("- `|corr| >= 0.8`: supporting pairwise-correlation flag")
    lines.append("- `condition number > 30`: matrix-conditioning warning")
    lines.append("- `near-zero eigenvalue` or eigenvalue-ratio collapse (`min/max <= 1e-6`): near-singularity warning")
    lines.append("")
    lines.append("## Artifact Inventory")
    lines.append(f"- summary: `{summary_path.name}`")
    lines.append("- tables: `tables/`")
    lines.append("- figures: `figures/`")
    lines.append("")

    for target in TARGETS:
        lines.append(f"## Target: `{target}`")
        lines.append("")
        comp = comparisons[target]
        lines.append("### Raw vs diff comparison")
        lines.append(
            f"- high-VIF counts: raw={comp['raw_high_count']}, diff1={comp['diff_high_count']}"
        )
        lines.append(
            f"- severe-VIF counts: raw={comp['raw_severe_count']}, diff1={comp['diff_severe_count']}"
        )
        lines.append(
            f"- persisting high-VIF variables: {', '.join(comp['persisting_high'][:12]) if comp['persisting_high'] else 'none'}"
        )
        lines.append(
            f"- raw-only high-VIF variables: {', '.join(comp['raw_only_high'][:12]) if comp['raw_only_high'] else 'none'}"
        )
        lines.append(
            f"- diff-only high-VIF variables: {', '.join(comp['diff_only_high'][:12]) if comp['diff_only_high'] else 'none'}"
        )
        lines.append(
            f"- comparison figure: `figures/vif_count_compare_{target.lower()}.png`"
        )
        lines.append("")

        for level in ["raw", "diff1"]:
            result = results[(target, level)]
            lines.append(f"### {level}")
            lines.append(f"- observations used: {result.n_obs}")
            lines.append(f"- predictors used: {result.n_predictors}")
            lines.append(
                f"- high-VIF predictors: {(result.vif_table['vif'] > VIF_HIGH).sum()} | severe predictors: {(result.vif_table['vif'] > VIF_SEVERE).sum()}"
            )
            lines.append(
                f"- condition number: {result.condition_number:.2f} {'(warning)' if result.condition_number > CONDITION_WARNING else '(ok)'}"
            )
            lines.append(
                f"- min eigenvalue: {result.min_eigenvalue:.6g}; eigenvalue ratio: {result.eigenvalue_ratio:.6g}; near singularity: {'yes' if result.near_singularity else 'no'}"
            )
            if result.dropped_zero_variance:
                lines.append(
                    f"- dropped zero-variance predictors: {', '.join(result.dropped_zero_variance)}"
                )
            lines.append(
                f"- top VIF figure: `figures/{result.top_vif_plot}`"
            )
            if result.flagged_heatmap:
                lines.append(
                    f"- flagged correlation heatmap: `figures/{result.flagged_heatmap}`"
                )
            else:
                lines.append("- flagged correlation heatmap: none (no flagged support group)")
            lines.append(
                f"- full VIF table: `tables/vif_{target.lower()}_{level}.csv`"
            )
            lines.append(
                f"- flagged pairwise correlations: `tables/flagged_pairwise_{target.lower()}_{level}.csv`"
            )
            lines.append(
                f"- grouped proposals: `tables/group_summary_{target.lower()}_{level}.csv`"
            )
            lines.append("- variable reduction suggestions:")
            lines.extend(recommendation_lines(result))
            lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(results: dict[tuple[str, str], LevelTargetResult], comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_result = {}
    for (target, level), result in results.items():
        per_result[f"{target}:{level}"] = {
            "n_obs": result.n_obs,
            "n_predictors": result.n_predictors,
            "dropped_zero_variance": result.dropped_zero_variance,
            "high_vif_count": int((result.vif_table["vif"] > VIF_HIGH).sum()),
            "severe_vif_count": int((result.vif_table["vif"] > VIF_SEVERE).sum()),
            "condition_number": result.condition_number,
            "condition_warning": result.condition_number > CONDITION_WARNING,
            "min_eigenvalue": result.min_eigenvalue,
            "max_eigenvalue": result.max_eigenvalue,
            "eigenvalue_ratio": result.eigenvalue_ratio,
            "near_singularity": result.near_singularity,
            "top_vif_variables": result.vif_table.head(10)[["variable", "vif", "severity"]].to_dict(orient="records"),
            "group_count": int(len(result.group_summary)),
        }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rule_set": {
            "vif_high": VIF_HIGH,
            "vif_severe": VIF_SEVERE,
            "pairwise_abs_corr_flag": CORR_FLAG,
            "condition_warning": CONDITION_WARNING,
            "eigenvalue_ratio_warning": EIGENVALUE_RATIO_WARNING,
        },
        "results": per_result,
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if args.output_dir is None:
        output_dir = default_output_dir()
    else:
        output_dir = Path(args.output_dir)

    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")

    frame = load_frame(input_path)
    levels = build_level_frames(frame)

    results: dict[tuple[str, str], LevelTargetResult] = {}
    for target in TARGETS:
        for level_name, level_frame in levels.items():
            results[(target, level_name)] = analyze_level_target(
                frame=level_frame,
                target=target,
                level=level_name,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
            )

    comparisons = {}
    for target in TARGETS:
        raw_result = results[(target, "raw")]
        diff_result = results[(target, "diff1")]
        comparisons[target] = comparison_summary(raw_result, diff_result)
        render_target_comparison_plot(
            raw_result,
            diff_result,
            figures_dir / f"vif_count_compare_{target.lower()}.png",
        )

    summary = build_summary(results, comparisons)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, results, comparisons, summary_path)

    print(json.dumps({
        "output_dir": str(output_dir),
        "report_path": str(output_dir / 'report.md'),
        "summary_path": str(summary_path),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
