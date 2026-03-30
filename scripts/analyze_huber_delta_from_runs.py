#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OIL_TARGETS = ("Com_BrentCrudeOil", "Com_CrudeOil")
MAD_NORMALIZER = 0.6744897501960817
HUBER_EFFICIENCY_COEF = 1.345


@dataclass(frozen=True)
class ResidualScope:
    label: str
    rows: int
    files: int
    run_roots: int
    median_resid: float
    median_abs: float
    mean_abs: float
    rmse: float
    mad: float
    sigma_mad: float
    delta_exact: float
    delta_round_1dp: float
    delta_round_0_5: float
    p75_abs: float
    p80_abs: float
    p85_abs: float
    p90_abs: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "rows": self.rows,
            "files": self.files,
            "run_roots": self.run_roots,
            "median_resid": self.median_resid,
            "median_abs": self.median_abs,
            "mean_abs": self.mean_abs,
            "rmse": self.rmse,
            "mad": self.mad,
            "sigma_mad": self.sigma_mad,
            "delta_exact": self.delta_exact,
            "delta_round_1dp": self.delta_round_1dp,
            "delta_round_0_5": self.delta_round_0_5,
            "p75_abs": self.p75_abs,
            "p80_abs": self.p80_abs,
            "p85_abs": self.p85_abs,
            "p90_abs": self.p90_abs,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate historical run forecasts and recommend a HuberLoss delta."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory that contains historical run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary/report artifacts will be written.",
    )
    return parser.parse_args()


def derive_run_root(path: Path) -> str:
    parts = list(path.parts)
    if "scheduler" in parts:
        return str(path.parents[4])
    if len(parts) >= 2 and parts[-2] == "cv":
        return str(path.parents[1])
    return str(path.parent)


def normalize_modes(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("requested_mode", "validated_mode", "fold_idx"):
        if column not in frame.columns:
            frame[column] = pd.NA
    if (
        frame["validated_mode"].isna().all()
        and frame["model"].astype(str).eq("Naive").all()
    ):
        frame["validated_mode"] = "baseline_fixed"
        frame["requested_mode"] = frame["requested_mode"].fillna("baseline_fixed")
    return frame


def load_forecasts(runs_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    records: list[pd.DataFrame] = []
    skipped: list[str] = []
    for path in sorted(runs_dir.glob("**/*_forecasts.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - surfaced in summary.json
            skipped.append(f"{path}: read_error={exc}")
            continue
        required = {"model", "unique_id", "y", "y_hat"}
        if not required.issubset(frame.columns):
            skipped.append(
                f"{path}: missing_columns={sorted(required - set(frame.columns))}"
            )
            continue
        frame = normalize_modes(frame).dropna(subset=["y", "y_hat"]).copy()
        if frame.empty:
            skipped.append(f"{path}: empty_after_dropna")
            continue
        frame["source_path"] = str(path)
        frame["run_root"] = derive_run_root(path)
        frame["is_legacy"] = frame["source_path"].str.contains(r"/legacy/")
        frame["resid"] = frame["y"] - frame["y_hat"]
        frame["abs_resid"] = frame["resid"].abs()
        records.append(
            frame[
                [
                    "model",
                    "requested_mode",
                    "validated_mode",
                    "fold_idx",
                    "unique_id",
                    "source_path",
                    "run_root",
                    "is_legacy",
                    "y",
                    "y_hat",
                    "resid",
                    "abs_resid",
                ]
            ]
        )
    if not records:
        raise SystemExit(f"No usable *_forecasts.csv files found under {runs_dir}")
    return pd.concat(records, ignore_index=True), skipped


def round_half(value: float) -> float:
    return round(value * 2.0) / 2.0


def robust_delta(residuals: np.ndarray) -> tuple[float, float, float]:
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    sigma_mad = 0.0 if mad == 0.0 else mad / MAD_NORMALIZER
    return mad, sigma_mad, HUBER_EFFICIENCY_COEF * sigma_mad


def summarize_scope(label: str, frame: pd.DataFrame) -> ResidualScope:
    residuals = frame["resid"].to_numpy(dtype=float)
    absolute = np.abs(residuals)
    mad, sigma_mad, delta_exact = robust_delta(residuals)
    return ResidualScope(
        label=label,
        rows=int(len(frame)),
        files=int(frame["source_path"].nunique()),
        run_roots=int(frame["run_root"].nunique()),
        median_resid=float(np.median(residuals)),
        median_abs=float(np.median(absolute)),
        mean_abs=float(np.mean(absolute)),
        rmse=float(np.sqrt(np.mean(np.square(residuals)))),
        mad=mad,
        sigma_mad=sigma_mad,
        delta_exact=float(delta_exact),
        delta_round_1dp=round(float(delta_exact), 1),
        delta_round_0_5=round_half(float(delta_exact)),
        p75_abs=float(np.quantile(absolute, 0.75)),
        p80_abs=float(np.quantile(absolute, 0.80)),
        p85_abs=float(np.quantile(absolute, 0.85)),
        p90_abs=float(np.quantile(absolute, 0.90)),
    )


def collect_scope_rows(scopes: Iterable[ResidualScope]) -> pd.DataFrame:
    return pd.DataFrame([scope.to_dict() for scope in scopes])


def sensitivity_table(frame: pd.DataFrame, deltas: Iterable[float]) -> pd.DataFrame:
    absolute = frame["abs_resid"].to_numpy(dtype=float)
    rows = []
    for delta in deltas:
        rows.append(
            {
                "delta": float(delta),
                "fraction_abs_resid_gt_delta": float(np.mean(absolute > delta)),
            }
        )
    return pd.DataFrame(rows)


def frame_to_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = [headers, ["---"] * len(headers)]
    for record in frame.itertuples(index=False, name=None):
        rows.append([str(value) for value in record])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_report(
    output_dir: Path,
    total_frame: pd.DataFrame,
    oil_learned_scope: ResidualScope,
    target_scopes: list[ResidualScope],
    legacy_scope: ResidualScope,
    current_scope: ResidualScope,
    per_model: pd.DataFrame,
    per_run: pd.DataFrame,
    sensitivity: pd.DataFrame,
    skipped: list[str],
) -> None:
    target_rows = {
        scope.label: scope for scope in target_scopes if scope.label.startswith("target:")
    }
    brent = target_rows["target:Com_BrentCrudeOil:learned"]
    wti = target_rows["target:Com_CrudeOil:learned"]
    model_table = frame_to_markdown_table(per_model)
    sensitivity_markdown = frame_to_markdown_table(sensitivity)
    report = f"""# HuberLoss delta recommendation from historical runs

## Recommendation
- **Primary global delta for oil learned models:** **{oil_learned_scope.delta_round_1dp:.1f}** (exact robust estimate {oil_learned_scope.delta_exact:.4f})
- **Config-friendly rounded default:** **{oil_learned_scope.delta_round_0_5:.1f}**
- **Target-specific fallback:** Brent {brent.delta_round_1dp:.1f}, WTI {wti.delta_round_1dp:.1f}

## Method
1. Scan every `*_forecasts.csv` reachable under `runs/`.
2. Compute residuals `e = y - y_hat`.
3. Estimate a robust scale with `sigma ~= MAD(e) / 0.67448975`.
4. Convert that scale into a Huber threshold with the common 95 percent Gaussian-efficiency rule `delta = 1.345 * sigma`.

## Coverage
- forecast files scanned: {total_frame['source_path'].nunique()}
- run roots scanned: {total_frame['run_root'].nunique()}
- residual rows scanned: {len(total_frame)}
- oil learned residual rows used for the main recommendation: {oil_learned_scope.rows}
- skipped forecast files: {len(skipped)}

## Why {oil_learned_scope.delta_round_1dp:.1f}
- Oil learned residuals produced an exact robust delta of **{oil_learned_scope.delta_exact:.4f}**.
- Brent and WTI learned subsets landed at **{brent.delta_exact:.4f}** and **{wti.delta_exact:.4f}**, so {oil_learned_scope.delta_round_1dp:.1f} sits between both targets instead of overfitting one side.
- At delta={oil_learned_scope.delta_round_1dp:.1f}, about **{100 * sensitivity.loc[sensitivity['delta'].eq(oil_learned_scope.delta_round_1dp), 'fraction_abs_resid_gt_delta'].iloc[0]:.1f}%** of oil learned residuals remain in the linear Huber regime.

## Distribution notes
- Legacy oil learned delta: {legacy_scope.delta_round_1dp:.1f}
- Non-legacy oil learned delta: {current_scope.delta_round_1dp:.1f}
- Per-run oil learned delta median: {per_run['delta_exact'].median():.4f}
- Per-run oil learned delta IQR: {per_run['delta_exact'].quantile(0.25):.4f} to {per_run['delta_exact'].quantile(0.75):.4f}

## Model-level context (oil learned only)
{model_table}

## Sensitivity
{sensitivity_markdown}

## Notes
- `BS_Core_Index_Integrated` is tracked separately in `per_target_stats.csv`; it was excluded from the primary oil recommendation because its scale is materially smaller than Brent/WTI.
- `Naive` rows are included in coverage summaries but excluded from the primary recommendation because HuberLoss is intended for trainable learned models.
- Full tables are saved beside this report.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_frame, skipped = load_forecasts(args.runs_dir)

    oil_frame = forecast_frame[forecast_frame["unique_id"].isin(OIL_TARGETS)].copy()
    oil_learned = oil_frame[
        oil_frame["validated_mode"].fillna("").str.contains("learned")
    ].copy()
    oil_legacy = oil_learned[oil_learned["is_legacy"]].copy()
    oil_current = oil_learned[~oil_learned["is_legacy"]].copy()

    scope_rows = [
        summarize_scope("all_forecasts", forecast_frame),
        summarize_scope("oil_all", oil_frame),
        summarize_scope("oil_learned", oil_learned),
        summarize_scope("oil_learned_legacy", oil_legacy),
        summarize_scope("oil_learned_non_legacy", oil_current),
    ]
    for target, frame in forecast_frame.groupby("unique_id"):
        scope_rows.append(summarize_scope(f"target:{target}", frame))
        learned = frame[frame["validated_mode"].fillna("").str.contains("learned")]
        if not learned.empty:
            scope_rows.append(summarize_scope(f"target:{target}:learned", learned))

    scope_table = collect_scope_rows(scope_rows)
    scope_table.to_csv(output_dir / "scope_stats.csv", index=False)

    oil_per_model_rows = []
    for model, group in oil_learned.groupby("model", sort=True):
        row = summarize_scope(model, group).to_dict()
        row["model"] = row.pop("label")
        oil_per_model_rows.append(row)
    oil_per_model = pd.DataFrame(oil_per_model_rows).sort_values("delta_exact")
    oil_per_model.to_csv(output_dir / "oil_learned_per_model_stats.csv", index=False)

    oil_per_run_rows = []
    for (run_root, unique_id), group in oil_learned.groupby(
        ["run_root", "unique_id"], sort=True
    ):
        scope = summarize_scope("tmp", group)
        oil_per_run_rows.append(
            {
                "run_root": run_root,
                "unique_id": unique_id,
                "rows": int(len(group)),
                "files": int(group["source_path"].nunique()),
                "delta_exact": scope.delta_exact,
                "delta_round_1dp": scope.delta_round_1dp,
                "delta_round_0_5": scope.delta_round_0_5,
                "mean_abs": float(group["abs_resid"].mean()),
                "median_abs": float(group["abs_resid"].median()),
                "rmse": float(
                    np.sqrt(np.mean(np.square(group["resid"].to_numpy(dtype=float))))
                ),
            }
        )
    oil_per_run = pd.DataFrame(oil_per_run_rows).sort_values(
        ["unique_id", "delta_exact"]
    )
    oil_per_run.to_csv(output_dir / "oil_learned_per_run_deltas.csv", index=False)

    oil_scope = scope_table.loc[scope_table["label"].eq("oil_learned")].iloc[0]
    sensitivity = sensitivity_table(
        oil_learned,
        [
            5.0,
            5.5,
            round(float(oil_scope["delta_round_1dp"]), 1),
            6.5,
            7.0,
            8.0,
        ],
    ).drop_duplicates(subset=["delta"])
    sensitivity.to_csv(output_dir / "oil_learned_delta_sensitivity.csv", index=False)

    write_report(
        output_dir=output_dir,
        total_frame=forecast_frame,
        oil_learned_scope=summarize_scope("oil_learned", oil_learned),
        target_scopes=[
            summarize_scope("target:Com_BrentCrudeOil:learned", oil_learned[oil_learned["unique_id"].eq("Com_BrentCrudeOil")]),
            summarize_scope("target:Com_CrudeOil:learned", oil_learned[oil_learned["unique_id"].eq("Com_CrudeOil")]),
        ],
        legacy_scope=summarize_scope("oil_learned_legacy", oil_legacy),
        current_scope=summarize_scope("oil_learned_non_legacy", oil_current),
        per_model=oil_per_model[
            ["model", "rows", "delta_exact", "delta_round_1dp", "mean_abs", "rmse"]
        ],
        per_run=oil_per_run,
        sensitivity=sensitivity,
        skipped=skipped,
    )

    summary_payload = {
        "runs_dir": str(args.runs_dir),
        "output_dir": str(output_dir),
        "files_scanned": int(forecast_frame["source_path"].nunique()),
        "run_roots_scanned": int(forecast_frame["run_root"].nunique()),
        "rows_scanned": int(len(forecast_frame)),
        "skipped_file_count": len(skipped),
        "recommended_delta": {
            "scope": "oil_learned",
            "exact": float(oil_scope["delta_exact"]),
            "rounded_1dp": float(oil_scope["delta_round_1dp"]),
            "rounded_0_5": float(oil_scope["delta_round_0_5"]),
        },
        "target_specific": {
            "Com_BrentCrudeOil": float(
                scope_table.loc[
                    scope_table["label"].eq("target:Com_BrentCrudeOil:learned"),
                    "delta_round_1dp",
                ].iloc[0]
            ),
            "Com_CrudeOil": float(
                scope_table.loc[
                    scope_table["label"].eq("target:Com_CrudeOil:learned"),
                    "delta_round_1dp",
                ].iloc[0]
            ),
            "BS_Core_Index_Integrated": float(
                scope_table.loc[
                    scope_table["label"].eq("target:BS_Core_Index_Integrated:learned"),
                    "delta_round_1dp",
                ].iloc[0]
            ),
        },
        "notes": [
            "Primary recommendation excludes BS_Core_Index_Integrated because its scale is much smaller than Brent/WTI.",
            "Primary recommendation excludes Naive baseline rows because HuberLoss applies to trainable learned models.",
        ],
        "skipped_files": skipped[:50],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
