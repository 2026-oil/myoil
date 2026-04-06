from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import genextreme
from statsmodels.tsa.stattools import adfuller

TARGET_COLUMNS = ["Com_CrudeOil", "Com_BrentCrudeOil"]
TAIL_DIRECTIONS = ["upper", "lower"]
REPORT_PROBABILITIES = [0.8, 0.9, 0.95]
FORECAST_HORIZON_WEEKS = 8
MIN_EXTREME_BLOCKS = 8


@dataclass
class AnalysisContext:
    input_path: Path
    output_dir: Path
    contract_dir: Path
    tables_dir: Path


@dataclass
class RepresentationCandidate:
    name: str
    label: str
    rationale: str
    valid: bool
    adf_pvalue: float | None
    n_obs: int
    reason_rejected: str | None


@dataclass
class TailFitResult:
    target: str
    tail: str
    status: str
    sign_convention: str
    block_count: int
    block_size_weeks: int
    fit_shape: float | None
    fit_loc: float | None
    fit_scale: float | None
    observed_extreme: float | None
    fitted_quantiles: dict[str, float]
    notes: list[str]
    csv_name: str


@dataclass
class TargetAnalysis:
    target: str
    chosen_representation: RepresentationCandidate
    representation_reason: str
    candidates: list[RepresentationCandidate]
    upper_tail: TailFitResult
    lower_tail: TailFitResult


@dataclass
class BlockExtreme:
    block_id: int
    dt_start: str
    dt_end: str
    original_value: float
    transformed_value: float


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"dfcsv-oil-gev-{timestamp}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_context(input_path: Path, output_dir: Path | None = None) -> AnalysisContext:
    actual_output = ensure_dir(output_dir or default_output_dir())
    return AnalysisContext(
        input_path=input_path,
        output_dir=actual_output,
        contract_dir=ensure_dir(actual_output / "contract"),
        tables_dir=ensure_dir(actual_output / "tables"),
    )


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "dt" not in df.columns:
        raise KeyError("Input frame must include dt column.")
    missing_targets = [target for target in TARGET_COLUMNS if target not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing required target columns: {missing_targets}")
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    if df["dt"].isna().any():
        raise ValueError("dt column contains non-parsable timestamps.")
    return df.sort_values("dt").reset_index(drop=True)


def build_audit(df: pd.DataFrame) -> dict[str, Any]:
    dt = df["dt"]
    diffs = dt.diff().dropna().dt.days.astype(int)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_min": dt.min().strftime("%Y-%m-%d"),
        "date_max": dt.max().strftime("%Y-%m-%d"),
        "day_diffs": sorted(diffs.unique().tolist()),
        "all_monday": bool((dt.dt.dayofweek == 0).all()),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
        "targets": {
            target: {
                "min": float(df[target].min()),
                "max": float(df[target].max()),
                "mean": float(df[target].mean()),
            }
            for target in TARGET_COLUMNS
        },
    }


def _adf_pvalue(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 20:
        return None
    if float(clean.std(ddof=0)) == 0.0:
        return None
    try:
        return float(adfuller(clean.to_numpy(), autolag="AIC")[1])
    except Exception:
        return None


def build_representation_candidates(series: pd.Series) -> list[tuple[RepresentationCandidate, pd.Series]]:
    clean = pd.to_numeric(series, errors="coerce")
    raw = clean.copy()
    diff1 = clean.diff()
    candidates: list[tuple[RepresentationCandidate, pd.Series]] = []

    raw_candidate = RepresentationCandidate(
        name="level",
        label="raw level",
        rationale="Directly interpretable oil-price level, but may retain trend/nonstationarity.",
        valid=True,
        adf_pvalue=_adf_pvalue(raw),
        n_obs=int(raw.dropna().shape[0]),
        reason_rejected=None,
    )
    candidates.append((raw_candidate, raw))

    diff_candidate = RepresentationCandidate(
        name="diff1",
        label="first difference",
        rationale="Captures weekly price change and often improves stationarity for EVT over finite horizons.",
        valid=bool(diff1.dropna().shape[0] > 0),
        adf_pvalue=_adf_pvalue(diff1),
        n_obs=int(diff1.dropna().shape[0]),
        reason_rejected=None if diff1.dropna().shape[0] > 0 else "No finite first differences available.",
    )
    candidates.append((diff_candidate, diff1))

    strictly_positive = bool((clean.dropna() > 0).all())
    if strictly_positive:
        log_return = np.log(clean).diff()
        log_candidate = RepresentationCandidate(
            name="log_return",
            label="log return",
            rationale="Scale-free weekly change; usually the most defensible EVT input for positive price series.",
            valid=bool(log_return.dropna().shape[0] > 0),
            adf_pvalue=_adf_pvalue(log_return),
            n_obs=int(log_return.dropna().shape[0]),
            reason_rejected=None if log_return.dropna().shape[0] > 0 else "No finite log returns available.",
        )
        candidates.append((log_candidate, log_return))
    else:
        log_candidate = RepresentationCandidate(
            name="log_return",
            label="log return",
            rationale="Scale-free weekly change; usually the most defensible EVT input for positive price series.",
            valid=False,
            adf_pvalue=None,
            n_obs=0,
            reason_rejected="Series includes non-positive values, so log return is invalid.",
        )
        candidates.append((log_candidate, pd.Series(index=series.index, dtype=float)))
    return candidates


def choose_representation(candidates: list[tuple[RepresentationCandidate, pd.Series]]) -> tuple[RepresentationCandidate, pd.Series, str]:
    valid = [item for item in candidates if item[0].valid]
    if not valid:
        raise ValueError("No valid representation candidates available.")

    stationary = [
        item for item in valid if item[0].adf_pvalue is not None and item[0].adf_pvalue <= 0.10
    ]
    pool = stationary or valid
    chosen_candidate, chosen_series = min(
        pool,
        key=lambda item: (
            1.0 if item[0].adf_pvalue is None else item[0].adf_pvalue,
            0 if item[0].name == "log_return" else 1,
            0 if item[0].name == "diff1" else 1,
        ),
    )
    reason = (
        f"Selected `{chosen_candidate.name}` because it offers the strongest available stationarity signal "
        f"(ADF p-value={_format_float(chosen_candidate.adf_pvalue)}) while preserving weekly tail behavior for an {FORECAST_HORIZON_WEEKS}-week horizon."
    )
    if chosen_candidate.name == "level":
        reason += " No stationary transform beat raw levels on this dataset, so the report must warn about nonstationarity risk."
    return chosen_candidate, chosen_series, reason


def _format_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _block_extremes(series: pd.Series, dt: pd.Series, tail: str) -> list[BlockExtreme]:
    clean = pd.DataFrame({"dt": dt, "value": series}).dropna().reset_index(drop=True)
    extremes: list[BlockExtreme] = []
    for start in range(0, len(clean), FORECAST_HORIZON_WEEKS):
        block = clean.iloc[start : start + FORECAST_HORIZON_WEEKS]
        if len(block) < FORECAST_HORIZON_WEEKS:
            continue
        if tail == "upper":
            idx = int(block["value"].idxmax())
            transformed = float(clean.loc[idx, "value"])
        else:
            idx = int(block["value"].idxmin())
            transformed = float(-clean.loc[idx, "value"])
        row = clean.loc[idx]
        extremes.append(
            BlockExtreme(
                block_id=len(extremes) + 1,
                dt_start=block.iloc[0]["dt"].strftime("%Y-%m-%d"),
                dt_end=block.iloc[-1]["dt"].strftime("%Y-%m-%d"),
                original_value=float(row["value"]),
                transformed_value=transformed,
            )
        )
    if not extremes:
        raise ValueError(f"No complete {FORECAST_HORIZON_WEEKS}-week blocks available for {tail} tail analysis.")
    return extremes


def _quantile_map(fit_shape: float, fit_loc: float, fit_scale: float, tail: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for probability in REPORT_PROBABILITIES:
        fitted = float(genextreme.ppf(probability, fit_shape, loc=fit_loc, scale=fit_scale))
        if tail == "lower":
            fitted = -fitted
        values[f"q{int(probability * 100)}"] = fitted
    return values


def fit_tail(
    target: str,
    series: pd.Series,
    dt: pd.Series,
    tail: str,
) -> tuple[TailFitResult, pd.DataFrame]:
    sign_convention = (
        "Upper tail is modeled on the chosen representation directly."
        if tail == "upper"
        else "Lower tail is modeled by fitting GEV to the negated representation so larger transformed values mean deeper downside events."
    )
    extremes = _block_extremes(series, dt, tail)
    table = pd.DataFrame([asdict(item) for item in extremes])
    csv_name = f"{target.lower()}_{tail}_tail.csv"
    table.insert(0, "target", target)
    table.insert(1, "tail", tail)
    table["sign_convention"] = sign_convention
    transformed = table["transformed_value"].to_numpy(dtype=float)
    notes: list[str] = []

    if len(transformed) < MIN_EXTREME_BLOCKS:
        notes.append(
            f"Insufficient tail evidence: only {len(transformed)} complete {FORECAST_HORIZON_WEEKS}-week blocks, need at least {MIN_EXTREME_BLOCKS} for a stable GEV fit."
        )
        return (
            TailFitResult(
                target=target,
                tail=tail,
                status="insufficient_evidence",
                sign_convention=sign_convention,
                block_count=int(len(transformed)),
                block_size_weeks=FORECAST_HORIZON_WEEKS,
                fit_shape=None,
                fit_loc=None,
                fit_scale=None,
                observed_extreme=float(table["original_value"].max() if tail == "upper" else table["original_value"].min()),
                fitted_quantiles={},
                notes=notes,
                csv_name=csv_name,
            ),
            table,
        )

    fit_shape, fit_loc, fit_scale = [float(value) for value in genextreme.fit(transformed)]
    quantiles = _quantile_map(fit_shape, fit_loc, fit_scale, tail)
    observed_extreme = float(table["original_value"].max() if tail == "upper" else table["original_value"].min())
    notes.append(
        f"Fitted on {len(transformed)} non-overlapping {FORECAST_HORIZON_WEEKS}-week block extremes."
    )
    if tail == "lower":
        notes.append("Reported fitted quantiles were mapped back from the negated scale into the original representation.")
    return (
        TailFitResult(
            target=target,
            tail=tail,
            status="ok",
            sign_convention=sign_convention,
            block_count=int(len(transformed)),
            block_size_weeks=FORECAST_HORIZON_WEEKS,
            fit_shape=fit_shape,
            fit_loc=fit_loc,
            fit_scale=fit_scale,
            observed_extreme=observed_extreme,
            fitted_quantiles=quantiles,
            notes=notes,
            csv_name=csv_name,
        ),
        table,
    )


def analyze_target(df: pd.DataFrame, target: str) -> tuple[TargetAnalysis, dict[str, pd.DataFrame]]:
    candidates = build_representation_candidates(df[target])
    chosen, chosen_series, reason = choose_representation(candidates)
    upper_tail, upper_table = fit_tail(target, chosen_series, df["dt"], tail="upper")
    lower_tail, lower_table = fit_tail(target, chosen_series, df["dt"], tail="lower")
    return (
        TargetAnalysis(
            target=target,
            chosen_representation=chosen,
            representation_reason=reason,
            candidates=[item[0] for item in candidates],
            upper_tail=upper_tail,
            lower_tail=lower_tail,
        ),
        {
            "upper": upper_table,
            "lower": lower_table,
        },
    )


def build_manifest(input_path: Path, analyses: list[TargetAnalysis]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "target_columns": TARGET_COLUMNS,
        "tail_directions": TAIL_DIRECTIONS,
        "block_policy": {
            "kind": "non_overlapping_fixed_horizon",
            "block_size_weeks": FORECAST_HORIZON_WEEKS,
            "rationale": f"Matches the user’s {FORECAST_HORIZON_WEEKS}-week forecasting horizon.",
        },
        "lower_tail_sign_convention": "Fit GEV to the negated chosen representation; convert fitted quantiles back to original sign for reporting.",
        "targets": {
            analysis.target: {
                "chosen_representation": {
                    "name": analysis.chosen_representation.name,
                    "label": analysis.chosen_representation.label,
                    "adf_pvalue": analysis.chosen_representation.adf_pvalue,
                    "n_obs": analysis.chosen_representation.n_obs,
                },
                "representation_reason": analysis.representation_reason,
                "candidate_representations": [asdict(candidate) for candidate in analysis.candidates],
            }
            for analysis in analyses
        },
    }


def build_summary(context: AnalysisContext, analyses: list[TargetAnalysis]) -> dict[str, Any]:
    artifacts = {
        "manifest": str(context.contract_dir / "analysis_manifest.json"),
        "audit": str(context.contract_dir / "dataset_audit.json"),
        "report": str(context.output_dir / "report.md"),
    }
    targets: dict[str, Any] = {}
    for analysis in analyses:
        targets[analysis.target] = {
            "representation": analysis.chosen_representation.name,
            "representation_reason": analysis.representation_reason,
            "tails": {
                "upper": asdict(analysis.upper_tail),
                "lower": asdict(analysis.lower_tail),
            },
        }
        artifacts[f"{analysis.target}_upper_table"] = str(context.tables_dir / analysis.upper_tail.csv_name)
        artifacts[f"{analysis.target}_lower_table"] = str(context.tables_dir / analysis.lower_tail.csv_name)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(context.output_dir),
        "target_count": len(targets),
        "tail_directions": TAIL_DIRECTIONS,
        "targets": targets,
        "artifacts": artifacts,
    }


def _tail_lines(result: TailFitResult) -> list[str]:
    lines = [
        f"- status: {result.status}",
        f"- block_count: {result.block_count}",
        f"- block_size_weeks: {result.block_size_weeks}",
        f"- observed_extreme: {_format_float(result.observed_extreme)}",
        f"- sign_convention: {result.sign_convention}",
        f"- table: `tables/{result.csv_name}`",
    ]
    if result.status == "ok":
        lines.extend(
            [
                f"- fit_shape: {_format_float(result.fit_shape)}",
                f"- fit_loc: {_format_float(result.fit_loc)}",
                f"- fit_scale: {_format_float(result.fit_scale)}",
                f"- fitted_quantiles: {json.dumps(result.fitted_quantiles, ensure_ascii=False)}",
            ]
        )
    for note in result.notes:
        lines.append(f"- note: {note}")
    return lines


def build_report(context: AnalysisContext, analyses: list[TargetAnalysis], audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GEV Oil Extremes Analysis for 8-Week Forecasting")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Input: `data/df.csv`")
    lines.append("- Targets: `Com_CrudeOil`, `Com_BrentCrudeOil`")
    lines.append("- Tails: upside + downside")
    lines.append("- Method family: block-extreme GEV framing")
    lines.append("- Non-goal: recommendation-only output; no feature engineering or target-transform implementation commitment")
    lines.append("")
    lines.append("## Dataset audit")
    lines.append(f"- rows: {audit['rows']}")
    lines.append(f"- date range: {audit['date_min']} .. {audit['date_max']}")
    lines.append(f"- weekly diffs (days): {audit['day_diffs']}")
    lines.append("")
    lines.append("## Block policy")
    lines.append(f"- Non-overlapping block size: {FORECAST_HORIZON_WEEKS} weeks")
    lines.append(f"- Rationale: align extremes with the user\'s {FORECAST_HORIZON_WEEKS}-week forecasting horizon")
    lines.append("")
    lines.append("## Representation rationale")
    lines.append(
        "The analysis compares raw level, first difference, and log return when valid. For each target, the chosen representation is the most stationary viable candidate, with explicit ADF evidence and no silent substitution."
    )
    lines.append("")
    for analysis in analyses:
        lines.append(f"## Target: `{analysis.target}`")
        lines.append(
            f"- chosen_representation: `{analysis.chosen_representation.name}` ({analysis.chosen_representation.label})"
        )
        lines.append(f"- rationale: {analysis.representation_reason}")
        lines.append("- candidate_representations:")
        for candidate in analysis.candidates:
            status = "valid" if candidate.valid else f"rejected: {candidate.reason_rejected}"
            lines.append(
                f"  - `{candidate.name}` | status={status} | adf_pvalue={_format_float(candidate.adf_pvalue)} | n_obs={candidate.n_obs}"
            )
        lines.append("")
        lines.append("### Upper tail")
        lines.extend(_tail_lines(analysis.upper_tail))
        lines.append("")
        lines.append("### Lower tail")
        lines.extend(_tail_lines(analysis.lower_tail))
        lines.append("")
    lines.append("## Cross-series comparison")
    for analysis in analyses:
        upper_status = analysis.upper_tail.status
        lower_status = analysis.lower_tail.status
        lines.append(
            f"- `{analysis.target}`: representation=`{analysis.chosen_representation.name}`, upper_status={upper_status}, lower_status={lower_status}"
        )
    lines.append("")
    lines.append("## 8-week forecasting strategy recommendations")
    lines.append(
        "- Use the chosen representation and tail quantiles as a risk overlay for the next 8-week horizon: when current conditions resemble historically extreme blocks, widen scenario ranges rather than pretending the point forecast is equally reliable."
    )
    lines.append(
        "- Treat upside and downside tails separately. A market that is calm on average can still have asymmetric downside risk, so the forecasting workflow should track upper-tail and lower-tail alerts independently."
    )
    lines.append(
        "- If one target shows weaker stationarity or insufficient tail evidence, keep that series in a lower-confidence regime bucket instead of forcing precise tail-based rules."
    )
    lines.append(
        "- Recommendation only: these findings suggest possible model-side features or scenario gates, but they do not commit the project to a final feature-engineering or target-transform design in this task."
    )
    lines.append("")
    lines.append("## Recommendation boundary")
    lines.append("- This report stops at analytical interpretation and forecasting strategy guidance.")
    lines.append("- It does not implement or finalize feature, loss, target, or model configuration changes.")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
