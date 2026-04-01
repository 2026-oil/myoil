from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

PRIMARY_COMPONENTS = ["Com_CrudeOil", "Com_BrentCrudeOil"]
EXCLUDED_COLUMNS = ["dt", *PRIMARY_COMPONENTS]
SCREEN_LAGS = list(range(-8, 9))
POSITIVE_LAGS = list(range(1, 9))
HORIZONS = [1, 2, 4, 8]
SIGNIFICANCE_ALPHA = 0.10
MIN_EFFECTIVE_SAMPLE = 156
MIN_TRAIN_SIZE = 260
MAX_FACTOR_COUNT = 5
TOP_LOADING_QUANTILE = 0.9
COHERENCE_THRESHOLD = 0.30
CCF_CORR_THRESHOLD = 0.20
CONTRADICTION_TOLERANCE = 1
HP_LAMBDA = 6.25 * (52**4)
FAMILY_ORDER = [
    "turning_point",
    "ccf",
    "predictive",
    "dfm",
    "oos_dm",
    "frequency",
]
BAND_DEFS = {
    "2_8_weeks": (1 / 8, 1 / 2),
    "8_26_weeks": (1 / 26, 1 / 8),
    "26_52_weeks": (1 / 52, 1 / 26),
}
REQUIRED_FAMILY_COLUMNS = [
    "variable",
    "family",
    "eligible",
    "blocked_reason",
    "representation",
    "support_class",
    "best_lag_weeks",
    "band_label",
    "support_strength",
    "effect_sign",
    "raw_pvalue",
    "fdr_pvalue",
    "stability_flag",
    "source_artifact",
    "notes",
]
FAMILY_EXTRA_COLUMNS = {
    "turning_point": [
        "matched_turns",
        "mean_lead_weeks",
        "median_lead_weeks",
        "false_cycles",
        "missed_cycles",
    ],
    "ccf": ["corr_at_best_lag", "best_lag_abs_corr"],
    "predictive": [
        "selected_lag",
        "path_used",
        "forward_raw_pvalue",
        "forward_fdr_pvalue",
        "reverse_raw_pvalue",
        "reverse_fdr_pvalue",
        "selected_model",
        "directional_effect",
        "generic_pvalue_rule",
    ],
    "dfm": ["factor_id", "abs_loading", "factor_explained_variance", "factor_class", "factor_count"],
    "oos_dm": [
        "row_grain",
        "horizon_weeks",
        "baseline_rmse",
        "candidate_rmse",
        "dm_stat",
        "dm_hln_adjusted",
        "horizon_fdr_pvalue",
        "is_best_horizon",
    ],
    "frequency": ["dominant_band", "coherence_max", "phase_lead_weeks"],
}
REQUIRED_SYNTHESIS_COLUMNS = [
    "variable",
    "eligible_family_count",
    "leading_votes",
    "coincident_votes",
    "lagging_votes",
    "inconclusive_votes",
    "predictive_support",
    "contradiction_count",
    "stability_flag",
    "turning_point_class",
    "turning_point_best_lag_weeks",
    "ccf_class",
    "ccf_best_lag_weeks",
    "predictive_class",
    "predictive_best_lag_weeks",
    "dfm_class",
    "dfm_best_lag_weeks",
    "oos_dm_class",
    "oos_dm_best_lag_weeks",
    "frequency_class",
    "frequency_best_lag_weeks",
    "frequency_band_label",
    "final_class",
    "final_reason_code",
]


@dataclass
class AnalysisContext:
    input_path: Path
    output_dir: Path
    contract_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass
class FamilySelector:
    family: str
    sort_columns: list[str]
    ascending: list[bool]


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"dfcsv-leading-indicators-oil-{timestamp}"


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
        figures_dir=ensure_dir(actual_output / "figures"),
    )


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "dt" not in df.columns:
        raise KeyError("Input frame must include dt column.")
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric columns found in input data.")
    return numeric


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0 or math.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - float(series.mean())) / std


def primary_target(numeric: pd.DataFrame) -> pd.Series:
    zcols = [zscore(numeric[column]) for column in PRIMARY_COMPONENTS]
    combined = sum(zcols) / len(zcols)
    combined.name = "oil_target_primary"
    return combined


def sensitivity_target(numeric: pd.DataFrame) -> pd.Series:
    matrix = np.column_stack([zscore(numeric[column]).to_numpy() for column in PRIMARY_COMPONENTS])
    pca = PCA(n_components=1)
    component = pca.fit_transform(matrix).reshape(-1)
    series = pd.Series(component, index=numeric.index, name="oil_target_pca1")
    corr = float(np.corrcoef(series.to_numpy(), primary_target(numeric).to_numpy())[0, 1])
    if corr < 0:
        series *= -1
    return series


def build_audit(df: pd.DataFrame, numeric: pd.DataFrame) -> dict[str, Any]:
    dt = df["dt"]
    diffs = dt.diff().dropna().dt.days.astype(int)
    predictors = [column for column in numeric.columns if column not in PRIMARY_COMPONENTS]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": int(len(numeric.columns)),
        "predictor_count": int(len(predictors)),
        "excluded_columns": EXCLUDED_COLUMNS,
        "all_monday": bool((dt.dt.dayofweek == 0).all()),
        "day_diffs": sorted(diffs.unique().tolist()),
        "date_min": dt.min().strftime("%Y-%m-%d"),
        "date_max": dt.max().strftime("%Y-%m-%d"),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
    }


def build_manifest(input_path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "primary_target": {
            "kind": "zscore_average",
            "components": PRIMARY_COMPONENTS,
        },
        "sensitivity_target": {
            "kind": "pca1_appendix",
            "components": PRIMARY_COMPONENTS,
        },
        "candidate_universe": {
            "exclude": EXCLUDED_COLUMNS,
            "count": audit["predictor_count"],
        },
        "representations": {
            "turning_point": "cycle_smoothed_level",
            "ccf": ["cycle", "diff"],
            "predictive": "stationary",
            "dfm": "standardized_stationary_panel",
            "oos_dm": "direct_horizon_change",
            "frequency": "stationary",
        },
        "screen_lags": POSITIVE_LAGS,
        "signed_support_lags": SCREEN_LAGS,
        "reporting_buckets": {
            "short_weeks": [1, 2, 3, 4],
            "medium_weeks": [5, 6, 7, 8],
        },
        "horizons": HORIZONS,
        "significance_alpha": SIGNIFICANCE_ALPHA,
        "stability_rule": {
            "lag_tolerance_weeks": 2,
            "sign_flip_unstable": True,
        },
        "lag_sign_convention": "positive means candidate leads oil target by k weeks; negative means candidate lags oil target by k weeks",
        "blocked_policy": {
            "emit_explicit_blocked_rows": True,
            "silent_substitution_forbidden": True,
        },
        "family_vote_rules": {
            "turning_point": "leading if median lead >1 and matched_turns>=3; coincident if abs(median lead)<=1 and matched_turns>=3; lagging if median lead<-1 and matched_turns>=3; else inconclusive",
            "ccf": "leading if lag in +1..+8 and fdr<=0.10 and abs corr>=0.20; coincident if lag==0 and same significance rule; lagging if lag in -1..-8 and same significance rule; else inconclusive",
            "predictive": "leading if forward_fdr<=0.10 and reverse_fdr>0.10; lagging if reverse_fdr<=0.10 and forward_fdr>0.10; coincident if both significant and lag==1; else inconclusive",
            "dfm": "candidate inherits strongest-loading eligible factor class; else inconclusive",
            "oos_dm": "leading if selected best-horizon row improves rmse and horizon_fdr<=0.10; else inconclusive",
            "frequency": "leading if coherence>=0.30 and phase lead >1; coincident if coherence>=0.30 and abs phase lead<=1; lagging if coherence>=0.30 and phase lead<-1; else inconclusive",
        },
        "synthesis_thresholds": {
            "leading": {"eligible_families": 4, "votes": 3, "predictive_support": True, "max_opposite_votes": 1},
            "coincident": {"eligible_families": 4, "votes": 3},
            "lagging": {"eligible_families": 4, "votes": 3, "max_opposite_votes": 1},
            "mixed": {"eligible_families": 4, "min_non_inconclusive_votes": 2},
            "inconclusive": {"min_significant_votes": 2},
        },
        "verifier_selector_policy": {
            "turning_point": "highest matched_turns then largest abs(median_lead_weeks) then variable asc",
            "ccf": "smallest fdr_pvalue then largest abs(corr_at_best_lag) then variable asc",
            "predictive": "smallest min(forward_fdr_pvalue, reverse_fdr_pvalue) then smallest selected_lag then variable asc",
            "dfm": "largest abs_loading among eligible non-inconclusive rows then factor_id then variable asc",
            "oos_dm": "is_best_horizon=true then smallest horizon_fdr_pvalue then smallest horizon_weeks then variable asc",
            "frequency": "largest coherence_max then largest abs(phase_lead_weeks) then variable asc",
        },
        "oos_dm": {
            "fdr_universe": "within_horizon_across_variables",
            "dm_adjustment": "Harvey-Leybourne-Newbold",
            "newey_west_bandwidth": "h-1",
            "min_train_size": MIN_TRAIN_SIZE,
        },
        "predictive": {
            "fdr_universe": "separate_across_all_variables_for_forward_and_reverse_after_lag_path_selection",
            "max_lag": 8,
            "sample_guard": {
                "min_effective_sample": MIN_EFFECTIVE_SAMPLE,
                "min_ratio_per_lag": 20,
            },
        },
        "bands": BAND_DEFS,
        "hp_lambda": HP_LAMBDA,
        "family_order": FAMILY_ORDER,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bh_adjust(values: pd.Series) -> pd.Series:
    series = values.astype(float).copy()
    mask = series.notna()
    if not mask.any():
        return pd.Series(np.nan, index=series.index, dtype=float)
    ranked = series[mask].sort_values()
    m = len(ranked)
    adjusted = pd.Series(index=ranked.index, dtype=float)
    prev = 1.0
    for rank, (idx, value) in enumerate(reversed(list(ranked.items())), start=1):
        k = m - rank + 1
        candidate = float(value) * m / k
        prev = min(prev, candidate)
        adjusted[idx] = min(prev, 1.0)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[adjusted.index] = adjusted.sort_index()
    return result


def safe_adf_pvalue(series: pd.Series) -> float:
    clean = pd.Series(series).dropna()
    if len(clean) < 25 or clean.nunique() < 3:
        return math.nan
    return float(adfuller(clean, autolag="AIC")[1])


def stationary_representation(series: pd.Series) -> tuple[pd.Series, str]:
    raw = series.dropna().astype(float)
    if safe_adf_pvalue(raw) <= 0.05:
        return raw, "raw"
    diff = series.diff().dropna().astype(float)
    if len(diff) == 0:
        return diff, "none"
    if safe_adf_pvalue(diff) <= 0.05:
        return diff, "diff"
    return pd.Series(dtype=float), "none"


def cycle_representation(series: pd.Series) -> pd.Series:
    clean = pd.Series(series).astype(float)
    if clean.isna().all():
        return clean.fillna(0.0)
    interpolated = clean.interpolate(limit_direction="both")
    cycle, _ = hpfilter(interpolated, lamb=HP_LAMBDA)
    return pd.Series(cycle, index=clean.index, name=f"cycle_{series.name}")


def align_series(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    return pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()


def best_lag_and_corr(candidate: pd.Series, target: pd.Series, lags: list[int]) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for lag in lags:
        if lag > 0:
            pair = pd.DataFrame({"x": candidate.iloc[:-lag], "y": target.iloc[lag:]})
        elif lag < 0:
            pair = pd.DataFrame({"x": candidate.iloc[-lag:], "y": target.iloc[:lag]})
        else:
            pair = pd.DataFrame({"x": candidate, "y": target})
        pair = pair.dropna()
        if len(pair) < 12 or pair["x"].nunique() < 3 or pair["y"].nunique() < 3:
            continue
        corr = float(pair["x"].corr(pair["y"]))
        abs_corr = abs(corr)
        if math.isnan(corr):
            continue
        if best is None or abs_corr > float(best["abs_corr"]):
            best = {
                "best_lag": lag,
                "corr": corr,
                "abs_corr": abs_corr,
                "n_obs": len(pair),
            }
    if best is None:
        return {"best_lag": 0, "corr": 0.0, "abs_corr": 0.0, "n_obs": 0}
    return best


def corr_pvalue(corr: float, n_obs: int) -> float:
    if n_obs <= 3 or abs(corr) >= 1:
        return 0.0 if abs(corr) >= 1 else 1.0
    t_stat = abs(corr) * math.sqrt((n_obs - 2) / max(1e-12, 1 - corr**2))
    return float(2 * stats.t.sf(t_stat, df=n_obs - 2))


def family_stub(variable: str, family: str) -> dict[str, Any]:
    return {
        "variable": variable,
        "family": family,
        "eligible": True,
        "blocked_reason": "",
        "representation": "",
        "support_class": "inconclusive",
        "best_lag_weeks": np.nan,
        "band_label": "",
        "support_strength": 0.0,
        "effect_sign": "neutral",
        "raw_pvalue": np.nan,
        "fdr_pvalue": np.nan,
        "stability_flag": False,
        "source_artifact": "",
        "notes": "",
    }


def turning_points(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    array = pd.Series(values).fillna(0.0).to_numpy(dtype=float)
    peaks, _ = signal.find_peaks(array, distance=4)
    troughs, _ = signal.find_peaks(-array, distance=4)
    return peaks, troughs


def turning_point_summary(candidate_cycle: pd.Series, target_cycle: pd.Series, variable: str) -> dict[str, Any]:
    row = family_stub(variable, "turning_point")
    peaks_t, troughs_t = turning_points(target_cycle)
    peaks_x, troughs_x = turning_points(candidate_cycle)
    target_turns = np.sort(np.concatenate([peaks_t, troughs_t]))
    candidate_turns = np.sort(np.concatenate([peaks_x, troughs_x]))
    if len(target_turns) == 0 or len(candidate_turns) == 0:
        row.update({
            "eligible": False,
            "blocked_reason": "no_turning_points",
            "source_artifact": "tables/family_turning_point.csv",
        })
        return row
    diffs: list[int] = []
    matched = 0
    for turn in target_turns:
        nearest = candidate_turns[np.argmin(np.abs(candidate_turns - turn))]
        delta = int(turn - nearest)
        if abs(delta) <= 12:
            matched += 1
            diffs.append(delta)
    if matched == 0:
        row.update({
            "eligible": False,
            "blocked_reason": "no_matched_turns",
            "source_artifact": "tables/family_turning_point.csv",
        })
        return row
    median_delta = float(np.median(diffs))
    mean_delta = float(np.mean(diffs))
    false_cycles = max(0, len(candidate_turns) - matched)
    missed_cycles = max(0, len(target_turns) - matched)
    support_class = "inconclusive"
    if median_delta > 1 and matched >= 3:
        support_class = "leading"
    elif abs(median_delta) <= 1 and matched >= 3:
        support_class = "coincident"
    elif median_delta < -1 and matched >= 3:
        support_class = "lagging"
    row.update(
        {
            "representation": "cycle",
            "support_class": support_class,
            "best_lag_weeks": median_delta,
            "support_strength": float(matched / max(len(target_turns), 1)),
            "effect_sign": "positive" if mean_delta >= 0 else "negative",
            "raw_pvalue": np.nan,
            "fdr_pvalue": np.nan,
            "matched_turns": matched,
            "mean_lead_weeks": mean_delta,
            "median_lead_weeks": median_delta,
            "false_cycles": false_cycles,
            "missed_cycles": missed_cycles,
            "stability_flag": False,
            "source_artifact": "tables/family_turning_point.csv",
        }
    )
    return row


def ccf_family_rows(
    variables: list[str],
    candidate_cycles: dict[str, pd.Series],
    candidate_stationary: dict[str, pd.Series],
    target_cycle: pd.Series,
    target_stationary: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        for representation, source, target in [
            ("cycle", candidate_cycles[variable], target_cycle),
            ("diff", candidate_stationary[variable], target_stationary),
        ]:
            summary = best_lag_and_corr(source, target, SCREEN_LAGS)
            row = family_stub(variable, "ccf")
            pvalue = corr_pvalue(float(summary["corr"]), int(summary["n_obs"]))
            row.update(
                {
                    "representation": representation,
                    "best_lag_weeks": int(summary["best_lag"]),
                    "support_strength": float(summary["abs_corr"]),
                    "effect_sign": "positive" if float(summary["corr"]) >= 0 else "negative",
                    "raw_pvalue": pvalue,
                    "corr_at_best_lag": float(summary["corr"]),
                    "best_lag_abs_corr": float(summary["abs_corr"]),
                    "source_artifact": "tables/family_ccf.csv",
                }
            )
            if int(summary["n_obs"]) == 0:
                row.update(
                    {
                        "eligible": False,
                        "blocked_reason": "no_valid_alignment",
                        "support_strength": 0.0,
                        "effect_sign": "neutral",
                    }
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["fdr_pvalue"] = bh_adjust(frame["raw_pvalue"])
    support_class: list[str] = []
    for row in frame.itertuples(index=False):
        label = "inconclusive"
        if row.fdr_pvalue <= SIGNIFICANCE_ALPHA and abs(row.corr_at_best_lag) >= CCF_CORR_THRESHOLD:
            if 1 <= row.best_lag_weeks <= 8:
                label = "leading"
            elif row.best_lag_weeks == 0:
                label = "coincident"
            elif -8 <= row.best_lag_weeks <= -1:
                label = "lagging"
        support_class.append(label)
    frame["support_class"] = support_class
    frame["stability_flag"] = False
    for variable, subset in frame.groupby("variable"):
        eligible_subset = subset[subset["eligible"].astype(bool)]
        sign_values = {value for value in eligible_subset["effect_sign"].tolist() if value != "neutral"}
        lag_values = [float(value) for value in eligible_subset["best_lag_weeks"].dropna().tolist()]
        lag_span = (max(lag_values) - min(lag_values)) if lag_values else 0.0
        unstable = len(sign_values) > 1 or lag_span > 2
        frame.loc[frame["variable"] == variable, "stability_flag"] = unstable
    return frame


def select_var_lag(data: pd.DataFrame, maxlags: int = 8) -> tuple[int | None, str | None]:
    safe_max = min(maxlags, max(1, len(data) // 10))
    if safe_max < 1:
        return None, "insufficient_rows_for_lag_selection"
    try:
        order = VAR(data).select_order(safe_max)
    except Exception as exc:
        return None, f"lag_selection_failed:{type(exc).__name__}"
    bic = getattr(order, "selected_orders", {}).get("bic")
    if bic is None or bic < 1:
        return None, "lag_selection_undefined"
    return int(bic), None


def ecm_directional_pvalue(target_level: pd.Series, candidate_level: pd.Series, lag: int) -> tuple[float, float, str]:
    pair = align_series(target_level, candidate_level)
    if len(pair) < MIN_EFFECTIVE_SAMPLE:
        return math.nan, math.nan, "insufficient_sample"
    levels = pair.rename(columns={"left": "y", "right": "x"})
    regression = LinearRegression().fit(levels[["x"]], levels["y"])
    ec = levels["y"] - regression.predict(levels[["x"]])
    dy = levels["y"].diff()
    dx = levels["x"].diff()

    def build(direction: str) -> pd.DataFrame:
        lhs = dy if direction == "forward" else dx
        own = dy if direction == "forward" else dx
        other = dx if direction == "forward" else dy
        design = pd.DataFrame({"lhs": lhs, "ec_lag1": ec.shift(1)})
        for step in range(1, lag + 1):
            design[f"own_lag_{step}"] = own.shift(step)
            design[f"other_lag_{step}"] = other.shift(step)
        return design.dropna()

    def test(direction: str) -> float:
        frame = build(direction)
        if len(frame) < MIN_EFFECTIVE_SAMPLE:
            return math.nan
        y = frame["lhs"]
        X = add_constant(frame.drop(columns=["lhs"]))
        model = OLS(y, X).fit()
        other_cols = [column for column in X.columns if column.startswith("other_lag_")]
        if not other_cols:
            return math.nan
        restriction = " = 0, ".join(other_cols) + " = 0"
        return float(model.f_test(restriction).pvalue)

    return test("forward"), test("reverse"), "ecm"


def predictive_family(
    variables: list[str],
    numeric: pd.DataFrame,
    primary: pd.Series,
    target_stationary: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        row = family_stub(variable, "predictive")
        candidate_raw = numeric[variable]
        candidate_stationary, candidate_repr = stationary_representation(candidate_raw)
        aligned_stationary = pd.concat(
            [target_stationary.rename("y"), candidate_stationary.rename("x")], axis=1
        ).dropna()
        if (
            candidate_repr == "none"
            or len(aligned_stationary) < MIN_EFFECTIVE_SAMPLE
            or aligned_stationary["x"].nunique() < 3
            or aligned_stationary["y"].nunique() < 3
        ):
            row.update(
                {
                    "eligible": False,
                    "blocked_reason": "no_stationary_representation" if candidate_repr == "none" else "insufficient_sample_or_variation",
                    "selected_model": "none",
                    "directional_effect": "none",
                    "generic_pvalue_rule": "min_directional_pvalue",
                    "source_artifact": "tables/family_predictive.csv",
                }
            )
            rows.append(row)
            continue

        lag, lag_reason = select_var_lag(aligned_stationary[["y", "x"]])
        if lag is None:
            row.update(
                {
                    "eligible": False,
                    "blocked_reason": lag_reason or "lag_selection_failed",
                    "selected_model": "none",
                    "directional_effect": "none",
                    "generic_pvalue_rule": "min_directional_pvalue",
                    "source_artifact": "tables/family_predictive.csv",
                }
            )
            rows.append(row)
            continue
        effective_required = max(MIN_EFFECTIVE_SAMPLE, 20 * lag)
        if len(aligned_stationary) < effective_required:
            row.update(
                {
                    "eligible": False,
                    "blocked_reason": "insufficient_effective_sample_for_selected_lag",
                    "selected_model": "none",
                    "directional_effect": "none",
                    "generic_pvalue_rule": "min_directional_pvalue",
                    "source_artifact": "tables/family_predictive.csv",
                }
            )
            rows.append(row)
            continue
        forward_raw = math.nan
        reverse_raw = math.nan
        path_used = "var_granger"
        target_adf = safe_adf_pvalue(primary)
        candidate_adf = safe_adf_pvalue(candidate_raw)
        if pd.isna(target_adf) or pd.isna(candidate_adf):
            row.update(
                {
                    "eligible": False,
                    "blocked_reason": "adf_infeasible",
                    "selected_model": "none",
                    "directional_effect": "none",
                    "generic_pvalue_rule": "min_directional_pvalue",
                    "source_artifact": "tables/family_predictive.csv",
                }
            )
            rows.append(row)
            continue
        raw_target_nonstationary = target_adf > 0.05
        raw_candidate_nonstationary = candidate_adf > 0.05
        if raw_target_nonstationary and raw_candidate_nonstationary:
            try:
                johansen = coint_johansen(
                    pd.concat([primary.rename("y"), candidate_raw.rename("x")], axis=1).dropna(),
                    det_order=0,
                    k_ar_diff=max(1, lag - 1),
                )
                has_cointegration = float(johansen.lr1[0]) > float(johansen.cvt[0, 1])
            except Exception:
                row.update(
                    {
                        "eligible": False,
                        "blocked_reason": "cointegration_test_infeasible",
                        "selected_model": "none",
                        "directional_effect": "none",
                        "generic_pvalue_rule": "min_directional_pvalue",
                        "source_artifact": "tables/family_predictive.csv",
                    }
                )
                rows.append(row)
                continue
            if has_cointegration:
                try:
                    forward_raw, reverse_raw, path_used = ecm_directional_pvalue(primary, candidate_raw, lag)
                except Exception:
                    row.update(
                        {
                            "eligible": False,
                            "blocked_reason": "ecm_path_infeasible",
                            "selected_model": "none",
                            "directional_effect": "none",
                            "generic_pvalue_rule": "min_directional_pvalue",
                            "source_artifact": "tables/family_predictive.csv",
                        }
                    )
                    rows.append(row)
                    continue
            else:
                try:
                    forward_raw = float(
                        grangercausalitytests(
                            aligned_stationary[["y", "x"]], maxlag=[lag], verbose=False
                        )[lag][0]["ssr_ftest"][1]
                    )
                    reverse_raw = float(
                        grangercausalitytests(
                            aligned_stationary[["x", "y"]], maxlag=[lag], verbose=False
                        )[lag][0]["ssr_ftest"][1]
                    )
                    path_used = "var_granger"
                except Exception:
                    row.update(
                        {
                            "eligible": False,
                            "blocked_reason": "var_granger_infeasible",
                            "selected_model": "none",
                            "directional_effect": "none",
                            "generic_pvalue_rule": "min_directional_pvalue",
                            "source_artifact": "tables/family_predictive.csv",
                        }
                    )
                    rows.append(row)
                    continue
        else:
            try:
                forward_raw = float(
                    grangercausalitytests(
                        aligned_stationary[["y", "x"]], maxlag=[lag], verbose=False
                    )[lag][0]["ssr_ftest"][1]
                )
                reverse_raw = float(
                    grangercausalitytests(
                        aligned_stationary[["x", "y"]], maxlag=[lag], verbose=False
                    )[lag][0]["ssr_ftest"][1]
                )
                path_used = "var_granger"
            except Exception:
                row.update(
                    {
                        "eligible": False,
                        "blocked_reason": "var_granger_infeasible",
                        "selected_model": "none",
                        "directional_effect": "none",
                        "generic_pvalue_rule": "min_directional_pvalue",
                        "source_artifact": "tables/family_predictive.csv",
                    }
                )
                rows.append(row)
                continue

        row.update(
            {
                "representation": "stationary",
                "best_lag_weeks": lag,
                "selected_lag": lag,
                "path_used": path_used,
                "forward_raw_pvalue": forward_raw,
                "reverse_raw_pvalue": reverse_raw,
                "selected_model": path_used,
                "directional_effect": "none",
                "generic_pvalue_rule": "min_directional_pvalue",
                "source_artifact": "tables/family_predictive.csv",
            }
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    for column in [
        "forward_raw_pvalue",
        "forward_fdr_pvalue",
        "reverse_raw_pvalue",
        "reverse_fdr_pvalue",
        "selected_lag",
        "path_used",
        "selected_model",
        "directional_effect",
        "generic_pvalue_rule",
    ]:
        if column not in frame.columns:
            frame[column] = np.nan
    eligible = frame["eligible"].astype(bool)
    frame.loc[eligible, "forward_fdr_pvalue"] = bh_adjust(frame.loc[eligible, "forward_raw_pvalue"])
    frame.loc[eligible, "reverse_fdr_pvalue"] = bh_adjust(frame.loc[eligible, "reverse_raw_pvalue"])
    frame["raw_pvalue"] = frame[["forward_raw_pvalue", "reverse_raw_pvalue"]].min(axis=1)
    frame["fdr_pvalue"] = frame[["forward_fdr_pvalue", "reverse_fdr_pvalue"]].min(axis=1)
    support_class: list[str] = []
    directional_effect: list[str] = []
    support_strength: list[float] = []
    effect_sign: list[str] = []
    for row in frame.itertuples(index=False):
        label = "inconclusive"
        effect = "none"
        if bool(row.eligible):
            if row.forward_fdr_pvalue <= SIGNIFICANCE_ALPHA and row.reverse_fdr_pvalue > SIGNIFICANCE_ALPHA:
                label = "leading"
                effect = "forward"
            elif row.reverse_fdr_pvalue <= SIGNIFICANCE_ALPHA and row.forward_fdr_pvalue > SIGNIFICANCE_ALPHA:
                label = "lagging"
                effect = "reverse"
            elif row.forward_fdr_pvalue <= SIGNIFICANCE_ALPHA and row.reverse_fdr_pvalue <= SIGNIFICANCE_ALPHA and row.selected_lag == 1:
                label = "coincident"
                effect = "bidirectional"
        support_class.append(label)
        directional_effect.append(effect)
        support_strength.append(float(1 - min([value for value in [row.fdr_pvalue] if not pd.isna(value)] or [1.0])))
        effect_sign.append("positive" if label in {"leading", "coincident"} else ("negative" if label == "lagging" else "neutral"))
    frame["support_class"] = support_class
    frame["directional_effect"] = directional_effect
    frame["support_strength"] = support_strength
    frame["effect_sign"] = effect_sign
    frame["stability_flag"] = False
    return frame


def dfm_family(
    variables: list[str],
    numeric: pd.DataFrame,
    target_stationary: pd.Series,
) -> pd.DataFrame:
    stationary_panel: dict[str, pd.Series] = {}
    missing_stationary: set[str] = set()
    for variable in variables:
        series, _ = stationary_representation(numeric[variable])
        if len(series) == 0:
            missing_stationary.add(variable)
            continue
        stationary_panel[variable] = series
    panel = pd.DataFrame(stationary_panel).dropna(axis=0, how="any")
    if len(panel) < MIN_EFFECTIVE_SAMPLE or panel.shape[1] < 2:
        rows = []
        for variable in variables:
            row = family_stub(variable, "dfm")
            row.update(
                {
                    "eligible": False,
                    "blocked_reason": "no_stationary_representation" if variable in missing_stationary else "insufficient_panel",
                    "factor_count": 0,
                    "source_artifact": "tables/family_dfm.csv",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)
    standardized = (panel - panel.mean()) / panel.std(ddof=0).replace(0, 1.0)
    pca = PCA(n_components=min(MAX_FACTOR_COUNT, standardized.shape[1]))
    scores = pca.fit_transform(standardized)
    explained = np.cumsum(pca.explained_variance_ratio_)
    factor_count = int(min(MAX_FACTOR_COUNT, np.searchsorted(explained, 0.60) + 1))
    loads = pd.DataFrame(
        pca.components_[:factor_count].T,
        index=standardized.columns,
        columns=[f"F{i+1}" for i in range(factor_count)],
    )
    target_aligned = target_stationary.loc[standardized.index]
    factor_rows: dict[str, dict[str, Any]] = {}
    for factor_name in loads.columns:
        factor_series = pd.Series(scores[:, int(factor_name[1:]) - 1], index=standardized.index)
        best = best_lag_and_corr(factor_series, target_aligned, SCREEN_LAGS)
        pvalue = corr_pvalue(float(best["corr"]), int(best["n_obs"]))
        label = "inconclusive"
        if abs(float(best["corr"])) >= CCF_CORR_THRESHOLD:
            if 1 <= int(best["best_lag"]) <= 8:
                label = "leading"
            elif int(best["best_lag"]) == 0:
                label = "coincident"
            elif -8 <= int(best["best_lag"]) <= -1:
                label = "lagging"
        factor_rows[factor_name] = {
            "factor_class": label,
            "best_lag_weeks": int(best["best_lag"]),
            "raw_pvalue": pvalue,
            "support_strength": abs(float(best["corr"])),
        }
    factor_p = pd.Series({k: v["raw_pvalue"] for k, v in factor_rows.items()})
    factor_fdr = bh_adjust(factor_p)
    rows: list[dict[str, Any]] = []
    for variable in variables:
        row = family_stub(variable, "dfm")
        if variable in missing_stationary:
            row.update(
                {
                    "eligible": False,
                    "blocked_reason": "no_stationary_representation",
                    "factor_count": factor_count,
                    "source_artifact": "tables/family_dfm.csv",
                }
            )
            rows.append(row)
            continue
        if variable not in loads.index:
            row.update(
                {
                    "eligible": False,
                    "blocked_reason": "variable_missing_after_panel_align",
                    "source_artifact": "tables/family_dfm.csv",
                }
            )
            rows.append(row)
            continue
        loadings = loads.loc[variable].abs()
        strongest = str(loadings.idxmax())
        threshold = float(loads[strongest].abs().quantile(TOP_LOADING_QUANTILE))
        if float(loadings[strongest]) < threshold:
            row.update(
                {
                    "representation": "stationary_panel",
                    "factor_id": strongest,
                    "factor_count": factor_count,
                    "abs_loading": float(loadings[strongest]),
                    "factor_explained_variance": float(pca.explained_variance_ratio_[int(strongest[1:]) - 1]),
                    "factor_class": "inconclusive",
                    "notes": f"factor_count={factor_count}",
                    "source_artifact": "tables/family_dfm.csv",
                }
            )
            rows.append(row)
            continue
        factor_meta = factor_rows[strongest]
        row.update(
            {
                "representation": "stationary_panel",
                "support_class": factor_meta["factor_class"],
                "best_lag_weeks": factor_meta["best_lag_weeks"],
                "support_strength": float(loadings[strongest]),
                "effect_sign": "positive" if factor_meta["factor_class"] in {"leading", "coincident"} else ("negative" if factor_meta["factor_class"] == "lagging" else "neutral"),
                "raw_pvalue": factor_meta["raw_pvalue"],
                "fdr_pvalue": float(factor_fdr[strongest]),
                "factor_id": strongest,
                "factor_count": factor_count,
                "abs_loading": float(loadings[strongest]),
                "factor_explained_variance": float(pca.explained_variance_ratio_[int(strongest[1:]) - 1]),
                "factor_class": factor_meta["factor_class"],
                "stability_flag": False,
                "notes": f"factor_count={factor_count}",
                "source_artifact": "tables/family_dfm.csv",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_horizon_target(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-horizon) - series


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def dm_test(errors_a: np.ndarray, errors_b: np.ndarray, horizon: int) -> tuple[float, float, float]:
    d = np.square(errors_a) - np.square(errors_b)
    n = len(d)
    if n < 10:
        return math.nan, math.nan, math.nan
    mean_d = float(np.mean(d))
    demeaned = d - mean_d
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    variance = gamma0
    bandwidth = max(horizon - 1, 0)
    for lag in range(1, bandwidth + 1):
        cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        weight = 1 - lag / (bandwidth + 1)
        variance += 2 * weight * cov
    variance = max(variance / n, 1e-12)
    dm_stat = mean_d / math.sqrt(variance)
    hln = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    adjusted = dm_stat * hln
    pvalue = float(2 * stats.t.sf(abs(adjusted), df=n - 1))
    return float(dm_stat), float(adjusted), pvalue


def direct_ols_forecast(train: pd.DataFrame, test_row: pd.Series, features: list[str]) -> float:
    model = LinearRegression()
    model.fit(train[features], train["target"])
    return float(model.predict(test_row[features].to_frame().T)[0])


def oos_family(
    variables: list[str],
    primary_target: pd.Series,
    candidate_stationary: dict[str, pd.Series],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        stationary = candidate_stationary[variable]
        for horizon in HORIZONS:
            row = family_stub(variable, "oos_dm")
            row.update(
                {
                    "row_grain": "variable_horizon",
                    "horizon_weeks": horizon,
                    "representation": "target_change",
                    "best_lag_weeks": horizon,
                    "source_artifact": "tables/family_oos_dm.csv",
                }
            )
            target_h = build_horizon_target(primary_target, horizon)
            design = pd.DataFrame({"target": target_h})
            for lag in POSITIVE_LAGS:
                design[f"target_lag_{lag}"] = primary_target.shift(lag)
                design[f"cand_lag_{lag}"] = stationary.shift(lag)
            design = design.dropna().reset_index(drop=True)
            if len(design) <= MIN_TRAIN_SIZE + 5:
                row.update({"eligible": False, "blocked_reason": "insufficient_sample"})
                rows.append(row)
                continue
            baseline_features = [f"target_lag_{lag}" for lag in POSITIVE_LAGS]
            candidate_features = baseline_features + [f"cand_lag_{lag}" for lag in POSITIVE_LAGS]
            baseline_errors: list[float] = []
            candidate_errors: list[float] = []
            for idx in range(MIN_TRAIN_SIZE, len(design)):
                train = design.iloc[:idx]
                test_row = design.iloc[idx]
                truth = float(test_row["target"])
                baseline_pred = direct_ols_forecast(train, test_row, baseline_features)
                candidate_pred = direct_ols_forecast(train, test_row, candidate_features)
                baseline_errors.append(truth - baseline_pred)
                candidate_errors.append(truth - candidate_pred)
            baseline_arr = np.asarray(baseline_errors)
            candidate_arr = np.asarray(candidate_errors)
            dm_stat, dm_adj, pvalue = dm_test(baseline_arr, candidate_arr, horizon)
            row.update(
                {
                    "baseline_rmse": rmse(baseline_arr),
                    "candidate_rmse": rmse(candidate_arr),
                    "dm_stat": dm_stat,
                    "dm_hln_adjusted": dm_adj,
                    "raw_pvalue": pvalue,
                    "support_strength": float(max(0.0, rmse(baseline_arr) - rmse(candidate_arr))),
                    "effect_sign": "positive" if rmse(candidate_arr) < rmse(baseline_arr) else "neutral",
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    for column in [
        "baseline_rmse",
        "candidate_rmse",
        "dm_stat",
        "dm_hln_adjusted",
        "horizon_fdr_pvalue",
        "is_best_horizon",
        "row_grain",
        "horizon_weeks",
    ]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["horizon_fdr_pvalue"] = np.nan
    for horizon in HORIZONS:
        mask = frame["horizon_weeks"].eq(horizon) & frame["eligible"].astype(bool)
        if mask.any():
            frame.loc[mask, "horizon_fdr_pvalue"] = bh_adjust(frame.loc[mask, "raw_pvalue"])
    frame["is_best_horizon"] = False
    for variable, subset in frame.groupby("variable"):
        eligible = subset[subset["eligible"].astype(bool)]
        if eligible.empty:
            continue
        improved = eligible[eligible["candidate_rmse"] < eligible["baseline_rmse"]]
        chosen = improved if not improved.empty else eligible
        winner = chosen.sort_values(["horizon_fdr_pvalue", "horizon_weeks"], ascending=[True, True]).iloc[0]
        frame.loc[(frame["variable"] == variable) & (frame["horizon_weeks"] == winner["horizon_weeks"]), "is_best_horizon"] = True
    frame["support_class"] = np.where(
        frame["is_best_horizon"]
        & frame["eligible"].astype(bool)
        & (frame["candidate_rmse"] < frame["baseline_rmse"])
        & (frame["horizon_fdr_pvalue"] <= SIGNIFICANCE_ALPHA),
        "leading",
        "inconclusive",
    )
    frame["fdr_pvalue"] = frame["horizon_fdr_pvalue"]
    frame["stability_flag"] = False
    for variable, subset in frame.groupby("variable"):
        improving = subset[
            subset["eligible"].astype(bool)
            & (subset["candidate_rmse"] < subset["baseline_rmse"])
            & (subset["horizon_fdr_pvalue"] <= SIGNIFICANCE_ALPHA)
        ]
        unstable = bool(len(improving) > 1 and (improving["horizon_weeks"].max() - improving["horizon_weeks"].min()) > 2)
        frame.loc[frame["variable"] == variable, "stability_flag"] = unstable
    return frame


def frequency_family(
    variables: list[str],
    candidate_stationary: dict[str, pd.Series],
    target_stationary: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        row = family_stub(variable, "frequency")
        pair = pd.concat([candidate_stationary[variable].rename("x"), target_stationary.rename("y")], axis=1).dropna()
        if len(pair) < 64:
            row.update(
                {
                    "eligible": False,
                    "blocked_reason": "insufficient_frequency_sample",
                    "source_artifact": "tables/family_frequency.csv",
                }
            )
            rows.append(row)
            continue
        x = pair["x"].to_numpy(dtype=float)
        y = pair["y"].to_numpy(dtype=float)
        nperseg = min(128, len(pair))
        freqs, coh = signal.coherence(x, y, fs=1.0, nperseg=nperseg)
        _, csd_vals = signal.csd(x, y, fs=1.0, nperseg=nperseg)
        phase = np.angle(csd_vals)
        best_band = ""
        best_coh = -1.0
        best_phase_weeks = math.nan
        for label, (low, high) in BAND_DEFS.items():
            band_mask = (freqs >= low) & (freqs < high)
            if not band_mask.any():
                continue
            band_index = np.argmax(coh[band_mask])
            band_freqs = freqs[band_mask]
            band_coh = coh[band_mask]
            band_phase = phase[band_mask]
            current_coh = float(band_coh[band_index])
            current_freq = float(band_freqs[band_index])
            current_phase = float(band_phase[band_index])
            phase_weeks = current_phase / (2 * math.pi * current_freq) if current_freq > 0 else math.nan
            if current_coh > best_coh:
                best_band = label
                best_coh = current_coh
                best_phase_weeks = phase_weeks
        raw_p = float(max(0.0, 1.0 - best_coh)) if best_coh >= 0 else 1.0
        support_class = "inconclusive"
        if best_coh >= COHERENCE_THRESHOLD:
            if best_phase_weeks > 1:
                support_class = "leading"
            elif best_phase_weeks < -1:
                support_class = "lagging"
            elif abs(best_phase_weeks) <= 1:
                support_class = "coincident"
        row.update(
            {
                "representation": "stationary",
                "support_class": support_class,
                "best_lag_weeks": float(best_phase_weeks) if not math.isnan(best_phase_weeks) else np.nan,
                "band_label": best_band,
                "support_strength": float(max(best_coh, 0.0)),
                "effect_sign": "positive" if support_class in {"leading", "coincident"} else ("negative" if support_class == "lagging" else "neutral"),
                "raw_pvalue": raw_p,
                "dominant_band": best_band,
                "coherence_max": float(max(best_coh, 0.0)),
                "phase_lead_weeks": float(best_phase_weeks) if not math.isnan(best_phase_weeks) else np.nan,
                "source_artifact": "tables/family_frequency.csv",
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["fdr_pvalue"] = bh_adjust(frame["raw_pvalue"])
    frame["stability_flag"] = False
    return frame


def family_vote(frame: pd.DataFrame, variable: str, family: str) -> str:
    subset = frame[frame["variable"] == variable]
    if subset.empty:
        return "inconclusive"
    selected = select_representative_row(subset, family)
    if selected is None:
        return "inconclusive"
    return str(selected["support_class"])


def synthesis_table(family_frames: dict[str, pd.DataFrame], variables: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in variables:
        votes = {family: family_vote(frame, variable, family) for family, frame in family_frames.items()}
        selected_rows: dict[str, pd.Series] = {}
        eligible_family_count = 0
        for family, frame in family_frames.items():
            subset = frame[frame["variable"] == variable]
            if family == "oos_dm":
                subset = subset[subset["is_best_horizon"].astype(bool)] if not subset.empty else subset
            if subset.empty:
                continue
            selected = select_representative_row(subset, family)
            if selected is None:
                continue
            selected_rows[family] = selected
            if bool(selected.get("eligible", True)):
                eligible_family_count += 1
        leading_votes = sum(v == "leading" for v in votes.values())
        coincident_votes = sum(v == "coincident" for v in votes.values())
        lagging_votes = sum(v == "lagging" for v in votes.values())
        inconclusive_votes = sum(v == "inconclusive" for v in votes.values())
        contradiction_count = int(leading_votes > 0 and lagging_votes > 0)
        predictive_support = votes["predictive"] == "leading" or votes["oos_dm"] == "leading"
        stability_flag = contradiction_count > 0 or any(bool(row.get("stability_flag", False)) for row in selected_rows.values())
        predictive_leading_fdr = min(
            [
                float(selected_rows[family].get("fdr_pvalue", np.nan))
                for family in ["predictive", "oos_dm"]
                if family in selected_rows and votes.get(family) == "leading" and not pd.isna(selected_rows[family].get("fdr_pvalue", np.nan))
            ]
            or [np.nan]
        )
        coincident_fdr = min(
            [
                float(selected_rows[family].get("fdr_pvalue", np.nan))
                for family in selected_rows
                if votes.get(family) == "coincident" and not pd.isna(selected_rows[family].get("fdr_pvalue", np.nan))
            ]
            or [np.nan]
        )
        predictive_stronger_than_coincident = (
            not pd.isna(predictive_leading_fdr)
            and not pd.isna(coincident_fdr)
            and predictive_leading_fdr < coincident_fdr
        )
        final_class = "inconclusive"
        final_reason_code = "insufficient_evidence"
        if eligible_family_count >= 4:
            if leading_votes >= 3 and predictive_support and lagging_votes <= CONTRADICTION_TOLERANCE:
                final_class = "leading"
                final_reason_code = "leading_threshold_met"
            elif coincident_votes >= 3 and not predictive_stronger_than_coincident:
                final_class = "coincident"
                final_reason_code = "coincident_threshold_met"
            elif lagging_votes >= 3 and leading_votes <= CONTRADICTION_TOLERANCE:
                final_class = "lagging"
                final_reason_code = "lagging_threshold_met"
            elif leading_votes + coincident_votes + lagging_votes >= 2:
                final_class = "mixed"
                final_reason_code = "mixed_conflicting_votes"
        rows.append(
            {
                "variable": variable,
                "eligible_family_count": eligible_family_count,
                "leading_votes": leading_votes,
                "coincident_votes": coincident_votes,
                "lagging_votes": lagging_votes,
                "inconclusive_votes": inconclusive_votes,
                "predictive_support": bool(predictive_support),
                "contradiction_count": contradiction_count,
                "stability_flag": stability_flag,
                "turning_point_class": votes.get("turning_point", "inconclusive"),
                "turning_point_best_lag_weeks": selected_rows.get("turning_point", {}).get("best_lag_weeks", np.nan) if "turning_point" in selected_rows else np.nan,
                "ccf_class": votes.get("ccf", "inconclusive"),
                "ccf_best_lag_weeks": selected_rows.get("ccf", {}).get("best_lag_weeks", np.nan) if "ccf" in selected_rows else np.nan,
                "predictive_class": votes.get("predictive", "inconclusive"),
                "predictive_best_lag_weeks": selected_rows.get("predictive", {}).get("best_lag_weeks", np.nan) if "predictive" in selected_rows else np.nan,
                "dfm_class": votes.get("dfm", "inconclusive"),
                "dfm_best_lag_weeks": selected_rows.get("dfm", {}).get("best_lag_weeks", np.nan) if "dfm" in selected_rows else np.nan,
                "oos_dm_class": votes.get("oos_dm", "inconclusive"),
                "oos_dm_best_lag_weeks": selected_rows.get("oos_dm", {}).get("best_lag_weeks", np.nan) if "oos_dm" in selected_rows else np.nan,
                "frequency_class": votes.get("frequency", "inconclusive"),
                "frequency_best_lag_weeks": selected_rows.get("frequency", {}).get("best_lag_weeks", np.nan) if "frequency" in selected_rows else np.nan,
                "frequency_band_label": selected_rows.get("frequency", {}).get("band_label", "") if "frequency" in selected_rows else "",
                "final_class": final_class,
                "final_reason_code": final_reason_code,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["final_class", "leading_votes", "coincident_votes", "lagging_votes", "variable"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)


def top_candidates(synthesis: pd.DataFrame, label: str, limit: int = 10) -> pd.DataFrame:
    return synthesis[synthesis["final_class"] == label].head(limit)


def write_family_csv(frame: pd.DataFrame, family: str, tables_dir: Path) -> Path:
    required = REQUIRED_FAMILY_COLUMNS + FAMILY_EXTRA_COLUMNS[family]
    for column in required:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[required]
    path = tables_dir / f"family_{family}.csv"
    frame.to_csv(path, index=False)
    return path


def write_synthesis_csv(frame: pd.DataFrame, tables_dir: Path) -> Path:
    for column in REQUIRED_SYNTHESIS_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    path = tables_dir / "synthesis.csv"
    frame[REQUIRED_SYNTHESIS_COLUMNS].to_csv(path, index=False)
    return path


def plot_family_strength(frame: pd.DataFrame, family: str, figure_path: Path) -> None:
    if frame.empty:
        return
    subset = frame.copy()
    if family == "oos_dm":
        subset = subset[subset["is_best_horizon"].astype(bool)]
    subset = subset.sort_values(["support_strength", "variable"], ascending=[False, True]).head(20)
    if subset.empty:
        return
    colors = subset["support_class"].map(
        {
            "leading": "#4c78a8",
            "coincident": "#72b7b2",
            "lagging": "#e45756",
            "inconclusive": "#bab0ab",
        }
    ).fillna("#bab0ab")
    fig_height = max(6, 0.35 * len(subset))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(subset["variable"], subset["support_strength"], color=colors)
    ax.set_title(f"Top support strengths | {family}")
    ax.set_xlabel("support_strength")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_synthesis_counts(synthesis: pd.DataFrame, figure_path: Path) -> None:
    counts = synthesis["final_class"].value_counts().reindex(
        ["leading", "coincident", "lagging", "mixed", "inconclusive"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=["#4c78a8", "#72b7b2", "#e45756", "#f58518", "#bab0ab"])
    ax.set_title("Final classification counts")
    ax.set_ylabel("variables")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def build_summary(context: AnalysisContext, synthesis: pd.DataFrame, family_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_dir": str(context.output_dir),
        "input_path": str(context.input_path),
        "artifact_dirs": {
            "contract": str(context.contract_dir),
            "tables": str(context.tables_dir),
            "figures": str(context.figures_dir),
        },
        "family_counts": {
            family: int(len(frame if family != "oos_dm" else frame[frame["is_best_horizon"].astype(bool)]))
            for family, frame in family_frames.items()
        },
        "final_class_counts": synthesis["final_class"].value_counts().to_dict(),
        "top_leading": top_candidates(synthesis, "leading")["variable"].tolist(),
        "top_coincident": top_candidates(synthesis, "coincident")["variable"].tolist(),
        "top_lagging": top_candidates(synthesis, "lagging")["variable"].tolist(),
    }


def build_report(context: AnalysisContext, audit: dict[str, Any], synthesis: pd.DataFrame, family_frames: dict[str, pd.DataFrame]) -> None:
    lines: list[str] = []
    lines.append("# Oil Leading / Coincident / Lagging Indicator Analysis")
    lines.append("")
    lines.append("## Dataset Contract")
    lines.append(f"- input: `{context.input_path}`")
    lines.append(f"- rows: {audit['rows']}")
    lines.append(f"- columns: {audit['columns']}")
    lines.append(f"- numeric columns: {audit['numeric_columns']}")
    lines.append(f"- predictors: {audit['predictor_count']}")
    lines.append(f"- all Monday: {audit['all_monday']}")
    lines.append(f"- cadence days: {audit['day_diffs']}")
    lines.append("")
    lines.append("## Final Class Counts")
    counts = synthesis["final_class"].value_counts().to_dict()
    for label in ["leading", "coincident", "lagging", "mixed", "inconclusive"]:
        lines.append(f"- {label}: {counts.get(label, 0)}")
    lines.append("")
    for label in ["leading", "coincident", "lagging", "mixed", "inconclusive"]:
        subset = synthesis[synthesis["final_class"] == label].head(10)
        if subset.empty:
            continue
        lines.append(f"## Top {label}")
        for row in subset.itertuples(index=False):
            lines.append(
                f"- `{row.variable}` | votes L/C/G/I = {row.leading_votes}/{row.coincident_votes}/{row.lagging_votes}/{row.inconclusive_votes} | predictive_support={row.predictive_support}"
            )
        lines.append("")
    lines.append("## Family Artifact Inventory")
    for family in FAMILY_ORDER:
        lines.append(f"- `{family}`: `tables/family_{family}.csv` and `figures/{family}_top_support.png`")
    lines.append("- synthesis: `tables/synthesis.csv`")
    lines.append("- report: `report.md`")
    lines.append("- summary: `summary.json`")
    lines.append("")
    lines.append("## Caveats")
    unstable = synthesis[synthesis["stability_flag"]].head(10)
    if unstable.empty:
        lines.append("- No stability flags in final synthesis.")
    else:
        lines.append("- unstable candidates: " + ", ".join(f"`{value}`" for value in unstable["variable"].tolist()))
    contradictions = synthesis[synthesis["contradiction_count"] > 0].head(10)
    if contradictions.empty:
        lines.append("- contradictory cases: none")
    else:
        lines.append("- contradictory cases: " + ", ".join(f"`{value}`" for value in contradictions["variable"].tolist()))
    blocked: list[str] = []
    for family, frame in family_frames.items():
        blocked_count = int((~frame["eligible"].astype(bool)).sum())
        blocked.append(f"{family}={blocked_count}")
    lines.append("- blocked rows by family: " + ", ".join(blocked))
    lines.append("")
    lines.append("## Blocked Family Rationale")
    for family, frame in family_frames.items():
        blocked_subset = frame[~frame["eligible"].astype(bool)]
        if blocked_subset.empty:
            lines.append(f"- `{family}`: none")
            continue
        reasons = (
            blocked_subset["blocked_reason"]
            .fillna("unknown")
            .replace("", "unknown")
            .value_counts()
            .to_dict()
        )
        reason_line = ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items()))
        lines.append(f"- `{family}`: {reason_line}")
    lines.append("")
    (context.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selector_for_family(family: str) -> FamilySelector:
    selectors = {
        "turning_point": FamilySelector(family, ["matched_turns", "median_lead_abs", "variable"], [False, False, True]),
        "ccf": FamilySelector(family, ["fdr_pvalue", "best_lag_abs_corr", "variable"], [True, False, True]),
        "predictive": FamilySelector(family, ["fdr_pvalue", "selected_lag", "variable"], [True, True, True]),
        "dfm": FamilySelector(family, ["abs_loading", "factor_id", "variable"], [False, True, True]),
        "oos_dm": FamilySelector(family, ["horizon_fdr_pvalue", "horizon_weeks", "variable"], [True, True, True]),
        "frequency": FamilySelector(family, ["coherence_max", "phase_lead_abs", "variable"], [False, False, True]),
    }
    return selectors[family]


def select_representative_row(frame: pd.DataFrame, family: str) -> pd.Series | None:
    working = frame.copy()
    if family == "oos_dm":
        working = working[working["is_best_horizon"].astype(bool)]
    working = working[working["eligible"].astype(bool)]
    if family == "dfm":
        working = working[working["support_class"] != "inconclusive"]
    if working.empty:
        return None
    if "median_lead_weeks" in working.columns:
        working["median_lead_abs"] = working["median_lead_weeks"].abs()
    if "phase_lead_weeks" in working.columns:
        working["phase_lead_abs"] = working["phase_lead_weeks"].abs()
    selector = selector_for_family(family)
    return working.sort_values(selector.sort_columns, ascending=selector.ascending).iloc[0]
