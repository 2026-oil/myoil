from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ValueWarning)
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests

from scripts.oil_leading_indicators_lib import (
    BAND_DEFS,
    FAMILY_EXTRA_COLUMNS,
    FAMILY_ORDER,
    POSITIVE_LAGS,
    REQUIRED_FAMILY_COLUMNS,
    REQUIRED_SYNTHESIS_COLUMNS,
    best_lag_and_corr,
    bh_adjust,
    build_audit,
    build_horizon_target,
    cycle_representation,
    dm_test,
    ecm_directional_pvalue,
    load_frame,
    numeric_frame,
    primary_target,
    rmse,
    select_var_lag,
    stationary_representation,
    turning_points,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify oil leading indicator run artifacts.")
    parser.add_argument("--run-dir", required=True, help="Run directory to verify.")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    require(abs(float(actual) - float(expected)) <= tolerance, f"{label} mismatch: {actual} vs {expected}")


def load_family_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for family in FAMILY_ORDER:
        path = run_dir / "tables" / f"family_{family}.csv"
        require(path.exists(), f"missing required family table: {path}")
        frame = pd.read_csv(path)
        required = REQUIRED_FAMILY_COLUMNS + FAMILY_EXTRA_COLUMNS[family]
        missing = [column for column in required if column not in frame.columns]
        require(not missing, f"{path.name} missing required columns: {missing}")
        tables[family] = frame
    synthesis_path = run_dir / "tables" / "synthesis.csv"
    require(synthesis_path.exists(), f"missing synthesis table: {synthesis_path}")
    synthesis = pd.read_csv(synthesis_path)
    missing = [column for column in REQUIRED_SYNTHESIS_COLUMNS if column not in synthesis.columns]
    require(not missing, f"synthesis.csv missing required columns: {missing}")
    tables["synthesis"] = synthesis
    return tables


def load_source_context(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, pd.Series], pd.Series, dict[str, pd.Series], list[str]]:
    manifest = json.loads((run_dir / "contract" / "analysis_manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["input_path"])
    frame = load_frame(source)
    numeric = numeric_frame(frame)
    primary = primary_target(numeric)
    variables = [column for column in numeric.columns if column not in {"Com_CrudeOil", "Com_BrentCrudeOil"}]
    candidate_cycles = {variable: cycle_representation(numeric[variable]) for variable in variables}
    candidate_stationary = {variable: stationary_representation(numeric[variable])[0] for variable in variables}
    target_stationary = stationary_representation(primary)[0]
    return frame, numeric, primary, candidate_cycles, target_stationary, candidate_stationary, variables


def exact_dataset_contract_check(run_dir: Path, frame: pd.DataFrame, numeric: pd.DataFrame) -> None:
    stored = json.loads((run_dir / "contract" / "dataset_audit.json").read_text(encoding="utf-8"))
    input_name = Path(json.loads((run_dir / "contract" / "analysis_manifest.json").read_text(encoding="utf-8"))["input_path"]).name
    if input_name == "df.csv" and len(frame) == 584:
        require(stored["rows"] == 584, f"dataset audit mismatch for rows: {stored['rows']}")
        require(stored["columns"] == 119, f"dataset audit mismatch for columns: {stored['columns']}")
        require(stored["numeric_columns"] == 118, f"dataset audit mismatch for numeric_columns: {stored['numeric_columns']}")
        require(stored["predictor_count"] == 116, f"dataset audit mismatch for predictor_count: {stored['predictor_count']}")
        require(stored["all_monday"] is True, "dataset audit mismatch for all_monday")
        require(stored["day_diffs"] == [7], f"dataset audit mismatch for day_diffs: {stored['day_diffs']}")
    else:
        recomputed = build_audit(frame, numeric)
        for key in ["rows", "columns", "numeric_columns", "predictor_count", "all_monday", "day_diffs", "excluded_columns"]:
            require(stored[key] == recomputed[key], f"dataset audit mismatch for {key}: {stored[key]} vs {recomputed[key]}")


def select_row(frame: pd.DataFrame, family: str) -> pd.Series | None:
    working = frame.copy()
    if family == "oos_dm":
        working = working[working["is_best_horizon"].astype(bool)]
    working = working[working["eligible"].astype(bool)]
    if family == "dfm":
        working = working[working["support_class"] != "inconclusive"]
    if working.empty:
        return None
    if family == "turning_point":
        working = working.assign(_abs=working["median_lead_weeks"].abs())
        return working.sort_values(["matched_turns", "_abs", "variable"], ascending=[False, False, True]).iloc[0]
    if family == "ccf":
        return working.sort_values(["fdr_pvalue", "best_lag_abs_corr", "variable"], ascending=[True, False, True]).iloc[0]
    if family == "predictive":
        working = working.assign(_min_directional_fdr=working[["forward_fdr_pvalue", "reverse_fdr_pvalue"]].min(axis=1))
        return working.sort_values(["_min_directional_fdr", "selected_lag", "variable"], ascending=[True, True, True]).iloc[0]
    if family == "dfm":
        return working.sort_values(["abs_loading", "factor_id", "variable"], ascending=[False, True, True]).iloc[0]
    if family == "oos_dm":
        return working.sort_values(["horizon_fdr_pvalue", "horizon_weeks", "variable"], ascending=[True, True, True]).iloc[0]
    if family == "frequency":
        working = working.assign(_abs=working["phase_lead_weeks"].abs())
        return working.sort_values(["coherence_max", "_abs", "variable"], ascending=[False, False, True]).iloc[0]
    raise ValueError(f"Unknown family {family}")


def verify_turning_row(actual: pd.Series, target_cycle: pd.Series, candidate_cycles: dict[str, pd.Series]) -> None:
    variable = actual["variable"]
    peaks_t, troughs_t = turning_points(target_cycle)
    peaks_x, troughs_x = turning_points(candidate_cycles[variable])
    target_turns = np.sort(np.concatenate([peaks_t, troughs_t]))
    candidate_turns = np.sort(np.concatenate([peaks_x, troughs_x]))
    diffs = []
    matched = 0
    for turn in target_turns:
        nearest = candidate_turns[np.argmin(np.abs(candidate_turns - turn))]
        delta = int(turn - nearest)
        if abs(delta) <= 12:
            matched += 1
            diffs.append(delta)
    assert_close(actual["matched_turns"], matched, 1e-6, "turning:matched_turns")
    assert_close(actual["median_lead_weeks"], float(np.median(diffs)), 1e-6, "turning:median_lead_weeks")
    assert_close(actual["mean_lead_weeks"], float(np.mean(diffs)), 1e-6, "turning:mean_lead_weeks")


def verify_ccf_row(actual: pd.Series, candidate_cycles: dict[str, pd.Series], candidate_stationary: dict[str, pd.Series], target_cycle: pd.Series, target_stationary: pd.Series) -> None:
    variable = actual["variable"]
    if actual["representation"] == "cycle":
        summary = best_lag_and_corr(candidate_cycles[variable], target_cycle, list(range(-8, 9)))
    else:
        summary = best_lag_and_corr(candidate_stationary[variable], target_stationary, list(range(-8, 9)))
    assert_close(actual["best_lag_weeks"], summary["best_lag"], 1e-6, "ccf:best_lag")
    assert_close(actual["corr_at_best_lag"], summary["corr"], 1e-6, "ccf:corr")


def verify_predictive_row(actual: pd.Series, actual_frame: pd.DataFrame, numeric: pd.DataFrame, primary: pd.Series, target_stationary: pd.Series) -> None:
    variable = actual["variable"]
    candidate_raw = numeric[variable]
    candidate_stationary = stationary_representation(candidate_raw)[0]
    aligned = pd.concat([target_stationary.rename("y"), candidate_stationary.rename("x")], axis=1).dropna()
    lag, lag_reason = select_var_lag(aligned[["y", "x"]])
    require(lag is not None, f"predictive lag selection failed during verify: {lag_reason}")
    if actual["path_used"] == "ecm":
        forward_raw, reverse_raw, _ = ecm_directional_pvalue(primary, candidate_raw, lag)
    else:
        forward_raw = float(grangercausalitytests(aligned[["y", "x"]], maxlag=[lag], verbose=False)[lag][0]["ssr_ftest"][1])
        reverse_raw = float(grangercausalitytests(aligned[["x", "y"]], maxlag=[lag], verbose=False)[lag][0]["ssr_ftest"][1])
    eligible = actual_frame[actual_frame["eligible"].astype(bool)]
    forward_series = eligible.set_index("variable")["forward_raw_pvalue"]
    reverse_series = eligible.set_index("variable")["reverse_raw_pvalue"]
    forward_fdr = bh_adjust(forward_series).loc[variable]
    reverse_fdr = bh_adjust(reverse_series).loc[variable]
    assert_close(actual["selected_lag"], lag, 1e-6, "predictive:selected_lag")
    assert_close(actual["forward_raw_pvalue"], forward_raw, 1e-6, "predictive:forward_raw")
    assert_close(actual["reverse_raw_pvalue"], reverse_raw, 1e-6, "predictive:reverse_raw")
    assert_close(actual["forward_fdr_pvalue"], forward_fdr, 1e-6, "predictive:forward_fdr")
    assert_close(actual["reverse_fdr_pvalue"], reverse_fdr, 1e-6, "predictive:reverse_fdr")
    require(actual["generic_pvalue_rule"] == "min_directional_pvalue", "predictive generic_pvalue_rule mismatch")
    require(actual["directional_effect"] in {"forward", "reverse", "bidirectional", "none"}, "predictive directional_effect invalid")


def verify_ccf_family(actual_frame: pd.DataFrame, candidate_cycles: dict[str, pd.Series], candidate_stationary: dict[str, pd.Series], target_cycle: pd.Series, target_stationary: pd.Series) -> None:
    for target_class in ["leading", "coincident"]:
        subset = actual_frame[(actual_frame["eligible"].astype(bool)) & (actual_frame["support_class"] == target_class)]
        if subset.empty:
            continue
        row = select_row(subset, "ccf")
        require(row is not None, f"ccf {target_class} selector failed")
        verify_ccf_row(row, candidate_cycles, candidate_stationary, target_cycle, target_stationary)


def verify_dfm_row(actual: pd.Series, numeric: pd.DataFrame, target_stationary: pd.Series, variables: list[str]) -> None:
    stationary_panel = {
        variable: stationary_representation(numeric[variable])[0]
        for variable in variables
        if len(stationary_representation(numeric[variable])[0]) > 0
    }
    panel = pd.DataFrame(stationary_panel).dropna(axis=0, how="any")
    require(not panel.empty, "dfm verification panel is empty")
    standardized = (panel - panel.mean()) / panel.std(ddof=0).replace(0, 1.0)
    pca = PCA(n_components=min(5, standardized.shape[1]))
    pca.fit_transform(standardized)
    explained = np.cumsum(pca.explained_variance_ratio_)
    factor_count = int(min(5, np.searchsorted(explained, 0.60) + 1))
    loads = pd.DataFrame(pca.components_[:factor_count].T, index=standardized.columns, columns=[f"F{i+1}" for i in range(factor_count)])
    variable = actual["variable"]
    strongest = loads.loc[variable].abs().idxmax()
    loading = float(loads.loc[variable].abs().max())
    assert_close(actual["abs_loading"], loading, 1e-4, "dfm:abs_loading")
    require(str(actual["factor_id"]) == str(strongest), "dfm:factor_id mismatch")
    require(int(actual["factor_count"]) == factor_count, "dfm factor_count mismatch")


def direct_ols_forecast(train: pd.DataFrame, test_row: pd.Series, features: list[str]) -> float:
    model = LinearRegression()
    model.fit(train[features], train["target"])
    return float(model.predict(test_row[features].to_frame().T)[0])


def verify_oos_row(actual: pd.Series, primary: pd.Series, candidate_stationary: dict[str, pd.Series]) -> None:
    variable = actual["variable"]
    horizon = int(actual["horizon_weeks"])
    stationary = candidate_stationary[variable]
    target_h = build_horizon_target(primary, horizon)
    design = pd.DataFrame({"target": target_h})
    for lag in POSITIVE_LAGS:
        design[f"target_lag_{lag}"] = primary.shift(lag)
        design[f"cand_lag_{lag}"] = stationary.shift(lag)
    design = design.dropna().reset_index(drop=True)
    baseline_features = [f"target_lag_{lag}" for lag in POSITIVE_LAGS]
    candidate_features = baseline_features + [f"cand_lag_{lag}" for lag in POSITIVE_LAGS]
    baseline_errors = []
    candidate_errors = []
    for idx in range(260, len(design)):
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
    assert_close(actual["baseline_rmse"], rmse(baseline_arr), 1e-6, "oos:baseline_rmse")
    assert_close(actual["candidate_rmse"], rmse(candidate_arr), 1e-6, "oos:candidate_rmse")
    assert_close(actual["dm_stat"], dm_stat, 1e-6, "oos:dm_stat")
    assert_close(actual["dm_hln_adjusted"], dm_adj, 1e-6, "oos:dm_adj")
    require(actual["row_grain"] == "variable_horizon", "oos row_grain mismatch")
    require(bool(actual["is_best_horizon"]) is True, "oos best horizon flag mismatch")
    require(np.isfinite(pvalue), "oos raw pvalue not finite")


def verify_frequency_row(actual: pd.Series, candidate_stationary: dict[str, pd.Series], target_stationary: pd.Series) -> None:
    variable = actual["variable"]
    pair = pd.concat([candidate_stationary[variable].rename("x"), target_stationary.rename("y")], axis=1).dropna()
    x = pair["x"].to_numpy(dtype=float)
    y = pair["y"].to_numpy(dtype=float)
    nperseg = min(128, len(pair))
    freqs, coh = signal.coherence(x, y, fs=1.0, nperseg=nperseg)
    _, csd_vals = signal.csd(x, y, fs=1.0, nperseg=nperseg)
    phase = np.angle(csd_vals)
    best_band = None
    best_coh = -1.0
    best_phase_weeks = np.nan
    for label, (low, high) in BAND_DEFS.items():
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            continue
        idx = np.argmax(coh[mask])
        band_freqs = freqs[mask]
        band_coh = coh[mask]
        band_phase = phase[mask]
        candidate_coh = float(band_coh[idx])
        candidate_freq = float(band_freqs[idx])
        candidate_phase_weeks = float(band_phase[idx] / (2 * np.pi * candidate_freq)) if candidate_freq > 0 else np.nan
        if candidate_coh > best_coh:
            best_band = label
            best_coh = candidate_coh
            best_phase_weeks = candidate_phase_weeks
    require(str(actual["dominant_band"]) == str(best_band), "frequency dominant_band mismatch")
    assert_close(actual["coherence_max"], best_coh, 1e-4, "frequency coherence")
    assert_close(actual["phase_lead_weeks"], best_phase_weeks, 1e-4, "frequency phase")


def verify_synthesis_thresholds(synthesis: pd.DataFrame, family_tables: dict[str, pd.DataFrame]) -> None:
    for row in synthesis.itertuples(index=False):
        predictive_leading_fdrs = []
        coincident_fdrs = []
        for family, frame in family_tables.items():
            if family == "synthesis":
                continue
            subset = frame[frame["variable"] == row.variable]
            if family == "oos_dm":
                subset = subset[subset["is_best_horizon"].astype(bool)]
            if subset.empty or not bool(subset.iloc[0].get("eligible", True)):
                continue
            current = subset.iloc[0]
            if current["support_class"] == "leading" and family in {"predictive", "oos_dm"} and not pd.isna(current.get("fdr_pvalue", np.nan)):
                predictive_leading_fdrs.append(float(current["fdr_pvalue"]))
            if current["support_class"] == "coincident" and not pd.isna(current.get("fdr_pvalue", np.nan)):
                coincident_fdrs.append(float(current["fdr_pvalue"]))
        predictive_stronger = bool(predictive_leading_fdrs and coincident_fdrs and min(predictive_leading_fdrs) < min(coincident_fdrs))
        if row.final_class == "leading":
            require(row.eligible_family_count >= 4 and row.leading_votes >= 3 and bool(row.predictive_support) and row.lagging_votes <= 1, f"leading threshold broken for {row.variable}")
        elif row.final_class == "coincident":
            require(row.eligible_family_count >= 4 and row.coincident_votes >= 3 and not predictive_stronger, f"coincident threshold broken for {row.variable}")
        elif row.final_class == "lagging":
            require(row.eligible_family_count >= 4 and row.lagging_votes >= 3 and row.leading_votes <= 1, f"lagging threshold broken for {row.variable}")
        elif row.final_class == "mixed":
            require(row.eligible_family_count >= 4 and (row.leading_votes + row.coincident_votes + row.lagging_votes) >= 2, f"mixed threshold broken for {row.variable}")
        elif row.final_class == "inconclusive":
            require(row.eligible_family_count < 4 or max(row.leading_votes, row.coincident_votes, row.lagging_votes) < 2, f"inconclusive threshold broken for {row.variable}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    require(run_dir.exists(), f"run dir not found: {run_dir}")
    require((run_dir / "contract" / "dataset_audit.json").exists(), "missing dataset_audit.json")
    require((run_dir / "contract" / "analysis_manifest.json").exists(), "missing analysis_manifest.json")
    require((run_dir / "report.md").exists(), "missing report.md")
    require((run_dir / "summary.json").exists(), "missing summary.json")

    family_tables = load_family_tables(run_dir)
    frame, numeric, primary, candidate_cycles, target_stationary, candidate_stationary, variables = load_source_context(run_dir)
    target_cycle = cycle_representation(primary)
    exact_dataset_contract_check(run_dir, frame, numeric)

    manifest_mtime = (run_dir / "contract" / "analysis_manifest.json").stat().st_mtime
    for family in FAMILY_ORDER:
        require((run_dir / "tables" / f"family_{family}.csv").stat().st_mtime >= manifest_mtime, f"family_{family}.csv predates analysis_manifest.json")

    for family in FAMILY_ORDER:
        actual = family_tables[family]
        selector = select_row(actual, family)
        if selector is None:
            if family == "dfm":
                stationary_candidates = [
                    variable for variable in variables if len(stationary_representation(numeric[variable])[0]) > 0
                ]
                if len(stationary_candidates) >= 2:
                    require(False, "dfm should expose an eligible stationary subset but is fully blocked")
            require((~actual["eligible"].astype(bool)).any() or actual["support_class"].eq("inconclusive").all(), f"{family} lacks representative row and lacks blocked/inconclusive evidence")
            continue
        if family == "turning_point":
            verify_turning_row(selector, target_cycle, candidate_cycles)
        elif family == "ccf":
            verify_ccf_family(actual, candidate_cycles, candidate_stationary, target_cycle, target_stationary)
            verify_ccf_row(selector, candidate_cycles, candidate_stationary, target_cycle, target_stationary)
        elif family == "predictive":
            verify_predictive_row(selector, actual, numeric, primary, target_stationary)
        elif family == "dfm":
            verify_dfm_row(selector, numeric, target_stationary, variables)
        elif family == "oos_dm":
            verify_oos_row(selector, primary, candidate_stationary)
        elif family == "frequency":
            verify_frequency_row(selector, candidate_stationary, target_stationary)

    verify_synthesis_thresholds(family_tables["synthesis"], family_tables)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    require("blocked rows by family" in report, "report missing blocked-family summary")
    require("mixed" in report and "inconclusive" in report, "report missing mixed/inconclusive language")
    require("unstable candidates" in report, "report missing unstable-candidate coverage")
    require("contradictory cases" in report, "report missing contradictory-case coverage")
    require("## Blocked Family Rationale" in report, "report missing blocked-family rationale section")

    print(json.dumps({"run_dir": str(run_dir), "status": "ok"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
