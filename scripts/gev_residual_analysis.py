"""
Task-specific GEV analysis for ML/DL 8-week Brent oil forecasting.

Fits GEV to model residuals to quantify:
1. Prediction uncertainty by horizon step
2. Extreme error risk (VaR/CVaR)
3. Asymmetric risk profile (upside vs downside miss)

Usage:
    uv run python scripts/gev_residual_analysis.py \
        --forecast-runs runs/feature_set_brentoil_case1_jobs_1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_forecasts(run_dir: Path) -> pd.DataFrame:
    """Load and merge all model forecasts from a run directory."""
    frames = []
    cv_dir = run_dir / "scheduler" / "workers"
    if not cv_dir.exists():
        raise FileNotFoundError(f"No workers directory: {cv_dir}")

    for worker_dir in cv_dir.iterdir():
        if not worker_dir.is_dir():
            continue
        forecast_files = list(worker_dir.glob("cv/*_forecasts.csv"))
        for fpath in forecast_files:
            df = pd.read_csv(fpath)
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No forecast CSV files found")

    combined = pd.concat(frames, ignore_index=True)
    combined["ds"] = pd.to_datetime(combined["ds"])
    combined["cutoff"] = pd.to_datetime(combined["cutoff"])
    combined["residual"] = combined["y"] - combined["y_hat"]
    combined["abs_residual"] = combined["residual"].abs()
    combined["pct_error"] = combined["residual"] / combined["y"] * 100
    combined["log_error"] = np.log(combined["y_hat"] / combined["y"])

    return combined


def fit_gev_residual(data: np.ndarray, label: str) -> dict:
    """Fit GEV to residual array and return diagnostics."""
    shape, loc, scale = stats.genextreme.fit(data)
    xi = -shape  # scipy convention

    # Goodness of fit
    ks_stat, ks_p = stats.kstest(data, "genextreme", args=(shape, loc, scale))

    # Quantiles
    quantiles = {}
    for p in [0.01, 0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975, 0.99]:
        quantiles[f"q{int(p * 100)}"] = float(
            stats.genextreme.ppf(p, shape, loc, scale)
        )

    # Log-likelihood
    ll = float(np.sum(stats.genextreme.logpdf(data, shape, loc, scale)))
    aic = 2 * 3 - 2 * ll
    bic = 3 * np.log(len(data)) - 2 * ll

    return {
        "label": label,
        "n_obs": len(data),
        "params": {
            "shape_xi": float(xi),
            "location_mu": float(loc),
            "scale_sigma": float(scale),
        },
        "distribution_type": (
            "Gumbel (thin tail)"
            if abs(xi) < 0.01
            else "Fréchet (heavy tail)"
            if xi > 0
            else "Weibull (bounded tail)"
        ),
        "goodness_of_fit": {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "fit_ok": bool(ks_p > 0.05),
        },
        "quantiles": quantiles,
        "model_fit": {"log_likelihood": ll, "aic": aic, "bic": bic},
        "empirical": {
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
            "skew": float(pd.Series(data).skew()),
            "kurtosis": float(pd.Series(data).kurtosis()),
        },
    }


def compute_var_cvar(residuals: np.ndarray, confidence: float = 0.95) -> dict:
    """Compute VaR and CVaR (Expected Shortfall)."""
    sorted_res = np.sort(residuals)
    n = len(sorted_res)

    # Lower tail (worst losses)
    lower_idx = int((1 - confidence) * n)
    var_lower = float(sorted_res[lower_idx])
    cvar_lower = float(sorted_res[:lower_idx].mean()) if lower_idx > 0 else var_lower

    # Upper tail (best case)
    upper_idx = int(confidence * n)
    var_upper = float(sorted_res[upper_idx])
    cvar_upper = float(sorted_res[upper_idx:].mean()) if upper_idx < n else var_upper

    return {
        "confidence": confidence,
        "lower_tail": {
            "VaR": var_lower,
            "CVaR": cvar_lower,
            "interpretation": f"Worst {(1 - confidence) * 100:.1f}% of errors are below {var_lower:.2f}",
        },
        "upper_tail": {
            "VaR": var_upper,
            "CVaR": cvar_upper,
            "interpretation": f"Best {confidence * 100:.1f}% of errors are above {var_upper:.2f}",
        },
        "asymmetry": {
            "range_lower": abs(var_lower),
            "range_upper": abs(var_upper),
            "ratio": abs(var_lower) / abs(var_upper)
            if var_upper != 0
            else float("inf"),
            "note": (
                "Downside risk is LARGER"
                if abs(var_lower) > abs(var_upper)
                else "Upside potential is LARGER"
            ),
        },
    }


def build_prediction_interval(
    point_forecast: float,
    gev_quantiles: dict,
    method: str = "residual",
) -> dict:
    """Build prediction intervals from GEV residual quantiles."""
    intervals = {}
    for qname, qval in gev_quantiles.items():
        level = int(qname.replace("q", ""))
        if level < 50:
            lower = point_forecast + qval
            intervals[f"{level}%"] = {"lower": lower, "upper": None}
        elif level > 50:
            upper = point_forecast + qval
            key = f"{100 - level}%"
            if key in intervals:
                intervals[key]["upper"] = upper
            else:
                intervals[key] = {"lower": None, "upper": upper}
    return intervals


def main():
    parser = argparse.ArgumentParser(
        description="GEV analysis on model residuals for 8-week Brent forecasting"
    )
    parser.add_argument(
        "--forecast-runs",
        required=True,
        help="Path to a run directory containing forecast CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <run_dir>/gev_residual_analysis)",
    )
    args = parser.parse_args()

    run_dir = Path(args.forecast_runs)
    output_dir = (
        Path(args.output_dir) if args.output_dir else run_dir / "gev_residual_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading forecasts from: {run_dir}")
    df = load_forecasts(run_dir)
    print(f"  Total forecast rows: {len(df)}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Folds: {sorted(df['fold_idx'].unique().tolist())}")
    print(f"  Horizon steps: {sorted(df['horizon_step'].unique().tolist())}")
    print()

    results = {"run_dir": str(run_dir), "analyses": {}}

    # --- 1. Overall residual GEV ---
    print("=" * 60)
    print("1. OVERALL RESIDUAL GEV (all models, all horizons)")
    print("=" * 60)
    residuals_all = df["residual"].values
    overall = fit_gev_residual(residuals_all, "overall")
    overall["var_cvar"] = compute_var_cvar(residuals_all)
    results["analyses"]["overall"] = overall
    print(json.dumps(overall, indent=2))
    print()

    # --- 2. Per-model residual GEV ---
    print("=" * 60)
    print("2. PER-MODEL RESIDUAL GEV")
    print("=" * 60)
    per_model = {}
    for model in df["model"].unique():
        model_res = df[df["model"] == model]["residual"].values
        if len(model_res) < 10:
            print(f"  Skipping {model}: too few observations ({len(model_res)})")
            continue
        print(f"\n  Model: {model} (n={len(model_res)})")
        mresult = fit_gev_residual(model_res, f"model_{model}")
        mresult["var_cvar"] = compute_var_cvar(model_res)
        per_model[model] = mresult
        print(
            f"    ξ={mresult['params']['shape_xi']:.4f} ({mresult['distribution_type']})"
        )
        print(f"    KS p-value: {mresult['goodness_of_fit']['ks_pvalue']:.4f}")
        print(f"    95% VaR: {mresult['var_cvar']['lower_tail']['VaR']:.2f}")
    results["analyses"]["per_model"] = per_model
    print()

    # --- 3. Per-horizon residual GEV ---
    print("=" * 60)
    print("3. PER-HORIZON RESIDUAL GEV (1-8 weeks)")
    print("=" * 60)
    per_horizon = {}
    for h in sorted(df["horizon_step"].unique()):
        h_res = df[df["horizon_step"] == h]["residual"].values
        if len(h_res) < 10:
            continue
        print(f"\n  Horizon: {h} (n={len(h_res)})")
        hresult = fit_gev_residual(h_res, f"horizon_{h}")
        hresult["var_cvar"] = compute_var_cvar(h_res)
        per_horizon[int(h)] = hresult
        print(
            f"    ξ={hresult['params']['shape_xi']:.4f} ({hresult['distribution_type']})"
        )
        print(f"    Mean error: {hresult['empirical']['mean']:.2f}")
        print(f"    95% VaR (down): {hresult['var_cvar']['lower_tail']['VaR']:.2f}")
        print(f"    95% VaR (up): {hresult['var_cvar']['upper_tail']['VaR']:.2f}")
    results["analyses"]["per_horizon"] = per_horizon
    print()

    # --- 4. Prediction interval example ---
    print("=" * 60)
    print("4. PREDICTION INTERVAL EXAMPLE")
    print("=" * 60)
    # Use last known price as example
    last_price = df.sort_values("ds").iloc[-1]["y"]
    print(f"  Current price: {last_price:.2f}")
    print()

    for h in [1, 4, 8]:
        if h not in per_horizon:
            continue
        hq = per_horizon[h]["quantiles"]
        print(f"  Horizon {h} week(s):")
        for qname in ["q2", "q5", "q10"]:
            if qname in hq:
                lower = last_price + hq[qname]
                print(f"    {qname} lower bound: {lower:.2f}")
        for qname in ["q90", "q95", "q97", "q99"]:
            if qname in hq:
                upper = last_price + hq[qname]
                print(f"    {qname} upper bound: {upper:.2f}")
        print()

    # --- 5. Asymmetric risk summary ---
    print("=" * 60)
    print("5. ASYMMETRIC RISK SUMMARY")
    print("=" * 60)
    for h in sorted(per_horizon.keys()):
        vc = per_horizon[h]["var_cvar"]
        print(
            f"  Horizon {h}: Downside VaR={vc['lower_tail']['VaR']:.2f}, "
            f"Upside VaR={vc['upper_tail']['VaR']:.2f}, "
            f"Ratio={vc['asymmetry']['ratio']:.2f}x"
        )
    print()

    # --- Save results ---
    output_path = output_dir / "gev_residual_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

    # --- Save horizon summary table ---
    horizon_rows = []
    for h, hdata in sorted(per_horizon.items()):
        horizon_rows.append(
            {
                "horizon_week": h,
                "n_obs": hdata["n_obs"],
                "shape_xi": hdata["params"]["shape_xi"],
                "dist_type": hdata["distribution_type"],
                "mean_error": hdata["empirical"]["mean"],
                "std_error": hdata["empirical"]["std"],
                "q5_lower": hdata["quantiles"]["q5"],
                "q95_upper": hdata["quantiles"]["q95"],
                "VaR_95": hdata["var_cvar"]["lower_tail"]["VaR"],
                "CVaR_95": hdata["var_cvar"]["lower_tail"]["CVaR"],
                "ks_pvalue": hdata["goodness_of_fit"]["ks_pvalue"],
            }
        )
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_csv = output_dir / "horizon_gev_summary.csv"
    horizon_df.to_csv(horizon_csv, index=False)
    print(f"Horizon summary saved to: {horizon_csv}")


if __name__ == "__main__":
    main()
