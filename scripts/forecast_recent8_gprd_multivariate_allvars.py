from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_TARGETS = ("GPRD", "GPRD_ACT", "GPRD_THREAT")
DEFAULT_HORIZON = 8
DEFAULT_PLOT_CONTEXT = 32
ID_COLUMN = "item_id"
TIME_COLUMN = "dt"


def chronos2_multivariate_predict_df(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame,
    targets: list[str],
    horizon: int,
) -> pd.DataFrame:
    context_payload = context_df.copy()
    future_payload = future_df.copy()
    context_payload[TIME_COLUMN] = context_payload[TIME_COLUMN].astype(str)
    future_payload[TIME_COLUMN] = future_payload[TIME_COLUMN].astype(str)
    payload = {
        "context_records": context_payload.to_dict(orient="records"),
        "future_records": future_payload.to_dict(orient="records"),
        "targets": targets,
        "prediction_length": horizon,
    }
    script = """
import json
import sys
import pandas as pd
import torch
from chronos import Chronos2Pipeline

payload = json.load(sys.stdin)
context_df = pd.DataFrame(payload["context_records"])
future_df = pd.DataFrame(payload["future_records"])
context_df["dt"] = pd.to_datetime(context_df["dt"])
future_df["dt"] = pd.to_datetime(future_df["dt"])

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
    torch_dtype=torch.float32,
)

pred_df = pipeline.predict_df(
    df=context_df,
    future_df=future_df,
    id_column="item_id",
    timestamp_column="dt",
    target=payload["targets"],
    prediction_length=payload["prediction_length"],
    quantile_levels=[0.5],
    batch_size=128,
    cross_learning=False,
)
pred_df["dt"] = pred_df["dt"].astype(str)
sys.stdout.write(pred_df.to_json(orient="records"))
"""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "chronos-forecasting>=2.0,<3",
            "python",
            "-c",
            script,
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Chronos2 multivariate subprocess failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    records = json.loads(completed.stdout)
    pred_df = pd.DataFrame(records)
    pred_df["dt"] = pd.to_datetime(pred_df["dt"])
    return pred_df


def _plot_target(
    output_path: Path,
    target: str,
    history_dates: pd.Series,
    history_values: pd.Series,
    forecast_dates: pd.Series,
    actual_values: pd.Series,
    prediction_values: pd.Series,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(history_dates, history_values, color="0.75", linewidth=2, label="history")
    plt.plot(
        forecast_dates,
        actual_values,
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
        label="actual(last 8w)",
    )
    plt.plot(
        forecast_dates,
        prediction_values,
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Chronos2_multivariate_allvars",
    )
    plt.axvline(forecast_dates.iloc[0], color="0.4", linestyle="--", linewidth=1)
    plt.title(target)
    plt.xlabel("date")
    plt.ylabel(target)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/df.csv")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--plot-context", type=int, default=DEFAULT_PLOT_CONTEXT)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    df = pd.read_csv(data_path)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    numeric_columns = [c for c in df.columns if c != TIME_COLUMN]
    covariate_columns = [c for c in numeric_columns if c not in args.targets]

    context_df = df.iloc[:-args.horizon].copy()
    future_df = df.iloc[-args.horizon :][[TIME_COLUMN, *covariate_columns]].copy()
    context_df.insert(0, ID_COLUMN, "gprd")
    future_df.insert(0, ID_COLUMN, "gprd")

    pred_df = chronos2_multivariate_predict_df(
        context_df=context_df,
        future_df=future_df,
        targets=list(args.targets),
        horizon=args.horizon,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("runs")
        / f"gprd_recent8_multivariate_allvars_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    metrics: list[dict] = []
    panel_data: list[tuple[str, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]] = []

    for target in args.targets:
        pred_target = pred_df.loc[pred_df["target_name"] == target, [TIME_COLUMN, "predictions"]].reset_index(drop=True)
        actual_target = (
            df.loc[df[TIME_COLUMN].isin(pred_target[TIME_COLUMN]), [TIME_COLUMN, target]]
            .reset_index(drop=True)
            .rename(columns={target: "actual"})
        )
        merged = actual_target.merge(pred_target, on=TIME_COLUMN, how="inner")
        merged["target"] = target
        rows.append(merged[[TIME_COLUMN, "target", "actual", "predictions"]])

        err = merged["actual"] - merged["predictions"]
        metrics.append(
            {
                "target": target,
                "model": "Chronos2_multivariate_allvars",
                "mae": float(err.abs().mean()),
                "mse": float((err**2).mean()),
                "rmse": float(((err**2).mean()) ** 0.5),
            }
        )

        history_start = max(0, len(df) - args.horizon - args.plot_context)
        history_dates = df[TIME_COLUMN].iloc[history_start:-args.horizon].reset_index(drop=True)
        history_values = df[target].astype(float).iloc[history_start:-args.horizon].reset_index(drop=True)
        forecast_dates = merged[TIME_COLUMN].reset_index(drop=True)
        actual_values = merged["actual"].reset_index(drop=True)
        prediction_values = merged["predictions"].reset_index(drop=True)
        panel_data.append((target, history_dates, history_values, forecast_dates, actual_values, prediction_values))

        _plot_target(
            output_path=output_dir / f"{target}_recent8_multivariate_allvars.png",
            target=target,
            history_dates=history_dates,
            history_values=history_values,
            forecast_dates=forecast_dates,
            actual_values=actual_values,
            prediction_values=prediction_values,
        )

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / "recent8_multivariate_allvars_forecasts.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "recent8_multivariate_allvars_metrics.csv", index=False)

    fig, axes = plt.subplots(len(panel_data), 1, figsize=(12, 4 * len(panel_data)), sharex=False)
    if len(panel_data) == 1:
        axes = [axes]
    for ax, (target, history_dates, history_values, forecast_dates, actual_values, prediction_values) in zip(axes, panel_data):
        ax.plot(history_dates, history_values, color="0.75", linewidth=2, label="history")
        ax.plot(forecast_dates, actual_values, color="#1f77b4", marker="o", linewidth=2.5, label="actual(last 8w)")
        ax.plot(
            forecast_dates,
            prediction_values,
            color="#d62728",
            marker="o",
            linewidth=2,
            label="Chronos2_multivariate_allvars",
        )
        ax.axvline(forecast_dates.iloc[0], color="0.4", linestyle="--", linewidth=1)
        ax.set_title(target)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "recent8_multivariate_allvars_compare_all_targets.png", dpi=180)
    plt.close(fig)

    metadata = {
        "data_path": str(data_path),
        "targets": args.targets,
        "horizon": args.horizon,
        "context_rows_used": int(len(context_df)),
        "future_rows_compared": int(args.horizon),
        "covariate_count": int(len(covariate_columns)),
        "covariates": covariate_columns,
        "model": "Chronos2_multivariate_allvars",
        "output_dir": str(output_dir),
        "combined_csv": str(output_dir / "recent8_multivariate_allvars_forecasts.csv"),
        "metrics_csv": str(output_dir / "recent8_multivariate_allvars_metrics.csv"),
        "combined_png": str(output_dir / "recent8_multivariate_allvars_compare_all_targets.png"),
        "target_pngs": {
            target: str(output_dir / f"{target}_recent8_multivariate_allvars.png")
            for target in args.targets
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
