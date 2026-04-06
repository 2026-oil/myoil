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


def _run_subprocess(command: list[str], payload: dict) -> list[float]:
    completed = subprocess.run(
        command,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"subprocess failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    data = json.loads(completed.stdout)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]
    return [float(value) for value in data]


def forecast_with_chronos2(series: list[float], horizon: int) -> list[float]:
    payload = {
        "model_id": "amazon/chronos-2",
        "torch_dtype": "float32",
        "device": "cpu",
        "prediction_length": horizon,
        "series": series,
    }
    script = """
import json
import sys
import torch
from chronos import Chronos2Pipeline

payload = json.load(sys.stdin)
pipeline = Chronos2Pipeline.from_pretrained(
    payload["model_id"],
    device_map=payload["device"],
    torch_dtype=torch.float32,
)
context = [torch.tensor(payload["series"], dtype=torch.float32)]
_, mean = pipeline.predict_quantiles(
    inputs=context,
    prediction_length=payload["prediction_length"],
    quantile_levels=[0.5],
    batch_size=1,
    limit_prediction_length=True,
)
item = torch.as_tensor(mean[0], dtype=torch.float32).cpu()
if item.ndim == 2:
    item = item.squeeze(0)
json.dump(item.tolist(), sys.stdout)
"""
    return _run_subprocess(
        [
            "uv",
            "run",
            "--with",
            "chronos-forecasting>=2.0,<3",
            "python",
            "-c",
            script,
        ],
        payload,
    )


def forecast_with_timesfm(series: list[float], horizon: int) -> list[float]:
    payload = {
        "model_id": "google/timesfm-2.5-200m-transformers",
        "prediction_length": horizon,
        "series": series,
    }
    script = """
import json
import sys
import torch
from transformers import TimesFm2_5ModelForPrediction

payload = json.load(sys.stdin)
model = TimesFm2_5ModelForPrediction.from_pretrained(payload["model_id"])
model = model.to(device="cpu", dtype=torch.float32).eval()
context = [torch.tensor(payload["series"], dtype=torch.float32)]
with torch.no_grad():
    outputs = model(past_values=context)
item = torch.as_tensor(outputs.mean_predictions, dtype=torch.float32).cpu()
if item.ndim == 2:
    item = item[0]
json.dump(item[: payload["prediction_length"]].tolist(), sys.stdout)
"""
    return _run_subprocess(
        [
            "uv",
            "run",
            "--with",
            "transformers>=5.4.0",
            "python",
            "-c",
            script,
        ],
        payload,
    )


def _metric_rows(target: str, actual: list[float], forecasts: dict[str, list[float]]) -> list[dict]:
    rows: list[dict] = []
    actual_series = pd.Series(actual, dtype=float)
    for model_name, prediction in forecasts.items():
        pred_series = pd.Series(prediction, dtype=float)
        error = actual_series - pred_series
        rows.append(
            {
                "target": target,
                "model": model_name,
                "mae": float(error.abs().mean()),
                "mse": float((error**2).mean()),
                "rmse": float(((error**2).mean()) ** 0.5),
            }
        )
    return rows


def _plot_target(
    output_path: Path,
    title: str,
    history_dates: pd.Series,
    history_values: pd.Series,
    forecast_dates: pd.Series,
    actual_values: pd.Series,
    forecasts: dict[str, list[float]],
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(history_dates, history_values, color="0.7", linewidth=2, label="history")
    plt.plot(
        forecast_dates,
        actual_values,
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
        label="actual(last 8w)",
    )
    palette = {
        "Chronos2": "#ff7f0e",
        "TimesFM2_5": "#2ca02c",
    }
    for model_name, prediction in forecasts.items():
        plt.plot(
            forecast_dates,
            prediction,
            marker="o",
            linewidth=2,
            label=model_name,
            color=palette.get(model_name),
        )
    plt.axvline(forecast_dates.iloc[0], color="0.4", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel(title)
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
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("runs") / f"gprd_recent8_compare_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    panel_data: list[tuple[str, pd.Series, pd.Series, pd.Series, pd.Series, dict[str, list[float]]]] = []

    for target in args.targets:
        series = df[target].astype(float)
        train = series.iloc[:-args.horizon]
        actual = series.iloc[-args.horizon:]
        forecast_dates = df["dt"].iloc[-args.horizon:].reset_index(drop=True)

        chronos_pred = forecast_with_chronos2(train.tolist(), args.horizon)
        timesfm_pred = forecast_with_timesfm(train.tolist(), args.horizon)
        forecasts = {
            "Chronos2": chronos_pred,
            "TimesFM2_5": timesfm_pred,
        }

        forecast_frame = pd.DataFrame(
            {
                "dt": forecast_dates,
                "target": target,
                "actual": actual.to_numpy(),
                "Chronos2": chronos_pred,
                "TimesFM2_5": timesfm_pred,
            }
        )
        forecast_frames.append(forecast_frame)
        metric_rows.extend(_metric_rows(target, actual.tolist(), forecasts))

        history_start = max(0, len(df) - args.horizon - args.plot_context)
        history_dates = df["dt"].iloc[history_start:-args.horizon].reset_index(drop=True)
        history_values = series.iloc[history_start:-args.horizon].reset_index(drop=True)
        actual_values = actual.reset_index(drop=True)
        panel_data.append(
            (
                target,
                history_dates,
                history_values,
                forecast_dates,
                actual_values,
                forecasts,
            )
        )
        _plot_target(
            output_dir / f"{target}_recent8_compare.png",
            title=target,
            history_dates=history_dates,
            history_values=history_values,
            forecast_dates=forecast_dates,
            actual_values=actual_values,
            forecasts=forecasts,
        )

    combined = pd.concat(forecast_frames, ignore_index=True)
    combined.to_csv(output_dir / "recent8_forecasts.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "recent8_metrics.csv", index=False)

    fig, axes = plt.subplots(len(panel_data), 1, figsize=(12, 4 * len(panel_data)), sharex=False)
    if len(panel_data) == 1:
        axes = [axes]
    palette = {"Chronos2": "#ff7f0e", "TimesFM2_5": "#2ca02c"}
    for ax, (target, history_dates, history_values, forecast_dates, actual_values, forecasts) in zip(axes, panel_data):
        ax.plot(history_dates, history_values, color="0.7", linewidth=2, label="history")
        ax.plot(forecast_dates, actual_values, color="#1f77b4", marker="o", linewidth=2.5, label="actual(last 8w)")
        for model_name, prediction in forecasts.items():
            ax.plot(
                forecast_dates,
                prediction,
                marker="o",
                linewidth=2,
                label=model_name,
                color=palette.get(model_name),
            )
        ax.axvline(forecast_dates.iloc[0], color="0.4", linestyle="--", linewidth=1)
        ax.set_title(target)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "recent8_compare_all_targets.png", dpi=180)
    plt.close(fig)

    metadata = {
        "data_path": str(data_path),
        "targets": args.targets,
        "horizon": args.horizon,
        "output_dir": str(output_dir),
        "combined_csv": str(output_dir / "recent8_forecasts.csv"),
        "metrics_csv": str(output_dir / "recent8_metrics.csv"),
        "combined_png": str(output_dir / "recent8_compare_all_targets.png"),
        "target_pngs": {
            target: str(output_dir / f"{target}_recent8_compare.png")
            for target in args.targets
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
