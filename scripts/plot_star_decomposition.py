from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plugins.aa_forecast import STARFeatureExtractor


TARGET_COLUMNS = {
    "wti": "Com_CrudeOil",
    "brent": "Com_BrentCrudeOil",
    "gprd_threat": "GPRD_THREAT",
}
TARGET_TAILS = {
    "wti": "two_sided",
    "brent": "two_sided",
    "gprd_threat": "upward",
}
DEFAULT_LOWESS_FRAC = 0.6
DEFAULT_LOWESS_DELTA = 0.01
DEFAULT_SEASON_LENGTH = 4
DEFAULT_P_VALUE = 0.05
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AA-Forecast STAR decomposition for WTI and Brent from data/df.csv."
    )
    parser.add_argument(
        "--input",
        default="data/df.csv",
        help="Input CSV path (default: data/df.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to runs/star-decomposition-<timestamp>.",
    )
    parser.add_argument(
        "--lowess-frac",
        type=float,
        default=DEFAULT_LOWESS_FRAC,
        help=f"LOWESS frac parameter (default: {DEFAULT_LOWESS_FRAC}).",
    )
    parser.add_argument(
        "--lowess-delta",
        type=float,
        default=DEFAULT_LOWESS_DELTA,
        help=f"LOWESS delta parameter (default: {DEFAULT_LOWESS_DELTA}).",
    )
    parser.add_argument(
        "--season-length",
        type=int,
        default=DEFAULT_SEASON_LENGTH,
        help=f"STAR seasonal period (default: {DEFAULT_SEASON_LENGTH}).",
    )
    parser.add_argument(
        "--p-value",
        type=float,
        default=DEFAULT_P_VALUE,
        help=f"Shared STAR anomaly p-value (default: {DEFAULT_P_VALUE}).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGET_COLUMNS),
        default=["wti", "brent"],
        help="Target aliases to plot (default: wti brent).",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "runs" / f"star-decomposition-{timestamp}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "dt" not in df.columns:
        raise ValueError(f"Input CSV must include a 'dt' column: {csv_path}")
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Input CSV has no valid datetime rows after parsing 'dt': {csv_path}")
    return df


def extract_target_frame(df: pd.DataFrame, label: str, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Required {label} target column is missing: {column}")
    target_frame = df[["dt", column]].dropna(subset=[column]).reset_index(drop=True)
    if target_frame.empty:
        raise ValueError(f"Target {label} has no non-null rows in column {column}")
    return target_frame


def to_component_series(
    target_frame: pd.DataFrame,
    extractor: STARFeatureExtractor,
) -> dict[str, np.ndarray]:
    values = target_frame.iloc[:, 1].to_numpy(dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Target series must be one-dimensional and non-empty.")
    tensor = torch.as_tensor(values, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        components = extractor(tensor)
    return {
        name: component.detach().cpu().numpy().reshape(-1)
        for name, component in components.items()
    }


def compute_robustness_score(raw_residual: np.ndarray) -> np.ndarray:
    residual_center = np.median(raw_residual)
    mad = np.median(np.abs(raw_residual - residual_center))
    mad = max(float(mad), 1e-4)
    return 0.6745 * np.abs(raw_residual - residual_center) / mad


def compute_signed_robustness_score(raw_residual: np.ndarray) -> np.ndarray:
    residual_center = np.median(raw_residual)
    mad = np.median(np.abs(raw_residual - residual_center))
    mad = max(float(mad), 1e-4)
    return 0.6745 * (raw_residual - residual_center) / mad


def compute_selected_count(sequence_length: int, p_value: float) -> int:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not 0 < p_value < 1:
        raise ValueError("p_value must satisfy 0 < p_value < 1")
    return max(1, math.ceil(p_value * sequence_length))


def compute_tail_priority(raw_residual: np.ndarray, *, tail: str) -> np.ndarray:
    signed_score = compute_signed_robustness_score(raw_residual)
    if tail == "two_sided":
        return np.abs(signed_score)
    if tail == "upward":
        return np.maximum(signed_score, 0.0)
    raise ValueError(f"Unsupported tail: {tail}")


def select_tail_anomaly_mask(
    raw_residual: np.ndarray,
    *,
    p_value: float,
    tail: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    selected_count = compute_selected_count(len(raw_residual), p_value)
    priority = compute_tail_priority(raw_residual, tail=tail)
    candidate_order = np.argsort(priority, kind="stable")
    selected_idx = candidate_order[-selected_count:]
    mask = np.zeros(len(raw_residual), dtype=bool)
    mask[selected_idx] = True
    if tail == "upward":
        signed_score = compute_signed_robustness_score(raw_residual)
        mask &= signed_score > 0
    return mask, priority, selected_count


def render_target_plot(
    *,
    label: str,
    column: str,
    target_frame: pd.DataFrame,
    components: dict[str, np.ndarray],
    output_path: Path,
    p_value: float,
    tail: str,
) -> None:
    dt = target_frame["dt"]
    observed = target_frame[column].to_numpy(dtype=float)
    trend = components["trend"]
    seasonal = components["seasonal"]
    legacy_mask = components["critical_mask"].astype(bool)
    raw_residual = np.where(legacy_mask, components["anomalies"], components["residual"])
    anomaly_mask, tail_priority, selected_count = select_tail_anomaly_mask(
        raw_residual,
        p_value=p_value,
        tail=tail,
    )
    anomalies = np.where(anomaly_mask, raw_residual, 1.0)
    cleaned_residual = np.where(anomaly_mask, 1.0, raw_residual)
    robustness_score = compute_robustness_score(raw_residual)

    fig, axes = plt.subplots(7, 1, figsize=(16, 24), sharex=True)
    fig.suptitle(
        (
            f"{label.upper()} STAR decomposition ({column}) | "
            f"tail={tail} | p_value={p_value} | selected_count={selected_count}"
        ),
        fontsize=16,
    )

    axes[0].plot(dt, observed, color="#1f77b4", linewidth=1.8)
    axes[0].set_title("Observed series")
    axes[0].set_ylabel("price")

    axes[1].plot(dt, trend, color="#ff7f0e", linewidth=1.8)
    axes[1].set_title("LOWESS trend")
    axes[1].set_ylabel("trend")

    axes[2].plot(dt, seasonal, color="#2ca02c", linewidth=1.8)
    axes[2].axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    axes[2].set_title("Seasonal multiplier")
    axes[2].set_ylabel("seasonal")

    axes[3].plot(dt, raw_residual, color="#9467bd", linewidth=1.5)
    axes[3].axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    axes[3].scatter(
        dt[anomaly_mask],
        raw_residual[anomaly_mask],
        color="#d62728",
        s=18,
        label="selected anomaly residual",
        zorder=3,
    )
    axes[3].set_title("Raw residual multiplier")
    axes[3].set_ylabel("raw residual")
    if anomaly_mask.any():
        axes[3].legend(loc="upper left")

    axes[4].plot(dt, anomalies, color="#d62728", linewidth=1.4)
    axes[4].axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    axes[4].set_title("Anomalies only (normal points forced to 1)")
    axes[4].set_ylabel("anomalies")

    axes[5].plot(dt, robustness_score, color="#17becf", linewidth=1.4, label="robustness score")
    axes[5].plot(dt, tail_priority, color="#111111", linewidth=1.0, linestyle=":", label=f"{tail} priority")
    axes[5].scatter(
        dt[anomaly_mask],
        robustness_score[anomaly_mask],
        color="#d62728",
        s=20,
        label="selected anomalies",
        zorder=3,
    )
    axes[5].set_title("Robustness score and tail-priority ranking")
    axes[5].set_ylabel("score")
    axes[5].legend(loc="upper left")

    axes[6].plot(dt, cleaned_residual, color="#8c564b", linewidth=1.4, label="cleaned residual")
    axes[6].axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    axes[6].scatter(
        dt[anomaly_mask],
        cleaned_residual[anomaly_mask],
        color="#d62728",
        s=20,
        label="masked anomaly -> 1",
        zorder=3,
    )
    mask_axis = axes[6].twinx()
    mask_axis.step(
        dt,
        anomaly_mask.astype(int),
        where="mid",
        color="#111111",
        linewidth=1.2,
        label="selected anomaly mask",
    )
    mask_axis.set_ylim(-0.05, 1.05)
    mask_axis.set_yticks([0, 1])
    mask_axis.set_ylabel("mask")
    axes[6].set_title("Cleaned residual (selected anomalies forced to 1) + mask")
    axes[6].set_ylabel("cleaned residual")
    axes[6].set_xlabel("dt")

    handles, labels = axes[6].get_legend_handles_labels()
    mask_handles, mask_labels = mask_axis.get_legend_handles_labels()
    if handles or mask_handles:
        axes[6].legend(handles + mask_handles, labels + mask_labels, loc="upper left")

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, str]:
    input_path = Path(args.input)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else default_output_dir())
    frame = load_frame(input_path)
    extractor = STARFeatureExtractor(
        season_length=args.season_length,
        lowess_frac=args.lowess_frac,
        lowess_delta=args.lowess_delta,
        p_value=args.p_value,
    )

    output_paths: dict[str, str] = {}
    selected_targets = list(dict.fromkeys(args.targets))
    for label in selected_targets:
        column = TARGET_COLUMNS[label]
        target_frame = extract_target_frame(frame, label=label, column=column)
        components = to_component_series(target_frame, extractor)
        output_path = output_dir / f"{label}_star_decomposition.png"
        render_target_plot(
            label=label,
            column=column,
            target_frame=target_frame,
            components=components,
            output_path=output_path,
            p_value=args.p_value,
            tail=TARGET_TAILS[label],
        )
        output_paths[label] = str(output_path)

    png_outputs = sorted(output_dir.glob("*.png"))
    if len(png_outputs) != len(selected_targets):
        raise RuntimeError(
            f"Expected exactly {len(selected_targets)} PNG outputs in {output_dir}, found {len(png_outputs)}"
        )
    return output_paths


def main() -> None:
    output_paths = run(parse_args())
    print(json.dumps(output_paths, indent=2))


if __name__ == "__main__":
    main()
