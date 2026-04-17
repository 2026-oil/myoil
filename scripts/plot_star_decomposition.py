from __future__ import annotations

import argparse
from typing import cast
import json
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
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import torch
import yaml

from app_config import _coerce_bool, _load_document, _unknown_keys
from plugins.aa_forecast import STARFeatureExtractor
from plugins.retrieval.config import RetrievalStarConfig, normalize_retrieval_detail_payload
from plugins.retrieval.signatures import _resolve_tail_modes, retrieval_timestep_combined_critical_mask


TARGET_COLUMNS = {
    "wti": "Com_CrudeOil",
    "brent": "Com_BrentCrudeOil",
    "gprd_threat": "GPRD_THREAT",
    "dubai": "Com_DubaiOil",
}

DEFAULT_RETRIEVAL_CONFIG = Path("yaml/plugins/retrieval/baseline_retrieval.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot AA-Forecast STAR decomposition (WTI, Brent, Dubai, GPRD_THREAT) from a df.csv-style CSV."
        )
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
        "--retrieval-config",
        type=Path,
        default=DEFAULT_RETRIEVAL_CONFIG,
        help=(
            "Retrieval plugin detail YAML (top-level 'retrieval:' block). "
            "STAR uses retrieval.star (season_length, lowess_*, thresh) and "
            "retrieval.star.anomaly_tails for per-column tail mode."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=sorted(TARGET_COLUMNS),
        default=["wti", "brent"],
        help="Target aliases to plot (default: wti brent).",
    )
    parser.add_argument(
        "--setting",
        type=Path,
        default=ROOT / "yaml" / "setting" / "setting.yaml",
        help="Shared setting YAML (training.input_size used in figure title).",
    )
    parser.add_argument(
        "--cutoff-date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "When set, use the last observation on that calendar date as the window end, then "
            "take the previous training.input_size-1 rows plus that row (exactly input_size "
            "contiguous rows). STAR is fit on that window only. Requires training.input_size in "
            "--setting YAML."
        ),
    )
    parser.add_argument(
        "--hist-exog-cols",
        nargs="*",
        default=[],
        metavar="COL",
        help=(
            "Optional input CSV columns included in retrieval-style combined critical_mask: "
            "OR over target STAR critical (always two_sided) and each hist column (tail from "
            "retrieval.star.anomaly_tails). Must not include the plotted target column."
        ),
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "runs" / f"star-decomposition-{timestamp}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(path: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def load_retrieval_star_config(path: Path) -> RetrievalStarConfig:
    resolved = resolve_repo_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Retrieval plugin YAML not found: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        raise ValueError(
            f"Retrieval plugin YAML must use extension .yaml or .yml, got {suffix!r}: {resolved}"
        )
    payload = _load_document(resolved, "yaml")
    plugin_cfg = normalize_retrieval_detail_payload(
        payload,
        unknown_keys=_unknown_keys,
        coerce_bool=_coerce_bool,
    )
    return plugin_cfg.star


def align_frame_to_target_window(full_frame: pd.DataFrame, target_window: pd.DataFrame) -> pd.DataFrame:
    """Rows from ``full_frame`` matching ``target_window['dt']`` order and length."""
    want = target_window["dt"].to_numpy()
    aligned = full_frame[full_frame["dt"].isin(want)].sort_values("dt").reset_index(drop=True)
    if len(aligned) != len(target_window):
        raise ValueError(
            f"After aligning on target window dates, expected {len(target_window)} rows, "
            f"found {len(aligned)} (check duplicate or missing 'dt' in input CSV)."
        )
    if not (aligned["dt"].to_numpy() == target_window["dt"].to_numpy()).all():
        raise ValueError("Date order mismatch between target window and aligned full-frame slice.")
    return aligned


def tail_for_column(column: str, star: RetrievalStarConfig) -> str:
    """Match ``plugins.retrieval.runtime`` hist-exog tail assignment for a single series name."""
    if column in star.anomaly_tails.get("two_sided", ()):
        return "two_sided"
    if column in star.anomaly_tails.get("upward", ()):
        return "upward"
    return "two_sided"


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


def slice_input_window_ending_on_cutoff_calendar_day(
    target_frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    input_size: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """``input_size`` contiguous rows ending at the last row on the cutoff *calendar* date.

    Example: cutoff ``2026-02-23`` → window ends at the last timestamp in the series whose
    calendar date is 2026-02-23 (inclusive of that day), and includes the ``input_size-1`` prior
    rows in the same frame (64 points = that day plus 63 earlier observations).
    """
    if input_size < 2:
        raise ValueError("input_size must be at least 2 for cutoff-window plotting.")
    cutoff_day = pd.Timestamp(cutoff).normalize()
    series_days = target_frame["dt"].dt.normalize()
    on_cutoff = series_days == cutoff_day
    if not on_cutoff.any():
        raise ValueError(
            f"No observation on cutoff calendar date {cutoff_day.date()} in the target series; "
            "cannot anchor a window that includes that day."
        )
    anchor_pos = int(np.flatnonzero(on_cutoff.to_numpy())[-1])
    start_pos = anchor_pos - input_size + 1
    if start_pos < 0:
        raise ValueError(
            f"Need {input_size} rows ending on {cutoff_day.date()}, but only {anchor_pos + 1} "
            "rows exist on or before that anchor in the series."
        )
    window = target_frame.iloc[start_pos : anchor_pos + 1].reset_index(drop=True)
    if len(window) != input_size:
        raise ValueError(
            f"Internal error: expected window length {input_size}, got {len(window)}."
        )
    marker_ts = pd.Timestamp(window["dt"].iloc[-1])
    return window, marker_ts


def load_training_input_size(setting_path: Path) -> int | None:
    if not setting_path.is_file():
        return None
    with setting_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Shared setting YAML must be a mapping: {setting_path}")
    training = payload.get("training")
    if not isinstance(training, dict):
        return None
    size = training.get("input_size")
    if size is None:
        return None
    return int(size)


def extract_star_components(
    target_frame: pd.DataFrame,
    extractor: STARFeatureExtractor,
    *,
    tail: str,
) -> dict[str, np.ndarray]:
    values = target_frame.iloc[:, 1].to_numpy(dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Target series must be one-dimensional and non-empty.")
    tensor = torch.as_tensor(values, dtype=torch.float32).view(1, -1, 1)
    tail_modes = (tail,)
    with torch.no_grad():
        components = extractor(tensor, tail_modes=tail_modes)
    return {
        name: component.detach().cpu().numpy().reshape(-1)
        for name, component in components.items()
    }


def _scatter_cutoff_marker(ax: Axes, x: pd.Timestamp, y: float) -> None:
    """Hollow (transparent face) marker shared across panels for the cutoff/end-of-window time."""
    ax.scatter(
        [x],
        [y],
        s=120,
        facecolors="none",
        edgecolors="#111111",
        linewidths=2.0,
        zorder=6,
    )


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


def compute_tail_score(raw_residual: np.ndarray, *, tail: str) -> np.ndarray:
    signed_score = compute_signed_robustness_score(raw_residual)
    if tail == "two_sided":
        return np.abs(signed_score)
    if tail == "upward":
        return signed_score
    raise ValueError(f"Unsupported tail: {tail}")


def select_tail_anomaly_mask(
    raw_residual: np.ndarray,
    *,
    thresh: float,
    tail: str,
) -> tuple[np.ndarray, np.ndarray]:
    if thresh < 0:
        raise ValueError("thresh must be non-negative")
    tail_score = compute_tail_score(raw_residual, tail=tail)
    if tail == "two_sided":
        mask = tail_score > thresh
    elif tail == "upward":
        mask = tail_score > thresh
    else:
        raise ValueError(f"Unsupported tail: {tail}")
    return mask, tail_score


def render_target_plot(
    *,
    label: str,
    column: str,
    target_frame: pd.DataFrame,
    components: dict[str, np.ndarray],
    output_path: Path,
    thresh: float,
    tail: str,
    cutoff: pd.Timestamp | None = None,
    input_size: int | None = None,
    cutoff_marker_ts: pd.Timestamp | None = None,
    retrieval_config_name: str | None = None,
    retrieval_combined_critical: np.ndarray | None = None,
    hist_exog_title: str | None = None,
) -> None:
    dt = target_frame["dt"]
    observed = target_frame[column].to_numpy(dtype=float)
    trend = components["trend"]
    seasonal = components["seasonal"]
    raw_residual = components["anomalies"] * components["residual"]
    robust_score_abs = components["robust_score_abs"].astype(float).reshape(-1)
    anomaly_mask, tail_score = select_tail_anomaly_mask(
        raw_residual,
        thresh=thresh,
        tail=tail,
    )
    anomalies = np.where(anomaly_mask, raw_residual, 1.0)
    cleaned_residual = np.where(anomaly_mask, 1.0, raw_residual)
    robustness_score = compute_robustness_score(raw_residual)

    title_core = (
        f"{label.upper()} STAR decomposition ({column}) | tail={tail} | thresh={thresh}"
    )
    if input_size is not None:
        title_core += f" | input_size={input_size}"
    if cutoff is not None:
        title_core += f" | cutoff={cutoff.date()}"
    if retrieval_config_name:
        title_core += f" | retrieval={retrieval_config_name}"
    if hist_exog_title:
        title_core += f" | hist_exog={hist_exog_title}"

    if cutoff is not None:
        if cutoff_marker_ts is None:
            raise ValueError("cutoff_marker_ts is required when cutoff is set.")
        _render_target_plot_input_window(
            dt=dt,
            observed=observed,
            trend=trend,
            seasonal=seasonal,
            raw_residual=raw_residual,
            anomaly_mask=anomaly_mask,
            robust_score_abs=robust_score_abs,
            thresh=thresh,
            output_path=output_path,
            title=title_core,
            cutoff_marker_ts=cutoff_marker_ts,
            retrieval_combined_critical=retrieval_combined_critical,
        )
        return

    n_panels = 8 if retrieval_combined_critical is not None else 7
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.45 * n_panels), sharex=True)
    fig.suptitle(title_core, fontsize=16)

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
        label="threshold anomaly residual",
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

    axes[5].plot(
        dt,
        robustness_score,
        color="#17becf",
        linewidth=1.4,
        label="robustness score",
    )
    axes[5].plot(
        dt,
        tail_score,
        color="#111111",
        linewidth=1.0,
        linestyle=":",
        label=f"{tail} score",
    )
    axes[5].axhline(
        thresh,
        color="#d62728",
        linewidth=0.9,
        linestyle="--",
        label=f"thresh={thresh}",
    )
    axes[5].scatter(
        dt[anomaly_mask],
        robustness_score[anomaly_mask],
        color="#d62728",
        s=20,
        label="threshold anomalies",
        zorder=3,
    )
    axes[5].set_title("Robustness score and tail score")
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
    axes[6].set_title("Cleaned residual (threshold anomalies forced to 1)")
    axes[6].set_ylabel("cleaned residual")

    handles, labels_leg = axes[6].get_legend_handles_labels()
    if handles:
        axes[6].legend(handles, labels_leg, loc="upper left")

    last_ax = axes[6]
    if retrieval_combined_critical is not None:
        comb = retrieval_combined_critical.astype(float)
        axes[7].step(dt.to_numpy(), comb, where="mid", color="#1f77b4", linewidth=1.2, label="retrieval combined")
        axes[7].set_ylim(-0.05, 1.05)
        axes[7].set_yticks([0.0, 1.0])
        axes[7].set_title("critical_mask retrieval (target two_sided OR hist exog)")
        axes[7].set_ylabel("mask")
        axes[7].legend(loc="upper left")
        last_ax = axes[7]
    last_ax.set_xlabel("dt")

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_target_plot_input_window(
    *,
    dt: pd.Series,
    observed: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    raw_residual: np.ndarray,
    anomaly_mask: np.ndarray,
    robust_score_abs: np.ndarray,
    thresh: float,
    output_path: Path,
    title: str,
    cutoff_marker_ts: pd.Timestamp,
    retrieval_combined_critical: np.ndarray | None = None,
) -> None:
    line_kw = {"linewidth": 1.8}
    n_panels = 5 + (1 if retrieval_combined_critical is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.45 * n_panels), sharex=True)
    fig.suptitle(title, fontsize=16)

    axes[0].plot(dt, observed, color="#1f77b4", linewidth=1.8)
    axes[0].set_title("Observed series (input window)")
    axes[0].set_ylabel("price")
    _scatter_cutoff_marker(axes[0], cutoff_marker_ts, float(observed[-1]))

    axes[1].plot(dt, trend, color="#ff7f0e", **line_kw)
    axes[1].set_title("LOWESS trend")
    axes[1].set_ylabel("trend")
    _scatter_cutoff_marker(axes[1], cutoff_marker_ts, float(trend[-1]))

    axes[2].plot(dt, seasonal, color="#2ca02c", **line_kw)
    axes[2].axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    axes[2].set_title("Seasonal multiplier")
    axes[2].set_ylabel("seasonal")
    _scatter_cutoff_marker(axes[2], cutoff_marker_ts, float(seasonal[-1]))

    axes[3].plot(dt, raw_residual, color="#9467bd", linewidth=1.5)
    axes[3].axhline(1.0, color="#444444", linewidth=0.8, linestyle="--")
    if anomaly_mask.any():
        axes[3].scatter(
            dt[anomaly_mask],
            raw_residual[anomaly_mask],
            color="#d62728",
            s=18,
            label="threshold anomaly residual",
            zorder=3,
        )
    axes[3].set_title("Raw residual multiplier")
    axes[3].set_ylabel("raw residual")
    if anomaly_mask.any():
        axes[3].legend(loc="upper left")
    _scatter_cutoff_marker(axes[3], cutoff_marker_ts, float(raw_residual[-1]))

    axes[4].plot(dt, robust_score_abs, color="#17becf", linewidth=1.4)
    axes[4].axhline(thresh, color="#d62728", linewidth=0.9, linestyle="--", label=f"thresh={thresh}")
    axes[4].set_title("robust_score_abs (STAR)")
    axes[4].set_ylabel("robust score")
    axes[4].legend(loc="upper left")
    _scatter_cutoff_marker(axes[4], cutoff_marker_ts, float(robust_score_abs[-1]))

    last_ax = axes[4]
    if retrieval_combined_critical is not None:
        comb = retrieval_combined_critical.astype(float)
        axes[5].step(
            dt.to_numpy(),
            comb,
            where="mid",
            color="#1f77b4",
            linewidth=1.2,
            label="retrieval combined",
        )
        axes[5].set_ylim(-0.05, 1.05)
        axes[5].set_yticks([0.0, 1.0])
        axes[5].set_title("critical_mask retrieval (target two_sided OR hist exog)")
        axes[5].set_ylabel("mask")
        axes[5].legend(loc="upper left")
        _scatter_cutoff_marker(axes[5], cutoff_marker_ts, float(comb[-1]))
        last_ax = axes[5]
    last_ax.set_xlabel("dt")

    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, str]:
    input_path = Path(args.input)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else default_output_dir())
    frame = load_frame(input_path)
    setting_path = Path(args.setting)
    input_size = load_training_input_size(setting_path)
    cutoff_ts: pd.Timestamp | None = None
    if args.cutoff_date is not None:
        cutoff_ts = pd.Timestamp(args.cutoff_date)
        if input_size is None:
            raise ValueError(
                "cutoff-date plotting requires training.input_size in the YAML passed via "
                f"--setting (missing or invalid file: {setting_path})."
            )

    star_cfg = load_retrieval_star_config(Path(args.retrieval_config))
    retrieval_config_name = Path(args.retrieval_config).name
    hist_cols_tuple = tuple(dict.fromkeys(list(args.hist_exog_cols or [])))
    hist_exog_title = ",".join(hist_cols_tuple) if hist_cols_tuple else None
    extractor = STARFeatureExtractor(
        season_length=star_cfg.season_length,
        lowess_frac=star_cfg.lowess_frac,
        lowess_delta=star_cfg.lowess_delta,
        thresh=star_cfg.thresh,
    )

    output_paths: dict[str, str] = {}
    selected_targets = list(dict.fromkeys(args.targets))
    for label in selected_targets:
        column = TARGET_COLUMNS[label]
        tail = tail_for_column(column, star_cfg)
        target_frame = extract_target_frame(frame, label=label, column=column)
        cutoff_marker_ts: pd.Timestamp | None = None
        if cutoff_ts is not None:
            target_frame, cutoff_marker_ts = slice_input_window_ending_on_cutoff_calendar_day(
                target_frame,
                cutoff=cutoff_ts,
                input_size=cast(int, input_size),
            )
        components = extract_star_components(
            target_frame,
            extractor,
            tail=tail,
        )
        retrieval_combined_critical: np.ndarray | None = None
        if hist_cols_tuple:
            for name in hist_cols_tuple:
                if name not in frame.columns:
                    raise ValueError(f"--hist-exog-cols: column not in input CSV: {name}")
            if column in hist_cols_tuple:
                raise ValueError(
                    f"--hist-exog-cols must not include the plotted target column {column!r}."
                )
            full_window = align_frame_to_target_window(frame, target_frame)
            hist_modes = _resolve_tail_modes(hist_cols_tuple, star_cfg.anomaly_tails)
            retrieval_combined_critical = retrieval_timestep_combined_critical_mask(
                star=extractor,
                window_df=full_window,
                target_col=column,
                hist_exog_cols=hist_cols_tuple,
                hist_exog_tail_modes=hist_modes,
            )
            if retrieval_combined_critical.shape[0] != len(target_frame):
                raise ValueError(
                    "Internal error: combined retrieval mask length does not match target window."
                )
        output_path = output_dir / f"{label}_star_decomposition.png"
        render_target_plot(
            label=label,
            column=column,
            target_frame=target_frame,
            components=components,
            output_path=output_path,
            thresh=star_cfg.thresh,
            tail=tail,
            cutoff=cutoff_ts,
            input_size=input_size,
            cutoff_marker_ts=cutoff_marker_ts,
            retrieval_config_name=retrieval_config_name,
            retrieval_combined_critical=retrieval_combined_critical,
            hist_exog_title=hist_exog_title,
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
