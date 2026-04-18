"""Generate target-specific F3 last-fold comparison plots.

This script reads the completed runs under ``runs/F3`` and writes six PNGs:
for each target (WTI / Brent / Dubai), one full-input plot and one
``window_16`` plot.

Legend labels follow the requested convention:
- ``AA-GRU`` / ``AA-GRU+Retrival``
- ``AA-Informer`` / ``AA-Informer+Retrival``
- ``AA-TimeXer`` / ``AA-TimeXer+Retrival``
- ``GRU`` / ``GRU+Retrival``
- ``Informer`` / ``Informer+Retrival``
- ``TimeXer`` / ``TimeXer+Retrival``

Example:
  UV_CACHE_DIR=/tmp/uvcache uv run python scripts/plot_f3_last_fold_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import runtime_support.runner as runtime  # noqa: E402

_TARGETS = ("wti", "brent", "dubai")
_AA_BACKBONES = ("gru", "informer", "timexer")
_BASELINE_MODELS = ("GRU", "Informer", "TimeXer")
_MODEL_CHOICES = ("gru", "informer", "timexer")
_LABEL_SET_CHOICES = ("all", "paper_plain_vs_all_aa_ret", "aa_timexer_pair")
_COMPARISON_VARIANT_CHOICES = ("delta_panel", "zoom", "pointplot")


def _model_label_order(selected_backbones: tuple[str, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for backbone in selected_backbones:
        labels.append(_aa_label(backbone, retrieval=False))
    for backbone in selected_backbones:
        labels.append(_aa_label(backbone, retrieval=True))
    for backbone in selected_backbones:
        labels.append(_baseline_label(_title_backbone(backbone), retrieval=False))
    for backbone in selected_backbones:
        labels.append(_baseline_label(_title_backbone(backbone), retrieval=True))
    return tuple(labels)




def _resolve_label_order(
    selected_backbones: tuple[str, ...],
    *,
    label_set: str,
) -> tuple[str, ...]:
    if label_set == "all":
        return _model_label_order(selected_backbones)
    if label_set == "paper_plain_vs_all_aa_ret":
        if selected_backbones != _MODEL_CHOICES:
            raise ValueError(
                "paper_plain_vs_all_aa_ret requires the default model families: gru informer timexer"
            )
        return (
            "GRU",
            "Informer",
            "TimeXer",
            "AA-GRU+Retrival",
            "AA-Informer+Retrival",
            "AA-TimeXer+Retrival",
        )
    if label_set == "aa_timexer_pair":
        if "timexer" not in selected_backbones:
            raise ValueError("aa_timexer_pair requires timexer to be selected")
        return (
            "AA-TimeXer",
            "AA-TimeXer+Retrival",
        )
    raise ValueError(f"unsupported label set: {label_set}")

def _parse_selected_backbones(raw_models: Iterable[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for raw in raw_models:
        normalized = raw.strip().lower()
        if not normalized:
            continue
        if normalized not in _MODEL_CHOICES:
            raise ValueError(f"unsupported model selection: {raw}")
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("at least one model must be selected")
    return tuple(selected)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate F3 last-fold comparison plots with custom labels.",
    )
    parser.add_argument(
        "--f3-root",
        type=Path,
        default=Path("runs/F3"),
        help="Root directory containing the completed F3 runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/F3/_custom_last_fold_plots"),
        help="Directory where the six PNG outputs will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(_MODEL_CHOICES),
        choices=list(_MODEL_CHOICES),
        help="Subset of model families to include (default: all).",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="*",
        type=int,
        default=[8, 16],
        help="Additional observed-history window sizes to render besides the full-input figure.",
    )
    parser.add_argument(
        "--label-set",
        choices=list(_LABEL_SET_CHOICES),
        default="all",
        help="Which display-label subset to render.",
    )
    parser.add_argument(
        "--comparison-variants",
        nargs="*",
        choices=list(_COMPARISON_VARIANT_CHOICES),
        default=[],
        help="Additional comparison-focused plot variants to write separately.",
    )
    args = parser.parse_args()
    args.models = _parse_selected_backbones(args.models)
    return args


def _title_backbone(name: str) -> str:
    lower = name.lower()
    if lower == "gru":
        return "GRU"
    if lower == "timexer":
        return "TimeXer"
    if lower == "informer":
        return "Informer"
    raise ValueError(f"unsupported backbone label: {name}")


def _aa_label(backbone: str, *, retrieval: bool) -> str:
    suffix = "+Retrival" if retrieval else ""
    return f"AA-{_title_backbone(backbone)}{suffix}"


def _baseline_label(model_name: str, *, retrieval: bool) -> str:
    suffix = "+Retrival" if retrieval else ""
    return f"{model_name}{suffix}"


def _discover_target_run_roots(
    f3_root: Path,
    *,
    target: str,
    selected_backbones: tuple[str, ...],
) -> dict[str, Path]:
    target_root_map: dict[str, Path] = {}
    for backbone in selected_backbones:
        run_root = f3_root / f"feature_set_aaforecast_{target}_aaforecast_{backbone}-ret"
        if not run_root.is_dir():
            raise FileNotFoundError(f"missing AAForecast run for {target}/{backbone}: {run_root}")
        target_root_map[f"aa:{backbone}"] = run_root
    baseline_root = f3_root / f"feature_set_aaforecast_{target}_brentoil_baseline-ret_gru_informer"
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"missing baseline run for {target}: {baseline_root}")
    target_root_map["baseline"] = baseline_root
    return target_root_map


def _load_run_result(run_root: Path) -> pd.DataFrame:
    result_path = run_root / "summary" / "result.csv"
    if not result_path.is_file():
        raise FileNotFoundError(f"missing summary/result.csv: {result_path}")
    frame = pd.read_csv(result_path)
    if frame.empty:
        raise ValueError(f"empty summary/result.csv: {result_path}")
    required = {"model", "fold_idx", "ds", "y_hat", "train_end_ds", "cutoff", "horizon_step"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"summary/result.csv missing columns {sorted(missing)}: {result_path}")
    return frame


def _last_fold_only(frame: pd.DataFrame, *, run_root: Path) -> pd.DataFrame:
    normalized = runtime._normalize_summary_window_frame(
        frame,
        frame_name=f"summary/result.csv for {run_root}",
    )
    last_fold = int(normalized["fold_idx"].max())
    scoped = normalized[normalized["fold_idx"] == last_fold].copy()
    if scoped.empty:
        raise ValueError(f"no last-fold rows found in {run_root / 'summary' / 'result.csv'}")
    scoped["_summary_source_root"] = str(run_root)
    return scoped


def _timestamp_slug(value: object) -> str:
    timestamp = runtime._normalize_summary_timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"invalid cutoff timestamp: {value}")
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d_%H%M%S")


def _coerce_prediction_values(values: Iterable[object], *, label: str) -> list[float]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    if series.isna().any():
        raise ValueError(f"{label} contains non-numeric prediction values")
    return [float(value) for value in series.tolist()]


def _clone_prediction_series(
    frame: pd.DataFrame,
    *,
    display_label: str,
    prediction_values: Iterable[object],
) -> pd.DataFrame:
    cloned = frame.copy()
    values = _coerce_prediction_values(prediction_values, label=display_label)
    if len(values) != len(cloned):
        raise ValueError(
            f"{display_label} prediction length mismatch: expected {len(cloned)}, got {len(values)}"
        )
    cloned["display_label"] = display_label
    cloned["y_hat"] = values
    return cloned


def _baseline_retrieval_payload_path(run_root: Path, *, model_name: str, cutoff: object) -> Path:
    slug = _timestamp_slug(cutoff)
    return (
        run_root
        / "scheduler"
        / "workers"
        / model_name
        / "retrieval"
        / f"retrieval_summary_{slug}.json"
    )


def _load_baseline_base_predictions(run_root: Path, *, model_name: str, cutoff: object) -> list[float]:
    payload_path = _baseline_retrieval_payload_path(run_root, model_name=model_name, cutoff=cutoff)
    if not payload_path.is_file():
        raise FileNotFoundError(f"missing baseline retrieval payload: {payload_path}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    base_prediction = payload.get("base_prediction")
    if not isinstance(base_prediction, list):
        raise ValueError(f"retrieval payload missing base_prediction list: {payload_path}")
    return _coerce_prediction_values(base_prediction, label=f"base_prediction in {payload_path}")


def _build_aa_series(run_root: Path, *, backbone: str) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    scoped = _last_fold_only(_load_run_result(run_root), run_root=run_root)
    if "aaforecast_base_prediction" not in scoped.columns:
        raise ValueError(f"AAForecast run missing aaforecast_base_prediction column: {run_root}")
    plain = _clone_prediction_series(
        scoped,
        display_label=_aa_label(backbone, retrieval=False),
        prediction_values=scoped["aaforecast_base_prediction"],
    )
    retrieval = _clone_prediction_series(
        scoped,
        display_label=_aa_label(backbone, retrieval=True),
        prediction_values=scoped["y_hat"],
    )
    train_end_values = pd.Series(scoped["train_end_ds"]).dropna().map(runtime._normalize_summary_timestamp).tolist()
    return plain, retrieval, train_end_values


def _build_baseline_series(
    run_root: Path,
    *,
    target: str,
    selected_backbones: tuple[str, ...],
) -> tuple[list[pd.DataFrame], list[pd.Timestamp]]:
    scoped = _last_fold_only(_load_run_result(run_root), run_root=run_root)
    selected_models = tuple(_title_backbone(backbone) for backbone in selected_backbones)
    available_models = set(scoped["model"].astype(str).unique())
    missing_baseline = [model for model in selected_models if model not in available_models]
    if missing_baseline:
        raise ValueError(f"baseline run for {target} is missing models {missing_baseline}: {run_root}")

    series_frames: list[pd.DataFrame] = []
    train_end_values = pd.Series(scoped["train_end_ds"]).dropna().map(runtime._normalize_summary_timestamp).tolist()
    for model_name in selected_models:
        model_frame = scoped[scoped["model"].astype(str) == model_name].copy()
        model_frame = model_frame.sort_values(["cutoff", "horizon_step", "ds"], kind="stable").reset_index(drop=True)
        if model_frame.empty:
            raise ValueError(f"baseline run for {target} has no rows for model {model_name}: {run_root}")
        plain_prediction_values: list[float] = []
        for cutoff, cutoff_frame in model_frame.groupby("cutoff", sort=False):
            base_prediction = _load_baseline_base_predictions(run_root, model_name=model_name, cutoff=cutoff)
            ordered = cutoff_frame.sort_values(["horizon_step", "ds"], kind="stable")
            horizon_steps = pd.to_numeric(ordered["horizon_step"], errors="coerce")
            if horizon_steps.isna().any():
                raise ValueError(f"baseline horizon_step contains invalid values for {model_name} cutoff={cutoff}")
            for horizon_step in horizon_steps.astype(int).tolist():
                if horizon_step < 1 or horizon_step > len(base_prediction):
                    raise ValueError(
                        f"base_prediction length mismatch for {model_name} cutoff={cutoff}: horizon_step={horizon_step}, len={len(base_prediction)}"
                    )
                plain_prediction_values.append(float(base_prediction[horizon_step - 1]))
        plain = _clone_prediction_series(
            model_frame,
            display_label=_baseline_label(model_name, retrieval=False),
            prediction_values=plain_prediction_values,
        )
        retrieval = _clone_prediction_series(
            model_frame,
            display_label=_baseline_label(model_name, retrieval=True),
            prediction_values=model_frame["y_hat"],
        )
        series_frames.extend([plain, retrieval])
    return series_frames, train_end_values


def _labelled_last_fold_frame(
    *,
    target: str,
    run_roots: dict[str, Path],
    selected_backbones: tuple[str, ...],
    label_set: str = "all",
) -> tuple[pd.DataFrame, Path]:
    frames: list[pd.DataFrame] = []
    representative_run_root: Path | None = None
    train_end_values: list[pd.Timestamp] = []

    for backbone in selected_backbones:
        run_root = run_roots[f"aa:{backbone}"]
        plain, retrieval, aa_train_end_values = _build_aa_series(run_root, backbone=backbone)
        frames.extend([plain, retrieval])
        train_end_values.extend(aa_train_end_values)
        representative_run_root = run_root if representative_run_root is None else representative_run_root

    baseline_root = run_roots["baseline"]
    baseline_frames, baseline_train_end_values = _build_baseline_series(
        baseline_root,
        target=target,
        selected_backbones=selected_backbones,
    )
    frames.extend(baseline_frames)
    train_end_values.extend(baseline_train_end_values)
    representative_run_root = representative_run_root or baseline_root

    if representative_run_root is None:
        raise ValueError(f"no run roots discovered for target={target}")
    if not frames:
        raise ValueError(f"no result frames discovered for target={target}")

    unique_train_end = pd.Series(train_end_values).dropna().drop_duplicates()
    if unique_train_end.empty:
        raise ValueError(f"target={target} did not provide any train_end_ds values")
    if len(unique_train_end) != 1:
        raise ValueError(
            f"target={target} requires exactly one shared train_end_ds across all runs; got {unique_train_end.tolist()}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["ds"] = pd.to_datetime(combined["ds"], errors="coerce").map(runtime._normalize_summary_timestamp)
    combined["y"] = pd.to_numeric(combined.get("y"), errors="coerce")
    combined["y_hat"] = pd.to_numeric(combined["y_hat"], errors="coerce")
    combined = combined.dropna(subset=["ds", "y_hat"]).copy()
    if combined.empty:
        raise ValueError(f"target={target} has no plottable last-fold rows")

    expected_labels = _resolve_label_order(selected_backbones, label_set=label_set)
    observed_labels = set(combined["display_label"].astype(str).unique())
    missing_labels = [label for label in expected_labels if label not in observed_labels]
    if missing_labels:
        raise ValueError(f"target={target} missing expected labels: {missing_labels}")
    combined = combined[combined["display_label"].astype(str).isin(expected_labels)].copy()
    combined["display_label"] = pd.Categorical(
        combined["display_label"], categories=list(expected_labels), ordered=True
    )
    combined = combined.sort_values(["display_label", "cutoff", "horizon_step", "ds"], kind="stable").reset_index(drop=True)
    return combined, representative_run_root


def _ordered_labels(combined: pd.DataFrame) -> list[str]:
    return combined["display_label"].astype(str).drop_duplicates().tolist()


def _reference_label(ordered_labels: list[str]) -> str:
    if "AA-GRU+Retrival" in ordered_labels:
        return "AA-GRU+Retrival"
    for label in ordered_labels:
        if label.startswith("AA-") and label.endswith("+Retrival"):
            return label
    return ordered_labels[0]


def _history_window_label(history_steps_override: int | None) -> str:
    return "full_input_window" if history_steps_override is None else f"window_{int(history_steps_override)}"


def _load_overlay_frames(
    combined: pd.DataFrame,
    *,
    run_root: Path,
    history_steps_override: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_actual_frame, output_actual_frame = runtime._summary_overlay_actual_frames(
        run_root,
        combined,
        history_steps_override=history_steps_override,
    )
    actual_anchor_frame = (
        input_actual_frame.tail(1)[["ds", "y"]].copy()
        if not input_actual_frame.empty
        else pd.DataFrame(columns=["ds", "y"])
    )
    return input_actual_frame, output_actual_frame, actual_anchor_frame


def _model_prediction_frames(
    combined: pd.DataFrame,
    *,
    actual_anchor_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    model_frames: dict[str, pd.DataFrame] = {}
    for label in _ordered_labels(combined):
        part = combined[combined["display_label"].astype(str) == label].copy()
        part = part.sort_values(["cutoff", "horizon_step", "ds"], kind="stable").reset_index(drop=True)
        if "y" in part.columns:
            part = part[part["y"].notna()].reset_index(drop=True)
        if part.empty:
            continue
        connected = _connected_plot_frame(actual_anchor_frame, part, value_col="y_hat")
        if not connected.empty:
            model_frames[label] = connected
    return model_frames


def _variant_output_path(output_path: Path, variant: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{variant}{output_path.suffix}")


def _connected_plot_frame(anchor_frame: pd.DataFrame, frame: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    plot_frame = frame[["ds", value_col]].copy()
    plot_frame["ds"] = pd.to_datetime(plot_frame["ds"], errors="coerce")
    plot_frame[value_col] = pd.to_numeric(plot_frame[value_col], errors="coerce")
    plot_frame = plot_frame.dropna(subset=["ds", value_col]).reset_index(drop=True)
    if anchor_frame.empty or plot_frame.empty:
        return plot_frame
    anchor = anchor_frame.rename(columns={"y": value_col})
    return pd.concat([anchor, plot_frame], ignore_index=True)


def _plot_target_last_fold(
    combined: pd.DataFrame,
    *,
    target: str,
    run_root: Path,
    output_path: Path,
    history_steps_override: int | None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_actual_frame, output_actual_frame, actual_anchor_frame = _load_overlay_frames(
        combined,
        run_root=run_root,
        history_steps_override=history_steps_override,
    )
    model_frames = _model_prediction_frames(combined, actual_anchor_frame=actual_anchor_frame)

    fig, ax = plt.subplots(figsize=(12, 6))
    if not input_actual_frame.empty:
        ax.plot(
            input_actual_frame["ds"],
            input_actual_frame["y"],
            label="Observed history",
            linewidth=2.0,
            color="black",
            marker="o",
            markersize=4,
        )
    if not output_actual_frame.empty and output_actual_frame["y"].notna().any():
        observed_output = _connected_plot_frame(actual_anchor_frame, output_actual_frame, value_col="y")
        ax.plot(
            observed_output["ds"],
            observed_output["y"],
            label="Observed future",
            linewidth=1.8,
            linestyle="--",
            color="dimgray",
            marker="o",
            markersize=4,
        )

    for label, connected_model_frame in model_frames.items():
        prediction_point_indices = list(range(1, len(connected_model_frame)))
        ax.plot(
            connected_model_frame["ds"],
            connected_model_frame["y_hat"],
            label=label,
            linewidth=1.8,
            marker="o",
            markersize=5,
            markevery=prediction_point_indices if prediction_point_indices else None,
        )

    window_label = _history_window_label(history_steps_override)
    ax.set_title(f"{target.upper()} last fold predictions ({window_label})")
    ax.set_xlabel("ds")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_target_last_fold_delta_panel(
    combined: pd.DataFrame,
    *,
    target: str,
    run_root: Path,
    output_path: Path,
    history_steps_override: int | None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_actual_frame, output_actual_frame, actual_anchor_frame = _load_overlay_frames(
        combined,
        run_root=run_root,
        history_steps_override=history_steps_override,
    )
    model_frames = _model_prediction_frames(combined, actual_anchor_frame=actual_anchor_frame)
    ordered_labels = list(model_frames)
    reference_label = _reference_label(ordered_labels)
    reference = model_frames[reference_label].iloc[1:].reset_index(drop=True)

    fig, (top_ax, bottom_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    if not input_actual_frame.empty:
        top_ax.plot(input_actual_frame["ds"], input_actual_frame["y"], label="Observed history", linewidth=2.0, color="black", marker="o", markersize=4)
    if not output_actual_frame.empty and output_actual_frame["y"].notna().any():
        observed_output = _connected_plot_frame(actual_anchor_frame, output_actual_frame, value_col="y")
        top_ax.plot(observed_output["ds"], observed_output["y"], label="Observed future", linewidth=1.8, linestyle="--", color="dimgray", marker="o", markersize=4)
    for label, frame in model_frames.items():
        prediction_point_indices = list(range(1, len(frame)))
        top_ax.plot(frame["ds"], frame["y_hat"], label=label, linewidth=1.8, marker="o", markersize=5, markevery=prediction_point_indices if prediction_point_indices else None)
    bottom_ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
    for label, frame in model_frames.items():
        aligned = frame.iloc[1:].reset_index(drop=True)
        delta = pd.to_numeric(aligned["y_hat"], errors="coerce") - pd.to_numeric(reference["y_hat"], errors="coerce")
        bottom_ax.plot(aligned["ds"], delta, label=f"{label} - {reference_label}", linewidth=1.8, marker="o", markersize=5)
    top_ax.set_title(f"{target.upper()} last fold predictions ({_history_window_label(history_steps_override)})")
    top_ax.set_ylabel("y")
    top_ax.grid(True, alpha=0.3)
    top_ax.legend(loc="best", fontsize=8)
    bottom_ax.set_ylabel("delta")
    bottom_ax.set_xlabel("ds")
    bottom_ax.grid(True, alpha=0.3)
    bottom_ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_target_last_fold_zoom(
    combined: pd.DataFrame,
    *,
    target: str,
    run_root: Path,
    output_path: Path,
    history_steps_override: int | None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_actual_frame, output_actual_frame, actual_anchor_frame = _load_overlay_frames(
        combined,
        run_root=run_root,
        history_steps_override=history_steps_override,
    )
    model_frames = _model_prediction_frames(combined, actual_anchor_frame=actual_anchor_frame)
    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 2]})
    for axis, zoom_only in ((left_ax, False), (right_ax, True)):
        if not zoom_only and not input_actual_frame.empty:
            axis.plot(input_actual_frame["ds"], input_actual_frame["y"], label="Observed history", linewidth=2.0, color="black", marker="o", markersize=4)
        if not output_actual_frame.empty and output_actual_frame["y"].notna().any():
            observed_output = _connected_plot_frame(actual_anchor_frame, output_actual_frame, value_col="y")
            axis.plot(observed_output["ds"], observed_output["y"], label="Observed future", linewidth=1.8, linestyle="--", color="dimgray", marker="o", markersize=4)
        for label, frame in model_frames.items():
            prediction_point_indices = list(range(1, len(frame)))
            axis.plot(frame["ds"], frame["y_hat"], label=label, linewidth=1.8, marker="o", markersize=5, markevery=prediction_point_indices if prediction_point_indices else None)
        axis.grid(True, alpha=0.3)
    future_dates = pd.to_datetime(output_actual_frame["ds"], errors="coerce") if not output_actual_frame.empty else pd.Series(dtype='datetime64[ns]')
    if not future_dates.empty:
        right_ax.set_xlim(future_dates.min() - pd.Timedelta(days=3), future_dates.max() + pd.Timedelta(days=3))
        future_values = []
        if not output_actual_frame.empty:
            future_values.extend(pd.to_numeric(output_actual_frame["y"], errors="coerce").dropna().tolist())
        for frame in model_frames.values():
            future_values.extend(pd.to_numeric(frame.iloc[1:]["y_hat"], errors="coerce").dropna().tolist())
        if future_values:
            ymin, ymax = min(future_values), max(future_values)
            pad = (ymax - ymin) * 0.15 if ymax > ymin else 1.0
            right_ax.set_ylim(ymin - pad, ymax + pad)
    left_ax.set_title(f"{target.upper()} full context ({_history_window_label(history_steps_override)})")
    right_ax.set_title("Forecast zoom")
    left_ax.set_xlabel("ds")
    right_ax.set_xlabel("ds")
    left_ax.set_ylabel("y")
    left_ax.legend(loc="best", fontsize=8)
    right_ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_target_last_fold_pointplot(
    combined: pd.DataFrame,
    *,
    target: str,
    run_root: Path,
    output_path: Path,
    history_steps_override: int | None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, output_actual_frame, actual_anchor_frame = _load_overlay_frames(
        combined,
        run_root=run_root,
        history_steps_override=history_steps_override,
    )
    model_frames = _model_prediction_frames(combined, actual_anchor_frame=actual_anchor_frame)
    ordered_labels = list(model_frames)
    fig, ax = plt.subplots(figsize=(12, 6))
    horizon_positions = range(len(output_actual_frame))
    if not output_actual_frame.empty:
        ax.plot(list(horizon_positions), pd.to_numeric(output_actual_frame["y"], errors="coerce"), label="Observed future", linewidth=1.8, linestyle="--", color="black", marker="o", markersize=6)
    offsets = [((idx - (len(ordered_labels) - 1) / 2) * 0.06) for idx in range(len(ordered_labels))]
    for offset, label in zip(offsets, ordered_labels):
        frame = model_frames[label].iloc[1:].reset_index(drop=True)
        xs = [pos + offset for pos in horizon_positions]
        ax.plot(xs, pd.to_numeric(frame["y_hat"], errors="coerce"), label=label, linewidth=1.4, marker="o", markersize=6)
    ax.set_xticks(list(horizon_positions))
    ax.set_xticklabels([f"h={idx + 1}" for idx in horizon_positions])
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("y")
    ax.set_title(f"{target.upper()} forecast point comparison ({_history_window_label(history_steps_override)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_target_plots(
    *,
    f3_root: Path,
    output_dir: Path,
    selected_backbones: tuple[str, ...] = _MODEL_CHOICES,
    window_sizes: tuple[int, ...] = (8, 16),
    label_set: str = "all",
    comparison_variants: tuple[str, ...] = (),
) -> list[Path]:
    outputs: list[Path] = []
    for target in _TARGETS:
        run_roots = _discover_target_run_roots(
            f3_root,
            target=target,
            selected_backbones=selected_backbones,
        )
        combined, representative_run_root = _labelled_last_fold_frame(
            target=target,
            run_roots=run_roots,
            selected_backbones=selected_backbones,
            label_set=label_set,
        )
        primary_path = output_dir / f"{target}_last_fold_all_models.png"
        outputs.append(
            _plot_target_last_fold(
                combined,
                target=target,
                run_root=representative_run_root,
                output_path=primary_path,
                history_steps_override=None,
            )
        )
        for variant in comparison_variants:
            variant_path = _variant_output_path(primary_path, variant)
            if variant == "delta_panel":
                outputs.append(_plot_target_last_fold_delta_panel(combined, target=target, run_root=representative_run_root, output_path=variant_path, history_steps_override=None))
            elif variant == "zoom":
                outputs.append(_plot_target_last_fold_zoom(combined, target=target, run_root=representative_run_root, output_path=variant_path, history_steps_override=None))
            elif variant == "pointplot":
                outputs.append(_plot_target_last_fold_pointplot(combined, target=target, run_root=representative_run_root, output_path=variant_path, history_steps_override=None))
            else:
                raise ValueError(f"unsupported comparison variant: {variant}")
        for window_size in window_sizes:
            window_output_path = output_dir / f"{target}_last_fold_all_models_window_{int(window_size)}.png"
            outputs.append(
                _plot_target_last_fold(
                    combined,
                    target=target,
                    run_root=representative_run_root,
                    output_path=window_output_path,
                    history_steps_override=int(window_size),
                )
            )
            for variant in comparison_variants:
                variant_path = _variant_output_path(window_output_path, variant)
                if variant == "delta_panel":
                    outputs.append(_plot_target_last_fold_delta_panel(combined, target=target, run_root=representative_run_root, output_path=variant_path, history_steps_override=int(window_size)))
                elif variant == "zoom":
                    outputs.append(_plot_target_last_fold_zoom(combined, target=target, run_root=representative_run_root, output_path=variant_path, history_steps_override=int(window_size)))
                elif variant == "pointplot":
                    outputs.append(_plot_target_last_fold_pointplot(combined, target=target, run_root=representative_run_root, output_path=variant_path, history_steps_override=int(window_size)))
                else:
                    raise ValueError(f"unsupported comparison variant: {variant}")
    return outputs


def main() -> int:
    args = _parse_args()
    outputs = generate_target_plots(
        f3_root=args.f3_root.resolve(),
        output_dir=args.output_dir,
        selected_backbones=args.models,
        window_sizes=tuple(dict.fromkeys(int(value) for value in args.window_sizes)),
        label_set=args.label_set,
        comparison_variants=tuple(dict.fromkeys(args.comparison_variants)),
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
