from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast

from . import config as _cfg


def _stage_root(run_root: Path) -> Path:
    root = run_root / "aa_forecast"
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    (root / "uncertainty").mkdir(parents=True, exist_ok=True)
    (root / "context").mkdir(parents=True, exist_ok=True)
    (root / "retrieval").mkdir(parents=True, exist_ok=True)
    return root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def materialize_aa_forecast_stage(
    *,
    loaded: Any,
    selected_jobs: Any,
    run_root: Path,
    main_resolved_path: Path,
    main_capability_path: Path,
    main_manifest_path: Path,
    entrypoint_version: str,
    validate_only: bool,
) -> None:
    stage_root = _stage_root(run_root)
    stage_cfg = loaded.config.stage_plugin_config
    selected = _cfg.aa_forecast_resolved_selected_path(
        stage_cfg, loaded.stage_plugin_loaded
    )
    _write_json(
        stage_root / "config" / "stage_config.json",
        _cfg.aa_forecast_plugin_state_dict(
            stage_cfg, selected_config_path=selected
        ),
    )
    _write_json(
        stage_root / "manifest" / "stage_manifest.json",
        {
            "entrypoint_version": entrypoint_version,
            "validate_only": bool(validate_only),
            "main_resolved_path": str(main_resolved_path),
            "main_capability_path": str(main_capability_path),
            "main_manifest_path": str(main_manifest_path),
            "selected_jobs": [job.model for job in selected_jobs],
        },
    )


def _aa_params_override(loaded: Any) -> dict[str, Any]:
    config = getattr(loaded, "config", loaded)
    stage_cfg = config.stage_plugin_config
    scaler_type = config.training.scaler_type
    if scaler_type == "robust":
        scaler_type = None
    return {
        "backbone": stage_cfg.model,
        "thresh": stage_cfg.thresh,
        "star_hist_exog_list": list(stage_cfg.star_hist_exog_cols_resolved),
        "non_star_hist_exog_list": list(stage_cfg.non_star_hist_exog_cols_resolved),
        "star_hist_exog_tail_modes": list(stage_cfg.star_anomaly_tail_modes_resolved),
        "lowess_frac": stage_cfg.lowess_frac,
        "lowess_delta": stage_cfg.lowess_delta,
        "uncertainty_enabled": stage_cfg.uncertainty.enabled,
        "uncertainty_dropout_candidates": list(
            stage_cfg.uncertainty.dropout_candidates
        ),
        "uncertainty_sample_count": stage_cfg.uncertainty.sample_count,
        "scaler_type": scaler_type,
    }


def _extract_target_prediction_frame(
    predictions: pd.DataFrame,
    *,
    target_col: str,
    model_name: str,
    diff_context: Any,
    restore_target_predictions: Any,
) -> pd.DataFrame:
    target_predictions = predictions[
        predictions["unique_id"] == target_col
    ].reset_index(drop=True)
    return restore_target_predictions(
        target_predictions,
        prediction_col=model_name,
        diff_context=diff_context,
    )


def _context_slug(train_end_ds: pd.Timestamp) -> str:
    return pd.Timestamp(train_end_ds).strftime("%Y%m%dT%H%M%S")


def _build_fold_context_frame(
    *,
    model: Any,
    train_df: pd.DataFrame,
    dt_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, bool]:
    required_attrs = (
        "star",
        "_reduce_critical_mask",
        "_select_hist_exog",
        "star_hist_exog_indices",
        "star_hist_exog_tail_modes",
    )
    if any(not hasattr(model, attr) for attr in required_attrs):
        empty_frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(train_df[dt_col]).reset_index(drop=True),
                "context_active": np.zeros(len(train_df), dtype=int),
                "context_label": ["normal_context"] * len(train_df),
            }
        )
        return empty_frame, False
    insample_y = torch.as_tensor(
        train_df[target_col].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    ).reshape(1, -1, 1)
    hist_exog = None
    if getattr(model, "hist_exog_list", ()):
        hist_exog = torch.as_tensor(
            train_df[list(model.hist_exog_list)].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ).reshape(1, len(train_df), -1)
    with torch.no_grad():
        target_star = model.star(insample_y, tail_modes=("two_sided",))
        target_mask = model._reduce_critical_mask(
            target_star["critical_mask"],
            template=insample_y,
        )
        star_hist_exog = (
            model._select_hist_exog(hist_exog, model.star_hist_exog_indices)
            if hist_exog is not None
            else None
        )
        star_hist_outputs = (
            model.star(
                star_hist_exog,
                tail_modes=model.star_hist_exog_tail_modes,
            )
            if star_hist_exog is not None
            else None
        )
        star_hist_mask = model._reduce_critical_mask(
            None if star_hist_outputs is None else star_hist_outputs["critical_mask"],
            template=insample_y,
        )
        context_mask = (
            (target_mask | star_hist_mask)
            .squeeze(0)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(train_df[dt_col]).reset_index(drop=True),
            "context_active": context_mask.astype(int),
            "context_label": np.where(
                context_mask,
                "anomaly_context",
                "normal_context",
            ),
        }
    )
    return frame, bool(context_mask.any())


def _write_context_artifacts(
    *,
    run_root: Path,
    train_end_ds: pd.Timestamp,
    context_frame: pd.DataFrame,
    context_active: bool,
) -> str:
    stage_root = _stage_root(run_root)
    slug = _context_slug(train_end_ds)
    relative_csv_path = Path("aa_forecast") / "context" / f"{slug}.csv"
    csv_path = run_root / relative_csv_path
    json_path = stage_root / "context" / f"{slug}.json"
    context_frame.to_csv(csv_path, index=False)
    _write_json(
        json_path,
        {
            "train_end_ds": str(pd.Timestamp(train_end_ds)),
            "context_active": bool(context_active),
            "context_points": int(len(context_frame)),
            "active_points": int(context_frame["context_active"].sum()),
            "csv_path": str(relative_csv_path),
        },
    )
    return str(relative_csv_path)


def _predict_with_adapter(
    nf: NeuralForecast,
    adapter_inputs: Any,
    *,
    random_seed: int | None = None,
) -> pd.DataFrame:
    predict_kwargs = {
        "df": adapter_inputs.fit_df,
        "static_df": adapter_inputs.static_df,
    }
    if adapter_inputs.futr_df is not None:
        predict_kwargs["futr_df"] = adapter_inputs.futr_df
    if random_seed is not None:
        predict_kwargs["random_seed"] = int(random_seed)
    return nf.predict(**predict_kwargs)


def _select_uncertainty_predictions(
    *,
    nf: NeuralForecast,
    adapter_inputs: Any,
    model: Any,
    model_name: str,
    target_col: str,
    diff_context: Any,
    restore_target_predictions: Any,
    prediction_column: str,
    dropout_candidates: tuple[float, ...],
    sample_count: int,
) -> dict[str, Any]:
    candidate_means: list[np.ndarray] = []
    candidate_stds: list[np.ndarray] = []
    candidate_samples: dict[str, list[list[float]]] = {}
    base_seed = int(getattr(model, "random_seed", 1) or 1)
    seed_stride = max(sample_count, 1) + 1
    for dropout_idx, dropout_p in enumerate(dropout_candidates):
        model.configure_stochastic_inference(enabled=True, dropout_p=dropout_p)
        samples: list[np.ndarray] = []
        for sample_idx in range(sample_count):
            sample_seed = base_seed + dropout_idx * seed_stride + sample_idx
            predictions = _predict_with_adapter(
                nf,
                adapter_inputs,
                random_seed=sample_seed,
            )
            restored = _extract_target_prediction_frame(
                predictions,
                target_col=target_col,
                model_name=model_name,
                diff_context=diff_context,
                restore_target_predictions=restore_target_predictions,
            )
            samples.append(restored[prediction_column].to_numpy(dtype=float))
        stacked = np.vstack(samples)
        candidate_means.append(stacked.mean(axis=0))
        candidate_stds.append(stacked.std(axis=0))
        candidate_samples[f"{dropout_p:.2f}"] = stacked.tolist()
    model.configure_stochastic_inference(enabled=False)

    std_grid = np.vstack(candidate_stds)
    mean_grid = np.vstack(candidate_means)
    best_indices = std_grid.argmin(axis=0)
    horizon_idx = np.arange(std_grid.shape[1])
    selected_mean = mean_grid[best_indices, horizon_idx]
    selected_std = std_grid[best_indices, horizon_idx]
    selected_dropout = np.asarray(dropout_candidates, dtype=float)[best_indices]
    return {
        "mean": selected_mean,
        "std": selected_std,
        "selected_dropout": selected_dropout,
        "candidate_mean_grid": mean_grid,
        "candidate_std_grid": std_grid,
        "candidate_dropout_values": np.asarray(dropout_candidates, dtype=float),
        "candidate_samples": candidate_samples,
    }


def _write_uncertainty_artifacts(
    *,
    run_root: Path,
    stage_cfg: Any,
    train_end_ds: pd.Timestamp,
    summary: dict[str, Any],
    target_actuals: pd.Series,
) -> None:
    stage_root = _stage_root(run_root)
    slug = pd.Timestamp(train_end_ds).strftime("%Y%m%dT%H%M%S")
    summary_path = stage_root / "uncertainty" / f"{slug}.json"
    csv_path = stage_root / "uncertainty" / f"{slug}.csv"
    candidate_stats_path = stage_root / "uncertainty" / f"{slug}.candidate_stats.csv"
    candidate_samples_path = stage_root / "uncertainty" / f"{slug}.candidate_samples.csv"
    candidate_plot_path = stage_root / "uncertainty" / f"{slug}.dropout_mae_sd.png"
    distribution_summary_path = (
        stage_root / "uncertainty" / f"{slug}.prediction_distribution_by_dropout.csv"
    )
    distribution_combined_path = (
        stage_root
        / "uncertainty"
        / f"{slug}.prediction_distribution_by_dropout.combined.csv"
    )
    distribution_plot_path = (
        stage_root / "uncertainty" / f"{slug}.prediction_distribution_by_dropout.png"
    )
    _write_json(
        summary_path,
        {
            "train_end_ds": str(pd.Timestamp(train_end_ds)),
            "thresh": stage_cfg.thresh,
            "star_hist_exog_cols_resolved": list(stage_cfg.star_hist_exog_cols_resolved),
            "non_star_hist_exog_cols_resolved": list(
                stage_cfg.non_star_hist_exog_cols_resolved
            ),
            "star_anomaly_tails": {
                "upward": list(stage_cfg.star_anomaly_tails_resolved["upward"]),
                "two_sided": list(stage_cfg.star_anomaly_tails_resolved["two_sided"]),
            },
            "star_anomaly_tail_modes_resolved": list(
                stage_cfg.star_anomaly_tail_modes_resolved
            ),
            "dropout_candidates": list(stage_cfg.uncertainty.dropout_candidates),
            "sample_count": stage_cfg.uncertainty.sample_count,
            "selected_dropout_by_horizon": summary["selected_dropout"].tolist(),
            "selected_std_by_horizon": summary["std"].tolist(),
        },
    )
    pd.DataFrame(
        {
            "horizon_step": np.arange(1, len(summary["mean"]) + 1),
            "selected_dropout": summary["selected_dropout"],
            "uncertainty_std": summary["std"],
            "prediction_mean": summary["mean"],
        }
    ).to_csv(csv_path, index=False)
    candidate_stats_rows: list[dict[str, float]] = []
    for dropout_idx, dropout_p in enumerate(summary["candidate_dropout_values"]):
        for horizon_idx in range(len(summary["mean"])):
            candidate_stats_rows.append(
                {
                    "horizon_step": horizon_idx + 1,
                    "dropout_p": float(dropout_p),
                    "prediction_mean": float(
                        summary["candidate_mean_grid"][dropout_idx, horizon_idx]
                    ),
                    "prediction_std": float(
                        summary["candidate_std_grid"][dropout_idx, horizon_idx]
                    ),
                }
            )
    pd.DataFrame(candidate_stats_rows).to_csv(candidate_stats_path, index=False)
    candidate_sample_rows: list[dict[str, float]] = []
    for dropout_key, sample_grid in summary["candidate_samples"].items():
        for sample_idx, horizon_values in enumerate(sample_grid):
            for horizon_idx, prediction in enumerate(horizon_values):
                candidate_sample_rows.append(
                    {
                        "horizon_step": horizon_idx + 1,
                        "dropout_p": float(dropout_key),
                        "sample_idx": sample_idx,
                        "prediction": float(prediction),
                    }
                )
    candidate_sample_frame = pd.DataFrame(candidate_sample_rows)
    candidate_sample_frame.to_csv(candidate_samples_path, index=False)
    distribution_summary = _build_uncertainty_prediction_distribution_summary(
        candidate_sample_frame=candidate_sample_frame
    )
    distribution_summary.to_csv(distribution_summary_path, index=False)
    distribution_combined = _build_uncertainty_prediction_distribution_summary(
        candidate_sample_frame=candidate_sample_frame,
        combine_horizons=True,
    )
    distribution_combined.to_csv(distribution_combined_path, index=False)
    error_summary = _build_uncertainty_error_summary(
        candidate_samples=summary["candidate_samples"],
        target_actuals=target_actuals,
    )
    _write_uncertainty_error_plot(
        error_summary=error_summary,
        plot_path=candidate_plot_path,
    )
    _write_uncertainty_prediction_distribution_plot(
        candidate_sample_frame=candidate_sample_frame,
        plot_path=distribution_plot_path,
    )


def _build_uncertainty_prediction_distribution_summary(
    *,
    candidate_sample_frame: pd.DataFrame,
    combine_horizons: bool = False,
) -> pd.DataFrame:
    if candidate_sample_frame.empty:
        raise ValueError("uncertainty candidate sample frame must not be empty")

    required_columns = {"dropout_p", "prediction"}
    if not combine_horizons:
        required_columns.add("horizon_step")
    missing_columns = required_columns.difference(candidate_sample_frame.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            "uncertainty candidate sample frame missing required columns: "
            f"{missing_text}"
        )

    group_columns = ["dropout_p"]
    if not combine_horizons:
        group_columns.append("horizon_step")

    summary = (
        candidate_sample_frame.groupby(group_columns, sort=True)["prediction"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            q05=lambda series: series.quantile(0.05),
            q25=lambda series: series.quantile(0.25),
            median="median",
            q75=lambda series: series.quantile(0.75),
            q95=lambda series: series.quantile(0.95),
            max="max",
        )
        .reset_index()
    )
    sort_columns = ["dropout_p"] if combine_horizons else ["dropout_p", "horizon_step"]
    return summary.sort_values(sort_columns).reset_index(drop=True)


def _build_uncertainty_error_summary(
    *,
    candidate_samples: dict[str, list[list[float]]],
    target_actuals: pd.Series | np.ndarray,
) -> pd.DataFrame:
    actual_values = np.asarray(target_actuals, dtype=float).reshape(-1)
    error_rows: list[dict[str, float]] = []
    for dropout_key, sample_grid in candidate_samples.items():
        sample_array = np.asarray(sample_grid, dtype=float)
        if sample_array.ndim != 2:
            raise ValueError("uncertainty candidate samples must be a 2D array")
        if sample_array.shape[1] != len(actual_values):
            raise ValueError(
                "uncertainty candidate sample horizon width must match target actuals"
            )
        absolute_errors = np.abs(sample_array - actual_values.reshape(1, -1))
        sample_mae = absolute_errors.mean(axis=1)
        error_rows.append(
            {
                "dropout_p": float(dropout_key),
                "mae_mean": float(sample_mae.mean()),
                "mae_sd": float(sample_mae.std(ddof=0)),
            }
        )
    return pd.DataFrame(error_rows).sort_values("dropout_p").reset_index(drop=True)


def _write_uncertainty_error_plot(
    *,
    error_summary: pd.DataFrame,
    plot_path: Path,
) -> None:
    if error_summary.empty:
        raise ValueError("uncertainty error summary must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_frame = error_summary.copy()
    plot_frame["dropout_p"] = pd.to_numeric(plot_frame["dropout_p"], errors="raise")
    plot_frame["mae_mean"] = pd.to_numeric(plot_frame["mae_mean"], errors="raise")
    plot_frame["mae_sd"] = pd.to_numeric(plot_frame["mae_sd"], errors="raise")
    x = plot_frame["dropout_p"].to_numpy(dtype=float)
    y = plot_frame["mae_mean"].to_numpy(dtype=float)
    sd = plot_frame["mae_sd"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x, y, label="MAE", linewidth=2.0)
    axis.fill_between(x, y - sd, y + sd, alpha=0.2, label="SD")
    axis.set_title("AAForecast dropout uncertainty error summary")
    axis.set_xlabel("Dynamic Dropout Probability")
    axis.set_ylabel("Error")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def _write_uncertainty_prediction_distribution_plot(
    *,
    candidate_sample_frame: pd.DataFrame,
    plot_path: Path,
) -> None:
    if candidate_sample_frame.empty:
        raise ValueError("uncertainty candidate sample frame must not be empty")
    required_columns = {"horizon_step", "dropout_p", "prediction"}
    missing_columns = required_columns.difference(candidate_sample_frame.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            "uncertainty candidate sample frame missing required columns: "
            f"{missing_text}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_frame = candidate_sample_frame.copy()
    plot_frame["horizon_step"] = pd.to_numeric(
        plot_frame["horizon_step"], errors="raise"
    ).astype(int)
    plot_frame["dropout_p"] = pd.to_numeric(plot_frame["dropout_p"], errors="raise")
    plot_frame["prediction"] = pd.to_numeric(plot_frame["prediction"], errors="raise")
    sort_columns = ["horizon_step", "dropout_p"]
    if "sample_idx" in plot_frame.columns:
        sort_columns.append("sample_idx")
    plot_frame = plot_frame.sort_values(sort_columns)

    horizons = sorted(plot_frame["horizon_step"].unique().tolist())
    if not horizons:
        raise ValueError("uncertainty candidate sample frame must include horizons")

    figure, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(12, 4.8 * len(horizons)),
        sharex=True,
    )
    if len(horizons) == 1:
        axes = [axes]

    for axis, horizon_step in zip(axes, horizons):
        horizon_frame = plot_frame[plot_frame["horizon_step"] == horizon_step]
        dropout_values = sorted(horizon_frame["dropout_p"].unique().tolist())
        boxplot_values = [
            horizon_frame.loc[horizon_frame["dropout_p"] == dropout_p, "prediction"]
            .to_numpy(dtype=float)
            for dropout_p in dropout_values
        ]
        tick_labels = [f"{dropout_p:.2f}" for dropout_p in dropout_values]
        axis.boxplot(boxplot_values, tick_labels=tick_labels, showfliers=False)
        mean_values = (
            horizon_frame.groupby("dropout_p")["prediction"]
            .mean()
            .reindex(dropout_values)
            .to_numpy(dtype=float)
        )
        median_values = (
            horizon_frame.groupby("dropout_p")["prediction"]
            .median()
            .reindex(dropout_values)
            .to_numpy(dtype=float)
        )
        x_positions = np.arange(1, len(dropout_values) + 1, dtype=float)
        axis.plot(
            x_positions,
            mean_values,
            color="tab:red",
            marker="o",
            linewidth=1.3,
            label="mean",
        )
        axis.plot(
            x_positions,
            median_values,
            color="tab:blue",
            marker="x",
            linewidth=1.0,
            label="median",
        )
        axis.set_title(
            "AAForecast MC dropout prediction distribution "
            f"(horizon {horizon_step})"
        )
        axis.set_ylabel("prediction")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")

    axes[-1].set_xlabel("dropout p")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _normalize_signature(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError("retrieval signature must not be empty")
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def _require_star_payload(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("critical_mask", "count_active_channels", "channel_activity")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            "aa_forecast retrieval requires STAR payload keys: "
            + ", ".join(required)
            + f"; missing: {', '.join(missing)}"
        )
    critical_mask = (
        payload["critical_mask"].detach().cpu().numpy().astype(float).reshape(-1)
    )
    count_active_channels = (
        payload["count_active_channels"].detach().cpu().numpy().astype(float).reshape(-1)
    )
    channel_activity = (
        payload["channel_activity"].detach().cpu().numpy().astype(float)
    )
    if channel_activity.ndim != 3:
        raise ValueError("aa_forecast retrieval requires 3D channel_activity payload")
    return critical_mask, count_active_channels, channel_activity.reshape(
        channel_activity.shape[1], channel_activity.shape[2]
    )


def _build_retrieval_signature(
    *,
    model: Any,
    transformed_window_df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any]:
    if not hasattr(model, "_compute_star_outputs"):
        raise ValueError(
            "aa_forecast retrieval requires model._compute_star_outputs for V0"
        )
    insample_y = torch.as_tensor(
        transformed_window_df[target_col].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    ).reshape(1, -1, 1)
    hist_exog = None
    if getattr(model, "hist_exog_list", ()):
        hist_exog = torch.as_tensor(
            transformed_window_df[list(model.hist_exog_list)].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ).reshape(1, len(transformed_window_df), -1)
    with torch.no_grad():
        payload = model._compute_star_outputs(insample_y, hist_exog)
    critical_mask, count_active_channels, channel_activity = _require_star_payload(
        payload
    )
    activity_sums = channel_activity.sum(axis=0)
    activity_max = channel_activity.max(axis=0)
    event_vector = np.concatenate(
        [
            critical_mask,
            count_active_channels,
            channel_activity.reshape(-1),
            activity_sums,
            activity_max,
        ]
    )
    shape_vector = transformed_window_df[target_col].to_numpy(dtype=float)
    event_score = float(
        count_active_channels.sum() + np.abs(channel_activity).sum()
    )
    return {
        "shape_vector": _normalize_signature(shape_vector),
        "event_vector": _normalize_signature(event_vector),
        "event_score": event_score,
    }


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("retrieval cosine similarity requires matching vector shapes")
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _build_event_memory_bank(
    *,
    model: Any,
    transformed_train_df: pd.DataFrame,
    raw_train_df: pd.DataFrame,
    dt_col: str,
    target_col: str,
    retrieval_cfg: _cfg.AAForecastRetrievalConfig,
    input_size: int,
    horizon: int,
) -> tuple[list[dict[str, Any]], int]:
    last_idx = len(raw_train_df) - 1
    max_end_idx = last_idx - horizon - retrieval_cfg.recency_gap_steps
    if max_end_idx < input_size - 1:
        return [], 0
    candidate_count = max_end_idx - (input_size - 1) + 1
    bank: list[dict[str, Any]] = []
    for end_idx in range(input_size - 1, max_end_idx + 1):
        start_idx = end_idx - input_size + 1
        transformed_window = transformed_train_df.iloc[start_idx : end_idx + 1].reset_index(
            drop=True
        )
        signature = _build_retrieval_signature(
            model=model,
            transformed_window_df=transformed_window,
            target_col=target_col,
        )
        if signature["event_score"] < retrieval_cfg.event_score_threshold:
            continue
        anchor_value = float(raw_train_df[target_col].iloc[end_idx])
        future_values = raw_train_df[target_col].iloc[end_idx + 1 : end_idx + 1 + horizon]
        if len(future_values) != horizon:
            continue
        scale = max(abs(anchor_value), 1e-8)
        future_returns = (
            future_values.to_numpy(dtype=float) - anchor_value
        ) / scale
        bank.append(
            {
                "candidate_end_ds": str(pd.Timestamp(raw_train_df[dt_col].iloc[end_idx])),
                "candidate_future_end_ds": str(
                    pd.Timestamp(raw_train_df[dt_col].iloc[end_idx + horizon])
                ),
                "shape_vector": signature["shape_vector"],
                "event_vector": signature["event_vector"],
                "event_score": signature["event_score"],
                "anchor_target_value": anchor_value,
                "future_returns": future_returns,
            }
        )
    return bank, candidate_count


def _build_event_query(
    *,
    model: Any,
    transformed_train_df: pd.DataFrame,
    target_col: str,
    input_size: int,
) -> dict[str, Any]:
    transformed_window = transformed_train_df.iloc[-input_size:].reset_index(drop=True)
    return _build_retrieval_signature(
        model=model,
        transformed_window_df=transformed_window,
        target_col=target_col,
    )


def _retrieve_event_neighbors(
    *,
    query: dict[str, Any],
    bank: list[dict[str, Any]],
    retrieval_cfg: _cfg.AAForecastRetrievalConfig,
) -> dict[str, Any]:
    if query["event_score"] < retrieval_cfg.event_score_threshold:
        return {
            "retrieval_attempted": True,
            "retrieval_applied": False,
            "skip_reason": "below_event_threshold",
            "top_neighbors": [],
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
        }
    if not bank:
        return {
            "retrieval_attempted": True,
            "retrieval_applied": False,
            "skip_reason": "empty_bank",
            "top_neighbors": [],
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
        }
    scored_neighbors: list[dict[str, Any]] = []
    for entry in bank:
        shape_similarity = _cosine_similarity(
            query["shape_vector"], entry["shape_vector"]
        )
        event_similarity = _cosine_similarity(
            query["event_vector"], entry["event_vector"]
        )
        similarity = 0.4 * shape_similarity + 0.6 * event_similarity
        if similarity < retrieval_cfg.min_similarity:
            continue
        scored_neighbors.append(
            {
                **entry,
                "shape_similarity": shape_similarity,
                "event_similarity": event_similarity,
                "similarity": similarity,
            }
        )
    if not scored_neighbors:
        return {
            "retrieval_attempted": True,
            "retrieval_applied": False,
            "skip_reason": "min_similarity",
            "top_neighbors": [],
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
        }
    scored_neighbors.sort(key=lambda item: item["similarity"], reverse=True)
    top_neighbors = scored_neighbors[: retrieval_cfg.top_k]
    logits = np.asarray(
        [neighbor["similarity"] for neighbor in top_neighbors],
        dtype=float,
    ) / retrieval_cfg.temperature
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights = weights / weights.sum()
    for neighbor, weight in zip(top_neighbors, weights, strict=True):
        neighbor["softmax_weight"] = float(weight)
    similarities = np.asarray(
        [neighbor["similarity"] for neighbor in top_neighbors],
        dtype=float,
    )
    return {
        "retrieval_attempted": True,
        "retrieval_applied": True,
        "skip_reason": None,
        "top_neighbors": top_neighbors,
        "mean_similarity": float(similarities.mean()),
        "max_similarity": float(similarities.max()),
    }


def _blend_event_memory_prediction(
    *,
    base_prediction: np.ndarray,
    memory_prediction: np.ndarray,
    uncertainty_std: np.ndarray | None,
    retrieval_cfg: _cfg.AAForecastRetrievalConfig,
    mean_similarity: float,
) -> tuple[np.ndarray, np.ndarray]:
    similarity_scale = float(np.clip(mean_similarity, 0.0, 1.0))
    if retrieval_cfg.use_uncertainty_gate and uncertainty_std is not None:
        std_values = np.asarray(uncertainty_std, dtype=float)
        max_std = float(np.max(std_values))
        if max_std > 1e-12:
            uncertainty_scale = std_values / max_std
        else:
            uncertainty_scale = np.ones_like(std_values)
    else:
        uncertainty_scale = np.ones_like(base_prediction, dtype=float)
    blend_weight = retrieval_cfg.blend_floor + (
        retrieval_cfg.blend_max - retrieval_cfg.blend_floor
    ) * similarity_scale * uncertainty_scale
    blend_weight = np.clip(blend_weight, retrieval_cfg.blend_floor, retrieval_cfg.blend_max)
    final_prediction = (
        (1.0 - blend_weight) * np.asarray(base_prediction, dtype=float)
        + blend_weight * np.asarray(memory_prediction, dtype=float)
    )
    return final_prediction, np.asarray(blend_weight, dtype=float)


def _write_retrieval_artifacts(
    *,
    run_root: Path,
    train_end_ds: pd.Timestamp,
    retrieval_summary: dict[str, Any],
) -> str:
    _stage_root(run_root)
    slug = pd.Timestamp(train_end_ds).strftime("%Y%m%dT%H%M%S")
    relative_json_path = Path("aa_forecast") / "retrieval" / f"{slug}.json"
    relative_neighbors_path = Path("aa_forecast") / "retrieval" / f"{slug}.neighbors.csv"
    json_payload = dict(retrieval_summary)
    json_payload["neighbors"] = [
        {
            "candidate_end_ds": neighbor["candidate_end_ds"],
            "candidate_future_end_ds": neighbor["candidate_future_end_ds"],
            "similarity": neighbor["similarity"],
            "shape_similarity": neighbor["shape_similarity"],
            "event_similarity": neighbor["event_similarity"],
            "softmax_weight": neighbor["softmax_weight"],
            "event_score": neighbor["event_score"],
            "anchor_target_value": neighbor["anchor_target_value"],
            "future_returns": np.asarray(neighbor["future_returns"], dtype=float).tolist(),
        }
        for neighbor in retrieval_summary["neighbors"]
    ]
    _write_json(run_root / relative_json_path, json_payload)
    neighbor_rows: list[dict[str, Any]] = []
    horizon = len(retrieval_summary["base_prediction"])
    for rank, neighbor in enumerate(retrieval_summary["neighbors"], start=1):
        row = {
            "rank": rank,
            "candidate_end_ds": neighbor["candidate_end_ds"],
            "candidate_future_end_ds": neighbor["candidate_future_end_ds"],
            "similarity": neighbor["similarity"],
            "shape_similarity": neighbor["shape_similarity"],
            "event_similarity": neighbor["event_similarity"],
            "softmax_weight": neighbor["softmax_weight"],
            "event_score": neighbor["event_score"],
            "anchor_target_value": neighbor["anchor_target_value"],
        }
        for horizon_idx in range(horizon):
            row[f"future_return_step_{horizon_idx + 1}"] = neighbor["future_returns"][
                horizon_idx
            ]
        neighbor_rows.append(row)
    pd.DataFrame(neighbor_rows).to_csv(
        run_root / relative_neighbors_path, index=False
    )
    return str(relative_json_path)


def predict_aa_forecast_fold(
    loaded: Any,
    job: Any,
    *,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    run_root: Path | None,
    params_override: dict[str, Any] | None = None,
    training_override: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp, pd.DataFrame, Any | None]:
    from runtime_support.forecast_models import build_model
    from runtime_support.runner import (
        _build_adapter_inputs,
        _build_fold_diff_context,
        _effective_config,
        _resolve_freq,
        _restore_target_predictions,
        _transform_training_frame,
    )

    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col
    source_df = pd.concat([train_df, future_df], ignore_index=True)
    freq = _resolve_freq(loaded, source_df)
    effective_config = _effective_config(loaded, training_override)
    diff_context = _build_fold_diff_context(loaded, train_df)
    transformed_train_df = _transform_training_frame(train_df, diff_context)
    adapter_inputs = _build_adapter_inputs(
        loaded,
        transformed_train_df,
        future_df,
        job,
        dt_col,
    )
    merged_params_override = {
        **_aa_params_override(effective_config),
        **(params_override or {}),
    }
    model = build_model(
        effective_config,
        job,
        n_series=adapter_inputs.metadata.get("n_series"),
        params_override=merged_params_override,
    )
    if hasattr(model, "set_star_precompute_context"):
        model.set_star_precompute_context(
            enabled=True,
            fold_key=json.dumps(
                {
                    "job": job.model,
                    "train_rows": len(transformed_train_df),
                    "train_end": str(train_df[dt_col].iloc[-1]),
                    "params_override": merged_params_override,
                    "training_override": training_override or {},
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
        )
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(
        adapter_inputs.fit_df,
        static_df=adapter_inputs.static_df,
        val_size=effective_config.training.val_size,
    )
    predictions = _predict_with_adapter(nf, adapter_inputs)
    target_predictions = _extract_target_prediction_frame(
        predictions,
        target_col=target_col,
        model_name=job.model,
        diff_context=diff_context,
        restore_target_predictions=_restore_target_predictions,
    )
    stage_cfg = loaded.config.stage_plugin_config
    context_frame, context_active = _build_fold_context_frame(
        model=nf.models[0],
        train_df=transformed_train_df,
        dt_col=dt_col,
        target_col=target_col,
    )
    context_artifact = None
    if run_root is not None:
        context_artifact = _write_context_artifacts(
            run_root=run_root,
            train_end_ds=pd.to_datetime(train_df[dt_col].iloc[-1]),
            context_frame=context_frame,
            context_active=context_active,
        )
    target_predictions["aaforecast_context_active"] = pd.Series(
        [pd.NA] * len(target_predictions),
        dtype="boolean",
    )
    target_predictions["aaforecast_context_label"] = pd.Series(
        [None] * len(target_predictions),
        dtype="object",
    )
    if context_artifact is not None:
        target_predictions["aaforecast_context_artifact"] = context_artifact
    uncertainty_summary: dict[str, Any] | None = None
    if stage_cfg.uncertainty.enabled:
        uncertainty_summary = _select_uncertainty_predictions(
            nf=nf,
            adapter_inputs=adapter_inputs,
            model=nf.models[0],
            model_name=job.model,
            target_col=target_col,
            diff_context=diff_context,
            restore_target_predictions=_restore_target_predictions,
            prediction_column=job.model,
            dropout_candidates=stage_cfg.uncertainty.dropout_candidates,
            sample_count=stage_cfg.uncertainty.sample_count,
        )
        target_predictions[job.model] = uncertainty_summary["mean"]
        target_predictions[f"{job.model}__uncertainty_std"] = uncertainty_summary["std"]
        target_predictions[f"{job.model}__selected_dropout"] = uncertainty_summary[
            "selected_dropout"
        ]
        if run_root is not None:
            _write_uncertainty_artifacts(
                run_root=run_root,
                stage_cfg=stage_cfg,
                train_end_ds=pd.to_datetime(train_df[dt_col].iloc[-1]),
                summary=uncertainty_summary,
                target_actuals=future_df[target_col].reset_index(drop=True),
            )
    retrieval_artifact = None
    retrieval_cfg = stage_cfg.retrieval
    if retrieval_cfg.enabled:
        if uncertainty_summary is None:
            raise ValueError(
                "aa_forecast retrieval requires uncertainty summary in predict_aa_forecast_fold"
            )
        input_size = int(getattr(nf.models[0], "input_size", len(transformed_train_df)))
        horizon = int(getattr(nf.models[0], "h", len(future_df)))
        if input_size <= 0:
            raise ValueError("aa_forecast retrieval requires positive input_size")
        if len(transformed_train_df) < input_size:
            raise ValueError(
                "aa_forecast retrieval requires transformed_train_df rows >= input_size"
            )
        bank, candidate_count = _build_event_memory_bank(
            model=nf.models[0],
            transformed_train_df=transformed_train_df,
            raw_train_df=train_df.reset_index(drop=True),
            dt_col=dt_col,
            target_col=target_col,
            retrieval_cfg=retrieval_cfg,
            input_size=input_size,
            horizon=horizon,
        )
        query = _build_event_query(
            model=nf.models[0],
            transformed_train_df=transformed_train_df,
            target_col=target_col,
            input_size=input_size,
        )
        retrieval_result = _retrieve_event_neighbors(
            query=query,
            bank=bank,
            retrieval_cfg=retrieval_cfg,
        )
        base_prediction = np.asarray(target_predictions[job.model], dtype=float)
        current_last_y = float(train_df[target_col].iloc[-1])
        retrieval_summary = {
            "cutoff": str(pd.Timestamp(train_df[dt_col].iloc[-1])),
            "train_end_ds": str(pd.Timestamp(train_df[dt_col].iloc[-1])),
            "retrieval_enabled": True,
            "retrieval_attempted": retrieval_result["retrieval_attempted"],
            "retrieval_applied": False,
            "skip_reason": retrieval_result["skip_reason"],
            "top_k_requested": retrieval_cfg.top_k,
            "top_k_used": len(retrieval_result["top_neighbors"]),
            "candidate_count": candidate_count,
            "eligible_candidate_count": len(bank),
            "event_score_threshold": retrieval_cfg.event_score_threshold,
            "recency_gap_steps": retrieval_cfg.recency_gap_steps,
            "min_similarity": retrieval_cfg.min_similarity,
            "mean_similarity": retrieval_result["mean_similarity"],
            "max_similarity": retrieval_result["max_similarity"],
            "query_event_score": query["event_score"],
            "blend_max": retrieval_cfg.blend_max,
            "blend_weight_by_horizon": [0.0] * len(base_prediction),
            "used_uncertainty_gate": retrieval_cfg.use_uncertainty_gate,
            "base_prediction": base_prediction.tolist(),
            "memory_prediction": base_prediction.tolist(),
            "final_prediction": base_prediction.tolist(),
            "neighbors": retrieval_result["top_neighbors"],
        }
        if retrieval_result["retrieval_applied"]:
            weighted_returns = np.zeros_like(base_prediction, dtype=float)
            for neighbor in retrieval_result["top_neighbors"]:
                weighted_returns = weighted_returns + (
                    float(neighbor["softmax_weight"])
                    * np.asarray(neighbor["future_returns"], dtype=float)
                )
            scale = max(abs(current_last_y), 1e-8)
            memory_prediction = current_last_y + scale * weighted_returns
            final_prediction, blend_weight = _blend_event_memory_prediction(
                base_prediction=base_prediction,
                memory_prediction=memory_prediction,
                uncertainty_std=np.asarray(uncertainty_summary["std"], dtype=float),
                retrieval_cfg=retrieval_cfg,
                mean_similarity=retrieval_result["mean_similarity"],
            )
            target_predictions[job.model] = final_prediction
            retrieval_summary["retrieval_applied"] = True
            retrieval_summary["skip_reason"] = None
            retrieval_summary["blend_weight_by_horizon"] = blend_weight.tolist()
            retrieval_summary["memory_prediction"] = memory_prediction.tolist()
            retrieval_summary["final_prediction"] = final_prediction.tolist()
        if run_root is not None:
            retrieval_artifact = _write_retrieval_artifacts(
                run_root=run_root,
                train_end_ds=pd.to_datetime(train_df[dt_col].iloc[-1]),
                retrieval_summary=retrieval_summary,
            )
        target_predictions["aaforecast_retrieval_enabled"] = pd.Series(
            [True] * len(target_predictions),
            dtype="boolean",
        )
        target_predictions["aaforecast_retrieval_applied"] = pd.Series(
            [retrieval_summary["retrieval_applied"]] * len(target_predictions),
            dtype="boolean",
        )
        target_predictions["aaforecast_retrieval_skip_reason"] = pd.Series(
            [retrieval_summary["skip_reason"]] * len(target_predictions),
            dtype="object",
        )
        if retrieval_artifact is not None:
            target_predictions["aaforecast_retrieval_artifact"] = retrieval_artifact
    target_actuals = future_df[target_col].reset_index(drop=True)
    train_end_ds = pd.to_datetime(train_df[dt_col].iloc[-1])
    return target_predictions, target_actuals, train_end_ds, train_df, nf
