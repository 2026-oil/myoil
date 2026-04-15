from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast

from plugins.retrieval.runtime import (
    _blend_prediction as _shared_blend_prediction,
    _retrieve_neighbors as _shared_retrieve_neighbors,
)

from . import config as _cfg


def _stage_root(run_root: Path) -> Path:
    root = run_root / "aa_forecast"
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    (root / "uncertainty").mkdir(parents=True, exist_ok=True)
    (root / "context").mkdir(parents=True, exist_ok=True)
    (root / "retrieval").mkdir(parents=True, exist_ok=True)
    (root / "encoding").mkdir(parents=True, exist_ok=True)
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
        _cfg.aa_forecast_plugin_state_dict(stage_cfg, selected_config_path=selected),
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


def _coerce_encoding_array(value: Any, *, field_name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(
            f"aa_forecast encoding export requires {field_name} to be rank-3 [batch, time, hidden], got shape {array.shape!r}"
        )
    return array.astype(float, copy=False)


def _build_encoding_time_frame(
    *,
    train_df: pd.DataFrame,
    dt_col: str,
    time_steps: int,
) -> pd.DataFrame:
    ds_values = pd.to_datetime(train_df[dt_col]).reset_index(drop=True)
    if time_steps <= 0:
        raise ValueError("aa_forecast encoding export requires positive time_steps")
    if time_steps > len(ds_values):
        raise ValueError(
            "aa_forecast encoding export requires time_steps to fit inside the transformed training frame"
        )
    time_window = ds_values.iloc[-time_steps:].reset_index(drop=True)
    return pd.DataFrame(
        {
            "time_index": np.arange(time_steps, dtype=int),
            "ds": time_window,
        }
    )


def _tensor_to_long_frame(
    *,
    tensor_name: str,
    values: np.ndarray,
    time_frame: pd.DataFrame,
) -> pd.DataFrame:
    batch_size, time_steps, hidden_size = values.shape
    batch_index = np.repeat(np.arange(batch_size, dtype=int), time_steps * hidden_size)
    time_index = np.tile(
        np.repeat(np.arange(time_steps, dtype=int), hidden_size),
        batch_size,
    )
    hidden_index = np.tile(np.arange(hidden_size, dtype=int), batch_size * time_steps)
    ds_values = time_frame["ds"].to_numpy()[time_index]
    return pd.DataFrame(
        {
            "tensor_name": tensor_name,
            "batch_index": batch_index,
            "time_index": time_index,
            "hidden_index": hidden_index,
            "ds": pd.to_datetime(ds_values),
            "value": values.reshape(-1),
        }
    )


def _summarize_tensor_frame(
    frame: pd.DataFrame,
    *,
    batch_size: int,
    hidden_size: int,
) -> pd.DataFrame:
    grouped = (
        frame.groupby(["tensor_name", "time_index", "ds"], dropna=False)["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "value_mean",
                "std": "value_std",
                "min": "value_min",
                "max": "value_max",
            }
        )
    )
    l2_norm = (
        frame.assign(value_sq=lambda df: np.square(df["value"].to_numpy(dtype=float)))
        .groupby(["tensor_name", "time_index", "ds"], dropna=False)["value_sq"]
        .sum()
        .reset_index(name="value_l2_sq")
    )
    grouped = grouped.merge(l2_norm, on=["tensor_name", "time_index", "ds"], how="left")
    grouped["value_l2_norm"] = np.sqrt(grouped.pop("value_l2_sq"))
    grouped["batch_size"] = int(batch_size)
    grouped["hidden_size"] = int(hidden_size)
    return grouped


def _write_encoding_artifacts(
    *,
    run_root: Path,
    train_end_ds: pd.Timestamp,
    model: Any,
    train_df: pd.DataFrame,
    dt_col: str,
) -> dict[str, str]:
    _stage_root(run_root)
    slug = _context_slug(train_end_ds)
    export_root = Path("aa_forecast") / "encoding" / slug
    (run_root / export_root).mkdir(parents=True, exist_ok=True)
    latest_export = getattr(model, "_latest_encoding_export", None)
    if not isinstance(latest_export, dict):
        raise ValueError(
            "aa_forecast encoding export requires model._latest_encoding_export after final inference"
        )
    required_keys = {"backbone_states", "hidden_states", "time_axis"}
    missing_keys = sorted(required_keys.difference(latest_export))
    if missing_keys:
        raise ValueError(
            "aa_forecast encoding export requires keys: "
            + ", ".join(sorted(required_keys))
            + f"; missing {', '.join(missing_keys)}"
        )
    time_axis = int(latest_export["time_axis"])
    if time_axis != 1:
        raise ValueError(
            f"aa_forecast encoding export expected time_axis=1 for informer, got {time_axis}"
        )

    tensor_paths: dict[str, str] = {}
    time_axis_rows: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    metadata_tensors: list[dict[str, Any]] = []
    for tensor_name in ("backbone_states", "hidden_states"):
        values = _coerce_encoding_array(
            latest_export[tensor_name],
            field_name=f"model._latest_encoding_export[{tensor_name!r}]",
        )
        time_frame = _build_encoding_time_frame(
            train_df=train_df,
            dt_col=dt_col,
            time_steps=values.shape[time_axis],
        )
        long_frame = _tensor_to_long_frame(
            tensor_name=tensor_name,
            values=values,
            time_frame=time_frame,
        )
        relative_tensor_path = export_root / f"{tensor_name}.parquet"
        long_frame.to_parquet(run_root / relative_tensor_path, index=False)
        tensor_paths[tensor_name] = str(relative_tensor_path)
        summary_frames.append(
            _summarize_tensor_frame(
                long_frame,
                batch_size=values.shape[0],
                hidden_size=values.shape[2],
            )
        )
        time_axis_rows.append(
            time_frame.assign(
                tensor_name=tensor_name,
                time_axis_dim=time_axis,
                axis_order="batch,time,hidden",
            )
        )
        metadata_tensors.append(
            {
                "tensor_name": tensor_name,
                "shape": list(values.shape),
                "batch_axis_dim": 0,
                "time_axis_dim": time_axis,
                "hidden_axis_dim": 2,
                "parquet_path": str(relative_tensor_path),
                "time_start_ds": str(time_frame["ds"].iloc[0]),
                "time_end_ds": str(time_frame["ds"].iloc[-1]),
            }
        )

    relative_time_axis_path = export_root / "time_axis.parquet"
    pd.concat(time_axis_rows, ignore_index=True).to_parquet(
        run_root / relative_time_axis_path,
        index=False,
    )
    relative_summary_path = export_root / "summary.parquet"
    pd.concat(summary_frames, ignore_index=True).to_parquet(
        run_root / relative_summary_path,
        index=False,
    )
    relative_metadata_path = export_root / "metadata.json"
    _write_json(
        run_root / relative_metadata_path,
        {
            "train_end_ds": str(pd.Timestamp(train_end_ds)),
            "source": "base_predict_before_uncertainty",
            "model_backbone": getattr(model, "backbone", None),
            "time_axis_contract": {
                "axis_order": "batch,time,hidden",
                "time_axis_dim": time_axis,
                "time_index_column": "time_index",
                "time_timestamp_column": "ds",
                "time_reference": "tail of transformed training frame aligned to encoder window",
            },
            "tensors": metadata_tensors,
            "time_axis_path": str(relative_time_axis_path),
            "summary_path": str(relative_summary_path),
        },
    )
    return {
        "metadata": str(relative_metadata_path),
        "time_axis": str(relative_time_axis_path),
        "summary": str(relative_summary_path),
        **tensor_paths,
    }


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
    def _extract_debug_scalar(
        debug: dict[str, Any] | None,
        key: str,
        *,
        positive_only: bool = False,
        negative_only: bool = False,
    ) -> float | None:
        if not isinstance(debug, dict):
            return None
        value = debug.get(key)
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            return None
        if positive_only:
            arr = np.clip(arr, 0.0, None)
        if negative_only:
            arr = np.clip(-arr, 0.0, None)
        return float(arr.mean())

    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        min_value = float(values.min())
        max_value = float(values.max())
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            return np.zeros_like(values)
        spread = max_value - min_value
        if spread <= 1e-8:
            return np.zeros_like(values)
        return (values - min_value) / spread

    candidate_means: list[np.ndarray] = []
    candidate_stds: list[np.ndarray] = []
    candidate_samples: dict[str, list[list[float]]] = {}
    candidate_spike_support: list[float] = []
    candidate_baseline_drag: list[float] = []
    candidate_direction_mean: list[float] = []
    base_seed = int(getattr(model, "random_seed", 1) or 1)
    seed_stride = max(sample_count, 1) + 1
    for dropout_idx, dropout_p in enumerate(dropout_candidates):
        model.configure_stochastic_inference(enabled=True, dropout_p=dropout_p)
        samples: list[np.ndarray] = []
        sample_spike_supports: list[float] = []
        sample_baseline_drags: list[float] = []
        sample_direction_means: list[float] = []
        sample_semantic_scores: list[float] = []
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
            active_model = None
            if hasattr(nf, "models") and getattr(nf, "models"):
                active_model = nf.models[0]
            if active_model is None:
                active_model = model
            debug = getattr(active_model, "_latest_decoder_debug", None)
            spike_support = _extract_debug_scalar(
                debug,
                "semantic_spike_component",
                positive_only=True,
            )
            baseline_drag = _extract_debug_scalar(
                debug,
                "semantic_baseline_curve",
                negative_only=True,
            )
            if baseline_drag is None:
                baseline_drag = _extract_debug_scalar(
                    debug,
                    "semantic_baseline_level",
                    negative_only=True,
                )
            direction_mean = _extract_debug_scalar(
                debug,
                "semantic_spike_direction",
            )
            if spike_support is not None:
                sample_spike_supports.append(spike_support)
            if baseline_drag is not None:
                sample_baseline_drags.append(baseline_drag)
            if direction_mean is not None:
                sample_direction_means.append(direction_mean)
            semantic_score = 0.0
            if spike_support is not None and direction_mean is not None:
                semantic_score = float(spike_support) * max(float(direction_mean), 0.0)
                if baseline_drag is not None:
                    semantic_score = semantic_score / (
                        1.0 + max(float(baseline_drag), 0.0)
                    )
            sample_semantic_scores.append(float(semantic_score))
        stacked = np.vstack(samples)
        candidate_means.append(stacked.mean(axis=0))
        candidate_stds.append(stacked.std(axis=0))
        candidate_samples[f"{dropout_p:.2f}"] = stacked.tolist()
        candidate_spike_support.append(
            float(np.mean(sample_spike_supports)) if sample_spike_supports else 0.0
        )
        candidate_baseline_drag.append(
            float(np.mean(sample_baseline_drags)) if sample_baseline_drags else 0.0
        )
        candidate_direction_mean.append(
            float(np.mean(sample_direction_means)) if sample_direction_means else 0.0
        )
    model.configure_stochastic_inference(enabled=False)

    std_grid = np.vstack(candidate_stds)
    mean_grid = np.vstack(candidate_means)
    dispersion_scores = np.sqrt(np.mean(np.square(std_grid), axis=1))
    spike_supports = np.asarray(candidate_spike_support, dtype=float)
    baseline_drags = np.asarray(candidate_baseline_drag, dtype=float)
    direction_means = np.asarray(candidate_direction_mean, dtype=float)
    has_semantic_signal = bool(
        np.nanmax(spike_supports) > 0.35 and np.nanmax(direction_means) > 0.45
    )
    if has_semantic_signal:
        semantic_scores = (
            spike_supports
            * np.clip(direction_means, 0.0, None)
            / (1.0 + np.clip(baseline_drags, 0.0, None))
        )
        min_dispersion = float(np.nanmin(dispersion_scores))
        semantic_tolerance = 0.15
        eligible_mask = dispersion_scores <= (
            min_dispersion * (1.0 + semantic_tolerance)
        )
        if not np.any(eligible_mask):
            eligible_mask = np.ones_like(dispersion_scores, dtype=bool)
        eligible_semantic = np.where(eligible_mask, semantic_scores, -np.inf)
        best_semantic = float(np.nanmax(eligible_semantic))
        trajectory_scores = dispersion_scores.copy()
        semantic_tie_mask = np.isfinite(eligible_semantic) & (
            eligible_semantic >= (best_semantic - 1e-12)
        )
        if np.any(semantic_tie_mask):
            tie_break = dispersion_scores.copy()
            tie_break[~semantic_tie_mask] = tie_break[~semantic_tie_mask] + 1e6
            trajectory_scores = tie_break
        selection_mode = "trajectory_semantic_tradeoff"
    else:
        trajectory_scores = dispersion_scores
        selection_mode = "trajectory_min_dispersion"
    selected_path_idx = int(trajectory_scores.argmin())
    selected_mean = mean_grid[selected_path_idx]
    selected_std = std_grid[selected_path_idx]
    selected_dropout_scalar = float(
        np.asarray(dropout_candidates, dtype=float)[selected_path_idx]
    )
    selected_dropout = np.full(std_grid.shape[1], selected_dropout_scalar, dtype=float)
    return {
        "mean": selected_mean,
        "std": selected_std,
        "selected_dropout": selected_dropout,
        "candidate_mean_grid": mean_grid,
        "candidate_std_grid": std_grid,
        "candidate_dropout_values": np.asarray(dropout_candidates, dtype=float),
        "candidate_samples": candidate_samples,
        "selection_mode": selection_mode,
        "candidate_path_scores": trajectory_scores,
        "candidate_dispersion_scores": dispersion_scores,
        "candidate_spike_support": spike_supports,
        "candidate_baseline_drag": baseline_drags,
        "candidate_direction_mean": direction_means,
        "candidate_semantic_scores": (
            semantic_scores if has_semantic_signal else np.zeros_like(dispersion_scores)
        ),
        "selected_path_idx": selected_path_idx,
        "selected_path_score": float(trajectory_scores[selected_path_idx]),
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
    candidate_samples_path = (
        stage_root / "uncertainty" / f"{slug}.candidate_samples.csv"
    )
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
            "star_hist_exog_cols_resolved": list(
                stage_cfg.star_hist_exog_cols_resolved
            ),
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
            "selection_mode": summary.get("selection_mode", "per_horizon_min_std"),
            "selected_path_idx": summary.get("selected_path_idx"),
            "selected_path_score": summary.get("selected_path_score"),
            "candidate_path_scores": (
                summary["candidate_path_scores"].tolist()
                if "candidate_path_scores" in summary
                else None
            ),
            "candidate_dispersion_scores": (
                summary["candidate_dispersion_scores"].tolist()
                if "candidate_dispersion_scores" in summary
                else None
            ),
            "candidate_spike_support": (
                summary["candidate_spike_support"].tolist()
                if "candidate_spike_support" in summary
                else None
            ),
            "candidate_baseline_drag": (
                summary["candidate_baseline_drag"].tolist()
                if "candidate_baseline_drag" in summary
                else None
            ),
            "candidate_direction_mean": (
                summary["candidate_direction_mean"].tolist()
                if "candidate_direction_mean" in summary
                else None
            ),
            "candidate_semantic_scores": (
                summary["candidate_semantic_scores"].tolist()
                if "candidate_semantic_scores" in summary
                else None
            ),
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
    path_scores = summary.get(
        "candidate_path_scores",
        np.zeros(len(summary["candidate_dropout_values"]), dtype=float),
    )
    dispersion_scores = summary.get(
        "candidate_dispersion_scores",
        np.zeros(len(summary["candidate_dropout_values"]), dtype=float),
    )
    spike_support = summary.get(
        "candidate_spike_support",
        np.zeros(len(summary["candidate_dropout_values"]), dtype=float),
    )
    baseline_drag = summary.get(
        "candidate_baseline_drag",
        np.zeros(len(summary["candidate_dropout_values"]), dtype=float),
    )
    direction_mean = summary.get(
        "candidate_direction_mean",
        np.zeros(len(summary["candidate_dropout_values"]), dtype=float),
    )
    semantic_scores = summary.get(
        "candidate_semantic_scores",
        np.zeros(len(summary["candidate_dropout_values"]), dtype=float),
    )
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
                    "path_score": float(path_scores[dropout_idx]),
                    "dispersion_score": float(dispersion_scores[dropout_idx]),
                    "semantic_spike_support": float(spike_support[dropout_idx]),
                    "semantic_baseline_drag": float(baseline_drag[dropout_idx]),
                    "semantic_direction_mean": float(direction_mean[dropout_idx]),
                    "semantic_score": float(semantic_scores[dropout_idx]),
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
            horizon_frame.loc[
                horizon_frame["dropout_p"] == dropout_p, "prediction"
            ].to_numpy(dtype=float)
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
            f"AAForecast MC dropout prediction distribution (horizon {horizon_step})"
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


def _require_star_payload(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        payload["count_active_channels"]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
        .reshape(-1)
    )
    channel_activity = payload["channel_activity"].detach().cpu().numpy().astype(float)
    if channel_activity.ndim != 3:
        raise ValueError("aa_forecast retrieval requires 3D channel_activity payload")
    return (
        critical_mask,
        count_active_channels,
        channel_activity.reshape(channel_activity.shape[1], channel_activity.shape[2]),
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
            transformed_window_df[list(model.hist_exog_list)].to_numpy(
                dtype=np.float32
            ),
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
    event_score = float(count_active_channels.sum() + np.abs(channel_activity).sum())
    return {
        "shape_vector": _normalize_signature(shape_vector),
        "event_vector": _normalize_signature(event_vector),
        "event_score": event_score,
    }


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
        transformed_window = transformed_train_df.iloc[
            start_idx : end_idx + 1
        ].reset_index(drop=True)
        signature = _build_retrieval_signature(
            model=model,
            transformed_window_df=transformed_window,
            target_col=target_col,
        )
        if (
            retrieval_cfg.trigger_quantile is None
            and signature["event_score"] < retrieval_cfg.event_score_threshold
        ):
            continue
        anchor_value = float(raw_train_df[target_col].iloc[end_idx])
        future_values = raw_train_df[target_col].iloc[
            end_idx + 1 : end_idx + 1 + horizon
        ]
        if len(future_values) != horizon:
            continue
        scale = max(abs(anchor_value), 1e-8)
        future_returns = (future_values.to_numpy(dtype=float) - anchor_value) / scale
        bank.append(
            {
                "candidate_end_ds": str(
                    pd.Timestamp(raw_train_df[dt_col].iloc[end_idx])
                ),
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
    return _shared_retrieve_neighbors(
        query=query,
        bank=bank,
        retrieval_cfg=retrieval_cfg,
    )


def _blend_event_memory_prediction(
    *,
    base_prediction: np.ndarray,
    memory_prediction: np.ndarray,
    uncertainty_std: np.ndarray | None,
    retrieval_cfg: _cfg.AAForecastRetrievalConfig,
    mean_similarity: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _shared_blend_prediction(
        base_prediction=base_prediction,
        memory_prediction=memory_prediction,
        uncertainty_std=uncertainty_std,
        retrieval_cfg=retrieval_cfg,
        mean_similarity=mean_similarity,
    )


def _write_retrieval_artifacts(
    *,
    run_root: Path,
    train_end_ds: pd.Timestamp,
    retrieval_summary: dict[str, Any],
) -> str:
    _stage_root(run_root)
    slug = pd.Timestamp(train_end_ds).strftime("%Y%m%dT%H%M%S")
    relative_json_path = Path("aa_forecast") / "retrieval" / f"{slug}.json"
    relative_neighbors_path = (
        Path("aa_forecast") / "retrieval" / f"{slug}.neighbors.csv"
    )
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
            "future_returns": np.asarray(
                neighbor["future_returns"], dtype=float
            ).tolist(),
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
    pd.DataFrame(neighbor_rows).to_csv(run_root / relative_neighbors_path, index=False)
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
    stage_cfg = loaded.config.stage_plugin_config
    model = build_model(
        effective_config,
        job,
        n_series=adapter_inputs.metadata.get("n_series"),
        params_override=merged_params_override,
    )
    setattr(
        model,
        "_capture_encoding_export",
        bool(stage_cfg.encoding_export.enabled),
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
    encoding_artifacts: dict[str, str] | None = None
    if stage_cfg.encoding_export.enabled:
        if run_root is None:
            raise ValueError(
                "aa_forecast encoding export requires a run_root to write Parquet artifacts"
            )
        encoding_artifacts = _write_encoding_artifacts(
            run_root=run_root,
            train_end_ds=pd.to_datetime(train_df[dt_col].iloc[-1]),
            model=nf.models[0],
            train_df=transformed_train_df,
            dt_col=dt_col,
        )
        target_predictions["aaforecast_encoding_metadata_artifact"] = (
            encoding_artifacts["metadata"]
        )
        target_predictions["aaforecast_encoding_time_axis_artifact"] = (
            encoding_artifacts["time_axis"]
        )
        target_predictions["aaforecast_encoding_summary_artifact"] = encoding_artifacts[
            "summary"
        ]
        target_predictions["aaforecast_encoding_backbone_artifact"] = (
            encoding_artifacts["backbone_states"]
        )
        target_predictions["aaforecast_encoding_hidden_artifact"] = encoding_artifacts[
            "hidden_states"
        ]
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
