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
        "top_k": stage_cfg.top_k,
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


def _predict_with_adapter(nf: NeuralForecast, adapter_inputs: Any) -> pd.DataFrame:
    predict_kwargs = {
        "df": adapter_inputs.fit_df,
        "static_df": adapter_inputs.static_df,
    }
    if adapter_inputs.futr_df is not None:
        predict_kwargs["futr_df"] = adapter_inputs.futr_df
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
    for dropout_p in dropout_candidates:
        model.configure_stochastic_inference(enabled=True, dropout_p=dropout_p)
        samples: list[np.ndarray] = []
        for _ in range(sample_count):
            predictions = _predict_with_adapter(nf, adapter_inputs)
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
) -> None:
    stage_root = _stage_root(run_root)
    slug = pd.Timestamp(train_end_ds).strftime("%Y%m%dT%H%M%S")
    summary_path = stage_root / "uncertainty" / f"{slug}.json"
    csv_path = stage_root / "uncertainty" / f"{slug}.csv"
    candidate_stats_path = stage_root / "uncertainty" / f"{slug}.candidate_stats.csv"
    candidate_samples_path = stage_root / "uncertainty" / f"{slug}.candidate_samples.csv"
    _write_json(
        summary_path,
        {
            "train_end_ds": str(pd.Timestamp(train_end_ds)),
            "top_k": stage_cfg.top_k,
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
    pd.DataFrame(candidate_sample_rows).to_csv(candidate_samples_path, index=False)


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
    target_predictions["aaforecast_context_active"] = bool(context_active)
    target_predictions["aaforecast_context_label"] = (
        "anomaly_context" if context_active else "normal_context"
    )
    if context_artifact is not None:
        target_predictions["aaforecast_context_artifact"] = context_artifact
    if stage_cfg.uncertainty.enabled:
        summary = _select_uncertainty_predictions(
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
        target_predictions[job.model] = summary["mean"]
        target_predictions[f"{job.model}__uncertainty_std"] = summary["std"]
        target_predictions[f"{job.model}__selected_dropout"] = summary[
            "selected_dropout"
        ]
        if run_root is not None:
            _write_uncertainty_artifacts(
                run_root=run_root,
                stage_cfg=stage_cfg,
                train_end_ds=pd.to_datetime(train_df[dt_col].iloc[-1]),
                summary=summary,
            )
    target_actuals = future_df[target_col].reset_index(drop=True)
    train_end_ds = pd.to_datetime(train_df[dt_col].iloc[-1])
    return target_predictions, target_actuals, train_end_ds, train_df, nf
