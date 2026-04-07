from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast

from . import config as _cfg


def _stage_root(run_root: Path) -> Path:
    root = run_root / "aa_forecast"
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    (root / "uncertainty").mkdir(parents=True, exist_ok=True)
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
            "compatibility_mapping_applied": bool(stage_cfg.compatibility_mode),
        },
    )


def _aa_params_override(loaded: Any) -> dict[str, Any]:
    stage_cfg = loaded.config.stage_plugin_config
    return {
        "p_value": stage_cfg.p_value,
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
    _write_json(
        summary_path,
        {
            "train_end_ds": str(pd.Timestamp(train_end_ds)),
            "p_value": stage_cfg.p_value,
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


def predict_aa_forecast_fold(
    loaded: Any,
    job: Any,
    *,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    run_root: Path | None,
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
    effective_config = _effective_config(loaded)
    diff_context = _build_fold_diff_context(loaded, train_df)
    transformed_train_df = _transform_training_frame(train_df, diff_context)
    adapter_inputs = _build_adapter_inputs(
        loaded,
        transformed_train_df,
        future_df,
        job,
        dt_col,
    )
    model = build_model(
        effective_config,
        job,
        n_series=adapter_inputs.metadata.get("n_series"),
        params_override=_aa_params_override(loaded),
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
