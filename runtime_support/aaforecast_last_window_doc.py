from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app_config import load_app_config
from neuralforecast import NeuralForecast
from neuralforecast.models.aaforecast.model import InformerHorizonAwareHead
from neuralforecast.models.aaforecast.gru import _align_horizon
from plugins.aa_forecast.runtime import (
    _aa_params_override,
    _extract_target_prediction_frame,
    _predict_with_adapter,
    _select_uncertainty_predictions,
)
from runtime_support.forecast_models import build_model
from runtime_support.runner import (
    _build_adapter_inputs,
    _build_fold_diff_context,
    _effective_config,
    _resolve_freq,
    _restore_prediction_series,
    _restore_target_predictions,
    _transform_training_frame,
)

TRACE_SCHEMA_FIELDS = (
    "stage",
    "tensor_name",
    "shape",
    "dtype",
    "meaning",
    "payload",
    "ordering",
)


@dataclass
class ReplayBundle:
    loaded: Any
    train_df: pd.DataFrame
    future_df: pd.DataFrame
    diff_context: Any
    transformed_train_df: pd.DataFrame
    adapter_inputs: Any
    nf: NeuralForecast
    model: Any
    deterministic_level_predictions: list[float]
    uncertainty_summary: dict[str, Any]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_native(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _to_native(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _to_native(value.tolist())
    if isinstance(value, pd.DataFrame):
        rows: list[dict[str, Any]] = []
        for row in value.to_dict(orient="records"):
            rows.append({str(key): _to_native(item) for key, item in row.items()})
        return rows
    if isinstance(value, pd.Series):
        return [_to_native(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _shape_of(value: Any) -> list[int] | list[str]:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if isinstance(value, pd.DataFrame):
        return [int(value.shape[0]), int(value.shape[1])]
    if isinstance(value, pd.Series):
        return [int(value.shape[0])]
    if isinstance(value, list):
        if not value:
            return [0]
        first = value[0]
        if isinstance(first, list):
            return [len(value), *(_shape_of(first))]
        if isinstance(first, dict):
            return [len(value), "records"]
        return [len(value)]
    return []


def _dtype_of(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.dtype)
    if isinstance(value, np.ndarray):
        return str(value.dtype)
    if isinstance(value, pd.DataFrame):
        return "dataframe"
    if isinstance(value, pd.Series):
        return str(value.dtype)
    if isinstance(value, list):
        if not value:
            return "list"
        first = value[0]
        if isinstance(first, dict):
            return "records"
        return type(first).__name__
    return type(value).__name__


class TraceBuilder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._ordering = 1

    def add(self, stage: str, tensor_name: str, value: Any, meaning: str) -> None:
        native = _to_native(value)
        self.records.append(
            {
                "stage": stage,
                "tensor_name": tensor_name,
                "shape": _shape_of(value),
                "dtype": _dtype_of(value),
                "meaning": meaning,
                "payload": native,
                "ordering": self._ordering,
            }
        )
        self._ordering += 1


def _build_replay_bundle(
    *,
    repo_root: Path,
    config_path: Path,
    cutoff: pd.Timestamp,
) -> ReplayBundle:
    loaded = load_app_config(repo_root, config_path=str(config_path))
    job = loaded.config.jobs[0]
    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col

    source_df = pd.read_csv(loaded.config.dataset.path)
    source_df[dt_col] = pd.to_datetime(source_df[dt_col])
    source_df = source_df.sort_values(dt_col).reset_index(drop=True)

    train_df = source_df[source_df[dt_col] <= cutoff].reset_index(drop=True)
    future_df = source_df[source_df[dt_col] > cutoff].head(loaded.config.cv.horizon).reset_index(drop=True)
    if len(future_df) != loaded.config.cv.horizon:
        raise ValueError("future_df length does not match configured horizon")

    diff_context = _build_fold_diff_context(loaded, train_df)
    transformed_train_df = _transform_training_frame(train_df, diff_context)
    effective_config = _effective_config(loaded, None)
    source_slice = pd.concat([train_df, future_df], ignore_index=True)
    freq = _resolve_freq(loaded, source_slice)
    adapter_inputs = _build_adapter_inputs(
        loaded,
        transformed_train_df,
        future_df,
        job,
        dt_col,
    )
    params_override = _aa_params_override(effective_config)
    model = build_model(
        effective_config,
        job,
        n_series=adapter_inputs.metadata.get("n_series"),
        params_override=params_override,
    )
    if hasattr(model, "set_star_precompute_context"):
        model.set_star_precompute_context(
            enabled=True,
            fold_key=json.dumps(
                {
                    "job": job.model,
                    "train_rows": len(transformed_train_df),
                    "train_end": str(train_df[dt_col].iloc[-1]),
                    "params_override": params_override,
                    "training_override": {},
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
    deterministic_level_predictions = (
        target_predictions[job.model].to_numpy(dtype=float).reshape(-1).tolist()
    )
    uncertainty_summary = _select_uncertainty_predictions(
        nf=nf,
        adapter_inputs=adapter_inputs,
        model=nf.models[0],
        model_name=job.model,
        target_col=target_col,
        diff_context=diff_context,
        restore_target_predictions=_restore_target_predictions,
        prediction_column=job.model,
        dropout_candidates=loaded.config.stage_plugin_config.uncertainty.dropout_candidates,
        sample_count=loaded.config.stage_plugin_config.uncertainty.sample_count,
    )
    return ReplayBundle(
        loaded=loaded,
        train_df=train_df,
        future_df=future_df,
        diff_context=diff_context,
        transformed_train_df=transformed_train_df,
        adapter_inputs=adapter_inputs,
        nf=nf,
        model=nf.models[0],
        deterministic_level_predictions=deterministic_level_predictions,
        uncertainty_summary=uncertainty_summary,
    )


def _add_raw_and_transformed_records(bundle: ReplayBundle, trace: TraceBuilder) -> None:
    loaded = bundle.loaded
    dt_col = loaded.config.dataset.dt_col
    target_col = loaded.config.dataset.target_col
    input_size = int(bundle.model.input_size)
    hist_cols = list(loaded.config.dataset.hist_exog_cols)

    diff_order = 0 if bundle.diff_context is None else max(
        int(getattr(bundle.diff_context, "target_diff_order", 0)),
        int(getattr(bundle.diff_context, "hist_exog_diff_order", 0)),
    )
    raw_support_window = bundle.train_df.tail(input_size + diff_order).reset_index(drop=True)
    transformed_input_window = bundle.transformed_train_df.tail(input_size).reset_index(drop=True)
    future_window = bundle.future_df.reset_index(drop=True)

    trace.add(
        "raw_rows",
        "raw_support_window_rows",
        raw_support_window,
        "All raw rows needed to compute the final diff-transformed input window. "
        f"Because the configured runtime diff order is {diff_order}, the "
        f"{input_size} transformed input steps depend on {input_size + diff_order} "
        "raw training rows.",
    )
    trace.add(
        "raw_rows",
        "future_horizon_rows",
        future_window,
        "Raw future rows for the two forecasted timestamps.",
    )
    trace.add(
        "raw_rows",
        f"raw_support_window.{dt_col}",
        raw_support_window[dt_col].astype(str).tolist(),
        "Raw support-window timestamps in chronological order.",
    )
    trace.add(
        "raw_rows",
        f"raw_support_window.{target_col}",
        raw_support_window[target_col].astype(float).tolist(),
        "Raw support-window target values before diff transformation.",
    )
    for column in hist_cols:
        trace.add(
            "raw_rows",
            f"raw_support_window.{column}",
            raw_support_window[column].astype(float).tolist(),
            f"Raw support-window exogenous values for {column} before diff transformation.",
        )

    trace.add(
        "transformed_inputs",
        "transformed_input_window_rows",
        transformed_input_window.assign(**{dt_col: transformed_input_window[dt_col].astype(str)}),
        "The exact transformed training rows (diff target + diff exogenous) that become the last 64-step model input window.",
    )
    trace.add(
        "transformed_inputs",
        "future_actual_level_target",
        future_window[target_col].astype(float).tolist(),
        "Actual future target values on the original level scale.",
    )
    future_actual_diff = np.diff(
        np.concatenate(
            [[float(bundle.train_df[target_col].iloc[-1])], future_window[target_col].to_numpy(dtype=float)]
        )
    )
    trace.add(
        "transformed_inputs",
        "future_actual_diff_target",
        future_actual_diff.tolist(),
        "Actual future target values expressed on the diff scale used by the model output restoration path.",
    )
    trace.add(
        "transformed_inputs",
        "diff_restore_anchor",
        float(bundle.train_df[target_col].iloc[-1]),
        "Last raw target level at the cutoff; diff-space forecasts are cumulatively added to this anchor.",
    )
    for column in [target_col, *hist_cols]:
        trace.add(
            "transformed_inputs",
            f"transformed_input_window.{column}",
            transformed_input_window[column].astype(float).tolist(),
            f"Final transformed 64-step vector for {column} after diff preprocessing.",
        )


def _tail_windows_for_model(bundle: ReplayBundle) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor | None]:
    model = bundle.model
    target_col = bundle.loaded.config.dataset.target_col
    input_window = bundle.transformed_train_df.tail(model.input_size).reset_index(drop=True)
    insample_y = torch.as_tensor(
        input_window[target_col].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    ).reshape(1, len(input_window), 1)
    hist_exog = None
    if getattr(model, "hist_exog_list", ()):
        hist_exog = torch.as_tensor(
            input_window[list(model.hist_exog_list)].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ).reshape(1, len(input_window), -1)
    return input_window, insample_y, hist_exog


def _trace_informer_head(
    head: InformerHorizonAwareHead,
    *,
    decoder_input: torch.Tensor,
    event_summary: torch.Tensor,
    event_path: torch.Tensor,
    raw_regime: torch.Tensor,
    pooled_context: torch.Tensor | None,
    memory_signal: torch.Tensor | None,
    anchor_value: torch.Tensor | None,
    memory_token: torch.Tensor | None,
    memory_bank: torch.Tensor | None,
    trace: TraceBuilder,
) -> torch.Tensor:
    head._validate_inputs(
        decoder_input,
        event_summary,
        event_path,
        raw_regime,
        pooled_context,
        memory_token,
        memory_bank,
    )
    horizon_context = head.build_horizon_context(
        batch_size=decoder_input.shape[0],
        device=decoder_input.device,
        dtype=decoder_input.dtype,
    )
    repeated_event = event_summary.unsqueeze(1).expand(-1, head.h, -1).to(dtype=decoder_input.dtype)
    repeated_path = event_path.unsqueeze(1).expand(-1, head.h, -1).to(dtype=decoder_input.dtype)
    context_features = torch.cat(
        [
            event_summary.to(dtype=decoder_input.dtype),
            event_path.to(dtype=decoder_input.dtype),
            raw_regime.to(dtype=decoder_input.dtype),
        ],
        dim=-1,
    )
    decoder_scale = 1.0 + torch.tanh(head.decoder_scale(context_features)).unsqueeze(1)
    decoder_shift = head.decoder_shift(context_features).unsqueeze(1)
    modulated_decoder_input = (decoder_input * decoder_scale) + decoder_shift
    regime_latent = head.regime_projector(raw_regime.to(dtype=decoder_input.dtype))
    regime_latent = F.gelu(
        torch.nan_to_num(regime_latent, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            min=-20.0,
            max=20.0,
        )
    )
    repeated_regime = regime_latent.unsqueeze(1).expand(-1, head.h, -1)
    event_gate = head.event_gate(torch.cat([repeated_event, horizon_context], dim=-1))
    path_gate = head.path_gate(torch.cat([repeated_path, horizon_context], dim=-1))
    regime_gate = head.regime_gate(torch.cat([repeated_regime, horizon_context], dim=-1))
    conditioned = torch.cat(
        [
            modulated_decoder_input,
            horizon_context,
            repeated_event,
            repeated_event * event_gate,
            repeated_path,
            repeated_path * path_gate,
            repeated_regime,
            repeated_regime * regime_gate,
        ],
        dim=-1,
    )
    trace.add("decoder", "head.decoder_input", decoder_input, "Decoder input passed into InformerHorizonAwareHead before FiLM-style conditioning.")
    trace.add("decoder", "head.horizon_context", horizon_context, "Learned per-horizon embedding vectors used to condition the shared decoder path.")
    trace.add("decoder", "head.repeated_event", repeated_event, "Event summary repeated across horizons.")
    trace.add("decoder", "head.repeated_path", repeated_path, "Event trajectory repeated across horizons.")
    trace.add("decoder", "head.context_features", context_features, "Context vector concatenating event summary, event trajectory, and raw regime descriptor.")
    trace.add("decoder", "head.decoder_scale", decoder_scale, "FiLM multiplicative scale applied to decoder_input.")
    trace.add("decoder", "head.decoder_shift", decoder_shift, "FiLM additive shift applied to decoder_input.")
    trace.add("decoder", "head.modulated_decoder_input", modulated_decoder_input, "Decoder input after FiLM scaling and shifting.")
    trace.add("decoder", "head.regime_latent", regime_latent, "Raw regime descriptor projected into decoder hidden space.")
    trace.add("decoder", "head.repeated_regime", repeated_regime, "Projected regime latent repeated across horizons.")
    trace.add("decoder", "head.event_gate", event_gate, "Per-horizon gate applied to the repeated event summary.")
    trace.add("decoder", "head.path_gate", path_gate, "Per-horizon gate applied to the repeated event trajectory.")
    trace.add("decoder", "head.regime_gate", regime_gate, "Per-horizon gate applied to the repeated regime latent.")
    trace.add("decoder", "head.conditioned", conditioned, "Final conditioned decoder feature tensor entering the shared trunk.")

    trunk_features_pre_attention = head.shared_trunk(conditioned)
    attended_path, path_attention_weights = head.path_attention(
        trunk_features_pre_attention,
        trunk_features_pre_attention,
        trunk_features_pre_attention,
        need_weights=True,
    )
    trunk_features = trunk_features_pre_attention + attended_path
    mixed_path, _ = head.path_mixer(trunk_features)
    mixed_path = mixed_path + trunk_features
    trace.add("decoder", "head.trunk_features_pre_attention", trunk_features_pre_attention, "Shared trunk features before path self-attention.")
    trace.add("decoder", "head.path_attention_output", attended_path, "Output of the decoder path self-attention block.")
    trace.add("decoder", "head.path_attention_weights", path_attention_weights, "Self-attention weights from the decoder path attention block.")
    trace.add("decoder", "head.trunk_features_post_attention", trunk_features, "Shared trunk features after adding the attention residual.")
    trace.add("decoder", "head.mixed_path", mixed_path, "Path-mixed decoder features after the GRU mixer residual.")

    if pooled_context is None:
        pooled_context = decoder_input.new_zeros((decoder_input.shape[0], head.pooled_features))
    else:
        pooled_context = pooled_context.to(dtype=decoder_input.dtype)
    if memory_signal is None:
        memory_signal = decoder_input.new_zeros((decoder_input.shape[0], 1))
    else:
        memory_signal = memory_signal.to(dtype=decoder_input.dtype)
    if memory_token is None:
        memory_token = decoder_input.new_zeros((decoder_input.shape[0], head.pooled_features))
    else:
        memory_token = memory_token.to(dtype=decoder_input.dtype)
    if memory_bank is None:
        memory_bank = memory_token.unsqueeze(1)
    else:
        memory_bank = memory_bank.to(dtype=decoder_input.dtype)
    memory_bank = head.memory_transport_projector(memory_bank)
    if anchor_value is None:
        anchor_value = decoder_input.new_zeros((decoder_input.shape[0], head.local_head.out_features))
    else:
        anchor_value = anchor_value.to(dtype=decoder_input.dtype)
    pooled_for_baseline = decoder_input.new_zeros((decoder_input.shape[0], head.pooled_features))
    baseline_context = torch.cat(
        [
            mixed_path.reshape(mixed_path.shape[0], -1),
            event_summary.to(dtype=decoder_input.dtype),
            event_path.to(dtype=decoder_input.dtype),
            pooled_for_baseline,
        ],
        dim=-1,
    )
    spike_context = torch.cat(
        [
            mixed_path.reshape(mixed_path.shape[0], -1),
            event_summary.to(dtype=decoder_input.dtype),
            event_path.to(dtype=decoder_input.dtype),
            pooled_context,
        ],
        dim=-1,
    )
    global_path = head.global_head(baseline_context).reshape(decoder_input.shape[0], head.h, -1)
    level = head.level_head(baseline_context).unsqueeze(1)
    delta_path_raw = head.delta_head(baseline_context).reshape(decoder_input.shape[0], head.h, -1)
    delta_path = torch.cumsum(delta_path_raw, dim=1)
    shock_context = torch.cat(
        [event_summary.to(dtype=decoder_input.dtype), event_path.to(dtype=decoder_input.dtype), pooled_context],
        dim=-1,
    )
    event_bias = head.event_bias_head(shock_context).reshape(decoder_input.shape[0], head.h, -1)
    event_delta_raw = head.event_delta_head(shock_context).reshape(decoder_input.shape[0], head.h, -1)
    event_delta = torch.cumsum(F.softplus(event_delta_raw), dim=1)
    event_delta_gate = (1.0 + F.softplus(head.event_delta_gate(shock_context))).unsqueeze(1)
    baseline_amplitude_context = torch.cat(
        [event_summary.to(dtype=decoder_input.dtype), event_path.to(dtype=decoder_input.dtype), raw_regime.to(dtype=decoder_input.dtype), pooled_for_baseline],
        dim=-1,
    )
    spike_amplitude_context = torch.cat(
        [event_summary.to(dtype=decoder_input.dtype), event_path.to(dtype=decoder_input.dtype), raw_regime.to(dtype=decoder_input.dtype), pooled_context],
        dim=-1,
    )
    level_shift = head.level_shift_head(baseline_amplitude_context).unsqueeze(1)
    path_amplitude = (1.0 + F.softplus(head.path_amplitude_head(spike_amplitude_context))).unsqueeze(1)
    normal_expert = head.normal_expert_head(baseline_context).reshape(decoder_input.shape[0], head.h, -1)
    spike_expert_raw = head.spike_expert_head(spike_context).reshape(decoder_input.shape[0], head.h, -1)
    spike_expert = torch.cumsum(F.softplus(spike_expert_raw), dim=1)
    expert_gate = torch.sigmoid(head.expert_gate(spike_amplitude_context) + (0.5 * memory_signal)).unsqueeze(1)
    spike_uplift = expert_gate * spike_expert
    expert_residual = normal_expert + spike_uplift
    local_path = head.local_head(mixed_path)
    residual_path = global_path + delta_path + local_path + event_bias + (event_delta * event_delta_gate) + expert_residual
    trace.add("decoder", "head.pooled_context", pooled_context, "Pooled decoder context used for spike-sensitive branches.")
    trace.add("decoder", "head.memory_signal", memory_signal, "Scalar memory-signal gate derived from retrieved memory logits.")
    trace.add("decoder", "head.memory_token", memory_token, "Top memory token selected by the outer decoder memory builder.")
    trace.add("decoder", "head.memory_bank_projected", memory_bank, "Projected memory bank fed into the decoder-level transport attention.")
    trace.add("decoder", "head.anchor_value", anchor_value, "Anchor value passed into the decoder head before final diff-to-level restoration.")
    trace.add("decoder", "head.baseline_context", baseline_context, "Baseline decoder context without pooled anomaly memory.")
    trace.add("decoder", "head.spike_context", spike_context, "Spike-aware decoder context including pooled anomaly memory.")
    trace.add("decoder", "head.global_path", global_path, "Global multi-horizon decoder path contribution.")
    trace.add("decoder", "head.level", level, "Decoder level component prior to final combination.")
    trace.add("decoder", "head.delta_path_raw", delta_path_raw, "Raw incremental delta path before cumulative summation.")
    trace.add("decoder", "head.delta_path", delta_path, "Cumulative delta path contribution.")
    trace.add("decoder", "head.shock_context", shock_context, "Shock context used by event bias/delta branches.")
    trace.add("decoder", "head.event_bias", event_bias, "Per-horizon event bias component.")
    trace.add("decoder", "head.event_delta_raw", event_delta_raw, "Raw event-delta increments before softplus+cumsum.")
    trace.add("decoder", "head.event_delta", event_delta, "Positive cumulative event-delta contribution.")
    trace.add("decoder", "head.event_delta_gate", event_delta_gate, "Gate scaling the event-delta contribution.")
    trace.add("decoder", "head.baseline_amplitude_context", baseline_amplitude_context, "Baseline amplitude context for level shifts.")
    trace.add("decoder", "head.spike_amplitude_context", spike_amplitude_context, "Spike amplitude context for anomaly-aware gains and experts.")
    trace.add("decoder", "head.level_shift", level_shift, "Level-shift component predicted from the baseline amplitude context.")
    trace.add("decoder", "head.path_amplitude", path_amplitude, "Positive gain applied to the spike path amplitude branch.")
    trace.add("decoder", "head.normal_expert", normal_expert, "Baseline expert contribution.")
    trace.add("decoder", "head.spike_expert_raw", spike_expert_raw, "Raw spike-expert increments before softplus+cumsum.")
    trace.add("decoder", "head.spike_expert", spike_expert, "Positive cumulative spike-expert path.")
    trace.add("decoder", "head.expert_gate", expert_gate, "Gate selecting how much of the spike expert is added.")
    trace.add("decoder", "head.spike_uplift", spike_uplift, "Spike-only uplift after gating the spike expert.")
    trace.add("decoder", "head.expert_residual", expert_residual, "Combined expert contribution added into the residual path.")
    trace.add("decoder", "head.local_path", local_path, "Local decoder contribution from the mixed path states.")
    trace.add("decoder", "head.residual_path", residual_path, "Residual path sum before semantic/trajectory post-processing branches.")

    trajectory_context = torch.cat(
        [
            event_summary.to(dtype=decoder_input.dtype),
            event_path.to(dtype=decoder_input.dtype),
            raw_regime.to(dtype=decoder_input.dtype),
            pooled_context,
        ],
        dim=-1,
    )
    trajectory_hidden = head.trajectory_seed_head(trajectory_context)
    trajectory_hidden = F.gelu(
        torch.nan_to_num(trajectory_hidden, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
    )
    memory_seed = head.trajectory_memory_seed_head(pooled_context.to(dtype=decoder_input.dtype))
    memory_seed = F.gelu(
        torch.nan_to_num(memory_seed, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
    )
    trajectory_hidden = trajectory_hidden + (0.5 * memory_seed)
    trajectory_steps: list[torch.Tensor] = []
    trace.add("decoder", "head.trajectory_context", trajectory_context, "Joint context used to seed the trajectory branch.")
    trace.add("decoder", "head.trajectory_hidden_seed", trajectory_hidden, "Initial hidden state for the trajectory branch after memory seeding.")
    trace.add("decoder", "head.memory_seed", memory_seed, "Auxiliary memory-derived seed added into the trajectory branch.")
    for step_idx in range(head.h):
        step_context = torch.cat([horizon_context[:, step_idx, :], trajectory_context], dim=-1)
        step_input = head.trajectory_input_head(step_context)
        trajectory_hidden = head.trajectory_cell(step_input, trajectory_hidden)
        step_output = 0.15 * torch.sigmoid(head.trajectory_output_head(trajectory_hidden))
        trajectory_steps.append(step_output)
        trace.add("decoder", f"head.trajectory_step_context_h{step_idx + 1}", step_context, f"Trajectory branch context for horizon step {step_idx + 1}.")
        trace.add("decoder", f"head.trajectory_step_input_h{step_idx + 1}", step_input, f"Trajectory branch GRU input for horizon step {step_idx + 1}.")
        trace.add("decoder", f"head.trajectory_hidden_h{step_idx + 1}", trajectory_hidden, f"Trajectory branch hidden state after horizon step {step_idx + 1}.")
        trace.add("decoder", f"head.trajectory_step_output_h{step_idx + 1}", step_output, f"Trajectory branch positive increment for horizon step {step_idx + 1}.")
    trajectory_shock = torch.cumsum(torch.stack(trajectory_steps, dim=1), dim=1)
    trajectory_gate = (1.0 + torch.sigmoid(head.trajectory_gate_head(trajectory_context))).unsqueeze(1)
    anchor_scale = torch.log1p(anchor_value.abs().clamp_min(1.0)).unsqueeze(1)
    trajectory_component = trajectory_shock * trajectory_gate * anchor_scale
    semantic_baseline_context = torch.cat(
        [mixed_path.reshape(mixed_path.shape[0], -1), event_summary.to(dtype=decoder_input.dtype), raw_regime.to(dtype=decoder_input.dtype)],
        dim=-1,
    )
    semantic_baseline_level = 0.1 * torch.tanh(head.semantic_baseline_level_head(semantic_baseline_context)).unsqueeze(1) * anchor_scale
    semantic_spike_context = torch.cat(
        [trajectory_context, memory_token, anchor_value, memory_signal],
        dim=-1,
    )
    semantic_baseline_delta_raw = head.semantic_baseline_delta_head(semantic_baseline_context).reshape(decoder_input.shape[0], head.h, -1)
    semantic_baseline_curve = torch.cumsum(0.1 * torch.tanh(semantic_baseline_delta_raw), dim=1) * anchor_scale
    semantic_spike_hidden = head.semantic_spike_seed_head(semantic_spike_context)
    semantic_spike_hidden = F.gelu(
        torch.nan_to_num(semantic_spike_hidden, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
    )
    semantic_spike_pos_steps: list[torch.Tensor] = []
    semantic_spike_neg_steps: list[torch.Tensor] = []
    trace.add("decoder", "head.trajectory_shock", trajectory_shock, "Cumulative positive trajectory shock across the two horizons.")
    trace.add("decoder", "head.trajectory_gate", trajectory_gate, "Gate scaling the trajectory shock.")
    trace.add("decoder", "head.anchor_scale", anchor_scale, "Log-scaled magnitude of the last input anchor used to scale semantic and trajectory curves.")
    trace.add("decoder", "head.trajectory_component", trajectory_component, "Trajectory-driven component before final combination.")
    trace.add("decoder", "head.semantic_baseline_context", semantic_baseline_context, "Context used for the semantic baseline level/curve heads.")
    trace.add("decoder", "head.semantic_baseline_level", semantic_baseline_level, "Low-amplitude semantic baseline level term.")
    trace.add("decoder", "head.semantic_spike_context", semantic_spike_context, "Spike-context vector used by the semantic spike branch.")
    trace.add("decoder", "head.semantic_baseline_delta_raw", semantic_baseline_delta_raw, "Raw semantic baseline increments before tanh+cumsum.")
    trace.add("decoder", "head.semantic_baseline_curve", semantic_baseline_curve, "Cumulative semantic baseline curve.")
    trace.add("decoder", "head.semantic_spike_hidden_seed", semantic_spike_hidden, "Initial hidden state for the semantic spike GRU branch.")
    for step_idx in range(head.h):
        semantic_query = semantic_spike_hidden.unsqueeze(1)
        semantic_memory_step, semantic_memory_weights = head.memory_transport_attention(
            semantic_query,
            memory_bank,
            memory_bank,
            need_weights=True,
        )
        semantic_step_features = torch.cat(
            [
                semantic_spike_hidden,
                semantic_memory_step.squeeze(1),
                horizon_context[:, step_idx, :],
                event_path.to(dtype=decoder_input.dtype),
                raw_regime.to(dtype=decoder_input.dtype),
            ],
            dim=-1,
        )
        semantic_step_input = head.semantic_spike_step_head(semantic_step_features)
        semantic_spike_hidden = head.semantic_spike_cell(semantic_step_input, semantic_spike_hidden)
        semantic_spike_pos = F.softplus(head.semantic_spike_pos_out_head(semantic_spike_hidden))
        semantic_spike_neg = F.softplus(head.semantic_spike_neg_out_head(semantic_spike_hidden))
        semantic_spike_pos_steps.append(semantic_spike_pos)
        semantic_spike_neg_steps.append(semantic_spike_neg)
        trace.add("decoder", f"head.semantic_query_h{step_idx + 1}", semantic_query, f"Semantic spike memory-attention query for horizon step {step_idx + 1}.")
        trace.add("decoder", f"head.semantic_memory_step_h{step_idx + 1}", semantic_memory_step, f"Memory-attention output for semantic spike step {step_idx + 1}.")
        trace.add("decoder", f"head.semantic_memory_weights_h{step_idx + 1}", semantic_memory_weights, f"Memory-attention weights for semantic spike step {step_idx + 1}.")
        trace.add("decoder", f"head.semantic_step_features_h{step_idx + 1}", semantic_step_features, f"Semantic spike feature vector for horizon step {step_idx + 1}.")
        trace.add("decoder", f"head.semantic_step_input_h{step_idx + 1}", semantic_step_input, f"Semantic spike GRU input for horizon step {step_idx + 1}.")
        trace.add("decoder", f"head.semantic_spike_hidden_h{step_idx + 1}", semantic_spike_hidden, f"Semantic spike hidden state after horizon step {step_idx + 1}.")
        trace.add("decoder", f"head.semantic_spike_pos_h{step_idx + 1}", semantic_spike_pos, f"Positive semantic spike increment for horizon step {step_idx + 1}.")
        trace.add("decoder", f"head.semantic_spike_neg_h{step_idx + 1}", semantic_spike_neg, f"Negative semantic spike increment for horizon step {step_idx + 1}.")
    semantic_spike_pos_curve = torch.cumsum(torch.stack(semantic_spike_pos_steps, dim=1), dim=1)
    semantic_spike_neg_curve = torch.cumsum(torch.stack(semantic_spike_neg_steps, dim=1), dim=1)
    semantic_spike_gate = torch.sigmoid(head.semantic_spike_gate_head(semantic_spike_context) + (0.5 * memory_signal)).unsqueeze(1)
    semantic_spike_gain = (1.0 + F.softplus(head.semantic_spike_gain_head(semantic_spike_context))).unsqueeze(1)
    semantic_spike_direction = torch.sigmoid(head.semantic_spike_direction_head(semantic_spike_context) + (0.5 * memory_signal)).unsqueeze(1)
    semantic_negative_weight = 0.9 * (1.0 - semantic_spike_direction).pow(2)
    semantic_spike_curve = (semantic_spike_direction * semantic_spike_pos_curve) - (semantic_negative_weight * semantic_spike_neg_curve)
    semantic_spike_component = semantic_spike_curve * semantic_spike_gate * semantic_spike_gain * anchor_scale
    memory_transport_states, memory_transport_weights = head.memory_transport_attention(
        mixed_path,
        memory_bank,
        memory_bank,
        need_weights=True,
    )
    memory_transport = torch.cumsum(F.softplus(head.local_head(memory_transport_states)), dim=1)
    memory_transport_gate = (1.0 + torch.sigmoid(head.memory_transport_gate_head(trajectory_context))).unsqueeze(1)
    analogue_component = trajectory_component + (memory_transport * memory_transport_gate * anchor_scale)
    final_output = semantic_baseline_level + semantic_baseline_curve + semantic_spike_component
    trace.add("decoder", "head.semantic_spike_pos_curve", semantic_spike_pos_curve, "Cumulative positive semantic spike curve.")
    trace.add("decoder", "head.semantic_spike_neg_curve", semantic_spike_neg_curve, "Cumulative negative semantic spike curve.")
    trace.add("decoder", "head.semantic_spike_gate", semantic_spike_gate, "Gate controlling the semantic spike branch.")
    trace.add("decoder", "head.semantic_spike_gain", semantic_spike_gain, "Positive gain applied to the semantic spike branch.")
    trace.add("decoder", "head.semantic_spike_direction", semantic_spike_direction, "Direction selector between positive and negative semantic curves.")
    trace.add("decoder", "head.semantic_negative_weight", semantic_negative_weight, "Penalty weight applied to the negative semantic curve.")
    trace.add("decoder", "head.semantic_spike_curve", semantic_spike_curve, "Signed semantic spike curve before gating and gain.")
    trace.add("decoder", "head.semantic_spike_component", semantic_spike_component, "Final semantic spike component added to the output.")
    trace.add("decoder", "head.memory_transport_states", memory_transport_states, "Decoder path states after memory-transport attention over the retrieved memory bank.")
    trace.add("decoder", "head.memory_transport_weights", memory_transport_weights, "Attention weights for the decoder memory-transport block.")
    trace.add("decoder", "head.memory_transport", memory_transport, "Positive cumulative memory transport contribution.")
    trace.add("decoder", "head.memory_transport_gate", memory_transport_gate, "Gate controlling the memory transport contribution.")
    trace.add("decoder", "head.analogue_component", analogue_component, "Trajectory-plus-memory analogue component (computed but not returned by the current decoder head).")
    trace.add("decoder", "head.final_output", final_output, "Final decoder-head diff-space output before the outer anchor is added back in `_decode_informer_forecast`. ")
    return final_output


def _trace_current_replay(bundle: ReplayBundle, trace: TraceBuilder) -> tuple[list[float], list[float]]:
    model = bundle.model
    model.eval()
    if hasattr(model, "configure_stochastic_inference"):
        model.configure_stochastic_inference(enabled=False, dropout_p=0.0)
    input_window, insample_y, hist_exog = _tail_windows_for_model(bundle)
    trace.add("transformed_inputs", "model_input_window_rows", input_window.assign(**{bundle.loaded.config.dataset.dt_col: input_window[bundle.loaded.config.dataset.dt_col].astype(str)}), "The exact transformed rows consumed by the current-code replay model forward pass.")
    trace.add("transformed_inputs", "model_input.insample_y", insample_y, "Current-code replay insample_y tensor fed into the model forward pass.")
    if hist_exog is not None:
        trace.add("transformed_inputs", "model_input.hist_exog", hist_exog, "Current-code replay historical exogenous tensor fed into the model forward pass.")
    with torch.no_grad():
        star_payload = model._compute_star_outputs(insample_y, hist_exog)
        trace.add("star", "star.target_trend", star_payload["target_trend"], "Target trend component from STAR decomposition.")
        trace.add("star", "star.target_seasonal", star_payload["target_seasonal"], "Target seasonal component from STAR decomposition.")
        trace.add("star", "star.target_anomalies", star_payload["target_anomalies"], "Target anomalies component from STAR decomposition.")
        trace.add("star", "star.target_residual", star_payload["target_residual"], "Target residual component from STAR decomposition.")
        trace.add("star", "star.target_activity", star_payload["target_activity"], "Target activity tensor = ranking_score * critical_mask.")
        trace.add("star", "star.target_signed_score", star_payload["target_signed_score"], "Signed STAR robust score for the target series.")
        trace.add("star", "star.star_hist_trend", star_payload["star_hist_trend"], "STAR trend components for the STAR-designated historical exogenous channels.")
        trace.add("star", "star.star_hist_seasonal", star_payload["star_hist_seasonal"], "STAR seasonal components for the STAR-designated historical exogenous channels.")
        trace.add("star", "star.star_hist_anomalies", star_payload["star_hist_anomalies"], "STAR anomalies for the STAR-designated historical exogenous channels.")
        trace.add("star", "star.star_hist_residual", star_payload["star_hist_residual"], "STAR residual components for the STAR-designated historical exogenous channels.")
        trace.add("star", "star.critical_mask", star_payload["critical_mask"].to(dtype=torch.int64), "Combined anomaly/context mask after merging target, STAR-hist, non-STAR activity, and regime activity.")
        trace.add("star", "star.count_active_channels", star_payload["count_active_channels"], "Count of active anomaly/regime channels at each timestep.")
        trace.add("star", "star.star_hist_activity", star_payload["star_hist_activity"], "STAR activity tensor for STAR-designated exogenous channels.")
        trace.add("star", "star.star_hist_signed_score", star_payload["star_hist_signed_score"], "Signed STAR robust scores for STAR-designated exogenous channels.")
        trace.add("star", "star.channel_activity", star_payload["channel_activity"], "Concatenated target/STAR/non-STAR/regime activity tensor used by attention and event builders.")
        trace.add("star", "star.event_summary", star_payload["event_summary"], "Event summary vector built from the STAR payload.")
        trace.add("star", "star.event_trajectory", star_payload["event_trajectory"], "Event trajectory vector built from the STAR payload.")
        trace.add("star", "star.non_star_regime", star_payload["non_star_regime"], "Non-STAR regime descriptor vector.")
        trace.add("star", "star.non_star_star_activity", star_payload["non_star_star_activity"], "Activity tensor for non-STAR exogenous channels after their own STAR decomposition.")
        trace.add("star", "star.non_star_regime_activity", star_payload["non_star_regime_activity"], "Non-STAR regime activity tensor aggregated from the non-STAR exogenous channels.")
        trace.add("star", "star.non_star_star_count", star_payload["non_star_star_count"], "Per-timestep active-channel count from non-STAR exogenous STAR paths.")
        trace.add("star", "star.non_star_regime_count", star_payload["non_star_regime_count"], "Per-timestep regime-count tensor from the non-STAR exogenous channels.")
        trace.add("star", "star.regime_intensity", star_payload["regime_intensity"], "Summed non-STAR regime intensity across channels.")
        trace.add("star", "star.regime_density", star_payload["regime_density"], "Normalized non-STAR regime density across channels.")

        encoder_parts = []
        if not model.exclude_insample_y:
            encoder_parts.append(insample_y)
        non_star_hist_exog = model._select_hist_exog(hist_exog, model.non_star_hist_exog_indices)
        if non_star_hist_exog is not None:
            encoder_parts.append(non_star_hist_exog)
        encoder_parts.extend(
            [
                star_payload["target_trend"],
                star_payload["target_seasonal"],
                star_payload["target_anomalies"],
                star_payload["target_residual"],
            ]
        )
        if star_payload["star_hist_anomalies"].size(-1) > 0:
            encoder_parts.extend(
                [
                    star_payload["star_hist_trend"],
                    star_payload["star_hist_seasonal"],
                    star_payload["star_hist_anomalies"],
                    star_payload["star_hist_residual"],
                ]
            )
        encoder_input = torch.cat(encoder_parts, dim=2)
        backbone_states = model.encoder(encoder_input)
        hidden_states = model.encoder.project_to_time_states(backbone_states)
        event_summary = star_payload["event_summary"].to(device=hidden_states.device, dtype=hidden_states.dtype)
        event_trajectory = star_payload["event_trajectory"].to(device=hidden_states.device, dtype=hidden_states.dtype)
        non_star_regime = star_payload["non_star_regime"].to(device=hidden_states.device, dtype=hidden_states.dtype)
        regime_intensity = star_payload["regime_intensity"].to(device=hidden_states.device, dtype=hidden_states.dtype)
        regime_density = star_payload["regime_density"].to(device=hidden_states.device, dtype=hidden_states.dtype)
        critical_mask = star_payload["critical_mask"].bool()
        count_active_channels = star_payload["count_active_channels"].to(device=hidden_states.device, dtype=hidden_states.dtype)
        channel_activity = star_payload["channel_activity"].to(device=hidden_states.device, dtype=hidden_states.dtype)
        attended_states, attention_weights = model.attention(
            hidden_states,
            critical_mask,
            count_active_channels,
            channel_activity,
        )
        if model.informer_decoder is None:
            decoder_input = model._build_time_decoder_input(
                hidden_states=hidden_states,
                attended_states=attended_states,
            )
            delta_forecast = model.decoder(decoder_input)
            final_diff = delta_forecast[:, -model.h :]
            trace.add(
                "decoder",
                "head.decoder_input",
                decoder_input,
                "Shared decoder input passed into the GRU-style AAForecast decoder path.",
            )
            trace.add(
                "forecast",
                "forecast.delta_forecast_diff",
                final_diff,
                "Shared decoder output on the diff scale.",
            )
        else:
            attention_hidden_states = hidden_states + model._project_regime_time_context(regime_intensity, regime_density)
            hidden_aligned, attended_aligned = model._build_time_decoder_features(
                hidden_states=attention_hidden_states,
                attended_states=attended_states,
            )
            regime_time_latent = model._project_regime_time_context(regime_intensity, regime_density)
            regime_time_aligned = _align_horizon(
                regime_time_latent,
                h=model.h,
                input_size=model.input_size,
                sequence_adapter=model.sequence_adapter,
            )
            event_context = model._project_event_summary(event_summary)
            event_path = model._project_event_trajectory(event_trajectory)
            pooled_context = model._build_memory_pooled_context(
                hidden_states=attention_hidden_states,
                attended_states=attended_states,
                event_context=event_context,
                event_path=event_path,
                non_star_regime=non_star_regime,
                regime_intensity=regime_intensity,
                regime_density=regime_density,
            )
            memory_token = getattr(model, "_latest_memory_token", None)
            if memory_token is None:
                memory_token = pooled_context
            memory_bank = getattr(model, "_latest_memory_bank", None)
            memory_signal = getattr(model, "_latest_memory_signal", None)
            decoder_input = torch.cat(
                [hidden_aligned + regime_time_aligned, attended_aligned + regime_time_aligned],
                dim=-1,
            )
            delta_forecast = _trace_informer_head(
                model.informer_decoder,
                decoder_input=decoder_input,
                event_summary=event_context,
                event_path=event_path,
                raw_regime=non_star_regime,
                pooled_context=pooled_context,
                memory_signal=memory_signal,
                anchor_value=insample_y[:, -1, :],
                memory_token=memory_token,
                memory_bank=memory_bank,
                trace=trace,
            )
            final_diff = insample_y[:, -1:, :].to(dtype=delta_forecast.dtype) + delta_forecast
            trace.add("attention", "attention.attention_hidden_states", attention_hidden_states, "Hidden states after adding regime-time conditioning before sparse attention.")
            trace.add("attention", "attention.hidden_aligned", hidden_aligned, "Hidden states aligned from input_size=64 down to horizon=2.")
            trace.add("attention", "attention.attended_aligned", attended_aligned, "Sparse-attended states aligned to the horizon dimension.")
            trace.add("attention", "attention.regime_time_latent", regime_time_latent, "Regime-time context projected into hidden space before horizon alignment.")
            trace.add("attention", "attention.regime_time_aligned", regime_time_aligned, "Regime-time context aligned to the horizon dimension.")
            trace.add("attention", "attention.event_context", event_context, "Projected event summary used by the decoder head.")
            trace.add("attention", "attention.event_path", event_path, "Projected event trajectory used by the decoder head.")
            trace.add("attention", "attention.pooled_context", pooled_context, "Memory-pooled context vector built from hidden states, attended states, and regime signals.")
            if memory_token is not None:
                trace.add("attention", "attention.memory_token", memory_token, "Top memory token selected by the pooled-context builder.")
            if memory_bank is not None:
                trace.add("attention", "attention.memory_bank", memory_bank, "Top-k memory bank gathered by the pooled-context builder.")
            if memory_signal is not None:
                trace.add("attention", "attention.memory_signal", memory_signal, "Mean top-k memory signal used by the decoder gates.")
        final_level = _restore_prediction_series(
            pd.Series(final_diff.detach().cpu().numpy().reshape(-1)),
            bundle.diff_context,
        ).tolist()
        trace.add("encoder", "encoder.encoder_input", encoder_input, "Concatenated encoder input tensor assembled from diff target, non-STAR exogenous channels, and STAR decomposition channels.")
        trace.add("encoder", "encoder.backbone_states", backbone_states, "Informer backbone output states before time projection.")
        trace.add("encoder", "encoder.hidden_states", hidden_states, "Backbone states projected into time-major hidden states.")
        trace.add("attention", "attention.event_summary", event_summary, "Event summary vector entering the Informer attention/decode path.")
        trace.add("attention", "attention.event_trajectory", event_trajectory, "Event trajectory vector entering the Informer attention/decode path.")
        trace.add("attention", "attention.non_star_regime", non_star_regime, "Raw non-STAR regime descriptor vector entering the Informer attention/decode path.")
        trace.add("attention", "attention.regime_intensity", regime_intensity, "Per-timestep regime intensity used for attention-time conditioning.")
        trace.add("attention", "attention.regime_density", regime_density, "Per-timestep regime density used for attention-time conditioning.")
        trace.add("attention", "attention.attention_weights", attention_weights, "Sparse-attention weights returned by the AAForecast CriticalSparseAttention module.")
        trace.add("attention", "attention.attended_states", attended_states, "Sparse-attention output states.")
        trace.add("forecast", "forecast.anchor_diff", insample_y[:, -1:, :], "Last diff-scale input value for the current replay window.")
        trace.add("forecast", "forecast.final_prediction_diff", final_diff, "Current-code replay final model prediction on the diff scale.")
        trace.add("forecast", "forecast.final_prediction_level", final_level, "Current-code replay final model prediction restored back to the raw level scale.")
        return final_diff.detach().cpu().numpy().reshape(-1).tolist(), final_level


def _load_archival_vectors(
    run_root: Path,
    *,
    cutoff: str,
) -> dict[str, Any]:
    result_df = pd.read_csv(run_root / "summary" / "result.csv")
    if "cutoff" in result_df.columns:
        result_df = result_df[result_df["cutoff"].astype(str) == f"{cutoff} 00:00:00"]
    result_df = result_df.sort_values("horizon_step").reset_index(drop=True)
    if result_df.empty:
        raise ValueError(f"No archived result rows found for cutoff {cutoff}")
    uncertainty_json = _read_json(run_root / "aa_forecast" / "uncertainty" / "20260223T000000.json")
    uncertainty_csv = pd.read_csv(run_root / "aa_forecast" / "uncertainty" / "20260223T000000.csv").sort_values("horizon_step")
    candidate_stats = pd.read_csv(run_root / "aa_forecast" / "uncertainty" / "20260223T000000.candidate_stats.csv")
    candidate_samples = pd.read_csv(run_root / "aa_forecast" / "uncertainty" / "20260223T000000.candidate_samples.csv")
    context_json = _read_json(run_root / "aa_forecast" / "context" / "20260223T000000.json")
    manifest = _read_json(run_root / "manifest" / "run_manifest.json")
    resolved = _read_json(run_root / "config" / "config.resolved.json")
    archived_final = result_df["y_hat"].astype(float).tolist()
    archived_actual = result_df["y"].astype(float).tolist()
    archived_std = uncertainty_csv["uncertainty_std"].astype(float).tolist()
    archived_selected_dropout = uncertainty_csv["selected_dropout"].astype(float).tolist()
    return {
        "result_df": result_df,
        "uncertainty_json": uncertainty_json,
        "uncertainty_csv": uncertainty_csv,
        "candidate_stats": candidate_stats,
        "candidate_samples": candidate_samples,
        "context_json": context_json,
        "manifest": manifest,
        "resolved": resolved,
        "archived_final": archived_final,
        "archived_actual": archived_actual,
        "archived_std": archived_std,
        "archived_selected_dropout": archived_selected_dropout,
        "archived_horizon": int(len(archived_final)),
    }


def _candidate_sample_payload(frame: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dropout_value, dropout_frame in frame.groupby("dropout_p"):
        ordered = dropout_frame.sort_values(["sample_idx", "horizon_step"])
        samples: list[list[float]] = []
        for sample_idx, sample_frame in ordered.groupby("sample_idx"):
            _ = sample_idx
            samples.append(sample_frame["prediction"].astype(float).tolist())
        payload[str(dropout_value)] = samples
    return payload


def _current_uncertainty_payload(bundle: ReplayBundle) -> dict[str, Any]:
    return {
        "selected_mean": np.asarray(bundle.uncertainty_summary["mean"], dtype=float).tolist(),
        "selected_std": np.asarray(bundle.uncertainty_summary["std"], dtype=float).tolist(),
        "selected_dropout": np.asarray(bundle.uncertainty_summary["selected_dropout"], dtype=float).tolist(),
        "candidate_mean_grid": np.asarray(bundle.uncertainty_summary["candidate_mean_grid"], dtype=float).tolist(),
        "candidate_std_grid": np.asarray(bundle.uncertainty_summary["candidate_std_grid"], dtype=float).tolist(),
        "candidate_dropout_values": np.asarray(bundle.uncertainty_summary["candidate_dropout_values"], dtype=float).tolist(),
        "selection_mode": str(bundle.uncertainty_summary["selection_mode"]),
        "candidate_spike_support": np.asarray(bundle.uncertainty_summary["candidate_spike_support"], dtype=float).tolist(),
        "candidate_baseline_drag": np.asarray(bundle.uncertainty_summary["candidate_baseline_drag"], dtype=float).tolist(),
        "candidate_direction_mean": np.asarray(bundle.uncertainty_summary["candidate_direction_mean"], dtype=float).tolist(),
        "candidate_dispersion_scores": np.asarray(bundle.uncertainty_summary["candidate_dispersion_scores"], dtype=float).tolist(),
        "candidate_semantic_scores": np.asarray(bundle.uncertainty_summary["candidate_semantic_scores"], dtype=float).tolist(),
        "candidate_path_scores": np.asarray(bundle.uncertainty_summary["candidate_path_scores"], dtype=float).tolist(),
        "candidate_samples": _to_native(bundle.uncertainty_summary["candidate_samples"]),
    }


def _build_provenance(
    *,
    repo_root: Path,
    config_path: Path,
    run_root: Path,
    bundle: ReplayBundle,
    archived: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    current_selected_mean = np.asarray(bundle.uncertainty_summary["mean"], dtype=float)
    archived_final = np.asarray(archived["archived_final"], dtype=float)
    current_horizon = int(current_selected_mean.shape[0])
    archived_horizon = int(archived["archived_horizon"])
    horizon_mismatch = current_horizon != archived_horizon
    if horizon_mismatch:
        selected_abs_diff: list[float] | None = None
        identity_lock_passed = False
    else:
        selected_abs_diff = np.abs(current_selected_mean - archived_final).tolist()
        identity_lock_passed = bool(np.all(np.asarray(selected_abs_diff, dtype=float) <= 1e-9))
    blocker = None
    if horizon_mismatch:
        blocker = (
            f"Run-root drift detected: archived result horizon={archived_horizon} but current replay/config horizon={current_horizon}. "
            "The run root no longer matches the requested final-window contract, so exact historical internal tensors cannot be proven from this path."
        )
    elif not identity_lock_passed:
        blocker = (
            "No archived fitted-state checkpoint exists under the completed run root, and a fresh current-code replay does not reproduce the archived final forecast exactly. "
            "Exact internal tensors for the historical run therefore cannot be proven from the saved artifacts alone."
        )
    return {
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "run_root": str(run_root),
        "target_cutoff": "2026-02-23",
        "horizon": int(bundle.loaded.config.cv.horizon),
        "input_size": int(bundle.loaded.config.training.input_size),
        "fitted_state_source": str(checkpoint_path),
        "archived_result_path": str(run_root / "summary" / "result.csv"),
        "archived_uncertainty_path": str(run_root / "aa_forecast" / "uncertainty" / "20260223T000000.json"),
        "archived_context_path": str(run_root / "aa_forecast" / "context" / "20260223T000000.json"),
        "retrieval_enabled": bool(bundle.loaded.config.stage_plugin_config.retrieval.enabled),
        "eval_mode": bool(not bundle.model.training),
        "device": str(next(bundle.model.parameters()).device),
        "dtype": str(next(bundle.model.parameters()).dtype),
        "seed_controls": {"runtime_random_seed": int(bundle.loaded.config.runtime.random_seed)},
        "archived_horizon": archived_horizon,
        "current_replay_horizon": current_horizon,
        "archived_final_prediction_level": archived["archived_final"],
        "current_replay_deterministic_prediction_level": bundle.deterministic_level_predictions,
        "current_replay_selected_mean_level": current_selected_mean.tolist(),
        "current_replay_selected_std_level": np.asarray(bundle.uncertainty_summary["std"], dtype=float).tolist(),
        "current_replay_selected_dropout": np.asarray(bundle.uncertainty_summary["selected_dropout"], dtype=float).tolist(),
        "archived_selected_dropout": archived["archived_selected_dropout"],
        "archived_selected_std": archived["archived_std"],
        "selected_mean_abs_diff_vs_archived": selected_abs_diff,
        "identity_lock_passed": identity_lock_passed,
        "blocker": blocker,
    }


def _build_completeness_manifest(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    manifest: dict[str, list[str]] = {}
    for record in records:
        manifest.setdefault(record["stage"], []).append(record["tensor_name"])
    return manifest


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    provenance = payload["provenance"]
    archived = payload["archival"]
    lines.append("# AAForecast Informer 마지막 window 실제 계산 문서")
    lines.append("")
    lines.append("이 문서는 `runs/feature_set_aaforecast_brentoil_case1_parity_aaforecast_informer` 의 **마지막 cutoff 1개 (`2026-02-23`)** 만 다룬다.")
    lines.append("실제 archived artifact에서 확인 가능한 값과, 현재 코드 기준 replay로 추출한 내부 tensor를 분리해서 적는다.")
    lines.append("")
    lines.append("## 0. Provenance lock")
    lines.append("")
    lines.append(f"- identity_lock_passed: **{provenance['identity_lock_passed']}**")
    lines.append(f"- retrieval_enabled: `{provenance['retrieval_enabled']}`")
    lines.append(f"- fitted_state_source: `{provenance['fitted_state_source']}`")
    lines.append(f"- archived_horizon: `{provenance['archived_horizon']}`")
    lines.append(f"- current_replay_horizon: `{provenance['current_replay_horizon']}`")
    lines.append(f"- archived final prediction: `{provenance['archived_final_prediction_level']}`")
    lines.append(f"- current replay deterministic prediction: `{provenance['current_replay_deterministic_prediction_level']}`")
    lines.append(f"- current replay selected mean prediction: `{provenance['current_replay_selected_mean_level']}`")
    lines.append(f"- selected mean abs diff vs archived: `{provenance['selected_mean_abs_diff_vs_archived']}`")
    if provenance["blocker"]:
        lines.append("")
        lines.append("> blocker: archived run root에는 fitted-state checkpoint가 없고, 현재 코드 replay도 archived 최종 예측을 정확히 재현하지 못했다.")
        lines.append("> 따라서 아래 내부 tensor는 **현재 코드 replay trace** 이고, archived 최종 예측 자체와 완전 동일하다고 증명된 tensor는 아니다.")
        lines.append(f"> detail: {provenance['blocker']}")
    lines.append("")
    lines.append("## 1. Archived final outputs (exact) ")
    lines.append("")
    lines.append(f"- archived actual future level target: `{archived['archived_actual']}`")
    lines.append(f"- archived final prediction level: `{archived['archived_final']}`")
    lines.append(f"- archived selected dropout: `{archived['archived_selected_dropout']}`")
    lines.append(f"- archived selected std: `{archived['archived_std']}`")
    lines.append(f"- archived context summary: `{archived['context_json']}`")
    lines.append("")
    lines.append("## 2. Canonical trace schema")
    lines.append("")
    lines.append("각 trace item은 `(stage, tensor_name, shape, dtype, meaning, payload, ordering)` 구조를 가진다.")
    lines.append("ordering은 문서 출력 순서이며, tensor ordering semantics는 기본적으로 **batch -> timestep -> feature** 이다.")
    lines.append("")
    lines.append("## 3. Completeness manifest")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload["completeness_manifest"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    stage_titles = {
        "raw_rows": "4. Raw rows",
        "transformed_inputs": "5. Transformed inputs",
        "star": "6. STAR decomposition and event payload",
        "encoder": "7. Encoder tensors",
        "attention": "8. Attention tensors",
        "decoder": "9. Decoder tensors",
        "forecast": "10. Forecast tensors and uncertainty",
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in payload["records"]:
        grouped.setdefault(record["stage"], []).append(record)
    for stage in ["raw_rows", "transformed_inputs", "star", "encoder", "attention", "decoder", "forecast"]:
        records = grouped.get(stage, [])
        if not records:
            continue
        lines.append(f"## {stage_titles[stage]}")
        lines.append("")
        for record in records:
            lines.append(f"### {record['tensor_name']}")
            lines.append("")
            lines.append(f"- shape: `{record['shape']}`")
            lines.append(f"- dtype: `{record['dtype']}`")
            lines.append(f"- meaning: {record['meaning']}")
            lines.append(f"- ordering: `{record['ordering']}`")
            lines.append("```json")
            lines.append(json.dumps(record["payload"], ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def generate_last_window_doc(
    *,
    repo_root: Path,
    config_path: Path,
    run_root: Path,
    output_md: Path,
    output_json: Path,
    output_checkpoint: Path,
    cutoff: str = "2026-02-23",
) -> dict[str, Any]:
    cutoff_ts = pd.Timestamp(cutoff)
    archived = _load_archival_vectors(run_root, cutoff=cutoff)
    bundle = _build_replay_bundle(repo_root=repo_root, config_path=config_path, cutoff=cutoff_ts)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    bundle.model.save(str(output_checkpoint))

    trace = TraceBuilder()
    _add_raw_and_transformed_records(bundle, trace)
    _trace_current_replay(bundle, trace)
    trace.add("forecast", "forecast.archived_selected_mean_level", archived["archived_final"], "Exact archived final level predictions from the completed run.")
    trace.add("forecast", "forecast.archived_selected_std_level", archived["archived_std"], "Exact archived uncertainty std values from the completed run.")
    trace.add("forecast", "forecast.archived_selected_dropout", archived["archived_selected_dropout"], "Exact archived selected dropout values from the completed run.")
    trace.add("forecast", "forecast.archived_candidate_stats_rows", archived["candidate_stats"], "Exact archived candidate statistics for every dropout candidate.")
    trace.add("forecast", "forecast.archived_candidate_samples", _candidate_sample_payload(archived["candidate_samples"]), "Exact archived per-dropout prediction samples stored by the completed run.")
    trace.add("forecast", "forecast.current_replay_uncertainty", _current_uncertainty_payload(bundle), "Current-code replay uncertainty-selection payload returned by `_select_uncertainty_predictions`.")

    provenance = _build_provenance(
        repo_root=repo_root,
        config_path=config_path,
        run_root=run_root,
        bundle=bundle,
        archived=archived,
        checkpoint_path=output_checkpoint,
    )
    payload = {
        "schema_version": 1,
        "ordering_semantics": "batch-major then timestep-major then feature-major; row-oriented records remain chronological.",
        "trace_schema_fields": list(TRACE_SCHEMA_FIELDS),
        "provenance": provenance,
        "archival": {
            "archived_actual": archived["archived_actual"],
            "archived_final": archived["archived_final"],
            "archived_std": archived["archived_std"],
            "archived_selected_dropout": archived["archived_selected_dropout"],
            "context_json": archived["context_json"],
            "manifest": archived["manifest"],
            "resolved": archived["resolved"],
        },
        "records": trace.records,
        "completeness_manifest": _build_completeness_manifest(trace.records),
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _render_markdown(payload)
    output_md.write_text(markdown, encoding="utf-8")
    return {
        "output_md": str(output_md),
        "output_json": str(output_json),
        "output_checkpoint": str(output_checkpoint),
        "identity_lock_passed": provenance["identity_lock_passed"],
        "selected_mean_abs_diff_vs_archived": provenance["selected_mean_abs_diff_vs_archived"],
    }
