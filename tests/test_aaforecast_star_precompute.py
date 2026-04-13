from __future__ import annotations

import pandas as pd
import pytest
import torch

from neuralforecast.models import AAForecast
from plugins.aa_forecast import CriticalSparseAttention


def _build_model() -> AAForecast:
    model = AAForecast(
        h=2,
        input_size=4,
        encoder_hidden_size=8,
        encoder_n_layers=1,
        encoder_dropout=0.1,
        decoder_hidden_size=8,
        decoder_layers=1,
        attention_hidden_size=4,
        season_length=2,
        lowess_frac=0.6,
        lowess_delta=0.01,
        thresh=3.5,
        hist_exog_list=["event", "macro"],
        star_hist_exog_list=["event"],
        non_star_hist_exog_list=["macro"],
        star_hist_exog_tail_modes=["upward"],
        max_steps=1,
        val_check_steps=1,
        batch_size=1,
        valid_batch_size=1,
        windows_batch_size=4,
        inference_windows_batch_size=4,
        scaler_type="robust",
    )
    model.val_size = 2
    model.test_size = 2
    model.predict_step_size = 1
    model.set_star_precompute_context(enabled=True, fold_key="fold-1")
    return model


def _build_batch() -> dict[str, object]:
    rows = 12
    frame = pd.DataFrame(
        {
            "y": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 50.0, 60.0, 70.0, 80.0],
            "event": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 90.0, 91.0, 92.0, 93.0],
            "macro": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 40.0, 41.0, 42.0, 43.0],
            "available_mask": [1.0] * rows,
        }
    )
    temporal = torch.tensor(frame.to_numpy().T, dtype=torch.float32).unsqueeze(0)
    return {
        "temporal": temporal,
        "temporal_cols": pd.Index(["y", "event", "macro", "available_mask"]),
        "y_idx": 0,
        "static": None,
        "static_cols": None,
    }


@pytest.mark.parametrize(
    ("phase", "w_idxs"),
    [
        ("train", [1, 0]),
        ("val", [0]),
        ("predict", [0]),
    ],
)
def test_sample_windows_preserves_window_ids_for_phase_paths(
    phase: str,
    w_idxs: list[int],
) -> None:
    model = _build_model()
    batch = _build_batch()

    windows_temporal, static, static_cols, final_condition = model._create_windows(
        batch,
        step=phase,
    )
    w_idx_tensor = torch.tensor(w_idxs, device=windows_temporal.device)
    windows = model._sample_windows(
        windows_temporal=windows_temporal,
        static=static,
        static_cols=static_cols,
        temporal_cols=batch["temporal_cols"],
        w_idxs=w_idx_tensor,
        final_condition=final_condition,
    )

    expected_window_ids = final_condition[w_idx_tensor]

    assert "window_ids" in windows
    assert torch.equal(windows["window_ids"], expected_window_ids)
    assert torch.equal(windows["temporal"], windows_temporal[expected_window_ids])


def test_star_precompute_validation_cache_reuses_phase_payload() -> None:
    model = _build_model()
    batch = _build_batch()

    windows_temporal, static, static_cols, final_condition = model._create_windows(
        batch,
        step="val",
    )
    w_idxs = torch.arange(len(final_condition), device=windows_temporal.device)
    windows = model._sample_windows(
        windows_temporal=windows_temporal,
        static=static,
        static_cols=static_cols,
        temporal_cols=batch["temporal_cols"],
        w_idxs=w_idxs,
        final_condition=final_condition,
    )
    payload1 = model.get_star_precomputed(
        batch=batch,
        phase="val",
        window_ids=windows["window_ids"],
        device=windows_temporal.device,
        dtype=windows_temporal.dtype,
    )
    cache_keys_after_first = set(model._star_phase_cache)
    payload2 = model.get_star_precomputed(
        batch=batch,
        phase="val",
        window_ids=windows["window_ids"],
        device=windows_temporal.device,
        dtype=windows_temporal.dtype,
    )

    assert cache_keys_after_first == {"val"}
    assert set(model._star_phase_cache) == cache_keys_after_first
    assert payload1 is not None and payload2 is not None
    for key in (
        "target_trend",
        "target_seasonal",
        "target_anomalies",
        "target_residual",
        "critical_mask",
        "count_active_channels",
        "channel_activity",
        "event_summary",
    ):
        assert key in payload1
        assert torch.equal(payload1[key], payload2[key])


def test_validation_phase_cache_excludes_test_rows() -> None:
    model = _build_model()
    batch = _build_batch()

    payload = model.get_star_precomputed(
        batch=batch,
        phase="val",
        window_ids=torch.tensor([0], dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    val_cache = model._star_phase_cache["val"]

    windows_temporal, static, static_cols, final_condition = model._create_windows(
        batch,
        step="val",
    )
    windows = model._sample_windows(
        windows_temporal=windows_temporal,
        static=static,
        static_cols=static_cols,
        temporal_cols=batch["temporal_cols"],
        w_idxs=torch.arange(len(final_condition), device=windows_temporal.device),
        final_condition=final_condition,
    )
    observed_targets = windows["temporal"][:, :, 0, 0].reshape(-1).tolist()

    assert payload is not None
    assert val_cache["window_ids"].numel() == len(final_condition)
    assert 70.0 not in observed_targets
    assert 80.0 not in observed_targets
    assert 50.0 in observed_targets
    assert 60.0 in observed_targets


def test_predict_phase_star_precompute_is_disabled() -> None:
    model = _build_model()
    batch = _build_batch()
    payload = model.get_star_precomputed(
        batch=batch,
        phase="predict",
        window_ids=torch.tensor([0], dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert payload is None


def test_star_precompute_supports_zero_anomaly_windows() -> None:
    model = _build_model()
    model.star.thresh = 1_000.0
    batch = _build_batch()
    batch["temporal"] = torch.ones_like(batch["temporal"])

    payload = model.get_star_precomputed(
        batch=batch,
        phase="val",
        window_ids=torch.tensor([0], dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert payload is not None
    assert torch.count_nonzero(payload["critical_mask"]) == 0
    assert torch.count_nonzero(payload["count_active_channels"]) == 0
    assert torch.count_nonzero(payload["channel_activity"]) == 0
    assert torch.count_nonzero(payload["event_summary"]) == 0


def test_event_summary_tracks_recent_upward_activity_strength() -> None:
    model = _build_model()
    earlier_payload = {
        "critical_mask": torch.tensor([[[True], [False], [False], [False]]]),
        "count_active_channels": torch.tensor([[[2.0], [0.0], [0.0], [0.0]]]),
        "channel_activity": torch.tensor(
            [[[3.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "target_activity": torch.tensor(
            [[[3.0], [0.0], [0.0], [0.0]]],
            dtype=torch.float32,
        ),
        "star_hist_activity": torch.tensor(
            [[[0.0], [0.0], [0.0], [0.0]]],
            dtype=torch.float32,
        ),
    }
    later_payload = {
        "critical_mask": torch.tensor([[[False], [False], [False], [True]]]),
        "count_active_channels": torch.tensor([[[0.0], [0.0], [0.0], [2.0]]]),
        "channel_activity": torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 0.0]]],
            dtype=torch.float32,
        ),
        "target_activity": torch.tensor(
            [[[0.0], [0.0], [0.0], [3.0]]],
            dtype=torch.float32,
        ),
        "star_hist_activity": torch.tensor(
            [[[0.0], [0.0], [0.0], [0.0]]],
            dtype=torch.float32,
        ),
    }

    earlier = model._build_event_summary_from_payload(
        earlier_payload,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    later = model._build_event_summary_from_payload(
        later_payload,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert earlier.shape == (1, model.EVENT_SUMMARY_SIZE)
    assert later.shape == (1, model.EVENT_SUMMARY_SIZE)
    assert later[0, 1] > earlier[0, 1]
    assert later[0, 5] > earlier[0, 5]


def test_count_active_channels_preserves_multi_channel_density() -> None:
    template = torch.zeros((1, 4, 1), dtype=torch.float32)
    mask = torch.tensor(
        [
            [
                [False, False, False],
                [True, False, True],
                [True, True, True],
                [False, True, False],
            ]
        ],
        dtype=torch.bool,
    )

    counts = AAForecast._count_active_channels(mask, template=template)
    reduced_mask = AAForecast._reduce_critical_mask(mask, template=template)

    assert torch.equal(
        counts,
        torch.tensor([[[0.0], [2.0], [3.0], [1.0]]], dtype=torch.float32),
    )
    assert torch.equal(reduced_mask, counts > 0)


def test_critical_sparse_attention_uses_count_signal() -> None:
    attention = CriticalSparseAttention(hidden_size=2, attention_hidden_size=2)
    with torch.no_grad():
        attention.proj.weight.zero_()
        attention.proj.bias.zero_()
        attention.score.weight.zero_()
        attention.score.bias.zero_()

    hidden_states = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    critical_mask = torch.ones((1, 3, 1), dtype=torch.bool)
    low_counts = torch.ones((1, 3, 1), dtype=torch.float32)
    high_counts = torch.tensor(
        [[[1.0], [3.0], [1.0]]],
        dtype=torch.float32,
    )
    channel_activity = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )

    attended_low, weights_low = attention(
        hidden_states,
        critical_mask,
        low_counts,
        channel_activity,
    )
    attended_high, weights_high = attention(
        hidden_states,
        critical_mask,
        high_counts,
        channel_activity,
    )

    assert weights_high[0, 0, 1] > weights_low[0, 0, 1]
    assert not torch.allclose(attended_high, attended_low)


def test_critical_sparse_attention_uses_channel_activity_bank() -> None:
    attention = CriticalSparseAttention(hidden_size=2, attention_hidden_size=2)
    with torch.no_grad():
        attention.proj.weight.zero_()
        attention.proj.bias.zero_()
        attention.score.weight.zero_()
        attention.score.bias.zero_()

    hidden_states = torch.tensor(
        [[[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    critical_mask = torch.ones((1, 3, 1), dtype=torch.bool)
    count_active_channels = torch.ones((1, 3, 1), dtype=torch.float32)
    first_identity = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    swapped_identity = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )

    attended_first, weights_first = attention(
        hidden_states,
        critical_mask,
        count_active_channels,
        first_identity,
    )
    attended_swapped, weights_swapped = attention(
        hidden_states,
        critical_mask,
        count_active_channels,
        swapped_identity,
    )

    assert not torch.allclose(attended_first, attended_swapped)
    assert not torch.allclose(weights_first, weights_swapped)


def test_star_precompute_preserves_active_scaler_state() -> None:
    model = _build_model()
    batch = _build_batch()

    windows_temporal, static, static_cols, final_condition = model._create_windows(
        batch,
        step="val",
    )
    windows = model._sample_windows(
        windows_temporal=windows_temporal,
        static=static,
        static_cols=static_cols,
        temporal_cols=batch["temporal_cols"],
        w_idxs=torch.tensor([0], device=windows_temporal.device),
        final_condition=final_condition,
    )
    windows = model._normalization(windows=windows, y_idx=batch["y_idx"])
    pre_shift = model.scaler.x_shift.detach().clone()
    pre_scale = model.scaler.x_scale.detach().clone()
    (
        insample_y,
        insample_mask,
        _outsample_y,
        _outsample_mask,
        hist_exog,
        futr_exog,
        stat_exog,
    ) = model._parse_windows(batch, windows)
    windows_batch = dict(
        insample_y=insample_y,
        insample_mask=insample_mask,
        futr_exog=futr_exog,
        hist_exog=hist_exog,
        stat_exog=stat_exog,
    )

    materialized = model._maybe_attach_star_precomputed(
        batch,
        windows,
        windows_batch,
        phase="val",
    )

    assert "star_precomputed" in materialized
    assert torch.equal(model.scaler.x_shift, pre_shift)
    assert torch.equal(model.scaler.x_scale, pre_scale)


def test_forward_uses_precomputed_payload_without_calling_star(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model()
    batch_size = 2
    seq_len = model.input_size
    insample_y = torch.ones(batch_size, seq_len, 1)
    hist_exog = torch.cat(
        [
            torch.full((batch_size, seq_len, 1), 9.0),
            torch.full((batch_size, seq_len, 1), 8.0),
        ],
        dim=2,
    )
    star_precomputed = {
        "target_trend": torch.full((batch_size, seq_len, 1), 1.0),
        "target_seasonal": torch.full((batch_size, seq_len, 1), 2.0),
        "target_anomalies": torch.full((batch_size, seq_len, 1), 3.0),
        "target_residual": torch.full((batch_size, seq_len, 1), 4.0),
        "star_hist_trend": torch.full((batch_size, seq_len, 1), 5.0),
        "star_hist_seasonal": torch.full((batch_size, seq_len, 1), 6.0),
        "star_hist_anomalies": torch.full((batch_size, seq_len, 1), 7.0),
        "star_hist_residual": torch.full((batch_size, seq_len, 1), 8.0),
        "critical_mask": torch.ones(batch_size, seq_len, 1, dtype=torch.bool),
        "count_active_channels": torch.ones(batch_size, seq_len, 1),
        "channel_activity": torch.ones(batch_size, seq_len, 2),
    }
    captured: dict[str, torch.Tensor] = {}

    def _raise(*args, **kwargs):
        raise AssertionError(
            "STAR should not be called when star_precomputed is provided"
        )

    original_encoder_forward = model.encoder.forward

    def _capture_encoder_input(x: torch.Tensor) -> torch.Tensor:
        captured["encoder_input"] = x.detach().clone()
        return original_encoder_forward(x)

    monkeypatch.setattr(model.star, "forward", _raise)
    monkeypatch.setattr(model.encoder, "forward", _capture_encoder_input)
    output = model(
        {
            "insample_y": insample_y,
            "insample_mask": torch.ones_like(insample_y),
            "hist_exog": hist_exog,
            "futr_exog": None,
            "stat_exog": None,
            "star_precomputed": star_precomputed,
        }
    )

    assert output.shape[0] == batch_size
    assert output.shape[1] == model.h
    encoder_input = captured["encoder_input"]
    assert encoder_input.shape == (batch_size, seq_len, 10)
    expected_channels = (1.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    for index, expected in enumerate(expected_channels):
        assert torch.allclose(
            encoder_input[:, :, index],
            torch.full((batch_size, seq_len), expected),
        )
