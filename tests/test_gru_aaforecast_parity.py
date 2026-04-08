from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

import runtime_support.runner as runtime
from neuralforecast.models import AAForecast, GRU


REPO_ROOT = Path(__file__).resolve().parents[1]
GRU_AUTO_PARITY_CONFIG = Path("tests/fixtures/gru_runtime_auto_parity_smoke.yaml")
AA_AUTO_PARITY_CONFIG = Path("tests/fixtures/aa_forecast_runtime_auto_model_only_smoke.yaml")
PARITY_SHARED_MODEL_SELECTORS = [
    "encoder_hidden_size",
    "encoder_n_layers",
    "encoder_dropout",
    "decoder_hidden_size",
    "decoder_layers",
]


def _windows_batch(batch_size: int = 2, input_size: int = 4) -> dict[str, torch.Tensor]:
    insample_y = torch.arange(batch_size * input_size, dtype=torch.float32).reshape(
        batch_size, input_size, 1
    )
    hist_exog = torch.linspace(
        0.0, 1.0, steps=batch_size * input_size, dtype=torch.float32
    ).reshape(batch_size, input_size, 1)
    return {
        "insample_y": insample_y,
        "hist_exog": hist_exog,
        "futr_exog": torch.empty(batch_size, input_size, 0),
        "stat_exog": torch.empty(batch_size, 0),
    }


def test_gru_parity_forward_shape_matches_aaforecast() -> None:
    torch.manual_seed(7)
    kwargs = dict(
        h=2,
        input_size=4,
        encoder_hidden_size=8,
        encoder_n_layers=2,
        encoder_dropout=0.1,
        decoder_hidden_size=8,
        decoder_layers=2,
        hist_exog_list=["event"],
        star_hist_exog_list=["event"],
        non_star_hist_exog_list=[],
        star_hist_exog_tail_modes=["upward"],
        scaler_type="identity",
        max_steps=1,
        val_check_steps=1,
        batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
    )
    gru = GRU(**kwargs)
    aaforecast = AAForecast(**kwargs)
    windows_batch = _windows_batch()

    gru.eval()
    aaforecast.eval()

    gru_out = gru(windows_batch)
    aa_out = aaforecast(windows_batch)

    assert gru_out.shape == aa_out.shape == (2, 2, 1)
    assert gru.decoder.layers[0].in_features == aaforecast.decoder.layers[0].in_features
    assert gru.sequence_adapter is None
    assert aaforecast.sequence_adapter is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"recurrent": True}, "recurrent=True"),
        ({"futr_exog_list": ["future_a"]}, "futr_exog_list"),
        ({"stat_exog_list": ["static_a"]}, "stat_exog_list"),
    ],
)
def test_gru_parity_rejects_unsupported_legacy_paths(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GRU(h=2, input_size=4, **kwargs)


def test_search_space_parity_selectors_align_between_gru_and_aaforecast() -> None:
    payload = yaml.safe_load((REPO_ROOT / "yaml/HPO/search_space.yaml").read_text())
    assert list(payload["models"]["GRU"]) == PARITY_SHARED_MODEL_SELECTORS
    assert list(payload["models"]["AAForecast"]) == PARITY_SHARED_MODEL_SELECTORS
    for selector in PARITY_SHARED_MODEL_SELECTORS:
        assert payload["models"]["GRU"][selector] == payload["models"]["AAForecast"][selector]


def test_runtime_validate_only_accepts_gru_auto_parity_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output_root = tmp_path / "validate-only-gru-auto-parity"
    code = runtime.main(
        [
            "--config",
            str(GRU_AUTO_PARITY_CONFIG),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    assert len(manifest["jobs"]) == 1
    assert manifest["jobs"][0]["model"] == "GRU"
    assert manifest["jobs"][0]["requested_mode"] == "learned_auto_requested"
    assert manifest["jobs"][0]["validated_mode"] == "learned_auto"
    assert manifest["jobs"][0]["selected_search_params"] == PARITY_SHARED_MODEL_SELECTORS
    assert manifest["training_search"] == {
        "requested_mode": "training_fixed",
        "validated_mode": "training_fixed",
        "selected_search_params": [],
    }


def test_runtime_validate_only_gru_and_aaforecast_share_parity_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gru_output_root = tmp_path / "validate-only-gru-parity-compare"
    aa_output_root = tmp_path / "validate-only-aa-parity-compare"

    gru_code = runtime.main(
        [
            "--config",
            str(GRU_AUTO_PARITY_CONFIG),
            "--output-root",
            str(gru_output_root),
            "--validate-only",
        ]
    )
    aa_code = runtime.main(
        [
            "--config",
            str(AA_AUTO_PARITY_CONFIG),
            "--output-root",
            str(aa_output_root),
            "--validate-only",
        ]
    )

    assert gru_code == 0
    assert aa_code == 0

    gru_manifest = json.loads((gru_output_root / "manifest" / "run_manifest.json").read_text())
    aa_manifest = json.loads((aa_output_root / "manifest" / "run_manifest.json").read_text())

    assert gru_manifest["jobs"][0]["selected_search_params"] == PARITY_SHARED_MODEL_SELECTORS
    assert aa_manifest["jobs"][0]["selected_search_params"] == PARITY_SHARED_MODEL_SELECTORS
