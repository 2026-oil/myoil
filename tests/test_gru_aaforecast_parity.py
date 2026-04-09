from __future__ import annotations

import csv
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
BRENT_CASE1_PARITY_GRU_CONFIG = Path(
    "yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-gru.yaml"
)
BRENT_CASE1_PARITY_AA_CONFIG = Path(
    "yaml/experiment/feature_set_aaforecast/brentoil-case1-parity-aaforecast.yaml"
)
BRENT_CASE1_PARITY_AA_PLUGIN = Path(
    "yaml/plugins/aa_forecast_brentoil_case1_parity.yaml"
)
PARITY_SHARED_MODEL_SELECTORS = [
    "encoder_hidden_size",
    "encoder_n_layers",
    "encoder_dropout",
    "decoder_hidden_size",
    "decoder_layers",
]
BRENT_CASE1_PARITY_PARAMS = {
    "encoder_hidden_size": 128,
    "encoder_n_layers": 3,
    "encoder_dropout": 0.1,
    "decoder_hidden_size": 128,
    "decoder_layers": 2,
}
BRENT_CASE1_PARITY_AA_PARAMS = {
    "encoder_hidden_size": 128,
    "encoder_n_layers": 4,
    "encoder_dropout": 0.1,
    "decoder_hidden_size": 128,
    "decoder_layers": 4,
    "season_length": 52,
}
BRENT_CASE1_PARITY_HIST_EXOG = [
    "Idx_OVX",
    "Com_Oil_Spread",
    "BS_Core_Index_A",
    "BS_Core_Index_B",
    "BS_Core_Index_C",
    "Com_LMEX",
    "Com_BloombergCommodity_BCOM",
    "GPRD_THREAT",
    "GPRD",
    "GPRD_ACT",
]
BRENT_CASE1_PARITY_AA_HIST_EXOG = [
    "GPRD_THREAT",
    "BS_Core_Index_A",
    "GPRD",
    "GPRD_ACT",
    "BS_Core_Index_B",
    "BS_Core_Index_C",
    "Idx_OVX",
    "Com_Oil_Spread",
    "Com_LMEX",
    "Com_BloombergCommodity_BCOM",
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


def test_gru_parity_horizon_context_varies_across_steps() -> None:
    gru = GRU(
        h=4,
        input_size=4,
        encoder_hidden_size=8,
        encoder_n_layers=2,
        encoder_dropout=0.1,
        decoder_hidden_size=8,
        decoder_layers=2,
        hist_exog_list=["event"],
        max_steps=1,
        val_check_steps=1,
        batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
    )
    context = gru._build_horizon_context(batch_size=2, device=torch.device("cpu"))
    assert context.shape == (2, 4, gru.encoder_hidden_size)
    assert not torch.allclose(context[:, 0, :], context[:, 1, :])


def test_aaforecast_encoder_uses_only_non_star_raw_hist_exog() -> None:
    model = AAForecast(
        h=2,
        input_size=4,
        encoder_hidden_size=8,
        encoder_n_layers=2,
        encoder_dropout=0.1,
        decoder_hidden_size=8,
        decoder_layers=2,
        hist_exog_list=["event", "macro"],
        star_hist_exog_list=["event"],
        non_star_hist_exog_list=["macro"],
        star_hist_exog_tail_modes=["upward"],
        scaler_type="identity",
        max_steps=1,
        val_check_steps=1,
        batch_size=1,
        windows_batch_size=1,
        inference_windows_batch_size=1,
    )

    assert model.encoder.input_size == 1 + 1 + 4 + 2 * 1


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


def test_gru_real_run_forecasts_are_not_flat_within_cutoff() -> None:
    forecast_path = REPO_ROOT / "runs/feature_set_aaforecast_brentoil_case1_parity_gru/cv/GRU_forecasts.csv"
    if not forecast_path.exists():
        pytest.skip("real-run artifact not present")

    with forecast_path.open() as f:
        rows = list(csv.DictReader(f))

    by_cutoff: dict[str, set[str]] = {}
    for row in rows:
        by_cutoff.setdefault(row["cutoff"], set()).add(row["y_hat"])

    assert by_cutoff, "expected at least one cutoff in GRU real-run forecasts"
    assert all(len(values) > 1 for values in by_cutoff.values())


def test_brent_case1_parity_experiment_configs_share_dataset_contract() -> None:
    gru_payload = yaml.safe_load((REPO_ROOT / BRENT_CASE1_PARITY_GRU_CONFIG).read_text())
    aa_payload = yaml.safe_load((REPO_ROOT / BRENT_CASE1_PARITY_AA_CONFIG).read_text())
    aa_plugin_payload = yaml.safe_load((REPO_ROOT / BRENT_CASE1_PARITY_AA_PLUGIN).read_text())

    for payload in (gru_payload, aa_payload):
        assert payload["dataset"]["path"] == "data/df.csv"
        assert payload["dataset"]["target_col"] == "Com_BrentCrudeOil"
        assert payload["dataset"]["dt_col"] == "dt"
        assert payload["dataset"]["futr_exog_cols"] == []
        assert payload["dataset"]["static_exog_cols"] == []
        assert payload["training_search"] == {"enabled": False}

    assert gru_payload["dataset"]["hist_exog_cols"] == BRENT_CASE1_PARITY_HIST_EXOG
    assert aa_payload["dataset"]["hist_exog_cols"] == BRENT_CASE1_PARITY_AA_HIST_EXOG

    assert gru_payload["jobs"] == [{"model": "GRU", "params": BRENT_CASE1_PARITY_PARAMS}]
    assert aa_payload["aa_forecast"]["enabled"] is True
    assert aa_payload["aa_forecast"]["config_path"] == str(BRENT_CASE1_PARITY_AA_PLUGIN)
    assert aa_plugin_payload["aa_forecast"]["model"] == "gru"
    assert aa_plugin_payload["aa_forecast"]["tune_training"] is True
    assert aa_plugin_payload["aa_forecast"]["model_params"] == BRENT_CASE1_PARITY_AA_PARAMS
    assert aa_plugin_payload["aa_forecast"]["uncertainty"] == {
        "enabled": True,
        "sample_count": 50,
    }
    assert aa_plugin_payload["aa_forecast"]["top_k"] == 0.1


def test_runtime_validate_only_accepts_brent_case1_fixed_parity_experiments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    gru_output_root = tmp_path / "validate-only-brent-case1-parity-gru"
    aa_output_root = tmp_path / "validate-only-brent-case1-parity-aa"

    gru_code = runtime.main(
        [
            "--config",
            str(BRENT_CASE1_PARITY_GRU_CONFIG),
            "--output-root",
            str(gru_output_root),
            "--validate-only",
        ]
    )
    aa_code = runtime.main(
        [
            "--config",
            str(BRENT_CASE1_PARITY_AA_CONFIG),
            "--output-root",
            str(aa_output_root),
            "--validate-only",
        ]
    )

    assert gru_code == 0
    assert aa_code == 0

    gru_manifest = json.loads((gru_output_root / "manifest" / "run_manifest.json").read_text())
    aa_manifest = json.loads((aa_output_root / "manifest" / "run_manifest.json").read_text())
    gru_resolved = json.loads((gru_output_root / "config" / "config.resolved.json").read_text())
    aa_resolved = json.loads((aa_output_root / "config" / "config.resolved.json").read_text())
    aa_stage_config = json.loads(
        (aa_output_root / "aa_forecast" / "config" / "stage_config.json").read_text()
    )

    assert len(gru_manifest["jobs"]) == 1
    assert gru_manifest["jobs"][0]["model"] == "GRU"
    assert gru_manifest["jobs"][0]["requested_mode"] == "learned_fixed"
    assert gru_manifest["jobs"][0]["validated_mode"] == "learned_fixed"
    assert gru_manifest["jobs"][0]["selected_search_params"] == []
    assert gru_resolved["runtime"]["random_seed"] == 1
    assert gru_resolved["jobs"] == [
        {
            "model": "GRU",
            "params": BRENT_CASE1_PARITY_PARAMS,
            "requested_mode": "learned_fixed",
            "validated_mode": "learned_fixed",
            "selected_search_params": [],
        }
    ]
    assert gru_manifest["training_search"] == {
        "requested_mode": "training_fixed",
        "validated_mode": "training_fixed",
        "selected_search_params": [],
    }

    assert len(aa_manifest["jobs"]) == 1
    assert aa_manifest["jobs"][0]["model"] == "AAForecast"
    assert aa_manifest["jobs"][0]["requested_mode"] == "learned_fixed"
    assert aa_manifest["jobs"][0]["validated_mode"] == "learned_fixed"
    assert aa_manifest["jobs"][0]["selected_search_params"] == []
    assert aa_resolved["runtime"]["random_seed"] == 1
    assert aa_resolved["jobs"] == [
        {
            "model": "AAForecast",
            "params": BRENT_CASE1_PARITY_AA_PARAMS,
            "requested_mode": "learned_fixed",
            "validated_mode": "learned_fixed",
            "selected_search_params": [],
        }
    ]
    assert aa_manifest["training_search"] == {
        "requested_mode": "training_fixed",
        "validated_mode": "training_fixed",
        "selected_search_params": [],
    }
    assert aa_manifest["aa_forecast"]["config_path"] == str(BRENT_CASE1_PARITY_AA_PLUGIN)
    assert aa_stage_config["model"] == "gru"
    assert aa_stage_config["model_params"] == BRENT_CASE1_PARITY_AA_PARAMS
    assert aa_stage_config["uncertainty"]["enabled"] is True
    assert aa_stage_config["uncertainty"]["sample_count"] == 50
    assert aa_stage_config["uncertainty"]["dropout_candidates"] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]
