from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.paperbanana_brent_architecture import (
    BrentArchitectureContext,
    build_caption,
    build_methodology_markdown,
    build_paperbanana_prompt,
    configure_paperbanana,
    write_bundle,
)


def _sample_context(tmp_path: Path) -> BrentArchitectureContext:
    return BrentArchitectureContext(
        task_name="aaforecast_gru-ret",
        target_col="Com_BrentCrudeOil",
        hist_exog_cols=("GPRD_THREAT", "GPRD", "Com_LMEX"),
        star_hist_exog_cols=("GPRD_THREAT", "GPRD"),
        non_star_hist_exog_cols=("Com_LMEX",),
        backbone="gru",
        retrieval_enabled=True,
        retrieval_top_k=1,
        retrieval_recency_gap_steps=16,
        retrieval_trigger_quantile=0.0126,
        retrieval_min_similarity=0.362,
        retrieval_blend_floor=0.5,
        retrieval_blend_max=0.75,
        retrieval_temperature=0.0105,
        input_size=64,
        horizon=2,
        n_windows=14,
        max_steps=400,
        val_size=32,
        loss="mae",
        run_root=tmp_path / "runs/example",
        config_path=tmp_path / "yaml/experiment/example.yaml",
        linked_stage_config_path="yaml/plugins/aa_forecast/example.yaml",
        selected_stage_config_path="/tmp/example.yaml",
        retrieval_detail_config_path="../retrieval/baseline_retrieval.yaml",
    )


def test_text_builders_include_expected_facts(tmp_path: Path) -> None:
    ctx = _sample_context(tmp_path)
    methodology = build_methodology_markdown(ctx)
    caption = build_caption(ctx)
    prompt = build_paperbanana_prompt(ctx)
    assert "Com_BrentCrudeOil" in methodology
    assert "top-1" in methodology
    assert "posthoc blend path" in prompt
    assert "Figure:" in caption


def test_write_bundle_creates_expected_files(tmp_path: Path) -> None:
    ctx = _sample_context(tmp_path)
    payload = write_bundle(tmp_path / "bundle", ctx)
    bundle_json = json.loads((tmp_path / "bundle/bundle.json").read_text(encoding="utf-8"))
    assert Path(payload["methodology_path"]).exists()
    assert Path(payload["caption_path"]).exists()
    assert Path(payload["paperbanana_prompt_path"]).exists()
    assert bundle_json["task_name"] == "aaforecast_gru-ret"


def test_configure_paperbanana_writes_expected_yaml(tmp_path: Path) -> None:
    paperbanana_dir = tmp_path / "PaperBanana"
    config_path = configure_paperbanana(
        paperbanana_dir=paperbanana_dir,
        main_model_name="gemini-main",
        image_gen_model_name="gemini-image",
        google_api_key="secret",
        embed_api_key=False,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8").split("\n", 2)[2])
    assert payload["defaults"]["main_model_name"] == "gemini-main"
    assert payload["defaults"]["image_gen_model_name"] == "gemini-image"
    assert payload["api_keys"]["google_api_key"] == ""
