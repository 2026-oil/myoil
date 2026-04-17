from __future__ import annotations

import argparse
import asyncio
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "yaml/experiment/feature_set_aaforecast_brent/aaforecast-gru-ret.yaml"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs/brent/feature_set_aaforecast_brent_aaforecast_gru-ret"
DEFAULT_BUNDLE_DIR = REPO_ROOT / "docs/paperbanana/brent_aaforecast_gru_ret"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/paperbanana_outputs/brent_aaforecast_gru_ret"
DEFAULT_PAPERBANANA_DIR = REPO_ROOT / "tools/PaperBanana"


@dataclass(frozen=True)
class BrentArchitectureContext:
    task_name: str
    target_col: str
    hist_exog_cols: tuple[str, ...]
    star_hist_exog_cols: tuple[str, ...]
    non_star_hist_exog_cols: tuple[str, ...]
    backbone: str
    retrieval_enabled: bool
    retrieval_top_k: int
    retrieval_recency_gap_steps: int
    retrieval_trigger_quantile: float | None
    retrieval_min_similarity: float
    retrieval_blend_floor: float
    retrieval_blend_max: float
    retrieval_temperature: float
    input_size: int
    horizon: int
    n_windows: int
    max_steps: int
    val_size: int
    loss: str
    run_root: Path
    config_path: Path
    linked_stage_config_path: str | None
    selected_stage_config_path: str | None
    retrieval_detail_config_path: str | None


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_repo_relative(base_path: Path, reference: str) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate.resolve()
    return (base_path.parent / candidate).resolve()


def _discover_run_root(run_root: Path) -> Path:
    if run_root.exists():
        return run_root
    candidates = sorted(
        REPO_ROOT.glob("runs/**/feature_set_aaforecast_brent_aaforecast_gru-ret")
    )
    if candidates:
        return candidates[0]
    return run_root


def load_context(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> BrentArchitectureContext:
    run_root = _discover_run_root(run_root)
    resolved_path = run_root / "config/config.resolved.json"
    manifest_path = run_root / "manifest/run_manifest.json"
    if resolved_path.exists() and manifest_path.exists():
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        top = _load_yaml(config_path)
        stage_path = _resolve_repo_relative(config_path, top["aa_forecast"]["config_path"])
        stage_doc = _load_yaml(stage_path)
        retrieval_ref = stage_doc["aa_forecast"].get("retrieval", {}).get("config_path")
        retrieval_doc = {}
        if retrieval_ref:
            retrieval_doc = _load_yaml(_resolve_repo_relative(stage_path, retrieval_ref))
        resolved = {
            "task": top["task"],
            "dataset": top["dataset"],
            "training": {
                "input_size": 64,
                "max_steps": 400,
                "val_size": 32,
                "loss": "mae",
            },
            "cv": {"horizon": 2, "n_windows": 14},
            "jobs": [{"params": stage_doc["aa_forecast"].get("model_params", {})}],
            "aa_forecast": {
                "model": stage_doc["aa_forecast"]["model"],
                "retrieval": {
                    **retrieval_doc.get("retrieval", {}),
                    **stage_doc["aa_forecast"].get("retrieval", {}),
                },
                "star_hist_exog_cols_resolved": tuple(
                    stage_doc["aa_forecast"].get("star_anomaly_tails", {}).get("upward", ())
                ),
                "non_star_hist_exog_cols_resolved": (),
            },
        }
        manifest = {
            "aa_forecast": {
                "selected_config_path": str(stage_path),
                "retrieval": resolved["aa_forecast"]["retrieval"],
            }
        }
    aa = resolved["aa_forecast"]
    retrieval = aa["retrieval"]
    return BrentArchitectureContext(
        task_name=str(resolved["task"]["name"]),
        target_col=str(resolved["dataset"]["target_col"]),
        hist_exog_cols=tuple(resolved["dataset"].get("hist_exog_cols", ())),
        star_hist_exog_cols=tuple(aa.get("star_hist_exog_cols_resolved", ())),
        non_star_hist_exog_cols=tuple(aa.get("non_star_hist_exog_cols_resolved", ())),
        backbone=str(aa["model"]),
        retrieval_enabled=bool(retrieval.get("enabled", False)),
        retrieval_top_k=int(retrieval.get("top_k", 0)),
        retrieval_recency_gap_steps=int(retrieval.get("recency_gap_steps", 0)),
        retrieval_trigger_quantile=(
            None
            if retrieval.get("trigger_quantile") is None
            else float(retrieval["trigger_quantile"])
        ),
        retrieval_min_similarity=float(retrieval.get("min_similarity", 0.0)),
        retrieval_blend_floor=float(retrieval.get("blend_floor", 0.0)),
        retrieval_blend_max=float(retrieval.get("blend_max", 0.0)),
        retrieval_temperature=float(retrieval.get("temperature", 0.0)),
        input_size=int(resolved["training"]["input_size"]),
        horizon=int(resolved["cv"]["horizon"]),
        n_windows=int(resolved["cv"]["n_windows"]),
        max_steps=int(resolved["training"]["max_steps"]),
        val_size=int(resolved["training"]["val_size"]),
        loss=str(resolved["training"]["loss"]),
        run_root=run_root,
        config_path=config_path,
        linked_stage_config_path=resolved.get("aa_forecast", {}).get("config_path"),
        selected_stage_config_path=manifest.get("aa_forecast", {}).get("selected_config_path"),
        retrieval_detail_config_path=retrieval.get("config_path"),
    )


def build_methodology_markdown(ctx: BrentArchitectureContext) -> str:
    hist = ", ".join(f"`{col}`" for col in ctx.hist_exog_cols)
    star = ", ".join(f"`{col}`" for col in ctx.star_hist_exog_cols) or "none"
    non_star = ", ".join(f"`{col}`" for col in ctx.non_star_hist_exog_cols) or "none"
    retrieval_block = (
        f"The AAForecast stage enables retrieval in posthoc blending mode. "
        f"For each fold, the retrieval branch builds a STAR-based memory bank over the transformed training history, "
        f"forms a query from the latest input window, filters candidate neighbors by event threshold and minimum similarity, "
        f"selects the top-{ctx.retrieval_top_k} neighbor set, and converts retrieved future returns into a memory prediction. "
        f"The final forecast is blended with the base AAForecast prediction using similarity-controlled weights in the "
        f"[{ctx.retrieval_blend_floor:.2f}, {ctx.retrieval_blend_max:.2f}] interval. "
        f"The retrieval hyperparameters for this Brent route are recency gap {ctx.retrieval_recency_gap_steps}, "
        f"trigger quantile {ctx.retrieval_trigger_quantile}, min similarity {ctx.retrieval_min_similarity:.3f}, "
        f"and temperature {ctx.retrieval_temperature:.4f}."
        if ctx.retrieval_enabled
        else "Retrieval is disabled in this route."
    )
    return f"""# Methodology: Brent AAForecast-GRU-Retrieval execution flow

This figure should illustrate the concrete runtime path of the Brent experiment launched with `main.py --config {ctx.config_path.as_posix()}`. The entrypoint bootstraps the project virtual environment, parses CLI arguments, and forwards execution into `runtime_support.runner.load_app_config`. The top-level experiment YAML selects the task `{ctx.task_name}` and the target series `{ctx.target_col}`.

The configuration loader resolves three layers of configuration. First it reads the experiment YAML under `yaml/experiment/feature_set_aaforecast_brent/`. Second it follows the linked AAForecast stage config under `{ctx.linked_stage_config_path}` and selects the GRU backbone. Third it merges retrieval detail settings from `{ctx.retrieval_detail_config_path}`. The resolved configuration and run manifest are materialized under `{ctx.run_root.as_posix()}` before fold execution starts.

The dataset branch loads `data/df.csv`, uses `dt` as the timestamp column, and models `{ctx.target_col}` as the forecasting target. Historical exogenous features are {hist}. Within AAForecast, STAR anomaly features are computed only for {star}. Remaining exogenous inputs {non_star} travel through the non-STAR path.

For each expanding-window cross-validation fold, the runtime applies differencing transforms to the target and exogenous history, builds adapter inputs, instantiates an `AAForecast` model with the `{ctx.backbone}` backbone, and trains it with input size {ctx.input_size}, horizon {ctx.horizon}, {ctx.n_windows} windows, max steps {ctx.max_steps}, validation size {ctx.val_size}, and `{ctx.loss}` loss. After fitting, the model emits the base forecast for the future horizon.

{retrieval_block}

The output side should show that every fold produces predictions, metrics, and plots under the run root. The AAForecast stage also owns its own sub-artifacts such as stage config snapshots, retrieval diagnostics, context traces, uncertainty outputs, and encoding exports when enabled. The full architecture diagram should therefore emphasize five visual groups: entry and config, runtime orchestration, AAForecast model path, retrieval path, and output artifacts.
"""


def build_caption(ctx: BrentArchitectureContext) -> str:
    return (
        "Figure: Brent AAForecast-GRU-Retrieval execution flow from `main.py` configuration "
        "loading through stage-plugin resolution, fold-wise training, STAR-based retrieval "
        "blending, and run artifact materialization."
    )


def build_paperbanana_prompt(ctx: BrentArchitectureContext) -> str:
    return f"""# PaperBanana prompt for Brent architecture diagram

Create a polished academic system diagram for a forecasting experiment. The diagram must be a left-to-right pipeline with five grouped regions: **Entry & Config**, **Runtime Orchestration**, **AAForecast Model Path**, **Retrieval Path**, and **Output Artifacts**.

## Hard constraints
- Base the diagram only on the execution path of `main.py --config {ctx.config_path.as_posix()}`.
- Do not generalize to the whole Brent batch matrix; focus on the single `aaforecast-gru-ret` route.
- Show that the active stage plugin is **AAForecast**, the backbone is **GRU**, and retrieval is a **posthoc blend path**, not the main backbone itself.
- Include the concrete target `{ctx.target_col}`.
- Show historical exogenous features entering the runtime, with STAR features visually separated from non-STAR features.
- Include run-manifest/config materialization under `{ctx.run_root.as_posix()}`.

## Key technical labels to preserve
- `main.py`
- `runtime_support.runner.load_app_config`
- experiment YAML
- AAForecast linked YAML
- retrieval detail YAML
- `AAForecastStagePlugin`
- expanding-window CV
- diff transform
- GRU backbone
- base forecast
- STAR memory bank
- query window
- neighbor search
- posthoc blend
- predictions / metrics / summary artifacts

## Required numeric callouts
- input size: {ctx.input_size}
- horizon: {ctx.horizon}
- CV windows: {ctx.n_windows}
- retrieval top-k: {ctx.retrieval_top_k}
- recency gap: {ctx.retrieval_recency_gap_steps}
- min similarity: {ctx.retrieval_min_similarity:.3f}
- blend range: {ctx.retrieval_blend_floor:.2f} to {ctx.retrieval_blend_max:.2f}

## Visual style
- publication-quality, clean, vector-like, no 3D effects
- blue for orchestration, purple for model internals, orange for retrieval, green for artifacts, gray for config inputs
- rounded subsystem boxes, directional arrows, concise labels
- no figure caption text inside the image body
- keep the content faithful to software architecture rather than abstract ML storytelling
"""


def write_bundle(output_dir: Path, ctx: BrentArchitectureContext) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methodology_path = output_dir / "methodology.md"
    caption_path = output_dir / "caption.txt"
    prompt_path = output_dir / "paperbanana_prompt.md"
    metadata_path = output_dir / "bundle.json"
    methodology_path.write_text(build_methodology_markdown(ctx), encoding="utf-8")
    caption_path.write_text(build_caption(ctx) + "\n", encoding="utf-8")
    prompt_path.write_text(build_paperbanana_prompt(ctx), encoding="utf-8")
    metadata = {
        "task_name": ctx.task_name,
        "config_path": str(ctx.config_path),
        "run_root": str(ctx.run_root),
        "methodology_path": str(methodology_path),
        "caption_path": str(caption_path),
        "paperbanana_prompt_path": str(prompt_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {k: str(v) for k, v in metadata.items()}


def configure_paperbanana(
    *,
    paperbanana_dir: Path,
    main_model_name: str,
    image_gen_model_name: str,
    google_api_key: str | None,
    embed_api_key: bool,
) -> Path:
    config_path = paperbanana_dir / "configs/model_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "defaults": {
            "main_model_name": main_model_name,
            "image_gen_model_name": image_gen_model_name,
        },
        "api_keys": {
            "google_api_key": google_api_key if (embed_api_key and google_api_key) else "",
            "openai_api_key": "",
            "anthropic_api_key": "",
            "openrouter_api_key": "",
        },
    }
    header = (
        "# Generated by scripts/paperbanana_brent_architecture.py\n"
        "# If google_api_key is empty, PaperBanana will fall back to the GOOGLE_API_KEY environment variable.\n"
    )
    config_path.write_text(header + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _extract_final_image_bytes(result: dict[str, Any], exp_mode: str) -> bytes | None:
    task_name = "diagram"
    final_key = None
    for round_idx in range(3, -1, -1):
        candidate = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if result.get(candidate):
            final_key = candidate
            break
    if final_key is None:
        final_key = (
            f"target_{task_name}_stylist_desc0_base64_jpg"
            if exp_mode == "demo_full"
            else f"target_{task_name}_desc0_base64_jpg"
        )
    payload = result.get(final_key)
    if not payload:
        return None
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


def _load_paperbanana_demo_module(paperbanana_dir: Path):
    site_packages = sorted((paperbanana_dir / ".venv/lib").glob("python*/site-packages"))
    for site_pkg in reversed(site_packages):
        sys.path.insert(0, str(site_pkg))
    sys.path.insert(0, str(paperbanana_dir))
    module_path = paperbanana_dir / "demo.py"
    spec = importlib.util.spec_from_file_location("paperbanana_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load PaperBanana demo module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paperbanana_has_credentials(paperbanana_dir: Path) -> bool:
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        return True
    config_path = paperbanana_dir / "configs/model_config.yaml"
    if not config_path.exists():
        return False
    payload = _load_yaml(config_path)
    api_keys = payload.get("api_keys", {})
    return any(
        bool(api_keys.get(key))
        for key in ("google_api_key", "openrouter_api_key", "openai_api_key", "anthropic_api_key")
    )


async def _run_paperbanana_async(
    *,
    bundle_dir: Path,
    output_dir: Path,
    paperbanana_dir: Path,
    exp_mode: str,
    retrieval_setting: str,
    num_candidates: int,
    max_critic_rounds: int,
    main_model_name: str,
    image_gen_model_name: str,
) -> dict[str, Any]:
    if not _paperbanana_has_credentials(paperbanana_dir):
        raise SystemExit(
            "PaperBanana is installed, but no API credentials are configured. "
            "Set GOOGLE_API_KEY (recommended) or write a key into tools/PaperBanana/configs/model_config.yaml first."
        )
    metadata = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    method_content = Path(metadata["methodology_path"]).read_text(encoding="utf-8")
    caption = Path(metadata["caption_path"]).read_text(encoding="utf-8").strip()
    demo = _load_paperbanana_demo_module(paperbanana_dir)
    inputs = demo.create_sample_inputs(
        method_content=method_content,
        caption=caption,
        num_copies=num_candidates,
        max_critic_rounds=max_critic_rounds,
    )
    results = await demo.process_parallel_candidates(
        inputs,
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        main_model_name=main_model_name,
        image_gen_model_name=image_gen_model_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    candidate_manifest = []
    for idx, result in enumerate(results):
        image_bytes = _extract_final_image_bytes(result, exp_mode)
        image_path = None
        if image_bytes is not None:
            image_path = output_dir / f"candidate_{idx:02d}.png"
            image_path.write_bytes(image_bytes)
        candidate_manifest.append(
            {
                "candidate_index": idx,
                "image_path": None if image_path is None else str(image_path),
                "keys": sorted(result.keys()),
            }
        )
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "results_path": str(output_dir / "results.json"),
        "candidate_manifest_path": str(output_dir / "candidate_manifest.json"),
        "output_dir": str(output_dir),
    }


def run_paperbanana(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(_run_paperbanana_async(**kwargs))


def refine_with_gemini(
    *,
    prompt_path: Path,
    output_path: Path,
    model: str | None,
) -> Path:
    gemini_bin = shutil.which("gemini")
    if gemini_bin is None:
        raise SystemExit("gemini CLI was not found on PATH.")
    prompt_text = prompt_path.read_text(encoding="utf-8")
    refine_prompt = (
        "Refine the following PaperBanana architecture-diagram prompt for clarity and compactness. "
        "Preserve all technical facts and numeric values. Output only the refined prompt.\n\n"
        + prompt_text
    )
    cmd = [gemini_bin, "-p", refine_prompt, "--output-format", "text"]
    if model:
        cmd.extend(["--model", model])
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    try:
        completed = subprocess.run(
            cmd,
            input="n\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = ((exc.stdout or "") + (exc.stderr or "")).strip()
        raise SystemExit(
            "gemini refinement timed out. If this is the first run, authenticate with "
            "`gemini` interactively and rerun refine-with-gemini.\n"
            + partial
        ) from exc
    merged = (completed.stdout or "") + (completed.stderr or "")
    if "Opening authentication page in your browser" in merged:
        raise SystemExit(
            "gemini CLI is installed but not authenticated in this environment yet. "
            "Run `gemini` once interactively to complete login, then rerun refine-with-gemini."
        )
    if completed.returncode != 0:
        raise SystemExit(
            f"gemini refinement failed with exit code {completed.returncode}:\n{merged.strip()}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(completed.stdout, encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and run the Brent AAForecast PaperBanana architecture workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("bundle", help="Write the architecture brief bundle.")
    bundle.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    bundle.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    bundle.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE_DIR)

    configure = subparsers.add_parser(
        "configure-paperbanana", help="Write PaperBanana model_config.yaml for Gemini."
    )
    configure.add_argument("--paperbanana-dir", type=Path, default=DEFAULT_PAPERBANANA_DIR)
    configure.add_argument("--main-model-name", default="gemini-3.1-pro-preview")
    configure.add_argument("--image-gen-model-name", default="gemini-3.1-flash-image-preview")
    configure.add_argument(
        "--google-api-key",
        default=os.environ.get("GOOGLE_API_KEY"),
        help="If omitted, GOOGLE_API_KEY from the environment is used if available.",
    )
    configure.add_argument(
        "--embed-api-key",
        action="store_true",
        help="Write the current Google API key into model_config.yaml. Default leaves it blank.",
    )

    refine = subparsers.add_parser(
        "refine-with-gemini", help="Refine the PaperBanana prompt using Gemini CLI."
    )
    refine.add_argument("--prompt-path", type=Path, default=DEFAULT_BUNDLE_DIR / "paperbanana_prompt.md")
    refine.add_argument("--output-path", type=Path, default=DEFAULT_BUNDLE_DIR / "paperbanana_prompt.refined.md")
    refine.add_argument("--model", default=None)

    run = subparsers.add_parser("run-paperbanana", help="Generate candidates with PaperBanana.")
    run.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--paperbanana-dir", type=Path, default=DEFAULT_PAPERBANANA_DIR)
    run.add_argument("--exp-mode", default="demo_planner_critic", choices=("demo_planner_critic", "demo_full"))
    run.add_argument("--retrieval-setting", default="none", choices=("auto", "manual", "random", "none"))
    run.add_argument("--num-candidates", type=int, default=4)
    run.add_argument("--max-critic-rounds", type=int, default=2)
    run.add_argument("--main-model-name", default="gemini-3.1-pro-preview")
    run.add_argument("--image-gen-model-name", default="gemini-3.1-flash-image-preview")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "bundle":
        ctx = load_context(config_path=args.config_path, run_root=args.run_root)
        payload = write_bundle(args.output_dir, ctx)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "configure-paperbanana":
        config_path = configure_paperbanana(
            paperbanana_dir=args.paperbanana_dir,
            main_model_name=args.main_model_name,
            image_gen_model_name=args.image_gen_model_name,
            google_api_key=args.google_api_key,
            embed_api_key=bool(args.embed_api_key),
        )
        print(json.dumps({"config_path": str(config_path)}, indent=2))
        return 0
    if args.command == "refine-with-gemini":
        output_path = refine_with_gemini(
            prompt_path=args.prompt_path,
            output_path=args.output_path,
            model=args.model,
        )
        print(json.dumps({"output_path": str(output_path)}, indent=2))
        return 0
    if args.command == "run-paperbanana":
        payload = run_paperbanana(
            bundle_dir=args.bundle_dir,
            output_dir=args.output_dir,
            paperbanana_dir=args.paperbanana_dir,
            exp_mode=args.exp_mode,
            retrieval_setting=args.retrieval_setting,
            num_candidates=args.num_candidates,
            max_critic_rounds=args.max_critic_rounds,
            main_model_name=args.main_model_name,
            image_gen_model_name=args.image_gen_model_name,
        )
        print(json.dumps(payload, indent=2))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
