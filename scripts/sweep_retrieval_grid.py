"""Generate exhaustive grid-sweep YAML configs for aa_forecast retrieval parameters.

Usage:
    uv run python scripts/sweep_retrieval_grid.py              # generate all configs
    uv run python scripts/sweep_retrieval_grid.py --dry-run    # print combo count only
    uv run python scripts/sweep_retrieval_grid.py --clean      # remove generated dirs
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Grid definition — edit values here to control the sweep
# ---------------------------------------------------------------------------

GRID: dict[str, list[Any]] = {
    "use_uncertainty_gate": [True, False],
    "use_shape_key": [True, False],
    "use_event_key": [True, False],
    "top_k": [1, 3, 5],
    "recency_gap_steps": [4, 8, 16],
    "trigger_quantile": [0.7, 0.8, 0.9],
    "min_similarity": [0.5, 0.7, 0.9],
    "blend_floor": [0.0],
    "blend_max": [0.5, 0.75, 1.0],
    "event_score_log_bonus_alpha": [0.0, 0.15, 0.3],
    "event_score_log_bonus_cap": [0.0, 0.1],
}

# ---------------------------------------------------------------------------
# Fixed parts of the plugin YAML (everything except retrieval params)
# ---------------------------------------------------------------------------

PLUGIN_FIXED: dict[str, Any] = {
    "aa_forecast": {
        "model": "informer",
        "tune_training": False,
        "lowess_frac": 0.35,
        "lowess_delta": 0.01,
        "uncertainty": {
            "enabled": True,
            "dropout_candidates": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            "sample_count": 50,
        },
        "model_params": {
            "hidden_size": 128,
            "n_head": 8,
            "encoder_layers": 2,
            "dropout": 0.1,
            "factor": 3,
            "decoder_hidden_size": 192,
            "decoder_layers": 4,
            "season_length": 4,
        },
        "star_anomaly_tails": {
            "upward": ["GPRD_THREAT", "BS_Core_Index_A", "BS_Core_Index_C"],
            "two_sided": [],
        },
        "thresh": 3.5,
    }
}

EXPERIMENT_TEMPLATE: dict[str, Any] = {
    "task": {"name": ""},
    "dataset": {
        "path": "data/df.csv",
        "target_col": "Com_BrentCrudeOil",
        "dt_col": "dt",
        "hist_exog_cols": [
            "GPRD_THREAT",
            "BS_Core_Index_A",
            "GPRD",
            "GPRD_ACT",
            "BS_Core_Index_B",
            "BS_Core_Index_C",
            "Idx_OVX",
            "Com_LMEX",
            "Com_BloombergCommodity_BCOM",
            "Idx_DxyUSD",
        ],
        "futr_exog_cols": [],
        "static_exog_cols": [],
    },
    "training_search": {"enabled": False},
    "aa_forecast": {"enabled": True, "config_path": ""},
}

PLUGIN_DIR = REPO_ROOT / "yaml" / "plugins" / "aa_forecast" / "sweep"
EXPERIMENT_DIR = REPO_ROOT / "yaml" / "experiment" / "sweep_retrieval"
CONFIGS_TXT = EXPERIMENT_DIR / "configs.txt"


def _is_valid_combo(combo: dict[str, Any]) -> bool:
    if not combo["use_shape_key"] and not combo["use_event_key"]:
        return False
    if combo["blend_floor"] > combo["blend_max"]:
        return False
    return True


def generate_combos() -> list[dict[str, Any]]:
    keys = list(GRID.keys())
    all_values = [GRID[k] for k in keys]
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*all_values):
        combo = dict(zip(keys, values))
        if _is_valid_combo(combo):
            combos.append(combo)
    return combos


def _build_retrieval_block(combo: dict[str, Any]) -> dict[str, Any]:
    return {"enabled": True, **combo}


def _build_plugin_yaml(combo: dict[str, Any]) -> dict[str, Any]:
    import copy

    doc = copy.deepcopy(PLUGIN_FIXED)
    doc["aa_forecast"]["retrieval"] = _build_retrieval_block(combo)
    return doc


def _build_experiment_yaml(idx: int, plugin_rel_path: str) -> dict[str, Any]:
    import copy

    doc = copy.deepcopy(EXPERIMENT_TEMPLATE)
    doc["task"]["name"] = f"sweep_ret_{idx:04d}"
    doc["aa_forecast"]["config_path"] = plugin_rel_path
    return doc


def _yaml_dump(data: dict[str, Any]) -> str:
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def generate(*, dry_run: bool = False) -> int:
    combos = generate_combos()
    total = len(combos)
    print(f"Total valid combinations: {total}")

    if dry_run:
        return total

    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    config_paths: list[str] = []
    for idx, combo in enumerate(combos, start=1):
        plugin_filename = f"aa_forecast_informer-ret-{idx:04d}.yaml"
        experiment_filename = f"sweep-{idx:04d}.yaml"

        plugin_path = PLUGIN_DIR / plugin_filename
        experiment_path = EXPERIMENT_DIR / experiment_filename

        plugin_doc = _build_plugin_yaml(combo)
        plugin_path.write_text(_yaml_dump(plugin_doc), encoding="utf-8")

        plugin_rel = str(plugin_path.relative_to(REPO_ROOT))
        experiment_doc = _build_experiment_yaml(idx, plugin_rel)
        experiment_path.write_text(_yaml_dump(experiment_doc), encoding="utf-8")

        config_paths.append(str(experiment_path.relative_to(REPO_ROOT)))

    CONFIGS_TXT.write_text("\n".join(config_paths) + "\n", encoding="utf-8")
    print(f"Generated {total} plugin YAMLs  -> {PLUGIN_DIR.relative_to(REPO_ROOT)}/")
    print(f"Generated {total} experiment YAMLs -> {EXPERIMENT_DIR.relative_to(REPO_ROOT)}/")
    print(f"Config list -> {CONFIGS_TXT.relative_to(REPO_ROOT)}")
    return total


def clean() -> None:
    for d in (PLUGIN_DIR, EXPERIMENT_DIR):
        if d.exists():
            shutil.rmtree(d)
            print(f"Removed {d.relative_to(REPO_ROOT)}")
        else:
            print(f"Already clean: {d.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrieval grid-sweep configs")
    parser.add_argument("--dry-run", action="store_true", help="Print combo count without generating files")
    parser.add_argument("--clean", action="store_true", help="Remove generated sweep directories")
    args = parser.parse_args()

    if args.clean:
        clean()
        return

    generate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
