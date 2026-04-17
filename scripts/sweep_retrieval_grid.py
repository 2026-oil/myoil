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
    "star_anomaly_tails_upward": [
        ["GPRD_THREAT"],
        ["GPRD"],
        ["GPRD_ACT"],
        ["Idx_OVX"],
        ["GPRD_THREAT", "GPRD"],
        ["GPRD_THREAT", "GPRD_ACT"],
        ["GPRD", "GPRD_ACT"],
        ["GPRD", "Idx_OVX"],
        ["GPRD_THREAT", "GPRD", "GPRD_ACT"],
        ["GPRD_THREAT", "GPRD", "GPRD_ACT", "Idx_OVX"],
    ],
    # NOTE: aa_forecast.retrieval.trigger_quantile requires 0 < value < 1.
    "trigger_quantile": [0.05, 0.1, 0.2],
    "min_similarity": [0.0, 0.3, 0.5],
    "top_k": [1, 3, 5, 7],
}

# ---------------------------------------------------------------------------
# Fixed parts of the plugin YAML (everything except retrieval params)
# ---------------------------------------------------------------------------

PLUGIN_FIXED: dict[str, Any] = {
    "aa_forecast": {
        "model": "timexer",
        "tune_training": False,
        "lowess_frac": 0.35,
        "lowess_delta": 0.01,
        "uncertainty": {
            "enabled": True,
            # Keep uncertainty enabled because aa_forecast retrieval requires it, but
            # keep the sweep lightweight (grid size already dominates runtime).
            "dropout_candidates": [0.2, 0.3],
            "sample_count": 3,
        },
        "model_params": {
            "hidden_size": 128,
            "n_heads": 8,
            "e_layers": 2,
            "dropout": 0.1,
            "d_ff": 256,
            "patch_len": 8,
            "use_norm": True,
            "decoder_hidden_size": 128,
            "decoder_layers": 4,
            "season_length": 4,
        },
        "star_anomaly_tails": {
            "upward": ["GPRD_THREAT"],  # will be overridden by grid
            "two_sided": [],
        },
        "thresh": 3.5,
        "retrieval": {
            "recency_gap_steps": 8,
            "similarity": "cosine",
            "temperature": 0.1,
            "blend_floor": 0.0,
            "blend_max": 1.0,
            "use_uncertainty_gate": True,
            "use_event_key": True,
            "event_score_log_bonus_alpha": 0.0,
            "event_score_log_bonus_cap": 0.0,
        },
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
            "GPRD",
            "GPRD_ACT",
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


def generate_combos() -> list[dict[str, Any]]:
    keys = list(GRID.keys())
    all_values = [GRID[k] for k in keys]
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*all_values):
        combo = dict(zip(keys, values))
        combos.append(combo)
    return combos


def _build_retrieval_block(combo: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": True,
        "top_k": combo["top_k"],
        "trigger_quantile": combo["trigger_quantile"],
        "min_similarity": combo["min_similarity"],
        "use_event_key": True,
        "recency_gap_steps": 8,
        "similarity": "cosine",
        "temperature": 0.1,
        "blend_floor": 0.0,
        "blend_max": 1.0,
        "use_uncertainty_gate": True,
        "event_score_log_bonus_alpha": 0.0,
        "event_score_log_bonus_cap": 0.0,
    }


def _build_plugin_yaml(combo: dict[str, Any]) -> dict[str, Any]:
    import copy

    doc = copy.deepcopy(PLUGIN_FIXED)
    # Override star_anomaly_tails with grid values
    doc["aa_forecast"]["star_anomaly_tails"]["upward"] = combo[
        "star_anomaly_tails_upward"
    ]
    # Set retrieval params from grid
    doc["aa_forecast"]["retrieval"] = _build_retrieval_block(combo)
    return doc


def _build_experiment_yaml(idx: int, plugin_rel_path: str) -> dict[str, Any]:
    import copy

    doc = copy.deepcopy(EXPERIMENT_TEMPLATE)
    doc["task"]["name"] = f"sweep_ret_{idx:04d}"
    doc["aa_forecast"]["config_path"] = plugin_rel_path
    return doc


def _yaml_dump(data: dict[str, Any]) -> str:
    return yaml.dump(
        data, default_flow_style=False, sort_keys=False, allow_unicode=True
    )


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
        plugin_filename = f"aa_forecast_timexer-ret-{idx:04d}.yaml"
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
    print(
        f"Generated {total} experiment YAMLs -> {EXPERIMENT_DIR.relative_to(REPO_ROOT)}/"
    )
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
    parser = argparse.ArgumentParser(
        description="Generate retrieval grid-sweep configs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combo count without generating files",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove generated sweep directories"
    )
    args = parser.parse_args()

    if args.clean:
        clean()
        return

    generate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
