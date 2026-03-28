from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BOMB_DIR = REPO_ROOT / "yaml" / "bomb"


EXPECTED_FILES = [
    "brentoil-case3-family-h8-diff-exloss-i48.yaml",
    "brentoil-case3-family-h8-diff-exloss-i48-res-level.yaml",
    "brentoil-case3-family-h8-diff-exloss-i48-res-delta.yaml",
    "brentoil-case3-family-h8-diff-exloss-i128.yaml",
    "brentoil-case3-family-h8-diff-exloss-i128-res-level.yaml",
    "brentoil-case3-family-h8-diff-exloss-i128-res-delta.yaml",
    "wti-case3-family-h8-diff-exloss-i48.yaml",
    "wti-case3-family-h8-diff-exloss-i48-res-level.yaml",
    "wti-case3-family-h8-diff-exloss-i48-res-delta.yaml",
    "wti-case3-family-h8-diff-exloss-i128.yaml",
    "wti-case3-family-h8-diff-exloss-i128-res-level.yaml",
    "wti-case3-family-h8-diff-exloss-i128-res-delta.yaml",
]


def test_bomb_yaml_inventory_matches_rebuilt_diff_exloss_matrix() -> None:
    actual = sorted(path.name for path in BOMB_DIR.glob("*.yaml"))
    assert actual == sorted(EXPECTED_FILES)


def test_bomb_yaml_configs_follow_requested_constraints() -> None:
    for filename in EXPECTED_FILES:
        payload = yaml.safe_load((BOMB_DIR / filename).read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        training = payload["training"]
        runtime = payload["runtime"]
        cv = payload["cv"]
        residual = payload["residual"]
        jobs = payload["jobs"]

        assert runtime["transformations_target"] == "diff"
        assert runtime["transformations_exog"] == "diff"
        assert training["loss"] == "exloss"
        assert training["input_size"] in {48, 128}
        assert training["lr_scheduler"]["max_lr"] == 0.003
        assert training["max_steps"] == 800
        assert cv["horizon"] == 8
        assert all(job["model"] not in {"Naive", "xLSTM"} for job in jobs)
        assert any(job["model"] == "LSTM" for job in jobs)

        if filename.endswith(("-res-level.yaml", "-res-delta.yaml")):
            assert residual["enabled"] is True
            assert residual["model"] == "xgboost"
            assert residual["target"] in {"level", "delta"}
            assert residual["features"]["exog_sources"]["hist"] == dataset["hist_exog_cols"]
            assert "learning_rate" not in residual["params"]
            assert residual["params"]["subsample"] == 0.5
            assert float(residual["params"]["colsample_bytree"]) == 0.9
        else:
            assert residual == {"enabled": False}
