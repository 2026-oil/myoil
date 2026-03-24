from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_ROOT = REPO_ROOT / "yaml"


def _iter_yaml_payloads():
    for path in sorted(YAML_ROOT.rglob("*.yaml")):
        yield path, yaml.safe_load(path.read_text(encoding="utf-8"))


def test_yaml_matrix_replaces_patchtst_with_timexer() -> None:
    for path, payload in _iter_yaml_payloads():
        models = [job["model"] for job in payload.get("jobs", [])]
        assert "PatchTST" not in models, path


def test_yaml_matrix_pins_model_step_size_when_timexer_is_present() -> None:
    timexer_files = 0

    for path, payload in _iter_yaml_payloads():
        models = [job["model"] for job in payload.get("jobs", [])]
        if "TimeXer" not in models:
            continue
        timexer_files += 1
        assert payload["training"]["model_step_size"] == 8, path

    assert timexer_files > 0
