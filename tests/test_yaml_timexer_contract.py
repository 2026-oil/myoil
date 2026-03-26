from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_ROOT = REPO_ROOT / "yaml"


def _resolve_jobs(path: Path, jobs: object) -> list[dict]:
    if isinstance(jobs, list):
        return jobs
    if isinstance(jobs, str):
        repo_candidate = (REPO_ROOT / jobs).resolve()
        local_candidate = (path.parent / jobs).resolve()
        jobs_path = local_candidate if local_candidate.exists() else repo_candidate
        loaded = yaml.safe_load(jobs_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return loaded
        return loaded["jobs"]
    return []


def _iter_yaml_payloads():
    for path in sorted(YAML_ROOT.rglob("*.yaml")):
        if path.name in {"jobs_default.yaml", "jobs_tune.yaml"}:
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        payload["jobs"] = _resolve_jobs(path, payload.get("jobs", []))
        yield path, payload


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
