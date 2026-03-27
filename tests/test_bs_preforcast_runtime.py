from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import bs_preforcast.runtime as bs_runtime
import pytest
import residual.runtime as residual_runtime


def test_residual_runtime_uses_authoritative_bs_preforcast_runtime_apis() -> None:
    assert residual_runtime.prepare_bs_preforcast_fold_inputs is bs_runtime.prepare_bs_preforcast_fold_inputs
    assert residual_runtime.materialize_bs_preforcast_stage is bs_runtime.materialize_bs_preforcast_stage


def test_validate_only_bs_preforcast_smoke_fixture_materializes_stage_metadata(
    tmp_path: Path,
) -> None:
    fixture_path = Path("tests/fixtures/bs_preforcast_runtime_smoke.yaml")
    output_root = tmp_path / "bs-preforcast-smoke"

    code = residual_runtime.main(
        [
            "--config",
            str(fixture_path),
            "--validate-only",
            "--output-root",
            str(output_root),
        ]
    )

    assert code == 0
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    capability = json.loads((output_root / "config" / "capability_report.json").read_text())
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    stage_resolved = output_root / "bs_preforcast" / "config" / "config.resolved.json"
    assert stage_resolved.is_file()
    assert resolved["bs_preforcast"]["selected_config_path"].endswith(
        "tests/fixtures/bs_preforcast_stage_smoke.yaml"
    )
    assert resolved["bs_preforcast"]["validate_only"] is True
    assert capability["bs_preforcast"]["enabled"] is True
    assert manifest["bs_preforcast"]["validate_only"] is True


def test_run_stage_variants_marks_output_root_as_internal(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "main.py").write_text("", encoding="utf-8")
    captured: dict[str, object] = {}
    stage_job_model = next(iter(bs_runtime.MODEL_CLASSES))

    monkeypatch.setattr(bs_runtime, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(
        bs_runtime,
        "_single_stage_job",
        lambda _loaded: SimpleNamespace(model=stage_job_model),
    )
    monkeypatch.setattr(
        bs_runtime,
        "_stage_variant_payloads",
        lambda _loaded: [("demo", {"jobs": [{"model": stage_job_model, "params": {}}]})],
    )

    def fake_run(cmd, *, cwd, check, env):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["check"] = check
        captured["env"] = dict(env)

    monkeypatch.setattr(bs_runtime.subprocess, "run", fake_run)

    run_root = tmp_path / "run-root"
    result = bs_runtime._run_stage_variants(object(), run_root=run_root)

    assert result == [run_root / "bs_preforcast" / "runs" / "demo"]
    assert captured["cmd"] == [
        bs_runtime.sys.executable,
        str(repo_root / "main.py"),
        "--config",
        str(run_root / "bs_preforcast" / "temp_configs" / "demo.yaml"),
        "--output-root",
        str(run_root / "bs_preforcast" / "runs" / "demo"),
    ]
    assert captured["cwd"] == repo_root
    assert captured["check"] is True
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[bs_runtime._ALLOW_INTERNAL_OUTPUT_ROOT_ENV] == "1"


def test_materialize_stage_fails_before_stage_side_effects_for_unsupported_futr_exog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    loaded = SimpleNamespace(
        config=SimpleNamespace(bs_preforcast=SimpleNamespace(enabled=True)),
        normalized_payload={},
    )
    stage_loaded = SimpleNamespace(normalized_payload={"demo": True})

    monkeypatch.setattr(
        bs_runtime,
        "load_bs_preforcast_stage_config",
        lambda *_args, **_kwargs: stage_loaded,
    )

    def fail_fast(*_args, **_kwargs):
        raise ValueError("unsupported futr_exog main job")

    monkeypatch.setattr(
        bs_runtime,
        "resolve_bs_preforcast_injection_mode",
        fail_fast,
    )
    monkeypatch.setattr(
        bs_runtime,
        "_write_json",
        lambda *_args, **_kwargs: calls.append("write_json"),
    )
    monkeypatch.setattr(
        bs_runtime,
        "write_manifest",
        lambda *_args, **_kwargs: calls.append("write_manifest"),
    )
    monkeypatch.setattr(
        bs_runtime,
        "_run_stage_variants",
        lambda *_args, **_kwargs: calls.append("run_stage_variants"),
    )
    monkeypatch.setattr(
        bs_runtime,
        "attach_bs_preforcast_stage_metadata",
        lambda *_args, **_kwargs: calls.append("attach_metadata"),
    )

    with pytest.raises(ValueError, match="unsupported futr_exog main job"):
        bs_runtime.materialize_bs_preforcast_stage(
            loaded=loaded,
            selected_jobs=[SimpleNamespace(model="Naive")],
            run_root=tmp_path / "run-root",
            main_resolved_path=tmp_path / "main.resolved.json",
            main_capability_path=tmp_path / "main.capability.json",
            main_manifest_path=tmp_path / "main.manifest.json",
            entrypoint_version="test",
            validate_only=False,
        )

    assert calls == []
