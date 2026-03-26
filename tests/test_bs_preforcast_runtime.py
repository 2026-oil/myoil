from __future__ import annotations

import json
from pathlib import Path

import bs_preforcast.runtime as bs_runtime
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
