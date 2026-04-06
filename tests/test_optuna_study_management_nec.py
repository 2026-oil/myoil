from __future__ import annotations

import json
from pathlib import Path

import runtime_support.runner as runtime
import yaml


NEC_CONFIG = Path("tests/fixtures/nec_runtime_smoke.yaml")


def test_validate_only_nec_multi_study_catalog_and_projection(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(NEC_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(Path(payload["dataset"]["path"]).resolve())
    payload.setdefault("runtime", {})["opt_study_count"] = 2
    config_path = tmp_path / "nec_multi.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    output_root = tmp_path / "validate-only-nec-multi"
    code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--validate-only",
        ]
    )

    assert code == 0
    manifest = json.loads((output_root / "manifest" / "run_manifest.json").read_text())
    resolved = json.loads((output_root / "config" / "config.resolved.json").read_text())
    study_catalog = json.loads(
        (output_root / "models" / "NEC" / "study_catalog.json").read_text()
    )
    assert resolved["nec"]["config_path"] == "tests/fixtures/nec_plugin_smoke.yaml"
    assert manifest["optuna"]["study_count"] == 2
    assert manifest["optuna"]["canonical_projection_study_index"] == 1
    assert study_catalog["study_count"] == 2
    assert study_catalog["canonical_projection_study_index"] == 1
    assert (output_root / "models" / "NEC" / "visualizations" / "cross_study_dashboard.html").exists()

    selected_output_root = tmp_path / "validate-only-nec-multi-selected"
    selected_code = runtime.main(
        [
            "--config",
            str(config_path),
            "--output-root",
            str(selected_output_root),
            "--validate-only",
            "--optuna-study",
            "2",
        ]
    )

    assert selected_code == 0
    selected_manifest = json.loads(
        (selected_output_root / "manifest" / "run_manifest.json").read_text()
    )
    selected_catalog = json.loads(
        (selected_output_root / "models" / "NEC" / "study_catalog.json").read_text()
    )
    assert selected_manifest["optuna"]["selected_study_index"] == 2
    assert selected_manifest["optuna"]["canonical_projection_study_index"] == 2
    assert selected_catalog["selected_study_index"] == 2
    assert selected_catalog["canonical_projection_study_index"] == 2
    assert (selected_output_root / "models" / "NEC" / "visualizations" / "cross_study_dashboard.html").exists()
