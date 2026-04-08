from __future__ import annotations

from pathlib import Path

import pytest

from app_config import load_app_config
import runtime_support.runner as runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RESIDUAL_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "legacy_residual_rejected.yaml"
)
ERROR_TEXT = (
    "legacy residual config is no longer supported; remove the top-level residual section"
)


def test_load_app_config_rejects_legacy_residual_section() -> None:
    with pytest.raises(ValueError, match=ERROR_TEXT):
        load_app_config(REPO_ROOT, config_path=LEGACY_RESIDUAL_FIXTURE)


def test_runtime_validate_only_rejects_legacy_residual_section() -> None:
    with pytest.raises(ValueError, match=ERROR_TEXT):
        runtime.main(["--validate-only", "--config", str(LEGACY_RESIDUAL_FIXTURE)])
