from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_timexer_script_runs_all_configs_with_timexer_only() -> None:
    script_path = REPO_ROOT / "timexer.sh"

    assert script_path.exists()
    assert script_path.stat().st_mode & 0o111

    script = script_path.read_text(encoding="utf-8")
    assert 'exec bash "$repo_root/all.sh" --jobs TimeXer "$@"' in script
