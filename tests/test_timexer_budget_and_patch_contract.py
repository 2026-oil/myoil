from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neuralforecast.models.timexer import TimeXer


REPO_ROOT = Path(__file__).resolve().parents[1]
TIME_XER_JOB_FILES = [
    REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_default.yaml",
    REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_1.yaml",
    REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_2.yaml",
    REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_3.yaml",
    REPO_ROOT / "yaml" / "jobs" / "main" / "jobs_4.yaml",
]
HIST_EXOG_LIST = [f"hist_{idx}" for idx in range(16)]


def _load_timexer_params(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job in payload:
        if job["model"] == "TimeXer":
            return dict(job["params"])
    raise AssertionError(f"TimeXer job not found in {path}")


def _build_timexer(params: dict) -> TimeXer:
    return TimeXer(
        h=8,
        input_size=64,
        n_series=17,
        hist_exog_list=HIST_EXOG_LIST,
        futr_exog_list=[],
        stat_exog_list=[],
        **params,
    )


@pytest.mark.parametrize("jobs_path", TIME_XER_JOB_FILES, ids=lambda path: path.stem)
def test_timexer_job_configs_stay_within_budget(jobs_path: Path) -> None:
    params = _load_timexer_params(jobs_path)
    model = _build_timexer(params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert params["patch_len"] in {8, 16}, jobs_path
    assert 80_000 <= trainable_params <= 150_000, (jobs_path, trainable_params)


def test_timexer_rejects_incompatible_patch_geometry() -> None:
    with pytest.raises(ValueError, match=r"patch_len.*divide input_size"):
        _build_timexer(
            {
                "patch_len": 24,
                "hidden_size": 64,
                "n_heads": 4,
                "e_layers": 2,
                "d_ff": 128,
                "factor": 1,
                "dropout": 0.1,
                "use_norm": True,
            }
        )
