from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import feature_set_aaforecast_postprocess as postprocess


def test_iter_passed_run_entries_expands_fanout_and_groups_configs(
    monkeypatch,
) -> None:
    baseline_loaded = SimpleNamespace(
        jobs_fanout_specs=(SimpleNamespace(route_slug="gru_informer"),),
        active_jobs_route_slug=None,
    )
    aa_loaded = SimpleNamespace(
        jobs_fanout_specs=(),
        active_jobs_route_slug=None,
    )

    def fake_load_app_config(_repo_root, *, config_path):
        if config_path.endswith("baseline-ret.yaml"):
            return baseline_loaded
        if config_path.endswith("aaforecast-gru.yaml"):
            return aa_loaded
        raise AssertionError(config_path)

    monkeypatch.setattr(postprocess, "load_app_config", fake_load_app_config)
    monkeypatch.setattr(
        postprocess,
        "loaded_config_for_jobs_fanout",
        lambda _repo_root, _loaded, spec: f"variant:{spec.route_slug}",
    )
    monkeypatch.setattr(
        postprocess.runtime,
        "_default_output_root",
        lambda _repo_root, loaded: Path("/tmp")
        / (
            "feature_set_aaforecast_brentoil_baseline-ret_gru_informer"
            if loaded == "variant:gru_informer"
            else "feature_set_aaforecast_aaforecast_gru"
        ),
    )

    entries = postprocess._iter_passed_run_entries(
        Path("/repo"),
        {
            "results": [
                {
                    "config": "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml",
                    "status": "passed",
                },
                {
                    "config": "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml",
                    "status": "passed",
                },
                {
                    "config": "yaml/experiment/feature_set_aaforecast/aaforecast-informer.yaml",
                    "status": "failed",
                },
            ]
        },
    )

    assert entries == [
        {
            "config": "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml",
            "group": "ret",
            "jobs_route": "gru_informer",
            "canonical_run_root": "/tmp/feature_set_aaforecast_brentoil_baseline-ret_gru_informer",
            "run_name": "feature_set_aaforecast_brentoil_baseline-ret_gru_informer",
        },
        {
            "config": "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml",
            "group": "nonret",
            "jobs_route": None,
            "canonical_run_root": "/tmp/feature_set_aaforecast_aaforecast_gru",
            "run_name": "feature_set_aaforecast_aaforecast_gru",
        },
    ]


def test_main_links_runs_writes_manifest_and_calls_both_group_plots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_dir = tmp_path / "logs-source"
    log_dir.mkdir()
    ret_root = tmp_path / "canonical-ret"
    ret_root.mkdir()
    nonret_root = tmp_path / "canonical-nonret"
    nonret_root.mkdir()
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "log_dir": str(log_dir),
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "config": "yaml/experiment/feature_set_aaforecast/baseline-ret.yaml",
            "group": "ret",
            "jobs_route": "gru_informer",
            "canonical_run_root": str(ret_root),
            "run_name": ret_root.name,
        },
        {
            "config": "yaml/experiment/feature_set_aaforecast/aaforecast-gru.yaml",
            "group": "nonret",
            "jobs_route": None,
            "canonical_run_root": str(nonret_root),
            "run_name": nonret_root.name,
        },
    ]
    plot_calls: list[tuple[str, str | None, str | None]] = []

    monkeypatch.setattr(
        postprocess,
        "_iter_passed_run_entries",
        lambda _repo_root, _summary_payload: entries,
    )

    def fake_write_group_plot(*, raw_batch_root, entries, group, x_start, x_end):
        plot_calls.append((group, x_start, x_end))
        plot_path = raw_batch_root / "plots" / group / "all_folds_continuous_overlay.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_path.write_text(group, encoding="utf-8")
        return plot_path

    monkeypatch.setattr(postprocess, "_write_group_plot", fake_write_group_plot)

    raw_batch_root = tmp_path / "raw_feature_set_aaforecast" / "20260417T000000Z"
    assert (
        postprocess.main(
            [
                "--summary-json",
                str(summary_json),
                "--raw-batch-root",
                str(raw_batch_root),
                "--x-start",
                "2025-08-15",
                "--x-end",
                "2026-03-09",
            ]
        )
        == 0
    )

    assert (raw_batch_root / "runs" / ret_root.name).is_symlink()
    assert (raw_batch_root / "runs" / nonret_root.name).is_symlink()
    assert (raw_batch_root / "logs").is_symlink()
    manifest = json.loads((raw_batch_root / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["log_dir"] == str(log_dir.resolve())
    assert len(manifest["entries"]) == 2
    assert plot_calls == [
        ("ret", "2025-08-15", "2026-03-09"),
        ("nonret", "2025-08-15", "2026-03-09"),
    ]
