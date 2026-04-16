from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts import plot_fold_prediction_overlay as overlay


class _FakeAxis:
    def __init__(self) -> None:
        self.plot_calls: list[tuple[object, object]] = []
        self.plot_kwargs: list[dict[str, object]] = []
        self.vline_calls: list[object] = []
        self.xlim_calls: list[dict[str, object]] = []

    def plot(self, x, y, **kwargs):
        self.plot_calls.append((x, y))
        self.plot_kwargs.append(kwargs)

    def fill_between(self, *_args, **_kwargs):
        return None

    def axvline(self, x, **_kwargs):
        self.vline_calls.append(x)

    def set_xlim(self, **kwargs):
        self.xlim_calls.append(kwargs)

    def set_title(self, *_args, **_kwargs):
        return None

    def set_xlabel(self, *_args, **_kwargs):
        return None

    def set_ylabel(self, *_args, **_kwargs):
        return None

    def grid(self, *_args, **_kwargs):
        return None

    def legend(self, *_args, **_kwargs):
        return None


class _FakeFigure:
    def tight_layout(self):
        return None

    def savefig(self, path, **_kwargs):
        Path(path).write_text("fake-figure", encoding="utf-8")


def test_plot_single_fold_overlay_connects_actual_and_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = _FakeAxis()
    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (_FakeFigure(), axis))
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    fold_frame = pd.DataFrame(
        {
            "run_id": ["run-a", "run-a", "run-b", "run-b"],
            "ds": ["2024-02-25", "2024-03-03", "2024-02-25", "2024-03-03"],
            "horizon_step": [1, 2, 1, 2],
            "y": [12.0, 13.0, 12.0, 13.0],
            "y_hat": [12.5, 13.5, 11.8, 12.8],
        }
    )
    input_actual_frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-02-11", "2024-02-18"]),
            "y": [10.0, 11.0],
        }
    )
    output_actual_frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-02-25", "2024-03-03"]),
            "y": [12.0, 13.0],
        }
    )

    output_path = overlay._plot_single_fold_overlay(
        fold_frame,
        input_actual_frame=input_actual_frame,
        output_actual_frame=output_actual_frame,
        output_path=tmp_path / "fold_000_predictions_overlay.png",
        title="Fold 000",
        show_mean_band=False,
        alpha_per_run=0.9,
    )

    assert output_path.exists()
    assert len(axis.plot_calls) == 4
    output_x, output_y = axis.plot_calls[1]
    run_a_x, run_a_y = axis.plot_calls[2]
    run_b_x, run_b_y = axis.plot_calls[3]
    assert [str(value.date()) for value in output_x.tolist()] == [
        "2024-02-18",
        "2024-02-25",
        "2024-03-03",
    ]
    assert output_y.tolist() == [11.0, 12.0, 13.0]
    assert [str(value.date()) for value in run_a_x.tolist()] == [
        "2024-02-18",
        "2024-02-25",
        "2024-03-03",
    ]
    assert run_a_y.tolist() == [11.0, 12.5, 13.5]
    assert run_b_y.tolist() == [11.0, 11.8, 12.8]
    assert axis.plot_kwargs[2]["label"] == "run-a"
    assert axis.plot_kwargs[3]["label"] == "run-b"


def test_resolve_actual_frames_for_fold_rejects_inconsistent_train_end_ds() -> None:
    fold_frame = pd.DataFrame(
        {
            "run_root": ["/tmp/run-a", "/tmp/run-b"],
            "run_id": ["run-a", "run-b"],
            "train_end_ds": ["2024-02-18", "2024-02-25"],
            "fold_idx": [0, 0],
            "ds": ["2024-02-25", "2024-02-25"],
            "y_hat": [12.5, 12.4],
        }
    )

    with pytest.raises(ValueError, match="inconsistent train_end_ds"):
        overlay._resolve_actual_frames_for_fold(
            fold_frame,
            fold_idx=0,
            history_steps_override=None,
        )


def test_plot_folds_writes_full_and_window_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_calls: list[tuple[int, int | None]] = []
    plot_calls: list[str] = []

    def fake_resolve_actual_frames(fold_frame, *, fold_idx, history_steps_override):
        resolve_calls.append((fold_idx, history_steps_override))
        return (
            pd.DataFrame({"ds": pd.to_datetime(["2024-02-18"]), "y": [11.0]}),
            pd.DataFrame({"ds": pd.to_datetime(["2024-02-25"]), "y": [12.0]}),
        )

    def fake_plot_single_fold_overlay(
        fold_frame,
        *,
        input_actual_frame,
        output_actual_frame,
        output_path,
        title,
        show_mean_band,
        alpha_per_run,
    ):
        plot_calls.append(output_path.name)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(title, encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr(overlay, "_resolve_actual_frames_for_fold", fake_resolve_actual_frames)
    monkeypatch.setattr(overlay, "_plot_single_fold_overlay", fake_plot_single_fold_overlay)

    combined = pd.DataFrame(
        {
            "fold_idx": [0, 0, 1, 1],
            "run_id": ["run-a", "run-b", "run-a", "run-b"],
            "run_root": ["/tmp/run-a", "/tmp/run-b", "/tmp/run-a", "/tmp/run-b"],
            "ds": ["2024-02-25", "2024-02-25", "2024-03-03", "2024-03-03"],
            "horizon_step": [1, 1, 1, 1],
            "y": [12.0, 12.0, 13.0, 13.0],
            "y_hat": [12.5, 12.4, 13.5, 13.4],
            "train_end_ds": ["2024-02-18", "2024-02-18", "2024-02-25", "2024-02-25"],
        }
    )

    paths = overlay.plot_folds(
        combined,
        tmp_path,
        show_mean_band=False,
        alpha_per_run=0.9,
        window_history_steps=16,
    )

    assert [path.name for path in paths] == [
        "fold_000_predictions_overlay.png",
        "fold_000_predictions_overlay_window_16.png",
        "fold_001_predictions_overlay.png",
        "fold_001_predictions_overlay_window_16.png",
    ]
    assert all(path.exists() for path in paths)
    assert resolve_calls == [
        (0, None),
        (0, 16),
        (1, None),
        (1, 16),
    ]
    assert plot_calls == [path.name for path in paths]


def test_plot_continuous_overlay_keeps_fold_segments_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = _FakeAxis()
    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (_FakeFigure(), axis))
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    combined = pd.DataFrame(
        {
            "run_id": ["run-a", "run-a", "run-a", "run-a", "run-b", "run-b", "run-b", "run-b"],
            "display_label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "fold_idx": [0, 0, 1, 1, 0, 0, 1, 1],
            "train_end_ds": [
                "2024-02-18",
                "2024-02-18",
                "2024-03-03",
                "2024-03-03",
                "2024-02-18",
                "2024-02-18",
                "2024-03-03",
                "2024-03-03",
            ],
            "ds": [
                "2024-02-25",
                "2024-03-03",
                "2024-03-10",
                "2024-03-17",
                "2024-02-25",
                "2024-03-03",
                "2024-03-10",
                "2024-03-17",
            ],
            "horizon_step": [1, 2, 1, 2, 1, 2, 1, 2],
            "y_hat": [12.5, 13.5, 14.5, 15.5, 11.5, 12.5, 13.5, 14.5],
        }
    )
    actual_series = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                ["2024-02-11", "2024-02-18", "2024-02-25", "2024-03-03", "2024-03-10", "2024-03-17"]
            ),
            "y": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )
    fold_boundaries = pd.DataFrame(
        {
            "fold_idx": [0, 1],
            "train_end_ds": pd.to_datetime(["2024-02-18", "2024-03-03"]),
        }
    )

    output_path = overlay._plot_continuous_overlay(
        combined,
        actual_series=actual_series,
        fold_boundaries=fold_boundaries,
        output_path=tmp_path / "all_folds_continuous_overlay.png",
        title="continuous",
        show_mean_band=False,
        alpha_per_run=0.9,
        x_start="2024-02-15",
        x_end="2024-03-17",
    )

    assert output_path.exists()
    assert len(axis.plot_calls) == 5
    assert axis.plot_kwargs[1]["label"] == "A"
    assert axis.plot_kwargs[2]["label"] == "_nolegend_"
    assert axis.plot_kwargs[3]["label"] == "B"
    assert axis.plot_kwargs[4]["label"] == "_nolegend_"
    first_run_second_fold_x = axis.plot_calls[2][0]
    assert [str(value.date()) for value in first_run_second_fold_x.tolist()] == [
        "2024-03-10",
        "2024-03-17",
    ]
    assert len(axis.vline_calls) == 2
    assert axis.xlim_calls == [
        {"left": pd.Timestamp("2024-02-18"), "right": pd.Timestamp("2024-03-17")}
    ]


def test_load_shared_actual_series_rejects_mixed_dataset_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        Path("/tmp/run-a"): SimpleNamespace(
            config=SimpleNamespace(
                dataset=SimpleNamespace(
                    path="/tmp/data-a.csv",
                    dt_col="dt",
                    target_col="y",
                )
            )
        ),
        Path("/tmp/run-b"): SimpleNamespace(
            config=SimpleNamespace(
                dataset=SimpleNamespace(
                    path="/tmp/data-b.csv",
                    dt_col="dt",
                    target_col="y",
                )
            )
        ),
    }

    monkeypatch.setattr(
        overlay.runtime,
        "_load_summary_loaded_config",
        lambda run_root: responses[Path(run_root)],
    )
    monkeypatch.setattr(
        overlay.pd,
        "read_csv",
        lambda path: pd.DataFrame({"dt": ["2024-02-18"], "y": [11.0]}),
    )

    with pytest.raises(ValueError, match="share one dataset path"):
        overlay._load_shared_actual_series([Path("/tmp/run-a"), Path("/tmp/run-b")])


def test_annotate_series_identity_splits_multi_model_baseline_run() -> None:
    frame = pd.DataFrame(
        {
            "model": ["GRU", "Informer", "GRU"],
            "y_hat": [1.0, 2.0, 3.0],
        }
    )

    annotated = overlay._annotate_series_identity(
        frame,
        run_id="feature_set_aaforecast_brentoil_baseline-ret_gru_informer",
    )

    assert annotated["series_id"].tolist() == [
        "feature_set_aaforecast_brentoil_baseline-ret_gru_informer::GRU",
        "feature_set_aaforecast_brentoil_baseline-ret_gru_informer::Informer",
        "feature_set_aaforecast_brentoil_baseline-ret_gru_informer::GRU",
    ]
    assert annotated["display_label"].tolist() == [
        "Baseline GRU RET",
        "Baseline Informer RET",
        "Baseline GRU RET",
    ]


def test_collect_hpo_trial_forecasts_builds_combined_and_coverage(tmp_path: Path) -> None:
    hpo_run_root = tmp_path / "feature_set_aaforecast_aaforecast_timexer-ret_HPO"

    def write_trial(
        *,
        study_label: str,
        trial_id: str,
        status: str,
        fold_predictions: dict[int, list[float]],
    ) -> None:
        trial_root = hpo_run_root / "models" / "AAForecast" / "studies" / study_label / "trials" / trial_id
        trial_root.mkdir(parents=True, exist_ok=True)
        (trial_root / "trial_result.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "trial_number": int(trial_id.removeprefix("trial-")),
                    "objective_value": 0.123,
                }
            ),
            encoding="utf-8",
        )
        for fold_idx, y_hats in fold_predictions.items():
            fold_root = trial_root / "folds" / f"fold_{fold_idx:03d}"
            fold_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "model": ["AAForecast"] * len(y_hats),
                    "fold_idx": [fold_idx] * len(y_hats),
                    "train_end_ds": ["2024-02-18"] * len(y_hats),
                    "ds": ["2024-02-25", "2024-03-03"][: len(y_hats)],
                    "horizon_step": [1, 2][: len(y_hats)],
                    "y": [12.0, 13.0][: len(y_hats)],
                    "y_hat": y_hats,
                }
            ).to_csv(fold_root / "predictions.csv", index=False)

    write_trial(
        study_label="study-01",
        trial_id="trial-0000",
        status="complete",
        fold_predictions={0: [12.5, 13.5], 1: [14.5, 15.5]},
    )
    write_trial(
        study_label="study-01",
        trial_id="trial-0001",
        status="failed",
        fold_predictions={0: [11.5, 12.5]},
    )
    write_trial(
        study_label="study-02",
        trial_id="trial-0000",
        status="pruned",
        fold_predictions={1: [10.5, 11.5]},
    )

    combined, coverage, summary = overlay._collect_hpo_trial_forecasts(
        hpo_run_root,
        model_name="AAForecast",
    )

    assert sorted(combined["run_id"].unique().tolist()) == [
        "study-01/trial-0000",
        "study-01/trial-0001",
        "study-02/trial-0000",
    ]
    assert combined["run_root"].drop_duplicates().tolist() == [str(hpo_run_root.resolve())]
    assert summary["study_count"] == 2
    assert summary["trial_count"] == 3
    assert summary["fold_indices"] == [0, 1]
    assert summary["status_counts"] == {"complete": 1, "failed": 1, "pruned": 1}
    assert summary["folds"]["fold_000"] == {
        "plotted_trial_count": 2,
        "skipped_trial_count": 1,
    }
    assert summary["folds"]["fold_001"] == {
        "plotted_trial_count": 2,
        "skipped_trial_count": 1,
    }
    assert coverage[["study_label", "trial_id", "status", "available_fold_count"]].to_dict(
        orient="records"
    ) == [
        {
            "study_label": "study-01",
            "trial_id": "trial-0000",
            "status": "complete",
            "available_fold_count": 2,
        },
        {
            "study_label": "study-01",
            "trial_id": "trial-0001",
            "status": "failed",
            "available_fold_count": 1,
        },
        {
            "study_label": "study-02",
            "trial_id": "trial-0000",
            "status": "pruned",
            "available_fold_count": 1,
        },
    ]
    assert coverage["has_fold_000"].tolist() == [True, True, False]
    assert coverage["has_fold_001"].tolist() == [True, False, True]


def test_plot_hpo_trial_folds_writes_one_png_per_fold_and_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    combined = pd.DataFrame(
        {
            "fold_idx": [0, 0, 1, 1],
            "run_id": [
                "study-01/trial-0000",
                "study-02/trial-0001",
                "study-01/trial-0000",
                "study-02/trial-0001",
            ],
            "run_root": ["/tmp/hpo"] * 4,
            "study_label": ["study-01", "study-02", "study-01", "study-02"],
            "ds": ["2024-02-25", "2024-02-25", "2024-03-03", "2024-03-03"],
            "horizon_step": [1, 1, 1, 1],
            "y": [12.0, 12.0, 13.0, 13.0],
            "y_hat": [12.5, 12.4, 13.5, 13.4],
            "train_end_ds": ["2024-02-18", "2024-02-18", "2024-02-25", "2024-02-25"],
        }
    )
    coverage = pd.DataFrame(
        {
            "study_label": ["study-01", "study-02"],
            "study_index": [1, 2],
            "trial_number": [0, 1],
            "trial_id": ["trial-0000", "trial-0001"],
            "status": ["complete", "failed"],
            "available_fold_count": [2, 2],
            "trial_root": ["/tmp/a", "/tmp/b"],
            "objective_value": [0.1, 0.2],
            "has_fold_000": [True, True],
            "has_fold_001": [True, True],
        }
    )
    summary = {
        "study_count": 2,
        "trial_count": 2,
        "plotted_trial_count": 2,
        "fold_indices": [0, 1],
        "status_counts": {"complete": 1, "failed": 1},
        "folds": {
            "fold_000": {"plotted_trial_count": 2, "skipped_trial_count": 0},
            "fold_001": {"plotted_trial_count": 2, "skipped_trial_count": 0},
        },
    }
    resolve_calls: list[int] = []
    plot_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        overlay,
        "_collect_hpo_trial_forecasts",
        lambda *_args, **_kwargs: (combined, coverage, summary),
    )

    def fake_resolve_actual_frames(fold_frame, *, fold_idx, history_steps_override):
        resolve_calls.append(fold_idx)
        return (
            pd.DataFrame({"ds": pd.to_datetime(["2024-02-18"]), "y": [11.0]}),
            pd.DataFrame({"ds": pd.to_datetime(["2024-02-25"]), "y": [12.0]}),
        )

    def fake_plot_single_fold_overlay(
        fold_frame,
        *,
        input_actual_frame,
        output_actual_frame,
        output_path,
        title,
        show_mean_band,
        alpha_per_run,
        color_by_col,
        color_map,
        show_series_legend,
        group_legend_entries,
    ):
        plot_calls.append(
            {
                "name": Path(output_path).name,
                "title": title,
                "color_by_col": color_by_col,
                "show_series_legend": show_series_legend,
                "group_legend_entries": list(group_legend_entries),
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(title, encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr(overlay, "_resolve_actual_frames_for_fold", fake_resolve_actual_frames)
    monkeypatch.setattr(overlay, "_plot_single_fold_overlay", fake_plot_single_fold_overlay)

    paths, coverage_path, summary_path = overlay.plot_hpo_trial_folds(
        Path("/tmp/hpo"),
        tmp_path,
        model_name="AAForecast",
        show_mean_band=False,
        alpha_per_run=0.9,
    )

    assert [path.name for path in paths] == [
        "fold_000_all_trials_overlay.png",
        "fold_001_all_trials_overlay.png",
    ]
    assert resolve_calls == [0, 1]
    assert plot_calls == [
        {
            "name": "fold_000_all_trials_overlay.png",
            "title": "Fold 000 — plotted 2/2 trials across 2 studies",
            "color_by_col": "study_label",
            "show_series_legend": False,
            "group_legend_entries": [
                ("study-01", overlay._HPO_STUDY_COLORS["study-01"]),
                ("study-02", overlay._HPO_STUDY_COLORS["study-02"]),
            ],
        },
        {
            "name": "fold_001_all_trials_overlay.png",
            "title": "Fold 001 — plotted 2/2 trials across 2 studies",
            "color_by_col": "study_label",
            "show_series_legend": False,
            "group_legend_entries": [
                ("study-01", overlay._HPO_STUDY_COLORS["study-01"]),
                ("study-02", overlay._HPO_STUDY_COLORS["study-02"]),
            ],
        },
    ]
    assert coverage_path.exists()
    assert summary_path.exists()
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["folds"]["fold_000"]["plotted_trial_count"] == 2
