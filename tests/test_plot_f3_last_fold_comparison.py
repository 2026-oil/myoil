from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from scripts import plot_f3_last_fold_comparison as script


@pytest.fixture()
def f3_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs" / "F3"
    last_cutoff = "2025-02-01 00:00:00"
    ds_values = ["2025-02-08", "2025-02-15"]
    horizon_steps = [1, 2]

    aa_ret_values = {
        "gru": [3.11, 4.11],
        "informer": [3.21, 4.21],
        "timexer": [3.31, 4.31],
    }
    aa_base_values = {
        "gru": [3.01, 4.01],
        "informer": [3.02, 4.02],
        "timexer": [3.03, 4.03],
    }
    baseline_ret_values = {
        "GRU": [3.4, 4.4],
        "Informer": [3.5, 4.5],
        "TimeXer": [3.6, 4.6],
    }
    baseline_base_values = {
        "GRU": [3.14, 4.14],
        "Informer": [3.15, 4.15],
        "TimeXer": [3.16, 4.16],
    }

    for target in ("wti", "brent", "dubai"):
        for backbone in ("gru", "informer", "timexer"):
            run_root = root / f"feature_set_aaforecast_{target}_aaforecast_{backbone}-ret"
            (run_root / "summary").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "model": ["AAForecast", "AAForecast", "AAForecast", "AAForecast"],
                    "fold_idx": [0, 0, 1, 1],
                    "cutoff": ["2025-01-01 00:00:00", "2025-01-01 00:00:00", last_cutoff, last_cutoff],
                    "train_end_ds": ["2025-01-01 00:00:00", "2025-01-01 00:00:00", last_cutoff, last_cutoff],
                    "ds": ["2025-01-08", "2025-01-15", *ds_values],
                    "horizon_step": [1, 2, *horizon_steps],
                    "y": [1.0, 2.0, 3.0, 4.0],
                    "y_hat": [1.1, 2.1, *aa_ret_values[backbone]],
                    "aaforecast_base_prediction": [1.01, 2.01, *aa_base_values[backbone]],
                }
            ).to_csv(run_root / "summary" / "result.csv", index=False)

        baseline_root = root / f"feature_set_aaforecast_{target}_brentoil_baseline-ret_gru_informer"
        (baseline_root / "summary").mkdir(parents=True, exist_ok=True)
        rows = []
        for model_name in ("GRU", "Informer", "TimeXer"):
            for ds, horizon_step, y_hat in zip(ds_values, horizon_steps, baseline_ret_values[model_name], strict=True):
                rows.append(
                    {
                        "model": model_name,
                        "fold_idx": 1,
                        "cutoff": last_cutoff,
                        "train_end_ds": last_cutoff,
                        "ds": ds,
                        "horizon_step": horizon_step,
                        "y": float(horizon_step + 2),
                        "y_hat": y_hat,
                    }
                )
        pd.DataFrame(rows).to_csv(baseline_root / "summary" / "result.csv", index=False)
        for model_name in ("GRU", "Informer", "TimeXer"):
            retrieval_dir = baseline_root / "scheduler" / "workers" / model_name / "retrieval"
            retrieval_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "cutoff": last_cutoff,
                "base_prediction": baseline_base_values[model_name],
                "final_prediction": baseline_ret_values[model_name],
                "memory_prediction": [value + 1.0 for value in baseline_ret_values[model_name]],
                "retrieval_applied": True,
                "skip_reason": None,
            }
            payload_path = retrieval_dir / "retrieval_summary_2025-02-01_000000.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_model_label_order_can_be_subset() -> None:
    assert script._model_label_order(("gru", "timexer")) == (
        "AA-GRU",
        "AA-TimeXer",
        "AA-GRU+Retrival",
        "AA-TimeXer+Retrival",
        "GRU",
        "TimeXer",
        "GRU+Retrival",
        "TimeXer+Retrival",
    )


def test_resolve_label_order_for_paper_subset() -> None:
    assert script._resolve_label_order(
        ("gru", "informer", "timexer"),
        label_set="paper_plain_vs_all_aa_ret",
    ) == (
        "GRU",
        "Informer",
        "TimeXer",
        "AA-GRU+Retrival",
        "AA-Informer+Retrival",
        "AA-TimeXer+Retrival",
    )


def test_resolve_label_order_for_aa_timexer_pair() -> None:
    assert script._resolve_label_order(
        ("timexer",),
        label_set="aa_timexer_pair",
    ) == (
        "AA-TimeXer",
        "AA-TimeXer+Retrival",
    )


def test_labelled_last_fold_frame_builds_requested_labels(f3_root: Path) -> None:
    run_roots = script._discover_target_run_roots(
        f3_root,
        target="wti",
        selected_backbones=("gru", "informer", "timexer"),
    )
    combined, representative_run_root = script._labelled_last_fold_frame(
        target="wti",
        run_roots=run_roots,
        selected_backbones=("gru", "informer", "timexer"),
    )

    assert representative_run_root == run_roots["aa:gru"]
    assert combined["display_label"].astype(str).drop_duplicates().tolist() == list(
        script._model_label_order(("gru", "informer", "timexer"))
    )
    assert set(combined["fold_idx"].unique()) == {1}

    label_to_values = {
        label: combined.loc[combined["display_label"].astype(str) == label, "y_hat"].tolist()
        for label in script._model_label_order(("gru", "informer", "timexer"))
    }
    assert label_to_values["AA-GRU"] == [3.01, 4.01]
    assert label_to_values["AA-GRU+Retrival"] == [3.11, 4.11]
    assert label_to_values["GRU"] == [3.14, 4.14]
    assert label_to_values["GRU+Retrival"] == [3.4, 4.4]


def test_labelled_last_fold_frame_supports_gru_timexer_subset(f3_root: Path) -> None:
    run_roots = script._discover_target_run_roots(
        f3_root,
        target="wti",
        selected_backbones=("gru", "timexer"),
    )
    combined, _ = script._labelled_last_fold_frame(
        target="wti",
        run_roots=run_roots,
        selected_backbones=("gru", "timexer"),
    )

    assert combined["display_label"].astype(str).drop_duplicates().tolist() == [
        "AA-GRU",
        "AA-TimeXer",
        "AA-GRU+Retrival",
        "AA-TimeXer+Retrival",
        "GRU",
        "TimeXer",
        "GRU+Retrival",
        "TimeXer+Retrival",
    ]
    assert "AA-Informer" not in set(combined["display_label"].astype(str))
    assert "Informer" not in set(combined["display_label"].astype(str))


def test_labelled_last_fold_frame_supports_paper_plain_vs_all_aa_ret(f3_root: Path) -> None:
    run_roots = script._discover_target_run_roots(
        f3_root,
        target="wti",
        selected_backbones=("gru", "informer", "timexer"),
    )
    combined, _ = script._labelled_last_fold_frame(
        target="wti",
        run_roots=run_roots,
        selected_backbones=("gru", "informer", "timexer"),
        label_set="paper_plain_vs_all_aa_ret",
    )

    assert combined["display_label"].astype(str).drop_duplicates().tolist() == [
        "GRU",
        "Informer",
        "TimeXer",
        "AA-GRU+Retrival",
        "AA-Informer+Retrival",
        "AA-TimeXer+Retrival",
    ]




def test_variant_output_path_appends_suffix() -> None:
    path = Path("runs/F3/_custom_last_fold_plots/wti_last_fold_all_models_window_16.png")
    assert script._variant_output_path(path, "delta_panel").name == "wti_last_fold_all_models_window_16_delta_panel.png"


def test_generate_target_plots_writes_nine_expected_paths(
    f3_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int | None]] = []

    def fake_plot(combined, *, target, run_root, output_path, history_steps_override):
        del combined, run_root
        calls.append((target, output_path.name, history_steps_override))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("fake", encoding="utf-8")
        return output_path

    monkeypatch.setattr(script, "_plot_target_last_fold", fake_plot)

    outputs = script.generate_target_plots(
        f3_root=f3_root,
        output_dir=tmp_path / "plots",
        selected_backbones=("gru", "timexer"),
    )

    assert len(outputs) == 9
    assert [path.name for path in outputs] == [
        "wti_last_fold_all_models.png",
        "wti_last_fold_all_models_window_8.png",
        "wti_last_fold_all_models_window_16.png",
        "brent_last_fold_all_models.png",
        "brent_last_fold_all_models_window_8.png",
        "brent_last_fold_all_models_window_16.png",
        "dubai_last_fold_all_models.png",
        "dubai_last_fold_all_models_window_8.png",
        "dubai_last_fold_all_models_window_16.png",
    ]
    assert calls == [
        ("wti", "wti_last_fold_all_models.png", None),
        ("wti", "wti_last_fold_all_models_window_8.png", 8),
        ("wti", "wti_last_fold_all_models_window_16.png", 16),
        ("brent", "brent_last_fold_all_models.png", None),
        ("brent", "brent_last_fold_all_models_window_8.png", 8),
        ("brent", "brent_last_fold_all_models_window_16.png", 16),
        ("dubai", "dubai_last_fold_all_models.png", None),
        ("dubai", "dubai_last_fold_all_models_window_8.png", 8),
        ("dubai", "dubai_last_fold_all_models_window_16.png", 16),
    ]




def test_generate_target_plots_writes_comparison_variants(
    f3_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_plot(combined, *, target, run_root, output_path, history_steps_override):
        del combined, target, run_root, history_steps_override
        calls.append(output_path.name)
        return output_path

    monkeypatch.setattr(script, "_plot_target_last_fold", fake_plot)
    monkeypatch.setattr(script, "_plot_target_last_fold_delta_panel", fake_plot)
    monkeypatch.setattr(script, "_plot_target_last_fold_zoom", fake_plot)
    monkeypatch.setattr(script, "_plot_target_last_fold_pointplot", fake_plot)

    outputs = script.generate_target_plots(
        f3_root=f3_root,
        output_dir=tmp_path / "plots",
        selected_backbones=("gru", "informer", "timexer"),
        label_set="paper_plain_vs_all_aa_ret",
        window_sizes=(16,),
        comparison_variants=("delta_panel", "zoom", "pointplot"),
    )

    assert len(outputs) == 24
    assert "wti_last_fold_all_models_window_16_delta_panel.png" in calls
    assert "wti_last_fold_all_models_window_16_zoom.png" in calls
    assert "wti_last_fold_all_models_window_16_pointplot.png" in calls


def test_labelled_last_fold_frame_supports_aa_timexer_pair(f3_root: Path) -> None:
    run_roots = script._discover_target_run_roots(
        f3_root,
        target="wti",
        selected_backbones=("timexer",),
    )
    combined, _ = script._labelled_last_fold_frame(
        target="wti",
        run_roots=run_roots,
        selected_backbones=("timexer",),
        label_set="aa_timexer_pair",
    )
    assert combined["display_label"].astype(str).drop_duplicates().tolist() == [
        "AA-TimeXer",
        "AA-TimeXer+Retrival",
    ]


def test_generate_target_plots_supports_paper_plain_vs_all_aa_ret(
    f3_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_labels: list[list[str]] = []

    def fake_plot(combined, *, target, run_root, output_path, history_steps_override):
        del target, run_root, output_path, history_steps_override
        captured_labels.append(combined["display_label"].astype(str).drop_duplicates().tolist())
        return tmp_path / "fake.png"

    monkeypatch.setattr(script, "_plot_target_last_fold", fake_plot)

    script.generate_target_plots(
        f3_root=f3_root,
        output_dir=tmp_path / "plots",
        selected_backbones=("gru", "informer", "timexer"),
        label_set="paper_plain_vs_all_aa_ret",
        window_sizes=(8,),
    )

    assert captured_labels[0] == [
        "GRU",
        "Informer",
        "TimeXer",
        "AA-GRU+Retrival",
        "AA-Informer+Retrival",
        "AA-TimeXer+Retrival",
    ]
    assert len(captured_labels) == 6


def test_plot_target_last_fold_uses_paper_observed_labels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    combined = pd.DataFrame(
        {
            "display_label": ["AA-GRU", "AA-GRU"],
            "cutoff": ["2025-02-01 00:00:00", "2025-02-01 00:00:00"],
            "ds": ["2025-02-08", "2025-02-15"],
            "horizon_step": [1, 2],
            "y": [3.0, 4.0],
            "y_hat": [3.1, 4.1],
        }
    )
    actual_history = pd.DataFrame({"ds": [pd.Timestamp("2025-02-01")], "y": [2.5]})
    actual_future = pd.DataFrame({"ds": [pd.Timestamp("2025-02-08"), pd.Timestamp("2025-02-15")], "y": [3.0, 4.0]})
    monkeypatch.setattr(
        script.runtime,
        "_summary_overlay_actual_frames",
        lambda *args, **kwargs: (actual_history, actual_future),
    )

    plot_calls: list[dict[str, object]] = []

    class FakeAxis:
        def plot(self, x, y, **kwargs):
            plot_calls.append(kwargs)
        def set_title(self, *args, **kwargs):
            pass
        def set_xlabel(self, *args, **kwargs):
            pass
        def set_ylabel(self, *args, **kwargs):
            pass
        def grid(self, *args, **kwargs):
            pass
        def legend(self, *args, **kwargs):
            pass

    class FakeFigure:
        def tight_layout(self):
            pass
        def savefig(self, *args, **kwargs):
            pass

    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_pyplot.subplots = lambda figsize=None: (FakeFigure(), FakeAxis())
    fake_pyplot.close = lambda fig: None
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.use = lambda backend: None
    fake_matplotlib.pyplot = fake_pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    script._plot_target_last_fold(
        combined,
        target="wti",
        run_root=tmp_path,
        output_path=tmp_path / "out.png",
        history_steps_override=None,
    )

    assert plot_calls[0]["label"] == "Observed history"
    assert plot_calls[0]["marker"] == "o"
    assert plot_calls[1]["label"] == "Observed future"
    assert plot_calls[1]["marker"] == "o"
