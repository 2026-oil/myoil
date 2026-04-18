from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import compare_feature_set_aaforecast_fold_mape as compare


def _build_row(*, family: str, backbone: str, retrieval: str, start: float, step: float = 0.0) -> dict[str, str]:
    folds = [round(start + step * idx, 2) for idx in range(16)]
    mean_mape = sum(folds) / len(folds)
    row = {
        "실험군": family,
        "Backbone": backbone,
        "Retrieval": retrieval,
        "Mean MAPE": f"{mean_mape:.2f}%",
    }
    for idx, value in enumerate(folds):
        row[f"Fold {idx}"] = f"{value:.2f}%"
    return row



def _write_result_md(tmp_path: Path, *, target: str, rows: list[dict[str, str]], filename: str) -> Path:
    headers = ["실험군", "Backbone", "Retrieval", *[f"Fold {idx}" for idx in range(16)], "Mean MAPE"]
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        markdown_rows.append("| " + " | ".join(row[header] for header in headers) + " |")

    payload = "\n".join(
        [
            "# synthetic",
            "",
            f"- **예측 타깃:** {target}",
            "",
            "## 04-04. Fold별 MAPE 요약",
            "",
            *markdown_rows,
            "",
            "## 04-05. trailing section",
            "",
        ]
    )
    path = tmp_path / filename
    path.write_text(payload, encoding="utf-8")
    return path



def _rows_for_result_fixture() -> list[dict[str, str]]:
    return [
        _build_row(family="baseline (derived non-ret)", backbone="GRU", retrieval="off", start=4.10),
        _build_row(family="baseline", backbone="GRU", retrieval="on", start=5.20),
        _build_row(family="baseline (derived non-ret)", backbone="Informer", retrieval="off", start=4.00),
        _build_row(family="baseline", backbone="Informer", retrieval="on", start=5.10),
        _build_row(family="baseline (derived non-ret)", backbone="TimeXer", retrieval="off", start=3.90),
        _build_row(family="baseline", backbone="TimeXer", retrieval="on", start=5.00),
        _build_row(family="AAForecast (derived non-ret)", backbone="gru", retrieval="off", start=4.05),
        _build_row(family="AAForecast", backbone="gru", retrieval="on", start=4.65),
        _build_row(family="AAForecast (derived non-ret)", backbone="informer", retrieval="off", start=4.15),
        _build_row(family="AAForecast", backbone="informer", retrieval="on", start=4.55),
        _build_row(family="AAForecast (derived non-ret)", backbone="timexer", retrieval="off", start=4.02),
        _build_row(family="AAForecast", backbone="timexer", retrieval="on", start=4.42),
    ]



def test_parse_result_markdown_extracts_expected_shapes(tmp_path: Path) -> None:
    result_md = _write_result_md(
        tmp_path,
        target="Com_CrudeOil",
        rows=_rows_for_result_fixture(),
        filename="wti_result.md",
    )

    condition_frame, tidy_frame = compare.parse_result_markdown(result_md)

    assert len(condition_frame) == 12
    assert len(tidy_frame) == 12 * 16
    assert condition_frame["commodity"].unique().tolist() == ["WTI"]
    assert condition_frame["fold_count"].unique().tolist() == [16]
    assert set(condition_frame["retrieval"].unique()) == {"off", "on"}
    assert tidy_frame["fold_idx"].min() == 0
    assert tidy_frame["fold_idx"].max() == 15



def test_build_outputs_writes_expected_csvs_and_counts(tmp_path: Path) -> None:
    result_paths = [
        _write_result_md(tmp_path, target="Com_CrudeOil", rows=_rows_for_result_fixture(), filename="wti.md"),
        _write_result_md(tmp_path, target="Com_BrentCrudeOil", rows=_rows_for_result_fixture(), filename="brent.md"),
        _write_result_md(tmp_path, target="Com_DubaiOil", rows=_rows_for_result_fixture(), filename="dubai.md"),
    ]

    output_dir = tmp_path / "out"
    path_map = compare.build_outputs(result_paths, output_dir)

    all_tidy = pd.read_csv(path_map["all_fold_mape_tidy"])
    all_summary = pd.read_csv(path_map["all_retrieval_summary"])
    all_conditions = pd.read_csv(path_map["all_conditions_summary"])
    all_fold_delta = pd.read_csv(path_map["all_fold_delta_summary"])

    assert len(all_tidy) == 3 * 12 * 16
    assert len(all_conditions) == 3 * 12
    assert len(all_summary) == 3 * 6
    assert len(all_fold_delta) == 3 * 6
    assert (output_dir / "wti_fold_mape_tidy.csv").exists()
    assert (output_dir / "brent_retrieval_summary.csv").exists()
    assert (output_dir / "report.md").exists()

    wti_baseline_gru = all_summary[
        (all_summary["commodity"] == "WTI")
        & (all_summary["family"] == "baseline")
        & (all_summary["backbone_key"] == "gru")
    ].iloc[0]
    assert wti_baseline_gru["off_mean_mape_pct"] == pytest.approx(4.10)
    assert wti_baseline_gru["on_mean_mape_pct"] == pytest.approx(5.20)
    assert wti_baseline_gru["delta_mape_pct"] == pytest.approx(1.10)
    assert not bool(wti_baseline_gru["improved_with_retrieval"])

    wti_aa_gru = all_summary[
        (all_summary["commodity"] == "WTI")
        & (all_summary["family"] == "AAForecast")
        & (all_summary["backbone_key"] == "gru")
    ].iloc[0]
    assert wti_aa_gru["delta_mape_pct"] == pytest.approx(0.60)



def test_parse_result_markdown_rejects_unknown_target(tmp_path: Path) -> None:
    result_md = _write_result_md(
        tmp_path,
        target="Com_UnknownOil",
        rows=_rows_for_result_fixture(),
        filename="unknown.md",
    )

    with pytest.raises(ValueError, match="Unsupported target"):
        compare.parse_result_markdown(result_md)
