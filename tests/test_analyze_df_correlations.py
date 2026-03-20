from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_df_correlations.py"
    spec = importlib.util.spec_from_file_location("analyze_df_correlations", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_correlation_script_emits_four_target_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()

    source = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "df.csv",
        encoding="utf-8-sig",
        usecols=[
            "dt",
            "Com_CrudeOil",
            "Com_BrentCrudeOil",
            "Com_Oil_Spread",
            "Com_Gasoline",
            "Idx_OVX",
        ],
        nrows=12,
    )
    input_path = tmp_path / "mini.csv"
    output_dir = tmp_path / "out"
    source.to_csv(input_path, index=False, encoding="utf-8-sig")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_df_correlations.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    module.main()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert sorted(summary["targets"]) == [
        "Com_BrentCrudeOil",
        "Com_CrudeOil",
        "diff(Com_BrentCrudeOil)",
        "diff(Com_CrudeOil)",
    ]

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    display_files = sorted(tables_dir.glob("*display.csv"))
    assert [path.name for path in display_files] == [
        "diff_Com_BrentCrudeOil_correlations_display.csv",
        "diff_Com_CrudeOil_correlations_display.csv",
        "raw_Com_BrentCrudeOil_correlations_display.csv",
        "raw_Com_CrudeOil_correlations_display.csv",
    ]

    for path in display_files:
        frame = pd.read_csv(path)
        assert len(frame) == 4
        assert {"variable", "pearson_corr", "abs_corr", "sign", "n_obs"} <= set(frame.columns)
        assert frame["pearson_corr"].round(3).equals(frame["pearson_corr"])

    assert (output_dir / "report.md").exists()
    assert (figures_dir / "raw_Com_CrudeOil_correlations_table.png").exists()
    assert (figures_dir / "diff_Com_BrentCrudeOil_correlations_bar.png").exists()
