from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_weekly_oil_gpr_df_new import OUTPUT_COLUMNS, build_weekly_frame, write_weekly_frame


def make_oil_daily_frame() -> pd.DataFrame:
    dt = pd.date_range("2024-01-01", "2024-01-21", freq="D")
    return pd.DataFrame(
        {
            "dt": dt,
            "Com_CrudeOil": range(10, 10 + len(dt)),
            "Com_BrentCrudeOil": range(20, 20 + len(dt)),
        }
    )


def make_gpr_daily_frame() -> pd.DataFrame:
    dt = pd.date_range("2024-01-08", "2024-01-21", freq="D")
    return pd.DataFrame(
        {
            "dt": dt,
            "GPRD": range(100, 100 + len(dt)),
            "GPRD_ACT": range(200, 200 + len(dt)),
            "GPRD_THREAT": range(300, 300 + len(dt)),
        }
    )


def test_build_weekly_frame_uses_overlap_only_monday_labeled_mean_contract() -> None:
    frame = build_weekly_frame(make_oil_daily_frame(), make_gpr_daily_frame())

    expected = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2024-01-08", "2024-01-15"]),
            "Com_CrudeOil": [20.0, 27.0],
            "Com_BrentCrudeOil": [30.0, 37.0],
            "Com_Oil_Spread": [10.0, 10.0],
            "GPRD": [103.0, 110.0],
            "GPRD_ACT": [203.0, 210.0],
            "GPRD_THREAT": [303.0, 310.0],
        }
    )

    pd.testing.assert_frame_equal(frame, expected)
    assert list(frame.columns) == OUTPUT_COLUMNS
    assert frame["dt"].is_monotonic_increasing
    assert frame["dt"].is_unique
    assert all(timestamp.weekday() == 0 for timestamp in frame["dt"])
    assert not frame[OUTPUT_COLUMNS[1:]].isna().any().any()


def test_write_weekly_frame_writes_utf8_sig_csv_with_expected_column_order(tmp_path: Path) -> None:
    output_path = tmp_path / "df_new.csv"
    weekly = build_weekly_frame(make_oil_daily_frame(), make_gpr_daily_frame())

    write_weekly_frame(weekly, output_path)

    written = pd.read_csv(output_path, encoding="utf-8-sig")
    assert list(written.columns) == OUTPUT_COLUMNS
    assert written["dt"].tolist() == ["2024-01-08", "2024-01-15"]
    assert written.to_dict(orient="records")[0]["Com_CrudeOil"] == 20.0
    assert written.to_dict(orient="records")[0]["Com_Oil_Spread"] == 10.0
