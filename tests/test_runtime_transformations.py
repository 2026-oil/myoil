from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from app_config import load_app_config
from runtime_support.runner import (
    _build_fold_diff_context,
    _restore_prediction_series,
    _transform_training_frame,
    _transform_training_series,
)


def _write_runtime_config(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]],
    runtime_payload: dict[str, object],
) -> tuple[object, pd.DataFrame]:
    dataset_path = tmp_path / "series.csv"
    train_df = pd.DataFrame(rows)
    train_df.to_csv(dataset_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "path": str(dataset_path),
                    "target_col": "target",
                    "dt_col": "dt",
                    "hist_exog_cols": ["hist1"],
                },
                "runtime": runtime_payload,
                "jobs": [{"model": "Naive", "params": {}}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    loaded = load_app_config(tmp_path, config_path=config_path)
    return loaded, train_df


def test_second_order_target_diff_restores_predictions_and_keeps_mixed_exog_order(
    tmp_path: Path,
) -> None:
    loaded, train_df = _write_runtime_config(
        tmp_path,
        rows=[
            {"dt": "2024-01-01", "target": 10.0, "hist1": 100.0},
            {"dt": "2024-01-02", "target": 13.0, "hist1": 101.0},
            {"dt": "2024-01-03", "target": 17.0, "hist1": 103.0},
            {"dt": "2024-01-04", "target": 24.0, "hist1": 106.0},
        ],
        runtime_payload={
            "transformations_target": "diff-diff",
            "transformations_exog": "diff",
        },
    )

    diff_context = _build_fold_diff_context(loaded, train_df)

    assert diff_context is not None
    assert diff_context.target_diff_order == 2
    assert diff_context.hist_exog_diff_order == 1

    transformed_train_df = _transform_training_frame(train_df, diff_context)
    assert transformed_train_df["target"].tolist() == [1.0, 3.0]
    assert transformed_train_df["hist1"].tolist() == [2.0, 3.0]

    transformed_target = _transform_training_series(train_df["target"], diff_context)
    assert transformed_target.tolist() == [1.0, 3.0]

    restored_predictions = _restore_prediction_series(
        pd.Series([1.0, 2.0, 0.0]),
        diff_context,
    )
    assert restored_predictions.tolist() == [32.0, 42.0, 52.0]


def test_second_order_exog_diff_uses_shared_alignment_trim(tmp_path: Path) -> None:
    loaded, train_df = _write_runtime_config(
        tmp_path,
        rows=[
            {"dt": "2024-01-01", "target": 10.0, "hist1": 100.0},
            {"dt": "2024-01-02", "target": 13.0, "hist1": 103.0},
            {"dt": "2024-01-03", "target": 17.0, "hist1": 109.0},
            {"dt": "2024-01-04", "target": 24.0, "hist1": 118.0},
        ],
        runtime_payload={
            "transformations_target": "diff",
            "transformations_exog": "diff-diff",
        },
    )

    diff_context = _build_fold_diff_context(loaded, train_df)

    assert diff_context is not None
    assert diff_context.target_diff_order == 1
    assert diff_context.hist_exog_diff_order == 2

    transformed_train_df = _transform_training_frame(train_df, diff_context)
    assert transformed_train_df["target"].tolist() == [4.0, 7.0]
    assert transformed_train_df["hist1"].tolist() == [3.0, 3.0]


def test_second_order_diff_requires_three_training_rows(tmp_path: Path) -> None:
    loaded, train_df = _write_runtime_config(
        tmp_path,
        rows=[
            {"dt": "2024-01-01", "target": 10.0, "hist1": 100.0},
            {"dt": "2024-01-02", "target": 13.0, "hist1": 101.0},
        ],
        runtime_payload={"transformations_target": "diff-diff"},
    )

    with pytest.raises(
        ValueError,
        match="requires at least 3 training rows per fold",
    ):
        _build_fold_diff_context(loaded, train_df)
