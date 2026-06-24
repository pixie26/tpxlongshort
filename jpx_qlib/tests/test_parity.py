from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jpx8qlib.config import Config
from jpx8qlib.constants import FEATURE_COLUMNS, LABEL_COLUMN
from jpx8qlib.features import build_legacy_optimized_features
from jpx8qlib.legacy import build_legacy_features
from jpx8qlib.parity import compare_frames


LEGACY_CODE_DIR = (
    Path(__file__).resolve().parents[2]
    / "JPXTokyoStockExchangePrediction"
    / "winner-models"
    / "8th"
)


def boundary_sample() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2021-01-01", periods=35, freq="D")
    for offset, date in enumerate(dates):
        open_price = 100.0 + offset
        close_price = open_price + 2.0
        rows.append(
            (
                date,
                1301,
                open_price,
                open_price + 3.0,
                open_price - 1.0,
                close_price,
                1000.0 + offset * 100.0,
                0.5 if offset == 2 else 1.0,
                0.0 if offset == 2 else np.nan,
                0.01 + offset / 1000.0,
            )
        )
        rows.append(
            (
                date,
                1332,
                200.0 + offset * 2.0,
                205.0 + offset * 2.0,
                198.0 + offset * 2.0,
                204.0 + offset * 2.0,
                2000.0 + offset * 100.0,
                2.0 if offset == 1 else 1.0,
                np.nan,
                -0.01 - offset / 1000.0,
            )
        )

    # Leading no-price row is dropped; a later suspended row is forward-filled.
    for column in range(2, 6):
        rows[0] = rows[0][:column] + (np.nan,) + rows[0][column + 1:]
    rows[0] = rows[0][:6] + (0.0,) + rows[0][7:]
    suspended_index = 3 * 2
    for column in range(2, 6):
        rows[suspended_index] = (
            rows[suspended_index][:column]
            + (np.nan,)
            + rows[suspended_index][column + 1:]
        )
    rows[suspended_index] = (
        rows[suspended_index][:6] + (0.0,) + rows[suspended_index][7:]
    )
    frame = pd.DataFrame(
        rows,
        columns=[
            "Date",
            "SecuritiesCode",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "AdjustmentFactor",
            "ExpectedDividend",
            "Target",
        ],
    )
    frame["RowId"] = np.arange(len(frame)).astype(str)
    frame["SupervisionFlag"] = False
    return frame


def test_compare_frames_handles_key_also_present_in_feature_columns():
    frame = build_legacy_optimized_features(boundary_sample())
    report = compare_frames(frame, frame)

    assert report["keys_equal"] is True
    assert report["all_close"] is True
    assert "SecuritiesCode" not in report["columns"]
    assert set(report["columns"]) == set(FEATURE_COLUMNS) - {"SecuritiesCode"}


def test_compare_frames_reports_numeric_difference():
    left = build_legacy_optimized_features(boundary_sample())
    right = left.copy()
    right.loc[right.index[0], "CloseSMA3"] += 1.0

    report = compare_frames(left, right)

    assert report["all_close"] is False
    assert report["columns"]["CloseSMA3"]["matching_fraction"] < 1.0
    assert report["columns"]["CloseSMA3"]["max_abs_difference"] == pytest.approx(1.0)


def test_legacy_optimized_matches_published_boundary_semantics():
    raw = boundary_sample()
    legacy = build_legacy_features(raw, LEGACY_CODE_DIR)
    optimized = build_legacy_optimized_features(raw)

    report = compare_frames(
        legacy,
        optimized,
        columns=FEATURE_COLUMNS + [LABEL_COLUMN],
    )

    assert report["keys_equal"] is True
    assert report["left_rows"] == len(raw) - 1
    assert report["all_close"] is True

    first = optimized.loc[optimized["SecuritiesCode"].eq(1301)].sort_values("Date").iloc[0]
    second = optimized.loc[optimized["SecuritiesCode"].eq(1301)].sort_values("Date").iloc[1]
    suspended = optimized.loc[
        optimized["SecuritiesCode"].eq(1301)
        & optimized["Date"].eq(pd.Timestamp("2021-01-04"))
    ].iloc[0]
    assert first["Return"] == 0.0
    assert first["Volatility10"] == 0.0
    assert second["Volatility10"] == pytest.approx(
        np.std([first["Return"], second["Return"]], ddof=1)
    )
    assert suspended["Open"] == 204.0
    assert suspended["Close"] == 208.0
    assert suspended["Volume"] == 0.0


def test_reimplemented_engine_name_is_compatibility_alias():
    cfg = Config(
        raw={
            "project": {"output_dir": "outputs"},
            "data": {
                "stock_prices_csv": "x",
                "feature_engine": "reimplemented",
            },
        },
        source_path=Path("configs/baseline.yaml"),
    )

    assert cfg.feature_engine == "legacy_optimized"
