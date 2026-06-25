import logging
from pathlib import Path

import numpy as np
import pandas as pd

from jpx8qlib.config import Config
from jpx8qlib.data import load_raw_stock_prices, prepare_panel


def test_load_raw_stock_prices_combines_sources_in_security_date_order(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({
        "Date": ["2021-01-04", "2021-01-05"],
        "SecuritiesCode": [2, 1],
    }).to_csv(first, index=False)
    pd.DataFrame({
        "Date": ["2021-01-06", "2021-01-05"],
        "SecuritiesCode": [1, 2],
    }).to_csv(second, index=False)

    combined = load_raw_stock_prices([first, second])

    assert list(
        combined[["SecuritiesCode", "Date"]].itertuples(index=False, name=None)
    ) == [
        (1, pd.Timestamp("2021-01-05")),
        (1, pd.Timestamp("2021-01-06")),
        (2, pd.Timestamp("2021-01-04")),
        (2, pd.Timestamp("2021-01-05")),
    ]


def test_prepare_panel_reports_progress(caplog, tmp_path):
    source = tmp_path / "stock_prices.csv"
    pd.DataFrame({
        "RowId": ["a", "b", "c"],
        "Date": ["2021-01-01", "2021-01-02", "2021-01-03"],
        "SecuritiesCode": [1301, 1301, 1301],
        "Open": [100.0, 101.0, 102.0],
        "High": [102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [101.0, 102.0, 103.0],
        "Volume": [1000.0, 1100.0, 1200.0],
        "AdjustmentFactor": [1.0, 1.0, 1.0],
        "ExpectedDividend": [np.nan, np.nan, np.nan],
        "SupervisionFlag": [False, False, False],
        "Target": [0.1, 0.2, 0.3],
    }).to_csv(source, index=False)

    cfg = Config(
        raw={
            "project": {"output_dir": str(tmp_path / "output")},
            "data": {
                "stock_prices_csv": str(source),
                "feature_engine": "reimplemented",
                "cache_file": "prepared.pkl.gz",
            },
        },
        source_path=Path(tmp_path / "configs" / "baseline.yaml"),
    )

    with caplog.at_level(logging.INFO, logger="jpx8qlib"):
        panel = prepare_panel(cfg, force=True)

    messages = [record.getMessage() for record in caplog.records]
    assert len(panel) == 3
    assert any("Reading stock prices CSV" in message for message in messages)
    assert any("Features: calculating Volatility10" in message for message in messages)
    assert any("Writing compressed cache" in message for message in messages)
    assert any("Preparation complete" in message for message in messages)
