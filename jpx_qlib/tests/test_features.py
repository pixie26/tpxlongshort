import numpy as np
import pandas as pd

from jpx8qlib.features import build_reimplemented_features


def sample():
    return pd.DataFrame({
        "RowId": ["a", "b", "c", "d"],
        "Date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
        "SecuritiesCode": [1301] * 4,
        "Open": [100.0, 101.0, 51.0, 52.0],
        "High": [102.0, 103.0, 53.0, 54.0],
        "Low": [99.0, 100.0, 50.0, 51.0],
        "Close": [101.0, 102.0, 52.0, 53.0],
        "Volume": [1000.0] * 4,
        "AdjustmentFactor": [1.0, 0.5, 1.0, 1.0],
        "ExpectedDividend": [np.nan] * 4,
        "SupervisionFlag": [False] * 4,
        "Target": [0.1, 0.2, 0.3, 0.4],
    })


def test_published_features_and_adjustment():
    out = build_reimplemented_features(sample())
    assert out["ExpectedDividend"].eq(0).all()
    assert out.loc[out["Date"].eq(pd.Timestamp("2021-01-03")), "Close"].iloc[0] == 104.0
    assert out["Amplitude"].iloc[0] == 3.0
    assert out["Return"].iloc[0] == 0.0
    assert np.isfinite(out.filter(regex="Volatility|SMA").to_numpy()).all()
