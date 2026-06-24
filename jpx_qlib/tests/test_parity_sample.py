from pathlib import Path

import pandas as pd

from jpx8qlib.config import Config
from jpx8qlib.data import select_parity_sample


def test_parity_sample_keeps_full_instrument_histories():
    raw = pd.DataFrame({
        "Date": pd.to_datetime(["2021-01-01", "2021-01-02"] * 3),
        "SecuritiesCode": [1001, 1001, 1002, 1002, 1003, 1003],
        "AdjustmentFactor": [1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
    })
    cfg = Config(
        raw={"project": {"output_dir": "outputs"}, "data": {"stock_prices_csv": "x"},
             "parity": {"max_instruments": 2, "prefer_adjustment_events": True}},
        source_path=Path("configs/baseline.yaml"),
    )
    sample = select_parity_sample(raw, cfg)
    assert sorted(sample["SecuritiesCode"].unique().tolist()) == [1001, 1002]
    assert sample.groupby("SecuritiesCode").size().to_dict() == {1001: 2, 1002: 2}
