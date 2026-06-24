import pandas as pd

from jpx8qlib.constants import FEATURE_COLUMNS
from jpx8qlib.data import to_qlib_frame


def test_qlib_frame_contract():
    row = {"Date": "2021-01-01", "SecuritiesCode": 1301, "Target": 0.1}
    row.update({name: 1.0 for name in FEATURE_COLUMNS})
    row["SecuritiesCode"] = 1301
    frame = to_qlib_frame(pd.DataFrame([row]))
    assert frame.index.names == ["datetime", "instrument"]
    assert ("feature", "Close") in frame.columns
    assert ("label", "Target") in frame.columns
