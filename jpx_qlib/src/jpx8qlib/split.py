from __future__ import annotations

import pandas as pd


def segment_mask(dates: pd.Series, start: str, end: str) -> pd.Series:
    values = pd.to_datetime(dates)
    return values.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")


def split_panel(panel: pd.DataFrame, split_cfg: dict) -> dict[str, pd.DataFrame]:
    result = {}
    for name in ("train", "valid", "test"):
        mask = segment_mask(panel["Date"], split_cfg[f"{name}_start"], split_cfg[f"{name}_end"])
        result[name] = panel.loc[mask].copy()
        if result[name].empty:
            raise ValueError(f"Segment {name} is empty under configured dates")
    return result
