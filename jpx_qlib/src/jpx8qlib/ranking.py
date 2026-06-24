from __future__ import annotations

import numpy as np
import pandas as pd


def _published_exact_one_day(predictions: pd.Series) -> np.ndarray:
    """Reproduce the assignment shown in the public submission code, including its likely bug."""
    ranks = np.arange(len(predictions), dtype=int)
    pairs = sorted(zip(predictions.to_numpy(), ranks), key=lambda item: -item[0])
    return np.asarray([original_position for _, original_position in pairs], dtype=int)


def _corrected_one_day(predictions: pd.Series) -> np.ndarray:
    order = np.argsort(-predictions.to_numpy(), kind="mergesort")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(len(order), dtype=int)
    return ranks


def add_rank(frame: pd.DataFrame, mode: str = "corrected_rank") -> pd.DataFrame:
    required = {"Date", "Prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ranking input missing columns: {sorted(missing)}")
    output = frame.copy()
    func = {
        "published_exact": _published_exact_one_day,
        "corrected_rank": _corrected_one_day,
    }.get(mode)
    if func is None:
        raise ValueError("mode must be 'published_exact' or 'corrected_rank'")
    ranked = []
    for _, group in output.groupby("Date", sort=False):
        part = group.copy()
        part["Rank"] = func(part["Prediction"])
        ranked.append(part)
    return pd.concat(ranked).sort_index()
