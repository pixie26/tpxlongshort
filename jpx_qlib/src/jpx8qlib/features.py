from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import DERIVED_FEATURE_COLUMNS

_REQUIRED = {
    "Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume",
    "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag", "Target",
}


def _validate(raw: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED - set(raw.columns))
    if missing:
        raise ValueError(f"stock_prices is missing columns: {missing}")


def preprocess_published(raw: pd.DataFrame) -> pd.DataFrame:
    """Independent implementation of the preprocessing shown in the public code.

    It intentionally preserves the published semantics:
    - ExpectedDividend NaN -> 0
    - drop leading rows for each stock until Open first exists
    - forward-fill later missing values inside each stock
    - cumulative adjustment factor uses prior rows only
    - adjusted OHLC = raw / CumAdjFactor; adjusted Volume = raw * CumAdjFactor
    """
    _validate(raw)
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values(["SecuritiesCode", "Date"], kind="mergesort").reset_index(drop=True)
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0.0)

    chunks: list[pd.DataFrame] = []
    for _, group in df.groupby("SecuritiesCode", sort=False, observed=True):
        valid = group["Open"].notna().to_numpy()
        if not valid.any():
            continue
        first = int(np.argmax(valid))
        group = group.iloc[first:].copy().ffill()
        chunks.append(group)
    if not chunks:
        raise ValueError("No stock has a non-null Open price")
    df = pd.concat(chunks, ignore_index=True)

    factors = df["AdjustmentFactor"].fillna(1.0)
    df["CumAdjFactor"] = (
        factors.groupby(df["SecuritiesCode"], sort=False)
        .cumprod()
        .groupby(df["SecuritiesCode"], sort=False)
        .shift(1, fill_value=1.0)
    )
    zero_factor = df["CumAdjFactor"].eq(0)
    if zero_factor.any():
        raise ValueError("CumAdjFactor contains zero; cannot adjust prices")

    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].div(
        df["CumAdjFactor"], axis=0
    )
    df["Volume"] = df["Volume"] * df["CumAdjFactor"]
    return df


def add_published_features(preprocessed: pd.DataFrame) -> pd.DataFrame:
    df = preprocessed.copy()
    df = df.sort_values(["SecuritiesCode", "Date"], kind="mergesort")
    grouped = df.groupby("SecuritiesCode", sort=False, observed=True)

    df["Amplitude"] = df["High"] - df["Low"]
    df["OpenCloseReturn"] = np.where(
        df["Open"].ne(0), (df["Close"] - df["Open"]) / df["Open"], 0.0
    )
    df["Return"] = grouped["Close"].pct_change(fill_method=None).fillna(0.0)

    # Public implementation uses all available observations until the window fills,
    # ddof=1, and then fills the one-observation NaN with zero.
    for window in (10, 30, 50):
        df[f"Volatility{window}"] = (
            df.groupby("SecuritiesCode", sort=False, observed=True)["Return"]
            .rolling(window=window, min_periods=1)
            .std(ddof=1)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )

    for col in ("Close", "Return"):
        for window in (3, 5, 10, 30):
            df[f"{col}SMA{window}"] = (
                df.groupby("SecuritiesCode", sort=False, observed=True)[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    df[DERIVED_FEATURE_COLUMNS] = df[DERIVED_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values(["Date", "SecuritiesCode"], kind="mergesort").reset_index(drop=True)


def build_reimplemented_features(raw: pd.DataFrame) -> pd.DataFrame:
    return add_published_features(preprocess_published(raw))
