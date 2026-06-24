from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
import pandas as pd

from .constants import DERIVED_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

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
    started = perf_counter()
    _validate(raw)
    logger.info("Preprocessing: sorting %s raw rows by security/date", f"{len(raw):,}")
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values(["SecuritiesCode", "Date"], kind="mergesort").reset_index(drop=True)
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0.0)

    group_started = perf_counter()
    instrument_count = df["SecuritiesCode"].nunique()
    logger.info("Preprocessing: forward-filling %s securities", f"{instrument_count:,}")
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
    logger.info(
        "Preprocessing: forward-fill complete, %s rows retained in %.1fs",
        f"{len(df):,}",
        perf_counter() - group_started,
    )

    adjustment_started = perf_counter()
    logger.info("Preprocessing: applying cumulative adjustment factors")
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
    logger.info(
        "Preprocessing complete in %.1fs (adjustments %.1fs)",
        perf_counter() - started,
        perf_counter() - adjustment_started,
    )
    return df


def add_published_features(preprocessed: pd.DataFrame) -> pd.DataFrame:
    started = perf_counter()
    logger.info("Features: calculating base returns for %s rows", f"{len(preprocessed):,}")
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
        window_started = perf_counter()
        logger.info("Features: calculating Volatility%d", window)
        df[f"Volatility{window}"] = (
            df.groupby("SecuritiesCode", sort=False, observed=True)["Return"]
            .rolling(window=window, min_periods=1)
            .std(ddof=1)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        logger.info("Features: Volatility%d complete in %.1fs", window, perf_counter() - window_started)

    for col in ("Close", "Return"):
        for window in (3, 5, 10, 30):
            window_started = perf_counter()
            logger.info("Features: calculating %sSMA%d", col, window)
            df[f"{col}SMA{window}"] = (
                df.groupby("SecuritiesCode", sort=False, observed=True)[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            logger.info(
                "Features: %sSMA%d complete in %.1fs",
                col,
                window,
                perf_counter() - window_started,
            )

    logger.info("Features: replacing non-finite values and sorting final panel")
    df[DERIVED_FEATURE_COLUMNS] = df[DERIVED_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result = df.sort_values(["Date", "SecuritiesCode"], kind="mergesort").reset_index(drop=True)
    logger.info("Feature calculation complete in %.1fs", perf_counter() - started)
    return result


def build_legacy_optimized_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the published feature contract with vectorized pandas operations."""
    return add_published_features(preprocess_published(raw))


def build_reimplemented_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for the former engine name."""
    return build_legacy_optimized_features(raw)
