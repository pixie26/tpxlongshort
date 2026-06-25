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


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    offset: float = 0.0,
) -> pd.Series:
    values = numerator.div(denominator.where(denominator.ne(0))) + offset
    return values.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_mean(
    values: pd.Series,
    groups: pd.Series,
    window: int,
) -> pd.Series:
    return (
        values.groupby(groups, sort=False)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _rolling_std(
    values: pd.Series,
    groups: pd.Series,
    window: int,
) -> pd.Series:
    return (
        values.groupby(groups, sort=False)
        .rolling(window=window, min_periods=2)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )


def _grouped_rolling_corr(
    left: pd.Series,
    right: pd.Series,
    groups: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Memory-bounded rolling correlation preserving the original row index."""
    left_values = left.to_numpy(dtype=float, copy=False)
    right_values = right.to_numpy(dtype=float, copy=False)
    result = np.zeros(len(left_values), dtype=float)
    for positions in groups.groupby(groups, sort=False).indices.values():
        loc = np.asarray(positions, dtype=int)
        local_left = pd.Series(left_values[loc], copy=False)
        local_right = pd.Series(right_values[loc], copy=False)
        result[loc] = (
            local_left.rolling(window=window, min_periods=min_periods)
            .corr(local_right)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy()
        )
    return pd.Series(result, index=left.index)


def add_experiment_feature_groups(
    panel: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    """Add fixed, point-in-time research features to the published panel.

    All time-series operations are grouped by security after an explicit
    security/date sort. Cross-sectional means use only rows from the same date.
    """
    if not groups:
        return panel
    df = panel.sort_values(
        ["SecuritiesCode", "Date"], kind="mergesort"
    ).copy()
    codes = df["SecuritiesCode"]
    close_grouped = df.groupby("SecuritiesCode", sort=False, observed=True)["Close"]
    previous_close = close_grouped.shift(1)

    if "relative_price" in groups:
        df["OpenToPrevClose"] = _safe_ratio(df["Open"], previous_close, offset=-1.0)
        df["HighToPrevClose"] = _safe_ratio(df["High"], previous_close, offset=-1.0)
        df["LowToPrevClose"] = _safe_ratio(df["Low"], previous_close, offset=-1.0)
        df["HighLowToPrevClose"] = _safe_ratio(
            df["High"] - df["Low"], previous_close
        )
        for window in (5, 10, 20, 60):
            average = _rolling_mean(df["Close"], codes, window)
            df[f"CloseToSMA{window}"] = _safe_ratio(
                df["Close"], average, offset=-1.0
            )

    traded_value = df["Close"] * df["Volume"]
    if "normalized_volume" in groups:
        df["LogVolume"] = np.log1p(df["Volume"].clip(lower=0.0))
        volume_means = {
            window: _rolling_mean(df["Volume"], codes, window)
            for window in (5, 20, 60)
        }
        for window, average in volume_means.items():
            df[f"VolumeToMean{window}"] = _safe_ratio(
                df["Volume"], average
            ).clip(0.1, 10.0)
        df["LogVolumeToADV20"] = np.log(
            df["VolumeToMean20"].clip(0.1, 10.0)
        )
        volume_std20 = _rolling_std(df["Volume"], codes, 20)
        df["VolumeZScore20"] = _safe_ratio(
            df["Volume"] - volume_means[20], volume_std20
        ).clip(-5.0, 5.0)
        df["TradedValue"] = traded_value
        traded_value_mean20 = _rolling_mean(traded_value, codes, 20)
        df["TradedValueToMean20"] = _safe_ratio(
            traded_value, traded_value_mean20
        ).clip(0.1, 10.0)

    if "momentum_reversal" in groups:
        returns: dict[int, pd.Series] = {1: df["Return"]}
        for window in (2, 5, 10, 20, 60):
            returns[window] = close_grouped.pct_change(
                periods=window, fill_method=None
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df[f"Return{window}d"] = returns[window]
        df["MomentumSlope5_20"] = returns[5] - returns[20] / 4.0
        df["MomentumSlope20_60"] = returns[20] - returns[60] / 3.0
        for window in (5, 20):
            universe = returns[window].groupby(df["Date"], sort=False).transform("mean")
            df[f"ExcessReturn{window}d"] = returns[window] - universe

    normalized_range = _safe_ratio(df["High"] - df["Low"], previous_close)
    if "volatility_range" in groups:
        negative_return = df["Return"].clip(upper=0.0)
        for window in (5, 20, 60):
            df[f"RealizedVol{window}d"] = _rolling_std(
                df["Return"], codes, window
            )
        downside_squared = negative_return.pow(2)
        df["DownsideVol20d"] = np.sqrt(
            _rolling_mean(downside_squared, codes, 20)
        )
        for window in (5, 20):
            df[f"HighLowRange{window}d"] = _rolling_mean(
                normalized_range, codes, window
            )
        df["VolatilityRatio5_20"] = _safe_ratio(
            df["RealizedVol5d"], df["RealizedVol20d"]
        ).clip(0.0, 10.0)
        df["VolatilityRatio20_60"] = _safe_ratio(
            df["RealizedVol20d"], df["RealizedVol60d"]
        ).clip(0.0, 10.0)

    if "liquidity_dynamics" in groups:
        adv = {
            window: _rolling_mean(traded_value, codes, window)
            for window in (5, 20, 60)
        }
        df["ADV5ToADV20"] = _safe_ratio(adv[5], adv[20]).clip(0.1, 10.0)
        df["ADV20ToADV60"] = _safe_ratio(adv[20], adv[60]).clip(0.1, 10.0)
        df["TradedValueShock20"] = _safe_ratio(
            traded_value, adv[20]
        ).clip(0.1, 10.0)
        df["ZeroVolumeFrequency20"] = _rolling_mean(
            df["Volume"].eq(0).astype(float), codes, 20
        )
        amihud_daily = df["Return"].abs().div(
            traded_value.where(traded_value.gt(0))
        ).replace([np.inf, -np.inf], np.nan)
        df["AmihudProxy20"] = _rolling_mean(
            amihud_daily.fillna(0.0), codes, 20
        )
        log_volume_change = np.log1p(df["Volume"].clip(lower=0.0)).groupby(
            codes, sort=False
        ).diff()
        df["VolumePriceDivergence20"] = _grouped_rolling_corr(
            log_volume_change.fillna(0.0),
            df["Return"],
            codes,
            window=20,
            min_periods=5,
        )

    added = [
        column for column in df.columns
        if column not in panel.columns
    ]
    if added:
        df[added] = df[added].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values(
        ["Date", "SecuritiesCode"], kind="mergesort"
    ).reset_index(drop=True)
