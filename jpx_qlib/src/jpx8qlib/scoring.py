from __future__ import annotations

import math

import numpy as np
import pandas as pd


def linear_weights(n: int, first: float = 2.0, last: float = 1.0) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    weights = np.linspace(first, last, n, dtype=float)
    return weights / weights.mean()


def daily_spread_return(
    ranked: pd.DataFrame,
    top_n: int = 200,
    weight_first: float = 2.0,
    weight_last: float = 1.0,
) -> pd.Series:
    required = {"Date", "Rank", "Target"}
    missing = required - set(ranked.columns)
    if missing:
        raise ValueError(f"Scoring input missing columns: {sorted(missing)}")

    values: dict[pd.Timestamp, float] = {}
    for date, group in ranked.groupby("Date", sort=True):
        group = group.dropna(subset=["Rank", "Target"]).sort_values("Rank")
        n = min(top_n, len(group) // 2)
        if n == 0:
            continue
        weights = linear_weights(n, weight_first, weight_last)
        top = np.average(group.head(n)["Target"].to_numpy(float), weights=weights)
        bottom = np.average(group.tail(n)["Target"].to_numpy(float), weights=weights[::-1])
        values[pd.Timestamp(date)] = float(top - bottom)
    return pd.Series(values, name="daily_spread").sort_index()


def score_summary(daily: pd.Series, annualization_days: int = 252) -> dict[str, float | int]:
    clean = daily.dropna().astype(float)
    if clean.empty:
        raise ValueError("No daily spread returns to score")
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    sharpe = float(mean / std) if std > 0 else math.nan
    return {
        "days": int(len(clean)),
        "mean_daily_spread": mean,
        "daily_volatility": std,
        "competition_sharpe": sharpe,
        "annualized_return_simple": mean * annualization_days,
        "annualized_volatility": std * math.sqrt(annualization_days),
        "annualized_sharpe": sharpe * math.sqrt(annualization_days) if math.isfinite(sharpe) else math.nan,
        "cumulative_spread_simple": float(clean.sum()),
    }
