from __future__ import annotations

import numpy as np
import pandas as pd

from jpx8qlib.strategy_experiments import (
    StrategySpec,
    construct_stateful_positions,
    evaluate_positions,
)


def _predictions(dates: int = 6, names: int = 20) -> pd.DataFrame:
    rows = []
    for day, date in enumerate(pd.bdate_range("2021-01-04", periods=dates)):
        order = np.roll(np.arange(names), day % 3)
        for rank, code_offset in enumerate(order):
            rows.append({
                "Date": date,
                "SecuritiesCode": 1000 + int(code_offset),
                "Prediction": float(names - rank),
                "Target": float((names / 2 - rank) / 10_000),
            })
    return pd.DataFrame(rows)


def test_buffer_reduces_membership_churn_without_breaking_exposure():
    predictions = _predictions()
    baseline, _ = construct_stateful_positions(
        predictions, StrategySpec("baseline", "baseline", top_n=4)
    )
    buffered, _ = construct_stateful_positions(
        predictions,
        StrategySpec(
            "buffer", "buffer", top_n=4,
            long_entry=3, long_exit=5, short_entry=3, short_exit=5,
        ),
    )
    _, base_summary = evaluate_positions(baseline, cost_bps=5, annualization_days=252)
    _, buffer_summary = evaluate_positions(buffered, cost_bps=5, annualization_days=252)
    assert buffer_summary["average_daily_traded_notional"] < (
        base_summary["average_daily_traded_notional"]
    )
    gross = buffered.groupby("Date")["Weight"].apply(lambda values: values.abs().sum())
    assert np.allclose(gross, 1.0)


def test_minimum_holding_retains_name_until_age_threshold():
    predictions = _predictions(dates=4)
    positions, _ = construct_stateful_positions(
        predictions,
        StrategySpec("hold3", "minimum_holding", top_n=2, minimum_holding_days=3),
    )
    first_long = set(
        positions.loc[
            (positions["Date"].eq(positions["Date"].min()))
            & positions["Side"].eq("Long"),
            "SecuritiesCode",
        ]
    )
    third_date = sorted(positions["Date"].unique())[2]
    third_long = set(
        positions.loc[
            positions["Date"].eq(third_date) & positions["Side"].eq("Long"),
            "SecuritiesCode",
        ]
    )
    assert first_long <= third_long


def test_smoothing_uses_only_current_and_past_predictions():
    predictions = _predictions(dates=3)
    original, _ = construct_stateful_positions(
        predictions, StrategySpec("smooth", "smoothing", top_n=2, smoothing_days=3)
    )
    changed = predictions.copy()
    changed.loc[changed["Date"].eq(changed["Date"].max()), "Prediction"] *= -100
    revised, _ = construct_stateful_positions(
        changed, StrategySpec("smooth", "smoothing", top_n=2, smoothing_days=3)
    )
    first_two = sorted(predictions["Date"].unique())[:2]
    left = original.loc[original["Date"].isin(first_two), ["Date", "SecuritiesCode", "Weight"]]
    right = revised.loc[revised["Date"].isin(first_two), ["Date", "SecuritiesCode", "Weight"]]
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))


def test_long_only_accounting_treats_missing_short_side_as_zero():
    positions, _ = construct_stateful_positions(
        _predictions(),
        StrategySpec(
            "long_only", "gross_control", top_n=4, long_gross=1.0, short_gross=0.0
        ),
    )
    daily, summary = evaluate_positions(positions, cost_bps=5, annualization_days=252)
    assert np.allclose(daily["ShortContribution"], 0.0)
    assert np.allclose(daily["GrossReturn"], daily["LongContribution"])
    assert np.isfinite(summary["net_sharpe"])
