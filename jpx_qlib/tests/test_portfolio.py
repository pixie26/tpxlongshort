from __future__ import annotations

import json

import numpy as np
import pandas as pd

from jpx8qlib.portfolio import (
    _render_report,
    construct_portfolio_positions,
    daily_portfolio_accounting,
    summarize_cost_scenario,
)


def _predictions(dates: int = 2, names: int = 400) -> pd.DataFrame:
    rows = []
    for date_number, date in enumerate(pd.date_range("2021-01-04", periods=dates, freq="B")):
        for rank in range(names):
            rows.append({
                "Date": date,
                "SecuritiesCode": 1000 + rank,
                "Prediction": float(names - rank),
                "Target": (200 - rank) / 10_000 + date_number / 100_000,
            })
    return pd.DataFrame(rows)


def test_positions_have_separately_normalized_market_neutral_exposure():
    positions = construct_portfolio_positions(_predictions())
    grouped = positions.groupby("Date")
    assert grouped.size().eq(400).all()
    assert np.allclose(grouped["Weight"].sum(), 0.0)
    assert np.allclose(grouped["Weight"].apply(lambda values: values.abs().sum()), 1.0)

    long_weight = positions.loc[positions["Side"].eq("Long")].groupby("Date")["Weight"].sum()
    short_weight = positions.loc[positions["Side"].eq("Short")].groupby("Date")["Weight"].sum()
    assert np.allclose(long_weight, 0.5)
    assert np.allclose(short_weight, -0.5)

    first = positions.loc[positions["Date"].eq(positions["Date"].min())]
    first_long = first.loc[first["Side"].eq("Long")].set_index("SideRank")["Weight"]
    assert first_long.loc[0] > first_long.loc[199]


def test_daily_accounting_charges_one_way_cost_on_actual_traded_notional():
    predictions = _predictions()
    second = predictions["Date"].eq(predictions["Date"].max())
    predictions.loc[second, "Prediction"] *= -1
    positions = construct_portfolio_positions(predictions)
    daily = daily_portfolio_accounting(positions)
    assert np.isclose(daily.iloc[0]["HalfTurnover"], 0.5)
    assert np.isclose(daily.iloc[0]["TradedNotional"], 1.0)
    assert np.isclose(daily.iloc[1]["HalfTurnover"], 1.0)
    assert np.isclose(daily.iloc[1]["TradedNotional"], 2.0)
    assert np.allclose(daily["TradedNotional"], 2.0 * daily["HalfTurnover"])
    assert np.allclose(
        daily["GrossReturn"],
        daily["LongContribution"] + daily["ShortContribution"],
    )

    gross, summary0 = summarize_cost_scenario(daily, cost_bps=0)
    net, summary15 = summarize_cost_scenario(daily, cost_bps=15)
    assert np.allclose(gross["NetReturn"], gross["GrossReturn"])
    assert np.allclose(
        net["NetReturn"],
        net["GrossReturn"] - net["TradedNotional"] * 0.0015,
    )
    assert np.isclose(net.iloc[0]["TradingCost"], 0.0015)
    assert np.isclose(net.iloc[1]["TradingCost"], 0.0030)
    assert np.allclose(
        net["GrossReturn"],
        net["NetReturn"] + net["TradingCost"],
    )
    assert summary15["net_ending_nav"] <= summary0["net_ending_nav"]


def test_report_is_self_contained_and_contains_valid_json(tmp_path):
    payload = {
        "meta": {
            "title": "test",
            "date_range": "test",
            "portfolio": "test",
            "weighting": "test",
            "cost": "test",
            "warning": "test",
            "primary_cost_bps": 0,
        },
        "summary": [],
        "yearly": [],
        "daily": [],
    }
    output = tmp_path / "report.html"
    _render_report(output, payload)
    text = output.read_text(encoding="utf-8")
    assert "__PORTFOLIO_PAYLOAD__" not in text
    assert "JPX Stitched OOS" in text
    embedded = text.split("const DATA=", 1)[1].split(";", 1)[0]
    assert json.loads(embedded) == payload
