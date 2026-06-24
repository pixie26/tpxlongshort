from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from jpx8qlib.diagnostics import (
    _render,
    build_side_attribution,
    portfolio_scenarios,
    side_turnover_decomposition,
)
from jpx8qlib.portfolio import construct_portfolio_positions


def _predictions() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2021-01-04", periods=3, freq="B")
    for date_number, date in enumerate(dates):
        for rank in range(400):
            prediction = float(400 - rank)
            if date_number == 1:
                prediction += 1000 if 100 <= rank < 300 else 0
            rows.append({
                "Date": date,
                "SecuritiesCode": 1000 + rank,
                "Prediction": prediction,
                "Target": (200 - rank) / 10_000,
            })
    return pd.DataFrame(rows)


def test_side_attribution_preserves_accounting_identities():
    positions = construct_portfolio_positions(_predictions())
    turnover = side_turnover_decomposition(positions)
    daily = build_side_attribution(
        positions,
        turnover,
        primary_cost_bps=5,
        fold_ranges=[("fold_01", pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))],
    )
    assert np.allclose(daily["GrossReturn"], daily["LongGross"] + daily["ShortGross"])
    assert np.allclose(daily["TradingCost"], daily["LongCost"] + daily["ShortCost"])
    assert np.allclose(daily["TotalNet"], daily["LongNet"] + daily["ShortNet"])
    assert np.allclose(
        daily["TradedNotional"],
        daily["LongTradedNotional"] + daily["ShortTradedNotional"],
    )


def test_standalone_portfolios_use_consistent_one_hundred_percent_gross():
    positions = construct_portfolio_positions(_predictions())
    turnover = side_turnover_decomposition(positions)
    daily = build_side_attribution(
        positions,
        turnover,
        primary_cost_bps=5,
        fold_ranges=[("fold_01", pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))],
    )
    summary, scenarios = portfolio_scenarios(daily, costs=[0, 5], annualization_days=252)
    assert set(summary["portfolio"]) == {"Long-only", "Short-only", "Long-short"}
    first = scenarios.loc[
        scenarios["Date"].eq(scenarios["Date"].min())
        & scenarios["CostBps"].eq(0)
    ]
    traded = first.set_index("Portfolio")["TradedNotional"]
    assert np.isclose(traded["Long-only"], 1.0)
    assert np.isclose(traded["Short-only"], 1.0)
    assert np.isclose(traded["Long-short"], 1.0)
    short = scenarios.loc[
        scenarios["Portfolio"].eq("Short-only") & scenarios["CostBps"].eq(0)
    ]
    assert np.allclose(short["GrossReturn"], 2.0 * daily["ShortGross"])


def test_diagnostics_report_embeds_valid_json(tmp_path):
    payload = {
        "meta": {"title": "test", "date_range": "test", "scope": "test", "primary_cost_bps": 5},
        "summary": {
            "side_summary_primary_cost": {},
            "turnover": {},
            "retention": {},
            "universe_benchmark": {},
            "diagnosis": {},
            "deferred_2B": [],
        },
        "portfolio_summary": [],
        "daily": [],
        "nav": [],
        "fold": [],
        "yearly": [],
        "monthly": [],
        "benchmark": [],
        "betas": [],
    }
    output = tmp_path / "portfolio_diagnostics.html"
    _render(output, payload)
    text = output.read_text(encoding="utf-8")
    embedded = re.search(r"const DATA=(.*?);\nconst pct=", text, re.S)
    assert embedded is not None
    assert json.loads(embedded.group(1)) == payload
