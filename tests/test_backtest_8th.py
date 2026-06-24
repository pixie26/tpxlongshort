from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines" / "8th"
sys.path.insert(0, str(BASELINE))
SPEC = importlib.util.spec_from_file_location("backtest_8th_runner", BASELINE / "run_backtest.py")
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def sample_predictions() -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.to_datetime(["2022-01-03", "2022-01-04"])):
        for rank in range(400):
            rows.append({
                "Date": date,
                "SecuritiesCode": 1000 + rank,
                "Target": (200 - rank) / 10000 + date_index / 100000,
                "Prediction_reference": 400 - rank,
                "Rank_reference": rank,
            })
    return pd.DataFrame(rows)


def test_portfolio_has_expected_exposure_and_competition_weight_shape():
    positions = runner.construct_positions(sample_predictions(), "test", "reference")
    by_day = positions.groupby("Date")
    assert by_day.size().eq(400).all()
    assert np.allclose(by_day["Weight"].sum(), 0.0)
    assert np.allclose(by_day["Weight"].apply(lambda x: x.abs().sum()), 1.0)
    day = positions.loc[positions["Date"].eq(positions["Date"].min())]
    assert day.loc[day["Rank"].eq(0), "Weight"].item() > day.loc[day["Rank"].eq(199), "Weight"].item()
    assert abs(day.loc[day["Rank"].eq(399), "Weight"].item()) > abs(day.loc[day["Rank"].eq(200), "Weight"].item())


def test_turnover_and_cost_use_actual_traded_notional_and_charge_entry():
    predictions = sample_predictions()
    second_date = predictions["Date"].max()
    second_day = predictions["Date"].eq(second_date)
    predictions.loc[second_day, "Rank_reference"] = 399 - predictions.loc[second_day, "Rank_reference"]
    positions = runner.construct_positions(predictions, "test", "reference")
    daily = runner.daily_from_positions(positions)
    assert np.isclose(daily.iloc[0]["HalfTurnover"], 0.5)
    assert np.isclose(daily.iloc[0]["TradedNotional"], 1.0)
    assert np.isclose(daily.iloc[1]["HalfTurnover"], 1.0)
    assert np.isclose(daily.iloc[1]["TradedNotional"], 2.0)
    assert np.allclose(daily["TradedNotional"], 2.0 * daily["HalfTurnover"])
    gross, summary0 = runner.summarize_returns(daily, "test", "reference", 0)
    net, summary15 = runner.summarize_returns(daily, "test", "reference", 15)
    assert np.allclose(gross["NetReturn"], gross["GrossReturn"])
    assert np.allclose(
        net["NetReturn"],
        net["GrossReturn"] - net["TradedNotional"] * 0.0015,
    )
    assert np.isclose(net.iloc[0]["TradingCost"], 0.0015)
    assert np.isclose(net.iloc[1]["TradingCost"], 0.0030)
    assert np.allclose(net["GrossReturn"], net["NetReturn"] + net["TradingCost"])
    assert summary15["ending_equity"] <= summary0["ending_equity"]


def test_target_timing_contract_matches_official_jpx_definition():
    spec = ROOT / "data" / "raw" / "jpx" / "data_specifications" / "stock_price_spec.csv"
    target = pd.read_csv(spec).set_index("Column").loc["Target", "Remarks"]
    assert "between t+2 and t+1 where t+0 is TradeDate" in target

    closes = pd.Series([100.0, 110.0, 121.0])
    target_at_t = closes.shift(-2) / closes.shift(-1) - 1.0
    assert np.isclose(target_at_t.iloc[0], 0.10)


def test_rank_validation_rejects_duplicates():
    frame = sample_predictions()
    frame.loc[frame.index[1], "Rank_reference"] = 0
    try:
        runner.validate_ranks(frame, "Rank_reference")
    except RuntimeError:
        return
    raise AssertionError("duplicate ranks should fail validation")
