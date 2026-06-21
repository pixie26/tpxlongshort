from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines" / "8th"
sys.path.insert(0, str(BASELINE))
SPEC = importlib.util.spec_from_file_location("baseline_8th_runner", BASELINE / "run_baseline.py")
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

from Preprocessing import StockDataPreprocessor  # noqa: E402
import Features  # noqa: E402


def test_training_columns_are_exact_reference_order():
    assert runner.training_columns() == [
        "SecuritiesCode", "Open", "High", "Low", "Close", "Volume",
        "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag",
        "Amplitude", "OpenCloseReturn", "Return", "Volatility10",
        "Volatility30", "Volatility50", "CloseSMA3", "CloseSMA5",
        "CloseSMA10", "CloseSMA30", "ReturnSMA3", "ReturnSMA5",
        "ReturnSMA10", "ReturnSMA30",
    ]


def test_preprocessing_drops_initial_nan_forwards_isolated_nan_and_adjusts():
    raw = pd.DataFrame({
        "Date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
        "SecuritiesCode": [1301] * 4,
        "Open": [np.nan, 100.0, np.nan, 52.0],
        "High": [np.nan, 110.0, np.nan, 55.0],
        "Low": [np.nan, 90.0, np.nan, 50.0],
        "Close": [np.nan, 104.0, np.nan, 54.0],
        "Volume": [0.0, 10.0, 0.0, 20.0],
        "AdjustmentFactor": [1.0, 0.5, 1.0, 1.0],
        "ExpectedDividend": [np.nan] * 4,
    })
    actual = StockDataPreprocessor.preprocess_for_training(raw)
    assert len(actual) == 3
    assert actual["ExpectedDividend"].eq(0).all()
    assert actual.iloc[1]["Close"] == 208.0
    assert actual.iloc[2]["Open"] == 104.0
    assert actual.iloc[2]["Volume"] == 10.0


def test_initial_rolling_windows_match_online_updates():
    base = pd.DataFrame({"Close": [10.0, 12.0, 9.0, 11.0]})
    returns = Features.Return().add_feature_pandas(base.copy())
    for feature in [Features.SMA("Close", 3), Features.Volatility(3)]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
            batch = feature.copy().add_feature_pandas(returns.copy())[feature.name].to_numpy()
        online_feature = feature.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
            warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
            online = returns.copy().apply(online_feature.update_row, axis=1)[feature.name].to_numpy()
        assert np.allclose(batch, online, equal_nan=True)
    assert Features.SMA("Close", 3).add_feature_pandas(base.copy())["CloseSMA3"].tolist() == [10.0, 11.0, 31.0 / 3.0, 32.0 / 3.0]


def test_ranks_are_unique_contiguous_and_prediction_descending():
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2022-01-01"] * 4),
        "SecuritiesCode": [4, 3, 2, 1],
        "Target": [0.1, 0.2, -0.1, -0.2],
        "Prediction": [0.4, 0.1, 0.3, 0.2],
    })
    ranked = runner.add_daily_ranks(frame)
    assert sorted(ranked["Rank"].tolist()) == [0, 1, 2, 3]
    by_rank = ranked.sort_values("Rank")
    assert by_rank["Prediction"].tolist() == [0.4, 0.3, 0.2, 0.1]


def test_competition_spread_formula():
    targets = np.arange(400, dtype=float) / 1000
    day = pd.DataFrame({"Rank": np.arange(400), "Target": targets})
    weights = np.linspace(2, 1, 200)
    expected = (
        np.sum(targets[:200] * weights) / weights.mean()
        - np.sum(targets[-200:] * weights[::-1]) / weights.mean()
    )
    assert np.isclose(runner.spread_return_per_day(day), expected)


def test_cloned_model_matches_upstream_hash():
    upstream = ROOT / "JPXTokyoStockExchangePrediction" / "winner-models" / "8th" / "lgbm.pickle"
    assert runner.sha256_file(BASELINE / "lgbm.pickle") == runner.sha256_file(upstream)


def test_old_lightgbm_pickle_predicts_through_unchanged_booster():
    model = runner.load_reference_model(BASELINE / "lgbm.pickle")
    sample = pd.DataFrame([[0.0] * 23], columns=runner.training_columns())
    actual = runner.model_predict(model, sample)
    expected = model.booster_.predict(sample)
    assert np.array_equal(actual, expected)
