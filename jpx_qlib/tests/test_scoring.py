import pandas as pd
import pytest

from jpx8qlib.scoring import daily_spread_return, score_summary


def test_spread_positive_for_correct_order():
    df = pd.DataFrame({
        "Date": ["2021-01-01"] * 4 + ["2021-01-02"] * 4,
        "Rank": [0, 1, 2, 3] * 2,
        "Target": [0.04, 0.03, -0.01, -0.02, 0.02, 0.01, -0.01, -0.03],
    })
    daily = daily_spread_return(df, top_n=2)
    assert (daily > 0).all()
    summary = score_summary(daily)
    assert summary["days"] == 2


def test_spread_matches_competition_weighted_sum_not_weighted_average():
    df = pd.DataFrame({
        "Date": ["2021-01-01"] * 4,
        "Rank": [0, 1, 2, 3],
        "Target": [0.04, 0.03, -0.01, -0.02],
    })

    daily = daily_spread_return(df, top_n=2)

    # Normalized weights are [4/3, 2/3]. Competition spread:
    # long = .04*4/3 + .03*2/3
    # short = -.01*2/3 + -.02*4/3
    assert daily.iloc[0] == pytest.approx(0.10666666666666667)


def test_competition_scaling_changes_returns_but_not_sharpe():
    df = pd.DataFrame({
        "Date": ["2021-01-01"] * 4 + ["2021-01-02"] * 4,
        "Rank": [0, 1, 2, 3] * 2,
        "Target": [0.04, 0.03, -0.01, -0.02, 0.02, 0.01, -0.01, -0.03],
    })

    competition_daily = daily_spread_return(df, top_n=2)
    old_weighted_average_daily = competition_daily / 2

    competition = score_summary(competition_daily)
    old = score_summary(old_weighted_average_daily)
    assert competition["mean_daily_spread"] == pytest.approx(2 * old["mean_daily_spread"])
    assert competition["daily_volatility"] == pytest.approx(2 * old["daily_volatility"])
    assert competition["cumulative_spread_simple"] == pytest.approx(2 * old["cumulative_spread_simple"])
    assert competition["competition_sharpe"] == pytest.approx(old["competition_sharpe"])
    assert competition["annualized_sharpe"] == pytest.approx(old["annualized_sharpe"])
