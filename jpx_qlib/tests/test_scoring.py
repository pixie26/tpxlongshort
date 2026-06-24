import pandas as pd

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
