import pandas as pd

from jpx8qlib.ranking import add_rank


def test_corrected_rank_is_inverse_permutation():
    df = pd.DataFrame({"Date": ["2021-01-01"] * 3, "Prediction": [0.2, 0.9, -0.1]})
    out = add_rank(df, "corrected_rank")
    assert out["Rank"].tolist() == [1, 0, 2]


def test_published_exact_reproduces_public_assignment():
    df = pd.DataFrame({"Date": ["2021-01-01"] * 3, "Prediction": [0.2, 0.9, -0.1]})
    out = add_rank(df, "published_exact")
    assert out["Rank"].tolist() == [1, 0, 2]
    # This tiny ordering happens to match; a different permutation exposes the bug.
    df2 = pd.DataFrame({"Date": ["2021-01-01"] * 4, "Prediction": [0.3, 0.1, 0.4, 0.2]})
    exact = add_rank(df2, "published_exact")["Rank"].tolist()
    corrected = add_rank(df2, "corrected_rank")["Rank"].tolist()
    assert exact == [2, 0, 3, 1]
    assert corrected == [1, 3, 0, 2]
