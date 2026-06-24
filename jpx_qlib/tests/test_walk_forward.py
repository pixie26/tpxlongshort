from pathlib import Path

import pandas as pd
import pytest

from jpx8qlib.config import Config
from jpx8qlib.walk_forward import _rank_ic, build_walk_forward_folds


def _config(tmp_path: Path, purge_days: int = 2) -> Config:
    return Config(
        raw={
            "project": {"output_dir": str(tmp_path / "output")},
            "walk_forward": {
                "purge_days": purge_days,
                "folds": [{
                    "name": "fold_01",
                    "train": ["2020-01-01", "2020-01-06"],
                    "valid": ["2020-01-07", "2020-01-10"],
                    "test": ["2020-01-13", "2020-01-15"],
                }],
            },
        },
        source_path=tmp_path / "configs" / "walk_forward.yaml",
    )


def test_walk_forward_purges_last_trading_days_from_train_and_valid(tmp_path):
    dates = pd.bdate_range("2020-01-01", "2020-01-15")
    panel = pd.DataFrame({"Date": dates, "SecuritiesCode": 1301})

    fold = build_walk_forward_folds(panel, _config(tmp_path))[0]

    assert fold.train_end == pd.Timestamp("2020-01-02")
    assert fold.purged_after_train == (
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-06"),
    )
    assert fold.valid_end == pd.Timestamp("2020-01-08")
    assert fold.purged_after_valid == (
        pd.Timestamp("2020-01-09"),
        pd.Timestamp("2020-01-10"),
    )
    assert fold.test_start == pd.Timestamp("2020-01-13")


def test_walk_forward_rejects_segment_too_short_for_purge(tmp_path):
    panel = pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", "2020-01-15"),
        "SecuritiesCode": 1301,
    })
    config = _config(tmp_path, purge_days=4)

    with pytest.raises(ValueError, match="too short"):
        build_walk_forward_folds(panel, config)


def test_vectorized_rank_ic_matches_daily_spearman():
    frame = pd.DataFrame({
        "Date": ["2020-01-01"] * 4 + ["2020-01-02"] * 4,
        "Prediction": [4.0, 3.0, 1.0, 2.0, 1.0, 2.0, 4.0, 3.0],
        "Target": [0.4, 0.2, 0.1, 0.3, 0.4, 0.3, 0.1, 0.2],
    })
    expected = frame.groupby("Date").apply(
        lambda group: group["Prediction"].corr(group["Target"], method="spearman"),
        include_groups=False,
    )

    mean, median = _rank_ic(frame)

    assert mean == pytest.approx(expected.mean())
    assert median == pytest.approx(expected.median())
