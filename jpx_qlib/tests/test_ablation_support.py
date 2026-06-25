from __future__ import annotations

import numpy as np
import pandas as pd

from jpx8qlib.config import Config, load_config
from jpx8qlib.data import apply_experiment_features
from jpx8qlib.features import add_experiment_feature_groups
from jpx8qlib.model import RidgeModel
from jpx8qlib.model import PublishedLGBM
from jpx8qlib.sector_diagnostics import (
    apply_sector_prediction_transform,
    sector_exposure_daily,
)
from jpx8qlib.training_report import _average_predictions


def test_nested_config_resolves_project_root_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    source = tmp_path / "configs" / "ablations" / "a1.yaml"
    source.parent.mkdir(parents=True)
    config = Config(raw={"project": {"output_dir": "outputs/a1"}}, source_path=source)
    assert config.project_root == tmp_path
    assert config.output_dir == tmp_path / "outputs" / "a1"


def test_config_extends_deep_merges_parent(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    parent = tmp_path / "configs" / "parent.yaml"
    child = tmp_path / "configs" / "research" / "child.yaml"
    child.parent.mkdir(parents=True)
    parent.write_text(
        "project: {output_dir: outputs/base}\n"
        "model: {type: lightgbm, params: {n_estimators: 100, num_leaves: 31}}\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: ../parent.yaml\n"
        "project: {output_dir: outputs/child}\n"
        "model: {params: {n_estimators: 1000}}\n",
        encoding="utf-8",
    )
    config = load_config(child)
    assert config.output_dir == tmp_path / "outputs" / "child"
    assert config.raw["model"]["params"] == {
        "n_estimators": 1000,
        "num_leaves": 31,
    }


def test_ridge_standardization_is_fit_on_training_rows_only():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0], "Target": [0.0, 1.0, 2.0]})
    future = pd.DataFrame({"x": [1000.0]})
    model = RidgeModel({"alpha": 1.0}, ["x"]).fit(train)
    center = model.estimator.named_steps["scale"].mean_[0]
    assert center == 1.0
    assert np.isfinite(model.predict(future).iloc[0])


def test_lightgbm_early_stopping_records_and_uses_best_iteration():
    train = pd.DataFrame({
        "x": np.arange(100, dtype=float),
        "Target": np.sin(np.arange(100, dtype=float) / 10),
    })
    valid = pd.DataFrame({
        "x": np.arange(100, 130, dtype=float),
        "Target": np.sin(np.arange(100, 130, dtype=float) / 10),
    })
    model = PublishedLGBM(
        {
            "objective": "regression",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "verbosity": -1,
            "random_state": 42,
            "early_stopping_rounds": 5,
            "eval_metric": "rmse",
        },
        feature_columns=["x"],
        categorical_features=[],
    ).fit(train, valid)
    assert 1 <= model.best_iteration <= 100
    assert np.isfinite(model.predict(valid)).all()


def test_cross_sectional_rank_uses_only_same_day_and_skips_binary_field(tmp_path):
    config = Config(
        raw={
            "project": {"output_dir": "outputs/test"},
            "features": {
                "columns": ["Close", "Volume", "SupervisionFlag"],
                "categorical": ["SupervisionFlag"],
                "cross_sectional_transform": "percentile_rank",
            },
        },
        source_path=tmp_path / "configs" / "rank.yaml",
    )
    panel = pd.DataFrame({
        "Date": pd.to_datetime(
            ["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-05"]
        ),
        "SecuritiesCode": [1, 2, 1, 2],
        "Close": [10.0, 20.0, 1000.0, 2000.0],
        "Volume": [100.0, 50.0, 1.0, 2.0],
        "SupervisionFlag": [False, True, False, True],
    })
    ranked = apply_experiment_features(panel, config)
    assert ranked["Close"].tolist() == [0.5, 1.0, 0.5, 1.0]
    assert ranked["Volume"].tolist() == [1.0, 0.5, 0.5, 1.0]
    assert ranked["SupervisionFlag"].tolist() == [False, True, False, True]


def _research_panel() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=8)
    return pd.DataFrame({
        "Date": dates,
        "SecuritiesCode": [1301] * len(dates),
        "Open": [100, 101, 102, 103, 104, 105, 106, 107],
        "High": [102, 103, 104, 105, 106, 107, 108, 109],
        "Low": [99, 100, 101, 102, 103, 104, 105, 106],
        "Close": [101, 102, 103, 104, 105, 106, 107, 108],
        "Volume": [100, 110, 90, 120, 130, 125, 140, 150],
        "Return": [0.0, 1 / 101, 1 / 102, 1 / 103, 1 / 104, 1 / 105, 1 / 106, 1 / 107],
    })


def test_relative_price_and_volume_features_are_finite_and_point_in_time():
    panel = _research_panel()
    groups = ["relative_price", "normalized_volume"]
    base = add_experiment_feature_groups(panel.iloc[:7], groups)
    with_future = panel.copy()
    with_future.loc[7, ["Open", "High", "Low", "Close", "Volume"]] = [
        10_000, 20_000, 1, 15_000, 1_000_000
    ]
    extended = add_experiment_feature_groups(with_future, groups).iloc[:7]
    columns = [
        "OpenToPrevClose", "HighLowToPrevClose", "CloseToSMA60",
        "LogVolume", "VolumeToMean20", "VolumeZScore20",
        "TradedValue", "TradedValueToMean20",
    ]
    np.testing.assert_allclose(base[columns], extended[columns])
    assert np.isfinite(base[columns].to_numpy()).all()
    assert base["OpenToPrevClose"].iloc[1] == 0.0
    assert base["TradedValue"].iloc[0] == 10_100


def test_momentum_volatility_and_liquidity_groups_do_not_use_future_rows():
    panel = _research_panel()
    groups = ["momentum_reversal", "volatility_range", "liquidity_dynamics"]
    base = add_experiment_feature_groups(panel.iloc[:7], groups)
    changed = panel.copy()
    changed.loc[7, ["Close", "Volume"]] = [50_000, 2_000_000]
    extended = add_experiment_feature_groups(changed, groups).iloc[:7]
    columns = [
        "Return5d", "MomentumSlope5_20", "ExcessReturn5d",
        "RealizedVol5d", "DownsideVol20d", "HighLowRange5d",
        "ADV5ToADV20", "AmihudProxy20", "VolumePriceDivergence20",
    ]
    np.testing.assert_allclose(base[columns], extended[columns])
    assert np.isfinite(base[columns].to_numpy()).all()


def test_sector_prediction_transforms_are_same_day_and_sector_only():
    predictions = pd.DataFrame({
        "Date": pd.to_datetime(["2021-01-04"] * 4 + ["2021-01-05"] * 4),
        "SecuritiesCode": [1, 2, 3, 4] * 2,
        "Prediction": [1.0, 3.0, 10.0, 14.0, 100.0, 300.0, 20.0, 40.0],
        "Target": [0.0] * 8,
    })
    sectors = pd.DataFrame({
        "SecuritiesCode": [1, 2, 3, 4],
        "Sector": ["A", "A", "B", "B"],
    })
    demeaned = apply_sector_prediction_transform(
        predictions, sectors, "sector_demean"
    )
    assert demeaned["Prediction"].tolist() == [
        -1.0, 1.0, -2.0, 2.0, -100.0, 100.0, -10.0, 10.0
    ]
    ranked = apply_sector_prediction_transform(predictions, sectors, "sector_rank")
    assert ranked["Prediction"].tolist() == [0.5, 1.0, 0.5, 1.0] * 2


def test_sector_exposure_preserves_portfolio_net_accounting():
    positions = pd.DataFrame({
        "Date": pd.to_datetime(["2021-01-04"] * 4),
        "SecuritiesCode": [1, 2, 3, 4],
        "Weight": [0.3, -0.1, 0.2, -0.4],
    })
    sectors = pd.DataFrame({
        "SecuritiesCode": [1, 2, 3, 4],
        "Sector": ["A", "A", "B", "B"],
    })
    exposure = sector_exposure_daily(positions, sectors)
    weights = exposure.set_index("Sector")["NetSectorWeight"]
    assert np.isclose(weights["A"], 0.2)
    assert np.isclose(weights["B"], -0.2)
    assert np.isclose(exposure["GrossNetSectorExposure"].iloc[0], 0.4)
    assert np.isclose(exposure["MaxAbsoluteSectorExposure"].iloc[0], 0.2)


def test_average_predictions_preserves_keys_targets_and_means_scores():
    left = pd.DataFrame({
        "Date": pd.to_datetime(["2021-01-04", "2021-01-04"]),
        "SecuritiesCode": [1, 2],
        "Prediction": [1.0, 3.0],
        "Target": [0.1, 0.2],
    })
    right = left.copy()
    right["Prediction"] = [3.0, 5.0]
    averaged = _average_predictions([left, right])
    assert averaged["Prediction"].tolist() == [2.0, 4.0]
    assert averaged["Target"].tolist() == [0.1, 0.2]
