from __future__ import annotations

import numpy as np
import pandas as pd

from jpx8qlib.config import Config
from jpx8qlib.data import apply_experiment_features
from jpx8qlib.model import RidgeModel


def test_nested_config_resolves_project_root_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    source = tmp_path / "configs" / "ablations" / "a1.yaml"
    source.parent.mkdir(parents=True)
    config = Config(raw={"project": {"output_dir": "outputs/a1"}}, source_path=source)
    assert config.project_root == tmp_path
    assert config.output_dir == tmp_path / "outputs" / "a1"


def test_ridge_standardization_is_fit_on_training_rows_only():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0], "Target": [0.0, 1.0, 2.0]})
    future = pd.DataFrame({"x": [1000.0]})
    model = RidgeModel({"alpha": 1.0}, ["x"]).fit(train)
    center = model.estimator.named_steps["scale"].mean_[0]
    assert center == 1.0
    assert np.isfinite(model.predict(future).iloc[0])


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
