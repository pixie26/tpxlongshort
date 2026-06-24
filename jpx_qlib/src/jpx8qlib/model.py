from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS, LABEL_COLUMN


@dataclass
class PublishedLGBM:
    params: dict

    def __post_init__(self) -> None:
        self.estimator = lgb.LGBMRegressor(**self.params)
        self._fitted = False

    @staticmethod
    def _prepare_x(frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[FEATURE_COLUMNS].copy()
        x["SecuritiesCode"] = x["SecuritiesCode"].astype("category")
        x["SupervisionFlag"] = x["SupervisionFlag"].astype("category")
        return x

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "PublishedLGBM":
        x_train = self._prepare_x(train)
        y_train = train[LABEL_COLUMN].astype(float)
        fit_kwargs = {
            "categorical_feature": ["SecuritiesCode", "SupervisionFlag"],
            "callbacks": [lgb.log_evaluation(period=0)],
        }
        if valid is not None:
            fit_kwargs["eval_set"] = [(self._prepare_x(valid), valid[LABEL_COLUMN].astype(float))]
            fit_kwargs["eval_metric"] = "rmse"
        self.estimator.fit(x_train, y_train, **fit_kwargs)
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        values = np.asarray(self.estimator.predict(self._prepare_x(frame)), dtype=float)
        return pd.Series(values, index=frame.index, name="Prediction")
