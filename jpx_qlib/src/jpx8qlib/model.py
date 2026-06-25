from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import FEATURE_COLUMNS, LABEL_COLUMN


@dataclass
class PublishedLGBM:
    params: dict
    feature_columns: list[str] | None = None
    categorical_features: list[str] | None = None

    def __post_init__(self) -> None:
        self.feature_columns = list(self.feature_columns or FEATURE_COLUMNS)
        configured = self.categorical_features or ["SecuritiesCode", "SupervisionFlag"]
        self.categorical_features = [
            column for column in configured if column in self.feature_columns
        ]
        self.estimator = lgb.LGBMRegressor(**self.params)
        self._fitted = False

    def _prepare_x(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.feature_columns].copy()
        for column in self.categorical_features:
            x[column] = x[column].astype("category")
        return x

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "PublishedLGBM":
        x_train = self._prepare_x(train)
        y_train = train[LABEL_COLUMN].astype(float)
        fit_kwargs = {
            "categorical_feature": self.categorical_features,
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


@dataclass
class RidgeModel:
    params: dict
    feature_columns: list[str]

    def __post_init__(self) -> None:
        ridge_params = {"alpha": 1.0, **self.params}
        self.estimator = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(**ridge_params)),
        ])
        self._fitted = False

    def _prepare_x(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[self.feature_columns].copy()
        for column in values.columns:
            if pd.api.types.is_bool_dtype(values[column]):
                values[column] = values[column].astype(float)
            elif not pd.api.types.is_numeric_dtype(values[column]):
                mapped = values[column].map({
                    True: 1.0, False: 0.0, "True": 1.0, "False": 0.0,
                })
                if mapped.isna().any():
                    raise ValueError(
                        f"Ridge feature {column} is non-numeric and cannot be mapped"
                    )
                values[column] = mapped
        return values.astype(float)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "RidgeModel":
        self.estimator.fit(
            self._prepare_x(train),
            train[LABEL_COLUMN].astype(float),
        )
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Model is not fitted")
        values = np.asarray(self.estimator.predict(self._prepare_x(frame)), dtype=float)
        return pd.Series(values, index=frame.index, name="Prediction")


def make_model(
    model_type: str,
    params: dict,
    feature_columns: list[str],
    categorical_features: list[str],
):
    if model_type == "lightgbm":
        return PublishedLGBM(
            params,
            feature_columns=feature_columns,
            categorical_features=categorical_features,
        )
    if model_type == "ridge":
        return RidgeModel(params, feature_columns=feature_columns)
    raise ValueError(f"Unsupported model type: {model_type}")
