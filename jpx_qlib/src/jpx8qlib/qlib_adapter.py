from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import FEATURE_COLUMNS, LABEL_COLUMN
from .data import to_qlib_frame
from .model import make_model


def require_qlib() -> dict[str, Any]:
    try:
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
    except ImportError as exc:
        raise RuntimeError(
            "Qlib is not installed. Run: pip install -e '.[qlib]'"
        ) from exc
    return {"DatasetH": DatasetH, "DataHandlerLP": DataHandlerLP}


def make_dataset(
    panel: pd.DataFrame,
    split_cfg: dict,
    feature_columns: list[str] | None = None,
):
    q = require_qlib()
    qframe = to_qlib_frame(panel, feature_columns=feature_columns)
    handler = q["DataHandlerLP"].from_df(qframe)
    segments = {
        "train": (split_cfg["train_start"], split_cfg["train_end"]),
        "valid": (split_cfg["valid_start"], split_cfg["valid_end"]),
        "test": (split_cfg["test_start"], split_cfg["test_end"]),
    }
    return q["DatasetH"](handler=handler, segments=segments)


def _flat_from_qlib(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        features = data["feature"].copy()
        if "label" in data.columns.get_level_values(0):
            labels = data["label"].copy()
            features[LABEL_COLUMN] = labels.iloc[:, 0]
    else:
        features = data.copy()
    idx = features.index.to_frame(index=False)
    features = features.reset_index(drop=True)
    features["Date"] = pd.to_datetime(idx["datetime"])
    instrument = idx["instrument"].astype(str).str.replace(r"^JP", "", regex=True)
    features["SecuritiesCode"] = pd.to_numeric(instrument, errors="raise").astype(int)
    return features


class QlibPublishedModel:
    """Qlib-compatible wrapper around the published LightGBM baseline."""

    def __init__(
        self,
        params: dict | None = None,
        *,
        model_type: str = "lightgbm",
        feature_columns: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ):
        self.params = params or {}
        self.model_type = model_type
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.categorical_features = list(
            categorical_features or ["SecuritiesCode", "SupervisionFlag"]
        )
        self.model = make_model(
            self.model_type,
            self.params,
            self.feature_columns,
            self.categorical_features,
        )

    def fit(self, dataset, **kwargs):
        q = require_qlib()
        train = dataset.prepare("train", col_set=["feature", "label"], data_key=q["DataHandlerLP"].DK_L)
        valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=q["DataHandlerLP"].DK_L)
        self.model.fit(_flat_from_qlib(train), _flat_from_qlib(valid))
        return self

    def predict(self, dataset, segment: str = "test", **kwargs) -> pd.Series:
        q = require_qlib()
        data = dataset.prepare(segment, col_set="feature", data_key=q["DataHandlerLP"].DK_I)
        flat = _flat_from_qlib(data)
        prediction = self.model.predict(flat)
        prediction.index = data.index
        prediction.name = "score"
        return prediction
