from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, columns: list[str] | None = None) -> dict:
    columns = list(columns or FEATURE_COLUMNS)
    key = ["Date", "SecuritiesCode"]
    value_columns = [column for column in columns if column not in key]
    selected_columns = key + value_columns
    missing_left = [column for column in selected_columns if column not in left.columns]
    missing_right = [column for column in selected_columns if column not in right.columns]
    if missing_left or missing_right:
        raise ValueError(
            "Cannot compare frames with missing columns: "
            f"left={missing_left}, right={missing_right}"
        )

    a = left[selected_columns].sort_values(key, kind="mergesort").reset_index(drop=True)
    b = right[selected_columns].sort_values(key, kind="mergesort").reset_index(drop=True)
    result = {
        "left_rows": len(a),
        "right_rows": len(b),
        "keys_equal": a[key].equals(b[key]),
        "columns": {},
    }
    if len(a) != len(b) or not result["keys_equal"]:
        result["all_close"] = False
        return result

    all_close = True
    for col in value_columns:
        av = pd.to_numeric(a[col], errors="coerce").to_numpy(float)
        bv = pd.to_numeric(b[col], errors="coerce").to_numpy(float)
        close = np.isclose(av, bv, rtol=1e-8, atol=1e-10, equal_nan=True)
        finite = np.isfinite(av) & np.isfinite(bv)
        max_abs = float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.0
        info = {"matching_fraction": float(close.mean()), "max_abs_difference": max_abs}
        result["columns"][col] = info
        all_close &= bool(close.all())
    result["all_close"] = all_close
    return result


def compare_predictions(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    key = ["Date", "SecuritiesCode"]
    merged = left[key + ["Prediction"]].merge(
        right[key + ["Prediction"]], on=key, how="outer", suffixes=("_left", "_right"), indicator=True
    )
    both = merged[merged["_merge"] == "both"].dropna()
    corr = float(both["Prediction_left"].corr(both["Prediction_right"])) if len(both) > 1 else float("nan")
    return {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "matched_rows": int(len(both)),
        "prediction_correlation": corr,
        "max_abs_difference": float((both["Prediction_left"] - both["Prediction_right"]).abs().max()) if len(both) else float("nan"),
    }


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
