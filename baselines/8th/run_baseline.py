"""Run the minimally adapted JPX 8th-place LightGBM baseline locally."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgbm
import numpy as np
import pandas as pd

import Features
from Trackers import StateTracker


BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parents[1]
UPSTREAM_DIR = REPO_ROOT / "JPXTokyoStockExchangePrediction" / "winner-models" / "8th"
EXPECTED_TRAIN_ROWS = 2_332_267
BASE_COLUMNS = [
    "SecuritiesCode", "Open", "High", "Low", "Close", "Volume",
    "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag",
]
CATEGORICAL_COLUMNS = ["SecuritiesCode", "SupervisionFlag"]


def make_features() -> list[Any]:
    return [
        Features.Amplitude(), Features.OpenCloseReturn(), Features.Return(),
        Features.Volatility(10), Features.Volatility(30), Features.Volatility(50),
        Features.SMA("Close", 3), Features.SMA("Close", 5),
        Features.SMA("Close", 10), Features.SMA("Close", 30),
        Features.SMA("Return", 3), Features.SMA("Return", 5),
        Features.SMA("Return", 10), Features.SMA("Return", 30),
    ]


def training_columns() -> list[str]:
    return BASE_COLUMNS + [feature.name for feature in make_features()]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def verify_upstream_unchanged(before: dict[str, str]) -> None:
    after = source_hashes(UPSTREAM_DIR)
    if after != before:
        raise RuntimeError("Upstream 8th-place source changed during the run")


def prepare_prices(path: Path) -> pd.DataFrame:
    print(f"Reading {path}", flush=True)
    prices = pd.read_csv(path)
    tracker = StateTracker(make_features())
    with warnings.catch_warnings():
        # The reference volatility code intentionally asks for sample standard
        # deviation on the first one-element window, then fills that NaN with 0.
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
        prepared = tracker.prepare_data_for_training(prices)
    prepared["Date"] = pd.to_datetime(prepared["Date"])
    return prepared


def load_reference_model(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def validate_model_features(model: Any, columns: list[str]) -> None:
    model_columns = list(getattr(model, "feature_name_", []))
    if not model_columns and hasattr(model, "booster_"):
        model_columns = list(model.booster_.feature_name())
    if model_columns != columns:
        raise RuntimeError(
            f"Model feature mismatch: expected {columns}, got {model_columns}"
        )
    if len(columns) != 23:
        raise RuntimeError(f"Expected 23 model features, got {len(columns)}")


def add_daily_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    ranked_days = []
    for _, day in frame.groupby("Date", sort=False):
        day = day.sort_values(by="Prediction", ascending=False).copy()
        day["Rank"] = np.arange(len(day), dtype=np.int64)
        ranked_days.append(day.sort_values(by="SecuritiesCode", ascending=True))
    ranked = pd.concat(ranked_days, ignore_index=True)
    for date, day in ranked.groupby("Date", sort=False):
        expected = np.arange(len(day), dtype=np.int64)
        actual = np.sort(day["Rank"].to_numpy())
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"Invalid ranks for {date.date()}")
    return ranked


def spread_return_per_day(day: pd.DataFrame, portfolio_size: int = 200) -> float:
    if len(day) < portfolio_size * 2:
        raise ValueError(f"Need at least {portfolio_size * 2} rows, got {len(day)}")
    ordered = day.sort_values("Rank")
    weights = np.linspace(2, 1, portfolio_size)
    purchase = (ordered["Target"].iloc[:portfolio_size].to_numpy() * weights).sum()
    purchase /= weights.mean()
    short = (ordered["Target"].iloc[-portfolio_size:].to_numpy() * weights[::-1]).sum()
    short /= weights.mean()
    return float(purchase - short)


def score_predictions(ranked: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = [
        {"Date": date, "SpreadReturn": spread_return_per_day(day)}
        for date, day in ranked.groupby("Date", sort=False)
    ]
    daily = pd.DataFrame(rows)
    score = float(daily["SpreadReturn"].mean() / daily["SpreadReturn"].std())
    if not math.isfinite(score):
        raise RuntimeError(f"Supplemental score is not finite: {score}")
    return daily, score


def model_predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    # Pickles produced by the older LightGBM sklearn wrapper have
    # `_n_classes=None`; LightGBM 4.6 fails before delegating to the unchanged
    # Booster. Calling that same Booster is a serialization compatibility shim.
    if getattr(model, "_n_classes", 1) is None and hasattr(model, "booster_"):
        return model.booster_.predict(features)
    return model.predict(features)


def predict(model: Any, prepared: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = prepared[["Date", "SecuritiesCode", "Target"]].copy()
    output["Prediction"] = model_predict(model, prepared[columns])
    return add_daily_ranks(output)


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_safe),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "jpx")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "baseline_8th")
    parser.add_argument("--model-source", choices=["reference", "retrain"], default="reference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_files" / "stock_prices.csv"
    supplemental_path = data_dir / "supplemental_files" / "stock_prices.csv"
    reference_model_path = BASELINE_DIR / "lgbm.pickle"
    for required in [train_path, supplemental_path, reference_model_path]:
        if not required.is_file():
            raise FileNotFoundError(required)

    upstream_before = source_hashes(UPSTREAM_DIR)
    columns = training_columns()
    reference_model = load_reference_model(reference_model_path)
    validate_model_features(reference_model, columns)

    supplemental = prepare_prices(supplemental_path)
    reference_predictions = predict(reference_model, supplemental, columns)
    reference_daily, reference_score = score_predictions(reference_predictions)

    metrics: dict[str, Any] = {
        "model_source": args.model_source,
        "reference_score": reference_score,
        "supplemental_rows": len(supplemental),
        "supplemental_dates": supplemental["Date"].nunique(),
    }
    selected_model = reference_model
    selected_predictions = reference_predictions
    selected_daily = reference_daily

    if args.model_source == "retrain":
        train = prepare_prices(train_path)
        if len(train) != EXPECTED_TRAIN_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_TRAIN_ROWS} preprocessed training rows, got {len(train)}"
            )
        print("Training original default LGBMRegressor...", flush=True)
        selected_model = lgbm.LGBMRegressor(**{})
        selected_model.fit(
            train[columns], train[["Target"]],
            categorical_feature=CATEGORICAL_COLUMNS, eval_metric="rmse",
        )
        validate_model_features(selected_model, columns)
        selected_predictions = predict(selected_model, supplemental, columns)
        selected_daily, retrained_score = score_predictions(selected_predictions)
        reference_vector = reference_predictions["Prediction"].to_numpy()
        retrained_vector = selected_predictions["Prediction"].to_numpy()
        metrics.update({
            "train_rows": len(train),
            "retrained_score": retrained_score,
            "score_difference_vs_reference": retrained_score - reference_score,
            "prediction_pearson_vs_reference": float(np.corrcoef(reference_vector, retrained_vector)[0, 1]),
            "prediction_spearman_vs_reference": float(
                pd.Series(reference_vector).corr(pd.Series(retrained_vector), method="spearman")
            ),
        })

    predictions_path = output_dir / "predictions.parquet"
    daily_path = output_dir / "daily_spread.csv"
    model_path = output_dir / "model.pickle"
    selected_predictions.to_parquet(predictions_path, index=False)
    selected_daily.to_csv(daily_path, index=False)
    if args.model_source == "reference":
        # Preserve the supplied model byte-for-byte instead of re-serializing it
        # through the current LightGBM version.
        shutil.copy2(reference_model_path, model_path)
    else:
        with model_path.open("wb") as handle:
            pickle.dump(selected_model, handle)

    input_hashes = {
        str(train_path): sha256_file(train_path),
        str(supplemental_path): sha256_file(supplemental_path),
        str(reference_model_path): sha256_file(reference_model_path),
    }
    metrics["selected_score"] = float(
        metrics.get("retrained_score", metrics["reference_score"])
    )
    write_json(output_dir / "metrics.json", metrics)
    shutil.copy2(BASELINE_DIR / "COMPATIBILITY.md", output_dir / "COMPATIBILITY.md")
    verify_upstream_unchanged(upstream_before)

    manifest = {
        "baseline": "JPX competition 8th-place minimally adapted local runner",
        "model_source": args.model_source,
        "command": " ".join(sys.argv),
        "data_dir": data_dir,
        "output_dir": output_dir,
        "features": columns,
        "categorical_features": CATEGORICAL_COLUMNS,
        "input_sha256": input_hashes,
        "upstream_sha256_before_and_after": upstream_before,
        "output_sha256": {
            "predictions.parquet": sha256_file(predictions_path),
            "daily_spread.csv": sha256_file(daily_path),
            "model.pickle": sha256_file(model_path),
        },
        "versions": package_versions(),
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"Artifacts written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
