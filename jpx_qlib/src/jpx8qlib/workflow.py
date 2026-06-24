from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import Config
from .data import prepare_panel
from .model import PublishedLGBM
from .qlib_adapter import QlibPublishedModel, make_dataset
from .ranking import add_rank
from .scoring import daily_spread_return, score_summary
from .split import split_panel


def _evaluate_predictions(frame: pd.DataFrame, config: Config, prefix: str) -> dict:
    ranking_cfg = config.raw["ranking"]
    ranked = add_rank(frame, mode=ranking_cfg["mode"])
    daily = daily_spread_return(
        ranked,
        top_n=int(ranking_cfg["top_n"]),
        weight_first=float(ranking_cfg["weight_first"]),
        weight_last=float(ranking_cfg["weight_last"]),
    )
    metrics = score_summary(
        daily,
        annualization_days=int(config.raw["evaluation"].get("annualization_days", 252)),
    )
    frame.to_pickle(config.output_dir / f"{prefix}_predictions.pkl.gz", compression="gzip")
    ranked.to_pickle(config.output_dir / f"{prefix}_ranked.pkl.gz", compression="gzip")
    daily.to_csv(config.output_dir / f"{prefix}_daily_spread.csv", header=True)
    (config.output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    return metrics


def run_native(config: Config, force_prepare: bool = False) -> dict:
    panel = prepare_panel(config, force=force_prepare)
    segments = split_panel(panel, config.raw["split"])
    model = PublishedLGBM(config.raw["model"]["params"])
    model.fit(segments["train"], segments["valid"])
    joblib.dump(model, config.output_dir / "native_model.joblib")

    test = segments["test"].copy()
    test["Prediction"] = model.predict(test).to_numpy()
    return _evaluate_predictions(test, config, "native")


def run_qlib(config: Config, force_prepare: bool = False) -> dict:
    panel = prepare_panel(config, force=force_prepare)
    dataset = make_dataset(panel, config.raw["split"])
    model = QlibPublishedModel(config.raw["model"]["params"])
    model.fit(dataset)
    prediction = model.predict(dataset, "test")

    test = split_panel(panel, config.raw["split"])["test"].copy()
    pred_flat = prediction.rename("Prediction").reset_index()
    pred_flat["Date"] = pd.to_datetime(pred_flat["datetime"])
    pred_flat["SecuritiesCode"] = pred_flat["instrument"].astype(str).str.replace(r"^JP", "", regex=True).astype(int)
    test = test.merge(pred_flat[["Date", "SecuritiesCode", "Prediction"]], on=["Date", "SecuritiesCode"], how="inner")
    joblib.dump(model, config.output_dir / "qlib_model.joblib")
    return _evaluate_predictions(test, config, "qlib")
