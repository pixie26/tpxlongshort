from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

from .config import Config
from .data import prepare_panel
from .model import PublishedLGBM
from .parity import compare_predictions, write_report
from .qlib_adapter import QlibPublishedModel, make_dataset
from .ranking import add_rank
from .scoring import daily_spread_return, score_summary
from .split import split_panel


NATIVE_REFERENCE_FILES = (
    "native_metrics.json",
    "native_predictions.pkl.gz",
    "native_ranked.pkl.gz",
    "native_daily_spread.csv",
    "native_in_sample_metrics.json",
    "feature_parity.json",
    "data_manifest.json",
)

QLIB_REFERENCE_FILES = (
    "qlib_model.joblib",
    "qlib_predictions.pkl.gz",
    "qlib_ranked.pkl.gz",
    "qlib_daily_spread.csv",
    "qlib_metrics.json",
    "prediction_parity.json",
    "environment_installed.txt",
)


def _score_predictions(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.Series, dict]:
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
    return ranked, daily, metrics


def _evaluate_predictions(
    frame: pd.DataFrame,
    config: Config,
    prefix: str,
    metadata: dict | None = None,
) -> dict:
    ranked, daily, metrics = _score_predictions(frame, config)
    if metadata:
        metrics.update(metadata)
    frame.to_pickle(config.output_dir / f"{prefix}_predictions.pkl.gz", compression="gzip")
    ranked.to_pickle(config.output_dir / f"{prefix}_ranked.pkl.gz", compression="gzip")
    daily.to_csv(config.output_dir / f"{prefix}_daily_spread.csv", header=True)
    (config.output_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8"
    )
    return metrics


def _freeze_native_reference(config: Config, force: bool = False) -> Path:
    reference_dir = config.output_dir / "native_reference"
    if reference_dir.exists() and not force:
        raise FileExistsError(
            f"Native reference already exists: {reference_dir}. "
            "Use --force-reference to replace its named artifacts."
        )

    missing = [
        name for name in NATIVE_REFERENCE_FILES
        if not (config.output_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot freeze native reference; missing artifacts: " + ", ".join(missing)
        )

    reference_dir.mkdir(parents=True, exist_ok=True)
    for name in NATIVE_REFERENCE_FILES:
        shutil.copy2(config.output_dir / name, reference_dir / name)
    shutil.copy2(config.source_path, reference_dir / "baseline.yaml")
    return reference_dir


def _write_installed_environment(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    path.write_text(result.stdout, encoding="utf-8")


def _freeze_qlib_reference(config: Config, force: bool = False) -> Path:
    reference_dir = config.output_dir / "qlib_reference"
    if reference_dir.exists() and not force:
        raise FileExistsError(
            f"Qlib reference already exists: {reference_dir}. "
            "Use --force-reference to replace its named artifacts."
        )

    missing = [
        name for name in QLIB_REFERENCE_FILES
        if not (config.output_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot freeze Qlib reference; missing artifacts: " + ", ".join(missing)
        )

    reference_dir.mkdir(parents=True, exist_ok=True)
    for name in QLIB_REFERENCE_FILES:
        shutil.copy2(config.output_dir / name, reference_dir / name)
    shutil.copy2(config.source_path, reference_dir / "baseline.yaml")
    return reference_dir


def run_native(
    config: Config,
    force_prepare: bool = False,
    freeze_reference: bool = False,
    force_reference: bool = False,
) -> dict:
    reference_dir = config.output_dir / "native_reference"
    if freeze_reference and reference_dir.exists() and not force_reference:
        raise FileExistsError(
            f"Native reference already exists: {reference_dir}. "
            "Use --force-reference to replace its named artifacts."
        )

    panel = prepare_panel(config, force=force_prepare)
    segments = split_panel(panel, config.raw["split"])
    model = PublishedLGBM(config.raw["model"]["params"])
    model.fit(segments["train"], segments["valid"])
    joblib.dump(model, config.output_dir / "native_model.joblib")

    test = segments["test"].copy()
    test["Prediction"] = model.predict(test).to_numpy()
    metrics = _evaluate_predictions(
        test,
        config,
        "native",
        metadata={
            "run_role": "native_reference",
            "evaluation_scope": "chronological_oos",
            "prediction_scope": "out_of_sample",
            "scoring_schema_version": 2,
            "test_target_used_for_training": False,
            "test_target_used_for_evaluation": True,
        },
    )

    train = segments["train"].copy()
    train["Prediction"] = model.predict(train).to_numpy()
    _, _, in_sample_metrics = _score_predictions(train, config)
    in_sample_metrics.update({
        "run_role": "native_reference",
        "evaluation_scope": "in_sample",
        "prediction_scope": "in_sample",
        "scoring_schema_version": 2,
        "target_used_for_training": True,
        "target_used_for_evaluation": True,
        "not_valid_for_strategy_assessment": True,
    })
    (config.output_dir / "native_in_sample_metrics.json").write_text(
        json.dumps(in_sample_metrics, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    if freeze_reference:
        frozen_dir = _freeze_native_reference(config, force=force_reference)
        metrics["native_reference_dir"] = str(frozen_dir)
    return metrics


def run_qlib(
    config: Config,
    force_prepare: bool = False,
    freeze_reference: bool = False,
    force_reference: bool = False,
) -> dict:
    reference_dir = config.output_dir / "qlib_reference"
    if freeze_reference and reference_dir.exists() and not force_reference:
        raise FileExistsError(
            f"Qlib reference already exists: {reference_dir}. "
            "Use --force-reference to replace its named artifacts."
        )

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
    metrics = _evaluate_predictions(
        test,
        config,
        "qlib",
        metadata={
            "run_role": "qlib_parity_reference",
            "evaluation_scope": "chronological_oos",
            "prediction_scope": "out_of_sample",
            "scoring_schema_version": 2,
            "test_target_used_for_training": False,
            "test_target_used_for_evaluation": True,
        },
    )

    if freeze_reference:
        native_path = config.output_dir / "native_predictions.pkl.gz"
        if not native_path.is_file():
            raise FileNotFoundError(
                "Cannot freeze Qlib parity reference without native_predictions.pkl.gz"
            )
        parity = compare_predictions(pd.read_pickle(native_path), test)
        write_report(parity, config.output_dir / "prediction_parity.json")
        _write_installed_environment(config.output_dir / "environment_installed.txt")
        frozen_dir = _freeze_qlib_reference(config, force=force_reference)
        metrics["qlib_reference_dir"] = str(frozen_dir)
        metrics["prediction_parity"] = parity
    return metrics
