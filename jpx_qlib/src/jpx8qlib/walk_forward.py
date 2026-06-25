from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .data import prepare_experiment_panel
from .model import make_model
from .parity import compare_predictions, write_report
from .qlib_adapter import QlibPublishedModel, make_dataset
from .ranking import add_rank
from .scoring import daily_spread_return, score_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    purge_days: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    nominal_train_end: pd.Timestamp
    nominal_valid_end: pd.Timestamp
    purged_after_train: tuple[pd.Timestamp, ...]
    purged_after_valid: tuple[pd.Timestamp, ...]

    def split_config(self) -> dict[str, str]:
        return {
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "valid_start": str(self.valid_start.date()),
            "valid_end": str(self.valid_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
        }

    def metadata(self) -> dict:
        return {
            "fold": self.name,
            **self.split_config(),
            "nominal_train_end": str(self.nominal_train_end.date()),
            "nominal_valid_end": str(self.nominal_valid_end.date()),
            "purge_days": self.purge_days,
            "purged_after_train": [str(value.date()) for value in self.purged_after_train],
            "purged_after_valid": [str(value.date()) for value in self.purged_after_valid],
        }


def _period_dates(
    trading_dates: pd.DatetimeIndex,
    start: str,
    end: str,
    label: str,
) -> pd.DatetimeIndex:
    values = trading_dates[
        (trading_dates >= pd.Timestamp(start)) & (trading_dates <= pd.Timestamp(end))
    ]
    if values.empty:
        raise ValueError(f"{label} has no trading dates between {start} and {end}")
    return values


def build_walk_forward_folds(panel: pd.DataFrame, config: Config) -> list[WalkForwardFold]:
    options = config.raw.get("walk_forward")
    if not isinstance(options, dict):
        raise ValueError("walk_forward config must be a mapping")
    purge_days = int(options.get("purge_days", 2))
    if purge_days < 0:
        raise ValueError("walk_forward.purge_days must be non-negative")
    specs = options.get("folds")
    if not isinstance(specs, list) or not specs:
        raise ValueError("walk_forward.folds must be a non-empty list")

    trading_dates = pd.DatetimeIndex(
        pd.to_datetime(panel["Date"], errors="raise").drop_duplicates().sort_values()
    )
    folds: list[WalkForwardFold] = []
    seen: set[str] = set()
    for spec in specs:
        name = str(spec["name"])
        if name in seen:
            raise ValueError(f"Duplicate walk-forward fold name: {name}")
        seen.add(name)
        train_dates = _period_dates(trading_dates, *spec["train"], f"{name}.train")
        valid_dates = _period_dates(trading_dates, *spec["valid"], f"{name}.valid")
        test_dates = _period_dates(trading_dates, *spec["test"], f"{name}.test")
        if purge_days and (
            len(train_dates) <= purge_days or len(valid_dates) <= purge_days
        ):
            raise ValueError(f"{name} is too short for purge_days={purge_days}")

        purged_train = tuple(train_dates[-purge_days:]) if purge_days else ()
        purged_valid = tuple(valid_dates[-purge_days:]) if purge_days else ()
        effective_train = train_dates[:-purge_days] if purge_days else train_dates
        effective_valid = valid_dates[:-purge_days] if purge_days else valid_dates
        if not (
            effective_train[-1] < valid_dates[0]
            and effective_valid[-1] < test_dates[0]
        ):
            raise ValueError(f"{name} segments are not strictly chronological")

        folds.append(WalkForwardFold(
            name=name,
            purge_days=purge_days,
            train_start=effective_train[0],
            train_end=effective_train[-1],
            valid_start=valid_dates[0],
            valid_end=effective_valid[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
            nominal_train_end=train_dates[-1],
            nominal_valid_end=valid_dates[-1],
            purged_after_train=purged_train,
            purged_after_valid=purged_valid,
        ))
    return folds


def _slice(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(panel["Date"])
    return panel.loc[dates.between(start, end, inclusive="both")].copy()


def _predict_native(model, frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["Prediction"] = model.predict(output).to_numpy()
    return output


def _prediction_from_qlib(
    model: QlibPublishedModel,
    dataset,
    segment: str,
    source: pd.DataFrame,
) -> pd.DataFrame:
    prediction = model.predict(dataset, segment)
    pred_flat = prediction.rename("Prediction").reset_index()
    pred_flat["Date"] = pd.to_datetime(pred_flat["datetime"])
    pred_flat["SecuritiesCode"] = (
        pred_flat["instrument"].astype(str).str.replace(r"^JP", "", regex=True).astype(int)
    )
    return source.merge(
        pred_flat[["Date", "SecuritiesCode", "Prediction"]],
        on=["Date", "SecuritiesCode"],
        how="inner",
        validate="one_to_one",
    )


def _rank_ic(frame: pd.DataFrame) -> tuple[float, float]:
    values = frame[["Date", "Prediction", "Target"]].dropna().copy()
    values["prediction_rank"] = values.groupby("Date")["Prediction"].rank(method="average")
    values["target_rank"] = values.groupby("Date")["Target"].rank(method="average")
    values["prediction_centered"] = (
        values["prediction_rank"]
        - values.groupby("Date")["prediction_rank"].transform("mean")
    )
    values["target_centered"] = (
        values["target_rank"]
        - values.groupby("Date")["target_rank"].transform("mean")
    )
    numerator = (
        values["prediction_centered"] * values["target_centered"]
    ).groupby(values["Date"]).sum()
    denominator = np.sqrt(
        values["prediction_centered"].pow(2).groupby(values["Date"]).sum()
        * values["target_centered"].pow(2).groupby(values["Date"]).sum()
    )
    daily = (numerator / denominator).replace([np.inf, -np.inf], np.nan)
    return float(daily.mean()), float(daily.median())


def _evaluate(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.Series, dict]:
    ranking = config.raw["ranking"]
    ranked = add_rank(frame, mode=ranking["mode"])
    daily = daily_spread_return(
        ranked,
        top_n=int(ranking["top_n"]),
        weight_first=float(ranking["weight_first"]),
        weight_last=float(ranking["weight_last"]),
    )
    metrics = score_summary(
        daily,
        annualization_days=int(config.raw["evaluation"].get("annualization_days", 252)),
    )
    rank_ic_mean, rank_ic_median = _rank_ic(frame)
    metrics.update({
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_median": rank_ic_median,
        "top_bottom_spread_mean": float(daily.mean()),
    })
    return ranked, daily, metrics


def _segment_metrics(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    config: Config,
    fold: WalkForwardFold,
    role: str,
    best_iteration: int,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    _, _, train_metrics = _evaluate(train, config)
    _, _, valid_metrics = _evaluate(valid, config)
    ranked, daily, test_metrics = _evaluate(test, config)
    metrics = {
        **fold.metadata(),
        "run_role": role,
        "evaluation_scope": "expanding_walk_forward",
        "scoring_schema_version": 2,
        "best_iteration": int(best_iteration),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "in_sample_sharpe": train_metrics["annualized_sharpe"],
        "valid_sharpe": valid_metrics["annualized_sharpe"],
        "oos_sharpe": test_metrics["annualized_sharpe"],
        "oos_competition_sharpe": test_metrics["competition_sharpe"],
        "rank_ic": test_metrics["rank_ic_mean"],
        "rank_ic_median": test_metrics["rank_ic_median"],
        "top_bottom_spread": test_metrics["top_bottom_spread_mean"],
        "oos_days": test_metrics["days"],
        "oos_cumulative_spread": test_metrics["cumulative_spread_simple"],
    }
    return ranked, daily, metrics


def _metrics_equal(left: dict, right: dict, atol: float = 1e-12) -> bool:
    keys = (
        "best_iteration", "train_rows", "valid_rows", "test_rows", "in_sample_sharpe",
        "valid_sharpe", "oos_sharpe", "oos_competition_sharpe",
        "rank_ic", "rank_ic_median", "top_bottom_spread", "oos_days",
        "oos_cumulative_spread",
    )
    for key in keys:
        a, b = left[key], right[key]
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            if int(a) != int(b):
                return False
        elif not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol):
            return False
    return True


def _fold_parity(
    native: pd.DataFrame,
    qlib: pd.DataFrame,
    native_metrics: dict,
    qlib_metrics: dict,
    ranking_mode: str,
    prediction_tolerance: float = 0.0,
) -> dict:
    prediction = compare_predictions(native, qlib)
    key = ["Date", "SecuritiesCode"]
    native_ranked = add_rank(native, mode=ranking_mode)[key + ["Rank"]]
    qlib_ranked = add_rank(qlib, mode=ranking_mode)[key + ["Rank"]]
    rank_merge = native_ranked.merge(
        qlib_ranked,
        on=key,
        how="outer",
        suffixes=("_native", "_qlib"),
        indicator=True,
    )
    keys_identical = bool(
        len(rank_merge) == len(native) == len(qlib)
        and rank_merge["_merge"].eq("both").all()
    )
    ranks_identical = bool(
        keys_identical
        and rank_merge["Rank_native"].eq(rank_merge["Rank_qlib"]).all()
    )
    max_difference = float(prediction["max_abs_difference"])
    return {
        **prediction,
        "prediction_tolerance": float(prediction_tolerance),
        "prediction_exact": max_difference == 0.0,
        "prediction_within_tolerance": max_difference <= prediction_tolerance,
        "keys_identical": keys_identical,
        "daily_rank_identical": ranks_identical,
        "metrics_identical": _metrics_equal(native_metrics, qlib_metrics),
    }


def _max_drawdown_additive(daily: pd.Series) -> float:
    curve = daily.cumsum()
    return float((curve - curve.cummax()).min())


def _write_summary(
    config: Config,
    output_dir: Path,
    folds: list[WalkForwardFold],
    mode: str,
) -> dict:
    rows = []
    stitched_frames = []
    for fold in folds:
        fold_dir = output_dir / fold.name
        metrics_path = fold_dir / f"{mode}_metrics.json"
        rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        stitched_frames.append(pd.read_pickle(fold_dir / f"{mode}_predictions.pkl.gz"))

    stitched = (
        pd.concat(stitched_frames, ignore_index=True)
        .sort_values(["Date", "SecuritiesCode"], kind="mergesort")
        .reset_index(drop=True)
    )
    if stitched.duplicated(["Date", "SecuritiesCode"]).any():
        raise ValueError("Stitched OOS predictions contain duplicate keys")
    stitched.to_pickle(
        output_dir / f"{mode}_stitched_oos_predictions.pkl.gz",
        compression="gzip",
    )
    stitched.to_pickle(output_dir / "stitched_oos_predictions.pkl.gz", compression="gzip")
    _, daily, aggregate = _evaluate(stitched, config)
    daily.to_csv(output_dir / f"{mode}_stitched_oos_daily_spread.csv", header=True)
    daily.to_csv(output_dir / "stitched_oos_daily_spread.csv", header=True)

    fold_sharpes = pd.Series([row["oos_sharpe"] for row in rows], dtype=float)
    fold_contributions = pd.Series(
        [row["oos_cumulative_spread"] for row in rows],
        index=[row["fold"] for row in rows],
        dtype=float,
    )
    positive_contributions = fold_contributions.clip(lower=0)
    contribution_share = (
        float(positive_contributions.max() / positive_contributions.sum())
        if positive_contributions.sum() > 0 else math.nan
    )
    monthly = daily.resample("ME").sum()
    yearly = {}
    for year, year_daily in daily.groupby(daily.index.year):
        year_metrics = score_summary(
            year_daily,
            annualization_days=int(config.raw["evaluation"].get("annualization_days", 252)),
        )
        yearly[str(int(year))] = {
            "days": year_metrics["days"],
            "annualized_sharpe": year_metrics["annualized_sharpe"],
            "cumulative_spread": year_metrics["cumulative_spread_simple"],
        }
    summary = {
        "run_role": f"{mode}_walk_forward",
        "evaluation_scope": "stitched_chronological_oos",
        "fold_count": len(rows),
        "test_start": str(stitched["Date"].min().date()),
        "test_end": str(stitched["Date"].max().date()),
        "oos_rows": int(len(stitched)),
        "oos_days": int(stitched["Date"].nunique()),
        "aggregate_oos_sharpe": aggregate["annualized_sharpe"],
        "aggregate_competition_sharpe": aggregate["competition_sharpe"],
        "positive_folds": int((fold_sharpes > 0).sum()),
        "median_fold_sharpe": float(fold_sharpes.median()),
        "mean_rank_ic": aggregate["rank_ic_mean"],
        "median_daily_rank_ic": aggregate["rank_ic_median"],
        "monthly_win_rate": float((monthly > 0).mean()),
        "max_drawdown_additive_spread": _max_drawdown_additive(daily),
        "largest_positive_fold_contribution_share": contribution_share,
        "yearly": yearly,
        "folds": rows,
    }
    pd.DataFrame(rows).to_csv(
        output_dir / f"{mode}_walk_forward_summary.csv",
        index=False,
    )
    write_report(summary, output_dir / f"{mode}_walk_forward_summary.json")
    return summary


def _write_combined_summary(output_dir: Path, native: dict, qlib: dict) -> dict:
    combined_rows = []
    parity_reports = []
    for native_fold, qlib_fold in zip(native["folds"], qlib["folds"], strict=True):
        fold = native_fold["fold"]
        if fold != qlib_fold["fold"]:
            raise ValueError("Native/Qlib summary fold order differs")
        parity = json.loads(
            (output_dir / fold / "prediction_parity.json").read_text(encoding="utf-8")
        )
        parity_reports.append({"fold": fold, **parity})
        row = {
            "fold": fold,
            "train_start": native_fold["train_start"],
            "train_end": native_fold["train_end"],
            "valid_start": native_fold["valid_start"],
            "valid_end": native_fold["valid_end"],
            "test_start": native_fold["test_start"],
            "test_end": native_fold["test_end"],
            "purge_days": native_fold["purge_days"],
        }
        for prefix, metrics in (("native", native_fold), ("qlib", qlib_fold)):
            for key in (
                "best_iteration", "train_rows", "valid_rows", "test_rows", "in_sample_sharpe",
                "valid_sharpe", "oos_sharpe", "rank_ic", "top_bottom_spread",
                "oos_cumulative_spread",
            ):
                row[f"{prefix}_{key}"] = metrics[key]
        row.update({
            "prediction_max_abs_difference": parity["max_abs_difference"],
            "prediction_correlation": parity["prediction_correlation"],
            "keys_identical": parity["keys_identical"],
            "daily_rank_identical": parity["daily_rank_identical"],
            "metrics_identical": parity["metrics_identical"],
        })
        combined_rows.append(row)

    positive_rank_ic_folds = sum(
        float(fold["rank_ic"]) > 0 for fold in native["folds"]
    )
    acceptance = {
        "aggregate_oos_sharpe_positive": native["aggregate_oos_sharpe"] > 0,
        "median_fold_sharpe_positive": native["median_fold_sharpe"] > 0,
        "at_least_three_positive_folds": native["positive_folds"] >= 3,
        "largest_positive_fold_share_below_80_percent": (
            math.isfinite(native["largest_positive_fold_contribution_share"])
            and native["largest_positive_fold_contribution_share"] < 0.8
        ),
        "rank_ic_positive_in_at_least_three_folds": positive_rank_ic_folds >= 3,
    }
    acceptance["passes_basic_diagnostic_bar"] = all(acceptance.values())
    engineering = {
        "all_fold_keys_identical": all(row["keys_identical"] for row in parity_reports),
        "all_fold_predictions_exact": all(
            row["max_abs_difference"] == 0.0 for row in parity_reports
        ),
        "all_fold_predictions_within_tolerance": all(
            row.get("prediction_within_tolerance", row["max_abs_difference"] == 0.0)
            for row in parity_reports
        ),
        "all_fold_ranks_identical": all(
            row["daily_rank_identical"] for row in parity_reports
        ),
        "all_fold_metrics_identical": all(
            row["metrics_identical"] for row in parity_reports
        ),
        "folds": parity_reports,
    }
    engineering["parity_passed"] = bool(
        engineering["all_fold_keys_identical"]
        and engineering["all_fold_predictions_within_tolerance"]
        and engineering["all_fold_ranks_identical"]
        and engineering["all_fold_metrics_identical"]
    )
    combined = {
        "run_role": "native_qlib_walk_forward",
        "evaluation_scope": "stitched_chronological_oos",
        "native": native,
        "qlib": qlib,
        "engineering_parity": engineering,
        "strategy_acceptance": acceptance,
    }
    pd.DataFrame(combined_rows).to_csv(
        output_dir / "walk_forward_summary.csv",
        index=False,
    )
    write_report(combined, output_dir / "walk_forward_summary.json")
    return combined


def finalize_existing_walk_forward(config: Config) -> dict:
    """Rebuild combined summaries from completed fold artifacts without retraining."""
    output_dir = config.output_dir
    current = json.loads(
        (output_dir / "walk_forward_summary.json").read_text(encoding="utf-8")
    )
    qlib = current["qlib"] if "qlib" in current else current
    native_folds = []
    for qlib_fold in qlib["folds"]:
        fold = qlib_fold["fold"]
        native_folds.append(json.loads(
            (output_dir / fold / "native_metrics.json").read_text(encoding="utf-8")
        ))
        parity = json.loads(
            (output_dir / fold / "prediction_parity.json").read_text(encoding="utf-8")
        )
        if not (
            parity["keys_identical"]
            and parity["daily_rank_identical"]
            and parity["metrics_identical"]
            and parity.get(
                "prediction_within_tolerance",
                parity["max_abs_difference"] == 0.0,
            )
        ):
            raise AssertionError(f"{fold} parity is not exact")

    native = {
        **qlib,
        "run_role": "native_walk_forward",
        "folds": native_folds,
    }
    qlib = {**qlib, "run_role": "qlib_walk_forward"}
    pd.DataFrame(native_folds).to_csv(
        output_dir / "native_walk_forward_summary.csv",
        index=False,
    )
    pd.DataFrame(qlib["folds"]).to_csv(
        output_dir / "qlib_walk_forward_summary.csv",
        index=False,
    )
    write_report(native, output_dir / "native_walk_forward_summary.json")
    write_report(qlib, output_dir / "qlib_walk_forward_summary.json")
    shutil.copy2(config.source_path, output_dir / "walk_forward.yaml")
    return _write_combined_summary(output_dir, native, qlib)


def finalize_completed_fold_artifacts(config: Config) -> dict:
    """Rebuild parity and summaries when every Native/Qlib fold artifact exists."""
    panel = prepare_experiment_panel(config)
    folds = build_walk_forward_folds(panel, config)
    tolerance = float(
        config.parity_config.get(
            "prediction_tolerance",
            1e-10 if config.model_type == "ridge" else 0.0,
        )
    )
    for fold in folds:
        fold_dir = config.output_dir / fold.name
        native = pd.read_pickle(fold_dir / "native_predictions.pkl.gz")
        qlib = pd.read_pickle(fold_dir / "qlib_predictions.pkl.gz")
        native_metrics = json.loads(
            (fold_dir / "native_metrics.json").read_text(encoding="utf-8")
        )
        qlib_metrics = json.loads(
            (fold_dir / "qlib_metrics.json").read_text(encoding="utf-8")
        )
        parity = _fold_parity(
            native,
            qlib,
            native_metrics,
            qlib_metrics,
            ranking_mode=config.raw["ranking"]["mode"],
            prediction_tolerance=tolerance,
        )
        write_report(parity, fold_dir / "prediction_parity.json")
        if not (
            parity["keys_identical"]
            and parity["daily_rank_identical"]
            and parity["metrics_identical"]
            and parity["prediction_within_tolerance"]
        ):
            raise AssertionError(f"{fold.name} parity failed: {parity}")
    native_summary = _write_summary(config, config.output_dir, folds, "native")
    qlib_summary = _write_summary(config, config.output_dir, folds, "qlib")
    return _write_combined_summary(config.output_dir, native_summary, qlib_summary)


def run_walk_forward(config: Config, mode: str, force_prepare: bool = False) -> dict:
    if mode not in {"native", "qlib"}:
        raise ValueError("mode must be 'native' or 'qlib'")
    started = perf_counter()
    panel = prepare_experiment_panel(config, force=force_prepare)
    folds = build_walk_forward_folds(panel, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.source_path, config.output_dir / "walk_forward.yaml")
    for number, fold in enumerate(folds, start=1):
        fold_started = perf_counter()
        logger.info(
            "Walk-forward %s %s/%s: %s train=%s..%s valid=%s..%s test=%s..%s",
            mode,
            number,
            len(folds),
            fold.name,
            fold.train_start.date(),
            fold.train_end.date(),
            fold.valid_start.date(),
            fold.valid_end.date(),
            fold.test_start.date(),
            fold.test_end.date(),
        )
        fold_dir = config.output_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        train = _slice(panel, fold.train_start, fold.train_end)
        valid = _slice(panel, fold.valid_start, fold.valid_end)
        test = _slice(panel, fold.test_start, fold.test_end)

        if mode == "native":
            model = make_model(
                config.model_type,
                config.raw["model"].get("params", {}),
                config.feature_columns,
                config.categorical_features,
            )
            model.fit(train, valid)
            train_pred = _predict_native(model, train)
            valid_pred = _predict_native(model, valid)
            test_pred = _predict_native(model, test)
        else:
            bounded = _slice(panel, fold.train_start, fold.test_end)
            dataset = make_dataset(
                bounded,
                fold.split_config(),
                feature_columns=config.feature_columns,
            )
            model = QlibPublishedModel(
                config.raw["model"].get("params", {}),
                model_type=config.model_type,
                feature_columns=config.feature_columns,
                categorical_features=config.categorical_features,
            )
            model.fit(dataset)
            train_pred = _prediction_from_qlib(model, dataset, "train", train)
            valid_pred = _prediction_from_qlib(model, dataset, "valid", valid)
            test_pred = _prediction_from_qlib(model, dataset, "test", test)

        joblib.dump(model, fold_dir / f"{mode}_model.joblib")
        ranked, daily, metrics = _segment_metrics(
            train_pred,
            valid_pred,
            test_pred,
            config,
            fold,
            role=f"{mode}_walk_forward_fold",
            best_iteration=model.best_iteration,
        )
        test_pred.to_pickle(fold_dir / f"{mode}_predictions.pkl.gz", compression="gzip")
        valid_pred.to_pickle(
            fold_dir / f"{mode}_valid_predictions.pkl.gz", compression="gzip"
        )
        ranked.to_pickle(fold_dir / f"{mode}_ranked.pkl.gz", compression="gzip")
        daily.to_csv(fold_dir / f"{mode}_daily_spread.csv", header=True)
        write_report(metrics, fold_dir / f"{mode}_metrics.json")

        if mode == "qlib":
            native_path = fold_dir / "native_predictions.pkl.gz"
            native_metrics_path = fold_dir / "native_metrics.json"
            if not native_path.is_file() or not native_metrics_path.is_file():
                raise FileNotFoundError(
                    f"{fold.name} requires Native outputs before Qlib parity"
                )
            parity = _fold_parity(
                pd.read_pickle(native_path),
                test_pred,
                json.loads(native_metrics_path.read_text(encoding="utf-8")),
                metrics,
                ranking_mode=config.raw["ranking"]["mode"],
                prediction_tolerance=float(
                    config.parity_config.get(
                        "prediction_tolerance",
                        1e-10 if config.model_type == "ridge" else 0.0,
                    )
                ),
            )
            write_report(parity, fold_dir / "prediction_parity.json")
            if not (
                parity["keys_identical"]
                and parity["daily_rank_identical"]
                and parity["metrics_identical"]
                and parity["prediction_within_tolerance"]
            ):
                raise AssertionError(f"{fold.name} Native/Qlib parity failed: {parity}")
        logger.info(
            "Walk-forward %s %s complete: rows=%s OOS Sharpe=%.6f in %.1fs",
            mode,
            fold.name,
            f"{len(test):,}",
            metrics["oos_sharpe"],
            perf_counter() - fold_started,
        )

    summary = _write_summary(config, config.output_dir, folds, mode)
    if mode == "qlib":
        native_summary_path = config.output_dir / "native_walk_forward_summary.json"
        if not native_summary_path.is_file():
            raise FileNotFoundError(
                "Native walk-forward summary is required before combined Qlib summary"
            )
        native_summary = json.loads(native_summary_path.read_text(encoding="utf-8"))
        _write_combined_summary(config.output_dir, native_summary, summary)
    logger.info(
        "Walk-forward %s complete: aggregate OOS Sharpe=%.6f positive_folds=%s/%s in %.1fs",
        mode,
        summary["aggregate_oos_sharpe"],
        summary["positive_folds"],
        summary["fold_count"],
        perf_counter() - started,
    )
    return summary
