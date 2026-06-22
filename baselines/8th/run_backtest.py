"""Build a reproducible long-short backtest for the frozen JPX 8th-place baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_baseline as baseline


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_COST_BPS = (0.0, 15.0)
PORTFOLIO_SIZE = 200
TRADING_DAYS = 252


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "jpx")
    parser.add_argument(
        "--baseline-artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "baseline_8th",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "backtest_8th"
    )
    parser.add_argument("--cost-bps", type=float, nargs="+", default=list(DEFAULT_COST_BPS))
    parser.add_argument("--force-rebuild-train-predictions", action="store_true")
    return parser.parse_args()


def file_fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": baseline.sha256_file(path)}


def cache_key(inputs: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(inputs, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_models(artifacts_dir: Path) -> dict[str, Any]:
    paths = {
        "reference": artifacts_dir / "reference" / "model.pickle",
        "retrain": artifacts_dir / "retrain" / "model.pickle",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    models = {name: baseline.load_reference_model(path) for name, path in paths.items()}
    columns = baseline.training_columns()
    for model in models.values():
        baseline.validate_model_features(model, columns)
    return models


def rank_vector(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    return frame.groupby("Date", sort=False)[prediction_column].rank(
        method="first", ascending=False
    ).astype(np.int64) - 1


def build_training_predictions(
    train_path: Path,
    models: dict[str, Any],
    cache_path: Path,
    cache_manifest_path: Path,
    expected_cache_key: str,
    force: bool,
) -> pd.DataFrame:
    if cache_path.is_file() and cache_manifest_path.is_file() and not force:
        cached_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cached_manifest.get("cache_key") == expected_cache_key:
            print(f"Loading valid training prediction cache: {cache_path}", flush=True)
            return pd.read_parquet(cache_path)

    prepared = baseline.prepare_prices(train_path)
    if len(prepared) != baseline.EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Expected {baseline.EXPECTED_TRAIN_ROWS} prepared training rows, got {len(prepared)}"
        )
    columns = baseline.training_columns()
    output = prepared[["Date", "SecuritiesCode", "Target"]].copy()
    for model_name, model in models.items():
        prediction_column = f"Prediction_{model_name}"
        rank_column = f"Rank_{model_name}"
        print(f"Predicting training period with {model_name} model", flush=True)
        output[prediction_column] = baseline.model_predict(model, prepared[columns])
        output[rank_column] = rank_vector(output, prediction_column)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(cache_path, index=False)
    baseline.write_json(
        cache_manifest_path,
        {
            "cache_key": expected_cache_key,
            "rows": len(output),
            "dates": int(output["Date"].nunique()),
            "date_min": output["Date"].min().date().isoformat(),
            "date_max": output["Date"].max().date().isoformat(),
            "columns": list(output.columns),
            "output_sha256": baseline.sha256_file(cache_path),
        },
    )
    return output


def load_supplemental_predictions(artifacts_dir: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for model_name in ("reference", "retrain"):
        path = artifacts_dir / model_name / "predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        current = pd.read_parquet(path)[["Date", "SecuritiesCode", "Target", "Prediction", "Rank"]]
        current = current.rename(
            columns={"Prediction": f"Prediction_{model_name}", "Rank": f"Rank_{model_name}"}
        )
        if merged is None:
            merged = current
        else:
            merged = merged.merge(
                current,
                on=["Date", "SecuritiesCode"],
                how="inner",
                validate="one_to_one",
                suffixes=("", "_other"),
            )
            if not np.allclose(merged["Target"], merged.pop("Target_other"), equal_nan=True):
                raise RuntimeError("Supplemental targets differ between model artifacts")
    assert merged is not None
    merged["Date"] = pd.to_datetime(merged["Date"])
    return merged


def validate_ranks(frame: pd.DataFrame, rank_column: str) -> None:
    counts = frame.groupby("Date", sort=False).size()
    distinct = frame.groupby("Date", sort=False)[rank_column].nunique()
    minimum = frame.groupby("Date", sort=False)[rank_column].min()
    maximum = frame.groupby("Date", sort=False)[rank_column].max()
    invalid = (distinct != counts) | (minimum != 0) | (maximum != counts - 1)
    if invalid.any():
        raise RuntimeError(f"Invalid daily ranks in {rank_column}: {invalid[invalid].index[:5].tolist()}")


def construct_positions(
    predictions: pd.DataFrame,
    period: str,
    model_name: str,
    portfolio_size: int = PORTFOLIO_SIZE,
) -> pd.DataFrame:
    rank_column = f"Rank_{model_name}"
    prediction_column = f"Prediction_{model_name}"
    validate_ranks(predictions, rank_column)
    day_size = predictions.groupby("Date", sort=False)["SecuritiesCode"].transform("size")
    selected = predictions.loc[
        (predictions[rank_column] < portfolio_size)
        | (predictions[rank_column] >= day_size - portfolio_size),
        ["Date", "SecuritiesCode", "Target", prediction_column, rank_column],
    ].copy()
    selected["Period"] = period
    selected["Model"] = model_name
    selected["Side"] = np.where(selected[rank_column] < portfolio_size, "Long", "Short")
    selected["SideRank"] = np.where(
        selected["Side"].eq("Long"), selected[rank_column], day_size.loc[selected.index] - 1 - selected[rank_column]
    ).astype(np.int64)
    raw_weight = 2.0 - selected["SideRank"] / (portfolio_size - 1)
    side_weight_sum = portfolio_size * 1.5
    selected["Weight"] = raw_weight / side_weight_sum * 0.5
    selected.loc[selected["Side"].eq("Short"), "Weight"] *= -1.0
    selected["Contribution"] = selected["Weight"] * selected["Target"]
    return selected.rename(columns={prediction_column: "Prediction", rank_column: "Rank"})


def daily_from_positions(positions: pd.DataFrame) -> pd.DataFrame:
    ordered = positions.sort_values(["Date", "SecuritiesCode"]).copy()
    previous = ordered[["Date", "SecuritiesCode", "Weight"]].copy()
    dates = np.sort(ordered["Date"].unique())
    previous_date = pd.Series(dates[1:], index=dates[:-1])
    previous["Date"] = previous["Date"].map(previous_date)
    previous = previous.dropna(subset=["Date"]).rename(columns={"Weight": "PreviousWeight"})
    changes = ordered[["Date", "SecuritiesCode", "Weight"]].merge(
        previous, on=["Date", "SecuritiesCode"], how="outer"
    ).fillna({"Weight": 0.0, "PreviousWeight": 0.0})
    changes["AbsChange"] = (changes["Weight"] - changes["PreviousWeight"]).abs()
    turnover = 0.5 * changes.groupby("Date", sort=True)["AbsChange"].sum()
    turnover.iloc[0] = 0.0  # Unknown pre-period holdings; do not charge artificial entry costs.

    grouped = ordered.groupby("Date", sort=True)
    daily = grouped.agg(
        GrossReturn=("Contribution", "sum"),
        GrossExposure=("Weight", lambda value: float(value.abs().sum())),
        NetExposure=("Weight", "sum"),
        Names=("SecuritiesCode", "size"),
    ).reset_index()
    long_return = ordered.loc[ordered["Side"].eq("Long")].groupby("Date")["Contribution"].sum()
    short_return = ordered.loc[ordered["Side"].eq("Short")].groupby("Date")["Contribution"].sum()
    daily["LongReturn"] = daily["Date"].map(long_return)
    daily["ShortReturn"] = daily["Date"].map(short_return)
    daily["Turnover"] = daily["Date"].map(turnover)
    return daily


def max_drawdown(returns: pd.Series) -> tuple[float, pd.Series, pd.Series]:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min()), equity, drawdown


def summarize_returns(daily: pd.DataFrame, period: str, model: str, cost_bps: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = daily.copy()
    result["Period"] = period
    result["Model"] = model
    result["CostBps"] = float(cost_bps)
    result["TradingCost"] = result["Turnover"] * cost_bps / 10_000.0
    result["NetReturn"] = result["GrossReturn"] - result["TradingCost"]
    maximum_drawdown, equity, drawdown = max_drawdown(result["NetReturn"])
    result["Equity"] = equity
    result["Drawdown"] = drawdown
    daily_std = float(result["NetReturn"].std())
    years = len(result) / TRADING_DAYS
    ending_equity = float(equity.iloc[-1])
    annualized_return = ending_equity ** (1.0 / years) - 1.0 if ending_equity > 0 and years > 0 else float("nan")
    annualized_volatility = daily_std * math.sqrt(TRADING_DAYS)
    sharpe = (
        float(result["NetReturn"].mean()) / daily_std * math.sqrt(TRADING_DAYS)
        if daily_std > 0 else float("nan")
    )
    summary = {
        "period": period,
        "model": model,
        "cost_bps": float(cost_bps),
        "start_date": result["Date"].min().date().isoformat(),
        "end_date": result["Date"].max().date().isoformat(),
        "trading_days": len(result),
        "total_return": ending_equity - 1.0,
        "annualized_return": annualized_return,
        "annualized_arithmetic_return": float(result["NetReturn"].mean() * TRADING_DAYS),
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": maximum_drawdown,
        "hit_rate": float((result["NetReturn"] > 0).mean()),
        "average_daily_turnover": float(result["Turnover"].iloc[1:].mean()),
        "annualized_cost_drag": float(result["TradingCost"].mean() * TRADING_DAYS),
        "average_gross_exposure": float(result["GrossExposure"].mean()),
        "average_net_exposure": float(result["NetExposure"].mean()),
        "average_names": float(result["Names"].mean()),
        "ending_equity": ending_equity,
    }
    return result, summary


def render_report(template_path: Path, output_path: Path, payload: dict[str, Any]) -> None:
    template = template_path.read_text(encoding="utf-8")
    output_path.write_text(
        template.replace("__BACKTEST_PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    started = time.time()
    data_dir = args.data_dir.resolve()
    artifacts_dir = args.baseline_artifacts_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_files" / "stock_prices.csv"
    model_paths = {
        "reference": artifacts_dir / "reference" / "model.pickle",
        "retrain": artifacts_dir / "retrain" / "model.pickle",
    }
    supplemental_prediction_paths = {
        name: artifacts_dir / name / "predictions.parquet" for name in ("reference", "retrain")
    }
    prediction_source_paths = {
        "Features.py": SCRIPT_DIR / "Features.py",
        "Preprocessing.py": SCRIPT_DIR / "Preprocessing.py",
        "Trackers.py": SCRIPT_DIR / "Trackers.py",
        "run_baseline.py": SCRIPT_DIR / "run_baseline.py",
        "run_backtest.py": Path(__file__).resolve(),
    }
    inputs = {"train": file_fingerprint(train_path)}
    inputs.update({f"model_{name}": file_fingerprint(path) for name, path in model_paths.items()})
    inputs.update(
        {f"supplemental_predictions_{name}": file_fingerprint(path) for name, path in supplemental_prediction_paths.items()}
    )
    inputs.update(
        {f"prediction_source_{name}": file_fingerprint(path) for name, path in prediction_source_paths.items()}
    )
    training_cache_inputs = {
        name: value
        for name, value in inputs.items()
        if name == "train" or name.startswith("model_") or name.startswith("prediction_source_")
    }
    models = load_models(artifacts_dir)
    training = build_training_predictions(
        train_path,
        models,
        cache_dir / "training_predictions.parquet",
        cache_dir / "training_predictions_manifest.json",
        cache_key(training_cache_inputs),
        args.force_rebuild_train_predictions,
    )
    supplemental = load_supplemental_predictions(artifacts_dir)

    all_positions: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for period, predictions in (("Training (in-sample)", training), ("Supplemental (OOS proxy)", supplemental)):
        for model_name in ("reference", "retrain"):
            positions = construct_positions(predictions, period, model_name)
            daily = daily_from_positions(positions)
            all_positions.append(positions)
            for cost_bps in sorted(set(args.cost_bps)):
                scenario_daily, summary = summarize_returns(daily, period, model_name, cost_bps)
                all_daily.append(scenario_daily)
                summaries.append(summary)

    positions_frame = pd.concat(all_positions, ignore_index=True)
    daily_frame = pd.concat(all_daily, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    positions_path = output_dir / "portfolio_positions.parquet"
    daily_path = output_dir / "daily_portfolio.csv"
    summary_path = output_dir / "summary.csv"
    positions_frame.to_parquet(positions_path, index=False)
    daily_frame.to_csv(daily_path, index=False)
    summary_frame.to_csv(summary_path, index=False)

    payload = {
        "summary": summaries,
        "daily": [
            {
                **row,
                "Date": pd.Timestamp(row["Date"]).date().isoformat(),
            }
            for row in daily_frame[["Date", "Period", "Model", "CostBps", "NetReturn", "Equity", "Drawdown", "Turnover"]].to_dict("records")
        ],
        "meta": {
            "generated_at": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat(),
            "portfolio": "Top 200 long / bottom 200 short; competition linear weights; 50% gross per side",
            "cost": "Cost = one-way turnover × bps; first date turnover and cost set to zero",
            "warning": "Training is in-sample because both supplied models were fit using the full training interval. Supplemental is an out-of-sample proxy, not a production execution test.",
        },
    }
    report_template = SCRIPT_DIR / "backtest_report_template.html"
    report_path = output_dir / "report.html"
    render_report(report_template, report_path, payload)

    manifest = {
        "name": "JPX 8th-place daily long-short backtest",
        "command": " ".join(sys.argv),
        "inputs": inputs,
        "cost_bps": sorted(set(args.cost_bps)),
        "portfolio_size_per_side": PORTFOLIO_SIZE,
        "annualization_days": TRADING_DAYS,
        "turnover_definition": "0.5 * sum(abs(weight_t - weight_t_minus_1)); first date set to zero",
        "cost_definition": "turnover * cost_bps / 10000",
        "period_interpretation": {
            "Training (in-sample)": "Full-period fitted-model backcast; descriptive only",
            "Supplemental (OOS proxy)": "Held-out competition supplemental interval; proxy only",
        },
        "outputs": {
            path.name: baseline.sha256_file(path)
            for path in [positions_path, daily_path, summary_path, report_path]
        },
        "versions": baseline.package_versions(),
        "elapsed_seconds": time.time() - started,
        "summary": summaries,
    }
    baseline.write_json(output_dir / "manifest.json", manifest)
    print(summary_frame.to_string(index=False), flush=True)
    print(f"Backtest report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
