from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, load_config
from .strategy_experiments import (
    StrategySpec,
    construct_stateful_positions,
    evaluate_positions,
)


SECTOR_PORTFOLIOS = (
    StrategySpec("baseline", "baseline"),
    StrategySpec("smooth_3d", "smoothing", smoothing_days=3),
)


def apply_sector_prediction_transform(
    predictions: pd.DataFrame,
    sector_map: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    if method not in {"raw", "sector_demean", "sector_rank"}:
        raise ValueError(f"Unsupported sector transform: {method}")
    values = predictions.merge(
        sector_map[["SecuritiesCode", "Sector"]],
        on="SecuritiesCode",
        how="left",
        validate="many_to_one",
    )
    values["Sector"] = values["Sector"].fillna("UNKNOWN")
    if method == "sector_demean":
        values["Prediction"] -= values.groupby(
            ["Date", "Sector"], sort=False
        )["Prediction"].transform("mean")
    elif method == "sector_rank":
        values["Prediction"] = values.groupby(
            ["Date", "Sector"], sort=False
        )["Prediction"].rank(method="average", pct=True)
    return values.drop(columns="Sector")


def sector_exposure_daily(
    positions: pd.DataFrame,
    sector_map: pd.DataFrame,
) -> pd.DataFrame:
    values = positions.merge(
        sector_map[["SecuritiesCode", "Sector"]],
        on="SecuritiesCode",
        how="left",
        validate="many_to_one",
    )
    values["Sector"] = values["Sector"].fillna("UNKNOWN")
    by_sector = (
        values.groupby(["Date", "Sector"], sort=True)["Weight"]
        .sum()
        .rename("NetSectorWeight")
        .reset_index()
    )
    totals = by_sector.groupby("Date", sort=True).agg(
        GrossNetSectorExposure=(
            "NetSectorWeight", lambda x: float(x.abs().sum())
        ),
        MaxAbsoluteSectorExposure=(
            "NetSectorWeight", lambda x: float(x.abs().max())
        ),
    )
    return by_sector.merge(totals, on="Date", how="left", validate="many_to_one")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _resolve(config: Config, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config.project_root / path).resolve()


def _load_sector_map(path: Path) -> pd.DataFrame:
    values = pd.read_csv(
        path,
        usecols=["SecuritiesCode", "33SectorCode", "33SectorName"],
        dtype={"SecuritiesCode": int, "33SectorCode": str, "33SectorName": str},
    )
    code = values["33SectorCode"].fillna("-").str.strip()
    name = values["33SectorName"].fillna("UNKNOWN").str.strip()
    values["Sector"] = np.where(
        code.eq("-") | code.eq(""),
        "UNKNOWN",
        code + " " + name,
    )
    return values[["SecuritiesCode", "Sector"]].drop_duplicates(
        "SecuritiesCode", keep="last"
    )


def run_sector_diagnostics(config: Config) -> dict[str, Any]:
    options = config.raw.get("sector_diagnostics", {})
    if not isinstance(options, dict):
        raise ValueError("sector_diagnostics must be a mapping")
    source = load_config(_resolve(config, str(options["source_config"])))
    sector_file = _resolve(config, str(options["stock_list_csv"]))
    sector_map = _load_sector_map(sector_file)
    methods = [
        str(value) for value in options.get(
            "methods", ["raw", "sector_demean", "sector_rank"]
        )
    ]
    cost_bps = float(options.get("cost_bps", 5))
    annualization_days = int(
        source.raw["evaluation"].get("annualization_days", 252)
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    exposures: list[pd.DataFrame] = []
    stitched: dict[str, list[pd.DataFrame]] = {method: [] for method in methods}
    parity_rows: list[dict[str, Any]] = []
    for fold_spec in source.raw["walk_forward"]["folds"]:
        fold = str(fold_spec["name"])
        fold_dir = source.output_dir / fold
        native_test = pd.read_pickle(fold_dir / "native_predictions.pkl.gz")
        native_valid = pd.read_pickle(fold_dir / "native_valid_predictions.pkl.gz")
        qlib_test = pd.read_pickle(fold_dir / "qlib_predictions.pkl.gz")
        qlib_valid = pd.read_pickle(fold_dir / "qlib_valid_predictions.pkl.gz")
        for method in methods:
            test = apply_sector_prediction_transform(native_test, sector_map, method)
            valid = apply_sector_prediction_transform(native_valid, sector_map, method)
            qtest = apply_sector_prediction_transform(qlib_test, sector_map, method)
            qvalid = apply_sector_prediction_transform(qlib_valid, sector_map, method)
            difference = float(
                (test["Prediction"] - qtest["Prediction"]).abs().max()
            )
            for portfolio in SECTOR_PORTFOLIOS:
                positions, _ = construct_stateful_positions(
                    test, portfolio, warmup=valid
                )
                qpositions, _ = construct_stateful_positions(
                    qtest, portfolio, warmup=qvalid
                )
                columns = ["Date", "SecuritiesCode", "Weight"]
                if not positions[columns].equals(qpositions[columns]):
                    raise AssertionError(
                        f"{fold} {method} {portfolio.name} portfolio parity failed"
                    )
                _, summary = evaluate_positions(
                    positions,
                    cost_bps=cost_bps,
                    annualization_days=annualization_days,
                )
                exposure = sector_exposure_daily(positions, sector_map)
                exposure["Fold"] = fold
                exposure["Method"] = method
                exposure["Portfolio"] = portfolio.name
                exposures.append(exposure)
                daily_sector = exposure.groupby("Date", sort=True).first()
                rows.append({
                    "Fold": fold,
                    "Method": method,
                    "Portfolio": portfolio.name,
                    "GrossSharpe": float(summary["gross_sharpe"]),
                    "NetSharpe": float(summary["net_sharpe"]),
                    "BreakEvenCostBps": float(summary["break_even_cost_bps"]),
                    "AverageTradedNotional": float(
                        summary["average_daily_traded_notional"]
                    ),
                    "NetMaxDrawdown": float(summary["net_max_drawdown"]),
                    "LongContribution": float(summary["long_contribution_sum"]),
                    "ShortContribution": float(summary["short_contribution_sum"]),
                    "AverageGrossNetSectorExposure": float(
                        daily_sector["GrossNetSectorExposure"].mean()
                    ),
                    "AverageMaxAbsoluteSectorExposure": float(
                        daily_sector["MaxAbsoluteSectorExposure"].mean()
                    ),
                })
            parity_rows.append({
                "Fold": fold,
                "Method": method,
                "PredictionMaxAbsDifference": difference,
                "PredictionWithinTolerance": difference <= 1e-15,
                "PortfolioAccountingExact": True,
            })
            stitched[method].append(test)

    fold_table = pd.DataFrame(rows)
    summary = (
        fold_table.groupby(["Method", "Portfolio"], as_index=False)
        .agg(
            MedianFoldGrossSharpe=("GrossSharpe", "median"),
            MedianFoldNetSharpe=("NetSharpe", "median"),
            PositiveNetFolds=("NetSharpe", lambda x: int((x > 0).sum())),
            WorstFoldNetSharpe=("NetSharpe", "min"),
            MedianBreakEvenCostBps=("BreakEvenCostBps", "median"),
            MedianTradedNotional=("AverageTradedNotional", "median"),
            MedianGrossNetSectorExposure=(
                "AverageGrossNetSectorExposure", "median"
            ),
            MedianMaxAbsoluteSectorExposure=(
                "AverageMaxAbsoluteSectorExposure", "median"
            ),
        )
    )
    fold_table.to_csv(output_dir / "sector_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "sector_summary.csv", index=False)
    pd.concat(exposures, ignore_index=True).to_csv(
        output_dir / "sector_exposure_daily.csv", index=False
    )
    parity = pd.DataFrame(parity_rows)
    parity.to_csv(output_dir / "sector_parity.csv", index=False)
    for method, frames in stitched.items():
        pd.concat(frames, ignore_index=True).to_pickle(
            output_dir / f"{method}_predictions.pkl.gz", compression="gzip"
        )
    report = {
        "run_role": "static_sector_prediction_diagnostics",
        "source_variant": str(source.raw["ablation"]["id"]),
        "stock_list_csv": str(sector_file),
        "point_in_time_warning": (
            "The supplied stock_list.csv is a static 2021-12-30 classification. "
            "Historical sector changes are not point-in-time and may drift."
        ),
        "methods": methods,
        "portfolios": [spec.name for spec in SECTOR_PORTFOLIOS],
        "fold_metrics": _records(fold_table),
        "summary": _records(summary),
        "parity": _records(parity),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    template = Path(__file__).with_name("templates") / "sector_diagnostics.html"
    (output_dir / "sector_diagnostics.html").write_text(
        template.read_text(encoding="utf-8").replace(
            "__SECTOR_PAYLOAD__",
            json.dumps(report, ensure_ascii=False, separators=(",", ":")),
        ),
        encoding="utf-8",
    )
    return report
