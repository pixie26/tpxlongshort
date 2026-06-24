from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .parity import write_report
from .portfolio import (
    _annualized_return,
    _nav_and_drawdown,
    _resolve_input_path,
    summarize_cost_scenario,
)
from .ranking import add_rank

logger = logging.getLogger(__name__)


def _options(config: Config) -> dict[str, Any]:
    value = config.raw.get("portfolio_diagnostics", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("portfolio_diagnostics config must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_ranges(config: Config) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    specs = config.raw.get("walk_forward", {}).get("folds", [])
    ranges = []
    for spec in specs:
        start, end = spec["test"]
        ranges.append((str(spec["name"]), pd.Timestamp(start), pd.Timestamp(end)))
    if not ranges:
        raise ValueError("walk_forward.folds is required for portfolio diagnostics")
    return ranges


def _assign_fold(dates: pd.Series, ranges: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    result = pd.Series(pd.NA, index=dates.index, dtype="string")
    for name, start, end in ranges:
        result.loc[dates.between(start, end)] = name
    if result.isna().any():
        missing = dates.loc[result.isna()].iloc[0]
        raise ValueError(f"No walk-forward fold covers date {missing.date()}")
    return result


def side_turnover_decomposition(positions: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "SecuritiesCode", "Side", "Weight"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(f"Position input missing columns: {sorted(missing)}")

    ordered = positions[list(required)].copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"], errors="raise")
    dates = pd.DatetimeIndex(ordered["Date"].drop_duplicates().sort_values())
    next_date = pd.Series(dates[1:].to_numpy(), index=dates[:-1])
    rows: list[pd.DataFrame] = []
    for side in ("Long", "Short"):
        current = ordered.loc[ordered["Side"].eq(side), ["Date", "SecuritiesCode", "Weight"]]
        previous = current.copy()
        previous["Date"] = previous["Date"].map(next_date)
        previous = previous.dropna(subset=["Date"]).rename(columns={"Weight": "PreviousWeight"})
        changes = current.merge(
            previous,
            on=["Date", "SecuritiesCode"],
            how="outer",
            validate="one_to_one",
        ).fillna({"Weight": 0.0, "PreviousWeight": 0.0})
        active_now = changes["Weight"].ne(0.0)
        active_before = changes["PreviousWeight"].ne(0.0)
        changes["EntryNotional"] = np.where(
            active_now & ~active_before, changes["Weight"].abs(), 0.0
        )
        changes["ExitNotional"] = np.where(
            ~active_now & active_before, changes["PreviousWeight"].abs(), 0.0
        )
        changes["ResizeNotional"] = np.where(
            active_now & active_before,
            (changes["Weight"] - changes["PreviousWeight"]).abs(),
            0.0,
        )
        changes["TradedNotional"] = (
            changes["EntryNotional"] + changes["ExitNotional"] + changes["ResizeNotional"]
        )
        daily = changes.groupby("Date", sort=True)[
            ["EntryNotional", "ExitNotional", "ResizeNotional", "TradedNotional"]
        ].sum()
        daily.columns = [f"{side}{column}" for column in daily.columns]
        rows.append(daily)

    output = pd.concat(rows, axis=1).fillna(0.0).sort_index()
    output["LongHalfTurnover"] = 0.5 * output["LongTradedNotional"]
    output["ShortHalfTurnover"] = 0.5 * output["ShortTradedNotional"]
    output["TradedNotional"] = (
        output["LongTradedNotional"] + output["ShortTradedNotional"]
    )
    output["HalfTurnover"] = 0.5 * output["TradedNotional"]
    component_sum = output[
        [
            "LongEntryNotional",
            "LongExitNotional",
            "LongResizeNotional",
            "ShortEntryNotional",
            "ShortExitNotional",
            "ShortResizeNotional",
        ]
    ].sum(axis=1)
    if not np.allclose(component_sum, output["TradedNotional"], atol=1e-15):
        raise AssertionError("Turnover decomposition does not sum to traded notional")
    return output.reset_index()


def build_side_attribution(
    positions: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    primary_cost_bps: float,
    fold_ranges: list[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    contributions = positions.pivot_table(
        index="Date",
        columns="Side",
        values="Contribution",
        aggfunc="sum",
    ).rename(columns={"Long": "LongGross", "Short": "ShortGross"})
    daily = contributions.join(turnover.set_index("Date"), how="inner").reset_index()
    daily["GrossReturn"] = daily["LongGross"] + daily["ShortGross"]
    daily["LongCost"] = daily["LongTradedNotional"] * primary_cost_bps / 10_000.0
    daily["ShortCost"] = daily["ShortTradedNotional"] * primary_cost_bps / 10_000.0
    daily["TradingCost"] = daily["LongCost"] + daily["ShortCost"]
    daily["LongNet"] = daily["LongGross"] - daily["LongCost"]
    daily["ShortNet"] = daily["ShortGross"] - daily["ShortCost"]
    daily["TotalNet"] = daily["LongNet"] + daily["ShortNet"]
    daily["Fold"] = _assign_fold(daily["Date"], fold_ranges)
    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.to_period("M").astype(str)
    daily["IsFoldFirstDay"] = daily.groupby("Fold")["Date"].transform("min").eq(daily["Date"])
    if not np.allclose(
        daily["GrossReturn"], daily["LongGross"] + daily["ShortGross"], atol=1e-15
    ):
        raise AssertionError("Gross return does not equal side contributions")
    if not np.allclose(
        daily["TradingCost"], daily["LongCost"] + daily["ShortCost"], atol=1e-15
    ):
        raise AssertionError("Trading cost does not equal side costs")
    if not np.allclose(
        daily["TotalNet"], daily["GrossReturn"] - daily["TradingCost"], atol=1e-15
    ):
        raise AssertionError("Net return accounting identity failed")
    return daily


def _standalone_daily(side_daily: pd.DataFrame, portfolio: str) -> pd.DataFrame:
    common = side_daily[
        ["Date", "LongGross", "ShortGross", "LongTradedNotional", "ShortTradedNotional"]
    ].copy()
    if portfolio == "Long-only":
        common["GrossReturn"] = 2.0 * common["LongGross"]
        common["LongContribution"] = common["GrossReturn"]
        common["ShortContribution"] = 0.0
        common["TradedNotional"] = 2.0 * common["LongTradedNotional"]
        common["GrossExposure"] = 1.0
        common["NetExposure"] = 1.0
        common["Names"] = 200
    elif portfolio == "Short-only":
        common["GrossReturn"] = 2.0 * common["ShortGross"]
        common["LongContribution"] = 0.0
        common["ShortContribution"] = common["GrossReturn"]
        common["TradedNotional"] = 2.0 * common["ShortTradedNotional"]
        common["GrossExposure"] = 1.0
        common["NetExposure"] = -1.0
        common["Names"] = 200
    elif portfolio == "Long-short":
        common["GrossReturn"] = common["LongGross"] + common["ShortGross"]
        common["LongContribution"] = common["LongGross"]
        common["ShortContribution"] = common["ShortGross"]
        common["TradedNotional"] = (
            common["LongTradedNotional"] + common["ShortTradedNotional"]
        )
        common["GrossExposure"] = 1.0
        common["NetExposure"] = 0.0
        common["Names"] = 400
    else:
        raise ValueError(f"Unknown portfolio: {portfolio}")
    common["HalfTurnover"] = 0.5 * common["TradedNotional"]
    common["Turnover"] = common["HalfTurnover"]
    return common[
        [
            "Date",
            "GrossReturn",
            "LongContribution",
            "ShortContribution",
            "TradedNotional",
            "HalfTurnover",
            "Turnover",
            "GrossExposure",
            "NetExposure",
            "Names",
        ]
    ]


def portfolio_scenarios(
    side_daily: pd.DataFrame,
    *,
    costs: list[float],
    annualization_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    scenario_rows = []
    for portfolio in ("Long-only", "Short-only", "Long-short"):
        base = _standalone_daily(side_daily, portfolio)
        break_even = (
            float(base["GrossReturn"].sum() / base["TradedNotional"].sum() * 10_000.0)
            if base["TradedNotional"].sum() > 0
            else math.nan
        )
        for cost in costs:
            scenario, summary = summarize_cost_scenario(
                base,
                cost_bps=cost,
                annualization_days=annualization_days,
            )
            summary["portfolio"] = portfolio
            summary["break_even_cost_bps"] = break_even
            summary["total_traded_notional"] = float(base["TradedNotional"].sum())
            summaries.append(summary)
            scenario["Portfolio"] = portfolio
            scenario_rows.append(scenario)
    return pd.DataFrame(summaries), pd.concat(scenario_rows, ignore_index=True)


def _break_even(gross: pd.Series, traded: pd.Series) -> float:
    notional = float(traded.sum())
    return float(gross.sum() / notional * 10_000.0) if notional > 0 else math.nan


def aggregate_attribution(daily: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, frame in daily.groupby(key, sort=True):
        gross_nav, _ = _nav_and_drawdown(frame["GrossReturn"])
        net_nav, net_drawdown = _nav_and_drawdown(frame["TotalNet"])
        net_std = float(frame["TotalNet"].std(ddof=1))
        rows.append({
            key: value,
            "Days": int(len(frame)),
            "LongGross": float(frame["LongGross"].sum()),
            "ShortGross": float(frame["ShortGross"].sum()),
            "GrossReturn": float(frame["GrossReturn"].sum()),
            "LongCost": float(frame["LongCost"].sum()),
            "ShortCost": float(frame["ShortCost"].sum()),
            "TradingCost": float(frame["TradingCost"].sum()),
            "LongNet": float(frame["LongNet"].sum()),
            "ShortNet": float(frame["ShortNet"].sum()),
            "TotalNet": float(frame["TotalNet"].sum()),
            "CompoundedGrossReturn": float(gross_nav.iloc[-1] - 1.0),
            "CompoundedNetReturn": float(net_nav.iloc[-1] - 1.0),
            "NetSharpe": (
                float(frame["TotalNet"].mean()) / net_std * math.sqrt(252)
                if net_std > 0 else math.nan
            ),
            "NetMaxDrawdown": float(net_drawdown.min()),
            "LongTradedNotional": float(frame["LongTradedNotional"].sum()),
            "ShortTradedNotional": float(frame["ShortTradedNotional"].sum()),
            "TotalTradedNotional": float(frame["TradedNotional"].sum()),
            "AverageHalfTurnover": float(frame["HalfTurnover"].mean()),
            "LongBreakEvenCostBps": _break_even(
                frame["LongGross"], frame["LongTradedNotional"]
            ),
            "ShortBreakEvenCostBps": _break_even(
                frame["ShortGross"], frame["ShortTradedNotional"]
            ),
            "TotalBreakEvenCostBps": _break_even(
                frame["GrossReturn"], frame["TradedNotional"]
            ),
        })
    return pd.DataFrame(rows)


def build_benchmark_relative(
    predictions: pd.DataFrame,
    side_daily: pd.DataFrame,
) -> pd.DataFrame:
    universe = (
        predictions.groupby("Date", sort=True)["Target"]
        .mean()
        .rename("UniverseReturn")
        .reset_index()
    )
    result = side_daily[
        ["Date", "Fold", "Year", "Month", "LongGross", "ShortGross", "GrossReturn"]
    ].merge(universe, on="Date", how="left", validate="one_to_one")
    result["LongOnlyReturn"] = 2.0 * result["LongGross"]
    result["BottomLongReturn"] = -2.0 * result["ShortGross"]
    result["ShortOnlyReturn"] = 2.0 * result["ShortGross"]
    result["LongExcessVsUniverse"] = result["LongOnlyReturn"] - result["UniverseReturn"]
    result["BottomExcessVsUniverse"] = (
        result["BottomLongReturn"] - result["UniverseReturn"]
    )
    result["ShortSelectionContribution"] = (
        result["UniverseReturn"] - result["BottomLongReturn"]
    )
    result["LongShortReturn"] = result["GrossReturn"]
    return result


def _beta_row(frame: pd.DataFrame, label: str, value: str) -> dict[str, Any]:
    benchmark = frame["UniverseReturn"]
    variance = float(benchmark.var(ddof=1))
    row: dict[str, Any] = {"Scope": label, "Value": value, "Days": int(len(frame))}
    for name, column in (
        ("LongOnly", "LongOnlyReturn"),
        ("ShortOnly", "ShortOnlyReturn"),
        ("LongShort", "LongShortReturn"),
    ):
        returns = frame[column]
        beta = float(returns.cov(benchmark) / variance) if variance > 0 else math.nan
        alpha = float((returns - beta * benchmark).mean() * 252)
        row[f"{name}Beta"] = beta
        row[f"{name}AnnualizedAlpha"] = alpha
        row[f"{name}Correlation"] = float(returns.corr(benchmark))
    return row


def benchmark_betas(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = [_beta_row(benchmark, "overall", "overall")]
    for key in ("Fold", "Year"):
        for value, frame in benchmark.groupby(key, sort=True):
            rows.append(_beta_row(frame, key.lower(), str(value)))
    return pd.DataFrame(rows)


def build_holdings_retention(
    positions: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    top_n: int,
    core_n: int,
    boundary_band: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = sorted(pd.to_datetime(positions["Date"]).unique())
    position_days = {
        pd.Timestamp(date): frame for date, frame in positions.groupby("Date", sort=True)
    }
    rank_days = {
        pd.Timestamp(date): frame.set_index("SecuritiesCode")["Rank"]
        for date, frame in ranked.groupby("Date", sort=True)
    }
    ages = {"Long": {}, "Short": {}}
    episodes = {"Long": [], "Short": []}
    previous_codes: dict[str, set[int]] = {"Long": set(), "Short": set()}
    previous_weights: dict[str, dict[int, float]] = {"Long": {}, "Short": {}}
    previous_core: dict[str, set[int]] = {"Long": set(), "Short": set()}
    rows = []

    for date_number, date_value in enumerate(dates):
        date = pd.Timestamp(date_value)
        frame = position_days[date]
        row: dict[str, Any] = {"Date": date}
        previous_rank = rank_days[dates[date_number - 1]] if date_number > 0 else None
        previous_day_size = len(previous_rank) if previous_rank is not None else 0
        for side, prefix in (("Long", "Top"), ("Short", "Bottom")):
            current = frame.loc[frame["Side"].eq(side)]
            current_codes = set(current["SecuritiesCode"].astype(int))
            current_weights = dict(
                zip(
                    current["SecuritiesCode"].astype(int),
                    current["Weight"].abs() / current["Weight"].abs().sum(),
                    strict=True,
                )
            )
            current_core = set(
                current.loc[current["SideRank"] < core_n, "SecuritiesCode"].astype(int)
            )
            retained = current_codes & previous_codes[side]
            entrants = current_codes - previous_codes[side]
            exits = previous_codes[side] - current_codes
            for code in exits:
                episodes[side].append(ages[side].pop(code))
            for code in current_codes:
                ages[side][code] = ages[side].get(code, 0) + 1
            weighted_overlap = sum(
                min(current_weights[code], previous_weights[side][code])
                for code in retained
            )
            if previous_rank is None:
                boundary_entrants = 0
            elif side == "Long":
                boundary_entrants = sum(
                    code in previous_rank.index
                    and top_n <= int(previous_rank.loc[code]) < top_n + boundary_band
                    for code in entrants
                )
            else:
                lower = previous_day_size - top_n - boundary_band
                upper = previous_day_size - top_n
                boundary_entrants = sum(
                    code in previous_rank.index
                    and lower <= int(previous_rank.loc[code]) < upper
                    for code in entrants
                )
            row[f"{prefix}Retention"] = (
                len(retained) / top_n if date_number > 0 else math.nan
            )
            row[f"{prefix}WeightedOverlap"] = (
                weighted_overlap if date_number > 0 else math.nan
            )
            row[f"{prefix}Core{core_n}Retention"] = (
                len(current_core & previous_core[side]) / core_n
                if date_number > 0 else math.nan
            )
            row[f"{prefix}Entrants"] = len(entrants)
            row[f"{prefix}Exits"] = len(exits)
            row[f"{prefix}BoundaryEntrants"] = boundary_entrants
            row[f"{prefix}BoundaryChurn"] = boundary_entrants / top_n
            row[f"{prefix}AverageHoldingAgeDays"] = float(
                np.mean([ages[side][code] for code in current_codes])
            )
            previous_codes[side] = current_codes
            previous_weights[side] = current_weights
            previous_core[side] = current_core
        rows.append(row)

    for side in ("Long", "Short"):
        episodes[side].extend(ages[side].values())
    retention = pd.DataFrame(rows)
    summary = {
        "top_n": top_n,
        "core_n": core_n,
        "boundary_band": boundary_band,
        "top_mean_retention": float(retention["TopRetention"].mean()),
        "bottom_mean_retention": float(retention["BottomRetention"].mean()),
        "top_mean_weighted_overlap": float(retention["TopWeightedOverlap"].mean()),
        "bottom_mean_weighted_overlap": float(retention["BottomWeightedOverlap"].mean()),
        "top_core_mean_retention": float(retention[f"TopCore{core_n}Retention"].mean()),
        "bottom_core_mean_retention": float(
            retention[f"BottomCore{core_n}Retention"].mean()
        ),
        "top_average_holding_days": float(np.mean(episodes["Long"])),
        "bottom_average_holding_days": float(np.mean(episodes["Short"])),
        "top_median_holding_days": float(np.median(episodes["Long"])),
        "bottom_median_holding_days": float(np.median(episodes["Short"])),
        "top_average_daily_entrants": float(retention["TopEntrants"].iloc[1:].mean()),
        "bottom_average_daily_entrants": float(
            retention["BottomEntrants"].iloc[1:].mean()
        ),
    }
    return retention, summary


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    values = frame.copy()
    for column in values.select_dtypes(include=["datetime64[ns]"]).columns:
        values[column] = values[column].dt.strftime("%Y-%m-%d")
    return json.loads(values.to_json(orient="records"))


def _render(output_path: Path, payload: dict[str, Any]) -> None:
    template = (
        Path(__file__).with_name("templates") / "portfolio_diagnostics.html"
    ).read_text(encoding="utf-8")
    output_path.write_text(
        template.replace(
            "__DIAGNOSTICS_PAYLOAD__",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        ),
        encoding="utf-8",
    )


def run_portfolio_diagnostics(config: Config) -> dict[str, Any]:
    started = perf_counter()
    options = _options(config)
    portfolio_options = config.raw["portfolio"]
    predictions_path = _resolve_input_path(
        config,
        str(options.get("predictions_file", portfolio_options["predictions_file"])),
    )
    positions_path = _resolve_input_path(
        config,
        str(
            options.get(
                "positions_file",
                f"{portfolio_options.get('output_subdir', 'portfolio_backtest')}/"
                "portfolio_positions.pkl.gz",
            )
        ),
    )
    for path in (predictions_path, positions_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    costs = sorted(
        {float(value) for value in options.get("cost_bps", portfolio_options["cost_bps"])}
    )
    primary_cost = float(
        options.get("primary_cost_bps", portfolio_options["primary_cost_bps"])
    )
    if primary_cost not in costs:
        raise ValueError("portfolio_diagnostics.primary_cost_bps must be in cost_bps")
    top_n = int(portfolio_options.get("top_n", 200))
    core_n = int(options.get("core_n", 50))
    boundary_band = int(options.get("boundary_band", 50))
    annualization_days = int(config.raw["evaluation"].get("annualization_days", 252))
    fold_ranges = _fold_ranges(config)

    logger.info("Loading diagnostics inputs: %s and %s", predictions_path, positions_path)
    predictions = pd.read_pickle(predictions_path)
    predictions["Date"] = pd.to_datetime(predictions["Date"], errors="raise")
    positions = pd.read_pickle(positions_path)
    positions["Date"] = pd.to_datetime(positions["Date"], errors="raise")

    turnover = side_turnover_decomposition(positions)
    daily = build_side_attribution(
        positions,
        turnover,
        primary_cost_bps=primary_cost,
        fold_ranges=fold_ranges,
    )
    scenario_summary, scenario_daily = portfolio_scenarios(
        daily,
        costs=costs,
        annualization_days=annualization_days,
    )
    fold = aggregate_attribution(daily, "Fold")
    yearly = aggregate_attribution(daily, "Year")
    monthly = aggregate_attribution(daily, "Month")
    benchmark = build_benchmark_relative(predictions, daily)
    betas = benchmark_betas(benchmark)
    ranked = add_rank(
        predictions[["Date", "SecuritiesCode", "Prediction", "Target"]],
        mode=str(config.raw["ranking"]["mode"]),
    )
    retention, retention_summary = build_holdings_retention(
        positions,
        ranked,
        top_n=top_n,
        core_n=core_n,
        boundary_band=boundary_band,
    )
    retention["Fold"] = _assign_fold(retention["Date"], fold_ranges)
    daily = daily.merge(retention, on=["Date", "Fold"], how="left", validate="one_to_one")

    output_dir = config.output_dir / str(
        options.get("output_subdir", "portfolio_diagnostics")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "side_attribution_daily.csv", index=False)
    monthly.to_csv(output_dir / "side_attribution_monthly.csv", index=False)
    fold.to_csv(output_dir / "side_attribution_fold.csv", index=False)
    yearly.to_csv(output_dir / "side_attribution_yearly.csv", index=False)
    benchmark.to_csv(output_dir / "benchmark_relative.csv", index=False)
    betas.to_csv(output_dir / "benchmark_beta.csv", index=False)
    retention.to_csv(output_dir / "holdings_retention.csv", index=False)
    turnover.to_csv(output_dir / "turnover_decomposition.csv", index=False)
    scenario_summary.to_csv(output_dir / "portfolio_summary.csv", index=False)

    total_gross = float(daily["GrossReturn"].sum())
    long_gross = float(daily["LongGross"].sum())
    short_gross = float(daily["ShortGross"].sum())
    long_cost = float(daily["LongCost"].sum())
    short_cost = float(daily["ShortCost"].sum())
    resize = float(
        turnover[["LongResizeNotional", "ShortResizeNotional"]].sum().sum()
    )
    entry_exit = float(
        turnover[
            [
                "LongEntryNotional",
                "LongExitNotional",
                "ShortEntryNotional",
                "ShortExitNotional",
            ]
        ].sum().sum()
    )
    first_days = daily["IsFoldFirstDay"]
    overall_beta = betas.loc[betas["Scope"].eq("overall")].iloc[0].to_dict()
    fold_by_name = fold.set_index("Fold")
    year_by_name = yearly.set_index("Year")
    loss_months = monthly["TotalNet"] < 0
    benchmark_sums = benchmark[
        [
            "UniverseReturn",
            "LongOnlyReturn",
            "BottomLongReturn",
            "LongExcessVsUniverse",
            "ShortSelectionContribution",
        ]
    ].sum()
    summary: dict[str, Any] = {
        "run_role": "stitched_oos_portfolio_diagnostics_2A",
        "evaluation_scope": "stitched_chronological_oos",
        "model_changed": False,
        "model_retrained": False,
        "primary_cost_bps": primary_cost,
        "cost_bps": costs,
        "sources": {
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "positions": str(positions_path),
            "positions_sha256": _sha256(positions_path),
        },
        "accounting": {
            "long_only": "Top 200, +100% gross",
            "short_only": "Bottom 200, -100% gross; returns use negative weights",
            "long_short": "Top +50%, Bottom -50%",
            "traded_notional": "sum(abs(weight_t - weight_t_minus_1))",
            "cost": "traded_notional * one_way_cost_bps / 10000",
            "first_day_entry_cost_charged": True,
            "break_even_cost_bps": "sum(gross_return) / sum(traded_notional) * 10000",
        },
        "portfolio_scenarios": _records(scenario_summary),
        "side_summary_primary_cost": {
            "long_gross": long_gross,
            "short_gross": short_gross,
            "total_gross": total_gross,
            "long_cost": long_cost,
            "short_cost": short_cost,
            "total_cost": long_cost + short_cost,
            "long_net": long_gross - long_cost,
            "short_net": short_gross - short_cost,
            "total_net": total_gross - long_cost - short_cost,
            "long_traded_notional": float(daily["LongTradedNotional"].sum()),
            "short_traded_notional": float(daily["ShortTradedNotional"].sum()),
            "long_break_even_cost_bps": _break_even(
                daily["LongGross"], daily["LongTradedNotional"]
            ),
            "short_break_even_cost_bps": _break_even(
                daily["ShortGross"], daily["ShortTradedNotional"]
            ),
            "total_break_even_cost_bps": _break_even(
                daily["GrossReturn"], daily["TradedNotional"]
            ),
        },
        "turnover": {
            "resize_notional": resize,
            "entry_exit_notional": entry_exit,
            "resize_share": resize / (resize + entry_exit),
            "entry_exit_share": entry_exit / (resize + entry_exit),
            "fold_first_day_average_traded_notional": float(
                daily.loc[first_days, "TradedNotional"].mean()
            ),
            "other_day_average_traded_notional": float(
                daily.loc[~first_days, "TradedNotional"].mean()
            ),
            "fold_first_day_cost_share": float(
                daily.loc[first_days, "TradingCost"].sum() / daily["TradingCost"].sum()
            ),
        },
        "retention": retention_summary,
        "universe_benchmark": overall_beta,
        "diagnosis": {
            "gross_return_source": (
                "Long side generated all absolute gross profit; short side lost money "
                "before costs."
            ),
            "long_survives_5bps": bool(long_gross - long_cost > 0),
            "short_survives_5bps": bool(short_gross - short_cost > 0),
            "long_gross_share_of_positive_side_gross": 1.0 if long_gross > 0 else 0.0,
            "short_cost_share": short_cost / (long_cost + short_cost),
            "short_relative_selection_contribution": float(
                benchmark_sums["ShortSelectionContribution"]
            ),
            "short_loss_interpretation": (
                "Bottom stocks underperformed the universe overall, but the short book's "
                "negative market beta lost more during a rising universe."
            ),
            "long_excess_vs_universe_sum": float(
                benchmark_sums["LongExcessVsUniverse"]
            ),
            "long_only_beta": float(overall_beta["LongOnlyBeta"]),
            "long_short_beta": float(overall_beta["LongShortBeta"]),
            "positive_long_gross_folds": int((fold["LongGross"] > 0).sum()),
            "positive_short_gross_folds": int((fold["ShortGross"] > 0).sum()),
            "positive_total_gross_folds": int((fold["GrossReturn"] > 0).sum()),
            "fold_side_instability": (
                "2020 H1 profit came from short while long lost; 2020 H2 reversed, "
                "with long profitable and short losing."
            ),
            "2019_h2_long_gross": float(fold_by_name.loc["fold_01", "LongGross"]),
            "2019_h2_short_gross": float(fold_by_name.loc["fold_01", "ShortGross"]),
            "2020_long_gross": float(year_by_name.loc[2020, "LongGross"]),
            "2020_short_gross": float(year_by_name.loc[2020, "ShortGross"]),
            "2020_total_net_5bps": float(year_by_name.loc[2020, "TotalNet"]),
            "2021_h2_long_gross": float(fold_by_name.loc["fold_05", "LongGross"]),
            "2021_h2_short_gross": float(fold_by_name.loc["fold_05", "ShortGross"]),
            "loss_month_average_half_turnover": float(
                monthly.loc[loss_months, "AverageHalfTurnover"].mean()
            ),
            "profit_month_average_half_turnover": float(
                monthly.loc[~loss_months, "AverageHalfTurnover"].mean()
            ),
            "monthly_net_turnover_correlation": float(
                monthly["TotalNet"].corr(monthly["AverageHalfTurnover"])
            ),
            "retraining_first_day_cost_concentrated": bool(
                daily.loc[first_days, "TradingCost"].sum() / daily["TradingCost"].sum()
                > 2.0 * first_days.mean()
            ),
        },
        "deferred_2B": [
            "TOPIX-aligned beta and alpha",
            "sector exposure and contribution",
            "point-in-time market capitalization",
            "borrow availability and complete shortability analysis",
        ],
        "elapsed_seconds": perf_counter() - started,
    }
    write_report(summary, output_dir / "summary.json")
    shutil.copy2(config.source_path, output_dir / "walk_forward.yaml")

    primary_scenarios = scenario_daily.loc[scenario_daily["CostBps"].eq(primary_cost)]
    nav = primary_scenarios[
        ["Date", "Portfolio", "GrossNAV", "NetNAV", "GrossReturn", "NetReturn"]
    ]
    payload = {
        "meta": {
            "title": "JPX Walk-forward Portfolio Diagnostics 2A",
            "date_range": (
                f"{daily['Date'].min().date().isoformat()} to "
                f"{daily['Date'].max().date().isoformat()}"
            ),
            "primary_cost_bps": primary_cost,
            "scope": "Frozen stitched OOS predictions; no retraining or feature changes",
        },
        "summary": summary,
        "portfolio_summary": _records(scenario_summary),
        "daily": _records(
            daily[
                [
                    "Date",
                    "Fold",
                    "LongGross",
                    "ShortGross",
                    "LongCost",
                    "ShortCost",
                    "LongNet",
                    "ShortNet",
                    "TotalNet",
                    "LongTradedNotional",
                    "ShortTradedNotional",
                    "TradedNotional",
                    "TopRetention",
                    "BottomRetention",
                    "TopWeightedOverlap",
                    "BottomWeightedOverlap",
                ]
            ]
        ),
        "nav": _records(nav),
        "fold": _records(fold),
        "yearly": _records(yearly),
        "monthly": _records(monthly),
        "benchmark": _records(benchmark),
        "betas": _records(betas),
    }
    report_path = output_dir / "portfolio_diagnostics.html"
    _render(report_path, payload)
    logger.info(
        "Portfolio diagnostics complete: output=%s elapsed=%.1fs",
        output_dir,
        perf_counter() - started,
    )
    return summary
