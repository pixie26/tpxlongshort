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
from .ranking import add_rank

logger = logging.getLogger(__name__)


def _portfolio_options(config: Config) -> dict[str, Any]:
    options = config.raw.get("portfolio")
    if not isinstance(options, dict):
        raise ValueError("portfolio config must be a mapping")
    return options


def _resolve_input_path(config: Config, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config.output_dir / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def construct_portfolio_positions(
    predictions: pd.DataFrame,
    *,
    top_n: int = 200,
    long_gross: float = 0.5,
    short_gross: float = 0.5,
    ranking_mode: str = "corrected_rank",
) -> pd.DataFrame:
    required = {"Date", "SecuritiesCode", "Prediction", "Target"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Portfolio input missing columns: {sorted(missing)}")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if long_gross <= 0 or short_gross <= 0:
        raise ValueError("long_gross and short_gross must be positive")

    values = predictions[list(required)].copy()
    values["Date"] = pd.to_datetime(values["Date"], errors="raise")
    if values.duplicated(["Date", "SecuritiesCode"]).any():
        raise ValueError("Portfolio input contains duplicate Date/SecuritiesCode keys")
    if values[["Prediction", "Target"]].isna().any().any():
        raise ValueError("Portfolio input contains missing Prediction or Target values")

    ranked = add_rank(values, mode=ranking_mode)
    day_size = ranked.groupby("Date", sort=False)["SecuritiesCode"].transform("size")
    too_small = ranked.loc[day_size < 2 * top_n, "Date"].drop_duplicates()
    if not too_small.empty:
        raise ValueError(
            f"Portfolio requires at least {2 * top_n} securities per day; "
            f"first short day={too_small.iloc[0].date()}"
        )

    selected = ranked.loc[
        (ranked["Rank"] < top_n) | (ranked["Rank"] >= day_size - top_n)
    ].copy()
    selected["Side"] = np.where(selected["Rank"] < top_n, "Long", "Short")
    selected["SideRank"] = np.where(
        selected["Side"].eq("Long"),
        selected["Rank"],
        day_size.loc[selected.index] - 1 - selected["Rank"],
    ).astype(np.int64)

    # Preserve the competition's 2-to-1 rank-weight shape, but normalize each
    # side independently to economically meaningful portfolio gross exposure.
    selected["RawWeight"] = 2.0 - selected["SideRank"] / max(top_n - 1, 1)
    side_sum = selected.groupby(["Date", "Side"])["RawWeight"].transform("sum")
    side_target = np.where(selected["Side"].eq("Long"), long_gross, short_gross)
    selected["Weight"] = selected["RawWeight"] / side_sum * side_target
    selected.loc[selected["Side"].eq("Short"), "Weight"] *= -1.0
    selected["Contribution"] = selected["Weight"] * selected["Target"]
    selected = selected.sort_values(
        ["Date", "Side", "SideRank", "SecuritiesCode"],
        kind="mergesort",
    ).reset_index(drop=True)

    by_day = selected.groupby("Date", sort=True)
    expected_names = 2 * top_n
    if not by_day.size().eq(expected_names).all():
        raise AssertionError("Portfolio did not select the expected number of names")
    long_weight = selected.loc[selected["Side"].eq("Long")].groupby("Date")["Weight"].sum()
    short_weight = selected.loc[selected["Side"].eq("Short")].groupby("Date")["Weight"].sum()
    if not np.allclose(long_weight, long_gross, atol=1e-12):
        raise AssertionError("Long weights do not sum to the configured gross exposure")
    if not np.allclose(short_weight, -short_gross, atol=1e-12):
        raise AssertionError("Short weights do not sum to the configured gross exposure")
    return selected


def daily_portfolio_accounting(positions: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "SecuritiesCode", "Side", "Weight", "Contribution"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(f"Position input missing columns: {sorted(missing)}")

    ordered = positions.sort_values(["Date", "SecuritiesCode"], kind="mergesort").copy()
    dates = pd.DatetimeIndex(ordered["Date"].drop_duplicates().sort_values())
    previous = ordered[["Date", "SecuritiesCode", "Weight"]].copy()
    previous_date = pd.Series(dates[1:].to_numpy(), index=dates[:-1])
    previous["Date"] = previous["Date"].map(previous_date)
    previous = previous.dropna(subset=["Date"]).rename(columns={"Weight": "PreviousWeight"})
    changes = ordered[["Date", "SecuritiesCode", "Weight"]].merge(
        previous,
        on=["Date", "SecuritiesCode"],
        how="outer",
        validate="one_to_one",
    ).fillna({"Weight": 0.0, "PreviousWeight": 0.0})
    changes["AbsChange"] = (changes["Weight"] - changes["PreviousWeight"]).abs()
    traded_notional = changes.groupby("Date", sort=True)["AbsChange"].sum()
    half_turnover = 0.5 * traded_notional

    grouped = ordered.groupby("Date", sort=True)
    daily = grouped.agg(
        GrossReturn=("Contribution", "sum"),
        GrossExposure=("Weight", lambda value: float(value.abs().sum())),
        NetExposure=("Weight", "sum"),
        Names=("SecuritiesCode", "size"),
    )
    daily["LongContribution"] = (
        ordered.loc[ordered["Side"].eq("Long")]
        .groupby("Date")["Contribution"]
        .sum()
    )
    daily["ShortContribution"] = (
        ordered.loc[ordered["Side"].eq("Short")]
        .groupby("Date")["Contribution"]
        .sum()
    )
    daily["TradedNotional"] = traded_notional
    daily["HalfTurnover"] = half_turnover
    # Backward-compatible alias. Economically this is half-turnover, not the
    # dollar notional to which a one-way execution cost should be applied.
    daily["Turnover"] = daily["HalfTurnover"]
    if not np.allclose(
        daily["GrossReturn"],
        daily["LongContribution"] + daily["ShortContribution"],
        atol=1e-15,
    ):
        raise AssertionError("Gross return does not equal long plus short contribution")
    if not np.allclose(
        daily["TradedNotional"],
        2.0 * daily["HalfTurnover"],
        atol=1e-15,
    ):
        raise AssertionError("Traded notional does not equal two times half-turnover")
    return daily.reset_index()


def _nav_and_drawdown(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    if (returns <= -1.0).any():
        raise ValueError("Daily portfolio return at or below -100% cannot be compounded")
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return nav, drawdown


def _annualized_return(nav: pd.Series, days: int, annualization_days: int) -> float:
    ending = float(nav.iloc[-1])
    years = days / annualization_days
    return ending ** (1.0 / years) - 1.0 if ending > 0 and years > 0 else math.nan


def summarize_cost_scenario(
    daily: pd.DataFrame,
    *,
    cost_bps: float,
    annualization_days: int = 252,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = daily.copy()
    result["CostBps"] = float(cost_bps)
    # cost_bps is a one-way charge per dollar actually traded. Buying and later
    # selling one dollar therefore costs twice cost_bps over the round trip.
    result["TradingCost"] = result["TradedNotional"] * float(cost_bps) / 10_000.0
    result["NetReturn"] = result["GrossReturn"] - result["TradingCost"]
    if not np.allclose(
        result["NetReturn"] + result["TradingCost"],
        result["GrossReturn"],
        atol=1e-15,
    ):
        raise AssertionError("Net return plus trading cost does not equal gross return")
    gross_nav, gross_drawdown = _nav_and_drawdown(result["GrossReturn"])
    net_nav, net_drawdown = _nav_and_drawdown(result["NetReturn"])
    result["GrossNAV"] = gross_nav
    result["NetNAV"] = net_nav
    result["GrossDrawdown"] = gross_drawdown
    result["NetDrawdown"] = net_drawdown
    result["CumulativeLongContribution"] = result["LongContribution"].cumsum()
    result["CumulativeShortContribution"] = result["ShortContribution"].cumsum()

    gross_std = float(result["GrossReturn"].std(ddof=1))
    net_std = float(result["NetReturn"].std(ddof=1))
    days = len(result)
    summary = {
        "cost_bps": float(cost_bps),
        "start_date": result["Date"].min().date().isoformat(),
        "end_date": result["Date"].max().date().isoformat(),
        "trading_days": days,
        "gross_total_return": float(gross_nav.iloc[-1] - 1.0),
        "net_total_return": float(net_nav.iloc[-1] - 1.0),
        "gross_annualized_return": _annualized_return(
            gross_nav, days, annualization_days
        ),
        "net_annualized_return": _annualized_return(
            net_nav, days, annualization_days
        ),
        "gross_annualized_volatility": gross_std * math.sqrt(annualization_days),
        "net_annualized_volatility": net_std * math.sqrt(annualization_days),
        "gross_sharpe": (
            float(result["GrossReturn"].mean()) / gross_std * math.sqrt(annualization_days)
            if gross_std > 0 else math.nan
        ),
        "net_sharpe": (
            float(result["NetReturn"].mean()) / net_std * math.sqrt(annualization_days)
            if net_std > 0 else math.nan
        ),
        "gross_max_drawdown": float(gross_drawdown.min()),
        "net_max_drawdown": float(net_drawdown.min()),
        "long_contribution_sum": float(result["LongContribution"].sum()),
        "short_contribution_sum": float(result["ShortContribution"].sum()),
        "average_daily_turnover": float(result["HalfTurnover"].mean()),
        "average_daily_half_turnover": float(result["HalfTurnover"].mean()),
        "average_daily_traded_notional": float(result["TradedNotional"].mean()),
        "average_rebalance_half_turnover": float(result["HalfTurnover"].iloc[1:].mean()),
        "average_rebalance_traded_notional": float(result["TradedNotional"].iloc[1:].mean()),
        "annualized_cost_drag": float(result["TradingCost"].mean() * annualization_days),
        "average_gross_exposure": float(result["GrossExposure"].mean()),
        "average_net_exposure": float(result["NetExposure"].mean()),
        "average_names": float(result["Names"].mean()),
        "gross_ending_nav": float(gross_nav.iloc[-1]),
        "net_ending_nav": float(net_nav.iloc[-1]),
    }
    return result, summary


def _yearly_summary(scenario: pd.DataFrame, annualization_days: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, frame in scenario.groupby(scenario["Date"].dt.year, sort=True):
        gross_nav, _ = _nav_and_drawdown(frame["GrossReturn"])
        net_nav, net_drawdown = _nav_and_drawdown(frame["NetReturn"])
        std = float(frame["NetReturn"].std(ddof=1))
        rows.append({
            "year": int(year),
            "days": int(len(frame)),
            "gross_return": float(gross_nav.iloc[-1] - 1.0),
            "net_return": float(net_nav.iloc[-1] - 1.0),
            "net_sharpe": (
                float(frame["NetReturn"].mean()) / std * math.sqrt(annualization_days)
                if std > 0 else math.nan
            ),
            "net_max_drawdown": float(net_drawdown.min()),
            "long_contribution_sum": float(frame["LongContribution"].sum()),
            "short_contribution_sum": float(frame["ShortContribution"].sum()),
            "average_turnover": float(frame["HalfTurnover"].mean()),
            "average_half_turnover": float(frame["HalfTurnover"].mean()),
            "average_traded_notional": float(frame["TradedNotional"].mean()),
        })
    return rows


def _json_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    values = frame[columns].copy()
    values["Date"] = values["Date"].dt.strftime("%Y-%m-%d")
    return values.to_dict(orient="records")


def _render_report(output_path: Path, payload: dict[str, Any]) -> None:
    template_path = Path(__file__).with_name("templates") / "portfolio_report.html"
    template = template_path.read_text(encoding="utf-8")
    output_path.write_text(
        template.replace(
            "__PORTFOLIO_PAYLOAD__",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        ),
        encoding="utf-8",
    )


def run_portfolio_backtest(config: Config) -> dict[str, Any]:
    started = perf_counter()
    options = _portfolio_options(config)
    predictions_path = _resolve_input_path(
        config,
        str(options.get("predictions_file", "native_stitched_oos_predictions.pkl.gz")),
    )
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Stitched OOS predictions not found: {predictions_path}")

    top_n = int(options.get("top_n", 200))
    long_gross = float(options.get("long_gross", 0.5))
    short_gross = float(options.get("short_gross", 0.5))
    costs = sorted({float(value) for value in options.get("cost_bps", [0, 5, 10, 20])})
    primary_cost = float(options.get("primary_cost_bps", 5))
    if primary_cost not in costs:
        raise ValueError("portfolio.primary_cost_bps must be included in portfolio.cost_bps")
    annualization_days = int(config.raw["evaluation"].get("annualization_days", 252))
    ranking_mode = str(config.raw["ranking"]["mode"])

    logger.info("Loading frozen stitched OOS predictions: %s", predictions_path)
    predictions = pd.read_pickle(predictions_path)
    logger.info(
        "Predictions loaded: rows=%s days=%s range=%s..%s",
        f"{len(predictions):,}",
        predictions["Date"].nunique(),
        pd.Timestamp(predictions["Date"].min()).date(),
        pd.Timestamp(predictions["Date"].max()).date(),
    )
    positions = construct_portfolio_positions(
        predictions,
        top_n=top_n,
        long_gross=long_gross,
        short_gross=short_gross,
        ranking_mode=ranking_mode,
    )
    daily = daily_portfolio_accounting(positions)

    scenario_frames: dict[float, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    for cost_bps in costs:
        scenario, summary = summarize_cost_scenario(
            daily,
            cost_bps=cost_bps,
            annualization_days=annualization_days,
        )
        scenario_frames[cost_bps] = scenario
        summaries.append(summary)
        logger.info(
            "Portfolio cost=%.1f bps: annualized_return=%.4f Sharpe=%.4f max_drawdown=%.4f",
            cost_bps,
            summary["net_annualized_return"],
            summary["net_sharpe"],
            summary["net_max_drawdown"],
        )

    output_dir = config.output_dir / str(options.get("output_subdir", "portfolio_backtest"))
    output_dir.mkdir(parents=True, exist_ok=True)
    positions.to_pickle(output_dir / "portfolio_positions.pkl.gz", compression="gzip")

    daily_output = daily.copy()
    for cost_bps, scenario in scenario_frames.items():
        suffix = f"{cost_bps:g}bps"
        for column in ("TradingCost", "NetReturn", "GrossNAV", "NetNAV", "NetDrawdown"):
            daily_output[f"{column}_{suffix}"] = scenario[column].to_numpy()
    daily_output.to_csv(output_dir / "portfolio_daily.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "portfolio_summary.csv", index=False)

    primary = scenario_frames[primary_cost]
    yearly = _yearly_summary(primary, annualization_days)
    primary_summary = next(item for item in summaries if item["cost_bps"] == primary_cost)
    report_summary = {
        "run_role": "stitched_oos_portfolio_backtest",
        "evaluation_scope": "stitched_chronological_oos",
        "prediction_scope": "out_of_sample",
        "test_target_used_for_training": False,
        "test_target_used_for_evaluation": True,
        "return_definition": (
            "JPX Target on prediction date t: adjusted close return from t+1 to t+2"
        ),
        "position_timing": (
            "weights selected from prediction date t are established at t+1 close "
            "and held until t+2 close"
        ),
        "weighting": "rank_linear_2_to_1_normalized_separately_by_side",
        "top_n_per_side": top_n,
        "long_gross_target": long_gross,
        "short_gross_target": short_gross,
        "net_exposure_target": long_gross - short_gross,
        "gross_exposure_target": long_gross + short_gross,
        "half_turnover_definition": "0.5 * sum(abs(weight_t - weight_t_minus_1))",
        "traded_notional_definition": "sum(abs(weight_t - weight_t_minus_1))",
        "cost_definition": "traded_notional * one_way_cost_bps_per_dollar / 10000",
        "cost_bps_interpretation": (
            "one-way cost per dollar traded; a buy then sell round trip costs twice the bps"
        ),
        "first_day_entry_cost_charged": True,
        "primary_cost_bps": primary_cost,
        "source_predictions": str(predictions_path),
        "source_predictions_sha256": _sha256(predictions_path),
        "source_rows": int(len(predictions)),
        "source_days": int(predictions["Date"].nunique()),
        "scenarios": summaries,
        "primary": primary_summary,
        "yearly_primary_cost": yearly,
    }
    write_report(report_summary, output_dir / "portfolio_metrics.json")
    shutil.copy2(config.source_path, output_dir / "walk_forward.yaml")

    payload = {
        "meta": {
            "title": "JPX Stitched OOS 市场中性组合回测",
            "source": str(predictions_path),
            "date_range": f"{primary_summary['start_date']} 至 {primary_summary['end_date']}",
            "portfolio": (
                f"每日 Top {top_n} 做多 / Bottom {top_n} 做空；"
                f"多头 +{long_gross:.0%}，空头 -{short_gross:.0%}，"
                f"总敞口 {long_gross + short_gross:.0%}，净敞口 {long_gross - short_gross:.0%}"
            ),
            "weighting": "各侧使用 2→1 线性排名权重，并分别归一化。",
            "cost": (
                "bps 表示每元实际成交额的单程成本；成本 = Σ|Δw| × bps。"
                "首日前持仓视为 0，因此收取首日建仓成本；完整买入再卖出的双程成本为 2×bps。"
            ),
            "warning": (
                "这是固定 stitched OOS 预测的经济回测，不是新的模型实验。"
                "尚未建模滑点、涨跌停、停牌、借券可得性、容量和成交冲击。"
            ),
            "primary_cost_bps": primary_cost,
        },
        "summary": summaries,
        "yearly": yearly,
        "daily": _json_records(
            pd.concat(
                [
                    scenario.assign(CostBps=cost_bps)
                    for cost_bps, scenario in scenario_frames.items()
                ],
                ignore_index=True,
            ),
            [
                "Date", "CostBps", "GrossReturn", "NetReturn", "GrossNAV", "NetNAV",
                "NetDrawdown", "LongContribution", "ShortContribution",
                "CumulativeLongContribution", "CumulativeShortContribution",
                "HalfTurnover", "TradedNotional", "Turnover",
                "GrossExposure", "NetExposure",
            ],
        ),
    }
    _render_report(output_dir / "report.html", payload)
    logger.info(
        "Portfolio backtest complete: output=%s elapsed=%.1fs",
        output_dir,
        perf_counter() - started,
    )
    return report_summary
