from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .data import prepare_panel
from .portfolio import daily_portfolio_accounting, summarize_cost_scenario
from .walk_forward import WalkForwardFold, build_walk_forward_folds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    top_n: int = 200
    long_gross: float = 0.5
    short_gross: float = 0.5
    weighting: str = "linear"
    rebalance_every: int = 1
    long_entry: int | None = None
    long_exit: int | None = None
    short_entry: int | None = None
    short_exit: int | None = None
    minimum_holding_days: int = 1
    smoothing_days: int = 1
    weight_inertia: float = 0.0
    no_trade_band: float = 0.0
    beta_neutral: bool = False
    selection_eligible: bool = True


def default_strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec("baseline", "baseline"),
        StrategySpec(
            "long_only_top200", "gross_control", long_gross=1.0, short_gross=0.0,
            selection_eligible=False,
        ),
        StrategySpec(
            "long75_short25", "gross_control", long_gross=0.75, short_gross=0.25,
            selection_eligible=False,
        ),
        StrategySpec("top_bottom_50", "concentration", top_n=50, selection_eligible=False),
        StrategySpec("top_bottom_100", "concentration", top_n=100, selection_eligible=False),
        StrategySpec("top_bottom_400", "concentration", top_n=400, selection_eligible=False),
        StrategySpec("equal_weight", "weighting", weighting="equal"),
        StrategySpec("rebalance_2d", "rebalance", rebalance_every=2),
        StrategySpec("rebalance_5d", "rebalance", rebalance_every=5, selection_eligible=False),
        StrategySpec(
            "buffer_150_250", "buffer",
            long_entry=150, long_exit=250, short_entry=150, short_exit=250,
        ),
        StrategySpec(
            "buffer_100_300", "buffer",
            long_entry=100, long_exit=300, short_entry=100, short_exit=300,
        ),
        StrategySpec("minimum_hold_2d", "minimum_holding", minimum_holding_days=2),
        StrategySpec("minimum_hold_3d", "minimum_holding", minimum_holding_days=3),
        StrategySpec("minimum_hold_5d", "minimum_holding", minimum_holding_days=5),
        StrategySpec(
            "asymmetric_buffer", "asymmetric",
            long_entry=150, long_exit=250, short_entry=100, short_exit=300,
        ),
        StrategySpec("smooth_3d", "smoothing", smoothing_days=3),
        StrategySpec("smooth_5d", "smoothing", smoothing_days=5),
        StrategySpec("turnover_penalty_50", "turnover_penalty", weight_inertia=0.5),
        StrategySpec("no_trade_band_25bp_weight", "no_trade_band", no_trade_band=0.00025),
        StrategySpec("beta_neutral_universe", "beta_control", beta_neutral=True),
        StrategySpec(
            "buffer150_smooth3_band", "combined",
            long_entry=150, long_exit=250, short_entry=150, short_exit=250,
            smoothing_days=3, no_trade_band=0.00025,
        ),
    ]


def _options(config: Config) -> dict[str, Any]:
    value = config.raw.get("strategy_experiments", {})
    if not isinstance(value, dict):
        raise ValueError("strategy_experiments config must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(frame["Date"], errors="raise")
    return frame.loc[dates.between(start, end, inclusive="both")].copy()


def _validation_predictions(
    config: Config,
    panel: pd.DataFrame,
    fold: WalkForwardFold,
    output_dir: Path,
) -> pd.DataFrame:
    target = output_dir / "validation_predictions" / f"{fold.name}.pkl.gz"
    if target.is_file():
        return pd.read_pickle(target)
    model_path = config.output_dir / fold.name / "native_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Frozen Native model not found: {model_path}")
    model = joblib.load(model_path)
    valid = _slice(panel, fold.valid_start, fold.valid_end)
    valid["Prediction"] = model.predict(valid).to_numpy()
    target.parent.mkdir(parents=True, exist_ok=True)
    valid.to_pickle(target, compression="gzip")
    return valid


def _rank_predictions(
    predictions: pd.DataFrame,
    smoothing_days: int,
    warmup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"Date", "SecuritiesCode", "Prediction", "Target"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Strategy input missing columns: {sorted(missing)}")
    current = predictions[list(required)].copy()
    current["Date"] = pd.to_datetime(current["Date"], errors="raise")
    current["_Current"] = True
    if warmup is not None and not warmup.empty and smoothing_days > 1:
        history = warmup[list(required)].copy()
        history["Date"] = pd.to_datetime(history["Date"], errors="raise")
        history["_Current"] = False
        values = pd.concat([history, current], ignore_index=True)
    else:
        values = current
    values = values.sort_values(["SecuritiesCode", "Date"], kind="mergesort")
    if smoothing_days > 1:
        values["Signal"] = values.groupby("SecuritiesCode", sort=False)["Prediction"].transform(
            lambda series: series.rolling(smoothing_days, min_periods=1).mean()
        )
    else:
        values["Signal"] = values["Prediction"]
    values = values.loc[values["_Current"]].copy()
    values["Rank"] = values.groupby("Date", sort=False)["Signal"].rank(
        method="first", ascending=False
    ).astype(np.int64) - 1
    values["DaySize"] = values.groupby("Date", sort=False)["SecuritiesCode"].transform("size")
    values["BottomRank"] = values["DaySize"] - 1 - values["Rank"]
    return values.sort_values(["Date", "Rank"], kind="mergesort").reset_index(drop=True)


def _side_target_weights(
    active: set[int],
    day: pd.DataFrame,
    side: str,
    gross: float,
    weighting: str,
) -> dict[int, float]:
    if gross <= 0 or not active:
        return {}
    rank_column = "Rank" if side == "Long" else "BottomRank"
    ordered = day.loc[day["SecuritiesCode"].isin(active)].sort_values(
        [rank_column, "SecuritiesCode"], kind="mergesort"
    )
    if weighting == "equal":
        raw = np.ones(len(ordered), dtype=float)
    elif weighting == "linear":
        raw = np.linspace(2.0, 1.0, len(ordered), dtype=float)
    else:
        raise ValueError("weighting must be 'linear' or 'equal'")
    sign = 1.0 if side == "Long" else -1.0
    weights = sign * gross * raw / raw.sum()
    return dict(zip(ordered["SecuritiesCode"].astype(int), weights, strict=True))


def _adjust_weights(
    target: dict[int, float],
    previous: dict[int, float],
    *,
    gross: float,
    inertia: float,
    no_trade_band: float,
) -> dict[int, float]:
    if gross <= 0 or not target:
        return {}
    adjusted: dict[int, float] = {}
    for code, target_weight in target.items():
        prior = previous.get(code, 0.0)
        value = inertia * prior + (1.0 - inertia) * target_weight
        if abs(target_weight - prior) <= no_trade_band:
            value = prior
        adjusted[code] = value
    denominator = sum(abs(value) for value in adjusted.values())
    if denominator <= 0:
        return target
    scale = gross / denominator
    return {code: value * scale for code, value in adjusted.items()}


def _rolling_beta_gross(
    history: pd.DataFrame,
    *,
    lookback: int,
    min_periods: int,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    values = history.tail(lookback)
    if len(values) < min_periods:
        return 0.5, 0.5
    variance = float(values["UniverseReturn"].var(ddof=1))
    if variance <= 0 or not math.isfinite(variance):
        return 0.5, 0.5
    long_beta = float(values["LongUnitReturn"].cov(values["UniverseReturn"]) / variance)
    short_beta = float(values["ShortUnitReturn"].cov(values["UniverseReturn"]) / variance)
    denominator = long_beta - short_beta
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.5, 0.5
    long_gross = float(np.clip(-short_beta / denominator, lower, upper))
    return long_gross, 1.0 - long_gross


def construct_stateful_positions(
    predictions: pd.DataFrame,
    spec: StrategySpec,
    *,
    warmup: pd.DataFrame | None = None,
    beta_history: pd.DataFrame | None = None,
    beta_lookback: int = 60,
    beta_min_periods: int = 20,
    beta_gross_bounds: tuple[float, float] = (0.25, 0.75),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = _rank_predictions(predictions, spec.smoothing_days, warmup=warmup)
    long_entry = spec.long_entry or spec.top_n
    long_exit = spec.long_exit or spec.top_n
    short_entry = spec.short_entry or spec.top_n
    short_exit = spec.short_exit or spec.top_n
    if min(long_entry, long_exit, short_entry, short_exit) <= 0:
        raise ValueError("entry and exit ranks must be positive")
    if long_entry > long_exit or short_entry > short_exit:
        raise ValueError("entry rank cannot exceed exit rank")

    long_active: set[int] = set()
    short_active: set[int] = set()
    long_age: dict[int, int] = {}
    short_age: dict[int, int] = {}
    previous_long: dict[int, float] = {}
    previous_short: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    history_rows = [] if beta_history is None else beta_history.to_dict(orient="records")

    for day_number, (date, day) in enumerate(ranked.groupby("Date", sort=True)):
        day = day.copy()
        available = set(day["SecuritiesCode"].astype(int))
        rebalance = day_number % spec.rebalance_every == 0
        rank_by_code = day.set_index("SecuritiesCode")["Rank"].to_dict()
        bottom_by_code = day.set_index("SecuritiesCode")["BottomRank"].to_dict()
        if rebalance:
            long_active = {
                code for code in long_active & available
                if long_age.get(code, 0) < spec.minimum_holding_days
                or rank_by_code[code] < long_exit
            }
            short_active = {
                code for code in short_active & available
                if short_age.get(code, 0) < spec.minimum_holding_days
                or bottom_by_code[code] < short_exit
            }
            long_active.update(
                day.loc[day["Rank"] < long_entry, "SecuritiesCode"].astype(int)
            )
            short_active.update(
                day.loc[day["BottomRank"] < short_entry, "SecuritiesCode"].astype(int)
            )
            overlap = long_active & short_active
            for code in overlap:
                if rank_by_code[code] <= bottom_by_code[code]:
                    short_active.remove(code)
                else:
                    long_active.remove(code)

        for code in long_active:
            long_age[code] = long_age.get(code, 0) + 1
        for code in short_active:
            short_age[code] = short_age.get(code, 0) + 1
        long_age = {code: long_age[code] for code in long_active}
        short_age = {code: short_age[code] for code in short_active}

        long_gross, short_gross = spec.long_gross, spec.short_gross
        history = pd.DataFrame(history_rows)
        if spec.beta_neutral:
            long_gross, short_gross = _rolling_beta_gross(
                history,
                lookback=beta_lookback,
                min_periods=beta_min_periods,
                lower=beta_gross_bounds[0],
                upper=beta_gross_bounds[1],
            )
        if rebalance:
            long_target = _side_target_weights(
                long_active, day, "Long", long_gross, spec.weighting
            )
            short_target = _side_target_weights(
                short_active, day, "Short", short_gross, spec.weighting
            )
            long_weights = _adjust_weights(
                long_target, previous_long, gross=long_gross,
                inertia=spec.weight_inertia, no_trade_band=spec.no_trade_band,
            )
            short_weights = _adjust_weights(
                short_target, previous_short, gross=short_gross,
                inertia=spec.weight_inertia, no_trade_band=spec.no_trade_band,
            )
        else:
            long_weights = previous_long
            short_weights = previous_short

        target_by_code = day.set_index("SecuritiesCode")["Target"].to_dict()
        long_return = 0.0
        short_return = 0.0
        for side, weights in (("Long", long_weights), ("Short", short_weights)):
            for code, weight in weights.items():
                target = float(target_by_code[code])
                contribution = weight * target
                if side == "Long":
                    long_return += contribution
                    side_rank = int(rank_by_code[code])
                else:
                    short_return += contribution
                    side_rank = int(bottom_by_code[code])
                rows.append({
                    "Date": pd.Timestamp(date),
                    "SecuritiesCode": int(code),
                    "Side": side,
                    "SideRank": side_rank,
                    "Weight": float(weight),
                    "Target": target,
                    "Contribution": contribution,
                })
        universe_return = float(day["Target"].mean())
        history_rows.append({
            "Date": pd.Timestamp(date),
            "UniverseReturn": universe_return,
            "LongUnitReturn": long_return / long_gross if long_gross > 0 else 0.0,
            "ShortUnitReturn": short_return / short_gross if short_gross > 0 else 0.0,
        })
        previous_long = long_weights
        previous_short = short_weights

    positions = pd.DataFrame(rows)
    if positions.empty:
        raise ValueError(f"Strategy {spec.name} produced no positions")
    history = pd.DataFrame(history_rows)
    return positions, history


def _break_even_cost_bps(daily: pd.DataFrame) -> float:
    traded = float(daily["TradedNotional"].sum())
    return float(daily["GrossReturn"].sum() / traded * 10_000.0) if traded > 0 else math.nan


def evaluate_positions(
    positions: pd.DataFrame,
    *,
    cost_bps: float,
    annualization_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily = daily_portfolio_accounting(positions)
    scenario, summary = summarize_cost_scenario(
        daily, cost_bps=cost_bps, annualization_days=annualization_days
    )
    summary["break_even_cost_bps"] = _break_even_cost_bps(daily)
    return scenario, summary


def _candidate_row(
    fold: str,
    segment: str,
    spec: StrategySpec,
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    turnover_lambda: float,
    turnover_ratio_limit: float,
    gross_sharpe_tolerance: float,
    required_break_even_bps: float,
) -> dict[str, Any]:
    average_notional = float(summary["average_daily_traded_notional"])
    net_sharpe = float(summary["net_sharpe"])
    row = {
        "Fold": fold,
        "Segment": segment,
        "Candidate": spec.name,
        "Family": spec.family,
        "SelectionEligible": spec.selection_eligible,
        "NetSharpe": net_sharpe,
        "GrossSharpe": float(summary["gross_sharpe"]),
        "NetAnnualizedReturn": float(summary["net_annualized_return"]),
        "GrossAnnualizedReturn": float(summary["gross_annualized_return"]),
        "AverageTradedNotional": average_notional,
        "BreakEvenCostBps": float(summary["break_even_cost_bps"]),
        "NetMaxDrawdown": float(summary["net_max_drawdown"]),
        "LongContribution": float(summary["long_contribution_sum"]),
        "ShortContribution": float(summary["short_contribution_sum"]),
        "Score": net_sharpe - turnover_lambda * average_notional,
    }
    if baseline is None:
        row.update({
            "TurnoverRatioVsBaseline": 1.0,
            "GrossSharpeChangeVsBaseline": 0.0,
            "PassesConstraints": spec.name == "baseline",
        })
    else:
        turnover_ratio = average_notional / float(baseline["average_daily_traded_notional"])
        gross_change = float(summary["gross_sharpe"]) - float(baseline["gross_sharpe"])
        row.update({
            "TurnoverRatioVsBaseline": turnover_ratio,
            "GrossSharpeChangeVsBaseline": gross_change,
            "PassesConstraints": bool(
                spec.selection_eligible
                and float(summary["break_even_cost_bps"]) > required_break_even_bps
                and turnover_ratio <= turnover_ratio_limit
                and gross_change >= -gross_sharpe_tolerance
                and float(summary["gross_sharpe"]) > 0
                and float(summary["net_annualized_return"]) > 0
            ),
        })
    return row


def _render_report(path: Path, payload: dict[str, Any]) -> None:
    template = Path(__file__).with_name("templates") / "strategy_experiments.html"
    path.write_text(
        template.read_text(encoding="utf-8").replace(
            "__STRATEGY_PAYLOAD__",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        ),
        encoding="utf-8",
    )


def run_strategy_experiments(config: Config) -> dict[str, Any]:
    started = perf_counter()
    options = _options(config)
    output_dir = config.output_dir / str(
        options.get("output_subdir", "strategy_experiments")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel(config)
    folds = build_walk_forward_folds(panel, config)
    specs = default_strategy_specs()
    cost_bps = float(options.get("cost_bps", 5))
    annualization_days = int(config.raw["evaluation"].get("annualization_days", 252))
    turnover_lambda = float(options.get("turnover_lambda", 0.5))
    turnover_ratio_limit = float(options.get("turnover_ratio_limit", 0.9))
    gross_sharpe_tolerance = float(options.get("gross_sharpe_tolerance", 0.25))
    required_break_even_bps = float(options.get("required_break_even_bps", cost_bps))
    beta_lookback = int(options.get("beta_lookback", 60))
    beta_min_periods = int(options.get("beta_min_periods", 20))
    beta_bounds = tuple(float(x) for x in options.get("beta_gross_bounds", [0.25, 0.75]))

    validation_rows: list[dict[str, Any]] = []
    fold_inputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    validation_cache: dict[
        tuple[str, str], tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]
    ] = {}
    for fold in folds:
        logger.info("Generating/loading validation predictions for %s", fold.name)
        valid = _validation_predictions(config, panel, fold, output_dir)
        test_path = config.output_dir / fold.name / "native_predictions.pkl.gz"
        test = pd.read_pickle(test_path)
        fold_inputs[fold.name] = (valid, test)
        baseline_summary: dict[str, Any] | None = None
        for spec in specs:
            positions, history = construct_stateful_positions(
                valid, spec,
                beta_lookback=beta_lookback,
                beta_min_periods=beta_min_periods,
                beta_gross_bounds=beta_bounds,
            )
            _, summary = evaluate_positions(
                positions, cost_bps=cost_bps, annualization_days=annualization_days
            )
            if spec.name == "baseline":
                baseline_summary = summary
            validation_cache[(fold.name, spec.name)] = (positions, history, summary)
        assert baseline_summary is not None
        for spec in specs:
            summary = validation_cache[(fold.name, spec.name)][2]
            validation_rows.append(_candidate_row(
                fold.name, "validation", spec, summary, baseline_summary,
                turnover_lambda=turnover_lambda,
                turnover_ratio_limit=turnover_ratio_limit,
                gross_sharpe_tolerance=gross_sharpe_tolerance,
                required_break_even_bps=required_break_even_bps,
            ))

    validation_table = pd.DataFrame(validation_rows)
    selected_by_fold: dict[str, StrategySpec] = {}
    selection_rows: list[dict[str, Any]] = []
    specs_by_name = {spec.name: spec for spec in specs}
    for fold in folds:
        rows = validation_table.loc[validation_table["Fold"].eq(fold.name)].copy()
        feasible = rows.loc[rows["PassesConstraints"]]
        pool = feasible if not feasible.empty else rows.loc[
            rows["SelectionEligible"] & rows["Candidate"].eq("baseline")
        ]
        selected = pool.sort_values(
            ["Score", "NetSharpe", "AverageTradedNotional"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        selected_by_fold[fold.name] = specs_by_name[str(selected["Candidate"])]
        selection_rows.append({
            **selected.to_dict(),
            "FeasibleCandidateCount": int(len(feasible)),
            "UsedBaselineFallback": bool(feasible.empty),
        })

    fixed_test_positions: dict[str, list[pd.DataFrame]] = {spec.name: [] for spec in specs}
    nested_positions: list[pd.DataFrame] = []
    fold_test_rows: list[dict[str, Any]] = []
    for fold in folds:
        valid, test = fold_inputs[fold.name]
        baseline_test_summary: dict[str, Any] | None = None
        per_spec: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
        for spec in specs:
            _, validation_history, _ = validation_cache[(fold.name, spec.name)]
            positions, _ = construct_stateful_positions(
                test, spec, warmup=valid, beta_history=validation_history,
                beta_lookback=beta_lookback, beta_min_periods=beta_min_periods,
                beta_gross_bounds=beta_bounds,
            )
            positions["Fold"] = fold.name
            fixed_test_positions[spec.name].append(positions)
            _, summary = evaluate_positions(
                positions, cost_bps=cost_bps, annualization_days=annualization_days
            )
            per_spec[spec.name] = (positions, summary)
            if spec.name == "baseline":
                baseline_test_summary = summary
        assert baseline_test_summary is not None
        for spec in specs:
            fold_test_rows.append(_candidate_row(
                fold.name, "test", spec, per_spec[spec.name][1], baseline_test_summary,
                turnover_lambda=turnover_lambda,
                turnover_ratio_limit=turnover_ratio_limit,
                gross_sharpe_tolerance=gross_sharpe_tolerance,
                required_break_even_bps=required_break_even_bps,
            ))
        nested_positions.append(per_spec[selected_by_fold[fold.name].name][0])

    fixed_rows: list[dict[str, Any]] = []
    fixed_daily: dict[str, pd.DataFrame] = {}
    for spec in specs:
        positions = pd.concat(fixed_test_positions[spec.name], ignore_index=True)
        daily, summary = evaluate_positions(
            positions, cost_bps=cost_bps, annualization_days=annualization_days
        )
        fixed_daily[spec.name] = daily
        fixed_rows.append(_candidate_row(
            "stitched", "test", spec, summary, None,
            turnover_lambda=turnover_lambda,
            turnover_ratio_limit=turnover_ratio_limit,
            gross_sharpe_tolerance=gross_sharpe_tolerance,
            required_break_even_bps=required_break_even_bps,
        ))

    nested = pd.concat(nested_positions, ignore_index=True)
    nested_daily, nested_summary = evaluate_positions(
        nested, cost_bps=cost_bps, annualization_days=annualization_days
    )
    baseline_fixed = next(row for row in fixed_rows if row["Candidate"] == "baseline")
    nested_row = _candidate_row(
        "stitched", "nested_test", StrategySpec("nested_selected", "nested"),
        nested_summary, {
            "average_daily_traded_notional": baseline_fixed["AverageTradedNotional"],
            "gross_sharpe": baseline_fixed["GrossSharpe"],
        },
        turnover_lambda=turnover_lambda,
        turnover_ratio_limit=turnover_ratio_limit,
        gross_sharpe_tolerance=gross_sharpe_tolerance,
        required_break_even_bps=required_break_even_bps,
    )

    validation_table.to_csv(output_dir / "candidate_validation.csv", index=False)
    pd.DataFrame(fold_test_rows).to_csv(output_dir / "candidate_test_by_fold.csv", index=False)
    pd.DataFrame(fixed_rows).to_csv(output_dir / "candidate_test_stitched.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "nested_fold_selection.csv", index=False)
    nested_daily.to_csv(output_dir / "nested_test_daily.csv", index=False)
    nested.to_pickle(output_dir / "nested_test_positions.pkl.gz", compression="gzip")
    pd.DataFrame([asdict(spec) for spec in specs]).to_csv(
        output_dir / "candidate_definitions.csv", index=False
    )
    shutil.copy2(config.source_path, output_dir / "walk_forward.yaml")

    summary = {
        "run_role": "nested_walk_forward_portfolio_experiments",
        "model_retrained": False,
        "validation_predictions_regenerated_from_frozen_models": True,
        "test_predictions": "existing frozen per-fold Native predictions",
        "selection_target": f"{cost_bps:g} bps net Sharpe minus turnover penalty",
        "selection_constraints": {
            "break_even_cost_bps_above": required_break_even_bps,
            "turnover_ratio_at_most": turnover_ratio_limit,
            "gross_sharpe_decline_at_most": gross_sharpe_tolerance,
            "gross_sharpe_positive": True,
            "net_annualized_return_positive": True,
        },
        "turnover_lambda": turnover_lambda,
        "cost_bps": cost_bps,
        "fold_selections": selection_rows,
        "fixed_candidate_test": fixed_rows,
        "nested_test": nested_row,
        "source_model_sha256": {
            fold.name: _sha256(config.output_dir / fold.name / "native_model.joblib")
            for fold in folds
        },
        "elapsed_seconds": perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    payload = {
        "meta": {
            "title": "JPX Controlled Portfolio Experiments",
            "subtitle": "Frozen predictions, nested walk-forward parameter selection",
            "cost": f"{cost_bps:g} bps one-way on actual traded notional",
        },
        "nested": nested_row,
        "selections": selection_rows,
        "fixed": fixed_rows,
        "validation": validation_rows,
        "daily": {
            "nested": nested_daily[["Date", "GrossNAV", "NetNAV", "TradedNotional"]]
            .assign(Date=lambda frame: frame["Date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records"),
            "baseline": fixed_daily["baseline"][
                ["Date", "GrossNAV", "NetNAV", "TradedNotional"]
            ].assign(Date=lambda frame: frame["Date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records"),
        },
    }
    _render_report(output_dir / "strategy_experiments.html", payload)
    logger.info(
        "Strategy experiments complete: nested net Sharpe=%.4f in %.1fs",
        nested_row["NetSharpe"], summary["elapsed_seconds"],
    )
    return summary
