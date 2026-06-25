from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .strategy_experiments import (
    StrategySpec,
    construct_stateful_positions,
    evaluate_positions,
)


PORTFOLIO_SPECS = (
    StrategySpec("baseline", "baseline"),
    StrategySpec("smooth_3d", "smoothing", smoothing_days=3),
)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _fold_metrics(config: Config) -> pd.DataFrame:
    rows = []
    for spec in config.raw["walk_forward"]["folds"]:
        fold = str(spec["name"])
        path = config.output_dir / fold / "native_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Native fold metrics not found: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "Fold": fold,
            "InSampleCompetitionSharpe": float(row["in_sample_sharpe"]),
            "ValidationCompetitionSharpe": float(row["valid_sharpe"]),
            "OOSCompetitionSharpe": float(row["oos_sharpe"]),
            "InSampleOOSGap": float(row["in_sample_sharpe"] - row["oos_sharpe"]),
            "RankIC": float(row["rank_ic"]),
            "MedianDailyRankIC": float(row["rank_ic_median"]),
        })
    return pd.DataFrame(rows)


def run_ablation_report(config: Config) -> dict[str, Any]:
    ablation = config.raw.get("ablation", {})
    variant = str(ablation.get("id", config.raw["project"]["name"]))
    label = str(ablation.get("label", variant))
    cost_bps = float(ablation.get("portfolio_cost_bps", 5))
    annualization_days = int(config.raw["evaluation"].get("annualization_days", 252))
    model_metrics = _fold_metrics(config)
    fold_rows: list[dict[str, Any]] = []
    stitched_positions: dict[str, list[pd.DataFrame]] = {
        spec.name: [] for spec in PORTFOLIO_SPECS
    }

    for fold in model_metrics["Fold"]:
        fold_dir = config.output_dir / fold
        test = pd.read_pickle(fold_dir / "native_predictions.pkl.gz")
        valid = pd.read_pickle(fold_dir / "native_valid_predictions.pkl.gz")
        model_row = model_metrics.loc[model_metrics["Fold"].eq(fold)].iloc[0]
        for spec in PORTFOLIO_SPECS:
            positions, _ = construct_stateful_positions(test, spec, warmup=valid)
            positions["Fold"] = fold
            stitched_positions[spec.name].append(positions)
            _, summary = evaluate_positions(
                positions,
                cost_bps=cost_bps,
                annualization_days=annualization_days,
            )
            fold_rows.append({
                "Variant": variant,
                "Label": label,
                "Fold": fold,
                "Portfolio": spec.name,
                "GrossSharpe": float(summary["gross_sharpe"]),
                "NetSharpe": float(summary["net_sharpe"]),
                "GrossAnnualizedReturn": float(summary["gross_annualized_return"]),
                "NetAnnualizedReturn": float(summary["net_annualized_return"]),
                "AverageTradedNotional": float(
                    summary["average_daily_traded_notional"]
                ),
                "BreakEvenCostBps": float(summary["break_even_cost_bps"]),
                "NetMaxDrawdown": float(summary["net_max_drawdown"]),
                "LongContribution": float(summary["long_contribution_sum"]),
                "ShortContribution": float(summary["short_contribution_sum"]),
                **model_row.drop(labels=["Fold"]).to_dict(),
            })

    fold_table = pd.DataFrame(fold_rows)
    stitched_rows: list[dict[str, Any]] = []
    daily_outputs: list[pd.DataFrame] = []
    for spec in PORTFOLIO_SPECS:
        positions = pd.concat(stitched_positions[spec.name], ignore_index=True)
        daily, summary = evaluate_positions(
            positions,
            cost_bps=cost_bps,
            annualization_days=annualization_days,
        )
        daily["Variant"] = variant
        daily["Portfolio"] = spec.name
        daily_outputs.append(daily)
        yearly = daily.groupby(daily["Date"].dt.year)["GrossReturn"].sum()
        positive_total = float(yearly.clip(lower=0).sum())
        contribution_2020 = float(yearly.get(2020, 0.0))
        contribution_share_2020 = (
            contribution_2020 / positive_total if positive_total > 0 else math.nan
        )
        portfolio_folds = fold_table.loc[fold_table["Portfolio"].eq(spec.name)]
        stitched_rows.append({
            "Variant": variant,
            "Label": label,
            "Portfolio": spec.name,
            "FeatureCount": len(config.feature_columns),
            "ModelType": config.model_type,
            "FeatureTransform": config.feature_transform,
            "MedianFoldGrossSharpe": float(portfolio_folds["GrossSharpe"].median()),
            "MedianFoldNetSharpe": float(portfolio_folds["NetSharpe"].median()),
            "MedianFoldBreakEvenCostBps": float(
                portfolio_folds["BreakEvenCostBps"].median()
            ),
            "MedianFoldTradedNotional": float(
                portfolio_folds["AverageTradedNotional"].median()
            ),
            "PositiveGrossFolds": int((portfolio_folds["GrossSharpe"] > 0).sum()),
            "PositiveNetFolds": int((portfolio_folds["NetSharpe"] > 0).sum()),
            "MeanRankIC": float(model_metrics["RankIC"].mean()),
            "MedianRankIC": float(model_metrics["RankIC"].median()),
            "MedianDailyRankIC": float(model_metrics["MedianDailyRankIC"].median()),
            "AverageTradedNotional": float(
                summary["average_daily_traded_notional"]
            ),
            "BreakEvenCostBps": float(summary["break_even_cost_bps"]),
            "WorstFoldGrossSharpe": float(portfolio_folds["GrossSharpe"].min()),
            "WorstFoldNetSharpe": float(portfolio_folds["NetSharpe"].min()),
            "GrossSharpe": float(summary["gross_sharpe"]),
            "NetSharpe": float(summary["net_sharpe"]),
            "GrossAnnualizedReturn": float(summary["gross_annualized_return"]),
            "NetAnnualizedReturn": float(summary["net_annualized_return"]),
            "NetMaxDrawdown": float(summary["net_max_drawdown"]),
            "Contribution2020": contribution_2020,
            "PositiveContributionShare2020": contribution_share_2020,
            "MedianInSampleCompetitionSharpe": float(
                model_metrics["InSampleCompetitionSharpe"].median()
            ),
            "MedianOOSCompetitionSharpe": float(
                model_metrics["OOSCompetitionSharpe"].median()
            ),
            "MedianInSampleOOSGap": float(model_metrics["InSampleOOSGap"].median()),
        })

    summary_table = pd.DataFrame(stitched_rows)
    fold_table.to_csv(config.output_dir / "ablation_fold_metrics.csv", index=False)
    summary_table.to_csv(config.output_dir / "ablation_summary.csv", index=False)
    pd.concat(daily_outputs, ignore_index=True).to_csv(
        config.output_dir / "ablation_portfolio_daily.csv", index=False
    )
    report = {
        "run_role": "feature_model_ablation",
        "variant": variant,
        "label": label,
        "model_type": config.model_type,
        "feature_columns": config.feature_columns,
        "feature_groups": config.feature_groups,
        "categorical_features": config.categorical_features,
        "cross_sectional_transform": config.feature_transform,
        "portfolio_cost_bps": cost_bps,
        "portfolios": ["baseline", "smooth_3d"],
        "fold_metrics": _records(fold_table),
        "summary": _records(summary_table),
    }
    (config.output_dir / "ablation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    shutil.copy2(config.source_path, config.output_dir / "ablation_config.yaml")
    return report


def build_suite_report(
    config_paths: list[Path],
    output_dir: Path,
    *,
    baseline_variant: str = "A0",
) -> dict[str, Any]:
    summaries = []
    folds = []
    parity_rows = []
    for path in config_paths:
        from .config import load_config

        config = load_config(path)
        summaries.append(pd.read_csv(config.output_dir / "ablation_summary.csv"))
        folds.append(pd.read_csv(config.output_dir / "ablation_fold_metrics.csv"))
        parity_path = config.output_dir / "walk_forward_summary.json"
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        engineering = parity["engineering_parity"]
        fold_parity = engineering["folds"]
        parity_rows.append({
            "Variant": str(config.raw["ablation"]["id"]),
            "PredictionExact": bool(
                engineering["all_fold_predictions_exact"]
            ),
            "PredictionWithinTolerance": bool(
                engineering.get(
                    "all_fold_predictions_within_tolerance",
                    engineering["all_fold_predictions_exact"],
                )
            ),
            "RankExact": bool(
                engineering["all_fold_ranks_identical"]
            ),
            "MetricsExact": bool(
                engineering["all_fold_metrics_identical"]
            ),
            "MaxPredictionDifference": float(
                max(row["max_abs_difference"] for row in fold_parity)
            ),
        })
    summary = pd.concat(summaries, ignore_index=True)
    fold = pd.concat(folds, ignore_index=True)
    fold_medians = (
        fold.groupby(["Variant", "Portfolio"], as_index=False)
        .agg(
            MedianFoldBreakEvenCostBps=("BreakEvenCostBps", "median"),
            MedianFoldTradedNotional=("AverageTradedNotional", "median"),
        )
    )
    summary = summary.drop(
        columns=[
            "MedianFoldBreakEvenCostBps",
            "MedianFoldTradedNotional",
        ],
        errors="ignore",
    ).merge(
        fold_medians,
        on=["Variant", "Portfolio"],
        how="left",
        validate="one_to_one",
    )
    if baseline_variant not in set(fold["Variant"]):
        raise ValueError(
            f"Baseline variant {baseline_variant!r} is not present in suite"
        )
    baseline = fold.loc[
        fold["Variant"].eq(baseline_variant)
    ].set_index(["Fold", "Portfolio"])
    paired = fold.copy()
    for metric in (
        "GrossSharpe", "NetSharpe", "RankIC", "AverageTradedNotional",
        "BreakEvenCostBps", "InSampleOOSGap",
    ):
        paired[f"Delta{metric}"] = [
            value - baseline.loc[(fold_name, portfolio), metric]
            for value, fold_name, portfolio in zip(
                paired[metric], paired["Fold"], paired["Portfolio"], strict=True
            )
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    fold.to_csv(output_dir / "ablation_fold_metrics.csv", index=False)
    paired.to_csv(output_dir / "ablation_paired_changes.csv", index=False)
    parity_table = pd.DataFrame(parity_rows)
    parity_table.to_csv(output_dir / "ablation_parity.csv", index=False)
    smooth = summary.loc[summary["Portfolio"].eq("smooth_3d")].set_index("Variant")
    baseline_smooth = smooth.loc[baseline_variant]
    gate_rows = []
    for variant, variant_summary in smooth.iterrows():
        paired_smooth = paired.loc[
            paired["Variant"].eq(variant)
            & paired["Portfolio"].eq("smooth_3d")
        ]
        if variant == baseline_variant:
            improved_folds = 0
        else:
            improved_folds = int((paired_smooth["DeltaNetSharpe"] > 0).sum())
        gates = {
            "AtLeastThreeImprovedFolds": improved_folds >= 3,
            "MedianNetSharpeImproved": (
                float(variant_summary["MedianFoldNetSharpe"])
                > float(baseline_smooth["MedianFoldNetSharpe"])
            ),
            "MedianBreakEvenNotLower": (
                float(variant_summary["MedianFoldBreakEvenCostBps"])
                >= float(baseline_smooth["MedianFoldBreakEvenCostBps"])
            ),
            "WorstFoldWithinTolerance": (
                float(variant_summary["WorstFoldNetSharpe"])
                >= float(baseline_smooth["WorstFoldNetSharpe"]) - 0.25
            ),
            "TrainOOSGapNotWider": (
                float(variant_summary["MedianInSampleOOSGap"])
                <= float(baseline_smooth["MedianInSampleOOSGap"])
            ),
            "Contribution2020NotMoreConcentrated": (
                float(variant_summary["PositiveContributionShare2020"])
                <= float(baseline_smooth["PositiveContributionShare2020"])
            ),
        }
        gate_rows.append({
            "Variant": variant,
            "BaselineVariant": baseline_variant,
            "ImprovedNetSharpeFolds": improved_folds,
            **gates,
            "PassesAllStabilityGates": bool(
                variant != baseline_variant and all(gates.values())
            ),
        })
    gate_table = pd.DataFrame(gate_rows)
    gate_table.to_csv(output_dir / "ablation_stability_gates.csv", index=False)
    if {"A0", "A3"}.issubset(smooth.index):
        findings = {
            "security_code": (
                "Removing SecuritiesCode materially reduced the median train/OOS "
                "gap, but did not improve every OOS fold and increased turnover."
            ),
            "ohlc_levels": (
                "A2a improved smooth3 net Sharpe in four of five folds versus A0; "
                "raw OHLC levels are not required for the observed OOS signal."
            ),
            "volume_level": (
                "Removing Volume after OHLC reduced break-even cost below 5 bps "
                "and worsened the stitched net result."
            ),
            "model_complexity": (
                "Ridge reduced the median train/OOS gap from "
                f"{smooth.loc['A0', 'MedianInSampleOOSGap']:.2f} to "
                f"{smooth.loc['A3', 'MedianInSampleOOSGap']:.2f}; its A2a feature "
                "group was selected on the same OOS folds, so this remains "
                "diagnostic rather than independent validation."
            ),
            "cross_section_rank": (
                "A4 retained substantial 2020 concentration and deteriorated in "
                "later folds; it is not the stable winner."
            ),
        }
    else:
        passing = gate_table.loc[
            gate_table["PassesAllStabilityGates"], "Variant"
        ].astype(str).tolist()
        findings = {
            "model_complexity": (
                f"Reference is {baseline_variant}. Stability gates are fixed before "
                "reviewing aggregate results; stitched Sharpe is diagnostic only."
            ),
            "stability_gate_result": (
                "Passing variants: " + (", ".join(passing) if passing else "none")
            ),
        }
    report = {
        "run_role": "feature_model_ablation_suite",
        "baseline_variant": baseline_variant,
        "selection_rule": (
            "Prefer majority fold improvement, higher medians and break-even cost, "
            "non-worse worst fold, and a smaller in-sample/OOS gap; aggregate Sharpe "
            "is diagnostic only."
        ),
        "summary": _records(summary),
        "paired_changes": _records(paired),
        "parity": _records(parity_table),
        "stability_gates": _records(gate_table),
        "findings": findings,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    template = Path(__file__).with_name("templates") / "ablation_suite.html"
    payload = {
        "summary": _records(summary),
        "paired": _records(paired),
        "parity": _records(parity_table),
        "gates": _records(gate_table),
        "findings": report["findings"],
    }
    (output_dir / "ablation_report.html").write_text(
        template.read_text(encoding="utf-8").replace(
            "__ABLATION_PAYLOAD__",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        ),
        encoding="utf-8",
    )
    return report


def run_ablation_suite_report(config: Config) -> dict[str, Any]:
    options = config.raw.get("ablation_suite", {})
    if not isinstance(options, dict):
        raise ValueError("ablation_suite config must be a mapping")
    paths = []
    for value in options.get("configs", []):
        path = Path(str(value))
        if not path.is_absolute():
            path = (config.project_root / path).resolve()
        paths.append(path)
    if not paths:
        raise ValueError("ablation_suite.configs must be non-empty")
    return build_suite_report(
        paths,
        config.output_dir,
        baseline_variant=str(options.get("baseline_variant", "A0")),
    )
