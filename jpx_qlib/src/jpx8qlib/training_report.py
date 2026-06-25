from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from .ablation_report import PORTFOLIO_SPECS
from .config import Config, load_config
from .strategy_experiments import construct_stateful_positions, evaluate_positions


def _resolve(config: Config, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config.project_root / path).resolve()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _average_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    key = ["Date", "SecuritiesCode"]
    base = frames[0][key + ["Target"]].copy()
    predictions = []
    for number, frame in enumerate(frames):
        values = frame[key + ["Prediction"]].rename(
            columns={"Prediction": f"Prediction_{number}"}
        )
        base = base.merge(values, on=key, how="inner", validate="one_to_one")
        predictions.append(f"Prediction_{number}")
    base["Prediction"] = base[predictions].mean(axis=1)
    return base[key + ["Prediction", "Target"]]


def run_seed_suite_report(config: Config) -> dict[str, Any]:
    options = config.raw.get("seed_suite", {})
    if not isinstance(options, dict):
        raise ValueError("seed_suite must be a mapping")
    configs = [
        load_config(_resolve(config, str(value)))
        for value in options.get("configs", [])
    ]
    if not configs:
        raise ValueError("seed_suite.configs must not be empty")
    reference = load_config(_resolve(config, str(options["reference_config"])))
    cost_bps = float(options.get("portfolio_cost_bps", 5))
    annualization_days = int(
        reference.raw["evaluation"].get("annualization_days", 252)
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    correlation_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    ensemble_daily: list[pd.DataFrame] = []
    fold_names = [
        str(spec["name"]) for spec in reference.raw["walk_forward"]["folds"]
    ]
    configs_by_variant = {
        str(item.raw["ablation"]["id"]): item for item in configs
    }
    for variant, item in configs_by_variant.items():
        combined = json.loads(
            (item.output_dir / "walk_forward_summary.json").read_text(
                encoding="utf-8"
            )
        )
        parity = combined["engineering_parity"]
        parity_rows.append({
            "Variant": variant,
            "ParityPassed": bool(parity["parity_passed"]),
            "KeysIdentical": bool(parity["all_fold_keys_identical"]),
            "RanksIdentical": bool(parity["all_fold_ranks_identical"]),
            "MetricsIdentical": bool(parity["all_fold_metrics_identical"]),
            "MaxPredictionDifference": max(
                float(row["max_abs_difference"]) for row in parity["folds"]
            ),
        })
    for fold in fold_names:
        tests = {
            str(item.raw["ablation"]["id"]): pd.read_pickle(
                item.output_dir / fold / "native_predictions.pkl.gz"
            )
            for item in configs
        }
        valids = {
            str(item.raw["ablation"]["id"]): pd.read_pickle(
                item.output_dir / fold / "native_valid_predictions.pkl.gz"
            )
            for item in configs
        }
        for left, right in combinations(tests, 2):
            key = ["Date", "SecuritiesCode"]
            merged = tests[left][key + ["Prediction"]].merge(
                tests[right][key + ["Prediction"]],
                on=key,
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            correlation_rows.append({
                "Fold": fold,
                "Left": left,
                "Right": right,
                "PredictionCorrelation": float(
                    merged["Prediction_left"].corr(merged["Prediction_right"])
                ),
                "RankCorrelation": float(
                    merged["Prediction_left"].corr(
                        merged["Prediction_right"], method="spearman"
                    )
                ),
                "MaxAbsDifference": float(
                    (
                        merged["Prediction_left"]
                        - merged["Prediction_right"]
                    ).abs().max()
                ),
            })
        for variant, test in tests.items():
            valid = valids[variant]
            model_metrics = json.loads(
                (
                    configs_by_variant[variant].output_dir
                    / fold
                    / "native_metrics.json"
                ).read_text(encoding="utf-8")
            )
            for spec in PORTFOLIO_SPECS:
                positions, _ = construct_stateful_positions(
                    test, spec, warmup=valid
                )
                _, summary = evaluate_positions(
                    positions,
                    cost_bps=cost_bps,
                    annualization_days=annualization_days,
                )
                single_rows.append({
                    "Variant": variant,
                    "Fold": fold,
                    "Portfolio": spec.name,
                    "GrossSharpe": float(summary["gross_sharpe"]),
                    "NetSharpe": float(summary["net_sharpe"]),
                    "BreakEvenCostBps": float(summary["break_even_cost_bps"]),
                    "AverageTradedNotional": float(
                        summary["average_daily_traded_notional"]
                    ),
                    "NetMaxDrawdown": float(summary["net_max_drawdown"]),
                    "BestIteration": int(
                        model_metrics.get(
                            "best_iteration",
                            configs_by_variant[variant]
                            .raw["model"]["params"]["n_estimators"],
                        )
                    ),
                    "InSampleSharpe": float(
                        model_metrics["in_sample_sharpe"]
                    ),
                    "ValidationSharpe": float(
                        model_metrics["valid_sharpe"]
                    ),
                    "OOSCompetitionSharpe": float(
                        model_metrics["oos_competition_sharpe"]
                    ),
                    "RankIC": float(model_metrics["rank_ic"]),
                    "TrainOOSGap": float(
                        model_metrics["in_sample_sharpe"]
                        - model_metrics["oos_sharpe"]
                    ),
                })
        test_ensemble = _average_predictions(list(tests.values()))
        valid_ensemble = _average_predictions(list(valids.values()))
        for spec in PORTFOLIO_SPECS:
            positions, _ = construct_stateful_positions(
                test_ensemble, spec, warmup=valid_ensemble
            )
            positions["Fold"] = fold
            daily, summary = evaluate_positions(
                positions,
                cost_bps=cost_bps,
                annualization_days=annualization_days,
            )
            daily["Fold"] = fold
            daily["Portfolio"] = spec.name
            ensemble_daily.append(daily)
            ensemble_rows.append({
                "Fold": fold,
                "Portfolio": spec.name,
                "GrossSharpe": float(summary["gross_sharpe"]),
                "NetSharpe": float(summary["net_sharpe"]),
                "BreakEvenCostBps": float(summary["break_even_cost_bps"]),
                "AverageTradedNotional": float(
                    summary["average_daily_traded_notional"]
                ),
                "NetMaxDrawdown": float(summary["net_max_drawdown"]),
                "LongContribution": float(summary["long_contribution_sum"]),
                "ShortContribution": float(summary["short_contribution_sum"]),
            })

    correlations = pd.DataFrame(correlation_rows)
    single_folds = pd.DataFrame(single_rows)
    single_summary = (
        single_folds.groupby(["Variant", "Portfolio"], as_index=False)
        .agg(
            MedianFoldNetSharpe=("NetSharpe", "median"),
            PositiveNetFolds=("NetSharpe", lambda x: int((x > 0).sum())),
            WorstFoldNetSharpe=("NetSharpe", "min"),
            MedianBreakEvenCostBps=("BreakEvenCostBps", "median"),
            MedianTradedNotional=("AverageTradedNotional", "median"),
            MedianRankIC=("RankIC", "median"),
            MedianTrainOOSGap=("TrainOOSGap", "median"),
        )
    )
    parity_summary = pd.DataFrame(parity_rows)
    ensemble_folds = pd.DataFrame(ensemble_rows)
    ensemble_summary = (
        ensemble_folds.groupby("Portfolio", as_index=False)
        .agg(
            MedianFoldGrossSharpe=("GrossSharpe", "median"),
            MedianFoldNetSharpe=("NetSharpe", "median"),
            PositiveNetFolds=("NetSharpe", lambda x: int((x > 0).sum())),
            WorstFoldNetSharpe=("NetSharpe", "min"),
            MedianBreakEvenCostBps=("BreakEvenCostBps", "median"),
            MedianTradedNotional=("AverageTradedNotional", "median"),
        )
    )
    correlations.to_csv(output_dir / "seed_correlations.csv", index=False)
    single_folds.to_csv(output_dir / "single_seed_fold_metrics.csv", index=False)
    single_summary.to_csv(output_dir / "single_seed_summary.csv", index=False)
    parity_summary.to_csv(output_dir / "seed_parity.csv", index=False)
    ensemble_folds.to_csv(output_dir / "ensemble_fold_metrics.csv", index=False)
    ensemble_summary.to_csv(output_dir / "ensemble_summary.csv", index=False)
    pd.concat(ensemble_daily, ignore_index=True).to_csv(
        output_dir / "ensemble_daily.csv", index=False
    )
    report = {
        "run_role": "lightgbm_seed_stability",
        "seeds": [
            int(item.raw["model"]["params"]["random_state"]) for item in configs
        ],
        "stochasticity_note": (
            "Only random_state changes. The observed predictions are not "
            "identical, so seed sensitivity is measured directly."
        ),
        "qlib_parity_note": (
            "Every seed variant has its own Native/Qlib prediction, rank, and "
            "metric parity check."
        ),
        "parity": _records(parity_summary),
        "single_seed_fold_metrics": _records(single_folds),
        "single_seed_summary": _records(single_summary),
        "correlations": _records(correlations),
        "ensemble_fold_metrics": _records(ensemble_folds),
        "ensemble_summary": _records(ensemble_summary),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    template = Path(__file__).with_name("templates") / "seed_stability.html"
    (output_dir / "seed_stability.html").write_text(
        template.read_text(encoding="utf-8").replace(
            "__SEED_PAYLOAD__",
            json.dumps(report, ensure_ascii=False, separators=(",", ":")),
        ),
        encoding="utf-8",
    )
    return report
