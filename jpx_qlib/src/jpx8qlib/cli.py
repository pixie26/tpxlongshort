from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from .ablation_report import run_ablation_report, run_ablation_suite_report
from .config import load_config
from .data import load_raw_stock_prices, prepare_panel, select_parity_sample
from .diagnostics import run_portfolio_diagnostics
from .features import build_legacy_optimized_features
from .legacy import build_legacy_features
from .parity import compare_frames, compare_predictions, write_report
from .portfolio import run_portfolio_backtest
from .strategy_experiments import run_strategy_experiments
from .workflow import run_native, run_qlib
from .walk_forward import run_walk_forward


def _configure_logging(output_dir, level_name: str) -> logging.Logger:
    logger = logging.getLogger("jpx8qlib")
    logger.setLevel(getattr(logging, level_name))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = output_dir / "jpx8.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JPX published baseline -> Qlib migration")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console and file log level (default: INFO)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Build and cache the full prepared feature panel")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("native", help="Run the transparent non-Qlib reference baseline")
    p.add_argument("--force-prepare", action="store_true")
    p.add_argument(
        "--freeze-reference",
        action="store_true",
        help="Copy the complete native reference artifact set into native_reference/",
    )
    p.add_argument(
        "--force-reference",
        action="store_true",
        help="Replace named files in an existing native_reference/ snapshot",
    )

    p = sub.add_parser("qlib", help="Run the same model through Qlib DatasetH")
    p.add_argument("--force-prepare", action="store_true")
    p.add_argument(
        "--freeze-reference",
        action="store_true",
        help="Create prediction parity and freeze the Qlib artifact set",
    )
    p.add_argument(
        "--force-reference",
        action="store_true",
        help="Replace named files in an existing qlib_reference/ snapshot",
    )

    p = sub.add_parser(
        "native-walk-forward",
        help="Run expanding walk-forward through the transparent Native path",
    )
    p.add_argument("--force-prepare", action="store_true")

    p = sub.add_parser(
        "qlib-walk-forward",
        help="Run the same expanding walk-forward through Qlib and verify parity",
    )
    p.add_argument("--force-prepare", action="store_true")

    sub.add_parser(
        "portfolio-backtest",
        help="Backtest frozen stitched OOS predictions as a market-neutral portfolio",
    )
    sub.add_parser(
        "portfolio-diagnostics",
        help="Attribute sides, turnover, retention, and universe-relative performance",
    )
    sub.add_parser(
        "strategy-experiments",
        help="Run controlled portfolio rules with nested walk-forward selection",
    )
    sub.add_parser(
        "ablation-report",
        help="Evaluate baseline and smooth-3d portfolios for one completed ablation",
    )
    sub.add_parser(
        "ablation-suite-report",
        help="Aggregate completed ablations into paired CSV, JSON, and HTML reports",
    )

    sub.add_parser(
        "feature-parity",
        help="Compare legacy and vectorized features on a small full-history instrument sample",
    )
    sub.add_parser("prediction-parity", help="Compare native and Qlib predictions")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(cfg.output_dir, args.log_level)
    logger.info(
        "Starting command=%s config=%s log=%s",
        args.command,
        cfg.source_path,
        cfg.output_dir / "jpx8.log",
    )

    if args.command == "prepare":
        panel = prepare_panel(cfg, force=args.force)
        print(json.dumps({"rows": len(panel), "cache": str(cfg.cache_path)}, indent=2))
    elif args.command == "native":
        print(json.dumps(
            run_native(
                cfg,
                force_prepare=args.force_prepare,
                freeze_reference=args.freeze_reference,
                force_reference=args.force_reference,
            ),
            indent=2,
            allow_nan=True,
        ))
    elif args.command == "qlib":
        print(json.dumps(
            run_qlib(
                cfg,
                force_prepare=args.force_prepare,
                freeze_reference=args.freeze_reference,
                force_reference=args.force_reference,
            ),
            indent=2,
            allow_nan=True,
        ))
    elif args.command == "native-walk-forward":
        print(json.dumps(
            run_walk_forward(cfg, mode="native", force_prepare=args.force_prepare),
            indent=2,
            allow_nan=True,
        ))
    elif args.command == "qlib-walk-forward":
        print(json.dumps(
            run_walk_forward(cfg, mode="qlib", force_prepare=args.force_prepare),
            indent=2,
            allow_nan=True,
        ))
    elif args.command == "portfolio-backtest":
        print(json.dumps(
            run_portfolio_backtest(cfg),
            indent=2,
            allow_nan=True,
        ))
    elif args.command == "portfolio-diagnostics":
        print(json.dumps(
            run_portfolio_diagnostics(cfg),
            indent=2,
            allow_nan=True,
        ))
    elif args.command == "strategy-experiments":
        print(json.dumps(
            run_strategy_experiments(cfg),
            indent=2,
            allow_nan=True,
        ))
    elif args.command == "ablation-report":
        print(json.dumps(
            run_ablation_report(cfg),
            indent=2,
            allow_nan=False,
        ))
    elif args.command == "ablation-suite-report":
        print(json.dumps(
            run_ablation_suite_report(cfg),
            indent=2,
            allow_nan=False,
        ))
    elif args.command == "feature-parity":
        raw = load_raw_stock_prices(cfg.stock_prices_csv)
        sample = select_parity_sample(raw, cfg)
        sample.to_pickle(cfg.output_dir / "parity_raw_sample.pkl.gz", compression="gzip")

        legacy = build_legacy_features(sample, cfg.legacy_code_dir)
        optimized = build_legacy_optimized_features(sample)
        legacy.to_pickle(cfg.output_dir / "legacy_parity_panel.pkl.gz", compression="gzip")
        optimized.to_pickle(
            cfg.output_dir / "legacy_optimized_parity_panel.pkl.gz",
            compression="gzip",
        )

        report = compare_frames(legacy, optimized)
        report["engines"] = {
            "left": "legacy",
            "right": "legacy_optimized",
        }
        report["sample"] = {
            "raw_rows": int(len(sample)),
            "instruments": [int(x) for x in sorted(sample["SecuritiesCode"].unique())],
            "first_date": str(sample["Date"].min().date()),
            "last_date": str(sample["Date"].max().date()),
        }
        write_report(report, cfg.output_dir / "feature_parity.json")
        print(json.dumps(report, indent=2, allow_nan=True))
    elif args.command == "prediction-parity":
        left = cfg.output_dir / "native_predictions.pkl.gz"
        right = cfg.output_dir / "qlib_predictions.pkl.gz"
        if not left.exists() or not right.exists():
            raise FileNotFoundError("Run both 'native' and 'qlib' before prediction-parity")
        report = compare_predictions(pd.read_pickle(left), pd.read_pickle(right))
        write_report(report, cfg.output_dir / "prediction_parity.json")
        print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
