from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import load_config
from .data import load_raw_stock_prices, prepare_panel, select_parity_sample
from .features import build_reimplemented_features
from .legacy import build_legacy_features
from .parity import compare_frames, compare_predictions, write_report
from .workflow import run_native, run_qlib


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JPX published baseline -> Qlib migration")
    parser.add_argument("--config", default="configs/baseline.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Build and cache the full prepared feature panel")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("native", help="Run the transparent non-Qlib reference baseline")
    p.add_argument("--force-prepare", action="store_true")

    p = sub.add_parser("qlib", help="Run the same model through Qlib DatasetH")
    p.add_argument("--force-prepare", action="store_true")

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

    if args.command == "prepare":
        panel = prepare_panel(cfg, force=args.force)
        print(json.dumps({"rows": len(panel), "cache": str(cfg.cache_path)}, indent=2))
    elif args.command == "native":
        print(json.dumps(run_native(cfg, args.force_prepare), indent=2, allow_nan=True))
    elif args.command == "qlib":
        print(json.dumps(run_qlib(cfg, args.force_prepare), indent=2, allow_nan=True))
    elif args.command == "feature-parity":
        raw = load_raw_stock_prices(cfg.stock_prices_csv)
        sample = select_parity_sample(raw, cfg)
        sample.to_pickle(cfg.output_dir / "parity_raw_sample.pkl.gz", compression="gzip")

        legacy = build_legacy_features(sample, cfg.legacy_code_dir)
        reimpl = build_reimplemented_features(sample)
        legacy.to_pickle(cfg.output_dir / "legacy_parity_panel.pkl.gz", compression="gzip")
        reimpl.to_pickle(cfg.output_dir / "reimplemented_parity_panel.pkl.gz", compression="gzip")

        report = compare_frames(legacy, reimpl)
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
