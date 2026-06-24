from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import prepare_panel
from .parity import compare_frames, compare_predictions, write_report
from .workflow import run_native, run_qlib


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JPX published baseline -> Qlib migration")
    parser.add_argument("--config", default="configs/baseline.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Build and cache the prepared feature panel")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("native", help="Run the transparent non-Qlib reference baseline")
    p.add_argument("--force-prepare", action="store_true")

    p = sub.add_parser("qlib", help="Run the same model through Qlib DatasetH")
    p.add_argument("--force-prepare", action="store_true")

    sub.add_parser("feature-parity", help="Compare legacy and reimplemented features")
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
        original_engine = cfg.raw["data"].get("feature_engine", "legacy")
        cfg.raw["data"]["feature_engine"] = "legacy"
        legacy = prepare_panel(cfg, force=True)
        legacy_cache = cfg.cache_path
        legacy.rename(columns={}).to_pickle(cfg.output_dir / "legacy_panel.pkl.gz", compression="gzip")
        cfg.raw["data"]["feature_engine"] = "reimplemented"
        reimpl = prepare_panel(cfg, force=True)
        reimpl.to_pickle(cfg.output_dir / "reimplemented_panel.pkl.gz", compression="gzip")
        cfg.raw["data"]["feature_engine"] = original_engine
        # Keep the default cache consistent with the configured engine after the comparison.
        (legacy if original_engine == "legacy" else reimpl).to_pickle(
            cfg.cache_path, compression="gzip"
        )
        report = compare_frames(legacy, reimpl)
        write_report(report, cfg.output_dir / "feature_parity.json")
        print(json.dumps(report, indent=2, allow_nan=True))
    elif args.command == "prediction-parity":
        left = cfg.output_dir / "native_predictions.pkl.gz"
        right = cfg.output_dir / "qlib_predictions.pkl.gz"
        if not left.exists() or not right.exists():
            raise FileNotFoundError("Run both 'native' and 'qlib' before prediction-parity")
        report = compare_predictions(__import__("pandas").read_pickle(left), __import__("pandas").read_pickle(right))
        write_report(report, cfg.output_dir / "prediction_parity.json")
        print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
