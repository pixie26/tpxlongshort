from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import Config
from .constants import FEATURE_COLUMNS, LABEL_COLUMN
from .features import build_reimplemented_features
from .legacy import build_legacy_features


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_stock_prices(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"JPX stock prices file not found: {source}")
    df = pd.read_csv(source)
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    return df


def prepare_panel(config: Config, force: bool = False) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cache = config.cache_path
    if cache.exists() and not force:
        return pd.read_pickle(cache)

    raw = load_raw_stock_prices(config.stock_prices_csv)
    if config.feature_engine == "legacy":
        prepared = build_legacy_features(raw, config.legacy_code_dir)
    elif config.feature_engine == "reimplemented":
        prepared = build_reimplemented_features(raw)
    else:
        raise ValueError("feature_engine must be 'legacy' or 'reimplemented'")

    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="raise")
    prepared = prepared.sort_values(["Date", "SecuritiesCode"], kind="mergesort").reset_index(drop=True)

    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in prepared.columns]
    if missing:
        raise ValueError(f"Prepared panel is missing columns: {missing}")

    prepared.to_pickle(cache, compression="gzip")
    manifest = {
        "source": str(config.stock_prices_csv),
        "source_sha256": file_sha256(config.stock_prices_csv),
        "feature_engine": config.feature_engine,
        "rows": int(len(prepared)),
        "dates": int(prepared["Date"].nunique()),
        "instruments": int(prepared["SecuritiesCode"].nunique()),
        "first_date": str(prepared["Date"].min().date()),
        "last_date": str(prepared["Date"].max().date()),
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
    }
    (config.output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return prepared


def to_qlib_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Convert a prepared flat panel to Qlib's MultiIndex row/column contract."""
    required = set(FEATURE_COLUMNS + [LABEL_COLUMN, "Date", "SecuritiesCode"])
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Cannot create Qlib frame; missing: {missing}")

    df = panel.copy()
    df["datetime"] = pd.to_datetime(df["Date"])
    df["instrument"] = "JP" + df["SecuritiesCode"].astype(int).astype(str)
    df = df.set_index(["datetime", "instrument"]).sort_index()

    features = df[FEATURE_COLUMNS].copy()
    label = df[[LABEL_COLUMN]].copy()
    features.columns = pd.MultiIndex.from_product([["feature"], features.columns])
    label.columns = pd.MultiIndex.from_product([["label"], label.columns])
    return pd.concat([features, label], axis=1).sort_index(axis=1)
