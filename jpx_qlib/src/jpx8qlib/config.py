from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .constants import FEATURE_COLUMNS


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    source_path: Path

    @property
    def project_root(self) -> Path:
        for parent in self.source_path.parents:
            if (parent / "pyproject.toml").is_file():
                return parent.resolve()
        return self.source_path.parent.parent.resolve()

    @property
    def output_dir(self) -> Path:
        value = Path(self.raw["project"]["output_dir"])
        return value if value.is_absolute() else self.project_root / value

    @property
    def stock_prices_csv(self) -> Path:
        return Path(self.raw["data"]["stock_prices_csv"])

    @property
    def stock_prices_csvs(self) -> list[Path]:
        values = self.raw["data"].get("stock_prices_csvs")
        if values is None:
            return [self.stock_prices_csv]
        if not isinstance(values, list) or not values:
            raise ValueError("data.stock_prices_csvs must be a non-empty list")
        return [Path(str(value)) for value in values]

    @property
    def legacy_code_dir(self) -> Path:
        return Path(self.raw["data"].get("legacy_code_dir", ""))

    @property
    def feature_engine(self) -> str:
        # "reimplemented" was the original migration name. Keep accepting it,
        # but expose the parity-validated engine under its precise canonical name.
        value = str(self.raw["data"].get("feature_engine", "legacy_optimized"))
        if value == "reimplemented":
            return "legacy_optimized"
        return value

    @property
    def cache_path(self) -> Path:
        return self.output_dir / self.raw["data"].get("cache_file", "prepared_panel.pkl.gz")

    @property
    def parity_config(self) -> dict[str, Any]:
        value = self.raw.get("parity", {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("parity config must be a mapping")
        return value

    @property
    def feature_columns(self) -> list[str]:
        value = self.raw.get("features", {}).get("columns", FEATURE_COLUMNS)
        if not isinstance(value, list) or not value:
            raise ValueError("features.columns must be a non-empty list")
        columns = [str(column) for column in value]
        if len(columns) != len(set(columns)):
            raise ValueError("features.columns contains duplicates")
        return columns

    @property
    def feature_groups(self) -> list[str]:
        value = self.raw.get("features", {}).get("groups", [])
        if not isinstance(value, list):
            raise ValueError("features.groups must be a list")
        groups = [str(group) for group in value]
        supported = {
            "relative_price",
            "normalized_volume",
            "momentum_reversal",
            "volatility_range",
            "liquidity_dynamics",
        }
        unknown = sorted(set(groups) - supported)
        if unknown:
            raise ValueError(f"Unsupported features.groups: {unknown}")
        if len(groups) != len(set(groups)):
            raise ValueError("features.groups contains duplicates")
        return groups

    @property
    def feature_transform(self) -> str:
        value = str(
            self.raw.get("features", {}).get("cross_sectional_transform", "none")
        )
        if value not in {"none", "percentile_rank"}:
            raise ValueError(
                "features.cross_sectional_transform must be 'none' or "
                "'percentile_rank'"
            )
        return value

    @property
    def categorical_features(self) -> list[str]:
        configured = self.raw.get("features", {}).get(
            "categorical", ["SecuritiesCode", "SupervisionFlag"]
        )
        if not isinstance(configured, list):
            raise ValueError("features.categorical must be a list")
        return [str(column) for column in configured if str(column) in self.feature_columns]

    @property
    def model_type(self) -> str:
        value = str(self.raw.get("model", {}).get("type", "lightgbm")).lower()
        if value not in {"lightgbm", "ridge"}:
            raise ValueError("model.type must be 'lightgbm' or 'ridge'")
        return value


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a mapping: {source}")
    parent_value = raw.pop("extends", None)
    if parent_value is not None:
        parent_path = Path(str(parent_value))
        if not parent_path.is_absolute():
            parent_path = (source.parent / parent_path).resolve()
        parent = load_config(parent_path).raw
        raw = _deep_merge(parent, raw)
    return Config(raw=raw, source_path=source)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
