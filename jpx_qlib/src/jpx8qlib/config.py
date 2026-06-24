from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    source_path: Path

    @property
    def project_root(self) -> Path:
        return self.source_path.parent.parent.resolve()

    @property
    def output_dir(self) -> Path:
        value = Path(self.raw["project"]["output_dir"])
        return value if value.is_absolute() else self.project_root / value

    @property
    def stock_prices_csv(self) -> Path:
        return Path(self.raw["data"]["stock_prices_csv"])

    @property
    def legacy_code_dir(self) -> Path:
        return Path(self.raw["data"].get("legacy_code_dir", ""))

    @property
    def feature_engine(self) -> str:
        return str(self.raw["data"].get("feature_engine", "legacy"))

    @property
    def cache_path(self) -> Path:
        return self.output_dir / self.raw["data"].get("cache_file", "prepared_panel.pkl.gz")


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a mapping: {source}")
    return Config(raw=raw, source_path=source)
