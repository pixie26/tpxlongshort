from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .constants import DERIVED_FEATURE_COLUMNS


@contextmanager
def _legacy_import_scope(code_dir: Path):
    code_dir = code_dir.expanduser().resolve()
    required = ["Features.py", "Preprocessing.py", "Trackers.py"]
    missing = [name for name in required if not (code_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Legacy code directory {code_dir} is missing: {', '.join(missing)}"
        )

    old_path = list(sys.path)
    previous = {name: sys.modules.get(name) for name in ("Features", "Preprocessing", "Trackers")}
    try:
        sys.path.insert(0, str(code_dir))
        for name in previous:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        yield
    finally:
        for name in previous:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]
        sys.path[:] = old_path


def build_legacy_features(raw: pd.DataFrame, code_dir: str | Path) -> pd.DataFrame:
    """Run the user's original StateTracker without modifying legacy source files."""
    with _legacy_import_scope(Path(code_dir)):
        Features = importlib.import_module("Features")
        Trackers = importlib.import_module("Trackers")

        features = [
            Features.Amplitude(),
            Features.OpenCloseReturn(),
            Features.Return(),
            Features.Volatility(10),
            Features.Volatility(30),
            Features.Volatility(50),
            Features.SMA("Close", 3),
            Features.SMA("Close", 5),
            Features.SMA("Close", 10),
            Features.SMA("Close", 30),
            Features.SMA("Return", 3),
            Features.SMA("Return", 5),
            Features.SMA("Return", 10),
            Features.SMA("Return", 30),
        ]
        tracker = Trackers.StateTracker(features)
        prepared = tracker.prepare_data_for_training(raw.copy())

    names = [feature.name for feature in features]
    if names != DERIVED_FEATURE_COLUMNS:
        raise ValueError(
            "Legacy feature names differ from the published contract. "
            f"Expected {DERIVED_FEATURE_COLUMNS}, got {names}"
        )
    return prepared
