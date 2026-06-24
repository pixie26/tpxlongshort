import json
from pathlib import Path

import pytest

from jpx8qlib.config import Config
from jpx8qlib.workflow import (
    NATIVE_REFERENCE_FILES,
    QLIB_REFERENCE_FILES,
    _freeze_native_reference,
    _freeze_qlib_reference,
)


def _config(tmp_path: Path) -> Config:
    source = tmp_path / "configs" / "baseline.yaml"
    source.parent.mkdir()
    source.write_text("project:\n  output_dir: output\n", encoding="utf-8")
    return Config(
        raw={"project": {"output_dir": str(tmp_path / "output")}},
        source_path=source,
    )


def test_freeze_native_reference_copies_complete_artifact_set(tmp_path):
    config = _config(tmp_path)
    config.output_dir.mkdir()
    for name in NATIVE_REFERENCE_FILES:
        (config.output_dir / name).write_text(
            json.dumps({"artifact": name}),
            encoding="utf-8",
        )

    reference_dir = _freeze_native_reference(config)

    assert (reference_dir / "baseline.yaml").read_text(encoding="utf-8") == (
        config.source_path.read_text(encoding="utf-8")
    )
    assert {path.name for path in reference_dir.iterdir()} == {
        *NATIVE_REFERENCE_FILES,
        "baseline.yaml",
    }


def test_freeze_native_reference_refuses_implicit_overwrite(tmp_path):
    config = _config(tmp_path)
    config.output_dir.mkdir()
    for name in NATIVE_REFERENCE_FILES:
        (config.output_dir / name).write_text(name, encoding="utf-8")
    _freeze_native_reference(config)

    with pytest.raises(FileExistsError, match="--force-reference"):
        _freeze_native_reference(config)


def test_freeze_qlib_reference_copies_complete_artifact_set(tmp_path):
    config = _config(tmp_path)
    config.output_dir.mkdir()
    for name in QLIB_REFERENCE_FILES:
        (config.output_dir / name).write_text(name, encoding="utf-8")

    reference_dir = _freeze_qlib_reference(config)

    assert {path.name for path in reference_dir.iterdir()} == {
        *QLIB_REFERENCE_FILES,
        "baseline.yaml",
    }
    with pytest.raises(FileExistsError, match="--force-reference"):
        _freeze_qlib_reference(config)
