from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "baseline.yaml"

for command in ("prepare", "native", "qlib", "prediction-parity"):
    subprocess.run([sys.executable, "-m", "jpx8qlib.cli", "--config", str(CONFIG), command], check=True, cwd=ROOT)
