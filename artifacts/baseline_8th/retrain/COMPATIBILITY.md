# Local compatibility changes

The files under `JPXTokyoStockExchangePrediction/winner-models/8th/` are the
immutable upstream reference. This directory is a clone with only these changes:

- `run_baseline.py` replaces Kaggle notebook paths and the Kaggle iterator with a
  local CLI, artifact manifest, local competition scoring, and integrity checks.
- The CLI delegates prediction directly to the stored LightGBM Booster when an
  old sklearn-wrapper pickle has `_n_classes=None`, which LightGBM 4.6 no longer
  accepts. The serialized trees and their predictions are unchanged.
- The CLI suppresses only NumPy's expected one-element sample-standard-deviation
  warnings; the reference code immediately replaces those NaNs with zero.
- `Features.py` replaces chained `Series.iloc` assignment with equivalent `.loc`
  assignment for pandas 2.2 compatibility.
- `Preprocessing.py` replaces deprecated `fillna(method="ffill")` with the
  equivalent `ffill()` call.

Feature formulas, feature order, preprocessing order, LightGBM defaults,
categorical columns, prediction direction, and daily ranking logic are unchanged.

## Run locally

From the repository root:

```powershell
python baselines\8th\run_baseline.py `
  --data-dir data\raw\jpx `
  --output-dir artifacts\baseline_8th\reference `
  --model-source reference
```

Use `--model-source retrain` and a separate output directory to rebuild the
default LightGBM model. Each run writes predictions, daily spread returns,
metrics, the selected model, compatibility notes, and a hash/version manifest.
