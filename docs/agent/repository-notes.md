# Repository notes

Read this document when orienting in the repository or planning the first runnable pipeline.

## Current state

- Competition data is under `data/raw/jpx/`, including train, supplemental, example-test,
  specifications, and the Linux CPython 3.7 Kaggle environment package.
- The primary training table is `data/raw/jpx/train_files/stock_prices.csv`.
- `01_data_audit.py` already checks keys, universe size, missing values, price consistency,
  adjustment events, and samples of target reconstruction. Its source currently contains
  mojibake in Chinese messages; preserve its analytical outputs if cleaning the text.
- `reports/data_audit/` contains generated audit results and should not be treated as raw input.
- This workspace root is not currently a Git repository; the nested winner-model collection
  is reference source rather than the runnable project structure.

## 8th-place reference

- The solution is a simple LightGBM regressor using security code, raw price/volume fields,
  amplitude, open-close return, close return, rolling volatility, and moving averages.
- `Preprocessing.py` fills expected dividends, forward-fills per-security observations, and
  applies cumulative adjustment factors to OHLC and volume.
- `Features.py` contains both batch and incremental implementations.
- `Trackers.py` maintains per-security state for online prediction.
- `Validation.py` partitions chronologically, while the notebooks contain the authoritative
  orchestration, exact feature list, training parameters, and submission rank conversion.
- `lgbm.pickle` may depend on historical library versions. Loading it and retraining a compatible
  model are separate validation paths.

## 5th-place reference

Use only after the 8th-place baseline is frozen. Its main ideas are an alpha-style target,
short-term mean reversion, lagged momentum, long-window volatility, cross-sectional ranks,
categorical security code, expanding-window evaluation, and training on return tails. Some
original pandas calls are obsolete and must be adapted without silently changing semantics.

## Known engineering constraints

- Kaggle notebook paths must be replaced by configurable local paths.
- The bundled competition environment binary targets Linux CPython 3.7 and is not expected to
  import directly on Windows; local replay should not depend on it.
- Large CSVs require selective reads, caching of derived features, and bounded smoke tests.
- Pickle files are executable artifacts. Load only this trusted local reference and record its
  hash; never generalize that permission to unknown pickle files.
