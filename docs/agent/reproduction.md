# Reproduction workflow

Read this document when connecting the downloaded data to the 8th-place model,
building a local runner, or deciding whether the baseline is reproduced.

## Phase 1: freeze the reference

- Treat `JPXTokyoStockExchangePrediction/winner-models/8th/` as upstream evidence.
- Extract the notebook's data columns, preprocessing order, feature list, LightGBM
  parameters, validation splits, prediction ordering, and rank direction into an
  explicit baseline config.
- Record source file hashes and the Python, pandas, NumPy, scikit-learn, and LightGBM
  versions needed to load or retrain the supplied model.
- Distinguish exact reproduction from compatibility edits required by current libraries.

## Phase 2: make one local path run

- Read `data/raw/jpx/train_files/stock_prices.csv` through a configurable path.
- Reproduce the reference preprocessing and feature calculations without editing raw data.
- Train or load the baseline model, create predictions, convert them to unique daily ranks,
  and calculate the competition metric locally.
- Start with a bounded date range for smoke testing, then run the complete historical period.
- Persist a run manifest containing data fingerprints, date range, feature list, parameters,
  seed, dependency versions, metric version, runtime, and output paths.

## Phase 3: validate online behavior

- Replay `example_test_files/` one date at a time using only data revealed by that date.
- Compare incremental features with batch features for the same security and timestamp.
- Check new securities, missing OHLC rows, adjustment events, suspended names, and changes in
  the daily universe.
- Require one rank per submitted security, integer ranks from `0` to `N-1`, and no duplicates.

## Baseline acceptance

The baseline is runnable when a clean command can regenerate its features, model or loaded
predictions, daily ranks, local score, and manifest from documented inputs. It is reproduced
only when remaining differences from the original are measured and explained. A successful
pickle load alone is not reproduction.

## Phase 4: controlled improvement

After freezing baseline outputs, introduce 5th-place ideas one at a time through separate
configs. Candidate experiments include market-relative targets, short-horizon mean reversion,
lagged momentum, historical volatility, cross-sectional price and volume ranks, categorical
security codes, expanding-window monthly evaluation, and tail-only training samples.

Do not combine these into a single first experiment. Each change needs an ablation against the
same baseline folds and metric, plus stability results by period rather than one aggregate score.
