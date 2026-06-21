# Project

Reproducible research and engineering for the Kaggle JPX Tokyo Stock Exchange
Prediction competition. The current priority is a faithful local reproduction of
the 8th-place LightGBM solution; later improvements may draw from the 5th-place
solution without obscuring the baseline.

## Core invariants

- Use only information available at the prediction timestamp when building features.
- Preserve chronological order, per-security state, universe membership, and target timing.
- Treat `data/raw/jpx/` as immutable source data; never rewrite files in place.
- Keep the original winner code as reference material. Implement runnable adaptations
  outside `JPXTokyoStockExchangePrediction/winner-models/` unless explicitly asked.
- Reproduce the 8th-place baseline before changing its features, model, validation, or
  ranking logic; label intentional deviations explicitly.
- Verify that batch-training and online/incremental feature calculations agree.
- Evaluate ranked predictions with the competition metric, not regression loss alone.
- Avoid look-ahead bias, survivorship bias, future-filled values, and leakage through
  adjustment factors, cross-sectional transforms, targets, or validation boundaries.
- Never fabricate market data, scores, model provenance, or reproduction fidelity.
- Make results reproducible with explicit configuration, deterministic seeds, and run metadata.
- Preserve user changes and unrelated files.

## Important paths

- `data/raw/jpx/`: downloaded competition data; read-only input.
- `01_data_audit.py`: initial stock-price audit and target checks.
- `reports/data_audit/`: generated audit summaries and plots.
- `JPXTokyoStockExchangePrediction/winner-models/8th/`: primary baseline reference.
- `JPXTokyoStockExchangePrediction/winner-models/5th/`: later optimization reference.
- `docs/agent/`: focused project guidance; load only what the task requires.

When adding the runnable project, prefer clear boundaries such as `src/` for reusable
pipeline code, `scripts/` for entry points, `configs/` for experiment definitions,
`tests/` for contracts and regression tests, and `artifacts/` or `reports/` for outputs.
Do not move existing files solely to match this layout.

## Working rules

- Inspect the relevant source, notebook cells, schemas, and tests before editing.
- Make small, reviewable changes; do not modernize and alter the algorithm in one step.
- Keep data loading, preprocessing, features, validation, training, ranking, and reporting
  separated by responsibility.
- Centralize filesystem paths and keep Kaggle paths out of reusable code.
- Preserve identifiers and date columns with explicit types; sort explicitly before
  grouped shifts, rolling windows, cumulative adjustments, or forward fills.
- Fit all learned transformations on training data only.
- Keep baseline and experimental configurations separate; do not overwrite baseline artifacts.
- Record package versions and model serialization compatibility when loading `lgbm.pickle`.
- Do not install dependencies or regenerate large outputs unless the task requires it.

## Context and large-file discipline

- Search first; read only relevant functions, notebook cells, schemas, and selected rows.
- Do not print or load full large CSV files when schema, dimensions, samples, or aggregates suffice.
- Do not inspect the 695 MB options file or other unused tables for the price-only baseline.
- Do not load every routed guidance document for a narrow task.

## Verification

- Use the narrowest relevant check first, then broaden in proportion to risk.
- For path or config edits, inspect configuration and run a focused smoke test.
- For preprocessing or feature changes, test ordering, adjustment handling, missing values,
  and batch-versus-online parity on multiple securities including edge cases.
- For validation, target, ranking, or metric changes, run chronological leakage tests and an
  end-to-end backtest on a bounded date range before a full run.
- For model or strategy changes, compare against the frozen 8th-place baseline using identical
  folds, dates, universe, seed, and metric implementation.
- Report tests run, tests not necessary, tests not run, and remaining risks separately.

## Task-specific guidance

Read only the documents relevant to the current task:

- `docs/agent/reproduction.md`: baseline phases, acceptance criteria, and deviation tracking.
- `docs/agent/quant-research.md`: time-series validation, metric, leakage, and experiment rules.
- `docs/agent/repository-notes.md`: current data, reference implementation, and known constraints.

## Handoff

- Report changed files, rationale, commands or tests run, resulting metrics, and residual risks.
- For research changes, state whether the result is baseline reproduction, engineering-only
  adaptation, or a strategy experiment.
- For fixes that can change historical results, state why and identify artifacts to regenerate.
