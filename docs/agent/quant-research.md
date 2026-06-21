# Quant research rules

Read this document for feature, target, validation, scoring, portfolio interpretation,
or model-comparison work.

## Information timing

- Define each row by an explicit decision date and available-information cutoff.
- The competition `Target` is a future return label. It may be used as supervised training
  output only after enforcing chronological train/validation separation.
- Grouped shifts and rolling values must be calculated after sorting by security and date.
- Cross-sectional ranks and means may use only the securities and values available that day.
- Never backfill a feature from the future. Forward fill only when economically justified,
  bounded, and consistent between training and online replay.
- Corporate-action adjustment must match the reference first, then be independently checked
  against target semantics before any correction is proposed.

## Validation and scoring

- Use walk-forward or expanding-window validation with an explicit gap when labels overlap
  the validation boundary. Never use random row splits.
- Keep folds date-based so all securities from one date remain in the same partition.
- Implement the competition ranking and portfolio score once, test it with hand-built examples,
  and use the same implementation everywhere.
- Track regression diagnostics only as secondary evidence; selection is based on out-of-sample
  competition score and its stability.
- Report per-fold and per-period scores, mean, dispersion, worst period, coverage, and turnover
  where available. Do not select from one favorable interval.

## Experiment discipline

- Freeze the baseline config and outputs before optimization.
- Change one conceptual component at a time: data, target, feature, model, validation, or ranking.
- Use identical data dates, universe, folds, seeds, and metric code for comparisons.
- Separate current defaults, tested alternatives, and ideas not yet tested.
- Persist enough metadata to reproduce every reported result and tie it to code and data.
- Treat Kaggle leaderboard scores as external evidence, not a substitute for local validation.

## Practical interpretation

- The competition score represents a stylized daily long-short ranking, not a production-ready
  trading result.
- Do not claim tradable alpha without testing costs, liquidity, turnover, borrowability,
  execution timing, capacity, and regime stability.
- Prefer simple features and models unless added complexity produces repeatable out-of-sample
  improvement and survives ablation.
