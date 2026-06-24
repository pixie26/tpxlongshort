# JPX published baseline → Qlib migration gates

## Gate 0 — Freeze facts

The official JPX winner repository says the 8th-place training source was not obtained. Therefore this project calls the target the **published 8th-place-style baseline**, not a verified reproduction of private leaderboard score 0.289.

## Gate 1 — Feature parity

Run both engines on the same raw CSV:

```bash
jpx8 --config configs/baseline.yaml feature-parity
```

Acceptance target:

- identical `(Date, SecuritiesCode)` keys;
- identical row count and NaN mask;
- every feature matching fraction 100% within `rtol=1e-8`, `atol=1e-10`.

Do not proceed by silently loosening tolerances. Investigate differences in sorting, rolling-window initialization, adjustment factors and forward filling.

## Gate 2 — Native reference baseline

```bash
jpx8 --config configs/baseline.yaml native
```

This creates predictions, ranks, daily spread returns, metrics and a serialized model without Qlib. It is the auditable reference.

## Gate 3 — Qlib dataset/model parity

```bash
jpx8 --config configs/baseline.yaml qlib
jpx8 --config configs/baseline.yaml prediction-parity
```

Acceptance target:

- identical test keys;
- prediction correlation > 0.999999;
- max absolute prediction difference close to floating-point noise;
- identical daily ranks and daily spread return.

## Gate 4 — Only then improve the strategy

One change per experiment:

1. fix ranking mode;
2. replace weak time split with expanding/walk-forward validation;
3. add supplemental history before live inference;
4. remove unstable or redundant raw-price level features;
5. compare target regression, pairwise ranking and cross-sectional objectives;
6. add costs, turnover and liquidity constraints.
