# JPX 第八名公开方案 → Qlib：Baseline-first 迁移

本工程把公开的 JPX “第八名风格”方案拆成两条完全相同的数据/模型链：

```text
raw JPX CSV
  → published preprocessing/features
  → fixed time segments
  → LightGBM
  → prediction/rank/JPX spread score
        ├─ native reference path
        └─ Qlib DatasetH path
```

目标是先证明 **Qlib 没有改变数据、特征、训练样本和预测**，之后才改 feature、validation 或 portfolio。

## 重要边界

JPX 官方仓库明确说明没有取得第八名训练部分的完整源码。因此，这里能复现的是**公开代码能力**，不能承诺复现 private leaderboard 0.289。公开提交代码的 Rank 赋值也存在疑似反向映射问题，所以配置提供：

- `published_exact`：逐字复现公开 Rank 赋值语义；
- `corrected_rank`：正确地把最高 Prediction 标为 Rank 0。

默认使用 `corrected_rank`，但两者的结果必须分别保存，不能混称为官方 baseline。

## 目录放置

建议把本文件夹复制到：

```text
D:\projects\jpx\jpx_qlib_baseline
```

原来的四个文件保持不动，例如：

```text
D:\projects\jpx\8th_place\
  Features.py
  Preprocessing.py
  Trackers.py
  Validation.py
  TrainingNotebook.ipynb
```

在 `configs/baseline.yaml` 修改两条路径：

```yaml
data:
  stock_prices_csv: D:/projects/jpx/data/raw/jpx/train_files/stock_prices.csv
  legacy_code_dir: D:/projects/jpx/8th_place
```

## Windows 安装

建议使用独立 Conda 环境：

```powershell
cd D:\projects\jpx\jpx_qlib_baseline
conda create -n jpx-qlib python=3.11 -y
conda activate jpx-qlib
python -m pip install --upgrade pip
pip install -e ".[qlib,dev]"
```

Qlib 官方当前支持 Python 3.8–3.12，并支持用配置或 Python workflow 运行模型；本工程采用代码式 workflow，更适合逐步做 parity。Qlib 的 `DataHandlerLP.from_df` / `DatasetH` 接收我们预先计算好的面板，因此不会替换旧方案的金融定义。

## 1. 先生成旧代码特征

配置保持：

```yaml
feature_engine: legacy
```

运行：

```powershell
jpx8 --config configs/baseline.yaml prepare --force
```

输出：

```text
outputs/jpx8_published_baseline/
  prepared_panel.pkl.gz
  data_manifest.json
```

## 2. 检查独立重写是否与旧文件一致

```powershell
jpx8 --config configs/baseline.yaml feature-parity
```

查看：

```text
outputs/jpx8_published_baseline/feature_parity.json
```

任何 feature 不一致，都先停在这里排查。最常见来源是：

- 每只股票排序不一致；
- `AdjustmentFactor` 作用日错一日；
- rolling `min_periods` 或 `ddof` 不一致；
- 初始 NaN 与中间 NaN 的处理不同。

## 3. 跑透明 Native baseline

```powershell
jpx8 --config configs/baseline.yaml native
```

首次确认 Native 结果后，用以下命令冻结 reference：

```powershell
jpx8 --config configs/baseline.yaml native --freeze-reference
```

冻结文件保存在 `outputs/jpx8_published_baseline/native_reference/`。该命令同时保存
明确标记为 `chronological_oos` 的 `native_metrics.json`，以及仅用于诊断、不得用于
策略评估的 `native_in_sample_metrics.json`。已有冻结目录默认不会被覆盖；确需替换时
显式增加 `--force-reference`。

主要输出：

```text
native_model.joblib
native_predictions.pkl.gz
native_ranked.pkl.gz
native_daily_spread.csv
native_metrics.json
```

这条链不依赖 Qlib，是裁判答案。

## 4. 跑 Qlib baseline

```powershell
jpx8 --config configs/baseline.yaml qlib
```

确认 Native reference 已存在后，冻结 Qlib parity baseline：

```powershell
jpx8 --config configs/baseline.yaml qlib --freeze-reference
```

该命令重新生成 Qlib 结果及 Native/Qlib prediction parity，并把模型、预测、排名、
评分、配置和实际安装包清单保存到
`outputs/jpx8_published_baseline/qlib_reference/`。已有快照默认拒绝覆盖。

## Expanding walk-forward

使用五折 expanding window，并在 Train/Valid 尾部各 purge 两个实际交易日：

```powershell
jpx8 --config configs/walk_forward.yaml native-walk-forward
jpx8 --config configs/walk_forward.yaml qlib-walk-forward
jpx8 --config configs/walk_forward.yaml portfolio-backtest
```

在不重训模型的前提下，对冻结的 stitched OOS 组合做多空归因、分侧成本、
换手拆解、持仓延续率和 universe-relative beta 诊断：

```powershell
jpx8 --config configs/walk_forward.yaml portfolio-diagnostics
```

产物写入 `outputs/walk_forward/portfolio_diagnostics/`。该命令属于 2A
诊断，不引入 TOPIX、真实市值或借券可得性假设。

必须先运行 Native，再运行 Qlib。结果保存在 `outputs/walk_forward/`；每折包含模型、
预测、排名、daily spread、metrics 和 Native/Qlib prediction parity。根目录包含
2019 H2 至 2021 H2 的 stitched OOS 预测、年度诊断和合并汇总。

Qlib 使用相同的 prepared panel、相同 segments、相同 LightGBM 参数和相同 scorer。

`portfolio-backtest` 不会重训模型。它读取冻结的 Native stitched OOS predictions，
构造每日 Top 200 多头 / Bottom 200 空头组合，排名线性权重分别归一化到 +50% / -50%，
并在 `outputs/walk_forward/portfolio_backtest/` 输出自包含 HTML 图表、持仓、每日会计、
0/5/10/20 bps 成本敏感度和复现元数据。

## 5. 验证 Qlib 没有改变预测

```powershell
jpx8 --config configs/baseline.yaml prediction-parity
```

期望：

```text
prediction_correlation > 0.999999
```

并且 max absolute difference 只剩浮点误差。若不一致，先检查：categorical dtype、行顺序、缺失值、LightGBM 版本和 seed。

## 测试

```powershell
pytest
```

## 当前版本已经完成

- 旧 `StateTracker` 的无侵入调用；
- 独立等价 preprocessing/features；
- Qlib MultiIndex 数据合同；
- 原生与 Qlib LightGBM 双路径；
- 正确/公开原样两种 Rank 模式；
- JPX daily spread 与 Sharpe；
- feature/prediction parity 报告；
- Windows 配置与自动化测试。

## 仍需用你的真实数据完成的验收

本交付环境没有你的 `D:\projects\jpx` 文件，也未安装 Qlib，所以我已运行本地单元测试，但无法在这里产生你的真实 baseline score。你本机第一次运行后，最重要的是发回：

```text
feature_parity.json
native_metrics.json
qlib_metrics.json
prediction_parity.json
```

届时才能判断差异来自公开旧代码、Qlib 适配，还是原方案本身无法复现。

## Performance note for the public legacy feature code

Do not use `feature_engine: legacy` for the full JPX training file.  The public
`Features.py` calculates rolling volatility with row-wise `DataFrame.apply` and
repeated NumPy slicing.  On roughly 2.33 million rows this is prohibitively slow
and emits chained-assignment warnings under modern pandas.

Use:

```yaml
data:
  feature_engine: legacy_optimized
```

The parity-validated vectorized implementation is used for full-data
preparation. The former name `reimplemented` remains accepted as a compatibility
alias. Run legacy parity separately on a small number of securities with
complete histories:

```powershell
jpx8 --config configs/baseline.yaml feature-parity
```

## Controlled portfolio experiments

Run:

```powershell
jpx8 --config configs/walk_forward.yaml strategy-experiments
```

This command does not retrain LightGBM. It regenerates each fold's validation
predictions from the frozen Native model, selects portfolio rules using only
that validation segment, and applies the selected rule to the existing frozen
test predictions. Diagnostic controls include long-only, 75/25 long-short,
Top/Bottom concentration, equal weighting, slower rebalancing, buffers,
minimum holding periods, prediction smoothing, turnover controls, and a
universe-beta-neutral variant.

### Controlled experiment findings

All figures below use a 5 bps one-way cost on actual traded notional.

| Rule | Net Sharpe | Average traded notional | Break-even cost |
| --- | ---: | ---: | ---: |
| Baseline 50/50 | -0.995 | 0.823 | 2.22 bps |
| Long-only Top 200 | 0.350 | 0.761 | 9.55 bps |
| 75% Long / 25% Short | 0.106 | 0.792 | 5.74 bps |
| Prediction smoothing 3 days | 0.227 | 0.418 | 6.34 bps |
| Prediction smoothing 5 days | 0.183 | 0.294 | 6.61 bps |
| Buffer 150/250 + smoothing 3 days + no-trade band | 0.314 | 0.335 | 7.38 bps |
| Strict nested-selected | -0.686 | 0.593 | 2.42 bps |

The fixed smoothing results are stitched-test diagnostics, not validated
parameter choices. Strict nested walk-forward selection remained negative
because validation winners did not transfer consistently to the following test
segments. Smoothing is the strongest next candidate; buffer alone reduces
turnover but does not reliably preserve enough gross alpha. Mechanical 2-day
and 5-day rebalancing also reduced gross performance.

Detailed reports:

- `outputs/walk_forward/portfolio_diagnostics/portfolio_diagnostics.html`
- `outputs/walk_forward/strategy_experiments/strategy_experiments.html`

## Feature and model ablations

Independent configs are under `configs/ablations/`. Run Native first, then
Qlib parity, and finally the fixed portfolio report:

```powershell
jpx8 --config configs/ablations/a1_no_security_code.yaml native-walk-forward
D:\anaconda3\envs\qlib\python.exe -m jpx8qlib.cli `
  --config configs/ablations/a1_no_security_code.yaml qlib-walk-forward
jpx8 --config configs/ablations/a1_no_security_code.yaml ablation-report
jpx8 --config configs/ablations/suite.yaml ablation-suite-report
```

The first-round matrix contains A0 baseline, A1 no security code, A2a no code
and no OHLC, A2b no code and no raw OHLCV levels, A3 Ridge on the A2a feature
group, and A4 daily cross-sectional percentile ranks.

The strongest paired diagnostic is A3 Ridge with fixed smooth3: median fold net
Sharpe `0.915`, worst fold `-0.443`, break-even cost `9.03 bps`, and median
train/OOS gap `0.51`. This is not an independently validated winner because
the A2a feature group was selected using the same OOS fold set. A2a is the
strongest LightGBM simplification: it improves smooth3 net Sharpe in 4/5 folds.
Removing Volume as well causes a material deterioration.

Report:

- `outputs/ablations/report/ablation_report.html`

## Controlled feature research

The next LightGBM round keeps A2a as the fixed reference and changes one
feature group at a time. Configs are under `configs/feature_research/`.

```powershell
jpx8 --config configs/feature_research/c3a_relative_price.yaml native-walk-forward
jpx8 --config configs/feature_research/c3a_relative_price.yaml qlib-walk-forward
jpx8 --config configs/feature_research/c3a_relative_price.yaml ablation-report
jpx8 --config configs/feature_research/suite_c3.yaml ablation-suite-report
jpx8 --config configs/feature_research/c2_sector_diagnostics.yaml sector-diagnostics
```

C3a relative price and C3b normalized Volume both failed the fixed stability
gates, so C3c was not run. C4 momentum/reversal, volatility/range, and
liquidity dynamics also failed; no `c4_selected` config was produced. The
static-sector diagnostics reduce sector exposure materially but worsen
smooth3 median net Sharpe and break-even cost. The sector classification is
the supplied `2021-12-30` snapshot and is not point-in-time.

Reports:

- `outputs/feature_research/report_c3/ablation_report.html`
- `outputs/feature_research/report_c4/ablation_report.html`
- `outputs/feature_research/c2_sector_diagnostics/sector_diagnostics.html`

## Phase D training stability and Fold 6

Configs are under `configs/training_research/`. A2a, fixed 100 trees, and
smooth3 are the frozen reference. Early stopping uses validation RMSE only;
test data never selects `best_iteration`.

```powershell
jpx8 --config configs/training_research/d1_early_stopping.yaml native-walk-forward
jpx8 --config configs/training_research/d1_early_stopping.yaml qlib-walk-forward
jpx8 --config configs/training_research/suite_seeds.yaml seed-suite-report
```

Early stopping selected 1/21/4/7/19 trees and reduced smooth3 median net Sharpe
to `-0.764`. None of the three fixed regularization variants passed the paired
stability gates. Five seeds produced materially different ranks
(pairwise rank correlation about `0.83` to `0.95`), while their mean-prediction
ensemble achieved only `0.284` median net Sharpe with 3/5 positive folds.
No Phase D challenger was frozen.

Fold 6 joins the immutable train and supplemental stock-price files before
feature construction so rolling state remains continuous. Parameters are
fixed before evaluating 2021-12-06 through 2022-06-24.

| Arm | Smooth3 gross Sharpe | Smooth3 5 bps net Sharpe | Rank IC |
| --- | ---: | ---: | ---: |
| Frozen A2a LightGBM | -0.951 | -1.526 | -0.0150 |
| Frozen Ridge diagnostic | -0.400 | -0.640 | -0.0304 |

Every Phase D and Fold 6 variant has its own Native/Qlib prediction, rank, and
metric parity result.

Reports:

- `outputs/training_research/report/ablation_report.html`
- `outputs/training_research/seed_report/seed_stability.html`
