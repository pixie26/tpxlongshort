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

Qlib 使用相同的 prepared panel、相同 segments、相同 LightGBM 参数和相同 scorer。

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
  feature_engine: reimplemented
```

The vectorized implementation is used for full-data preparation.  Run legacy
parity separately on a small number of securities with complete histories:

```powershell
jpx8 --config configs/baseline.yaml feature-parity
```
