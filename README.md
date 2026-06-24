# JPX 东京证券交易所预测：基线复现与可交易性研究

本工程围绕 Kaggle **JPX Tokyo Stock Exchange Prediction** 竞赛开展可复现的量化研究。
当前重点不是立即优化排行榜分数，而是先忠实复现第 8 名公开方案，再判断其预测能力能否转化为
样本外、扣除交易成本后仍然成立的多空收益。

项目遵循以下原则：

- 原始数据、公开获奖代码和本地适配代码分开保存；
- 所有特征、训练和验证严格遵守时间顺序，避免未来信息泄漏；
- 先冻结可复现基线，再进行消融、特征和模型实验；
- 同时报告竞赛指标、组合收益、换手率、交易成本和分阶段稳定性；
- 不把训练期回测或单次实验结果当作可交易证据。

## 当前已完成

### 1. 数据审计与第 8 名基线复现

- 完成 JPX 股票价格数据的字段、主键、缺失值、复权和目标检查；
- 保留原始获奖代码作为只读参考，在 `baselines/8th/` 中建立可运行适配版本；
- 复现公开方案的预处理、23 个特征、LightGBM 训练、每日排名和竞赛评分；
- 同时验证原始模型和本地重训模型，并记录数据、代码、模型、依赖和输出哈希；
- 明确区分必要的现代库兼容修改与原方案逻辑。

### 2. 正式基线回测

- 建立每日 Top 200 多头 / Bottom 200 空头组合；
- 多空两侧各占 50% gross exposure，组合净敞口接近 0；
- 实现基于单边换手率的交易成本扣减；
- 分别报告训练期回溯结果和 Supplemental 样本外代理结果；
- 输出收益、Sharpe、最大回撤、命中率、换手率、成本拖累和持仓明细；
- 生成可直接查看的 HTML 报告。

旧版 Supplemental 回测显示，第 8 名模型在样本外代理区间并未延续训练期的高 Sharpe：
0 bps 下参考模型 Sharpe 约为 **-0.51**，本地重训模型约为 **-0.38**。
这说明训练期的极高表现不能直接解释为稳定 alpha。

### 3. Qlib 迁移与 walk-forward 验证

- 将相同数据、特征、时间切分、模型参数、预测、排名和评分接入 Qlib；
- 验证 Native 与 Qlib 路径的特征和预测一致性；
- 建立 5 折 expanding walk-forward，每个训练/验证边界 purge 2 个交易日；
- 拼接 2019 年下半年至 2021 年末的严格样本外预测；
- 基于冻结的样本外预测建立 50/50 long-short 组合回测；
- 已实现多空贡献、换手率、年度诊断和 0/5/10/20 bps 成本敏感度；
- 已完成回测经济口径审计：首日从零持仓建仓并收费，区分 half-turnover 与实际成交额。

当前 walk-forward 结果仍属于待冻结的研究输出。初步结果如下：

| 指标 | 初步结果 |
| --- | ---: |
| 拼接 OOS 毛 Sharpe | 0.79 |
| 5 bps 净 Sharpe | -1.00 |
| 10 bps 净 Sharpe | -2.79 |
| 平均日 half-turnover | 41.1% |
| 平均日实际成交额 / NAV | 82.3% |
| 月度胜率 | 50% |

毛收益主要来自多头端，空头端累计贡献为负；加入较低交易成本后优势已基本消失。
因此目前尚不能认为该策略具备实际可交易性。

## 下一步研究路线

后续实验按以下顺序推进。每个实验必须使用相同的时间切分、股票池、成本定义和评估口径，
并与冻结基线单独比较，避免一次改变多个因素。

### Phase A｜先判断是否可交易

- **A1**：固定 50/50 long-short portfolio；
- **A2**：核对每日 half-turnover、实际成交额、换手来源和异常日期；
- **A3**：统一报告 0/5/10/20 bps 成本情景；
- **A4**：拆分 Long / Short 收益贡献；
- **A5**：补全月度、年度和市场状态诊断。

当前 A1–A4 已按统一口径实现。A5 已有年度结果，仍需补全月度和市场状态诊断。

回测采用以下固定经济口径：

- `Target(t)` 是 `t+1 close → t+2 close` 的复权收益；
- 根据预测日 `t` 产生的权重在 `t+1 close` 建立，并持有至 `t+2 close`；
- 实际成交额为 `Σ|w_t - w_{t-1}|`，half-turnover 为其一半；
- 5 bps 表示每元实际成交额的单程成本，完整买入再卖出的双程成本为 10 bps；
- 首日前持仓设为 0，收取首日建仓成本。

### Phase B｜高风险原版设计消融

- **B1**：移除 `SecuritiesCode`；
- **B2**：移除原始 OHLC 和 Volume 绝对水平；
- **B3**：用简单线性模型替代 LightGBM。

这是进入特征优化前的硬门槛。三项实验可以最快判断高 in-sample Sharpe
究竟来自证券身份记忆、价格尺度、树模型非线性，还是具有跨股票泛化能力的信号。

### Phase C｜改善特征表达

- **C1**：截面 rank normalization；
- **C2**：行业中性特征或行业中性预测；
- **C3**：相对价格和标准化成交量特征；
- **C4**：逐组加入新的特征，并进行单组消融。

### Phase D｜训练方法

- **D1**：使用验证集 early stopping；
- **D2**：比较线性模型、LightGBM 和其他受控候选模型；
- **D3**：multi-seed ensemble；
- **D4**：fold/model ensemble。

### Phase E｜最终组合

- **E1**：Top-N 敏感度；
- **E2**：排名权重与等权重比较；
- **E3**：换手约束；
- **E4**：行业敞口限制；
- **E5**：流动性和容量约束。

只有在 Phase B–D 显示样本外信号具有稳定性后，才值得系统优化 Phase E。
否则组合层优化容易掩盖模型本身缺乏泛化能力的问题。

## 主要目录

```text
data/raw/jpx/                         原始竞赛数据，只读
JPXTokyoStockExchangePrediction/      获奖方案参考代码，只读
baselines/8th/                        第 8 名方案的本地可运行适配
jpx_qlib/                             Qlib 迁移、parity 和 walk-forward 研究
artifacts/baseline_8th/               基线模型与预测产物
artifacts/backtest_8th/               已完成的正式基线回测
reports/data_audit/                   数据审计结果
docs/agent/                           复现与量化研究规则
```

主要报告：

- `artifacts/backtest_8th/report.html`：第 8 名模型训练期与 Supplemental 回测；
- `jpx_qlib/outputs/walk_forward/portfolio_backtest/report.html`：拼接 OOS 组合回测；
- `jpx_qlib/outputs/walk_forward/walk_forward_summary.json`：walk-forward 与 Native/Qlib parity 汇总。

## 运行入口

第 8 名基线与 Supplemental 评分：

```powershell
python baselines/8th/run_baseline.py --help
```

正式基线回测：

```powershell
python baselines/8th/run_backtest.py --help
```

Qlib walk-forward 与组合回测的安装、配置和命令见
[`jpx_qlib/README.md`](jpx_qlib/README.md)。

## 结果解释

- 训练期结果仅用于诊断拟合行为，不代表样本外收益；
- Supplemental 是竞赛数据中的 held-out proxy，但仍不是完整的真实交易检验；
- walk-forward 结果优先于单次训练/测试切分；
- 所有策略结论必须同时考虑交易成本、换手率、时期稳定性和多空两端贡献；
- 当前结论是：**基线已可复现，但可交易性尚未得到证明。**
