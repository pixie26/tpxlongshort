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

#### Phase A Findings

| 项目 | 状态 | Findings |
| --- | --- | --- |
| A1 50/50 long-short | 已完成 | 组合 gross exposure 为 100%，多头 +50%、空头 -50%，净敞口接近 0。拼接 OOS 毛 Sharpe 为 0.79。 |
| A2 Turnover | 已完成 | 平均日 half-turnover 为 41.1%，对应平均实际成交额 82.3% NAV，换手较高。旧实现把 half-turnover 当作实际成交额，低估了约一半成本。 |
| A3 Costs | 已完成 | 5 bps 单程成本下净 Sharpe 为 -1.00；10 bps 为 -2.79；20 bps 为 -6.39。毛收益无法覆盖较低交易成本。 |
| A4 Long/Short contribution | 已完成 | 多头累计贡献约 +21.6%，空头累计贡献约 -10.7%。毛收益主要来自多头端，空头端没有提供正贡献。 |
| A5 Monthly/regime diagnostics | 部分完成 | 年度结果均已生成；月度胜率为 50%。月度明细和市场状态划分仍待补全。 |

**Phase A 结论：** 原版信号存在一定毛收益，但高换手、空头端失效和成本敏感性使其目前不具备可交易性。

### Phase B｜高风险原版设计消融

- **B1**：移除 `SecuritiesCode`；
- **B2**：移除原始 OHLC 和 Volume 绝对水平；
- **B3**：用简单线性模型替代 LightGBM。

这是进入特征优化前的硬门槛。三项实验可以最快判断高 in-sample Sharpe
究竟来自证券身份记忆、价格尺度、树模型非线性，还是具有跨股票泛化能力的信号。

#### Phase B Findings

| 项目 | 状态 | Findings |
| --- | --- | --- |
| B1 Remove SecuritiesCode | 待执行 | 尚无实验结论。重点观察训练期 Sharpe 是否大幅下降，以及 OOS 表现是否更稳定。 |
| B2 Remove raw OHLC/Volume levels | 待执行 | 尚无实验结论。用于判断模型是否依赖不可跨股票泛化的价格和成交量尺度。 |
| B3 Simple linear model | 待执行 | 尚无实验结论。用于区分线性可泛化信号与 LightGBM 的非线性记忆能力。 |

**Phase B 结论：** 尚未开始；完成前不进入大规模特征扩展。

### Phase C｜改善特征表达

- **C1**：截面 rank normalization；
- **C2**：行业中性特征或行业中性预测；
- **C3**：相对价格和标准化成交量特征；
- **C4**：逐组加入新的特征，并进行单组消融。

#### Phase C Findings

| 项目 | 状态 | Findings |
| --- | --- | --- |
| C1 Cross-sectional rank normalization | 待执行 | 尚无实验结论。 |
| C2 Sector-neutral features/predictions | 待执行 | 尚无实验结论；需要先接入可靠的行业分类并检查时间可用性。 |
| C3 Relative price/normalized volume | 待执行 | 尚无实验结论。目标是消除绝对价格和成交量尺度依赖。 |
| C4 New feature groups | 待执行 | 尚无实验结论；每组特征必须单独消融，不一次混合多个改动。 |

**Phase C 结论：** 等待 Phase B 确认原版高 in-sample Sharpe 的来源。

### Phase D｜训练方法

- **D1**：使用验证集 early stopping；
- **D2**：比较线性模型、LightGBM 和其他受控候选模型；
- **D3**：multi-seed ensemble；
- **D4**：fold/model ensemble。

#### Phase D Findings

| 项目 | 状态 | Findings |
| --- | --- | --- |
| D1 Early stopping | 待执行 | 尚无实验结论。 |
| D2 Model comparison | 待执行 | 尚无实验结论；至少应包含线性模型和冻结参数的 LightGBM。 |
| D3 Multi-seed ensemble | 待执行 | 尚无实验结论；只有单模型存在稳定 OOS 信号时才有意义。 |
| D4 Fold/model ensemble | 待执行 | 尚无实验结论；需防止使用未来 fold 模型预测过去日期。 |

**Phase D 结论：** 当前不应通过 ensemble 掩盖基础信号不稳定的问题。

### Phase E｜最终组合

- **E1**：Top-N 敏感度；
- **E2**：排名权重与等权重比较；
- **E3**：换手约束；
- **E4**：行业敞口限制；
- **E5**：流动性和容量约束。

#### Phase E Findings

| 项目 | 状态 | Findings |
| --- | --- | --- |
| E1 Top-N sensitivity | 待执行 | 当前仅使用 Top/Bottom 200，尚无敏感度结论。 |
| E2 Rank/equal weighting | 待执行 | 当前使用每侧 2→1 线性排名权重，尚未与等权比较。 |
| E3 Turnover constraints | 待执行 | Phase A 已确认换手是核心问题，但尚未测试缓冲区、持仓保持或换手惩罚。 |
| E4 Industry exposure limits | 待执行 | 尚无实验结论。 |
| E5 Liquidity/capacity constraints | 待执行 | 尚未建模成交量参与率、冲击成本、涨跌停、停牌和借券约束。 |

**Phase E 结论：** 当前只有问题诊断，没有组合优化证据。

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

## 组合诊断与受控实验结论（2026-06-25）

本阶段保持 LightGBM、特征、训练折和冻结预测不变。2B 所需的 TOPIX、
真实市值及借券数据暂不引入。

### 2A 诊断

- 50/50 long-short 的 OOS gross Sharpe 为 `0.792`，5 bps net Sharpe 为
  `-0.995`，break-even cost 仅 `2.22 bps`。
- Long-only 在 5 bps 下 Sharpe 为 `0.350`；Short-only 为 `-0.887`。
- 50/50 组合累计 long gross contribution 为 `+21.55%`，short gross
  contribution 为 `-10.72%`。
- 交易成本的 `53.7%` 来自空头侧。
- `91.2%` 的换手来自股票进入和退出持仓名单，而不是存量权重调整。
- Top 200 / Bottom 200 日延续率分别约为 `63.4%` / `57.1%`，中位持有期均
  为 1 日。
- Long-only 对 universe equal-weight 的 beta 约为 `1.20`；long-short beta
  约为 `0.054`。

### Frozen-prediction 组合实验

参数选择严格按 fold 使用 validation，test 仅用于一次固定评估。Validation
预测由冻结的每折模型重新生成，没有重新训练模型。

| 规则 | 5 bps Net Sharpe | 平均实际成交额 / NAV | Break-even cost |
| --- | ---: | ---: | ---: |
| Baseline 50/50 | -0.995 | 0.823 | 2.22 bps |
| Long-only Top 200 | 0.350 | 0.761 | 9.55 bps |
| 75% Long / 25% Short | 0.106 | 0.792 | 5.74 bps |
| Prediction smoothing 3 days | 0.227 | 0.418 | 6.34 bps |
| Prediction smoothing 5 days | 0.183 | 0.294 | 6.61 bps |
| Buffer 150/250 + smoothing 3 days + no-trade band | 0.314 | 0.335 | 7.38 bps |
| Strict nested-selected | -0.686 | 0.593 | 2.42 bps |

固定 smoothing 和组合规则的 stitched test 结果为正，但不能当作已验证参数。
严格 nested selection 在多个 fold 上没有稳定迁移。因此当前结论是：

1. Prediction smoothing 是下一轮最值得继续验证的组合层方向。
2. Buffer 可以降低换手，但单独使用没有稳定提高 break-even cost。
3. 机械 2 日或 5 日再平衡明显损失 gross alpha。
4. Long-only 和降低 short gross 改善结果，但引入较大的方向性风险与回撤。
5. 目前没有足够证据宣布组合优化成功，也不应据此修改模型或扩大参数搜索。

报告：

- `jpx_qlib/outputs/walk_forward/portfolio_diagnostics/portfolio_diagnostics.html`
- `jpx_qlib/outputs/walk_forward/strategy_experiments/strategy_experiments.html`
