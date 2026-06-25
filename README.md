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
| B1 Remove SecuritiesCode | 已完成 | Median train/OOS competition Sharpe gap 从 29.67 降至 12.47，说明股票身份是主要拟合来源之一；但 smooth3 的 OOS 指标没有全面改善，代码仍携带部分稳定横截面信息。 |
| B2 Remove raw OHLC/Volume levels | 已完成 | 删除 code 和 OHLC 后，smooth3 median fold net Sharpe 提高至 0.55，4/5 folds 为正；继续删除 Volume 后 break-even 降至 4.29 bps，说明原始 Volume 仍包含有用信号或代理暴露。 |
| B3 Simple linear model | 已完成 | Ridge 将 median train/OOS gap 降至 0.51；smooth3 median fold net Sharpe 为 0.92，worst fold 为 -0.44。LightGBM 的大部分训练拟合没有形成稳定 OOS 增量。 |

**Phase B 结论：** 原模型存在明显的股票身份记忆和非线性过拟合。当前更可信的研究基线是无 code、无 OHLC 的简化特征组，并保留 Volume；Ridge 结果值得独立区间验证，但因其特征组由同一批 OOS folds 选出，暂不能视为最终 winner。

### Phase C｜改善特征表达

- **C1**：截面 rank normalization；
- **C2**：行业中性特征或行业中性预测；
- **C3**：相对价格和标准化成交量特征；
- **C4**：逐组加入新的特征，并进行单组消融。

#### Phase C Findings

| 项目 | 状态 | Findings |
| --- | --- | --- |
| C1 Cross-sectional rank normalization | 已完成 | A4 的 smooth3 median fold net Sharpe 为 0.22，2020 正贡献占比约 73%，未形成稳定改善。 |
| C2 Sector-neutral features/predictions | 已完成（诊断） | 静态 33-sector 去均值将 smooth3 median net Sharpe 从 0.55 降至 -0.17；行业内 rank 降至 -0.93。行业净敞口下降，但 alpha 和 break-even 同时显著下降。 |
| C3 Relative price/normalized volume | 已完成 | C3a、C3b 均未通过预注册稳定性门槛，因此未运行 C3c。相对价格提高部分毛指标，但成本后 median 与 break-even 均下降。 |
| C4 New feature groups | 已完成 | Momentum、volatility/range、liquidity 三组分别消融，均未通过全部门槛，不生成 `c4_selected`。 |

**Phase C 结论：** 继续保留 A2a LightGBM 作为当前非线性研究 reference。没有任何
C3/C4 特征组在多数 folds、median、break-even、worst fold、train/OOS gap 和
2020 集中度上同时改善。Ridge 仍是重要复杂度诊断，但不替代 LightGBM 的后续
factor 研究基线。

### Phase D｜训练方法

- **D1**：验证集 early stopping；
- **D2**：multi-seed 稳定性与简单 prediction ensemble；
- **D3**：有限、单因素 LightGBM 正则化；
- **D4**：冻结唯一 challenger；
- **D5**：在 Supplemental 上运行独立 Fold 6。

#### Phase D Findings

| 项目 | 状态 | Findings |
| --- | --- | --- |
| D1 Early stopping | 已完成 | 验证 RMSE 选出的 best iteration 为 1/21/4/7/19；smooth3 median net Sharpe 降至 -0.76。gap 缩小主要来自欠拟合，不是泛化改善。 |
| D2 Multi-seed stability | 已完成 | 五个 seed 的 rank correlation 约 0.83–0.95；smooth3 median net Sharpe 为 0.20–0.55。五 seed ensemble 仅 0.28、3/5 正 folds，未改善 reference。 |
| D3 Limited regularization | 已完成 | 增大叶节点样本、缩小树、增加 L2 均未通过 paired gates；最佳 D3c 也仅改善 2/5 folds，并恶化 worst fold。 |
| D4 Freeze challenger | 已完成 | 没有训练 variant 同时满足多数 folds、median、worst fold、break-even 和 gap 门槛，因此不冻结 challenger。 |
| D5 Fold 6 | 已完成 | Supplemental 2021-12-06 至 2022-06-24 上，A2a smooth3 Gross/Net Sharpe 为 -0.95/-1.53，Ridge 为 -0.40/-0.64；两者 Rank IC 和 break-even 均为负。 |

**Phase D 结论：** A2a 在原五折中的弱正结果没有通过独立 Fold 6。
Early stopping、seed ensemble 和有限正则化均未产生合格 challenger；当前应把
Phase C/D 的正结果视为样本依赖诊断，而不是可交易模型证据。

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

## 第一轮特征与模型消融（2026-06-25）

所有实验使用相同的五折 expanding walk-forward、相同 target、相同排名和
`5 bps` 单程成本。每个模型只报告原 baseline portfolio 与固定 smooth3，
没有重新搜索组合参数。

| 实验 | Smooth3 median fold net Sharpe | Positive net folds | Worst fold net Sharpe | Break-even | Median train/OOS gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 Baseline | 0.379 | 3/5 | -2.278 | 6.34 bps | 29.67 |
| A1 No SecuritiesCode | 0.213 | 3/5 | -1.358 | 5.77 bps | 12.47 |
| A2a No code / no OHLC | 0.551 | 4/5 | -1.040 | 6.88 bps | 12.92 |
| A2b No code / no raw levels | 0.452 | 3/5 | -1.747 | 4.29 bps | 12.41 |
| A3 Ridge on A2a features | 0.915 | 3/5 | -0.443 | 9.03 bps | 0.51 |
| A4 Cross-sectional rank | 0.223 | 3/5 | -2.012 | 5.93 bps | 10.05 |

Paired findings：

1. A1 的训练 Sharpe 大幅下降，而 OOS 没有同比恶化，确认
   `SecuritiesCode` 是原版高训练拟合的重要来源；但删除 code 会提高换手，
   且并非每折都改善。
2. A2a 在 smooth3 下相对 A0 改善 4/5 folds，并降低对 2020 的集中依赖。
   原始 OHLC 绝对水平不是必要 alpha 来源。
3. A2b 明显弱于 A2a。原始 Volume 不能直接归类为有害尺度，它可能包含流动性、
   规模或可交易性信息，下一步应做更严格的相对 Volume 表达，而不是直接删除。
4. A3 的训练/OOS gap 几乎消失，且 smooth3 的 worst fold 显著改善。这支持
   “LightGBM 大量非线性拟合属于过拟合”的判断。但 A3 的特征组是在同一批 OOS
   folds 上从 A1/A2 中选择的，因此该结果仍是诊断性证据，需要新的独立时间区间验证。
5. A4 的 median Rank IC 较高，但 2020 占正贡献约 73%，并在 2021 H2 失效；
   截面 rank 不能作为当前稳定 winner。

Native/Qlib parity：

- A0、A1、A2a、A2b、A4 predictions 精确一致；
- A3 Ridge 最大预测差为 `2.32e-12`，daily ranks 和 metrics 完全一致。

主报告：

- `jpx_qlib/outputs/ablations/report/ablation_report.html`

## Phase C 特征表达与行业诊断（2026-06-25）

本轮固定使用 A2a LightGBM 作为 C0：删除 `SecuritiesCode` 和 OHLC 原始水平，
保留原始 Volume。模型参数、五折 walk-forward、Top/Bottom 200、50/50 gross、
`5 bps` 单程成本以及 raw/smooth3 组合均不重新搜索。

### C3/C4 单组消融

以下均为 smooth3；break-even 为五折中位数。`Improved folds` 是相对 C0 的
5 bps Net Sharpe paired improvement count。

| Variant | Median fold net Sharpe | Improved folds | Positive folds | Worst fold | Median break-even | Train/OOS gap | 2020 positive share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 / A2a | 0.551 | — | 4/5 | -1.040 | 7.68 bps | 12.92 | 44.1% |
| C3a Relative price | 0.244 | 3/5 | 3/5 | -0.769 | 5.49 bps | 11.96 | 57.4% |
| C3b Normalized Volume | 0.312 | 2/5 | 3/5 | -1.255 | 6.45 bps | 13.34 | 43.4% |
| C4a Momentum/reversal | 0.470 | 2/5 | 4/5 | -0.760 | 7.62 bps | 12.54 | 61.7% |
| C4b Volatility/range | 0.572 | 2/5 | 3/5 | -1.111 | 6.29 bps | 13.43 | 48.8% |
| C4c Liquidity dynamics | 0.542 | 2/5 | 3/5 | -1.550 | 7.58 bps | 13.43 | 34.2% |

预注册门槛要求至少 3/5 folds 改善、median net 提高、median break-even 不下降、
worst fold 不低于 C0 超过 0.25、train/OOS gap 不扩大且 2020 集中度不增加。
没有 variant 全部通过：

1. C3a 虽改善 3/5 folds，并显著改善 worst fold，但 median net、break-even 和
   2020 集中度失败。其较好的 stitched gross 不能解释为成本后稳定改进。
2. C3b 未证明用这一组 normalized Volume 替换原 Volume 有益。
3. C4a 的改善偏向早期区间，2020 集中度升至 61.7%，fold 5 毛信号转负。
4. C4b 仅有 median net 小幅提高，其余关键稳定性指标恶化。
5. C4c 降低 2020 集中度，但 paired coverage、worst fold 和 gap 不合格。
6. 因 C3a/C3b 均未通过，按预先规则没有运行 C3c；也没有生成 `c4_selected`。

### C2 静态行业中性化诊断

行业分类来自比赛提供的 `stock_list.csv`，其 `EffectiveDate` 为
`2021-12-30`。这不是严格 point-in-time 的历史行业分类，可能存在分类漂移，
因此结果只作为 prediction 层诊断，不回写训练特征。

| Prediction transform | Smooth3 median net Sharpe | Positive folds | Median break-even | Median sector net-gross | Median max sector exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 0.551 | 4/5 | 7.68 bps | 0.329 | 0.060 |
| 33-sector demean | -0.166 | 2/5 | 4.36 bps | 0.153 | 0.028 |
| 33-sector percentile rank | -0.929 | 0/5 | 2.50 bps | 0.140 | 0.018 |

简单行业中性化确实降低行业集中度，但损失的毛收益和单位换手 alpha 更多。
当前不能把 A2a 收益全部解释为纯个股 alpha，但也不应采用机械行业去均值或
行业内 rank。所有 C3/C4 Native/Qlib daily ranks、metrics 和 portfolio
accounting 均一致；C4c 最大预测差低于 `1e-15`。

报告：

- `jpx_qlib/outputs/feature_research/report_c3/ablation_report.html`
- `jpx_qlib/outputs/feature_research/report_c4/ablation_report.html`
- `jpx_qlib/outputs/feature_research/c2_sector_diagnostics/sector_diagnostics.html`

## Phase D 训练稳定性与独立 Fold 6（2026-06-26）

本轮冻结 A2a 特征、100-tree LightGBM 和 smooth3 组合。D1 的 early stopping
只允许验证集选择树数；D2 只改变 seed；D3 每次只改变一类正则化参数。

| Variant | Smooth3 median fold net Sharpe | Positive folds | Worst fold | Median break-even | Median train/OOS gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 / A2a | 0.551 | 4/5 | -1.040 | 7.68 bps | 12.92 |
| D1 Early stopping | -0.764 | 1/5 | -2.537 | 0.58 bps | 2.98 |
| D3a min_child_samples=100 | 0.347 | 3/5 | -1.577 | 6.58 bps | 13.43 |
| D3b num_leaves=15, max_depth=5 | 0.252 | 3/5 | -1.570 | 6.24 bps | 8.03 |
| D3c lambda_l2=10 | 0.735 | 3/5 | -1.306 | 8.17 bps | 13.02 |
| Five-seed mean prediction | 0.284 | 3/5 | -1.250 | 6.44 bps | N/A |

D1 虽将 gap 降至 2.98，但 best iteration 中位数仅 7，且成本后表现转负。
D3c 的 median 较高，但仅改善 2/5 folds，positive-fold coverage、worst fold 和
gap 均未通过门槛。五 seed prediction ensemble 同样弱于 D0，因此 D4 没有
challenger。

Fold 6 使用 train + supplemental 连续面板，训练截至 2021-06-28，验证截至
2021-12-01，测试为 2021-12-06 至 2022-06-24；边界 purge 2 个交易日，且没有
在 Fold 6 内选择参数。

| Arm | Portfolio | Gross Sharpe | 5 bps Net Sharpe | Rank IC | Break-even |
| --- | --- | ---: | ---: | ---: | ---: |
| F6-A0 A2a LightGBM | smooth3 | -0.951 | -1.526 | -0.0150 | -8.33 bps |
| F6-Ridge diagnostic | smooth3 | -0.400 | -0.640 | -0.0304 | -8.35 bps |

Fold 6 否定了 A2a 和 Ridge 在原五折上的正面诊断。所有 Phase D 与 Fold 6
variant 均分别通过 Native/Qlib prediction、rank 和 metrics parity。

报告：

- `jpx_qlib/outputs/training_research/report/ablation_report.html`
- `jpx_qlib/outputs/training_research/seed_report/seed_stability.html`
