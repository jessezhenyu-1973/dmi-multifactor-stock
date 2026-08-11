---
name: dmi-multifactor-stock
description: >-
  9 因子 A 股多因子选股与回测引擎。因子含 DMI + MACD/KDJ/RSI/布林带/威科夫成交量/135 战法 + 基本面(PE/PB/ROE) + 舆情，含交易成本模型(佣金0.03%/印花0.05%/滑点0.1%)、-10% 硬止损、-20% 最大回撤清仓纪律。支持：沪深300 成分股抓取缓存、离线多因子回测、因子权重/Top-N 网格调优、样本外(过拟合)检验、对比报告。Triggers: 多因子选股、量化策略、股票回测、A股回测、DMI选股、参数调优、网格搜索、沪深300回测、因子权重、选股策略、因子模型、动量选股、价值选股。Use when the user wants to build/run/tune a quantitative stock-selection strategy, backtest A-share factors, or validate a strategy out-of-sample.
description_zh: "9 因子 A 股多因子选股与回测引擎：DMI+技术+基本面+舆情，含成本模型与样本外检验"
description_en: "9-factor A-share multifactor stock-selection & backtest engine with cost model and out-of-sample validation"
version: 1.0.2
homepage: https://www.workbuddy.cn
allowed-tools: Read,Write,Edit,Bash,Grep
display_name: "DMI多因子量化选股策略"
display_name_en: "DMI Multifactor Quant Stock Strategy"
visibility: "public"
---

# DMI 多因子量化选股策略引擎（A 股）

一个**自包含、可复现**的 9 因子 A 股量化回测与选股工具。本 skill 把完整管线打包在 `scripts/` 下，Hermes（或任意 agent）可直接调用。

## 策略构成

- **9 个因子**（权重顺序固定为 `dmi, macd, kdj, rsi, bol, wyk, m135, fund, sent`）：
  1. DMI（动向指标）2. MACD 3. KDJ 4. RSI 5. 布林带 6. 威科夫成交量 7. 135 战法 8. 基本面(PE/PB/ROE) 9. 舆情
- **交易成本模型**：佣金 0.03% 双边、印花税 0.05% 仅卖、滑点 0.1% 双边（默认开启）。
- **风险纪律**：个股 -10% 硬止损；组合 -20% 最大回撤触发清仓。
- **复权**：固定 **hfq（后复权）**，锚定上市首日 → 跨运行可复现（切勿用 qfq，其锚最新价会漂移）。
- **舆情**：当前为 **md5 确定性 mock**（非真实新闻），用于稳定复现，非真实情绪信号。

## 何时使用

- 用户要求「多因子选股 / 量化策略 / 股票回测 / 参数调优 / 沪深300 回测 / 因子权重」。
- 用户要评估某个选股逻辑的历史表现、做样本外检验、或对比不同权重。
- 用户要把选股策略接入实盘前先做回测验证。

## 运行环境

- **解释器**：使用 WorkBuddy 托管 Python（已装 akshare）：
  `C:\Users\jesse\.workbuddy\binaries\python\versions\3.13.12\python.exe`
  （若当前环境 akshare 已在 `python` 中，也可直接 `python`，但托管路径最稳。）
- **数据抓取**需要 akshare + 联网；**离线回测**只需已缓存数据，无需 akshare。
- 所有脚本位于本 skill 的 `scripts/` 目录；运行前 `cd` 到该目录（保证模块互相 import 成功）。

## 完整管线（5 步）

> 缓存目录可用环境变量 `QF_CACHE_DIR` 覆盖，便于复用已抓取的沪深300缓存（例如指向已有的 `cache/`），避免重复打接口。

### 0) 快速演示（无需缓存，需 akshare）
10 只权重股、2024 全年、真实行情+真实基本面：
```bash
cd "<skill>/scripts"
"<managed_python>" multi_factor_dmi_strategy.py --real
```

### 1) 抓取并缓存股票池（沪深300，需联网）
```bash
cd "<skill>/scripts"
"<managed_python>" fetch_universe.py
```
- 成分股：`ak.index_stock_cons_csindex("000300")`，区间 2023-01-01~2024-12-31。
- 落盘 `cache/<code>_hfq_kline.csv` 与 `cache/<code>_fund.csv`（文件名含复权后缀，避免 qfq/hfq 串味）。
- 可重入：已缓存自动跳过；`time.sleep(0.4)` 节流防限流。约 300 只全覆盖。

### 2) 网格搜索选最优参数（离线）
```bash
cd "<skill>/scripts"
"<managed_python>" tune.py            # 全量（沪深300 缓存齐全后）
"<managed_python>" tune.py --quick    # 仅前 60 只快速验证流程
```
- 网格：6 组权重预设(base/tech_heavy/value_heavy/even/momentum/sent_low) × Top-N(5/8/10)。
- 排序：回撤 ≤ -25% 约束下按夏普降序；最优配置写入 `tune_best.json`。

### 3) 最终对比报告（离线）
```bash
cd "<skill>/scripts"
"<managed_python>" run_tuned.py
```
- 基线（原 10 股 + base 权重） vs 调参后（沪深300 + 网格最优），写入 `final_backtest.log`。

### 4) 样本外 / 过拟合检验（强烈推荐，离线）
```bash
cd "<skill>/scripts"
"<managed_python>" robustness.py
```
- 在 **2023（训练）** 跑网格选冠军，再在**未参与选参的 2024（测试）**验证其是否存活；
  同时构建 6 预设 × (2023/2024/2年) 稳定性矩阵，挑出两年都正收益、最鲁棒的权重。
- 输出 `robustness.log` + `robustness_result.json`。

## 关键参数（如何改）

- **因子权重预设**：见 `tune.py` 的 `WEIGHT_PRESETS`（9 元组）。可在 `run_one`/回测中直接传权重元组自定义。
- **Top-N / 调仓频率**：`tune.py` 的 `TOP_NS`、`FREQS`（默认 20 交易日）。
- **交易成本**：`multi_factor_dmi_strategy.py` 的 `TransactionCost`（佣金/印花/滑点）。
- **风险纪律**：`RiskManager(stop_loss_pct=-0.10, max_drawdown_pct=-0.20)`。
- **回测区间**：`tune.py` 的 `START/END`（或 `robustness.py` 的 `WINDOWS`）。

## ⚠️ 重要注意事项（使用前必读）

1. **过拟合风险（最高优先级）**：单年（2024）网格冠军会随复权方式/基本面是否加载在 `value_heavy`→`even`→`sent_low` 间跳变。**单年冠军 `sent_low` 在 2023+2024 两年全样本中实际亏损（收益 -6.86%、夏普 -0.45）**，证明单年搜索严重过拟合，**严禁把单年冠军直接当生产默认**。
   - **推荐生产默认权重 = `base`**（经 2023–2024 两年全样本验证最稳）：
     `weights=(0.12, 0.12, 0.08, 0.08, 0.08, 0.10, 0.10, 0.18, 0.14)`，`Top-N=5`，调仓频率 `20` 交易日。
     两年全样本：收益 **+26.09%**、夏普 **0.55**、最大回撤 -23.5%。
   - **备选 = `even`（近等权，各因子 ~0.11）**：两年全样本收益 +25.28%、夏普 0.54，与 `base` 几乎并列，鲁棒性次优、最不容易翻车。
   - 2023 训练冠军 `sent_low(top_n=8)` 在 2024 样本外仍盈利（夏普 1.35/收益 32.43%），但两年拉垮，**仅作候选、不上线**。
   - 正式上线前，务必跑 `robustness.py` 做样本外检验，或按 Regime 用 2 年全样本冠军。
2. **可复现性铁律**：固定 hfq；缓存文件名带复权后缀；舆情用 md5 确定性种子（已归一化代码去后缀）；回测窗口必须显式裁剪，勿混入预热数据。
   - *实现注记（v1.0.1 优化）*：原 `data_source.py` 的兜底 mock（K线/基本面/舆情）误用 Python `hash()` 做随机种子，因字符串哈希默认加盐导致**跨进程不固定**，已统一改用 `zlib.crc32` 稳定种子；通达信 `TDXDataSource` 改为**懒连接**（仅在 AKShare 不可用时才连服务器，且多服务器轮询），避免无谓阻塞；腾讯接口作为主源时原本忽略 `period`，已修正为仅日线走腾讯、周/月线自动转东财。
3. **数据范围**：经典管线（`fetch_universe.py`/`tune.py`/`robustness.py`）缓存为 2023–2024 沪深300；新 AKShare 加载器（`real_data_loader.py` / `--real-hs300`）覆盖 2022–2025 全样本并含时点正确基本面。换年份/换股票池需重新抓取对应缓存。
4. **非投资建议**：回测为历史模拟，含真实成本与纪律，但过去表现不代表未来；舆情为 mock，真实情绪需另接数据源。
5. **可选动量因子（v1.0.1 新增）**：新增第 10 个因子 `momentum`（短 20 日 + 长 60 日价格动量加权），`FactorWeights.momentum_weight` 默认 **0（关闭）**，故默认仍是原 9 因子基线、可复现性不受影响。启用时建议将其余 9 个权重同比缩放使总和=1（例如 momentum=0.10 时其余 ×0.909）。
6. **可选质量因子（v1.0.1 新增）**：新增第 11 个因子 `quality`，`FactorWeights.quality_weight` 默认 **0（关闭）**。与价值因子正交——价值因子用 PE/PB 便宜度 + ROE 阈值过滤，质量因子用 **ROE 水平的连续打分**（高 ROE 更优，参考值 12%）；若真实基本面提供 `gross_margin`/`debt_ratio` 则自动合成更完整的质量分（高毛利、低负债更优）。启用时同样把其余权重同比缩放使总和=1。质量因子依赖基本面数据（`fundamental_data`），务必通过 `fetch_stock_data`/CLI 自动抓取或显式传入，否则保持中性 50 不生效。
7. **真实基本面 + 真实资金流舆情接入（v1.0.1 新增，westock-mcp）**：`DataSourceManager(use_westock_real=False)` 与 `SentimentMonitor(use_westock_real=False)` 旗标默认关闭，零基线风险。
   - 基本面：`fetch_fundamental` 开启后优先读取本地缓存 `{code}_realfund.csv`（pe/pb/roe/gross_margin/debt_ratio，由腾讯自选股 `data_finance`+`data_quote` 抓取推导后写入）。已实测贵州茅台真实 ROE=29.42%、质量因子 95.96。
   - 舆情：`SentimentMonitor` 开启后读取 `{code}_realflow.csv`，由 **主力净流入/总市值** 归一化（tanh(×1000)）得到 [-1,1] 舆情分，替换原 md5 mock（如恒瑞主力净流出→-0.87，五粮液/茅台净流入→+0.54/+0.40）。
   - 数据管线：先跑 `python scripts/gen_real_cache.py`（内嵌 10 只宇宙股票的真实财务/资金流快照）生成缓存，再跑 `python multi_factor_dmi_strategy.py --westock` 做回测。
   - ⚠️ 前瞻偏差：真实基本面/资金流均为单一最新快照（2026-03-31 / 2026-08-10），对 2023-2024 历史再平衡日属非时点正确信号，仅用于验证数据接入，非实盘信号。
   - 验证：纯 mock 基线 (-0.2285578, 771442.2032) 完全不变；`--westock` 回测跨进程可复现 (+0.26459208, 1264592.0781)，且 K线一致、差异纯来自真实因子输入。

8. **真实数据自包含加载器 + 沪深300 时点正确回测 + Regime + Walk-forward（v1.0.2 新增，推荐）**：`scripts/real_data_loader.py` 提供**完全不依赖 MCP、自包含、可离线重建**的真实数据层，**从根上消除项 7 的单一快照前瞻偏差**。
   - **时点正确（point-in-time）基本面**：由全市场季报（`ak.stock_yjbb_em`，13 个报告期 2022Q1~2025Q1）逐季推导 **TTM 净利**（滚动 4 个单季求和），PE=总市值/净利TTM、PB=PE×ROE（ROE 取小数），再按**报告披露日 ffill 到交易日**——只用已披露报告，绝不使用未来数据。
   - **市值单位坑**：`ak.stock_zh_valuation_baidu` 返回【亿元】，净利TTM 为【元】，须 `×1e8` 统一；周期须用 `近五年`（近三年只覆盖 ~2023-08，早段回测会缺 PE/PB）。
   - **估值恒等式**：`PB = PE × ROE`（ROE 取小数）。⚠️ 加载器输出 `roe` 为【百分比原值】（与 mock 及 `compute_fundamental_score`/`compute_quality_score` 约定一致），`gross_margin` 为【小数】——切勿把 roe 再 `/100`，否则 ROE 贡献被清零。
   - **时点正确技术舆情（2.5）** `compute_point_in_time_sentiment(df, i)`：由个股自身历史 20 日收益+量能比推导，只用 index≤i 数据，**完全时点正确**，替换项 7 的单一未来快照资金流舆情，默认关闭（零基线影响）。
   - **市场状态 Regime（2.6）** `MarketRegime(benchmark)`：沪深300 200 日均线趋势 + 60 日波动率 z-score → 目标暴露 {1.0 牛/0.5 震荡或高波动/0.0 熊市+高波动}，全时点正确；`_rebalance` 中 `regime_exposure` 调控可用现金与空仓，默认关闭。
   - **Walk-forward 样本外验证（2.7）** `walk_forward_validation(dm, fd, bench, start, end, n_folds, warmup_days)`：把区间切为 n_folds 段连续测试窗口，每段前接 `warmup_days` 历史做指标预热（不计入收益），拼接出连续样本外净值曲线并汇报每段 + 整体 OOS 收益/年化/夏普/回撤。因权重为先前一次性固定，**任何测试期天然样本外**。
   - **CLI**：`python multi_factor_dmi_strategy.py --real-hs300 [--pit-sentiment] [--regime] [--hs300-limit N]`。自动 `build_real_dataset`（沪深300 成分股）→ 真实回测 → walk-forward；结果落地 `cache/hs300_real_results.json`。`--hs300-limit N` 可限制成分股数做快速验证。
   - **缓存文件**：`{code}_kline.csv`（hfq 日线）、`{code}_mktcap_5y.csv`（日频总市值/元）、`quarterly_reports.json`（全市场季报，~1.1 万只）、`hs300_universe.json`（成分股）、`sh000300_kline.csv`（基准）、`hs300_real_results.json`（回测+WF 结果）。
   - **关于东方财富妙想 MCP（mx-ds-mcp）**：用户已连接，但实测连接不稳定（反复掉线 / 工具索引丢失）。本加载器以 AKShare 作为稳健、可离线重建的**可插拔数据层**达成同一目标（消除前瞻、接入真实 K线/基本面），是推荐路径；mx-ds-mcp 后续稳定后可经 `data_source.py` 的插拔接口替换接入，无需改动回测引擎。
   - ⚠️ AKShare 的东方财富解析依赖 `py_mini_racer`(V8)，**不可多线程并发**，故 K线/市值抓取均**串行**（V8 隔离区并发会崩溃 Check failed: !IsConfigurablePoolInitialized()）。

## 输出物

| 文件 | 含义 |
|------|------|
| `cache/<code>_hfq_kline.csv` / `_fund.csv` | 缓存的行情与基本面 |
| `universe_hs300.txt` | 沪深300 成分股清单 |
| `tune_best.json` | 网格搜索最优配置 |
| `final_backtest.log` | 基线 vs 调参对比报告 |
| `robustness.log` / `robustness_result.json` | 样本外检验与稳定性矩阵 |

## 示例输出（仓库内对照）

`examples/` 目录收录各脚本**真实跑出的结果**，调用前可先对照预期输出格式与数量级：

- `examples/final_backtest.log` — `run_tuned.py` 的基线 vs 调参对比报告
- `examples/robustness.log` / `robustness_result.json` — `robustness.py` 的样本外检验与稳定性矩阵
- `examples/tune_best.json` — `tune.py` 网格搜索最优配置
- `examples/real_backtest_fundamental.log` — `multi_factor_dmi_strategy.py --real` 单轮真实回测样例
- `examples/README.md` — 上述文件说明 + 关键结论速览
- `examples/hs300_real_backtest_2026-08-11.md` — `--real-hs300` 全样本 + walk-forward 实测（288 只、2022–2025 时点正确数据，含 OOS 诚实结论：全样本 +32% 在样本外塌到 ~0%，regime 在 V 型市拖累）

> 这些结果是 2023–2024 沪深300 的历史回测演示，仅作方法参考，不构成投资建议。

## 调用示例（给 agent）

- 「用沪深300 跑一轮多因子回测并调参」→ 执行步骤 1→2→3。
- 「这个策略会不会过拟合？」→ 执行步骤 4，汇报 2023 冠军在 2024 的夏普衰减与稳定预设。
- 「接入真实 A股 数据跑一轮」→ 先 `python scripts/gen_real_cache.py`（生成 `{code}_realfund.csv`/`{code}_realflow.csv`），再 `python multi_factor_dmi_strategy.py --westock`（确定性 mock K线 + 真实基本面 + 真实资金流舆情）。需先连接 westock-mcp 连接器并抓取数据。
- 「用沪深300 真实数据做时点正确回测 + 样本外验证」→ `python multi_factor_dmi_strategy.py --real-hs300`（默认 300 只、2022–2025、时点正确基本面 + 真实 K线，自动跑 walk-forward；加 `--pit-sentiment` 启用时点正确技术舆情，加 `--regime` 启用市场状态暴露调控，加 `--hs300-limit 30` 仅取前 30 只快速验证）。**无需任何 MCP、缓存齐全后可完全离线重跑**。
- 「换个权重试试价值因子」→ 在 `WEIGHT_PRESETS` 加/改预设后跑 `tune.py` 或 `run_tuned.py`。
