---
name: dmi-multifactor-stock
description: >-
  9 因子 A 股多因子选股与回测引擎。因子含 DMI + MACD/KDJ/RSI/布林带/威科夫成交量/135 战法 + 基本面(PE/PB/ROE) + 舆情，含交易成本模型(佣金0.03%/印花0.05%/滑点0.1%)、-10% 硬止损、-20% 最大回撤清仓纪律。支持：沪深300 成分股抓取缓存、离线多因子回测、因子权重/Top-N 网格调优、样本外(过拟合)检验、对比报告。Triggers: 多因子选股、量化策略、股票回测、A股回测、DMI选股、参数调优、网格搜索、沪深300回测、因子权重、选股策略、因子模型、动量选股、价值选股。Use when the user wants to build/run/tune a quantitative stock-selection strategy, backtest A-share factors, or validate a strategy out-of-sample.
description_zh: "9 因子 A 股多因子选股与回测引擎：DMI+技术+基本面+舆情，含成本模型与样本外检验"
description_en: "9-factor A-share multifactor stock-selection & backtest engine with cost model and out-of-sample validation"
version: 1.0.0
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
3. **数据范围**：当前缓存为 2023-2024 A 股。换年份/换股票池需重新 `fetch_universe.py` 抓取。
4. **非投资建议**：回测为历史模拟，含真实成本与纪律，但过去表现不代表未来；舆情为 mock，真实情绪需另接数据源。

## 输出物

| 文件 | 含义 |
|------|------|
| `cache/<code>_hfq_kline.csv` / `_fund.csv` | 缓存的行情与基本面 |
| `universe_hs300.txt` | 沪深300 成分股清单 |
| `tune_best.json` | 网格搜索最优配置 |
| `final_backtest.log` | 基线 vs 调参对比报告 |
| `robustness.log` / `robustness_result.json` | 样本外检验与稳定性矩阵 |

## 调用示例（给 agent）

- 「用沪深300 跑一轮多因子回测并调参」→ 执行步骤 1→2→3。
- 「这个策略会不会过拟合？」→ 执行步骤 4，汇报 2023 冠军在 2024 的夏普衰减与稳定预设。
- 「换个权重试试价值因子」→ 在 `WEIGHT_PRESETS` 加/改预设后跑 `tune.py` 或 `run_tuned.py`。
