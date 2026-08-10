"""
最终回测：用最优参数对沪深300跑真实回测（含成本），并与原10股基线对比
=========================================================================
- 复用 tune.py 的离线加载与单次回测函数（不联网）
- 基线：原始 10 只权重股 + base 权重 + 成本
- 调参后：沪深300 全样本 + 网格搜索最优权重/Top-N + 成本
- 结果写入 final_backtest.log
"""
import os
import json
import logging

from tune import load_universe, run_one, WEIGHT_PRESETS
from multi_factor_dmi_strategy import TransactionCost

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_tuned")

ORIGINAL_10 = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH",
               "000858.SZ", "600519.SH", "000333.SZ", "601318.SH",
               "002594.SZ", "600276.SH"]


def main():
    cost = TransactionCost()
    data_map, fundamental_data = load_universe()

    orig_keys = [c.split('.')[0] for c in ORIGINAL_10]
    orig_dm = {k: data_map[k] for k in orig_keys if k in data_map}
    orig_fd = {k: fundamental_data[k] for k in orig_keys if k in fundamental_data}

    # —— 基线：原始10股 + base权重 + 成本 ——
    base = run_one(orig_dm, orig_fd, WEIGHT_PRESETS["base"], top_n=5, freq=20, cost=cost)
    logger.info(f"基线(10股) 收益 {base['total_return']:.2%} 夏普 {base['sharpe_ratio']:.2f} "
                f"回撤 {base['max_drawdown']:.2%} 胜率 {base['win_rate']:.2%}")

    # —— 调参后：沪深300 + 最优权重 ——
    best = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_best.json")))
    best_w = tuple(best["weights"])
    tuned = run_one(data_map, fundamental_data, best_w, top_n=best["top_n"], freq=best["freq"], cost=cost)
    logger.info(f"调参(hs300) 收益 {tuned['total_return']:.2%} 夏普 {tuned['sharpe_ratio']:.2f} "
                f"回撤 {tuned['max_drawdown']:.2%} 胜率 {tuned['win_rate']:.2%}")

    # —— 对比报告 ——
    out = []
    out.append("=" * 72)
    out.append("最终对比报告（2024 全年 · 真实行情 · 含交易成本：佣金0.03%/印花0.05%/滑点0.1%）")
    out.append("=" * 72)
    out.append(f"{'指标':<14}{'原10股(base)':>20}{'沪深300(调参)':>20}")
    out.append("-" * 72)
    out.append(f"{'总收益率':<14}{base['total_return']:>19.2%}{tuned['total_return']:>19.2%}")
    out.append(f"{'年化收益':<14}{base['annual_return']:>19.2%}{tuned['annual_return']:>19.2%}")
    out.append(f"{'夏普比率':<14}{base['sharpe_ratio']:>20.2f}{tuned['sharpe_ratio']:>20.2f}")
    out.append(f"{'最大回撤':<14}{'-'+format(base['max_drawdown'],'.2%'):>20}{'-'+format(tuned['max_drawdown'],'.2%'):>20}")
    out.append(f"{'胜率':<14}{base['win_rate']:>19.2%}{tuned['win_rate']:>19.2%}")
    out.append(f"{'交易次数':<14}{base['total_trades']:>20d}{tuned['total_trades']:>20d}")
    out.append("-" * 72)
    out.append(f"沪深300 样本数: {len(data_map)} 只（其中含基本面 {len(fundamental_data)} 只）")
    out.append(f"最优配置: preset={best['preset']} top_n={best['top_n']} freq={best['freq']}")
    out.append(f"最优权重(dmi,macd,kdj,rsi,bol,wyk,m135,fund,sent): {[round(x,3) for x in best_w]}")
    out.append("=" * 72)

    report = "\n".join(out)
    print(report)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_backtest.log"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
