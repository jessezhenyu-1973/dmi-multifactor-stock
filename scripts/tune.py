"""
多因子策略参数网格搜索（离线，基于已缓存的沪深300真实数据）
============================================================
- 离线加载 ./cache 中的 K 线与基本面，不触发任何网络请求
- 网格：因子权重预设 × Top-N
- 目标：以夏普比率为主、回撤为约束挑选最优配置
- 输出：对比表 + 最优配置（写入 tune_best.json）

用法：
  python tune.py            # 全量搜索（沪深300 缓存齐全后）
  python tune.py --quick    # 仅用前 60 只缓存标的快速验证流程
"""
import os
import sys
import glob
import json
import logging
import argparse
import numpy as np
import pandas as pd

from multi_factor_dmi_strategy import (
    MultiFactorDMIStrategy, FactorWeights, RiskManager,
    SentimentMonitor, TransactionCost,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tune")

# 缓存目录：可用环境变量 QF_CACHE_DIR 覆盖（便于复用已抓取的沪深300缓存）
CACHE = os.environ.get(
    "QF_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"),
)
START = "20240101"
END = "20241231"

# 权重顺序：dmi, macd, kdj, rsi, bol, wyk, m135, fund, sent
WEIGHT_PRESETS = {
    "base":       (0.12, 0.12, 0.08, 0.08, 0.08, 0.10, 0.10, 0.18, 0.14),
    "tech_heavy": (0.15, 0.15, 0.10, 0.10, 0.10, 0.12, 0.12, 0.10, 0.06),
    "value_heavy":(0.08, 0.08, 0.06, 0.06, 0.06, 0.08, 0.08, 0.35, 0.15),
    "even":       (0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.12, 0.11),
    "momentum":   (0.16, 0.16, 0.09, 0.08, 0.07, 0.14, 0.14, 0.08, 0.08),
    "sent_low":   (0.13, 0.13, 0.09, 0.09, 0.09, 0.11, 0.11, 0.20, 0.05),
}
TOP_NS = [5, 8, 10]
FREQS = [20]  # 调仓频率固定，减少组合数（如需也可展开）


def load_universe(limit: int = None):
    """从本地缓存离线加载沪深300数据"""
    kline_files = sorted(glob.glob(os.path.join(CACHE, "*_kline.csv")))
    if limit:
        kline_files = kline_files[:limit]
    data_map, fundamental_data = {}, {}
    lo = pd.Timestamp(START)
    hi = pd.Timestamp(END)
    for kf in kline_files:
        # 缓存文件名形如 <code>_<adjust>_kline.csv，剥离后缀恢复纯代码
        raw = os.path.basename(kf).replace("_kline.csv", "")
        code = raw.replace("_hfq", "").replace("_qfq", "")
        k = pd.read_csv(kf, index_col=0, parse_dates=True)
        k.index = pd.to_datetime(k.index)
        # 裁剪到回测区间（避免混入 2023 预热数据导致窗口不一致）
        k = k[(k.index >= lo) & (k.index <= hi)]
        # 基本面缓存文件名不含复权后缀：<code>_fund.csv
        ff = os.path.join(CACHE, code + "_fund.csv")
        fund = None
        if os.path.exists(ff):
            fund = pd.read_csv(ff, index_col=0, parse_dates=True)
            fund.index = pd.to_datetime(fund.index)
            fund = fund.reindex(k.index).ffill().bfill()
            fund = fund[['pe', 'pb', 'roe']]
            fund = fund[(fund.index >= lo) & (fund.index <= hi)]
        if k.empty:
            continue
        data_map[code] = k
        if fund is not None and not fund.empty:
            fundamental_data[code] = fund
    logger.info(f"离线加载完成：K线 {len(data_map)} 只，基本面 {len(fundamental_data)} 只（区间 {START}~{END}）")
    return data_map, fundamental_data


def run_one(data_map, fundamental_data, weights_tuple, top_n, freq,
            cost: TransactionCost):
    w = FactorWeights(
        dmi_weight=weights_tuple[0], macd_weight=weights_tuple[1],
        kdj_weight=weights_tuple[2], rsi_weight=weights_tuple[3],
        bollinger_weight=weights_tuple[4], wyckoff_weight=weights_tuple[5],
        method135_weight=weights_tuple[6], fundamental_weight=weights_tuple[7],
        sentiment_weight=weights_tuple[8],
    )
    strat = MultiFactorDMIStrategy(
        initial_capital=1_000_000,
        top_n=top_n,
        factor_weights=w,
        risk_manager=RiskManager(stop_loss_pct=-0.10, max_drawdown_pct=-0.20, max_positions=max(top_n, 10)),
        sentiment_monitor=SentimentMonitor(use_real_data=False),
        rebalance_freq=freq,
        use_real_data=False,  # 数据已离线提供，不再联网
        cost=cost,
    )
    result = strat.run_backtest(
        data_map=data_map, fundamental_data=fundamental_data,
        start_date=START, end_date=END,
    )
    stats = strat._calculate_stats(result)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="仅用前 60 只缓存标的快速验证")
    args = ap.parse_args()

    cost = TransactionCost()  # 真实成本：佣金0.03%/印花税0.05%/滑点0.1%
    data_map, fundamental_data = load_universe(limit=60 if args.quick else None)

    rows = []
    for pname, wt in WEIGHT_PRESETS.items():
        for top_n in TOP_NS:
            for freq in FREQS:
                try:
                    s = run_one(data_map, fundamental_data, wt, top_n, freq, cost)
                    rows.append({
                        "preset": pname, "top_n": top_n, "freq": freq,
                        "total_return": s['total_return'],
                        "annual_return": s['annual_return'],
                        "sharpe": s['sharpe_ratio'],
                        "max_drawdown": s['max_drawdown'],
                        "win_rate": s['win_rate'],
                        "trades": s['total_trades'],
                    })
                    logger.info(f"{pname:10s} top_n={top_n} freq={freq} -> "
                                f"收益 {s['total_return']:.2%} 夏普 {s['sharpe_ratio']:.2f} "
                                f"回撤 {s['max_drawdown']:.2%} 胜率 {s['win_rate']:.2%}")
                except Exception as e:
                    logger.warning(f"{pname} top_n={top_n} 失败: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("无任何配置跑通，请检查缓存是否齐全")
        return

    # 排序：优先满足回撤约束（-25% 以内），再按夏普降序
    feasible = df[df['max_drawdown'] <= 0.25].copy()
    pool = feasible if not feasible.empty else df.copy()
    pool = pool.sort_values("sharpe", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 92)
    print("参数网格搜索结果（沪深300 · 2024 · 含交易成本） 按夏普降序")
    print("=" * 92)
    show = pool.copy()
    show['total_return'] = show['total_return'].map(lambda x: f"{x:.2%}")
    show['annual_return'] = show['annual_return'].map(lambda x: f"{x:.2%}")
    show['sharpe'] = show['sharpe'].map(lambda x: f"{x:.2f}")
    show['max_drawdown'] = show['max_drawdown'].map(lambda x: f"-{x:.2%}")
    show['win_rate'] = show['win_rate'].map(lambda x: f"{x:.2%}")
    print(show.to_string(index=False))
    print("=" * 92)

    best = pool.iloc[0]
    best_cfg = {
        "preset": best['preset'], "top_n": int(best['top_n']), "freq": int(best['freq']),
        "weights": list(WEIGHT_PRESETS[best['preset']]),
        "sharpe": float(best['sharpe']), "total_return": float(best['total_return']),
        "max_drawdown": float(best['max_drawdown']), "win_rate": float(best['win_rate']),
        "trades": int(best['trades']),
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_best.json"), "w") as f:
        json.dump(best_cfg, f, indent=2, ensure_ascii=False)
    logger.info(f"最优配置已写入 tune_best.json: {best_cfg}")


if __name__ == "__main__":
    main()
