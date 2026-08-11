"""后台任务：沪深300 全样本真实回测 + walk-forward 样本外验证。
构建数据(串行 AKShare) -> 时点正确回测 -> 5 折 walk-forward，结果落地 cache/hs300_real_results.json。
断点续跑：每只股票 K线/市值命中缓存即跳过，可重复执行。
"""
import os, json, logging, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hs300_full")

import pandas as pd, numpy as np
from real_data_loader import build_real_dataset, get_hs300_codes
from multi_factor_dmi_strategy import (
    MultiFactorDMIStrategy, FactorWeights, RiskManager, SentimentMonitor, MarketRegime,
)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
RESULT_PATH = os.path.join(CACHE, "hs300_real_results.json")

def main():
    t0 = time.time()
    codes = get_hs300_codes()  # 全部 300
    logger.info(f"HS300 成分股: {len(codes)} 只，开始构建真实数据集（串行 AKShare）")
    dm, fd, bench = build_real_dataset(codes, "20220101", "20251231")
    logger.info(f"数据集就绪 K线 {len(dm)} / 基本面 {len(fd)} / 基准 {len(bench)}，耗时 {time.time()-t0:.0f}s")

    fw = FactorWeights(
        dmi_weight=0.12, macd_weight=0.12, kdj_weight=0.08, rsi_weight=0.08,
        bollinger_weight=0.08, wyckoff_weight=0.10, method135_weight=0.10,
        fundamental_weight=0.18, sentiment_weight=0.14,
    )
    rm = RiskManager(stop_loss_pct=-0.10, max_drawdown_pct=-0.20, max_positions=30)

    out = {}
    for tag, pit_flag, reg_flag in [("base", False, False),
                                    ("pit", True, False),
                                    ("pit_regime", True, True)]:
        strat = MultiFactorDMIStrategy(
            initial_capital=1_000_000, top_n=15, factor_weights=fw, risk_manager=rm,
            sentiment_monitor=SentimentMonitor(use_real_data=False), rebalance_freq=20,
            use_real_data=True, use_point_in_time_sentiment=pit_flag,
            use_regime=reg_flag, regime=MarketRegime(bench) if reg_flag else None,
        )
        res = strat.run_backtest(data_map=dm, fundamental_data=fd, start_date="20220101", end_date="20251231")
        wf = strat.walk_forward_validation(dm, fd, bench, "20220101", "20251231", n_folds=5)
        stats = strat._calculate_stats(res)
        out[tag] = {
            "full_sample": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in stats.items()},
            "wf_overall": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in wf["overall"].items() if k != "n_folds"},
            "wf_folds": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in f.items()} for f in wf["folds"]],
            "trades": len(strat.trade_log),
        }
        logger.info(f"[{tag}] 全样本收益={stats['total_return']:.2%} 夏普={stats['sharpe_ratio']:.2f} "
                    f"回撤={stats['max_drawdown']:.2%} | WF-OOS 收益={wf['overall']['total_return']:.2%} "
                    f"夏普={wf['overall']['sharpe']:.2f} 回撤={wf['overall']['max_drawdown']:.2%}")
        # 落盘（每跑完一档即保存，防中断丢结果）
        json.dump(out, open(RESULT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    logger.info(f"全部完成，总耗时 {time.time()-t0:.0f}s，结果已写 {RESULT_PATH}")

if __name__ == "__main__":
    main()
