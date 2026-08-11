"""烟测：用已缓存 15 只验证 real-hs300 + walk-forward 整链路（无网络）。"""
import logging
logging.basicConfig(level=logging.INFO)
from real_data_loader import build_real_dataset
from multi_factor_dmi_strategy import (
    MultiFactorDMIStrategy, FactorWeights, RiskManager, SentimentMonitor, MarketRegime,
)

CODES = [
    "000657.SZ", "000988.SZ", "002202.SZ", "002532.SZ", "002558.SZ",
    "002602.SZ", "002837.SZ", "301165.SZ", "301308.SZ", "600118.SH",
    "600221.SH", "600549.SH", "688072.SH", "688183.SH", "688521.SH",
]

fw = FactorWeights(
    dmi_weight=0.12, macd_weight=0.12, kdj_weight=0.08, rsi_weight=0.08,
    bollinger_weight=0.08, wyckoff_weight=0.10, method135_weight=0.10,
    fundamental_weight=0.18, sentiment_weight=0.14,
)
rm = RiskManager(stop_loss_pct=-0.10, max_drawdown_pct=-0.20, max_positions=15)

dm, fd, bench = build_real_dataset(CODES, "20220101", "20251231")
print(f"数据集: K线 {len(dm)} 只, 基本面 {len(fd)} 只, 基准 {len(bench)} 行")

for pit, reg in [(False, False), (True, False), (True, True)]:
    strat = MultiFactorDMIStrategy(
        initial_capital=1_000_000, top_n=5, factor_weights=fw, risk_manager=rm,
        sentiment_monitor=SentimentMonitor(use_real_data=False), rebalance_freq=20,
        use_real_data=True, use_point_in_time_sentiment=pit,
        use_regime=reg, regime=MarketRegime(bench) if reg else None,
    )
    res = strat.run_backtest(data_map=dm, fundamental_data=fd, start_date="20220101", end_date="20251231")
    wf = strat.walk_forward_validation(dm, fd, bench, "20220101", "20251231", n_folds=4)
    o = wf["overall"]
    print(f"\n[pit={pit} regime={reg}] 全样本收益末值={res['return'].iloc[-1]:.2%} | "
          f"WF-OOS 收益={o['total_return']:.2%} 年化={o['annual_return']:.2%} 夏普={o['sharpe']:.2f} 回撤={o['max_drawdown']:.2%} | 交易={len(strat.trade_log)}")
    for f in wf["folds"]:
        print(f"   段{f['fold']} {f['start'][:10]}~{f['end'][:10]} 收益={f['total_return']:.2%} 夏普={f['sharpe']:.2f}")
print("\n烟测完成：链路无异常。")
