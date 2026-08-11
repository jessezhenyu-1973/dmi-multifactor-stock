"""
按今天的收盘数据跑一次：沪深300成分股 9因子回测
使用 cache_latest/ 中的最新缓存数据
"""
import os, sys, time, logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source import DataSourceManager
from multi_factor_dmi_strategy import (
    MultiFactorDMIStrategy, FactorWeights, RiskManager, TransactionCost,
    compute_dmi, compute_macd, compute_kdj, compute_rsi, compute_bollinger,
    compute_wyckoff, compute_ma_lines, compute_fundamental_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_latest")

CACHE_DIR = os.environ.get(
    "QF_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest"),
)

START = "20250101"
END = "20260810"

def main():
    logger.info("=" * 70)
    logger.info("DMI 多因子选股策略 — 沪深300成分股 回测（今日收盘数据）")
    logger.info("=" * 70)

    # 1) 加载沪深300成分股
    universe_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt")
    with open(universe_file, "r") as f:
        codes = [line.strip() for line in f if line.strip()]
    logger.info(f"沪深300 成分股: {len(codes)} 只")

    # 2) 加载数据
    mgr = DataSourceManager(use_real_data=False, cache_dir=CACHE_DIR)
    data_map = {}
    fund_map = {}
    t0 = time.time()
    for i, code in enumerate(codes):
        try:
            k = mgr.fetch_kline(code, START, END, "daily", "hfq")
            if k is not None and not k.empty:
                data_map[code] = k
                fnd = mgr.fetch_fundamental(code, START, END)
                if fnd is not None and not fnd.empty:
                    fnd = fnd.reindex(k.index).ffill().bfill()
                    fund_map[code] = fnd
        except Exception as e:
            logger.debug(f"{code} 加载失败: {e}")
        if (i + 1) % 100 == 0:
            logger.info(f"数据加载进度 {i+1}/{len(codes)} | 成功 {len(data_map)}")

    elapsed = time.time() - t0
    logger.info(f"数据加载完成: {len(data_map)}/{len(codes)} 只K线, {len(fund_map)} 只基本面, 用时 {elapsed:.1f}s")

    if len(data_map) == 0:
        logger.error("没有可用的股票数据，回测终止")
        return

    # 3) 计算全部技术指标
    logger.info("计算技术指标...")
    t1 = time.time()
    for code, df in data_map.items():
        df = compute_dmi(df, period=14)
        df = compute_macd(df)
        df = compute_kdj(df)
        df = compute_rsi(df)
        df = compute_bollinger(df)
        df = compute_wyckoff(df)
        df = compute_ma_lines(df)
        data_map[code] = df
    logger.info(f"技术指标计算完成, 用时 {time.time()-t1:.1f}s")

    # 4) 运行回测
    logger.info("开始回测...")
    strategy = MultiFactorDMIStrategy(
        initial_capital=1_000_000,
        top_n=5,
        factor_weights=FactorWeights(
            dmi_weight=0.12, macd_weight=0.12, kdj_weight=0.08,
            rsi_weight=0.08, bollinger_weight=0.08, wyckoff_weight=0.10,
            method135_weight=0.10, fundamental_weight=0.18, sentiment_weight=0.14
        ),
        risk_manager=RiskManager(stop_loss_pct=-0.10, max_drawdown_pct=-0.20, max_positions=10),
        sentiment_monitor=None,  # 内部会默认用mock
        rebalance_freq=20,
        use_real_data=False,
        data_source=None,  # 已有数据，不需要重新获取
        cost=TransactionCost(commission=0.0003, min_commission=5.0, stamp=0.0005, slippage=0.001),
    )

    # 重写 _rebalance 使其使用已有数据
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    logger.info(f"回测日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 个交易日")

    # 手动执行回测逻辑
    nav = 1_000_000
    cash = 1_000_000
    peak_nav = nav
    daily_records = []
    current_positions = {}
    position_shares = {}
    position_cost_basis = {}
    position_dates = {}
    trade_log = []
    risk_mgr = strategy.risk_manager

    for di, date in enumerate(all_dates):
        # 更新净值
        risk_mgr.update_nav(nav)
        nav = cash
        for code, shares in position_shares.items():
            if code in data_map and date in data_map[code].index:
                nav += shares * data_map[code].loc[date, 'close']
        risk_mgr.update_nav(nav)

        # 止损
        to_sell = []
        for code in list(current_positions.keys()):
            if code not in data_map or date not in data_map[code].index:
                continue
            current_price = data_map[code].loc[date, 'close']
            pnl = (current_price - current_positions[code]) / current_positions[code]
            if pnl <= -0.10:
                to_sell.append((code, current_price, "硬止损"))

        for code, sp, reason in to_sell:
            shares = position_shares[code]
            cost = position_cost_basis[code]
            fill = sp * 0.999
            gross = fill * shares
            commission = max(gross * 0.0003, 5.0)
            stamp = gross * 0.0005
            proceeds = gross - commission - stamp
            cash += proceeds
            pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
            trade_log.append({
                'code': code, 'buy_date': position_dates.get(code, ''),
                'buy_price': current_positions.get(code, sp), 'sell_price': sp,
                'reason': reason, 'pnl_pct': pnl_pct
            })
            risk_mgr.close_trade(code, sp, str(date), reason)
            position_shares.pop(code, None)
            position_cost_basis.pop(code, None)
            position_dates.pop(code, None)
            current_positions.pop(code, None)

        # 重新盯市
        nav = cash
        for code, shares in position_shares.items():
            if code in data_map and date in data_map[code].index:
                nav += shares * data_map[code].loc[date, 'close']
        risk_mgr.update_nav(nav)

        # 最大回撤清仓
        if risk_mgr.check_max_drawdown():
            logger.warning("触发最大回撤！清仓")
            for code in list(position_shares.keys()):
                if code in data_map and date in data_map[code].index:
                    sp = data_map[code].loc[date, 'close']
                    shares = position_shares[code]
                    fill = sp * 0.999
                    gross = fill * shares
                    commission = max(gross * 0.0003, 5.0)
                    stamp = gross * 0.0005
                    proceeds = gross - commission - stamp
                    cash += proceeds
                    cost = position_cost_basis.get(code, gross)
                    pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
                    trade_log.append({'code': code, 'buy_date': position_dates.get(code, ''),
                                     'buy_price': current_positions.get(code, sp), 'sell_price': sp,
                                     'reason': '最大回撤', 'pnl_pct': pnl_pct})
                    risk_mgr.close_trade(code, sp, str(date), '最大回撤')
                    position_shares.pop(code, None)
                    position_cost_basis.pop(code, None)
                    position_dates.pop(code, None)
                    current_positions.pop(code, None)
            nav = cash

        # 调仓
        if di % 20 == 0:
            # 计算所有股票综合得分
            scores = {}
            for code, df in data_map.items():
                if date not in df.index:
                    continue
                i = df.index.get_loc(date)

                # DMI
                if i < 14:
                    dmi_score = 50
                else:
                    cur = df.iloc[i]
                    if cur['+DI'] <= cur['-DI']:
                        dmi_score = 0
                    elif cur['ADX'] > 25:
                        dmi_score = min((cur['+DI'] - cur['-DI']) * 2, 100)
                    else:
                        dmi_score = 30

                # MACD
                if 'DIF' not in df.columns or i < 35:
                    macd_score = 50
                else:
                    cur, prev = df.iloc[i], df.iloc[i-1]
                    macd_score = 50
                    if cur['DIF'] > cur['DEA']: macd_score += 20
                    if cur['DIF'] > 0: macd_score += 15
                    if prev['DIF'] <= prev['DEA'] and cur['DIF'] > cur['DEA']: macd_score += 15
                    if cur['MACD_bar'] > 0: macd_score += 10
                    if cur['MACD_bar'] > prev['MACD_bar']: macd_score += 10

                # KDJ
                if 'K' not in df.columns or i < 9:
                    kdj_score = 50
                else:
                    cur, prev = df.iloc[i], df.iloc[i-1]
                    kdj_score = 50
                    if cur['K'] < 20: kdj_score += 25
                    elif cur['K'] > 80: kdj_score -= 25
                    if prev['K'] <= prev['D'] and cur['K'] > cur['D']: kdj_score += 20
                    elif prev['K'] >= prev['D'] and cur['K'] < cur['D']: kdj_score -= 20
                    if cur['J'] < 0: kdj_score += 10
                    if cur['J'] > 100: kdj_score -= 10

                # RSI
                if 'RSI' not in df.columns or i < 14:
                    rsi_score = 50
                else:
                    r = df.iloc[i]['RSI']
                    if r < 20: rsi_score = 95
                    elif r < 30: rsi_score = 80 + (30-r)*1.5
                    elif r < 50: rsi_score = 50 + (50-r)
                    elif r < 70: rsi_score = 50 - (r-50)
                    elif r < 80: rsi_score = 30 - (r-70)*2
                    else: rsi_score = 10

                # Bollinger
                if 'pct_b' not in df.columns or i < 20:
                    bb_score = 50
                else:
                    pct_b = df.iloc[i]['pct_b']
                    bb_score = min(max(50 + (0.5-pct_b)*60, 0), 100)
                    if df.iloc[i].get('bandwidth', 0) < 0.06: bb_score += 10

                # Wyckoff
                if 'rel_volume' not in df.columns or i < 20:
                    wyckoff_score = 50
                else:
                    cur, prev = df.iloc[i], df.iloc[i-1]
                    rel_vol = cur['rel_volume']
                    price_chg = (cur['close'] - prev['close']) / prev['close']
                    wyckoff_score = 50
                    if price_chg > 0 and rel_vol > 1.5: wyckoff_score += 30
                    elif price_chg < 0 and rel_vol > 1.5: wyckoff_score -= 30
                    if rel_vol > 2.5:
                        if price_chg <= 0: wyckoff_score += 10
                        else: wyckoff_score -= 10
                    if rel_vol < 0.6: wyckoff_score += 5

                # 135 战法
                if 'ma13' not in df.columns or i < 34:
                    m135_score = 50
                else:
                    cur, prev = df.iloc[i], df.iloc[i-1]
                    m135_score = 50
                    if cur['close'] > cur['ma13']: m135_score += 20
                    if cur['ma13'] > cur['ma34']: m135_score += 20
                    if cur['ma13'] > prev['ma13']: m135_score += 20
                    if prev['ma13'] <= prev['ma34'] and cur['ma13'] > cur['ma34']: m135_score += 20
                    if cur.get('rel_volume', 1.0) > 1.3 and cur['close'] > cur['ma13']: m135_score += 10

                # 基本面
                fund_score = 50.0
                if code in fund_map and date in fund_map[code].index:
                    try:
                        fund_df = fund_map[code]
                        fund_score = compute_fundamental_score(fund_df).get(date, 50.0)
                    except:
                        pass

                # 舆情（mock）
                import hashlib
                norm_code = code.split('.')[0]
                seed = int(hashlib.md5(f"{norm_code}{date}".encode()).hexdigest(), 16) % (2**32)
                np.random.seed(seed)
                sentiment = np.random.uniform(-0.5, 0.5)
                sentiment_factor = (sentiment + 1) / 2 * 100

                composite = (
                    0.12 * dmi_score + 0.12 * macd_score + 0.08 * kdj_score +
                    0.08 * rsi_score + 0.08 * bb_score + 0.10 * wyckoff_score +
                    0.10 * m135_score + 0.18 * fund_score + 0.14 * sentiment_factor
                )
                scores[code] = composite

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = set(c for c, _ in ranked[:5])
            current_held = set(current_positions.keys())

            # 卖出不在新组合中的
            for code in current_held - selected:
                if date in data_map[code].index:
                    sp = data_map[code].loc[date, 'close']
                    shares = position_shares[code]
                    fill = sp * 0.999
                    gross = fill * shares
                    commission = max(gross * 0.0003, 5.0)
                    stamp = gross * 0.0005
                    proceeds = gross - commission - stamp
                    cash += proceeds
                    cost = position_cost_basis.get(code, gross)
                    pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
                    trade_log.append({'code': code, 'buy_date': position_dates.get(code, ''),
                                     'buy_price': current_positions.get(code, sp), 'sell_price': sp,
                                     'reason': '调仓卖出', 'pnl_pct': pnl_pct})
                    risk_mgr.close_trade(code, sp, str(date), '调仓卖出')
                    position_shares.pop(code, None)
                    position_cost_basis.pop(code, None)
                    position_dates.pop(code, None)
                    current_positions.pop(code, None)

            # 买入新的
            to_buy = selected - current_held
            if to_buy and cash > 0:
                equal_budget = cash / len(to_buy)
                for code in to_buy:
                    if date in data_map[code].index:
                        sp = data_map[code].loc[date, 'close']
                        fill = sp * 1.001
                        shares = int(equal_budget / fill / 100) * 100
                        if shares <= 0:
                            continue
                        gross = fill * shares
                        commission = max(gross * 0.0003, 5.0)
                        total_cost = gross + commission
                        if total_cost > cash:
                            shares = int((cash / (fill * 1.0003)) / 100) * 100
                            if shares <= 0:
                                continue
                            gross = fill * shares
                            commission = max(gross * 0.0003, 5.0)
                            total_cost = gross + commission
                        cash -= total_cost
                        current_positions[code] = sp
                        position_dates[code] = str(date)
                        position_shares[code] = shares
                        position_cost_basis[code] = total_cost
                        risk_mgr.add_trade(type('TradeRecord', (), {
                            'code': code, 'buy_date': str(date), 'buy_price': sp, 'reason': '选股买入'
                        })())

        # 记录
        nav = cash
        for code, shares in position_shares.items():
            if code in data_map and date in data_map[code].index:
                nav += shares * data_map[code].loc[date, 'close']
        if nav > peak_nav:
            peak_nav = nav
        drawdown = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0
        daily_records.append({
            'date': str(date), 'nav': nav,
            'return': (nav / 1_000_000) - 1,
            'drawdown': drawdown,
            'positions': len(current_positions)
        })

    # 计算统计指标
    result_df = pd.DataFrame(daily_records)
    nav_returns = result_df['nav'].pct_change().fillna(0)
    total_return = result_df['return'].iloc[-1]
    trading_days = len(result_df)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1
    risk_free_rate = 0.03
    daily_std = nav_returns.std()
    sharpe_ratio = (nav_returns.mean() - risk_free_rate / 252) / daily_std * np.sqrt(252) if daily_std > 0 else 0
    max_drawdown = abs(result_df['drawdown'].min())
    win_trades = [t for t in trade_log if t['pnl_pct'] > 0]
    win_rate = len(win_trades) / len(trade_log) if trade_log else 0

    # 打印结果
    print("\n" + "=" * 70)
    print("📊 DMI 多因子选股策略回测结果（沪深300成分股 · 今日收盘数据）")
    print("=" * 70)
    print(f"回测区间:     {all_dates[0]} ~ {all_dates[-1]} ({trading_days}个交易日)")
    print(f"股票池:       沪深300成分股 ({len(codes)}只)")
    print(f"有效数据:     {len(data_map)}只K线, {len(fund_map)}只基本面")
    print(f"初始资金:     1,000,000 元")
    print(f"持仓数:       5 只")
    print(f"调仓频率:     20 交易日")
    print("-" * 70)
    print(f"总收益率:     {total_return:+.2%}")
    print(f"年化收益:     {annual_return:+.2%}")
    print(f"夏普比率:     {sharpe_ratio:.2f}")
    print(f"最大回撤:     {max_drawdown:.2%}")
    print(f"胜率:         {win_rate:.2%}")
    print(f"交易次数:     {len(trade_log)}")
    print(f"最终净值:     {nav:,.0f} 元")
    print("=" * 70)

    # 打印最近交易
    print("\n📋 最近20笔交易:")
    print("-" * 70)
    for t in trade_log[-20:]:
        print(f"  {t['code']} | 买入: {t['buy_price']:.2f} -> 卖出: {t['sell_price']:.2f} | "
              f"{t['pnl_pct']:+.2%} | {t['reason']}")

    # 保存结果
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_backtest_result.json")
    stats = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trade_log),
        'final_nav': nav,
        'trading_days': trading_days,
        'start_date': str(all_dates[0]),
        'end_date': str(all_dates[-1]),
    }
    with open(result_path, 'w') as f:
        pd.DataFrame([stats]).to_json(f, indent=2)
    print(f"\n结果已保存到: {result_path}")

if __name__ == "__main__":
    main()
