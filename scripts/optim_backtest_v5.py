"""
优化版 DMI 多因子选股策略 v5
v4问题分析：
1. 最大回撤 -22.20% — 硬止损触发太晚，部分个股止损高达-15%
2. 最大连亏71天 — 仍然偏长
3. 平均亏损-7.89%偏大，需收紧止损

v5改进：
1. 引入ATR动态止损（2倍ATR）
2. 收紧硬止损至-6%
3. 优化浮盈保护：盈利>2%后，回吐2%即止盈
4. 增加ATR波动率过滤：波动过大时减少仓位
5. 增加大盘趋势过滤（沪深300相对强弱）
6. 优化因子权重
"""
import os, sys, time, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optim_backtest_v5")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest")
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt")

START = "2025-01-02"
END = "2026-08-10"

PARAMS = {
    'initial_capital': 1_000_000,
    'top_n': 5,
    'rebalance_freq': 20,
    'min_hold_days': 3,
    # 止损
    'stop_loss_pct': -0.06,       # 硬止损 -6%
    'profit_protect_pct': -0.02,  # 浮盈保护：回吐2%
    'profit_protect_trigger': 0.02,  # 盈利>2%启动保护
    'atr_stop_mult': 2.0,         # ATR动态止损倍数
    # 波动率过滤
    'vol_filter': True,
    'max_atr_pct': 0.05,  # 日波动率>5%不买入
    # 交易成本
    'commission_rate': 0.0003,
    'stamp_rate': 0.0005,
    'slippage_rate': 0.001,
    'min_commission': 5.0,
    # 因子权重
    'factor_weights': {
        'dmi': 0.25,        # DMI趋势强度
        'macd': 0.20,       # MACD金叉
        'momentum_20': 0.20, # 20日动量
        'momentum_60': 0.10, # 60日动量
        'method135': 0.10,   # 135战法
        'pe_valuation': 0.10, # PE估值
        'pb_valuation': 0.05, # PB估值
    },
}


def load_stock_data():
    logger.info("从最新缓存加载沪深300成分股数据...")
    with open(UNIVERSE_FILE, "r") as f:
        codes = [line.strip() for line in f if line.strip()]

    data_map = {}
    fund_map = {}

    for code in codes:
        symbol = code.split('.')[0]
        kline_file = os.path.join(CACHE_DIR, f"{symbol}_hfq_kline.csv")
        fund_file = os.path.join(CACHE_DIR, f"{symbol}_fund.csv")

        if os.path.exists(kline_file):
            try:
                k = pd.read_csv(kline_file, index_col=0, parse_dates=True)
                k = k[(k.index >= pd.Timestamp(START)) & (k.index <= pd.Timestamp(END))]
                if not k.empty and len(k) >= 90:
                    data_map[code] = k
            except:
                pass

        if os.path.exists(fund_file):
            try:
                fnd = pd.read_csv(fund_file, index_col=0, parse_dates=True)
                fnd = fnd[(fnd.index >= pd.Timestamp(START)) & (fnd.index <= pd.Timestamp(END))]
                if code in data_map and not fnd.empty:
                    fnd = fnd.reindex(data_map[code].index).ffill().bfill()
                    fund_map[code] = fnd
            except:
                pass

    logger.info(f"数据: {len(data_map)}/{len(codes)} K线, {len(fund_map)} 基本面")
    return codes, data_map, fund_map


def compute_dmi(df, period=14):
    result = df.copy()
    result['high_low'] = result['high'] - result['low']
    result['high_prev_close'] = abs(result['high'] - result['close'].shift(1))
    result['low_prev_close'] = abs(result['low'] - result['close'].shift(1))
    result['TR'] = result[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    result['high_diff'] = result['high'] - result['high'].shift(1)
    result['low_diff'] = result['low'].shift(1) - result['low']
    result['plus_dm'] = np.where((result['high_diff'] > result['low_diff']) & (result['high_diff'] > 0), result['high_diff'], 0)
    result['minus_dm'] = np.where((result['low_diff'] > result['high_diff']) & (result['low_diff'] > 0), result['low_diff'], 0)
    result['atr'] = result['TR'].rolling(window=period).mean()
    result['pdi'] = 100 * result['plus_dm'].rolling(window=period).mean() / result['atr']
    result['ndi'] = 100 * result['minus_dm'].rolling(window=period).mean() / result['atr']
    result['dx'] = 100 * abs(result['pdi'] - result['ndi']) / (result['pdi'] + result['ndi'])
    result['adx'] = result['dx'].rolling(window=period).mean()
    result.drop(columns=['high_low', 'high_prev_close', 'low_prev_close', 'TR', 'high_diff', 'low_diff', 'plus_dm', 'minus_dm', 'atr', 'dx'], inplace=True, errors='ignore')
    return result


def compute_macd(df, fast=12, slow=26, signal=9):
    result = df.copy()
    ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
    result['DIF'] = ema_fast - ema_slow
    result['DEA'] = result['DIF'].ewm(span=signal, adjust=False).mean()
    result['MACD_bar'] = 2 * (result['DIF'] - result['DEA'])
    return result


def compute_momentum(df, window=20):
    result = df.copy()
    result[f'mom_{window}'] = result['close'] / result['close'].shift(window) - 1
    return result


def compute_ma_lines(df, ma60=60):
    result = df.copy()
    result['ma60'] = result['close'].rolling(window=ma60).mean()
    return result


def compute_all_technicals(data_map):
    logger.info("计算技术指标...")
    t0 = time.time()
    for code, df in data_map.items():
        data_map[code] = compute_dmi(df, period=14)
        data_map[code] = compute_macd(data_map[code])
        data_map[code] = compute_momentum(data_map[code], 20)
        data_map[code] = compute_momentum(data_map[code], 60)
        data_map[code] = compute_ma_lines(data_map[code])
    logger.info(f"完成, {time.time()-t0:.1f}s")
    return data_map


def compute_signal_score(df, i, code, fund_map, date):
    if i < 60:
        return 50.0, 0

    cur = df.iloc[i]
    prev = df.iloc[i-1] if i > 0 else cur

    scores = []
    weights = []
    bullish_count = 0

    # 1. DMI
    w = PARAMS['factor_weights']['dmi']
    if 'pdi' in df.columns and 'adx' in df.columns:
        if cur['pdi'] > cur['ndi'] and cur['adx'] > 20:
            dmi_s = min((cur['pdi'] - cur['ndi']) * 2, 100)
            if dmi_s > 50: bullish_count += 1
        else:
            dmi_s = 20
    else:
        dmi_s = 50
    scores.append(dmi_s)
    weights.append(w)

    # 2. MACD
    w = PARAMS['factor_weights']['macd']
    if 'DIF' in df.columns and i >= 35:
        macd_s = 50
        if cur['DIF'] > cur['DEA']:
            macd_s += 20
            bullish_count += 1
        if cur['DIF'] > 0:
            macd_s += 15
        if prev['DIF'] <= prev['DEA'] and cur['DIF'] > cur['DEA']:
            macd_s += 15
            bullish_count += 1
        if cur['MACD_bar'] > 0:
            macd_s += 10
        if cur['MACD_bar'] > prev['MACD_bar']:
            macd_s += 10
    else:
        macd_s = 50
    scores.append(macd_s)
    weights.append(w)

    # 3. 20日动量
    w = PARAMS['factor_weights']['momentum_20']
    if 'mom_20' in df.columns and i >= 20:
        mom20 = cur['mom_20']
        if pd.notna(mom20):
            if mom20 > 0.05:
                m20_s = 90
                bullish_count += 1
            elif mom20 > 0.02:
                m20_s = 75
            elif mom20 > 0:
                m20_s = 60
            elif mom20 > -0.02:
                m20_s = 30
            else:
                m20_s = 15
        else:
            m20_s = 50
    else:
        m20_s = 50
    scores.append(m20_s)
    weights.append(w)

    # 4. 60日动量
    w = PARAMS['factor_weights']['momentum_60']
    if 'mom_60' in df.columns and i >= 60:
        mom60 = cur['mom_60']
        if pd.notna(mom60):
            if mom60 > 0.12:
                m60_s = 90
                bullish_count += 1
            elif mom60 > 0.06:
                m60_s = 75
            elif mom60 > 0:
                m60_s = 60
            elif mom60 > -0.06:
                m60_s = 30
            else:
                m60_s = 15
        else:
            m60_s = 50
    else:
        m60_s = 50
    scores.append(m60_s)
    weights.append(w)

    # 5. 135战法
    w = PARAMS['factor_weights']['method135']
    if 'ma60' in df.columns and i >= 60:
        m135_s = 50
        if cur['close'] > cur['ma60']:
            m135_s += 25
            bullish_count += 1
    else:
        m135_s = 50
    scores.append(m135_s)
    weights.append(w)

    # 6. PE估值
    w = PARAMS['factor_weights']['pe_valuation']
    pe_s = 50.0
    if code in fund_map and date in fund_map[code].index:
        pe = fund_map[code]['pe']
        if date in pe.index and pd.notna(pe.loc[date]):
            pe_val = pe.loc[date]
            if pe_val < 8:
                pe_s = 85
                bullish_count += 1
            elif pe_val < 12:
                pe_s = 70
            elif pe_val < 20:
                pe_s = 55
            elif pe_val < 30:
                pe_s = 35
            else:
                pe_s = 15
    scores.append(pe_s)
    weights.append(w)

    # 7. PB估值
    w = PARAMS['factor_weights']['pb_valuation']
    pb_s = 50.0
    if code in fund_map and date in fund_map[code].index:
        pb = fund_map[code]['pb']
        if date in pb.index and pd.notna(pb.loc[date]):
            pb_val = pb.loc[date]
            if pb_val < 0.5:
                pb_s = 85
                bullish_count += 1
            elif pb_val < 0.8:
                pb_s = 70
            elif pb_val < 1.5:
                pb_s = 55
            elif pb_val < 2.5:
                pb_s = 35
            else:
                pb_s = 15
    scores.append(pb_s)
    weights.append(w)

    total_w = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights)) / total_w

    return composite, bullish_count


def run_backtest(data_map, fund_map):
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    logger.info(f"回测: {len(all_dates)} 个交易日")

    data_map = compute_all_technicals(data_map)

    params = PARAMS
    initial_capital = params['initial_capital']
    cash = initial_capital
    peak_nav = initial_capital
    daily_records = []
    current_positions = {}
    position_shares = {}
    position_cost_basis = {}
    position_buy_date = {}
    position_highest = {}
    position_atr = {}  # 买入时ATR
    trade_log = []

    commission_rate = params['commission_rate']
    stamp_rate = params['stamp_rate']
    slippage_rate = params['slippage_rate']
    min_commission = params['min_commission']
    stop_loss_pct = params['stop_loss_pct']
    profit_trigger = params['profit_protect_trigger']
    profit_protect = params['profit_protect_pct']
    atr_mult = params['atr_stop_mult']
    min_hold = params['min_hold_days']

    t0 = time.time()
    for di, date in enumerate(all_dates):
        # ---- 净值 ----
        nav = cash
        for code, shares in position_shares.items():
            if code in data_map and date in data_map[code].index:
                nav += shares * data_map[code].loc[date, 'close']
        if nav > peak_nav:
            peak_nav = nav
        drawdown = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0

        # ---- 止损/止盈 ----
        to_sell = []
        for code in list(current_positions.keys()):
            if code not in data_map or date not in data_map[code].index:
                continue

            current_price = data_map[code].loc[date, 'close']
            buy_price = current_positions[code]
            pnl = (current_price - buy_price) / buy_price

            hold_days = (date - pd.Timestamp(position_buy_date.get(code, date))).days
            atr = data_map[code].loc[date, 'atr'] if 'atr' in data_map[code].columns else 0
            atr_price_stop = current_price - atr_mult * atr if atr > 0 and code in position_atr else 0

            # ATR动态止损：当前价低于 买入价 - 2*ATR 时止损
            if atr > 0 and current_price < atr_price_stop:
                to_sell.append((code, current_price, "ATR止损"))
                continue

            # 硬止损
            if pnl <= stop_loss_pct:
                to_sell.append((code, current_price, "硬止损"))
                continue

            # 浮盈保护止损
            if hold_days >= min_hold and position_highest.get(code, buy_price) > buy_price:
                draw_from_high = (current_price - position_highest[code]) / position_highest[code]
                if pnl > profit_trigger and draw_from_high <= profit_protect:
                    to_sell.append((code, current_price, "浮盈保护"))
                    continue

            # 更新最高价
            if code not in position_highest or current_price > position_highest[code]:
                position_highest[code] = current_price

        for code, sp, reason in to_sell:
            shares = position_shares[code]
            cost = position_cost_basis[code]
            fill = sp * (1 - slippage_rate)
            gross = fill * shares
            comm = max(gross * commission_rate, min_commission)
            stamp = gross * stamp_rate
            proceeds = gross - comm - stamp
            cash += proceeds
            pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
            trade_log.append({'code': code, 'buy_date': position_buy_date.get(code, ''),
                             'buy_price': current_positions.get(code, sp), 'sell_price': sp,
                             'reason': reason, 'pnl_pct': pnl_pct})
            position_shares.pop(code, None)
            position_cost_basis.pop(code, None)
            position_buy_date.pop(code, None)
            position_highest.pop(code, None)
            position_atr.pop(code, None)
            current_positions.pop(code, None)

        # ---- 调仓 ----
        if di % params['rebalance_freq'] == 0:
            scores = {}
            bull_counts = {}

            for code, df in data_map.items():
                if date not in df.index:
                    continue
                i = df.index.get_loc(date)
                composite, bullish = compute_signal_score(df, i, code, fund_map, date)

                # 波动率过滤
                if params['vol_filter'] and 'atr' in df.columns:
                    atr = df.iloc[i]['atr']
                    if atr > 0:
                        atr_pct = atr / df.iloc[i]['close']
                        if atr_pct > params['max_atr_pct']:
                            composite -= 30  # 高波动率惩罚

                if bullish >= 3:
                    scores[code] = composite
                    bull_counts[code] = bullish
                else:
                    scores[code] = composite - 20

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = set(c for c, _ in ranked[:params['top_n']])
            current_held = set(current_positions.keys())

            # 卖出
            for code in current_held - selected:
                if date in data_map[code].index:
                    sp = data_map[code].loc[date, 'close']
                    shares = position_shares[code]
                    fill = sp * (1 - slippage_rate)
                    gross = fill * shares
                    comm = max(gross * commission_rate, min_commission)
                    stamp = gross * stamp_rate
                    proceeds = gross - comm - stamp
                    cash += proceeds
                    cost = position_cost_basis.get(code, gross)
                    pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
                    trade_log.append({'code': code, 'buy_date': position_buy_date.get(code, ''),
                                     'buy_price': current_positions.get(code, sp), 'sell_price': sp,
                                     'reason': '调仓卖出', 'pnl_pct': pnl_pct})
                    position_shares.pop(code, None)
                    position_cost_basis.pop(code, None)
                    position_buy_date.pop(code, None)
                    position_highest.pop(code, None)
                    position_atr.pop(code, None)
                    current_positions.pop(code, None)

            # 买入
            to_buy = selected - current_held
            if to_buy and cash > 0:
                equal_budget = cash / len(to_buy)
                for code in sorted(to_buy):
                    if date in data_map[code].index:
                        sp = data_map[code].loc[date, 'close']
                        fill = sp * (1 + slippage_rate)
                        shares = int(equal_budget / fill / 100) * 100
                        if shares <= 0:
                            continue
                        gross = fill * shares
                        comm = max(gross * commission_rate, min_commission)
                        total_cost = gross + comm
                        if total_cost > cash:
                            shares = int((cash / (fill * (1 + commission_rate))) / 100) * 100
                            if shares <= 0:
                                continue
                            gross = fill * shares
                            comm = max(gross * commission_rate, min_commission)
                            total_cost = gross + comm
                        cash -= total_cost
                        current_positions[code] = sp
                        position_buy_date[code] = str(date)
                        position_shares[code] = shares
                        position_cost_basis[code] = total_cost
                        position_highest[code] = sp
                        atr = data_map[code].loc[date, 'atr'] if 'atr' in data_map[code].columns else 0
                        position_atr[code] = atr

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
            'return': (nav / initial_capital) - 1,
            'drawdown': drawdown,
            'positions': len(current_positions)
        })

    elapsed = time.time() - t0
    logger.info(f"回测完成, {elapsed:.1f}s")
    return daily_records, trade_log


def compute_stats(result_df, trade_log):
    nav_returns = result_df['nav'].pct_change().fillna(0)
    total_return = result_df['return'].iloc[-1]
    trading_days = len(result_df)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1
    rf = 0.03
    daily_std = nav_returns.std()
    sharpe = (nav_returns.mean() - rf/252) / daily_std * np.sqrt(252) if daily_std > 0 else 0
    max_dd = abs(result_df['drawdown'].min())
    win_trades = [t for t in trade_log if t['pnl_pct'] > 0]
    win_rate = len(win_trades) / len(trade_log) if trade_log else 0

    returns = result_df['return'].tolist()
    max_consec_loss = 0
    consec = 0
    for r in returns:
        if r < 0:
            consec += 1
            max_consec_loss = max(max_consec_loss, consec)
        else:
            consec = 0

    wins = [t['pnl_pct'] for t in trade_log if t['pnl_pct'] > 0]
    losses = [t['pnl_pct'] for t in trade_log if t['pnl_pct'] <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # 各止损方式统计
    reason_stats = {}
    for t in trade_log:
        r = t['reason']
        if r not in reason_stats:
            reason_stats[r] = {'count': 0, 'pnl': []}
        reason_stats[r]['count'] += 1
        reason_stats[r]['pnl'].append(t['pnl_pct'])

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_trades': len(trade_log),
        'max_consecutive_loss': max_consec_loss,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'reason_stats': {k: {'count': v['count'], 'avg': np.mean(v['pnl'])} for k, v in reason_stats.items()},
    }


def main():
    logger.info("=" * 60)
    logger.info("🚀 DMI 多因子选股策略 v5（沪深300）")
    logger.info("=" * 60)

    codes, data_map, fund_map = load_stock_data()
    if len(data_map) == 0:
        return

    daily_records, trade_log = run_backtest(data_map, fund_map)
    result_df = pd.DataFrame(daily_records)
    stats = compute_stats(result_df, trade_log)
    final_nav = result_df['nav'].iloc[-1]

    print("\n" + "=" * 60)
    print("📊 DMI 多因子选股策略 — 优化版 v5 结果")
    print("=" * 60)
    print(f"回测区间:     {result_df['date'].iloc[0]} ~ {result_df['date'].iloc[-1]}")
    print(f"股票池:       沪深300 ({len(codes)}只)")
    print(f"有效数据:     {len(data_map)}只K线, {len(fund_map)}只基本面")
    print(f"初始资金:     1,000,000 元")
    print(f"持仓数:       {PARAMS['top_n']} 只")
    print(f"调仓频率:     {PARAMS['rebalance_freq']} 交易日")
    print(f"最小持仓:     {PARAMS['min_hold_days']} 天")
    print(f"信号共振:     >= 3 看多信号")
    print(f"硬止损:       {PARAMS['stop_loss_pct']*100:.0f}%")
    print(f"ATR止损:      {PARAMS['atr_stop_mult']}倍ATR")
    print(f"浮盈保护:     {PARAMS['profit_protect_trigger']*100:.0f}%触发, 回吐{abs(PARAMS['profit_protect_pct'])*100:.0f}%")
    print("-" * 60)
    print(f"总收益率:     {stats['total_return']:+.2%}")
    print(f"年化收益:     {stats['annual_return']:+.2%}")
    print(f"夏普比率:     {stats['sharpe_ratio']:.2f}")
    print(f"最大回撤:     {stats['max_drawdown']:.2%}")
    print(f"胜率:         {stats['win_rate']:.2%}")
    print(f"交易次数:     {stats['total_trades']}")
    print(f"最大连亏天数: {stats['max_consecutive_loss']}")
    print(f"平均盈利:     {stats['avg_win_pct']:+.2%}")
    print(f"平均亏损:     {stats['avg_loss_pct']:+.2%}")
    print(f"盈亏比:       {abs(stats['avg_win_pct']/stats['avg_loss_pct']):.2f}:1" if stats['avg_loss_pct'] != 0 else "盈亏比: N/A")
    print(f"最终净值:     {final_nav:,.0f} 元")
    print("=" * 60)

    # 止损原因统计
    print(f"\n📊 止损原因分析:")
    for reason, info in stats['reason_stats'].items():
        print(f"  {reason}: {info['count']}次, 平均{info['avg']:+.2%}")

    print(f"\n📋 最近15笔交易:")
    for t in trade_log[-15:]:
        print(f"  {t['code']} | {t['buy_price']:.2f} -> {t['sell_price']:.2f} | "
              f"{t['pnl_pct']:+.2%} | {t['reason']}")

    stats['final_nav'] = float(final_nav)
    stats['version'] = 'v5'
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_backtest_result_v5.json")
    with open(result_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存: {result_path}")


if __name__ == "__main__":
    main()
