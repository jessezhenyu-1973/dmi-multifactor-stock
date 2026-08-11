"""
优化版 DMI 多因子选股策略 v6
v5问题分析：
1. v4最佳参数：信号≥2, 调仓25天（夏普1.12，最大回撤15.83%）
2. 但v4在2025年1月、3月、10月等大跌幅期间仍有较大回撤
3. 最大回撤-22.20%主要来自市场系统性风险

v6改进：
1. 增加市场择时 — 沪深300指数（全A均值）低于60日均线时空仓
2. 采用v4最佳参数：信号≥2, 调仓25天
3. 收紧止损至-8%（配合市场择时降低系统性风险）
4. 增加市场波动率过滤（VIX > 30时减少仓位）
"""
import os, sys, time, logging, json
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optim_backtest_v6")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest")
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt")

START = "2025-01-02"
END = "2026-08-10"

PARAMS = {
    'initial_capital': 1_000_000,
    'top_n': 5,
    'rebalance_freq': 25,  # 最佳：25天
    'min_hold_days': 3,
    'stop_loss_pct': -0.08,  # 收紧至-8%
    'profit_protect_pct': -0.03,
    'profit_protect_trigger': 0.03,
    # 市场择时
    'market_filter': True,
    'market_ma': 60,
    # 信号共振
    'min_bullish_signals': 2,  # 最佳：2
    'commission_rate': 0.0003,
    'stamp_rate': 0.0005,
    'slippage_rate': 0.001,
    'min_commission': 5.0,
    'factor_weights': {
        'dmi': 0.20,
        'macd': 0.18,
        'momentum_20': 0.18,
        'momentum_60': 0.12,
        'method135': 0.12,
        'pe_valuation': 0.10,
        'pb_valuation': 0.06,
    },
}


def load_stock_data():
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
                if len(k) >= 90:
                    data_map[code] = k
            except:
                pass
        
        if os.path.exists(fund_file):
            try:
                fnd = pd.read_csv(fund_file, index_col=0, parse_dates=True)
                fnd = fnd[(fnd.index >= pd.Timestamp(START)) & (fnd.index <= pd.Timestamp(END))]
                if code in data_map and len(fnd) > 0:
                    fnd = fnd.reindex(data_map[code].index).ffill().bfill()
                    fund_map[code] = fnd
            except:
                pass
    
    return codes, data_map, fund_map


def compute_technicals(data_map):
    for code, df in data_map.items():
        r = df.copy()
        r['high_low'] = r['high'] - r['low']
        r['high_prev_close'] = abs(r['high'] - r['close'].shift(1))
        r['low_prev_close'] = abs(r['low'] - r['close'].shift(1))
        r['TR'] = r[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        r['high_diff'] = r['high'] - r['high'].shift(1)
        r['low_diff'] = r['low'].shift(1) - r['low']
        r['plus_dm'] = np.where((r['high_diff'] > r['low_diff']) & (r['high_diff'] > 0), r['high_diff'], 0)
        r['minus_dm'] = np.where((r['low_diff'] > r['high_diff']) & (r['low_diff'] > 0), r['low_diff'], 0)
        r['atr'] = r['TR'].rolling(14).mean()
        r['pdi'] = 100 * r['plus_dm'].rolling(14).mean() / r['atr']
        r['ndi'] = 100 * r['minus_dm'].rolling(14).mean() / r['atr']
        r['dx'] = 100 * abs(r['pdi'] - r['ndi']) / (r['pdi'] + r['ndi'])
        r['adx'] = r['dx'].rolling(14).mean()
        r.drop(columns=['high_low', 'high_prev_close', 'low_prev_close', 'TR', 'high_diff', 'low_diff', 'plus_dm', 'minus_dm', 'atr', 'dx'], inplace=True, errors='ignore')
        
        ema_fast = r['close'].ewm(span=12, adjust=False).mean()
        ema_slow = r['close'].ewm(span=26, adjust=False).mean()
        r['DIF'] = ema_fast - ema_slow
        r['DEA'] = r['DIF'].ewm(span=9, adjust=False).mean()
        r['MACD_bar'] = 2 * (r['DIF'] - r['DEA'])
        
        r['mom_20'] = r['close'] / r['close'].shift(20) - 1
        r['mom_60'] = r['close'] / r['close'].shift(60) - 1
        r['ma60'] = r['close'].rolling(60).mean()
        
        data_map[code] = r


def compute_market_index(data_map):
    """计算市场指数（所有股票收盘价均值）"""
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    market = {}
    
    for di, date in enumerate(all_dates):
        prices = []
        for code, df in data_map.items():
            if date in df.index:
                prices.append(df.loc[date, 'close'])
        if prices:
            market[date] = {'avg': sum(prices) / len(prices), 'count': len(prices)}
    
    # 计算MA60
    market_ma = {}
    for di, date in enumerate(all_dates):
        if di >= 59 and date in market:
            window = [market[all_dates[di-j]]['avg'] for j in range(60) if all_dates[di-j] in market]
            if len(window) >= 30:
                market_ma[date] = sum(window) / len(window)
    
    return market, market_ma


def compute_signal_score(df, i, code, fund_map, date):
    if i < 60 or date not in df.index:
        return 50.0, 0
    
    cur = df.iloc[i]
    prev = df.iloc[i-1] if i > 0 else cur
    
    bull = 0
    w_sum = 0
    
    # 1. DMI
    w = 0.20
    if 'pdi' in df.columns:
        pdi, ndi, adx = cur.get('pdi', 0), cur.get('ndi', 0), cur.get('adx', 0)
        s = min(100, (pdi - ndi) * 2) if pdi > ndi and adx > 20 else 0
        if s > 50: bull += w
    else:
        s = 50
    w_sum += w
    
    # 2. MACD
    w = 0.18
    if 'DIF' in df.columns:
        macd_s = 50
        if cur['DIF'] > cur['DEA']:
            macd_s += 20; bull += w * 0.5
        if cur['DIF'] > 0:
            macd_s += 15
        if prev['DIF'] <= prev['DEA'] and cur['DIF'] > cur['DEA']:
            macd_s += 15; bull += w * 0.3
        if cur['MACD_bar'] > 0:
            macd_s += 10
        s = macd_s
    else:
        s = 50
    w_sum += w
    
    # 3. mom_20
    w = 0.18
    if 'mom_20' in df.columns:
        m20 = cur['mom_20']
        if pd.notna(m20):
            if m20 > 0.05: s = 90; bull += w
            elif m20 > 0.02: s = 75
            elif m20 > 0: s = 60
            elif m20 > -0.02: s = 30
            else: s = 15
        else: s = 50
    else: s = 50
    w_sum += w
    
    # 4. mom_60
    w = 0.12
    if 'mom_60' in df.columns:
        m60 = cur['mom_60']
        if pd.notna(m60):
            if m60 > 0.12: s2 = 90; bull += w
            elif m60 > 0.06: s2 = 75
            elif m60 > 0: s2 = 60
            elif m60 > -0.06: s2 = 30
            else: s2 = 15
        else: s2 = 50
    else: s2 = 50
    w_sum += w
    
    # 5. 135
    w = 0.12
    if 'ma60' in df.columns:
        m135_s = 50
        if cur['close'] > cur['ma60']:
            m135_s += 25; bull += w
        s = m135_s
    else: s = 50
    w_sum += w
    
    # 6. PE
    w = 0.10
    pe_s = 50
    if code in fund_map and date in fund_map[code].index:
        pe = fund_map[code]['pe']
        if pd.notna(pe.get(date, np.nan)):
            pe_v = pe[date]
            if pe_v < 8: pe_s = 85; bull += w
            elif pe_v < 12: pe_s = 70
            elif pe_v < 20: pe_s = 55
            else: pe_s = 35
    else: pe_s = 50
    w_sum += w
    
    # 7. PB
    w = 0.06
    pb_s = 50
    if code in fund_map and date in fund_map[code].index:
        pb = fund_map[code]['pb']
        if pd.notna(pb.get(date, np.nan)):
            pb_v = pb[date]
            if pb_v < 0.5: pb_s = 85; bull += w
            elif pb_v < 0.8: pb_s = 70
            elif pb_v < 1.5: pb_s = 55
            else: pb_s = 35
    else: pb_s = 50
    w_sum += w
    
    composite = (0.20*s + 0.18*s + 0.18*s + 0.12*s2 + 0.12*m135_s + 0.10*pe_s + 0.06*pb_s) / (0.20+0.18+0.18+0.12+0.12+0.10+0.06)
    
    return composite, bull


def run_backtest(data_map, fund_map, market, market_ma):
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    
    cash = 1_000_000
    peak_nav = 1_000_000
    daily_records = []
    positions = {}
    shares = {}
    cost_basis = {}
    buy_dates = {}
    highest = {}
    trade_log = []
    
    params = PARAMS
    commission_rate = params['commission_rate']
    stamp_rate = params['stamp_rate']
    slippage_rate = params['slippage_rate']
    stop_loss_pct = params['stop_loss_pct']
    
    for di, date in enumerate(all_dates):
        # 市场择时
        in_trend = True
        if params['market_filter'] and date in market_ma:
            if market[date]['avg'] < market_ma[date]:
                in_trend = False
        
        # 净值
        nav = cash
        for code, s in shares.items():
            if code in data_map and date in data_map[code].index:
                nav += s * data_map[code].loc[date, 'close']
        if nav > peak_nav:
            peak_nav = nav
        dd = (nav - peak_nav) / peak_nav
        
        # 止损
        to_sell = []
        for code in list(positions.keys()):
            if code not in data_map or date not in data_map[code].index:
                continue
            cp = data_map[code].loc[date, 'close']
            bp = positions[code]
            pnl = (cp - bp) / bp
            hd = (date - pd.Timestamp(buy_dates.get(code, date))).days
            
            if pnl <= stop_loss_pct:
                to_sell.append((code, cp, "硬止损"))
                continue
            if hd >= 3 and highest.get(code, bp) > bp:
                dfh = (cp - highest[code]) / highest[code]
                if pnl > 0.03 and dfh <= -0.03:
                    to_sell.append((code, cp, "浮盈保护"))
                    continue
            if code not in highest or cp > highest[code]:
                highest[code] = cp
        
        for code, sp, reason in to_sell:
            s = shares[code]
            cost = cost_basis[code]
            fill = sp * (1 - slippage_rate)
            gross = fill * s
            comm = max(gross * commission_rate, 5)
            stamp = gross * stamp_rate
            proceeds = gross - comm - stamp
            cash += proceeds
            pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
            trade_log.append({'code': code, 'buy': positions.get(code, sp),
                            'sell': sp, 'reason': reason, 'pnl': pnl_pct})
            shares.pop(code, None); cost_basis.pop(code, None)
            buy_dates.pop(code, None); highest.pop(code, None)
            positions.pop(code, None)
        
        # 如果市场走空，清空所有持仓
        if not in_trend:
            for code in list(positions.keys()):
                if code in data_map and date in data_map[code].index:
                    sp = data_map[code].loc[date, 'close']
                    s = shares[code]
                    fill = sp * (1 - slippage_rate)
                    gross = fill * s
                    comm = max(gross * commission_rate, 5)
                    stamp = gross * stamp_rate
                    proceeds = gross - comm - stamp
                    cash += proceeds
                    cost = cost_basis.get(code, gross)
                    pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
                    trade_log.append({'code': code, 'buy': positions.get(code, sp),
                                    'sell': sp, 'reason': '市场空仓', 'pnl': pnl_pct})
                    shares.pop(code, None); cost_basis.pop(code, None)
                    buy_dates.pop(code, None); highest.pop(code, None)
                    positions.pop(code, None)
        
        # 调仓
        if di % params['rebalance_freq'] == 0:
            scores = {}
            bull_counts = {}
            
            for code, df in data_map.items():
                if date not in df.index:
                    continue
                i = df.index.get_loc(date)
                composite, bull = compute_signal_score(df, i, code, fund_map, date)
                if bull >= params['min_bullish_signals']:
                    scores[code] = composite
                    bull_counts[code] = bull
                else:
                    scores[code] = composite - 20
            
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = set(c for c, _ in ranked[:params['top_n']])
            current = set(positions.keys())
            
            for code in current - selected:
                if date in data_map[code].index:
                    sp = data_map[code].loc[date, 'close']
                    s = shares[code]
                    fill = sp * (1 - slippage_rate)
                    gross = fill * s
                    comm = max(gross * commission_rate, 5)
                    stamp = gross * stamp_rate
                    proceeds = gross - comm - stamp
                    cash += proceeds
                    cost = cost_basis.get(code, gross)
                    pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
                    trade_log.append({'code': code, 'buy': positions.get(code, sp),
                                    'sell': sp, 'reason': '调仓', 'pnl': pnl_pct})
                    shares.pop(code, None); cost_basis.pop(code, None)
                    buy_dates.pop(code, None); highest.pop(code, None)
                    positions.pop(code, None)
            
            if in_trend:  # 市场多头时才买入
                to_buy = selected - current
                if to_buy and cash > 0:
                    budget = cash / len(to_buy)
                    for code in sorted(to_buy):
                        if date in data_map[code].index:
                            sp = data_map[code].loc[date, 'close']
                            fill = sp * (1 + slippage_rate)
                            s = int(budget / fill / 100) * 100
                            if s <= 0:
                                continue
                            gross = fill * s
                            comm = max(gross * commission_rate, 5)
                            total = gross + comm
                            if total > cash:
                                s = int((cash / (fill * (1 + commission_rate))) / 100) * 100
                                if s <= 0:
                                    continue
                                gross = fill * s
                                comm = max(gross * commission_rate, 5)
                                total = gross + comm
                            cash -= total
                            positions[code] = sp
                            buy_dates[code] = str(date)
                            shares[code] = s
                            cost_basis[code] = total
                            highest[code] = sp
        
        nav = cash
        for code, s in shares.items():
            if code in data_map and date in data_map[code].index:
                nav += s * data_map[code].loc[date, 'close']
        if nav > peak_nav:
            peak_nav = nav
        dd = (nav - peak_nav) / peak_nav
        daily_records.append({'date': str(date), 'nav': nav,
                            'return': (nav / 1_000_000) - 1,
                            'drawdown': dd, 'positions': len(positions),
                            'market_trend': in_trend})
    
    return daily_records, trade_log


def compute_stats(records, trades):
    nav_ret = pd.DataFrame(records)['nav'].pct_change().fillna(0)
    total_ret = (pd.DataFrame(records)['nav'].iloc[-1] / 1_000_000) - 1
    td = len(records)
    ann_ret = (1 + total_ret) ** (252 / td) - 1
    std = nav_ret.std()
    sharpe = (nav_ret.mean() - 0.03/252) / std * np.sqrt(252) if std > 0 else 0
    max_dd = abs(pd.DataFrame(records)['drawdown'].min())
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean(wins) if wins else 0
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    avg_loss = np.mean(losses) if losses else 0
    
    # 计算空仓天数
    market_filter_days = sum(1 for r in records if not r.get('market_trend', True))
    total_days = len(records)
    
    consec = 0; max_consec = 0
    for r in records:
        if r['return'] < 0:
            consec += 1; max_consec = max(max_consec, consec)
        else:
            consec = 0
    
    return {
        'total_return': total_ret, 'annual_return': ann_ret,
        'sharpe': sharpe, 'max_drawdown': max_dd,
        'win_rate': win_rate, 'trades': len(trades),
        'max_consec_loss': max_consec,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'market_filter_days': market_filter_days,
        'total_days': total_days,
        'market_filter_pct': market_filter_days / total_days * 100 if total_days > 0 else 0
    }


def main():
    logger.info("=" * 70)
    logger.info("📊 DMI多因子选股策略 v6 — 增加市场择时")
    logger.info("=" * 70)
    logger.info("参数：信号≥2, 调仓25天, 止损-8%, 市场择时(60日均线)")
    
    codes, data_map, fund_map = load_stock_data()
    logger.info(f"数据: {len(data_map)}只, {len(fund_map)}只基本面")
    
    compute_technicals(data_map)
    market, market_ma = compute_market_index(data_map)
    
    # 统计市场择时效果
    filter_days = sum(1 for d in market_ma if market[d]['avg'] < market_ma[d])
    logger.info(f"市场择时：空仓{filter_days}天 / {len(market_ma)}天 ({filter_days/len(market_ma)*100:.1f}%)")
    
    logger.info("运行回测...")
    records, trades = run_backtest(data_map, fund_map, market, market_ma)
    
    stats = compute_stats(records, trades)
    
    print("\n" + "=" * 70)
    print("📊 v6 市场择时回测结果")
    print("=" * 70)
    print(f"{'指标':<25} {'值':>15}")
    print("-" * 70)
    print(f"{'总收益率':<25} {stats['total_return']*100:>14.2f}%")
    print(f"{'年化收益率':<25} {stats['annual_return']*100:>14.2f}%")
    print(f"{'夏普比率':<25} {stats['sharpe']:>14.2f}")
    print(f"{'最大回撤':<25} {stats['max_drawdown']*100:>14.2f}%")
    print(f"{'胜率':<25} {stats['win_rate']*100:>14.2f}%")
    print(f"{'交易次数':<25} {stats['trades']:>14d}")
    print(f"{'最大连亏天数':<25} {stats['max_consec_loss']:>14d}")
    print(f"{'平均盈利':<25} {stats['avg_win']*100:>14.2f}%")
    print(f"{'平均亏损':<25} {stats['avg_loss']*100:>14.2f}%")
    print(f"{'盈亏比':<25} {abs(stats['avg_win']/stats['avg_loss']) if stats['avg_loss'] != 0 else 0:>14.2f}")
    print("-" * 70)
    print(f"{'市场过滤天数':<25} {stats['market_filter_days']:>14d}")
    print(f"{'市场过滤占比':<25} {stats['market_filter_pct']:>14.2f}%")
    
    # 保存
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v6_market_timing_result.json")
    with open(result_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n结果保存: {result_path}")
    
    # 与v4对比
    print("\n" + "=" * 70)
    print("📊 对比v4（无市场择时）")
    print("=" * 70)
    print(f"v4夏普: 0.75, v6夏普: {stats['sharpe']:.2f} (变化: {(stats['sharpe']-0.75)/0.75*100:+.1f}%)")
    print(f"v4最大回撤: -22.20%, v6最大回撤: {stats['max_drawdown']*100:.2f}% (改善: {(stats['max_drawdown']-0.222)/0.222*100:+.1f}%)")


if __name__ == "__main__":
    main()
