"""
参数敏感性测试 v4
测试信号阈值和调仓频率对策略表现的影响
"""
import os, sys, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sensitivity_test")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest")
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt")

START = "2025-01-02"
END = "2026-08-10"

# 参数组合
PARAM_GRID = {
    'signal_threshold': [2, 3, 4],      # 信号共振阈值
    'rebalance_freq': [10, 15, 20, 25],  # 调仓频率
}


def load_data():
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
        ema_fast = r['close'].ewm(span=12, adjust=False).mean()
        ema_slow = r['close'].ewm(span=26, adjust=False).mean()
        r['DIF'] = ema_fast - ema_slow
        r['DEA'] = r['DIF'].ewm(span=9, adjust=False).mean()
        r['MACD_bar'] = 2 * (r['DIF'] - r['DEA'])
        r['mom_20'] = r['close'] / r['close'].shift(20) - 1
        r['mom_60'] = r['close'] / r['close'].shift(60) - 1
        r['ma60'] = r['close'].rolling(60).mean()
        data_map[code] = r


def signal_score(df, i, code, fund_map, date, min_bullish):
    if i < 60 or date not in df.index:
        return 50.0, 0
    
    cur = df.iloc[i]
    prev = df.iloc[i-1]
    
    bull = 0
    total = 0
    
    # DMI
    w = 0.25
    if 'pdi' in df.columns:
        pdi, ndi, adx = cur.get('pdi', 0), cur.get('ndi', 0), cur.get('adx', 0)
        dmi_s = min(100, (pdi - ndi) * 2) if pdi > ndi and adx > 20 else 0
        if dmi_s > 50:
            bull += 1
        s = dmi_s
    else:
        s = 50
    total += w
    bull *= (w/0.25)  # 按权重累加
    
    # MACD
    w = 0.20
    if 'DIF' in df.columns:
        macd_s = 50
        if cur['DIF'] > cur['DEA']:
            macd_s += 20
            bull += w
        if cur['DIF'] > 0:
            macd_s += 15
        if prev['DIF'] <= prev['DEA'] and cur['DIF'] > cur['DEA']:
            macd_s += 15
            bull += w
        if cur['MACD_bar'] > 0:
            macd_s += 10
        if cur['MACD_bar'] > prev['MACD_bar']:
            macd_s += 10
        s = macd_s
    else:
        s = 50
    total += w
    bull = bull + w * (macd_s if 'DIF' in df.columns else 50)
    bull -= w * (50)  # normalize
    
    # mom_20
    w = 0.20
    if 'mom_20' in df.columns:
        m20 = cur['mom_20']
        if pd.notna(m20):
            if m20 > 0.05:
                s = 90; bull += w
            elif m20 > 0.02:
                s = 75
            elif m20 > 0:
                s = 60
            elif m20 > -0.02:
                s = 30
            else:
                s = 15
        else:
            s = 50
    else:
        s = 50
    total += w
    
    # mom_60
    w = 0.12
    if 'mom_60' in df.columns:
        m60 = cur['mom_60']
        if pd.notna(m60):
            if m60 > 0.12:
                s2 = 90; bull += w
            elif m60 > 0.06:
                s2 = 75
            elif m60 > 0:
                s2 = 60
            elif m60 > -0.06:
                s2 = 30
            else:
                s2 = 15
        else:
            s2 = 50
    else:
        s2 = 50
    total += w
    
    # 135
    w = 0.10
    if 'ma60' in df.columns:
        m135_s = 50
        if cur['close'] > cur['ma60']:
            m135_s += 25
            bull += w
        s = m135_s
    else:
        s = 50
    total += w
    
    # PE/PB (简化)
    w_pe = 0.08
    w_pb = 0.05
    pe_s, pb_s = 50, 50
    if code in fund_map and date in fund_map[code].index:
        pe = fund_map[code]['pe']
        pb = fund_map[code]['pb']
        if pd.notna(pe.get(date, np.nan)):
            pe_v = pe[date]
            if pe_v < 8: pe_s = 85; bull += w_pe
            elif pe_v < 12: pe_s = 70
            elif pe_v < 20: pe_s = 55
            else: pe_s = 35
        if pd.notna(pb.get(date, np.nan)):
            pb_v = pb[date]
            if pb_v < 0.5: pb_s = 85; bull += w_pb
            elif pb_v < 0.8: pb_s = 70
            elif pb_v < 1.5: pb_s = 55
            else: pb_s = 35
    total += w_pe + w_pb
    
    # 简化版：用权重求和
    weights = [0.25, 0.20, 0.20, 0.12, 0.10, 0.08, 0.05]
    scores = [
        min(100, max(0, (cur.get('pdi', 0) - cur.get('ndi', 0)) * 2)) if 'pdi' in df.columns else 50,
        s if 'DIF' in df.columns else 50,
        s,
        s2,
        m135_s,
        pe_s,
        pb_s
    ]
    composite = sum(w * sc for w, sc in zip(weights, scores)) / sum(weights)
    
    # 计算看多信号数（基于因子得分 > 60）
    bull_count = 0
    factor_thresholds = [60, 60, 60, 60, 60, 65, 65]
    for sc, thresh in zip(scores, factor_thresholds):
        if sc >= thresh:
            bull_count += 1
    
    return composite, bull_count


def run_backtest(data_map, fund_map, min_bullish, rebal_freq):
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
    
    commission_rate = 0.0003
    stamp_rate = 0.0005
    slippage_rate = 0.001
    stop_loss_pct = -0.10
    profit_trigger = 0.03
    profit_protect = -0.03
    
    for di, date in enumerate(all_dates):
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
            if hd >= 5 and highest.get(code, bp) > bp:
                dfh = (cp - highest[code]) / highest[code]
                if pnl > profit_trigger and dfh <= profit_protect:
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
        
        # 调仓
        if di % rebal_freq == 0:
            scores = {}
            bull_counts = {}
            for code, df in data_map.items():
                if date not in df.index:
                    continue
                i = df.index.get_loc(date)
                composite, bull = signal_score(df, i, code, fund_map, date, min_bullish)
                if bull >= min_bullish:
                    scores[code] = composite
                    bull_counts[code] = bull
                else:
                    scores[code] = composite - 20
            
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = set(c for c, _ in ranked[:5])
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
        
        # 记录
        nav = cash
        for code, s in shares.items():
            if code in data_map and date in data_map[code].index:
                nav += s * data_map[code].loc[date, 'close']
        if nav > peak_nav:
            peak_nav = nav
        dd = (nav - peak_nav) / peak_nav
        daily_records.append({'date': str(date), 'nav': nav,
                            'return': (nav / 1_000_000) - 1,
                            'drawdown': dd, 'positions': len(positions)})
    
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
        'avg_win': avg_win, 'avg_loss': avg_loss
    }


def main():
    logger.info("=" * 60)
    logger.info("📊 参数敏感性测试")
    logger.info("=" * 60)
    
    codes, data_map, fund_map = load_data()
    logger.info(f"数据: {len(data_map)}只, {len(fund_map)}只基本面")
    
    compute_technicals(data_map)
    
    # 测试所有参数组合
    results = []
    for thresh in PARAM_GRID['signal_threshold']:
        for freq in PARAM_GRID['rebalance_freq']:
            key = f"信号={thresh},调仓={freq}天"
            logger.info(f"测试: {key}...")
            
            try:
                records, trades = run_backtest(data_map, fund_map, thresh, freq)
                if len(records) < 100:
                    continue
                stats = compute_stats(records, trades)
                stats['param'] = key
                results.append(stats)
                logger.info(f"  完成: 收益={stats['total_return']:+.2%}, 夏普={stats['sharpe']:.2f}, 回撤={stats['max_drawdown']:.2%}")
            except Exception as e:
                logger.warning(f"  失败: {e}")
    
    if not results:
        logger.error("无有效结果")
        return
    
    # 排序
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    # 输出表格
    print("\n" + "=" * 100)
    print("📊 参数敏感性测试结果（按夏普比率排序）")
    print("=" * 100)
    print(f"{'参数组合':<30} {'总收益':>10} {'年化':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8} {'交易':>6} {'连亏天':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['param']:<30} {r['total_return']*100:>8.2f}% {r['annual_return']*100:>8.2f}% "
              f"{r['sharpe']:>7.2f} {r['max_drawdown']*100:>8.2f}% {r['win_rate']*100:>6.2f}% "
              f"{r['trades']:>5d} {r['max_consec_loss']:>7d}")
    
    # 最佳参数
    best = results[0]
    print("\n🏆 最佳参数组合:")
    print(f"  {best['param']}")
    print(f"  总收益: {best['total_return']:+.2%}, 夏普: {best['sharpe']:.2f}, 最大回撤: {best['max_drawdown']:.2%}")
    print(f"  胜率: {best['win_rate']:.2%}, 平均盈利: {best['avg_win']:+.2%}, 平均亏损: {best['avg_loss']:+.2%}")
    
    # 保存
    import json
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitivity_test_results.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n结果保存: {result_path}")


if __name__ == "__main__":
    main()
