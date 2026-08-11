"""
按今天的收盘数据跑一次：沪深300成分股 9因子回测
直接从最新缓存目录读取数据，确保使用2025-2026最新数据
"""
import os, sys, time, logging, hashlib, json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_latest")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest")
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt")

START = "2025-01-02"
END = "2026-08-10"

def load_stock_data():
    """直接从缓存文件加载K线和基本面数据"""
    logger.info("从最新缓存加载沪深300成分股数据...")
    with open(UNIVERSE_FILE, "r") as f:
        codes = [line.strip() for line in f if line.strip()]
    logger.info(f"沪深300 成分股: {len(codes)} 只")

    data_map = {}
    fund_map = {}
    ok_k, ok_f = 0, 0
    t0 = time.time()

    for code in codes:
        symbol = code.split('.')[0]
        kline_file = os.path.join(CACHE_DIR, f"{symbol}_hfq_kline.csv")
        fund_file = os.path.join(CACHE_DIR, f"{symbol}_fund.csv")

        if os.path.exists(kline_file):
            try:
                k = pd.read_csv(kline_file, index_col=0, parse_dates=True)
                k = k[(k.index >= pd.Timestamp(START)) & (k.index <= pd.Timestamp(END))]
                if not k.empty and len(k) >= 60:  # 至少60天数据用于技术指标预热
                    data_map[code] = k
                    ok_k += 1
            except Exception as e:
                logger.debug(f"{code} K线加载失败: {e}")

        if os.path.exists(fund_file):
            try:
                fnd = pd.read_csv(fund_file, index_col=0, parse_dates=True)
                fnd = fnd[(fnd.index >= pd.Timestamp(START)) & (fnd.index <= pd.Timestamp(END))]
                if code in data_map and not fnd.empty:
                    fnd = fnd.reindex(data_map[code].index).ffill().bfill()
                    fund_map[code] = fnd
                    ok_f += 1
            except Exception as e:
                logger.debug(f"{code} 基本面加载失败: {e}")

    elapsed = time.time() - t0
    logger.info(f"数据加载完成: {ok_k}/{len(codes)} 只K线, {ok_f}/{len(codes)} 只基本面, 用时 {elapsed:.1f}s")

    # 检查日期范围
    if data_map:
        all_dates = sorted(set().union(*(df.index for df in data_map.values())))
        logger.info(f"数据日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 个交易日")

    return codes, data_map, fund_map


def compute_dmi(df, period=14):
    """计算DMI指标，返回含 pdi, ndi, adx 列的DataFrame（安全列名）"""
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
    # 用安全列名避免 pandas 特殊字符问题
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


def compute_kdj(df, n=9, k_period=3, d_period=3):
    result = df.copy()
    low_n = result['low'].rolling(window=n).min()
    high_n = result['high'].rolling(window=n).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = (result['close'] - low_n) / rng * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1/k_period, adjust=False).mean()
    d = k.ewm(alpha=1/d_period, adjust=False).mean()
    result['K'] = k
    result['D'] = d
    result['J'] = 3 * k - 2 * d
    return result


def compute_rsi(df, period=14):
    result = df.copy()
    delta = result['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50)
    result['RSI'] = rsi
    return result


def compute_bollinger(df, period=20, num_std=2.0):
    result = df.copy()
    mid = result['close'].rolling(window=period).mean()
    std = result['close'].rolling(window=period).std()
    result['bb_mid'] = mid
    result['bb_upper'] = mid + num_std * std
    result['bb_lower'] = mid - num_std * std
    width = result['bb_upper'] - result['bb_lower']
    result['pct_b'] = (result['close'] - result['bb_lower']) / width.replace(0, np.nan)
    result['pct_b'] = result['pct_b'].fillna(0.5)
    result['bandwidth'] = width / mid.replace(0, np.nan)
    result['bandwidth'] = result['bandwidth'].fillna(0)
    return result


def compute_wyckoff(df, vol_period=20):
    result = df.copy()
    vol_sma = result['volume'].rolling(window=vol_period).mean()
    result['rel_volume'] = result['volume'] / vol_sma.replace(0, np.nan)
    result['rel_volume'] = result['rel_volume'].fillna(1.0)
    return result


def compute_ma_lines(df, ma13=13, ma34=34):
    result = df.copy()
    result['ma13'] = result['close'].rolling(window=ma13).mean()
    result['ma34'] = result['close'].rolling(window=ma34).mean()
    return result


def compute_fundamental_score(df, pe_min=0, pe_max=40, pb_max=10, roe_min=8):
    pe = df['pe'] if 'pe' in df.columns else pd.Series(np.nan, index=df.index)
    pb = df['pb'] if 'pb' in df.columns else pd.Series(np.nan, index=df.index)
    roe = df['roe'] if 'roe' in df.columns else pd.Series(np.nan, index=df.index)

    pe_valid = pe.between(pe_min, pe_max)
    pe_score = pd.Series(0.0, index=df.index)
    pe_score[pe_valid] = (1 - (pe[pe_valid] - pe_min) / (pe_max - pe_min)) * 30

    pb_valid = pb <= pb_max
    pb_score = pd.Series(0.0, index=df.index)
    pb_score[pb_valid] = (1 - pb[pb_valid] / pb_max) * 30

    roe_valid = roe.notna() & (roe >= roe_min)
    roe_score = pd.Series(0.0, index=df.index)
    if roe_valid.any():
        roe_score[roe_valid] = (roe[roe_valid] / 50 * 40).clip(upper=40)

    scores = pe_score + pb_score + roe_score
    mask = pe_valid & pb_valid
    scores[~mask] *= 0.5
    return scores


def sentiment_score(code, date):
    """确定性mock舆情"""
    import hashlib
    norm_code = code.split('.')[0]
    seed = int(hashlib.md5(f"{norm_code}{date}".encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    return np.random.uniform(-0.5, 0.5)


def run_backtest(data_map, fund_map):
    """执行9因子回测"""
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    logger.info(f"回测日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 个交易日")

    # 计算技术指标
    for code, df in data_map.items():
        data_map[code] = compute_dmi(df, period=14)
        data_map[code] = compute_macd(data_map[code])
        data_map[code] = compute_kdj(data_map[code])
        data_map[code] = compute_rsi(data_map[code])
        data_map[code] = compute_bollinger(data_map[code])
        data_map[code] = compute_wyckoff(data_map[code])
        data_map[code] = compute_ma_lines(data_map[code])

    initial_capital = 1_000_000
    cash = initial_capital
    peak_nav = initial_capital
    daily_records = []
    current_positions = {}
    position_shares = {}
    position_cost_basis = {}
    position_dates = {}
    trade_log = []

    commission_rate = 0.0003
    stamp_rate = 0.0005
    slippage_rate = 0.001
    stop_loss = -0.10

    t0 = time.time()
    for di, date in enumerate(all_dates):
        # 净值
        nav = cash
        for code, shares in position_shares.items():
            if code in data_map and date in data_map[code].index:
                nav += shares * data_map[code].loc[date, 'close']
        if nav > peak_nav:
            peak_nav = nav
        drawdown = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0

        # 止损
        to_sell = []
        for code in list(current_positions.keys()):
            if code not in data_map or date not in data_map[code].index:
                continue
            cp = data_map[code].loc[date, 'close']
            pnl = (cp - current_positions[code]) / current_positions[code]
            if pnl <= stop_loss:
                to_sell.append((code, cp, "硬止损"))

        for code, sp, reason in to_sell:
            shares = position_shares[code]
            cost = position_cost_basis[code]
            fill = sp * (1 - slippage_rate)
            gross = fill * shares
            comm = max(gross * commission_rate, 5.0)
            stamp = gross * stamp_rate
            proceeds = gross - comm - stamp
            cash += proceeds
            pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
            trade_log.append({'code': code, 'buy_date': position_dates.get(code, ''),
                             'buy_price': current_positions.get(code, sp), 'sell_price': sp,
                             'reason': reason, 'pnl_pct': pnl_pct})
            position_shares.pop(code, None)
            position_cost_basis.pop(code, None)
            position_dates.pop(code, None)
            current_positions.pop(code, None)

        # 调仓日
        if di % 20 == 0:
            scores = {}
            for code, df in data_map.items():
                if date not in df.index:
                    continue
                i = df.index.get_loc(date)

                # DMI
                if i < 14:
                    dmi_s = 50
                else:
                    cur = df.iloc[i]
                    if cur['pdi'] <= cur['ndi']:
                        dmi_s = 0
                    elif cur['adx'] > 25:
                        dmi_s = min((cur['pdi'] - cur['ndi']) * 2, 100)
                    else:
                        dmi_s = 30

                # MACD
                if 'DIF' not in df.columns or i < 35:
                    macd_s = 50
                else:
                    cur, prev = df.iloc[i], df.iloc[i-1]
                    macd_s = 50
                    if cur['DIF'] > cur['DEA']: macd_s += 20
                    if cur['DIF'] > 0: macd_s += 15
                    if prev['DIF'] <= prev['DEA'] and cur['DIF'] > cur['DEA']: macd_s += 15
                    if cur['MACD_bar'] > 0: macd_s += 10
                    if cur['MACD_bar'] > prev['MACD_bar']: macd_s += 10

                # KDJ
                if 'K' not in df.columns or i < 9:
                    kdj_s = 50
                else:
                    cur, prev = df.iloc[i], df.iloc[i-1]
                    kdj_s = 50
                    if cur['K'] < 20: kdj_s += 25
                    elif cur['K'] > 80: kdj_s -= 25
                    if prev['K'] <= prev['D'] and cur['K'] > cur['D']: kdj_s += 20
                    elif prev['K'] >= prev['D'] and cur['K'] < cur['D']: kdj_s -= 20
                    if cur['J'] < 0: kdj_s += 10
                    if cur['J'] > 100: kdj_s -= 10

                # RSI
                if 'RSI' not in df.columns or i < 14:
                    rsi_s = 50
                else:
                    r = df.iloc[i]['RSI']
                    if r < 20: rsi_s = 95
                    elif r < 30: rsi_s = 80 + (30-r)*1.5
                    elif r < 50: rsi_s = 50 + (50-r)
                    elif r < 70: rsi_s = 50 - (r-50)
                    elif r < 80: rsi_s = 30 - (r-70)*2
                    else: rsi_s = 10

                # Bollinger
                if 'pct_b' not in df.columns or i < 20:
                    bb_s = 50
                else:
                    pct_b = df.iloc[i]['pct_b']
                    bb_s = min(max(50 + (0.5-pct_b)*60, 0), 100)
                    if df.iloc[i].get('bandwidth', 0) < 0.06: bb_s += 10

                # Wyckoff
                if 'rel_volume' not in df.columns or i < 20:
                    wyck_s = 50
                else:
                    cur, prev = df.iloc[i], df.iloc[i-1]
                    rv = cur['rel_volume']
                    pc = (cur['close'] - prev['close']) / prev['close']
                    wyck_s = 50
                    if pc > 0 and rv > 1.5: wyck_s += 30
                    elif pc < 0 and rv > 1.5: wyck_s -= 30
                    if rv > 2.5:
                        if pc <= 0: wyck_s += 10
                        else: wyck_s -= 10
                    if rv < 0.6: wyck_s += 5

                # 135
                if 'ma13' not in df.columns or i < 34:
                    m135_s = 50
                else:
                    cur, prev = df.iloc[i], df.iloc[i-1]
                    m135_s = 50
                    if cur['close'] > cur['ma13']: m135_s += 20
                    if cur['ma13'] > cur['ma34']: m135_s += 20
                    if cur['ma13'] > prev['ma13']: m135_s += 20
                    if prev['ma13'] <= prev['ma34'] and cur['ma13'] > cur['ma34']: m135_s += 20
                    if cur.get('rel_volume', 1.0) > 1.3 and cur['close'] > cur['ma13']: m135_s += 10

                # 基本面
                fund_s = 50.0
                if code in fund_map and date in fund_map[code].index:
                    try:
                        fund_s = compute_fundamental_score(fund_map[code]).get(date, 50.0)
                    except:
                        pass

                # 舆情
                sent = sentiment_score(code, str(date))
                sent_f = (sent + 1) / 2 * 100

                composite = (0.12*dmi_s + 0.12*macd_s + 0.08*kdj_s + 0.08*rsi_s +
                            0.08*bb_s + 0.10*wyck_s + 0.10*m135_s + 0.18*fund_s + 0.14*sent_f)
                scores[code] = composite

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = set(c for c, _ in ranked[:5])
            current_held = set(current_positions.keys())

            for code in current_held - selected:
                if date in data_map[code].index:
                    sp = data_map[code].loc[date, 'close']
                    shares = position_shares[code]
                    fill = sp * (1 - slippage_rate)
                    gross = fill * shares
                    comm = max(gross * commission_rate, 5.0)
                    stamp = gross * stamp_rate
                    proceeds = gross - comm - stamp
                    cash += proceeds
                    cost = position_cost_basis.get(code, gross)
                    pnl_pct = (proceeds - cost) / cost if cost > 0 else 0
                    trade_log.append({'code': code, 'buy_date': position_dates.get(code, ''),
                                     'buy_price': current_positions.get(code, sp), 'sell_price': sp,
                                     'reason': '调仓卖出', 'pnl_pct': pnl_pct})
                    position_shares.pop(code, None)
                    position_cost_basis.pop(code, None)
                    position_dates.pop(code, None)
                    current_positions.pop(code, None)

            to_buy = selected - current_held
            if to_buy and cash > 0:
                equal_budget = cash / len(to_buy)
                for code in to_buy:
                    if date in data_map[code].index:
                        sp = data_map[code].loc[date, 'close']
                        fill = sp * (1 + slippage_rate)
                        shares = int(equal_budget / fill / 100) * 100
                        if shares <= 0: continue
                        gross = fill * shares
                        comm = max(gross * commission_rate, 5.0)
                        total_cost = gross + comm
                        if total_cost > cash:
                            shares = int((cash / (fill * (1 + commission_rate))) / 100) * 100
                            if shares <= 0: continue
                            gross = fill * shares
                            comm = max(gross * commission_rate, 5.0)
                            total_cost = gross + comm
                        cash -= total_cost
                        current_positions[code] = sp
                        position_dates[code] = str(date)
                        position_shares[code] = shares
                        position_cost_basis[code] = total_cost

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
    logger.info(f"回测完成, 用时 {elapsed:.1f}s")
    return daily_records, trade_log


def main():
    logger.info("=" * 70)
    logger.info("DMI 多因子选股策略 — 沪深300成分股 回测（今日收盘数据）")
    logger.info("=" * 70)

    codes, data_map, fund_map = load_stock_data()

    if len(data_map) == 0:
        logger.error("没有可用的股票数据")
        return

    daily_records, trade_log = run_backtest(data_map, fund_map)

    result_df = pd.DataFrame(daily_records)
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

    print("\n" + "=" * 70)
    print("📊 DMI 多因子选股策略回测结果（沪深300成分股 · 最新收盘数据）")
    print("=" * 70)
    print(f"回测区间:     {result_df['date'].iloc[0]} ~ {result_df['date'].iloc[-1]} ({trading_days}个交易日)")
    print(f"股票池:       沪深300成分股 ({len(codes)}只)")
    print(f"有效数据:     {len(data_map)}只K线, {len(fund_map)}只基本面")
    print(f"初始资金:     1,000,000 元")
    print(f"持仓数:       5 只")
    print(f"调仓频率:     20 交易日")
    print("-" * 70)
    print(f"总收益率:     {total_return:+.2%}")
    print(f"年化收益:     {annual_return:+.2%}")
    print(f"夏普比率:     {sharpe:.2f}")
    print(f"最大回撤:     {max_dd:.2%}")
    print(f"胜率:         {win_rate:.2%}")
    print(f"交易次数:     {len(trade_log)}")
    final_nav = result_df['nav'].iloc[-1]
    print(f"最终净值:     {final_nav:,.0f} 元")
    print("=" * 70)

    print(f"\n📋 最近20笔交易:")
    print("-" * 70)
    for t in trade_log[-20:]:
        print(f"  {t['code']} | 买入: {t['buy_price']:.2f} -> 卖出: {t['sell_price']:.2f} | "
              f"{t['pnl_pct']:+.2%} | {t['reason']}")

    # 保存结果
    stats = {
        'total_return': total_return, 'annual_return': annual_return,
        'sharpe_ratio': sharpe, 'max_drawdown': max_dd,
        'win_rate': win_rate, 'total_trades': len(trade_log),
        'final_nav': float(final_nav), 'trading_days': trading_days,
        'start_date': str(result_df['date'].iloc[0]),
        'end_date': str(result_df['date'].iloc[-1]),
    }
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_backtest_result_v2.json")
    with open(result_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
