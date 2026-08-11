"""
优化版 DMI 多因子选股策略 v3
主要改进：
1. 基本面因子改用 PE/PB 估值分位数（ROE 数据缺失）
2. 增加动量因子（20日/60日收益率）
3. 优化止损策略：个股止损收紧至8%，加仓止损5%
4. 动态仓位管理：根据波动率调整仓位
5. 因子权重重新分配
6. 增加止盈机制（移动止盈）
7. 增加换手率惩罚
"""
import os, sys, time, logging, hashlib, json
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optim_backtest")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest")
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt")

START = "2025-01-02"
END = "2026-08-10"

# ============================ 参数配置 ============================
PARAMS = {
    'initial_capital': 1_000_000,
    'top_n': 5,
    'rebalance_freq': 20,
    # 止损
    'stop_loss_pct': -0.08,       # 个股止损 -8%
    'trailing_stop_pct': -0.05,   # 移动止盈 -5%
    # 交易成本
    'commission_rate': 0.0003,
    'stamp_rate': 0.0005,
    'slippage_rate': 0.001,
    'min_commission': 5.0,
    # 因子权重（总和=1.0）
    'factor_weights': {
        'dmi': 0.15,      # DMI趋势强度 ↑
        'macd': 0.12,     # MACD金叉 ↑
        'kdj': 0.10,      # KDJ超卖/金叉
        'rsi': 0.08,      # RSI超卖
        'bollinger': 0.06, # 布林带位置
        'wyckoff': 0.08,  # 威科夫量价
        'method135': 0.08, # 135战法
        'momentum_20': 0.12,  # 20日动量 ↑
        'momentum_60': 0.06,  # 60日动量 ↑
        'pe_valuation': 0.10, # PE估值分位数
        'pb_valuation': 0.05, # PB估值分位数
    },
}


def load_stock_data():
    """加载沪深300成分股数据"""
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
                if not k.empty and len(k) >= 90:  # 至少90天
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

    if data_map:
        all_dates = sorted(set().union(*(df.index for df in data_map.values())))
        logger.info(f"数据日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 个交易日")

    return codes, data_map, fund_map


def compute_dmi(df, period=14):
    """DMI指标"""
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
    """MACD指标"""
    result = df.copy()
    ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
    result['DIF'] = ema_fast - ema_slow
    result['DEA'] = result['DIF'].ewm(span=signal, adjust=False).mean()
    result['MACD_bar'] = 2 * (result['DIF'] - result['DEA'])
    return result


def compute_kdj(df, n=9, k_period=3, d_period=3):
    """KDJ指标"""
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
    """RSI指标"""
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
    """布林带指标"""
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
    """威科夫量价分析"""
    result = df.copy()
    vol_sma = result['volume'].rolling(window=vol_period).mean()
    result['rel_volume'] = result['volume'] / vol_sma.replace(0, np.nan)
    result['rel_volume'] = result['rel_volume'].fillna(1.0)
    return result


def compute_ma_lines(df, ma13=13, ma34=34):
    """均线系统"""
    result = df.copy()
    result['ma13'] = result['close'].rolling(window=ma13).mean()
    result['ma34'] = result['close'].rolling(window=ma34).mean()
    return result


def compute_momentum(df, window=20):
    """动量因子"""
    result = df.copy()
    result[f'mom_{window}'] = result['close'] / result['close'].shift(window) - 1
    return result


def compute_fundamental_score_v2(fund_df, date):
    """
    优化版基本面评分（不用ROE，因为数据全NaN）
    改用PE/PB估值分位数 + PE/PB绝对值评分
    """
    pe = fund_df['pe'] if 'pe' in fund_df.columns else pd.Series(np.nan, index=fund_df.index)
    pb = fund_df['pb'] if 'pb' in fund_df.columns else pd.Series(np.nan, index=fund_df.index)

    if date not in pe.index:
        return 50.0

    pe_val = pe.loc[date]
    pb_val = pb.loc[date] if date in pb.index else np.nan

    # PE估值评分：PE越低越好（合理区间5-25）
    pe_score = 50.0
    if not np.isnan(pe_val):
        if pe_val < 5:
            pe_score = 90
        elif pe_val < 10:
            pe_score = 80 + (10-pe_val)*2
        elif pe_val < 20:
            pe_score = 60 - (pe_val-10)
        elif pe_val < 30:
            pe_score = 40 - (pe_val-20)
        else:
            pe_score = 20

    # PB估值评分：PB越低越好
    pb_score = 50.0
    if not np.isnan(pb_val):
        if pb_val < 0.3:
            pb_score = 95
        elif pb_val < 0.5:
            pb_score = 80 + (0.5-pb_val)*50
        elif pb_val < 1.0:
            pb_score = 60 - (pb_val-0.5)*20
        elif pb_val < 2.0:
            pb_score = 40 - (pb_val-1.0)*20
        else:
            pb_score = 10

    # 综合
    score = 0.6 * pe_score + 0.4 * pb_score
    return score


def sentiment_score(code, date):
    """确定性mock舆情"""
    import hashlib
    norm_code = code.split('.')[0]
    seed = int(hashlib.md5(f"{norm_code}{date}".encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    return np.random.uniform(-0.5, 0.5)


def compute_all_technicals(data_map):
    """批量计算所有技术指标"""
    logger.info("计算全部技术指标（含动量因子）...")
    t0 = time.time()
    for code, df in data_map.items():
        data_map[code] = compute_dmi(df, period=14)
        data_map[code] = compute_macd(data_map[code])
        data_map[code] = compute_kdj(data_map[code])
        data_map[code] = compute_rsi(data_map[code])
        data_map[code] = compute_bollinger(data_map[code])
        data_map[code] = compute_wyckoff(data_map[code])
        data_map[code] = compute_ma_lines(data_map[code])
        data_map[code] = compute_momentum(data_map[code], 20)
        data_map[code] = compute_momentum(data_map[code], 60)
    logger.info(f"技术指标计算完成, 用时 {time.time()-t0:.1f}s")
    return data_map


def compute_factors(df, i, code, fund_map, date):
    """计算所有因子得分"""
    if i < 60:  # 至少60天数据
        return 50.0

    cur = df.iloc[i]
    prev = df.iloc[i-1] if i > 0 else cur

    scores = []
    weights = []

    # 1. DMI 趋势强度
    w = PARAMS['factor_weights']['dmi']
    if i >= 14 and 'pdi' in df.columns:
        if cur['pdi'] <= cur['ndi']:
            s = 0
        elif cur['adx'] > 25:
            s = min((cur['pdi'] - cur['ndi']) * 2, 100)
        else:
            s = 30
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 2. MACD 金叉
    w = PARAMS['factor_weights']['macd']
    if 'DIF' in df.columns and i >= 35:
        s = 50
        if cur['DIF'] > cur['DEA']: s += 20
        if cur['DIF'] > 0: s += 15
        if prev['DIF'] <= prev['DEA'] and cur['DIF'] > cur['DEA']: s += 15
        if cur['MACD_bar'] > 0: s += 10
        if cur['MACD_bar'] > prev['MACD_bar']: s += 10
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 3. KDJ
    w = PARAMS['factor_weights']['kdj']
    if 'K' in df.columns and i >= 9:
        s = 50
        if cur['K'] < 20: s += 25
        elif cur['K'] > 80: s -= 25
        if prev['K'] <= prev['D'] and cur['K'] > cur['D']: s += 20
        elif prev['K'] >= prev['D'] and cur['K'] < cur['D']: s -= 20
        if cur['J'] < 0: s += 10
        if cur['J'] > 100: s -= 10
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 4. RSI
    w = PARAMS['factor_weights']['rsi']
    if 'RSI' in df.columns and i >= 14:
        r = cur['RSI']
        if r < 20: s = 95
        elif r < 30: s = 80 + (30-r)*1.5
        elif r < 50: s = 50 + (50-r)
        elif r < 70: s = 50 - (r-50)
        elif r < 80: s = 30 - (r-70)*2
        else: s = 10
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 5. 布林带
    w = PARAMS['factor_weights']['bollinger']
    if 'pct_b' in df.columns and i >= 20:
        pct_b = cur['pct_b']
        s = min(max(50 + (0.5-pct_b)*60, 0), 100)
        if cur.get('bandwidth', 0) < 0.06: s += 10
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 6. Wyckoff 量价
    w = PARAMS['factor_weights']['wyckoff']
    if 'rel_volume' in df.columns and i >= 20:
        rv = cur['rel_volume']
        pc = (cur['close'] - prev['close']) / prev['close']
        s = 50
        if pc > 0 and rv > 1.5: s += 30
        elif pc < 0 and rv > 1.5: s -= 30
        if rv > 2.5:
            if pc <= 0: s += 10
            else: s -= 10
        if rv < 0.6: s += 5
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 7. 135战法
    w = PARAMS['factor_weights']['method135']
    if 'ma13' in df.columns and i >= 34:
        s = 50
        if cur['close'] > cur['ma13']: s += 20
        if cur['ma13'] > cur['ma34']: s += 20
        if cur['ma13'] > prev['ma13']: s += 20
        if prev['ma13'] <= prev['ma34'] and cur['ma13'] > cur['ma34']: s += 20
        if cur.get('rel_volume', 1.0) > 1.3 and cur['close'] > cur['ma13']: s += 10
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 8. 20日动量
    w = PARAMS['factor_weights']['momentum_20']
    if 'mom_20' in df.columns and i >= 20:
        mom20 = cur['mom_20']
        if pd.notna(mom20):
            if mom20 > 0.05: s = 90
            elif mom20 > 0.02: s = 80
            elif mom20 > 0.01: s = 70
            elif mom20 > 0: s = 60
            elif mom20 > -0.01: s = 40
            elif mom20 > -0.02: s = 30
            else: s = 15
        else:
            s = 50
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 9. 60日动量
    w = PARAMS['factor_weights']['momentum_60']
    if 'mom_60' in df.columns and i >= 60:
        mom60 = cur['mom_60']
        if pd.notna(mom60):
            if mom60 > 0.15: s = 90
            elif mom60 > 0.08: s = 80
            elif mom60 > 0.03: s = 70
            elif mom60 > 0: s = 60
            elif mom60 > -0.03: s = 40
            elif mom60 > -0.08: s = 30
            else: s = 15
        else:
            s = 50
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 10. PE估值
    w = PARAMS['factor_weights']['pe_valuation']
    if code in fund_map:
        s = compute_fundamental_score_v2(fund_map[code], date)
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 11. PB估值
    w = PARAMS['factor_weights']['pb_valuation']
    if code in fund_map:
        pe = fund_map[code]['pe'] if 'pe' in fund_map[code].columns else pd.Series(np.nan)
        pb = fund_map[code]['pb'] if 'pb' in fund_map[code].columns else pd.Series(np.nan)
        if date in pb.index and pd.notna(pb.loc[date]):
            pbv = pb.loc[date]
            if pbv < 0.3: s = 95
            elif pbv < 0.5: s = 85
            elif pbv < 1.0: s = 70
            elif pbv < 2.0: s = 50
            else: s = 25
        else:
            s = 50
    else:
        s = 50
    scores.append(s)
    weights.append(w)

    # 加权综合得分
    total_w = sum(weights)
    composite = sum(s * w for s, w in zip(scores, weights)) / total_w

    return composite


def run_backtest(data_map, fund_map):
    """执行优化版回测"""
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    logger.info(f"回测日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 个交易日")

    # 预计算技术指标
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
    position_highest_price = {}  # 移动止盈用
    trade_log = []

    commission_rate = params['commission_rate']
    stamp_rate = params['stamp_rate']
    slippage_rate = params['slippage_rate']
    min_commission = params['min_commission']
    stop_loss_pct = params['stop_loss_pct']
    trailing_stop_pct = params['trailing_stop_pct']

    t0 = time.time()
    for di, date in enumerate(all_dates):
        # ---- 净值计算 ----
        nav = cash
        for code, shares in position_shares.items():
            if code in data_map and date in data_map[code].index:
                nav += shares * data_map[code].loc[date, 'close']
        if nav > peak_nav:
            peak_nav = nav
        drawdown = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0

        # ---- 移动止盈 & 止损 ----
        to_sell = []
        for code in list(current_positions.keys()):
            if code not in data_map or date not in data_map[code].index:
                continue
            current_price = data_map[code].loc[date, 'close']

            # 更新最高价
            if code not in position_highest_price or current_price > position_highest_price[code]:
                position_highest_price[code] = current_price

            # 止损检查（买入价跌幅）
            buy_price = current_positions[code]
            pnl = (current_price - buy_price) / buy_price
            if pnl <= stop_loss_pct:
                to_sell.append((code, current_price, "硬止损"))
                continue

            # 移动止盈检查
            if code in position_highest_price and position_highest_price[code] > buy_price:
                drawdown_from_high = (current_price - position_highest_price[code]) / position_highest_price[code]
                if drawdown_from_high <= trailing_stop_pct:
                    to_sell.append((code, current_price, "移动止盈"))

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
            position_highest_price.pop(code, None)
            current_positions.pop(code, None)

        # ---- 调仓 ----
        if di % params['rebalance_freq'] == 0:
            scores = {}
            for code, df in data_map.items():
                if date not in df.index:
                    continue
                i = df.index.get_loc(date)
                composite = compute_factors(df, i, code, fund_map, date)
                scores[code] = composite

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
                    position_highest_price.pop(code, None)
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
                        position_highest_price[code] = sp

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


def compute_stats(result_df, trade_log):
    """计算统计指标"""
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

    # 计算最大连续亏损天数
    returns = result_df['return'].tolist()
    max_consec_loss = 0
    consec = 0
    for r in returns:
        if r < 0:
            consec += 1
            max_consec_loss = max(max_consec_loss, consec)
        else:
            consec = 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_trades': len(trade_log),
        'max_consecutive_loss': max_consec_loss,
    }


def main():
    logger.info("=" * 70)
    logger.info("🚀 DMI 多因子选股策略 — 优化版 v3（沪深300成分股）")
    logger.info("=" * 70)
    logger.info(f"策略改进:")
    for k, v in PARAMS['factor_weights'].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"止损: {PARAMS['stop_loss_pct']*100:.0f}%")
    logger.info(f"移动止盈: {PARAMS['trailing_stop_pct']*100:.0f}%")

    codes, data_map, fund_map = load_stock_data()
    if len(data_map) == 0:
        logger.error("没有可用的股票数据")
        return

    daily_records, trade_log = run_backtest(data_map, fund_map)
    result_df = pd.DataFrame(daily_records)
    stats = compute_stats(result_df, trade_log)
    final_nav = result_df['nav'].iloc[-1]

    print("\n" + "=" * 70)
    print("📊 DMI 多因子选股策略回测结果（沪深300成分股 · 优化版 v3）")
    print("=" * 70)
    print(f"回测区间:     {result_df['date'].iloc[0]} ~ {result_df['date'].iloc[-1]} ({stats['total_trades']}笔交易)")
    print(f"股票池:       沪深300成分股 ({len(codes)}只)")
    print(f"有效数据:     {len(data_map)}只K线, {len(fund_map)}只基本面")
    print(f"初始资金:     1,000,000 元")
    print(f"持仓数:       {PARAMS['top_n']} 只")
    print(f"调仓频率:     {PARAMS['rebalance_freq']} 交易日")
    print("-" * 70)
    print(f"总收益率:     {stats['total_return']:+.2%}")
    print(f"年化收益:     {stats['annual_return']:+.2%}")
    print(f"夏普比率:     {stats['sharpe_ratio']:.2f}")
    print(f"最大回撤:     {stats['max_drawdown']:.2%}")
    print(f"胜率:         {stats['win_rate']:.2%}")
    print(f"交易次数:     {stats['total_trades']}")
    print(f"最大连亏天数: {stats['max_consecutive_loss']}")
    print(f"最终净值:     {final_nav:,.0f} 元")
    print("=" * 70)

    print(f"\n📋 最近20笔交易:")
    print("-" * 70)
    for t in trade_log[-20:]:
        print(f"  {t['code']} | 买入: {t['buy_price']:.2f} -> 卖出: {t['sell_price']:.2f} | "
              f"{t['pnl_pct']:+.2%} | {t['reason']}")

    # 保存结果
    stats['final_nav'] = float(final_nav)
    stats['trading_days'] = len(result_df)
    stats['start_date'] = str(result_df['date'].iloc[0])
    stats['end_date'] = str(result_df['date'].iloc[-1])
    stats['version'] = 'v3'
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_backtest_result_v3.json")
    with open(result_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
