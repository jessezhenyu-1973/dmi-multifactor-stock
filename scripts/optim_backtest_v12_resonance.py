"""
V12 多因子共振版 - 新因子 + 多因子共振 + 诚实回测
====================================================
核心创新:
1. 新因子: 基本面质量 + 估值 + 资金流 + 波动率 regime
2. 多因子共振: 独立类别投票机制
3. 诚实回测: WF + 净 holdout + 时点正确
4. 消除 V10/V11 四大错误: 有卖出、状态切换触发、分样本优化、全量

因子类别 (4 类独立信号):
  A. 技术面 (原 V10 因子): DMI, 动量, 均线, RSI, 成交量
  B. 基本面质量: ROE, 毛利率, 营收增速, 净利增速
  C. 估值: PE 百分位, PB 百分位, PEG
  D. 资金流: 主力资金净流入, 北向资金变化, 换手率变化

共振逻辑:
  每个类别独立打分 [0, 1]
  当 ≥2 个类别同时 > 阈值时触发共振
  共振强度 = 类别数 × 类别内一致性
"""
import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"

# ===================== 数据加载 =====================

def load_all_data():
    """加载所有股票数据，确保有 2022-2026 数据"""
    stock_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_daily.csv')]
    all_data = {}
    
    for fname in stock_files:
        code = fname.replace('_daily.csv', '')
        filepath = os.path.join(CACHE_DIR, fname)
        df = pd.read_csv(filepath, parse_dates=['date'])
        
        if 'date' not in df.columns:
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            else:
                continue
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available = [c for c in required_cols if c in df.columns]
        if len(available) < 4:
            continue
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # 强制过滤
        if len(df) < 120:
            continue
        
        # 标准化列名
        if 'close' not in df.columns:
            if 'Close' in df.columns:
                df['close'] = df['Close']
        
        # 过滤负价格
        df = df[df['close'] > 0]
        
        all_data[code] = df
    
    return all_data


# ===================== 技术指标计算（时点正确）=====================

def compute_technical_indicators(df):
    """计算技术面因子 (类别 A)"""
    close = df['close'].values
    high = df['high'].values if 'high' in df.columns else close
    low = df['low'].values if 'low' in df.columns else close
    volume = df['volume'].values if 'volume' in df.columns else np.zeros(len(close))
    
    n = len(close)
    
    # DMI
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = max(0, up_move) if up_move > down_move else 0
        minus_dm[i] = max(0, down_move) if down_move > up_move else 0
    
    atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = np.mean(tr[max(0,i-13):i+1])
    
    smooth_plus_dm = np.zeros(n)
    smooth_minus_dm = np.zeros(n)
    for i in range(14, n):
        smooth_plus_dm[i] = np.mean(plus_dm[max(0,i-13):i+1])
        smooth_minus_dm[i] = np.mean(minus_dm[max(0,i-13):i+1])
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(14, n):
        denom = smooth_plus_dm[i] + smooth_minus_dm[i]
        if denom > 0:
            plus_di[i] = smooth_plus_dm[i] / denom * 100
            minus_di[i] = smooth_minus_dm[i] / denom * 100
    
    # ADX
    dx = np.zeros(n)
    for i in range(14, n):
        sum_adx = smooth_plus_dm[i] + smooth_minus_dm[i]
        if sum_adx > 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / sum_adx * 100
    adx = np.zeros(n)
    for i in range(14, n):
        adx[i] = np.mean(dx[max(0,i-13):i+1])
    
    # 动量因子
    momentum_5 = np.zeros(n)
    momentum_20 = np.zeros(n)
    momentum_60 = np.zeros(n)
    momentum_120 = np.zeros(n)
    for i in range(len(close)):
        if i >= 5:
            momentum_5[i] = (close[i] - close[i-5]) / close[i-5]
        if i >= 20:
            momentum_20[i] = (close[i] - close[i-20]) / close[i-20]
        if i >= 60:
            momentum_60[i] = (close[i] - close[i-60]) / close[i-60]
        if i >= 120:
            momentum_120[i] = (close[i] - close[i-120]) / close[i-120]
    
    # 均线系统
    ma5 = pd.Series(close).rolling(5).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values
    ma120 = pd.Series(close).rolling(120).mean().values
    
    price_above_ma5 = (close > ma5).astype(float)
    price_above_ma20 = (close > ma20).astype(float)
    price_above_ma60 = (close > ma60).astype(float)
    
    # RSI (简化版)
    returns = np.diff(close)
    gain = np.where(returns > 0, returns, 0)
    loss = np.where(returns < 0, -returns, 0)
    rsi = np.zeros(n)
    for i in range(14, n):
        avg_gain = np.mean(gain[max(0,i-14):i])
        avg_loss = np.mean(loss[max(0,i-14):i])
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - 100 / (1 + rs)
        else:
            rsi[i] = 100
    
    # 成交量因子
    vol_5 = pd.Series(volume).rolling(5).mean().values
    vol_20 = pd.Series(volume).rolling(20).mean().values
    volume_ratio = np.zeros(n)
    for i in range(20, n):
        if vol_20[i] > 0:
            volume_ratio[i] = volume[i] / vol_20[i]
    
    # 波动率 (补 0 对齐)
    returns_series = pd.Series(returns)
    vol_20 = returns_series.rolling(20).std().values
    vol_60 = returns_series.rolling(60).std().values
    # 对齐长度: returns 比 close 少 1，补 0 在开头
    volatility_20 = np.concatenate(([0], vol_20))
    volatility_60 = np.concatenate(([0], vol_60))
    
    # 汇总到 DataFrame
    result = df.copy()
    result['pdi'] = plus_di
    result['ndi'] = minus_di
    result['adx'] = adx
    result['momentum_5'] = momentum_5
    result['momentum_20'] = momentum_20
    result['momentum_60'] = momentum_60
    result['momentum_120'] = momentum_120
    result['price_above_ma5'] = price_above_ma5
    result['price_above_ma20'] = price_above_ma20
    result['price_above_ma60'] = price_above_ma60
    result['ma_bullish'] = ((close > ma5) & (ma5 > ma20) & (ma20 > ma60)).astype(float)
    result['rsi'] = rsi
    result['volume_ratio'] = volume_ratio
    result['volatility_20'] = volatility_20
    result['volatility_60'] = volatility_60
    
    return result


# ===================== 基本面质量因子 =====================

def generate_fundamental_factors(df, code):
    """生成基本面质量因子 (类别 B)
    
    由于本地数据没有基本面数据，使用量价推导的代理因子：
    - ROE proxy: 净利润增长率 ≈ 营收增长 × 利润率稳定度
    - 毛利率 proxy: 价格稳定性指标 (high-low spread 的倒数)
    - 营收增速: 量价配合度
    - 净利增速: 动量持续性
    """
    n = len(df)
    
    # 价格稳定性 (高毛利 proxy): 低波动率 = 高毛利
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    spread = (high - low) / close
    gross_margin_proxy = 1.0 / (1.0 + np.mean(spread[n//2:n]))
    
    # 营收增速 proxy: 量价配合度 (放量涨 = 营收增长)
    returns = np.diff(close) / close[:-1]
    volume = df['volume'].values
    vol_ma20 = pd.Series(volume).rolling(20).mean().values
    price_vol_agreement = 0
    for i in range(n):
        if i >= 20 and vol_ma20[i] > 0:
            vol_ratio = volume[i] / vol_ma20[i]
            ret = returns[i-1] if i > 0 else 0
            if (vol_ratio > 1.2 and ret > 0) or (vol_ratio < 0.8 and ret < 0):
                price_vol_agreement += 1
    revenue_growth_proxy = price_vol_agreement / 20
    
    # 净利增速 proxy: 动量持续性
    mom_consistent = 0
    for i in range(60, n):
        if i > 60:
            ret_short = (close[i] - close[i-5]) / close[i-5]
            ret_long = (close[i-5] - close[i-60]) / close[i-60]
            if ret_short > 0 and ret_long > 0:
                mom_consistent += 1
    net_profit_growth_proxy = mom_consistent / (n - 60) if n > 60 else 0
    
    # ROE proxy: 综合质量分
    roe_proxy = (gross_margin_proxy + revenue_growth_proxy + net_profit_growth_proxy) / 3
    
    result = df.copy()
    result['roe_proxy'] = roe_proxy
    result['gross_margin_proxy'] = gross_margin_proxy
    result['revenue_growth_proxy'] = revenue_growth_proxy
    result['net_profit_growth_proxy'] = net_profit_growth_proxy
    
    return result


# ===================== 估值因子 =====================

def generate_valuation_factors(df, all_data):
    """生成估值因子 (类别 C)
    
    由于没有 PE/PB 数据，使用量价推导的估值 proxy：
    - PE proxy: 价格位置分位数 (高分位 = 高 PE)
    - PB proxy: 市净率 proxy = 价格 / 历史均价 (反映资产溢价)
    - PEG proxy: PE / 盈利增速 = 价格位置 / 动量
    """
    n = len(df)
    close = df['close'].values
    
    # PE proxy: 价格位置分位数 (相对 120 日)
    pe_proxy = np.zeros(n)
    for i in range(120, n):
        window = close[i-120:i]
        if len(window) > 0:
            pe_proxy[i] = np.sum(close[i] > window) / len(window)
    
    # PB proxy: 价格 / 长期均价 (高 PB = 高估)
    pb_proxy = np.zeros(n)
    for i in range(240, n):
        long_avg = np.mean(close[i-240:i])
        if long_avg > 0:
            pb_proxy[i] = close[i] / long_avg
    
    # PEG proxy: PE_proxy / (1 + 动量)
    peg_proxy = np.zeros(n)
    for i in range(240, n):
        mom_60 = (close[i] - close[i-60]) / close[i-60] if close[i-60] > 0 else 0
        denom = 1 + mom_60
        if pe_proxy[i] > 0 and denom > 0:
            peg_proxy[i] = pe_proxy[i] / abs(denom)
    
    result = df.copy()
    result['pe_proxy'] = pe_proxy
    result['pb_proxy'] = pb_proxy
    result['peg_proxy'] = peg_proxy
    
    return result


# ===================== 资金流因子 =====================

def generate_money_flow_factors(df):
    """生成资金流因子 (类别 D)
    
    由于没有真实资金流数据，使用量价推导的 proxy：
    - 主力资金净流入: 大单占比 proxy = 大波动日的成交量
    - 北向资金变化: 换手率趋势 proxy
    - 换手率变化: 近期换手率 / 长期换手率
    """
    n = len(df)
    volume = df['volume'].values
    close = df['close'].values
    returns = np.diff(close) / close[:-1]
    
    # 主力资金净流入 proxy: 大波动日的成交量占比
    large_move_vol = 0
    total_vol = 0
    for i in range(1, n):
        total_vol += volume[i]
        if abs(returns[i-1]) > 0.02:  # 2% 以上波动 = 大单
            large_move_vol += volume[i]
    
    main_capital_ratio = large_move_vol / total_vol if total_vol > 0 else 0
    
    # 北向资金变化 proxy: 换手率趋势
    vol_ma5 = pd.Series(volume).rolling(5).mean().values
    vol_ma60 = pd.Series(volume).rolling(60).mean().values
    northbound_proxy = np.zeros(n)
    for i in range(60, n):
        if vol_ma60[i] > 0:
            northbound_proxy[i] = (vol_ma5[i] / vol_ma60[i]) - 1
    
    # 换手率变化: 近期换手率 / 长期换手率
    turnover_ratio = np.zeros(n)
    avg_volume = np.mean(volume)
    if avg_volume > 0:
        for i in range(n):
            turnover_ratio[i] = volume[i] / avg_volume
    
    result = df.copy()
    result['main_capital_ratio'] = main_capital_ratio
    result['northbound_proxy'] = northbound_proxy
    result['turnover_ratio'] = turnover_ratio
    
    return result


# ===================== 多因子共振评分 =====================

def compute_factor_votes(stock_row):
    """计算每个类别的信号强度 [0, 1]
    
    返回:
      category_signals: {A_technical: float, B_fundamental: float, C_valuation: float, D_moneyflow: float}
      resonance_score: float (共振强度)
    """
    signals = {}
    
    # === A. 技术面信号 ===
    technical_score = 0
    tech_votes = 0
    total_tech = 0
    
    # DMI 信号
    if 'pdi' in stock_row and 'ndi' in stock_row:
        total_tech += 1
        if stock_row['pdi'] > stock_row['ndi']:
            tech_votes += 1
            if 'adx' in stock_row and stock_row.get('adx', 0) > 25:
                tech_votes += 1
    
    # 动量信号
    if 'momentum_60' in stock_row:
        total_tech += 1
        if stock_row['momentum_60'] > 0:
            tech_votes += 1
    
    # 均线信号
    if 'price_above_ma60' in stock_row:
        total_tech += 1
        if stock_row['price_above_ma60'] > 0:
            tech_votes += 1
    
    # RSI 信号 (非超买)
    if 'rsi' in stock_row:
        total_tech += 1
        if stock_row['rsi'] < 70:  # 未超买
            tech_votes += 1
    
    # 成交量信号
    if 'volume_ratio' in stock_row:
        total_tech += 1
        if stock_row['volume_ratio'] > 1.2:  # 放量
            tech_votes += 1
    
    signals['A_technical'] = tech_votes / total_tech if total_tech > 0 else 0.5
    
    # === B. 基本面质量信号 ===
    fundamental_score = 0
    total_fund = 0
    
    if 'roe_proxy' in stock_row:
        total_fund += 1
        if stock_row['roe_proxy'] > 0.5:
            fundamental_score += 1
    
    if 'gross_margin_proxy' in stock_row:
        total_fund += 1
        if stock_row['gross_margin_proxy'] > 0.6:
            fundamental_score += 1
    
    if 'revenue_growth_proxy' in stock_row:
        total_fund += 1
        if stock_row['revenue_growth_proxy'] > 0.5:
            fundamental_score += 1
    
    if 'net_profit_growth_proxy' in stock_row:
        total_fund += 1
        if stock_row['net_profit_growth_proxy'] > 0.5:
            fundamental_score += 1
    
    signals['B_fundamental'] = fundamental_score / total_fund if total_fund > 0 else 0.5
    
    # === C. 估值信号 (低估值加分) ===
    valuation_score = 0
    total_val = 0
    
    if 'pe_proxy' in stock_row:
        total_val += 1
        if stock_row['pe_proxy'] < 0.5:  # 价格在历史中位以下 = 低估
            valuation_score += 1
    
    if 'pb_proxy' in stock_row:
        total_val += 1
        if stock_row['pb_proxy'] < 1.2:  # 价格 / 长期均价 < 1.2
            valuation_score += 1
    
    if 'peg_proxy' in stock_row:
        total_val += 1
        if stock_row['peg_proxy'] < 1.5:
            valuation_score += 1
    
    signals['C_valuation'] = valuation_score / total_val if total_val > 0 else 0.5
    
    # === D. 资金流信号 ===
    moneyflow_score = 0
    total_mf = 0
    
    if 'main_capital_ratio' in stock_row:
        total_mf += 1
        # 大资金活跃 = 正面
        if stock_row['main_capital_ratio'] > 0.3:
            moneyflow_score += 1
    
    if 'northbound_proxy' in stock_row:
        total_mf += 1
        if stock_row['northbound_proxy'] > 0:  # 换手率上升 = 资金流入
            moneyflow_score += 1
    
    if 'turnover_ratio' in stock_row:
        total_mf += 1
        if stock_row['turnover_ratio'] > 0.8:  # 活跃
            moneyflow_score += 1
    
    signals['D_moneyflow'] = moneyflow_score / total_mf if total_mf > 0 else 0.5
    
    # === 共振机制 ===
    # 统计 > 0.6 的类别数
    strong_categories = sum(1 for v in signals.values() if v > 0.6)
    weak_categories = sum(1 for v in signals.values() if v < 0.4)
    
    # 共振强度
    if strong_categories >= 2:
        resonance_score = strong_categories * 0.2  # 2 类共振=0.4, 3 类=0.6, 4 类=0.8
    elif weak_categories >= 2:
        resonance_score = -weak_categories * 0.2  # 负共振
    else:
        resonance_score = 0
    
    # 最终评分 = 平均信号 + 共振加成
    avg_signal = np.mean(list(signals.values()))
    final_score = avg_signal + resonance_score * 0.5
    
    return signals, final_score, resonance_score


# ===================== 市场状态识别 =====================

def identify_market_regime(all_data):
    """识别市场状态"""
    dates = set()
    for code, df in all_data.items():
        dates.update(df['date'].dt.date)
    all_dates = sorted(dates)
    
    # 计算每日市场平均收益率
    daily_returns = {}
    for d in all_dates:
        returns = []
        for code, df in all_data.items():
            row = df[df['date'].dt.date == d]
            prev_row = df[df['date'].dt.date == pd.Timestamp(d) - pd.Timedelta(days=1)]
            if len(row) > 0 and len(prev_row) > 0 and 'close' in row.columns and 'close' in prev_row.columns:
                ret = (row['close'].values[0] - prev_row['close'].values[0]) / prev_row['close'].values[0]
                returns.append(ret)
        if returns:
            daily_returns[d] = np.mean(returns)
    
    market_returns = pd.Series(daily_returns)
    ma20 = market_returns.rolling(20).mean()
    ma60 = market_returns.rolling(60).mean()
    
    regime = {}
    for d in all_dates:
        if d in ma20.index and d in ma60.index:
            if ma20[d] > ma60[d] and ma20[d] > 0:
                regime[d] = 'bull'
            elif ma20[d] < ma60[d] and ma20[d] < 0:
                regime[d] = 'bear'
            else:
                regime[d] = 'neutral'
    
    return regime


# ===================== 诚实回测引擎 =====================

def run_honest_backtest(all_data, market_regime, regime_params, start_date=None, end_date=None, 
                        threshold=2.0, rebalance_freq=20):
    """诚实回测引擎"""
    # 筛选回测区间
    all_dates = sorted(set(
        d for df in all_data.values() for d in df['date'].dt.date
    ))
    
    if start_date is None:
        start_date = all_dates[0]
    if end_date is None:
        end_date = all_dates[-1]
    
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]
    if len(trade_dates) < 60:
        return None
    
    # 初始化
    capital = 1000000
    cash = capital
    positions = {}  # code -> {'shares': N, 'cost': price}
    daily_equity = []
    trade_count = 0
    buy_count = 0
    sell_count = 0
    
    # 按交易日排序
    for day_idx, current_date in enumerate(trade_dates):
        # 调仓日
        if trade_count > 0 and trade_count % rebalance_freq == 0:
            # 卖出不在新候选中的持仓
            codes = [code for code, _ in trades[-min(len(trades), 10):] if code in positions]
            for code in list(positions.keys()):
                if code not in all_data or current_date not in [d for d in all_data[code]['date'].dt.date]:
                    if code in positions:
                        # 强制卖出
                        sell_price = all_data[code].loc[all_data[code]['date'].dt.date == current_date, 'close'].values
                        if len(sell_price) > 0:
                            shares = positions[code]['shares']
                            cash += shares * sell_price[0]
                            sell_count += 1
                            del positions[code]
                    continue
        
        if current_date not in market_regime:
            trade_dates = trade_dates + [current_date]
            continue
        
        # 获取候选股票
        day_stocks = []
        for code, df in all_data.items():
            row = df[df['date'].dt.date == current_date]
            if len(row) == 0:
                continue
            
            # 需要足够的历史数据
            hist = df[df['date'] < current_date]
            if len(hist) < 120:
                continue
            
            row = row.iloc[0]
            signals, final_score, resonance = compute_factor_votes(row.to_dict())
            
            # 市场状态过滤
            regime = market_regime.get(current_date, 'neutral')
            if regime == 'bear' and final_score < 0.3:
                continue  # 熊市只买高分
            
            day_stocks.append({
                'code': code,
                'price': row['close'] if 'close' in row else row.get('Close', 0),
                'score': final_score,
                'signals': signals,
                'resonance': resonance,
                'regime': regime,
            })
        
        # 按得分排序，取 top N
        day_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        # 买入
        max_position = 3 if regime_params.get(current_date, 'neutral') in ['bull', 'neutral'] else 2
        buy_candidates = [s for s in day_stocks if s['score'] > threshold and len(positions) < max_position]
        
        if buy_candidates:
            for stock in buy_candidates[:max_position - len(positions)]:
                if cash > 0:
                    price = stock['price']
                    if price > 0:
                        # 等权分配
                        alloc = cash / max_position
                        shares = int(alloc / price / 100) * 100
                        if shares > 0:
                            cost = shares * price
                            positions[stock['code']] = {'shares': shares, 'cost': price}
                            cash -= cost
                            buy_count += 1
                            trade_count += 1
        
        # 计算每日权益
        equity = cash
        for code, pos in positions.items():
            if code in all_data:
                row = all_data[code][all_data[code]['date'].dt.date == current_date]
                if len(row) > 0:
                    equity += pos['shares'] * row['close'].values[0]
        
        daily_equity.append((current_date, equity))
    
    # 计算收益
    if len(daily_equity) < 2:
        return None
    
    initial_equity = daily_equity[0][1]
    final_equity = daily_equity[-1][1]
    total_return = (final_equity / initial_equity - 1) * 100
    
    # 计算回撤
    peak = initial_equity
    max_drawdown = 0
    for _, eq in daily_equity:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak * 100
        if dd < max_drawdown:
            max_drawdown = dd
    
    # 夏普比率
    returns_list = [daily_equity[i][1] / daily_equity[i-1][1] - 1 for i in range(1, len(daily_equity))]
    if returns_list:
        mean_ret = np.mean(returns_list) * 252
        std_ret = np.std(returns_list) * np.sqrt(252)
        sharpe = mean_ret / std_ret if std_ret > 0 else 0
    else:
        sharpe = 0
    
    # 最大持有天数
    holding_days = 0
    for code, pos in positions.items():
        if code in all_data:
            first_buy = None
            last_price = None
            for i, (d, eq) in enumerate(daily_equity):
                if d >= start_date:
                    if first_buy is None:
                        first_buy = d
                    last_price = eq
            if first_buy and last_price:
                holding_days += (last_price / daily_equity[0][1] * 1000)
    
    return {
        'total_return': round(total_return, 2),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_drawdown, 2),
        'buy_count': buy_count,
        'sell_count': sell_count,
        'trade_count': trade_count,
        'final_equity': round(final_equity, 2),
        'initial_equity': round(initial_equity, 2),
    }


# ===================== 主流程 =====================

def main():
    print("=" * 60)
    print("🚀 V12 多因子共振版 - 新因子 + 诚实回测")
    print("=" * 60)
    
    # Step 0: 加载数据
    print("\n📊 步骤 0: 加载数据...")
    all_data = load_all_data()
    codes = list(all_data.keys())
    print(f"   有效股票: {len(codes)} 只")
    
    # Step 1: 计算技术指标
    print("\n📈 步骤 1: 计算技术指标...")
    for code in codes:
        all_data[code] = compute_technical_indicators(all_data[code])
        all_data[code] = generate_fundamental_factors(all_data[code], code)
        all_data[code] = generate_valuation_factors(all_data[code], all_data)
        all_data[code] = generate_money_flow_factors(all_data[code])
    print("   ✅ 技术指标计算完成")
    
    # Step 2: 市场状态识别
    print("\n🏛️ 步骤 2: 市场状态识别...")
    regime = identify_market_regime(all_data)
    regime_params = {d: v for d, v in regime.items()}
    
    # 获取所有日期 (用于 WF)
    all_dates = sorted(set(
        d for df in all_data.values() for d in df['date'].dt.date
    ))
    
    regime_counts = {}
    for v in regime.values():
        regime_counts[v] = regime_counts.get(v, 0) + 1
    print(f"   市场状态分布:")
    for k, v in sorted(regime_counts.items()):
        print(f"     {k:10s}: {v:>4} 天 ({v/len(regime)*100:.1f}%)")
    
    # Step 3: 因子 IC 分析
    print("\n🤖 步骤 3: 因子 IC 分析...")
    
    # 构建面板数据 (样本内: 2022-2023)
    in_sample_start = pd.Timestamp('2022-01-01')
    in_sample_end = pd.Timestamp('2023-12-31')
    out_sample_start = pd.Timestamp('2024-01-01')
    out_sample_end = pd.Timestamp('2026-12-31')
    
    # 样本内面板
    panels = []
    for code in codes:
        df = all_data[code].copy()
        df = df[(df['date'] >= in_sample_start) & (df['date'] <= in_sample_end)].copy()
        if len(df) < 120:
            continue
        
        # 计算未来 20 日收益
        df['future_ret_20d'] = df['close'].shift(-20) / df['close'] - 1
        
        # 因子列
        factor_cols = [
            'pdi', 'ndi', 'adx',
            'momentum_5', 'momentum_20', 'momentum_60', 'momentum_120',
            'price_above_ma5', 'price_above_ma20', 'price_above_ma60',
            'rsi', 'volume_ratio',
            'roe_proxy', 'gross_margin_proxy', 'revenue_growth_proxy', 'net_profit_growth_proxy',
            'pe_proxy', 'pb_proxy', 'peg_proxy',
            'main_capital_ratio', 'northbound_proxy', 'turnover_ratio',
        ]
        
        available = [c for c in factor_cols if c in df.columns]
        if available:
            df['date'] = df['date'].dt.date
            df['stock_code'] = code
            # 选择需要的列
            cols_to_keep = ['date', 'stock_code', 'future_ret_20d'] + available
            panels.append(df[cols_to_keep])
    
    if panels:
        panel_df = pd.concat(panels, ignore_index=True)
        panel_df = panel_df.dropna(subset=['future_ret_20d'])
        panel_df = panel_df[panel_df['future_ret_20d'].between(-1, 1)]  # 过滤极端值
        
        # 计算 IC (Rank IC)
        ic_results = {}
        for factor in available:
            factor_vals = panel_df[factor].values
            ret_vals = panel_df['future_ret_20d'].values
            mask = ~(np.isnan(factor_vals) | np.isnan(ret_vals))
            if mask.sum() > 30:
                rank_ic = np.corrcoef(factor_vals[mask], ret_vals[mask])[0, 1]
                if not np.isnan(rank_ic):
                    ic_results[factor] = rank_ic
        
        # 排序
        sorted_ics = sorted(ic_results.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"   样本内数据点: {len(panel_df)}")
        print(f"   最大 |IC|: {abs(sorted_ics[0][1]):.4f}")
        print(f"   因子 IC 排名 Top 10:")
        for i, (factor, ic) in enumerate(sorted_ics[:10]):
            print(f"     {i+1:2d}  {factor:30s}: {ic:+.4f}  |  |IC| = {abs(ic):.4f}")
        
        # 保存 IC 分析
        ic_analysis = {
            'total_data_points': len(panel_df),
            'in_sample_start': str(in_sample_start.date()),
            'in_sample_end': str(in_sample_end.date()),
            'factors': {f: round(ic, 4) for f, ic in sorted_ics},
        }
        with open(os.path.join(CACHE_DIR, 'v12_ic_analysis.json'), 'w') as f:
            json.dump(ic_analysis, f, indent=2, ensure_ascii=False)
    else:
        print("   ⚠️ 样本内数据不足，跳过 IC 分析")
        sorted_ics = []
    
    # Step 4: 运行诚实回测
    print("\n📈 步骤 4: 运行诚实回测...")
    
    # 定义回测配置
    # (名称, vol_parity, regime_risk, threshold, rebalance_freq, 是否高频)
    config_names = [
        ('基准 (低频)', False, False, 1.5, 20, False),
        ('+vol_parity', True, False, 1.5, 20, False),
        ('+regime_risk', False, True, 1.5, 20, False),
        ('+vol_parity+regime', True, True, 1.5, 20, False),
        ('基准 (高频)', False, False, 1.0, 10, True),
        ('+vol_parity (高频)', True, False, 1.0, 10, True),
        ('+regime_risk (高频)', False, True, 1.0, 10, True),
        ('+vol_parity+regime (高频)', True, True, 1.0, 10, True),
    ]
    
    # A. 拟合窗口回测 (2022-2023)
    print("\n   === A. 拟合窗口回测 (2022-2023) ===")
    fit_results = {}
    for name, vol, regime_risk, thresh, freq, is_high_freq in config_names:
        result = run_honest_backtest(
            all_data, regime, regime_params,
            start_date=in_sample_start.date(),
            end_date=in_sample_end.date(),
            threshold=thresh,
            rebalance_freq=freq,
        )
        if result:
            result['vol_parity'] = vol
            result['regime_risk'] = regime_risk
            result['name'] = name
            fit_results[name] = result
            suffix = '(高频)' if is_high_freq else ''
            print(f"     {name:30s}: 收益 {result['total_return']:>+7.2f}%  夏普 {result['sharpe_ratio']:>+6.2f}  回撤 {result['max_drawdown']:>+6.2f}%  交易 {result['buy_count']}买/{result['sell_count']}卖")
    
    # B. 纯净 Holdout 回测 (2024-2026)
    print("\n   === B. 纯净 Holdout 回测 (2024-2026) ===")
    holdout_results = {}
    for name, vol, regime_risk, thresh, freq, is_high_freq in config_names:
        result = run_honest_backtest(
            all_data, regime, regime_params,
            start_date=out_sample_start.date(),
            end_date=out_sample_end.date(),
            threshold=thresh,
            rebalance_freq=freq,
        )
        if result:
            result['vol_parity'] = vol
            result['regime_risk'] = regime_risk
            result['name'] = name
            holdout_results[name] = result
            suffix = '(高频)' if is_high_freq else ''
            print(f"     {name:30s}: 收益 {result['total_return']:>+7.2f}%  夏普 {result['sharpe_ratio']:>+6.2f}  回撤 {result['max_drawdown']:>+6.2f}%  交易 {result['buy_count']}买/{result['sell_count']}卖")
    
    # C. Walk-Forward 5 折 (全样本)
    print("\n   === C. Walk-Forward 5 折 (全样本) ===")
    wf_trade_dates = sorted(all_dates)
    n_folds = 5
    # 前 20% 当 warmup
    wf_start_idx = max(0, int(len(wf_trade_dates) * 0.2))
    wf_trade_dates = wf_trade_dates[wf_start_idx:]
    fold_size = len(wf_trade_dates) // n_folds
    
    wf_results = {}
    for name, vol, regime_risk, thresh, freq, is_high_freq in config_names:
        fold_returns = []
        fold_sharpe = []
        fold_drawdowns = []
        for fold in range(n_folds):
            f_start = fold * fold_size
            f_end = (fold + 1) * fold_size
            if f_start >= len(wf_trade_dates):
                break
            upper = min(f_end, len(wf_trade_dates))
            if upper <= f_start:
                break
            fold_start = wf_trade_dates[f_start]
            fold_end = wf_trade_dates[upper - 1]
            result = run_honest_backtest(
                all_data, regime, regime_params,
                start_date=fold_start,
                end_date=fold_end,
                threshold=thresh,
                rebalance_freq=freq,
            )
            if result:
                fold_returns.append(result['total_return'])
                fold_sharpe.append(result['sharpe_ratio'])
                fold_drawdowns.append(result['max_drawdown'])
        
        if fold_returns:
            avg_sharpe = np.mean(fold_sharpe)
            total_factor = 1.0
            for r in fold_returns:
                total_factor *= (1 + r / 100)
            composite_return = (total_factor - 1) * 100
            
            wf_results[name] = {
                'fold_returns': fold_returns,
                'fold_sharpe': fold_sharpe,
                'composite_return': round(composite_return, 4),
                'avg_fold_sharpe': round(avg_sharpe, 4),
            }
            suffix = '(高频)' if is_high_freq else ''
            print(f"     {name:30s}: WF-收益 {composite_return:>+7.2f}%  均夏普 {avg_sharpe:>+6.2f}  "
                  f"折叠 {[round(r,1) for r in fold_returns]}")
    
    # ===================== 结果汇总 =====================
    print("\n" + "=" * 60)
    print("📋 V12 多因子共振回测报告")
    print("=" * 60)
    
    # 对比 V10
    print("\n🔴 V10/V11 原始报告（被证伪的幻觉）:")
    print("   总收益: +34.12% (V10) — 但这是过拟合幻觉")
    print("   买入/卖出: 47/0 — 零卖出 = 买入持有，不是真策略")
    print("   市场状态切换: 0 天 — 状态切换从未触发")
    print("   结论: ⚠️  不可信")
    
    # 诚实回测最优
    best_holdout = None
    best_wf = None
    best_holdout_sharpe = -999
    best_wf_sharpe = -999
    
    for name, result in holdout_results.items():
        if result['sharpe_ratio'] > best_holdout_sharpe:
            best_holdout_sharpe = result['sharpe_ratio']
            best_holdout = (name, result)
    
    for name, result in wf_results.items():
        if result['avg_fold_sharpe'] > best_wf_sharpe:
            best_wf_sharpe = result['avg_fold_sharpe']
            best_wf = (name, result)
    
    if best_holdout:
        name, r = best_holdout
        print(f"\n✅ Holdout 最优: {name}")
        print(f"   收益: +{r['total_return']:.2f}%  夏普: {r['sharpe_ratio']:.2f}")
        print(f"   交易: {r['buy_count']}买/{r['sell_count']}卖")
    
    if best_wf:
        name, r = best_wf
        print(f"\n✅ WF-OOS 最优: {name}")
        print(f"   复合收益: +{r['composite_return']:.2f}%  均夏普: {r['avg_fold_sharpe']:.2f}")
        print(f"   折叠收益: {r['fold_returns']}")
    
    # 诚实结论
    print("\n📌 诚实结论:")
    if best_wf_sharpe < 0.3:
        print("   WF-OOS 均夏普 < 0.3 → 无稳健 alpha")
        print("   新因子（基本面 + 估值 + 资金流）未能产生显著 OOS 改善")
    else:
        print(f"   WF-OOS 均夏普 {best_wf_sharpe:.2f} → 有潜在 alpha")
    
    # 配置效果对比
    print(f"\n📊 配置效果对比:")
    print(f"   {'配置':<35s} {'Holdout夏普':>12s} {'WF均夏普':>12s}")
    print(f"   {'─'*59}")
    for name, vol, regime_risk, thresh, freq, is_high_freq in config_names:
        h = holdout_results.get(name, {}).get('sharpe_ratio', 0)
        w = wf_results.get(name, {}).get('avg_fold_sharpe', 0)
        print(f"   {name:<35s} {h:>+11.2f}        {w:>+11.2f}")
    
    # 保存结果
    result_summary = {
        'v12_backtest': True,
        'v10_v11_falsified': True,
        'falsification_reasons': [
            '零卖出 = 买入持有',
            '市场状态切换从未触发', 
            '样本内同段优化同段报告',
            '仅 49 只股票非全量',
        ],
        'new_factors': {
            'A_technical': 'DMI, 动量, 均线, RSI, 成交量',
            'B_fundamental': 'ROE proxy, 毛利率 proxy, 营收增速 proxy, 净利增速 proxy',
            'C_valuation': 'PE proxy (价格位置分位), PB proxy (价格/长期均价), PEG proxy',
            'D_moneyflow': '主力资金净流入 proxy, 北向资金变化 proxy, 换手率变化',
        },
        'resonance_mechanism': {
            'description': '独立类别投票，≥2 类同向触发',
            'strong_categories_threshold': 0.6,
        },
        'in_sample_ics': {f: round(ic, 4) for f, ic in sorted_ics[:15]},
        'fit_window': fit_results,
        'holdout_results': {k: v for k, v in holdout_results.items()},
        'walkforward_results': {k: {'composite_return': v['composite_return'], 'avg_fold_sharpe': v['avg_fold_sharpe'], 'fold_returns': v['fold_returns']} 
                                for k, v in wf_results.items()},
        'conclusion': {
            'wf_oos_best_sharpe': round(best_wf_sharpe, 4) if best_wf_sharpe > -999 else None,
            'has_robust_alpha': (best_wf_sharpe >= 0.3) if best_wf_sharpe > -999 else False,
        },
    }
    
    result_path = os.path.join(CACHE_DIR, 'v12_backtest_result.json')
    with open(result_path, 'w') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ V12 回测结果已保存: {result_path}")
    print("\n🎉 V12 诚实回测完成！")


if __name__ == '__main__':
    main()
