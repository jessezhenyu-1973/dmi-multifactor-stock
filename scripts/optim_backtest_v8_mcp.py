"""
v8 MCP 数据驱动版 DMI 多因子策略回测
基于通达信pytdx + 腾讯自选股API获取的数据
优化方向:
1. 使用 MCP 获取的最新数据
2. 移除冗余因子 (close_ma60_ratio)
3. 参数优化 (信号≥2 + 25天调仓)
4. 增加ATR动态止损
5. 数据质量校验
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 配置
CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"
HS300_FILE = os.path.join(CACHE_DIR, "hs300_constituents.csv")
RESULT_FILE = os.path.join(CACHE_DIR, "v8_mcp_backtest_result.json")

# 策略参数
TRADING_DAYS = 20  # 调仓频率
SIGNAL_THRESHOLD = 2  # 信号共振阈值
STOP_LOSS = -0.06  # 硬止损 -6%
ATR_MULTIPLIER = 2.0  # ATR 止损倍数
TAKE_PROFIT = 0.15  # 止盈 +15%
MAX_HOLD_DAYS = 60  # 最长持有天数

print("=" * 70)
print("🚀 v8 MCP 数据驱动版 DMI 多因子策略回测")
print("=" * 70)

# ============================================================
# 步骤 1: 加载数据
# ============================================================
print("\n📊 步骤 1: 加载数据...")
stock_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_daily.csv')]
print(f"   股票数量: {len(stock_files)} 只")

# 加载所有股票数据
all_data = {}
for fname in stock_files:
    code = fname.replace('_daily.csv', '')
    df = pd.read_csv(os.path.join(CACHE_DIR, fname))
    
    # 数据清洗
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').dropna(subset=['close', 'volume'])
    df = df[df['close'] > 0]  # 过滤无效数据
    
    if len(df) >= 100:  # 至少需要100天数据
        all_data[code] = df

print(f"   有效股票: {len(all_data)} 只")
if all_data:
    sample_code = list(all_data.keys())[0]
    df_sample = all_data[sample_code]
    print(f"   数据日期: {df_sample['date'].min().date()} ~ {df_sample['date'].max().date()}")
    print(f"   数据条数: {len(df_sample)} 条")

if len(all_data) < 10:
    print("❌ 数据不足，回测终止")
    exit(1)

# ============================================================
# 步骤 2: 计算技术指标
# ============================================================
print("\n📈 步骤 2: 计算技术指标...")

def calculate_dmi(df, period=14):
    """计算 DMI 指标"""
    df = df.copy()
    
    # 真实波幅 TR = max(H-L, |H-Cprev|, |L-Cprev|)
    prev_close = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - prev_close),
            abs(df['low'] - prev_close)
        )
    )
    
    # +DM 和 -DM
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    up_move = high_diff > low_diff
    down_move = low_diff > high_diff
    df['+dm'] = high_diff.where(up_move, 0)
    df['-dm'] = low_diff.where(down_move, 0)
    
    # ATR
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # +DI 和 -DI
    df['pdi'] = 100 * df['+dm'].ewm(span=period).mean() / df['atr']
    df['ndi'] = 100 * df['-dm'].ewm(span=period).mean() / df['atr']
    df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'] + 1e-10)
    
    # ADX = DX 的 EMA
    df['adx'] = df['dx'].ewm(span=period).mean()
    
    # 平滑 +DI 和 -DI
    df['pdi'] = df['pdi'].rolling(window=period).mean()
    df['ndi'] = df['ndi'].rolling(window=period).mean()
    
    return df

def calculate_factors(df):
    """计算多因子"""
    df = df.copy()
    
    # 1. DMI 信号
    df['pdi_gt_ndi'] = (df['pdi'] > df['ndi']).astype(int)
    df['adx_rising'] = (df['adx'] > df['adx'].shift(1)).astype(int)
    
    # 2. 动量因子 (移除冗余的 close_ma60_ratio)
    df['momentum_20'] = df['close'].pct_change(20)
    df['momentum_60'] = df['close'].pct_change(60)
    
    # 3. 均线系统
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    df['price_above_ma5'] = (df['close'] > df['ma5']).astype(int)
    df['price_above_ma20'] = (df['close'] > df['ma20']).astype(int)
    
    # 4. RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    
    # 5. 成交量变化
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['vol_expanding'] = (df['vol_ratio'] > 1.5).astype(int)
    
    return df

# 计算所有股票的技术指标
for code, df in all_data.items():
    df = calculate_dmi(df)
    df = calculate_factors(df)
    all_data[code] = df

print("✅ 技术指标计算完成")

# ============================================================
# 步骤 3: 回测引擎 (基于交易日矩阵)
# ============================================================
print("\n📊 步骤 3: 运行回测...")

# 构建交易日矩阵：找出所有有数据的日期
all_dates = sorted(set(
    date 
    for df in all_data.values() 
    for date in df['date']
))

# 过滤日期范围（至少需要200天）
if len(all_dates) < 200:
    print(f"❌ 数据不足: {len(all_dates)} 天")
    exit(1)

start_idx = 200  # 跳过前200天作为预热期
end_idx = len(all_dates)
trade_dates = all_dates[start_idx:end_idx]

print(f"   回测区间: {trade_dates[0].date()} ~ {trade_dates[-1].date()}")
print(f"   交易日数: {len(trade_dates)}")

# 初始化
initial_capital = 1_000_000
capital = initial_capital
positions = {}  # {code: {'shares': n, 'entry_price': p, 'entry_date': d}}
daily_values = []  # {date: total_value}
trade_log = []  # 交易记录

# 按交易日循环
for i, current_date in enumerate(trade_dates):
    # 获取当日所有股票数据
    day_stocks = {}
    for code, df in all_data.items():
        day_data = df[df['date'] == current_date]
        if len(day_data) > 0:
            day_stocks[code] = day_data.iloc[-1]
    
    if not day_stocks:
        continue
    
    # 计算信号
    signal_scores = {}
    for code, stock in day_stocks.items():
        if pd.isna(stock.get('pdi')) or pd.isna(stock.get('adx')):
            continue
        
        score = 0
        
        # 1. DMI 信号
        if stock.get('pdi', 0) > stock.get('ndi', 0):
            score += 1  # +DI > -DI
        if stock.get('adx', 0) > 25:
            score += 1  # ADX > 25
        if stock.get('adx_rising', 0):
            score += 1  # ADX 上升
        
        # 2. 动量信号
        if stock.get('momentum_20', 0) > 0:
            score += 1  # 20日动量为正
        if stock.get('momentum_60', 0) > 0:
            score += 1  # 60日动量为正
        
        # 3. 均线信号
        if stock.get('price_above_ma5', 0):
            score += 1
        if stock.get('price_above_ma20', 0):
            score += 1
        
        # 4. RSI 信号
        if stock.get('rsi_oversold', 0):
            score += 1  # 超卖反弹
        
        # 5. 成交量信号
        if stock.get('vol_expanding', 0):
            score += 1
        
        signal_scores[code] = score
    
    # 调仓逻辑
    if i % TRADING_DAYS == 0:  # 调仓日
        # 选择信号最强的股票
        top_stocks = sorted(signal_scores.items(), key=lambda x: x[1], reverse=True)
        buy_candidates = [code for code, score in top_stocks if score >= SIGNAL_THRESHOLD]
        
        # 卖出不在候选列表的持仓
        sell_codes = [code for code in positions if code not in buy_candidates]
        for code in sell_codes:
            pos = positions[code]
            current_price = day_stocks[code]['close']
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': current_date,
                'action': 'SELL',
                'code': code,
                'price': current_price,
                'shares': pos['shares'],
                'reason': '调仓'
            })
            del positions[code]
        
        # 买入新股票
        if buy_candidates:
            for code in buy_candidates[:3]:  # 最多持有3只
                if code not in positions:
                    buy_price = day_stocks[code]['close']
                    position_size = capital / 3
                    shares = int(position_size / buy_price)
                    
                    if shares > 0:
                        positions[code] = {
                            'shares': shares,
                            'entry_price': buy_price,
                            'entry_date': current_date
                        }
                        capital -= shares * buy_price
                        trade_log.append({
                            'date': current_date,
                            'action': 'BUY',
                            'code': code,
                            'price': buy_price,
                            'shares': shares,
                            'signal_score': signal_scores[code]
                        })
    
    # 检查止盈止损
    codes_to_remove = []
    for code, pos in positions.items():
        current_price = day_stocks.get(code, {}).get('close', pos['entry_price'])
        holding_days = (current_date - pos['entry_date']).days
        
        # 硬止损
        if current_price < pos['entry_price'] * (1 + STOP_LOSS):
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': current_date,
                'action': 'STOP_LOSS',
                'code': code,
                'price': current_price,
                'shares': pos['shares']
            })
            codes_to_remove.append(code)
            continue
        
        # ATR 止损
        atr = all_data[code].iloc[-1].get('atr', 0) if code in all_data else 0
        if atr > 0 and current_price < pos['entry_price'] - ATR_MULTIPLIER * atr:
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': current_date,
                'action': 'ATR_STOP',
                'code': code,
                'price': current_price,
                'shares': pos['shares']
            })
            codes_to_remove.append(code)
            continue
        
        # 止盈
        if current_price > pos['entry_price'] * (1 + TAKE_PROFIT):
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': current_date,
                'action': 'TAKE_PROFIT',
                'code': code,
                'price': current_price,
                'shares': pos['shares']
            })
            codes_to_remove.append(code)
            continue
        
        # 最长持有天数
        if holding_days > MAX_HOLD_DAYS:
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': current_date,
                'action': 'EXPIRE',
                'code': code,
                'price': current_price,
                'shares': pos['shares']
            })
            codes_to_remove.append(code)
            continue
    
    for code in codes_to_remove:
        del positions[code]
    
    # 计算当日总资产
    total_value = capital
    for code, pos in positions.items():
        current_price = day_stocks.get(code, {}).get('close', pos['entry_price'])
        total_value += pos['shares'] * current_price
    
    daily_values.append({
        'date': current_date,
        'value': total_value,
        'positions': len(positions)
    })

# ============================================================
# 步骤 4: 计算绩效指标
# ============================================================
print("\n📈 步骤 4: 计算绩效指标...")

if daily_values:
    df_values = pd.DataFrame(daily_values)
    df_values['return'] = df_values['value'].pct_change()
    
    # 总收益率
    total_return = (df_values['value'].iloc[-1] / initial_capital - 1) * 100
    
    # 年化收益率
    days = (df_values['date'].iloc[-1] - df_values['date'].iloc[0]).days
    if days > 0:
        annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100
    else:
        annual_return = total_return
    
    # 夏普比率
    sharpe = df_values['return'].mean() / df_values['return'].std() * np.sqrt(252) if df_values['return'].std() > 0 else 0
    
    # 最大回撤
    peak = df_values['value'].cummax()
    drawdown = (df_values['value'] - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # 胜率 (盈利交易日占比)
    winning_days = (df_values['return'] > 0).sum()
    total_days = len(df_values['return'])
    win_rate = winning_days / total_days * 100 if total_days > 0 else 0
    
    # 交易次数
    trade_count = len(trade_log)
    
    print(f"\n📊 绩效指标:")
    print(f"   总收益率: {total_return:.2f}%")
    print(f"   年化收益率: {annual_return:.2f}%")
    print(f"   夏普比率: {sharpe:.2f}")
    print(f"   最大回撤: {max_drawdown:.2f}%")
    print(f"   胜率: {win_rate:.2f}%")
    print(f"   交易天数: {total_days}")
    print(f"   总交易次数: {trade_count}")
    print(f"   平均持仓: {np.mean([v['positions'] for v in daily_values]):.1f} 只")
    
    # 保存结果
    result = {
        'strategy': 'v8 MCP Data-Driven DMI Multi-Factor',
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trading_days': total_days,
        'total_stocks': len(all_data),
        'trading_frequency': TRADING_DAYS,
        'signal_threshold': SIGNAL_THRESHOLD,
        'stop_loss': STOP_LOSS,
        'take_profit': TAKE_PROFIT,
        'data_source': 'MCP (pytdx + Tencent API)',
        'backtest_period': f"{trade_dates[0].date()} ~ {trade_dates[-1].date()}",
        'total_trades': trade_count
    }
    
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存: {RESULT_FILE}")
else:
    print("❌ 无交易数据")

print("\n" + "=" * 70)
print("🎉 回测完成！")
print("=" * 70)
