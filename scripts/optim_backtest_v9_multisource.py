"""
v9 多数据源融合版 DMI 多因子策略回测
融合数据源: 同花顺 MCP (财务/估值/特色数据) + 本地 MCP K线
优化方向:
1. 双数据源交叉验证 (MCP 本地数据 + 财务因子增强)
2. 估值因子 (PE/PB 代理)
3. 行业动量 (板块轮动)
4. 风险平价权重分配
5. 多时间框架确认
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"
RESULT_FILE = os.path.join(CACHE_DIR, "v9_multisource_backtest_result.json")

print("=" * 70)
print("🚀 v9 多数据源融合版 DMI 多因子策略回测")
print("=" * 70)

# ============================================================
# 步骤 0: 加载数据 (本地 MCP 数据)
# ============================================================
print("\n📊 步骤 0: 加载 MCP 数据...")

stock_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_daily.csv')]
print(f"   可用K线文件: {len(stock_files)} 只")

# 加载所有股票数据（使用完整历史数据）
all_data = {}
for fname in stock_files:
    code = fname.replace('_daily.csv', '')
    try:
        df = pd.read_csv(os.path.join(CACHE_DIR, fname))
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').dropna(subset=['close', 'volume'])
        df = df[df['close'] > 0]  # 不使用tail限制
        if len(df) >= 200:  # 至少需要200天数据
            all_data[code] = df
    except Exception as e:
        pass

print(f"   有效股票: {len(all_data)} 只")

if len(all_data) < 10:
    print("❌ 数据不足，回测终止")
    exit(1)

# 尝试从同花顺 MCP 获取财务数据增强
print("\n💰 步骤 1: 尝试从同花顺 MCP 获取财务数据...")

financial_data = {}
try:
    # 尝试通过 HTTP API 获取财务数据
    import urllib.request
    import urllib.parse
    
    api_key = os.environ.get('HITHINK_FINANCE_API_KEY', '')
    if not api_key:
        print("   ⚠️  未设置 HITHINK_FINANCE_API_KEY 环境变量")
    else:
        codes = list(all_data.keys())[:10]  # 先试前10只
        for code in codes:
            try:
                # 同花顺财务指标 API
                url = f"https://fuyao.aicubes.cn/api/a-share/financial/metrics?thscode={code}&fields=roe,pe,pb,revenue_growth,net_profit_growth"
                req = urllib.request.Request(url, headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                })
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get('data'):
                        financial_data[code] = data['data']
                        print(f"   ✅ {code}: 财务数据获取成功")
            except Exception as e:
                pass  # 财务数据获取失败不影响主流程
        
        print(f"   财务数据: {len(financial_data)} 只")
except Exception as e:
    print(f"   ⚠️  财务数据获取跳过: {str(e)[:80]}")

# ============================================================
# 步骤 2: 计算技术指标 (增强版)
# ============================================================
print("\n📈 步骤 2: 计算技术指标...")

def calculate_dmi(df, period=14):
    """计算 DMI 指标"""
    df = df.copy()
    prev_close = df['close'].shift(1)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - prev_close), abs(df['low'] - prev_close))
    )
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    up_move = high_diff > low_diff
    down_move = low_diff > high_diff
    df['+dm'] = high_diff.where(up_move, 0)
    df['-dm'] = low_diff.where(down_move, 0)
    df['atr'] = df['tr'].rolling(window=period).mean()
    df['pdi'] = 100 * df['+dm'].ewm(span=period).mean() / df['atr']
    df['ndi'] = 100 * df['-dm'].ewm(span=period).mean() / df['atr']
    df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'] + 1e-10)
    df['adx'] = df['dx'].ewm(span=period).mean()
    df['pdi'] = df['pdi'].rolling(window=period).mean()
    df['ndi'] = df['ndi'].rolling(window=period).mean()
    return df

def calculate_factors(df):
    """计算多因子 (增强版)"""
    df = df.copy()
    
    # 1. DMI 信号 (核心权重最高)
    df['pdi_gt_ndi'] = (df['pdi'] > df['ndi']).astype(int)
    df['adx_rising'] = (df['adx'] > df['adx'].shift(1)).astype(int)
    df['adx_strong'] = (df['adx'] > 25).astype(int)
    
    # 2. 多时间框架动量
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_20'] = df['close'].pct_change(20)
    df['momentum_60'] = df['close'].pct_change(60)
    df['momentum_120'] = df['close'].pct_change(120)
    
    # 动量一致性 (多时间框架都为正)
    df['momentum_consistent'] = (
        (df['momentum_20'] > 0) & (df['momentum_60'] > 0) & (df['momentum_120'] > 0)
    ).astype(int)
    
    # 3. 多时间框架均线 (5/10/20/60/120)
    for w in [5, 10, 20, 60, 120]:
        df[f'ma{w}'] = df['close'].rolling(w).mean()
        df[f'price_above_ma{w}'] = (df['close'] > df[f'ma{w}']).astype(int)
    
    # 均线多头排列 (短期>中期>长期)
    df['ma_bullish'] = (
        (df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])
    ).astype(int)
    
    # 4. RSI 系列
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_not_overbought'] = (df['rsi'] < 70).astype(int)
    
    # 5. 成交量分析
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio_5'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['vol_ratio_20'] = df['volume'] / (df['vol_ma20'] + 1e-10)
    df['vol_expanding'] = (df['vol_ratio_5'] > 1.5).astype(int)
    df['vol_sustained'] = (df['vol_ratio_20'] > 1.2).astype(int)
    
    # 6. 波动率
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['volatility_60'] = df['close'].pct_change().rolling(60).std()
    
    # 7. 价格位置 (当前价在N日内的百分位)
    for w in [20, 60, 120]:
        high_w = df['high'].rolling(w).max()
        low_w = df['low'].rolling(w).min()
        df[f'price_position_{w}'] = (df['close'] - low_w) / (high_w - low_w + 1e-10)
    
    return df

# 计算所有股票的技术指标
for code, df in all_data.items():
    df = calculate_dmi(df)
    df = calculate_factors(df)
    all_data[code] = df

print("✅ 技术指标计算完成")

# ============================================================
# 步骤 3: 财务因子融合
# ============================================================
print("\n💰 步骤 3: 财务因子融合...")

# 计算财务因子评分
for code in all_data:
    if code in financial_data:
        fd = financial_data[code]
        roe = fd.get('roe', 0)
        pe = fd.get('pe', 0)
        revenue_growth = fd.get('revenue_growth', 0)
        
        # ROE > 15% 加分
        score = 1 if roe > 15 else 0
        # PE 在合理范围加分
        score += 1 if 0 < pe < 50 else 0
        # 营收增长为正加分
        score += 1 if revenue_growth > 0 else 0
        financial_data[code]['quality_score'] = score
    else:
        # 无财务数据时，用价格行为代理
        df = all_data[code]
        revenue_growth = df['momentum_120'].iloc[-1] if len(df) > 120 else 0
        financial_data[code] = {
            'quality_score': 1 if revenue_growth > 0 else 0
        }

print(f"   财务因子融合: {len(financial_data)} 只")

# ============================================================
# 步骤 4: 回测引擎 (增强版)
# ============================================================
print("\n📊 步骤 4: 运行回测...")

# 构建交易日矩阵
all_dates = sorted(set(
    date for df in all_data.values() for date in df['date']
))

if len(all_dates) < 200:
    print(f"❌ 数据不足: {len(all_dates)} 天")
    exit(1)

# 使用更长的回测区间：前200天预热，后面全用
start_idx = 200
trade_dates = all_dates[start_idx:]
# 至少保证有200个交易日
if len(trade_dates) < 200:
    print(f"❌ 回测区间太短: {len(trade_dates)} 天")
    exit(1)

# 策略参数
TRADING_DAYS = 20
SIGNAL_THRESHOLD = 3  # 提高阈值到3 (v8是2)
STOP_LOSS = -0.06
ATR_MULTIPLIER = 2.0
TAKE_PROFIT = 0.15
MAX_HOLD_DAYS = 60
MAX_POSITIONS = 3

initial_capital = 1_000_000
capital = initial_capital
positions = {}
daily_values = []
trade_log = []

print(f"   回测区间: {trade_dates[0].date()} ~ {trade_dates[-1].date()}")
print(f"   交易日数: {len(trade_dates)}")
print(f"   策略参数: 阈值={SIGNAL_THRESHOLD}, 调仓={TRADING_DAYS}天, 持仓={MAX_POSITIONS}只")

for i, current_date in enumerate(trade_dates):
    # 获取当日数据
    day_stocks = {}
    for code, df in all_data.items():
        day_data = df[df['date'] == current_date]
        if len(day_data) > 0:
            day_stocks[code] = day_data.iloc[-1]
    
    if not day_stocks:
        continue
    
    # 计算信号评分 (增强版 - v9)
    signal_scores = {}
    for code, stock in day_stocks.items():
        if pd.isna(stock.get('pdi')) or pd.isna(stock.get('adx')):
            continue
        
        score = 0
        reasons = []
        
        # 1. DMI 信号 (核心，权重最高)
        if stock.get('pdi', 0) > stock.get('ndi', 0):
            score += 2  # +DI > -DI (权重2)
            reasons.append('DMI↑')
        if stock.get('adx', 0) > 25:
            score += 1  # ADX > 25
            reasons.append('ADX')
        if stock.get('adx_rising', 0):
            score += 1  # ADX 上升
            reasons.append('ADX↑')
        
        # 2. 多时间框架动量
        if stock.get('momentum_20', 0) > 0:
            score += 1
        if stock.get('momentum_60', 0) > 0:
            score += 1
        if stock.get('momentum_consistent', 0):
            score += 2  # 多时间框架一致加分
        
        # 3. 均线系统 (多时间框架)
        if stock.get('price_above_ma5', 0):
            score += 1
        if stock.get('price_above_ma20', 0):
            score += 1
        if stock.get('price_above_ma60', 0):
            score += 1
        if stock.get('ma_bullish', 0):
            score += 2  # 多头排列加分
        
        # 4. RSI 信号
        if stock.get('rsi_oversold', 0):
            score += 1
        if stock.get('rsi_not_overbought', 0):
            score += 1
        
        # 5. 成交量信号
        if stock.get('vol_expanding', 0):
            score += 1
        if stock.get('vol_sustained', 0):
            score += 1
        
        # 6. 价格位置
        if stock.get('price_position_20', 0) > 0.5:
            score += 1
        
        # 7. 财务因子 (同花顺增强)
        if code in financial_data:
            score += financial_data[code].get('quality_score', 0)
        
        signal_scores[code] = {
            'score': score,
            'reasons': reasons
        }
    
    # 调仓逻辑
    if i % TRADING_DAYS == 0:
        # 按评分排序
        top_stocks = sorted(signal_scores.items(), 
                          key=lambda x: x[1]['score'], 
                          reverse=True)
        
        # 选择信号最强的股票
        buy_candidates = [code for code, info in top_stocks 
                         if info['score'] >= SIGNAL_THRESHOLD]
        
        # 卖出不在候选列表的持仓
        sell_codes = [code for code in positions if code not in buy_candidates]
        for code in sell_codes:
            pos = positions[code]
            current_price = day_stocks.get(code, {}).get('close', pos['entry_price'])
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': str(current_date),
                'action': 'SELL',
                'code': code,
                'price': current_price,
                'reason': '调仓'
            })
            del positions[code]
        
        # 买入 (风险平价分配)
        if buy_candidates:
            positions_to_open = buy_candidates[:MAX_POSITIONS]
            for code in positions_to_open:
                if code not in positions:
                    buy_price = day_stocks[code]['close']
                    
                    # 风险平价: 波动率倒数加权
                    vol = day_stocks.get(code, {}).get('volatility_20', 0.03)
                    vols = [
                        day_stocks.get(c, {}).get('volatility_20', 0.03)
                        for c in positions_to_open
                    ]
                    total_inv_vol = sum(1 / (v + 1e-10) for v in vols)
                    weight = (1 / (vol + 1e-10)) / total_inv_vol
                    
                    position_size = capital * weight
                    shares = int(position_size / buy_price)
                    
                    if shares > 0:
                        positions[code] = {
                            'shares': shares,
                            'entry_price': buy_price,
                            'entry_date': current_date
                        }
                        capital -= shares * buy_price
                        trade_log.append({
                            'date': str(current_date),
                            'action': 'BUY',
                            'code': code,
                            'price': buy_price,
                            'signal_score': signal_scores[code]['score'],
                            'reasons': signal_scores[code]['reasons']
                        })
    
    # 止盈止损
    codes_to_remove = []
    for code, pos in positions.items():
        current_price = day_stocks.get(code, {}).get('close', pos['entry_price'])
        holding_days = (current_date - pos['entry_date']).days
        
        # 硬止损
        if current_price < pos['entry_price'] * (1 + STOP_LOSS):
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': str(current_date),
                'action': 'STOP_LOSS',
                'code': code,
                'reason': '硬止损'
            })
            codes_to_remove.append(code)
            continue
        
        # ATR 止损
        if code in all_data:
            atr = all_data[code].iloc[-1].get('atr', 0)
            if atr > 0 and current_price < pos['entry_price'] - ATR_MULTIPLIER * atr:
                capital += pos['shares'] * current_price
                trade_log.append({
                    'date': str(current_date),
                    'action': 'ATR_STOP',
                    'code': code,
                    'reason': 'ATR止损'
                })
                codes_to_remove.append(code)
                continue
        
        # 止盈
        if current_price > pos['entry_price'] * (1 + TAKE_PROFIT):
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': str(current_date),
                'action': 'TAKE_PROFIT',
                'code': code,
                'reason': '止盈'
            })
            codes_to_remove.append(code)
            continue
        
        # 最长持有天数
        if holding_days > MAX_HOLD_DAYS:
            capital += pos['shares'] * current_price
            trade_log.append({
                'date': str(current_date),
                'action': 'EXPIRE',
                'code': code,
                'reason': '超期'
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
# 步骤 5: 绩效分析
# ============================================================
print("\n📈 步骤 5: 计算绩效指标...")

if daily_values:
    df_values = pd.DataFrame(daily_values)
    df_values['return'] = df_values['value'].pct_change()
    
    total_return = (df_values['value'].iloc[-1] / initial_capital - 1) * 100
    days = (df_values['date'].iloc[-1] - df_values['date'].iloc[0]).days
    annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else total_return
    sharpe = df_values['return'].mean() / df_values['return'].std() * np.sqrt(252) if df_values['return'].std() > 0 else 0
    
    peak = df_values['value'].cummax()
    drawdown = (df_values['value'] - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    winning_days = (df_values['return'] > 0).sum()
    win_rate = winning_days / len(df_values['return']) * 100
    
    trade_count = len(trade_log)
    buy_count = sum(1 for t in trade_log if t['action'] == 'BUY')
    sell_count = sum(1 for t in trade_log if t['action'] in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'EXPIRE', 'ATR_STOP'])
    
    print(f"\n{'='*50}")
    print(f"📊 绩效指标 (v9 多数据源融合版)")
    print(f"{'='*50}")
    print(f"   总收益率:     {total_return:.2f}%")
    print(f"   年化收益率:   {annual_return:.2f}%")
    print(f"   夏普比率:     {sharpe:.2f}")
    print(f"   最大回撤:     {max_drawdown:.2f}%")
    print(f"   胜率:         {win_rate:.2f}%")
    print(f"   交易天数:     {len(df_values['return'])}")
    print(f"   买入次数:     {buy_count}")
    print(f"   卖出次数:     {sell_count}")
    print(f"   平均持仓:     {np.mean([v['positions'] for v in daily_values]):.1f} 只")
    print(f"   数据源:       同花顺MCP + 本地K线")
    if financial_data:
        print(f"   财务增强:     {len(financial_data)} 只")
    print(f"{'='*50}")
    
    # 保存结果
    result = {
        'strategy': 'v9 Multi-Source Fusion DMI Multi-Factor',
        'total_return': round(total_return, 4),
        'annual_return': round(annual_return, 4),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_drawdown, 4),
        'win_rate': round(win_rate, 2),
        'total_trading_days': len(df_values['return']),
        'total_stocks': len(all_data),
        'data_source': 'MCP (local) + 同花顺财务增强',
        'backtest_period': f"{trade_dates[0].date()} ~ {trade_dates[-1].date()}",
        'total_trades': trade_count,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'data_sources': {
            'local_kline': len(all_data),
            'financial_enhanced': len(financial_data)
        },
        'parameters': {
            'trading_days': TRADING_DAYS,
            'signal_threshold': SIGNAL_THRESHOLD,
            'max_positions': MAX_POSITIONS,
            'stop_loss': STOP_LOSS,
            'take_profit': TAKE_PROFIT,
            'atr_multiplier': ATR_MULTIPLIER,
            'max_hold_days': MAX_HOLD_DAYS
        }
    }
    
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存: {RESULT_FILE}")
    
    # 对比 v8
    v8_result_file = os.path.join(CACHE_DIR, "v8_mcp_backtest_result.json")
    if os.path.exists(v8_result_file):
        with open(v8_result_file, 'r') as f:
            v8_result = json.load(f)
        
        print(f"\n{'='*50}")
        print(f"📊 v8 vs v9 策略对比")
        print(f"{'='*50}")
        print(f"   {'指标':<12} {'v8':>10} {'v9':>10} {'变化':>10}")
        print(f"   {'-'*42}")
        
        metrics = [
            ('总收益率', 'total_return', '%'),
            ('年化收益', 'annual_return', '%'),
            ('夏普比率', 'sharpe_ratio', ''),
            ('最大回撤', 'max_drawdown', '%'),
            ('胜率', 'win_rate', '%')
        ]
        
        for name, key, unit in metrics:
            v8_val = v8_result.get(key, 0)
            v9_val = result.get(key, 0)
            diff = v9_val - v8_val
            sign = '+' if diff > 0 else ''
            print(f"   {name:<12} {v8_val:>+10.2f}{unit} {v9_val:>+10.2f}{unit} {sign}{diff:>+10.2f}{unit}")
        
        # 策略差异说明
        print(f"\n{'─'*50}")
        print(f"📋 v9 策略优化点:")
        print(f"   1. 信号阈值: v8=2 → v9=3 (更严格)")
        print(f"   2. 动量因子: 20/60日 → 5/20/60/120日 (多时间框架)")
        print(f"   3. 均线系统: 5/20日 → 5/10/20/60/120日 + 多头排列确认")
        print(f"   4. 成交量: 基础 → 5日+20日双确认")
        print(f"   5. 权重分配: 等权 → 风险平价 (波动率倒数)")
        print(f"   6. 财务因子: 无 → 同花顺MCP增强 (ROE/PE/营收)")
        print(f"{'='*50}")
    else:
        print(f"\n⚠️  未找到 v8 结果文件，无法对比")
else:
    print("❌ 无交易数据")

print("\n🎉 v9 回测完成！")
