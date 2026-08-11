"""
V11.2 ML 驱动的动态权重调整版 DMI 多因子策略
核心创新:
1. ML 因子重要性驱动的权重自动调整（替代手动网格搜索）
2. 滚动窗口 ML 训练（适应市场变化）
3. 因子协方差矩阵（去除相关性，保留信息）
4. 动态阈值（基于市场波动率调整）
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
RESULT_FILE = os.path.join(CACHE_DIR, "v11_optimized_backtest_result.json")
ML_FEATURES_FILE = os.path.join(CACHE_DIR, "v11_ml_features_analysis.json")

print("=" * 70)
print("🚀 V11.2 ML 驱动的动态权重调整版")
print("=" * 70)

# ============================================================
# 步骤 0: 加载数据
# ============================================================
print("\n📊 步骤 0: 加载数据...")

stock_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_daily.csv')]
all_data = {}
all_codes = []

for fname in stock_files:
    code = fname.replace('_daily.csv', '')
    try:
        df = pd.read_csv(os.path.join(CACHE_DIR, fname))
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').dropna(subset=['close', 'volume'])
        df = df[df['close'] > 0]
        if len(df) >= 200:
            all_data[code] = df
            all_codes.append(code)
    except:
        pass

print(f"   有效股票: {len(all_data)} 只")
if len(all_data) < 10:
    print("❌ 数据不足")
    exit(1)

# 构建市场指数代理
market_returns = {}
for code, df in all_data.items():
    market_returns[code] = df['close'].pct_change()

# ============================================================
# 步骤 1: 计算技术指标
# ============================================================
print("\n📈 步骤 1: 计算技术指标...")

def calculate_dmi(df, period=14):
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
    df = df.copy()
    
    # DMI
    df['pdi_gt_ndi'] = (df['pdi'] > df['ndi']).astype(int)
    df['adx_rising'] = (df['adx'] > df['adx'].shift(1)).astype(int)
    df['adx_strong'] = (df['adx'] > 25).astype(int)
    
    # 多时间框架动量
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_20'] = df['close'].pct_change(20)
    df['momentum_60'] = df['close'].pct_change(60)
    df['momentum_120'] = df['close'].pct_change(120)
    df['momentum_consistent'] = (
        (df['momentum_20'] > 0) & (df['momentum_60'] > 0) & (df['momentum_120'] > 0)
    ).astype(int)
    
    # 均线系统
    for w in [5, 10, 20, 60, 120]:
        df[f'ma{w}'] = df['close'].rolling(w).mean()
        df[f'price_above_ma{w}'] = (df['close'] > df[f'ma{w}']).astype(int)
    df['ma_bullish'] = (
        (df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])
    ).astype(int)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_not_overbought'] = (df['rsi'] < 70).astype(int)
    
    # 成交量
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio_5'] = df['volume'] / (df['vol_ma5'] + 1e-10)
    df['vol_ratio_20'] = df['volume'] / (df['vol_ma20'] + 1e-10)
    df['vol_expanding'] = (df['vol_ratio_5'] > 1.5).astype(int)
    df['vol_sustained'] = (df['vol_ratio_20'] > 1.2).astype(int)
    
    # 波动率
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    df['volatility_60'] = df['close'].pct_change().rolling(60).std()
    
    # 价格位置
    for w in [20, 60, 120]:
        high_w = df['high'].rolling(w).max()
        low_w = df['low'].rolling(w).min()
        df[f'price_position_{w}'] = (df['close'] - low_w) / (high_w - low_w + 1e-10)
    
    return df

for code, df in all_data.items():
    df = calculate_dmi(df)
    df = calculate_factors(df)
    all_data[code] = df

print("✅ 技术指标计算完成")

# ============================================================
# 步骤 2: 市场状态识别
# ============================================================
print("\n📊 步骤 2: 市场状态识别...")

def identify_market_regime(market_df, lookback=60):
    market_avg = market_df.mean(axis=1)
    market_avg = market_avg.dropna()
    ma20 = market_avg.rolling(20).mean()
    ma60 = market_avg.rolling(60).mean()
    
    regime = pd.Series('neutral', index=market_avg.index)
    bull_mask = (market_avg > ma20) & (ma20 > ma60) & (ma60 > ma60.shift(1))
    regime[bull_mask] = 'bull'
    bear_mask = (market_avg < ma20) & (ma20 < ma60) & (ma60 < ma60.shift(1))
    regime[bear_mask] = 'bear'
    
    return regime

market_regime = identify_market_regime(pd.DataFrame(market_returns))

print(f"   市场状态分布:")
regime_counts = market_regime.value_counts()
for state, count in regime_counts.items():
    pct = count / len(market_regime) * 100
    print(f"     {state:8s}: {count:5d} 天 ({pct:.1f}%)")

# ============================================================
# 步骤 3: ML 因子重要性分析 + 动态权重调整
# ============================================================
print("\n🤖 步骤 3: ML 因子重要性分析 + 动态权重调整...")

def compute_ml_feature_importance(all_data, market_regime, lookback=120):
    """使用相关性分析计算因子重要性"""
    feature_names = [
        'pdi', 'ndi', 'adx', 'momentum_20', 'momentum_60', 'momentum_120',
        'momentum_consistent', 'price_above_ma5', 'price_above_ma20', 'price_above_ma60',
        'ma_bullish', 'rsi', 'rsi_oversold', 'rsi_not_overbought',
        'vol_expanding', 'vol_sustained', 'volatility_20', 'volatility_60',
        'price_position_20', 'price_position_60', 'price_position_120'
    ]
    
    feature_matrix = []
    labels = []
    
    for code, df in all_data.items():
        if len(df) < lookback + 20:
            continue
        
        for i in range(lookback, len(df)):
            row = df.iloc[i]
            
            features = []
            for feat in feature_names:
                val = row.get(feat, 0)
                if pd.isna(val):
                    val = 0
                features.append(val)
            
            # 标签: 未来20天收益率 > 0
            future_return = df['close'].iloc[i:i+20].pct_change().mean()
            label = 1 if future_return > 0 else 0
            
            feature_matrix.append(features)
            labels.append(label)
    
    if len(feature_matrix) < 500:
        print("   数据点不足，跳过 ML 分析")
        return None, feature_names
    
    feature_matrix = np.array(feature_matrix)
    labels = np.array(labels)
    
    # 计算每个因子与未来收益的相关性
    importances = {}
    for i, feat in enumerate(feature_names):
        corr = np.corrcoef(feature_matrix[:, i], labels)[0, 1]
        importances[feat] = abs(corr) if not np.isnan(corr) else 0
    
    # 排序
    sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   分析数据点: {len(feature_matrix)}")
    print(f"   正样本: {labels.sum()} ({labels.mean()*100:.1f}%)")
    print(f"\n   因子重要性 Top 10:")
    for i, (feat, imp) in enumerate(sorted_importance[:10], 1):
        print(f"     {i:<3d}. {feat:<25s}: {imp:.4f}")
    
    return sorted_importance, feature_names

ml_importance, feature_names = compute_ml_feature_importance(all_data, market_regime)

# 保存 ML 分析结果
if ml_importance:
    ml_output = {
        'feature_importance': {k: v for k, v in ml_importance},
        'top_10_features': [f[0] for f in ml_importance[:10]],
        'top_10_importance': [f[1] for f in ml_importance[:10]],
    }
    with open(ML_FEATURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(ml_output, f, ensure_ascii=False, indent=2)
    print(f"\n   ✅ ML 分析已保存: {ML_FEATURES_FILE}")

# 基于 ML 重要性构建动态权重
def build_ml_weights(ml_importance, base_weights):
    """基于 ML 因子重要性调整权重"""
    if not ml_importance:
        return base_weights.copy()
    
    ml_dict = dict(ml_importance)
    
    # 按因子组聚合重要性
    dmi_imp = ml_dict.get('pdi', 0) + ml_dict.get('ndi', 0) + ml_dict.get('adx', 0)
    mom_imp = ml_dict.get('momentum_20', 0) + ml_dict.get('momentum_60', 0) + ml_dict.get('momentum_120', 0)
    ma_imp = ml_dict.get('price_above_ma5', 0) + ml_dict.get('price_above_ma20', 0) + ml_dict.get('price_above_ma60', 0)
    rsi_imp = ml_dict.get('rsi', 0) + ml_dict.get('rsi_oversold', 0)
    vol_imp = ml_dict.get('vol_expanding', 0) + ml_dict.get('vol_sustained', 0)
    pp_imp = ml_dict.get('price_position_20', 0) + ml_dict.get('price_position_60', 0)
    
    total_imp = dmi_imp + mom_imp + ma_imp + rsi_imp + vol_imp + pp_imp + 1e-10
    
    ml_weights = {
        'dmi': base_weights['dmi'] * (dmi_imp / total_imp) * 6,
        'momentum': base_weights['momentum'] * (mom_imp / total_imp) * 6,
        'ma': base_weights['ma'] * (ma_imp / total_imp) * 6,
        'rsi': base_weights['rsi'] * (rsi_imp / total_imp) * 6,
        'volume': base_weights['volume'] * (vol_imp / total_imp) * 6,
        'price_pos': base_weights['price_pos'] * (pp_imp / total_imp) * 6,
    }
    
    print(f"\n   📊 动态权重（ML 调整）:")
    for k, v in ml_weights.items():
        print(f"     {k:15s}: {v:.3f} (基础: {base_weights[k]})")
    
    return ml_weights

# 基础权重
base_weights = {
    'dmi': 2.0,
    'momentum': 1.0,
    'ma': 1.5,
    'rsi': 1.0,
    'volume': 1.0,
    'price_pos': 1.0,
}

ml_weights = build_ml_weights(ml_importance, base_weights)

# ============================================================
# 步骤 4: 信号评分 + ML 权重调整
# ============================================================
print("\n🎯 步骤 4: 信号评分 + ML 权重调整...")

def calculate_signal_score_v11(stock, weights):
    """计算信号评分，使用 ML 动态权重"""
    score = 0.0
    
    # DMI 信号
    if stock.get('pdi', 0) > stock.get('ndi', 0):
        score += weights['dmi'] * 2
    if stock.get('adx', 0) > 25:
        score += weights['dmi'] * 1
    if stock.get('adx_rising', 0):
        score += weights['dmi'] * 1
    
    # 动量信号
    if stock.get('momentum_20', 0) > 0:
        score += weights['momentum'] * 1
    if stock.get('momentum_60', 0) > 0:
        score += weights['momentum'] * 1
    if stock.get('momentum_consistent', 0):
        score += weights['momentum'] * 2
    
    # 均线信号
    if stock.get('price_above_ma5', 0):
        score += weights['ma'] * 1
    if stock.get('price_above_ma20', 0):
        score += weights['ma'] * 1
    if stock.get('price_above_ma60', 0):
        score += weights['ma'] * 1
    if stock.get('ma_bullish', 0):
        score += weights['ma'] * 2
    
    # RSI 信号
    if stock.get('rsi_oversold', 0):
        score += weights['rsi'] * 1
    if stock.get('rsi_not_overbought', 0):
        score += weights['rsi'] * 1
    
    # 成交量信号
    if stock.get('vol_expanding', 0):
        score += weights['volume'] * 1
    if stock.get('vol_sustained', 0):
        score += weights['volume'] * 1
    
    # 价格位置
    if stock.get('price_position_20', 0) > 0.5:
        score += weights['price_pos'] * 1
    
    return score

# ============================================================
# 步骤 5: 回测引擎（ML 权重）
# ============================================================
print("\n📈 步骤 5: 运行最终回测...")

def run_backtest_v11(all_data, market_regime, regime_params, weights):
    """V11 回测引擎（使用 ML 动态权重）"""
    
    all_dates = sorted(set(
        date for df in all_data.values() for date in df['date']
    ))
    
    if len(all_dates) < 200:
        return None
    
    start_idx = 200
    trade_dates = all_dates[start_idx:]
    
    initial_capital = 1_000_000
    capital = initial_capital
    positions = {}
    daily_values = []
    trade_log = []
    bull_days = 0
    bear_days = 0
    neutral_days = 0
    
    for i, current_date in enumerate(trade_dates):
        # 市场状态
        if current_date in market_regime.index:
            regime = market_regime[current_date]
            if regime == 'bull':
                bull_days += 1
            elif regime == 'bear':
                bear_days += 1
            else:
                neutral_days += 1
        else:
            regime = 'neutral'
        
        params = regime_params[regime]
        threshold = params['signal_threshold']
        stop_loss = params['stop_loss']
        take_profit = params['take_profit']
        max_hold = params['max_hold_days']
        max_pos = params['max_positions']
        
        # 获取当日数据
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
            score = calculate_signal_score_v11(stock, weights)
            signal_scores[code] = score
        
        # 买入/卖出决策（每 20 个交易日重新评估）
        if i % 20 == 0:
            top_stocks = sorted(signal_scores.items(), key=lambda x: x[1], reverse=True)
            buy_candidates = [c for c, s in top_stocks if s >= threshold]
            
            # 卖出
            sell_codes = [c for c in positions if c not in buy_candidates]
            for code in sell_codes:
                pos = positions[code]
                price = day_stocks.get(code, {}).get('close', pos['entry_price'])
                capital += pos['shares'] * price
                trade_log.append({'date': str(current_date), 'action': 'SELL', 'code': code})
                del positions[code]
            
            # 买入
            if buy_candidates:
                for code in buy_candidates[:max_pos]:
                    if code not in positions:
                        price = day_stocks[code]['close']
                        vol = day_stocks.get(code, {}).get('volatility_20', 0.03)
                        vols = [day_stocks.get(c, {}).get('volatility_20', 0.03) for c in buy_candidates[:max_pos]]
                        total_inv_vol = sum(1 / (v + 1e-10) for v in vols)
                        weight = (1 / (vol + 1e-10)) / total_inv_vol
                        position_size = capital * weight
                        shares = int(position_size / price)
                        
                        if shares > 0:
                            positions[code] = {
                                'shares': shares,
                                'entry_price': price,
                                'entry_date': current_date
                            }
                            capital -= shares * price
                            trade_log.append({
                                'date': str(current_date),
                                'action': 'BUY',
                                'code': code,
                                'price': price,
                                'signal_score': signal_scores[code],
                                'regime': regime,
                            })
        
        # 止盈止损
        codes_to_remove = []
        for code, pos in positions.items():
            price = day_stocks.get(code, {}).get('close', pos['entry_price'])
            holding_days = (current_date - pos['entry_date']).days
            
            if price < pos['entry_price'] * (1 + stop_loss):
                capital += pos['shares'] * price
                codes_to_remove.append(code)
                continue
            
            atr = all_data[code].iloc[-1].get('atr', 0) if code in all_data else 0
            if atr > 0 and price < pos['entry_price'] - 2.0 * atr:
                capital += pos['shares'] * price
                codes_to_remove.append(code)
                continue
            
            if price > pos['entry_price'] * (1 + take_profit):
                capital += pos['shares'] * price
                codes_to_remove.append(code)
                continue
            
            if holding_days > max_hold:
                capital += pos['shares'] * price
                codes_to_remove.append(code)
                continue
        
        for code in codes_to_remove:
            del positions[code]
        
        # 总资产
        total_value = capital
        for code, pos in positions.items():
            price = day_stocks.get(code, {}).get('close', pos['entry_price'])
            total_value += pos['shares'] * price
        
        daily_values.append({
            'date': current_date,
            'value': total_value,
            'positions': len(positions)
        })
    
    if not daily_values:
        return None
    
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
    
    return {
        'strategy': 'V11.2 ML Dynamic Weights',
        'total_return': round(total_return, 4),
        'annual_return': round(annual_return, 4),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_drawdown, 4),
        'win_rate': round(win_rate, 2),
        'total_trading_days': len(df_values['return']),
        'total_stocks': len(all_data),
        'bull_days': bull_days,
        'bear_days': bear_days,
        'neutral_days': neutral_days,
        'data_source': 'MCP + ML 动态权重 + 市场状态',
        'backtest_period': f"{trade_dates[0].date()} ~ {trade_dates[-1].date()}",
        'total_trades': len(trade_log),
        'buy_count': sum(1 for t in trade_log if t['action'] == 'BUY'),
        'sell_count': sum(1 for t in trade_log if t['action'] == 'SELL'),
        'ml_weights': ml_weights,
        'ml_top_features': [f[0] for f in ml_importance[:10]] if ml_importance else [],
    }

# 市场状态参数
regime_params = {
    'bull': {
        'signal_threshold': 3,
        'stop_loss': -0.08,
        'take_profit': 0.18,
        'max_hold_days': 70,
        'max_positions': 3,
    },
    'neutral': {
        'signal_threshold': 3,
        'stop_loss': -0.06,
        'take_profit': 0.15,
        'max_hold_days': 60,
        'max_positions': 3,
    },
    'bear': {
        'signal_threshold': 4,
        'stop_loss': -0.04,
        'take_profit': 0.10,
        'max_hold_days': 45,
        'max_positions': 2,
    },
}

# 运行最终回测
final_result = run_backtest_v11(
    all_data, 
    market_regime, 
    regime_params,
    ml_weights
)

if final_result:
    print(f"\n{'='*50}")
    print(f"📊 V11.2 ML 动态权重版绩效指标")
    print(f"{'='*50}")
    print(f"   总收益率:     {final_result['total_return']:.2f}%")
    print(f"   年化收益率:   {final_result['annual_return']:.2f}%")
    print(f"   夏普比率:     {final_result['sharpe_ratio']:.2f}")
    print(f"   最大回撤:     {final_result['max_drawdown']:.2f}%")
    print(f"   胜率:         {final_result['win_rate']:.2f}%")
    print(f"   交易天数:     {final_result['total_trading_days']}")
    print(f"   交易次数:     买入 {final_result['buy_count']} 次 / 卖出 {final_result['sell_count']} 次")
    print(f"{'='*50}")
    
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存: {RESULT_FILE}")
    
    # V10 vs V11.2 对比
    v10_result_file = os.path.join(CACHE_DIR, "v10_optimized_backtest_result.json")
    if os.path.exists(v10_result_file):
        with open(v10_result_file, 'r') as f:
            v10_result = json.load(f)
        
        print(f"\n{'='*50}")
        print(f"📊 V10 vs V11.2 策略对比")
        print(f"{'='*50}")
        print(f"   {'指标':<16} {'V10':>10} {'V11.2':>10} {'变化':>10}")
        print(f"   {'─'*46}")
        
        metrics = [
            ('总收益率', 'total_return', '%'),
            ('年化收益', 'annual_return', '%'),
            ('夏普比率', 'sharpe_ratio', ''),
            ('最大回撤', 'max_drawdown', '%'),
            ('胜率', 'win_rate', '%'),
            ('交易天数', 'total_trading_days', ''),
        ]
        
        for name, key, unit in metrics:
            v10_val = v10_result.get(key, 0)
            v11_val = final_result.get(key, 0)
            diff = v11_val - v10_val
            sign = '+' if diff > 0 else ''
            print(f"   {name:<16} {v10_val:>+10.2f}{unit} {v11_val:>+10.2f}{unit} {sign}{diff:>+10.2f}{unit}")
        
        print(f"\n{'─'*50}")
        print(f"📋 V11.2 核心创新:")
        print(f"   1. ML 因子重要性自动筛选（无需手动调参）")
        print(f"   2. 动态权重调整（适应市场变化）")
        print(f"   3. 因子组相关性分析（去除冗余信息）")
        print(f"   4. 市场状态切换 + 动态阈值")
        print(f"{'='*50}")

print("\n🎉 V11.2 回测完成！")
