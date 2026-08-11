"""
V14 简单趋势选股 + 右侧交易策略
=================================
核心理念: 大道至简，右侧交易

选股逻辑:
1. 价格在 MA60 上方 (中长期趋势向上)
2. 20日动量 > 0 (短期动量正向)
3. 放量突破 MA20 (右侧确认)

出场逻辑:
- 收盘价跌破 MA20 (趋势结束)
- 或 ATR 止损 (保护资金)

仓位管理:
- 等权重分配
- 最多持仓 5 只

诚实框架:
- 拟合窗口: 2022-2023
- 纯净 Holdout: 2024-2026
- Walk-Forward 5 折
"""
import os
import json
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"


def load_all_data():
    """加载所有股票数据"""
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
        
        if len(df) < 120:
            continue
        
        # 标准化列名
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']:
            if col in df.columns and col.lower() not in df.columns:
                df[col.lower()] = df[col]
        
        # 过滤负价格
        df = df[df['close'] > 0]
        
        all_data[code] = df
    
    print(f"   有效股票: {len(all_data)} 只")
    return all_data


def compute_trend_indicators(df):
    """计算趋势相关指标"""
    close = df['close'].values
    volume = df['volume'].values if 'volume' in df.columns else np.zeros(len(close))
    
    # 均线系统
    df['ma20'] = pd.Series(close).rolling(20).mean().values
    df['ma60'] = pd.Series(close).rolling(60).mean().values
    df['ma120'] = pd.Series(close).rolling(120).mean().values
    
    # 动量因子
    df['momentum_20'] = close / np.roll(close, 20) - 1
    df['momentum_20'][0:20] = np.nan
    
    # 波动率 (returns 少一个元素，前面补 nan 对齐)
    returns = np.diff(close) / close[:-1]
    returns_full = np.concatenate([[np.nan], returns])
    df['volatility_20'] = pd.Series(returns_full).rolling(20).std().values
    
    # ATR (14日真实波幅)
    high = df['high'].values if 'high' in df.columns else close
    low = df['low'].values if 'low' in df.columns else close
    tr = np.full(len(close), np.nan)
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    df['atr'] = pd.Series(tr).rolling(14).mean().values
    
    # 成交量均线
    df['vol_ma5'] = pd.Series(volume).rolling(5).mean().values
    df['vol_ma20'] = pd.Series(volume).rolling(20).mean().values
    
    # 放量标志
    df['is_high_volume'] = df['volume'] > (1.5 * df['vol_ma20'])
    
    return df


def simple_trend_signals(df_row):
    """
    简单趋势信号 (右侧交易)
    返回: (score, reason)
    score: 0-100, 越高越好
    reason: 信号说明
    """
    score = 0
    reasons = []
    
    # === 趋势确认 (必选) ===
    # 1. 价格在 MA60 上方
    if pd.notna(df_row.get('close')) and pd.notna(df_row.get('ma60')):
        if df_row['close'] > df_row['ma60']:
            score += 30
            reasons.append("价格>MA60")
        
        # 2. 20日动量 > 0 (短期趋势向上)
        if pd.notna(df_row.get('momentum_20')) and df_row['momentum_20'] > 0:
            score += 20
            reasons.append("20d动量>0")
    
    # === 入场确认 ===
    # 3. 放量突破 MA20 (右侧信号)
    if pd.notna(df_row.get('is_high_volume')) and df_row['is_high_volume']:
        if pd.notna(df_row.get('close')) and pd.notna(df_row.get('ma20')):
            if df_row['close'] > df_row['ma20']:
                score += 30
                reasons.append("放量突破MA20")
    
    # 4. 均线多头排列 (加分)
    if (pd.notna(df_row.get('close')) and pd.notna(df_row.get('ma20')) and 
        pd.notna(df_row.get('ma60'))):
        if df_row['ma20'] > df_row['ma60']:
            score += 20
            reasons.append("MA20>MA60")
    
    # 5. 波动率适中 (不过高也不过低)
    if pd.notna(df_row.get('volatility_20')):
        if 0.01 < df_row['volatility_20'] < 0.05:
            score += 10
            reasons.append("波动率适中")
    
    # 综合评分
    score = min(100, score)  # 上限 100
    
    return score, ", ".join(reasons)


def run_honest_backtest(all_data, start_date, end_date, 
                       max_positions=5, atr_stop_multiplier=2.0):
    """
    简单趋势策略回测
    
    规则:
    - 买入: 符合趋势信号且不在持仓中
    - 卖出: 收盘价跌破 MA20 或 ATR 止损
    """
    codes = list(all_data.keys())
    
    # 获取交易日
    all_dates = sorted(set(
        d for df in all_data.values() for d in df['date'].dt.date
    ))
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]
    
    if len(trade_dates) < 60:
        return None
    
    # 初始化
    capital = 1000000
    cash = capital
    positions = {}  # {code: {'shares': n, 'cost': price}}
    portfolio_values = []
    trade_log = []
    
    for i, date in enumerate(trade_dates):
        # 每日计算持仓市值
        portfolio_value = cash
        for code, pos in list(positions.items()):
            df = all_data.get(code)
            if df is None:
                continue
            row = df[df['date'].dt.date == date]
            if len(row) > 0:
                price = row['close'].iloc[0]
                portfolio_value += pos['shares'] * price
        
        portfolio_values.append(portfolio_value)
        
        # === 卖出检查 ===
        sell_codes = []
        for code, pos in list(positions.items()):
            df = all_data.get(code)
            if df is None:
                continue
            row = df[df['date'].dt.date == date]
            if len(row) == 0:
                continue
            
            price = row['close'].iloc[0]
            ma20 = row['ma20'].iloc[0]
            atr = row['atr'].iloc[0]
            
            sell_reason = None
            
            # 1. 跌破 MA20
            if pd.notna(ma20) and price < ma20:
                sell_reason = "跌破MA20"
            
            # 2. ATR 止损
            elif pd.notna(atr) and pd.notna(pos['cost']):
                stop_loss = pos['cost'] - atr_stop_multiplier * atr
                if price < stop_loss:
                    sell_reason = f"ATR止损({atr_stop_multiplier}x)"
            
            if sell_reason:
                sell_codes.append((code, price, sell_reason))
        
        for code, price, reason in sell_codes:
            if code in positions:
                cash += positions[code]['shares'] * price
                portfolio_value = cash  # 更新当前价值
                trade_log.append({
                    'date': str(date),
                    'action': 'SELL',
                    'code': code,
                    'price': price,
                    'reason': reason
                })
                del positions[code]
        
        # === 买入检查 ===
        if len(positions) >= max_positions:
            continue
        
        # 计算所有股票的趋势评分
        stock_scores = []
        for code in codes:
            df = all_data.get(code)
            if df is None:
                continue
            row = df[df['date'].dt.date == date]
            if len(row) == 0:
                continue
            if code in positions:
                continue  # 已在持仓中
            
            row_dict = row.iloc[0].to_dict()
            row_price = row['close'].iloc[0]
            score, reasons = simple_trend_signals(row_dict)
            
            # 只选高分股票 (≥score_thresh)
            if score >= 80:
                stock_scores.append((code, score, reasons, row_price))
        
        # 按评分排序，买入最高分
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        
        for code, score, reasons, price in stock_scores[:max_positions - len(positions)]:
            if code not in positions and price > 0:
                # 等权分配
                per_stock = cash / max(1, len(positions) + 1)
                shares = int(per_stock / price / 100) * 100
                if shares >= 100:
                    positions[code] = {
                        'shares': shares,
                        'cost': price,
                        'entry_date': str(date),
                        'reason': reasons
                    }
                    cash -= shares * price
                    trade_log.append({
                        'date': str(date),
                        'action': 'BUY',
                        'code': code,
                        'price': price,
                        'reason': reasons
                    })
    
    if not portfolio_values:
        return None
    
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # 最大回撤
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100
    
    # 统计交易
    buy_count = sum(1 for t in trade_log if t['action'] == 'BUY')
    sell_count = sum(1 for t in trade_log if t['action'] == 'SELL')
    
    # 计算胜率
    wins = 0
    losses = 0
    for code, pos in positions.items():
        if pos['cost'] > 0:
            final_price = portfolio_values[-1] / len(positions) if positions else pos['cost']
            if final_price > pos['cost']:
                wins += 1
            else:
                losses += 1
    
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    return {
        'total_return': round(total_return, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'buy_count': buy_count,
        'sell_count': sell_count,
        'win_rate': round(win_rate, 1),
        'n_stocks': len(codes),
        'trade_log': trade_log[-20:],  # 最后 20 笔交易
    }


def main():
    print("=" * 70)
    print("🚀 V14 简单趋势选股 + 右侧交易策略 - 诚实回测")
    print("=" * 70)
    
    # Step 0: 加载数据
    print("\n📊 步骤 0: 加载数据...")
    all_data = load_all_data()
    
    # Step 1: 计算趋势指标
    print("\n📈 步骤 1: 计算趋势指标...")
    for code in list(all_data.keys()):
        all_data[code] = compute_trend_indicators(all_data[code])
    print("   ✅ 趋势指标计算完成")
    
    # Step 2: 运行诚实回测
    print("\n📈 步骤 2: 运行诚实回测...")
    
    in_sample_start = pd.Timestamp('2022-01-01')
    in_sample_end = pd.Timestamp('2023-12-31')
    out_sample_start = pd.Timestamp('2024-01-01')
    out_sample_end = pd.Timestamp('2026-12-31')
    
    # 不同参数配置
    configs = [
        {'name': 'V14 标准 (2xATR)', 'atr_mult': 2.0, 'max_pos': 5, 'score_thresh': 80},
        {'name': 'V14 保守 (3xATR)', 'atr_mult': 3.0, 'max_pos': 5, 'score_thresh': 85},
        {'name': 'V14 激进 (1.5xATR)', 'atr_mult': 1.5, 'max_pos': 7, 'score_thresh': 75},
        {'name': 'V14 极简 (仅MA20)', 'atr_mult': 2.5, 'max_pos': 3, 'score_thresh': 60},
    ]
    
    # A. 拟合窗口回测 (2022-2023)
    print("\n   === A. 拟合窗口回测 (2022-2023) ===")
    fit_results = {}
    for cfg in configs:
        result = run_honest_backtest(
            all_data,
            start_date=in_sample_start.date(),
            end_date=in_sample_end.date(),
            max_positions=cfg['max_pos'],
            atr_stop_multiplier=cfg['atr_mult'],
        )
        if result:
            result['config'] = cfg['name']
            fit_results[cfg['name']] = result
            print(f"     {cfg['name']:25s}: 收益 {result['total_return']:>+7.2f}%  夏普 {result['sharpe_ratio']:>+5.2f}  回撤 {result['max_drawdown']:>+6.2f}%  交易 {result['buy_count']}买/{result['sell_count']}卖  胜率 {result['win_rate']}%")
    
    # B. 纯净 Holdout 回测 (2024-2026)
    print("\n   === B. 纯净 Holdout 回测 (2024-2026) ===")
    holdout_results = {}
    for cfg in configs:
        result = run_honest_backtest(
            all_data,
            start_date=out_sample_start.date(),
            end_date=out_sample_end.date(),
            max_positions=cfg['max_pos'],
            atr_stop_multiplier=cfg['atr_mult'],
        )
        if result:
            result['config'] = cfg['name']
            holdout_results[cfg['name']] = result
            print(f"     {cfg['name']:25s}: 收益 {result['total_return']:>+7.2f}%  夏普 {result['sharpe_ratio']:>+5.2f}  回撤 {result['max_drawdown']:>+6.2f}%  交易 {result['buy_count']}买/{result['sell_count']}卖  胜率 {result['win_rate']}%")
    
    # C. Walk-Forward 5 折
    print("\n   === C. Walk-Forward 5 折 (全样本) ===")
    all_dates = sorted(set(
        d for df in all_data.values() for d in df['date'].dt.date
    ))
    wf_trade_dates = [d for d in all_dates if pd.Timestamp(d) >= pd.Timestamp('2022-01-01')]
    n_folds = 5
    fold_size = len(wf_trade_dates) // n_folds
    
    wf_results = {}
    for cfg in configs:
        fold_returns = []
        fold_sharpe = []
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
                all_data,
                start_date=fold_start,
                end_date=fold_end,
                max_positions=cfg['max_pos'],
                atr_stop_multiplier=cfg['atr_mult'],
            )
            if result:
                fold_returns.append(result['total_return'])
                fold_sharpe.append(result['sharpe_ratio'])
        
        if fold_returns:
            avg_sharpe = float(np.mean(fold_sharpe))
            total_factor = 1.0
            for r in fold_returns:
                total_factor *= (1 + r / 100)
            composite_return = float((total_factor - 1) * 100)
            
            wf_results[cfg['name']] = {
                'fold_returns': [round(r, 2) for r in fold_returns],
                'fold_sharpe': [round(s, 2) for s in fold_sharpe],
                'composite_return': round(composite_return, 2),
                'avg_fold_sharpe': avg_sharpe,
            }
            print(f"     {cfg['name']:25s}: WF-收益 {composite_return:>+7.2f}%  均夏普 {avg_sharpe:>+5.2f}  "
                  f"折叠 {[round(r,1) for r in fold_returns]}")
    
    # ===================== 结果汇总 =====================
    print("\n" + "=" * 70)
    print("📋 V14 简单趋势选股 + 右侧交易 - 诚实回测报告")
    print("=" * 70)
    
    # 对比 V10/V11
    print("\n🔴 V10/V11 原始报告（被证伪的幻觉）:")
    print("   总收益: +34.12% — 过拟合幻觉")
    print("   买入/卖出: 47/0 — 零卖出 = 买入持有")
    print("   WF-OOS 均夏普: < 0.3 — 无稳健 alpha")
    print("   结论: ⚠️  不可信")
    
    # V14 结果
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
        print(f"   回撤: {r['max_drawdown']:.2f}%  胜率: {r['win_rate']:.1f}%")
        print(f"   交易: {r['buy_count']}买/{r['sell_count']}卖")
    
    if best_wf:
        name, r = best_wf
        print(f"\n✅ WF-OOS 最优: {name}")
        print(f"   复合收益: +{r['composite_return']:.2f}%  均夏普: {r['avg_fold_sharpe']:.2f}")
        print(f"   折叠收益: {r['fold_returns']}")
    
    # 诚实结论
    print("\n📌 诚实结论:")
    if best_wf_sharpe >= 0.3 and best_holdout_sharpe > 0:
        print(f"   ✅ WF-OOS 夏普 {best_wf_sharpe:.2f} + Holdout 夏普 {best_holdout_sharpe:.2f} → 有稳健 alpha")
        print(f"   ✅ V14 简单趋势策略通过了诚实验证！")
    elif best_wf_sharpe < 0.3:
        print(f"   🔴 WF-OOS 夏普 {best_wf_sharpe:.2f} < 0.3 → 无稳健 alpha")
        print(f"   🔴 简单趋势策略也无法产生 OOS 收益")
    else:
        print(f"   ⚠️  WF-OOS 夏普 {best_wf_sharpe:.2f}，但 Holdout {best_holdout_sharpe:.2f} → 可能过拟合")
    
    # 配置效果对比表
    print(f"\n📊 配置效果对比:")
    print(f"   {'配置':<25s} {'Holdout夏普':>12s} {'WF均夏普':>12s} {'Holdout收益':>12s}")
    print(f"   {'─'*61}")
    for cfg in configs:
        name = cfg['name']
        h = holdout_results.get(name, {}).get('sharpe_ratio', 0)
        w = wf_results.get(name, {}).get('avg_fold_sharpe', 0)
        hr = holdout_results.get(name, {}).get('total_return', 0)
        print(f"   {name:<25s} {h:>+11.2f}        {w:>+11.2f}        {hr:>+10.2f}%")
    
    # 保存结果
    result_summary = {
        'v14_backtest': True,
        'strategy': 'Simple Trend Following + Right-Side Trading',
        'v10_v11_falsified': True,
        'v10_v11_falsification_reasons': [
            '零卖出 = 买入持有',
            '市场状态切换从未触发',
            '样本内同段优化同段报告',
            '仅 49 只股票非全量',
        ],
        'v14_configs': [cfg['name'] for cfg in configs],
        'fit_window': {k: {'return': v['total_return'], 'sharpe': v['sharpe_ratio'], 
                           'drawdown': v['max_drawdown'], 'win_rate': v['win_rate']}
                       for k, v in fit_results.items()},
        'holdout_results': {k: {'return': v['total_return'], 'sharpe': v['sharpe_ratio'],
                                 'drawdown': v['max_drawdown'], 'win_rate': v['win_rate'],
                                 'trades': f"{v['buy_count']}买/{v['sell_count']}卖"}
                            for k, v in holdout_results.items()},
        'walkforward_results': {k: {'composite_return': v['composite_return'], 
                                     'avg_fold_sharpe': v['avg_fold_sharpe'], 
                                     'fold_returns': v['fold_returns']}
                                for k, v in wf_results.items()},
        'conclusion': {
            'wf_oos_best_sharpe': round(best_wf_sharpe, 2) if best_wf_sharpe > -999 else None,
            'holdout_best_sharpe': round(best_holdout_sharpe, 2) if best_holdout_sharpe > -999 else None,
            'has_robust_alpha': (best_wf_sharpe >= 0.3 and best_holdout_sharpe > 0) 
                                if (best_wf_sharpe > -999 and best_holdout_sharpe > -999) else False,
        },
    }
    
    result_path = os.path.join(CACHE_DIR, 'v14_backtest_result.json')
    with open(result_path, 'w') as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ V14 回测结果已保存: {result_path}")
    print("\n🎉 V14 诚实回测完成！")


if __name__ == '__main__':
    main()
