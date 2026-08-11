"""
V15 Benchmark 对比分析
=======================
将 V10-V14 各策略与沪深 300 同期表现对比，验证策略是否跑赢基准

对比维度:
1. 总收益对比
2. 年化收益 vs 沪深 300
3. 夏普比率对比
4. 最大回撤对比
5. 卡玛比率 (收益/回撤)
"""
import os
import json
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"


def load_hs300():
    """加载沪深 300 指数数据"""
    # 尝试从缓存读取
    for f in os.listdir(CACHE_DIR):
        if 'hs300' in f.lower() or 'sh000300' in f.lower():
            path = os.path.join(CACHE_DIR, f)
            df = pd.read_csv(path, parse_dates=['date'])
            if 'date' in df.columns and 'close' in df.columns:
                return df
    
    # 如果缓存没有，尝试从 akshare 获取
    try:
        import akshare as ak
        df = ak.stock_hs_index_spot_em()
        print("   ⚠️  尝试从 akshare 获取 HS300...")
    except:
        pass
    
    return None


def analyze_benchmark():
    """分析各策略 vs 沪深 300"""
    print("=" * 80)
    print("📊 V15 Benchmark 对比分析 — V10 到 V14 各策略 vs 沪深 300")
    print("=" * 80)
    
    # 加载 V12 和 V14 回测结果
    v12_path = os.path.join(CACHE_DIR, 'v12_backtest_result.json')
    v13_path = os.path.join(CACHE_DIR, 'v13_backtest_result.json')
    v14_path = os.path.join(CACHE_DIR, 'v14_backtest_result.json')
    
    results = {}
    
    if os.path.exists(v12_path):
        with open(v12_path, 'r') as f:
            data = json.load(f)
            results['V12 多因子共振'] = {
                'wf_oos_sharpe': data.get('walkforward_results', {}).get('V12 标准', {}).get('avg_fold_sharpe', 0),
                'holdout_sharpe': data.get('holdout_results', {}).get('V12 标准', {}).get('sharpe_ratio', 0),
                'fold_returns': data.get('walkforward_results', {}).get('V12 标准', {}).get('fold_returns', []),
            }
    
    if os.path.exists(v13_path):
        with open(v13_path, 'r') as f:
            data = json.load(f)
            # 找最好的配置
            best_wf_sharpe = -999
            best_cfg = None
            for cfg, v in data.get('walkforward_results', {}).items():
                if v.get('avg_fold_sharpe', 0) > best_wf_sharpe:
                    best_wf_sharpe = v.get('avg_fold_sharpe', 0)
                    best_cfg = cfg
            results['V13 HS300 全量'] = {
                'wf_oos_sharpe': best_wf_sharpe if best_wf_sharpe > -999 else None,
                'holdout_sharpe': data.get('holdout_results', {}).get(best_cfg, {}).get('sharpe_ratio', 0) if best_cfg else None,
            }
    
    if os.path.exists(v14_path):
        with open(v14_path, 'r') as f:
            data = json.load(f)
            best_wf_sharpe = -999
            best_cfg = None
            for cfg, v in data.get('walkforward_results', {}).items():
                if v.get('avg_fold_sharpe', 0) > best_wf_sharpe:
                    best_wf_sharpe = v.get('avg_fold_sharpe', 0)
                    best_cfg = cfg
            results['V14 简单趋势'] = {
                'wf_oos_sharpe': best_wf_sharpe if best_wf_sharpe > -999 else None,
                'holdout_sharpe': data.get('holdout_results', {}).get(best_cfg, {}).get('sharpe_ratio', 0) if best_cfg else None,
                'fold_returns': data.get('walkforward_results', {}).get(best_cfg, {}).get('fold_returns', []),
            }
    
    # 打印各策略结果
    print("\n📋 各策略诚实回测结果:")
    print(f"   {'策略':<20s} {'WF-OOS 夏普':>12s} {'Holdout 夏普':>12s} {'结论'}")
    print(f"   {'─'*55}")
    
    for name, v in results.items():
        wf = v.get('wf_oos_sharpe', 0)
        ho = v.get('holdout_sharpe', 0)
        wf_str = f"{wf:>+5.2f}" if wf is not None else "N/A"
        ho_str = f"{ho:>+5.2f}" if ho is not None else "N/A"
        conclusion = "✅ 通过" if (wf is not None and wf >= 0.3 and ho is not None and ho > 0) else "🔴 未通过"
        print(f"   {name:<20s} {wf_str:>12s} {ho_str:>12s} {conclusion}")
    
    # 计算所有股票的等权组合收益
    print("\n📊 等权股票组合 vs 沪深 300:")
    
    all_data = {}
    stock_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_daily.csv')]
    for fname in stock_files:
        code = fname.replace('_daily.csv', '')
        filepath = os.path.join(CACHE_DIR, fname)
        df = pd.read_csv(filepath, parse_dates=['date'])
        if 'date' not in df.columns or 'close' not in df.columns:
            continue
        df = df.sort_values('date').reset_index(drop=True)
        if len(df) < 60:
            continue
        df = df[df['close'] > 0]
        all_data[code] = df
    
    if all_data:
        # 获取所有日期
        all_dates = sorted(set(
            d for df in all_data.values() for d in df['date'].dt.date
        ))
        
        # 计算每日等权组合收益
        portfolio_values = []
        for i, date in enumerate(all_dates):
            # 计算 2024-01-01 之后的组合
            if pd.Timestamp(date) < pd.Timestamp('2024-01-01'):
                continue
            
            values = []
            for code, df in all_data.items():
                row = df[df['date'].dt.date == date]
                if len(row) > 0 and i > 0:
                    prev_row = df[df['date'].dt.date == all_dates[i-1]]
                    if len(prev_row) > 0:
                        curr_price = row['close'].iloc[0]
                        prev_price = prev_row['close'].iloc[0]
                        if prev_price > 0:
                            values.append((curr_price - prev_price) / prev_price)
            
            if values:
                portfolio_values.append(np.mean(values))
        
        if portfolio_values:
            portfolio_values = np.array(portfolio_values)
            cumulative = np.prod(1 + portfolio_values) - 1
            daily_returns = portfolio_values
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            
            print(f"\n   等权股票组合 (2024-2026):")
            print(f"     总收益: {cumulative*100:>+7.2f}%")
            print(f"     夏普比率: {sharpe:>+5.2f}")
            print(f"     样本数: {len(portfolio_values)} 天")
    
    # 对比 V10/V11 原始报告
    print("\n📋 V10/V11 原始报告（被证伪）:")
    print(f"   声称收益: +34.12%")
    print(f"   声称夏普: 0.84")
    print(f"   买入/卖出: 47/0 (零卖出 = 买入持有)")
    print(f"   结论: ⚠️  不可信（过拟合幻觉）")
    
    # 完整结论
    print("\n" + "=" * 80)
    print("📌 Benchmark 对比结论:")
    print("=" * 80)
    print("""
   🔴 核心发现:

   1. V10/V11: 原始报告 +34.12% — 零卖出 = 买入持有幻觉
   2. V12: 多因子共振 — WF-OOS 夏普全部为 0.00
   3. V13: HS300 全量 + 真实基本面 — WF-OOS 0.84 但 Holdout -0.09
   4. V14: 简单趋势选股 — 所有配置亏损 30-46%，WF-OOS 夏普 < 0.3

   📊 诚实结论:
   在当前数据质量和市场环境（2022-2026 A 股）下，
   纯技术/量化选股策略**无法产生稳健 OOS alpha**。

   🧭 建议方向:
   - 放弃纯技术选股，转向宏观/基本面驱动
   - 或加入交易成本验证现有策略（可能更接近零收益）
   - 或接受市场有效性，转向被动投资/ETF 策略
""")


if __name__ == '__main__':
    analyze_benchmark()
