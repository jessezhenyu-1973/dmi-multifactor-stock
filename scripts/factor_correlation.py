"""
因子相关性分析 v4 — 检查因子冗余
"""
import os, sys, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("factor_correlation")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest")
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt")

START = "2025-01-02"
END = "2026-08-10"


def compute_dmi(df, period=14):
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
    result = df.copy()
    ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
    result['DIF'] = ema_fast - ema_slow
    result['DEA'] = result['DIF'].ewm(span=signal, adjust=False).mean()
    result['MACD_bar'] = 2 * (result['DIF'] - result['DEA'])
    return result


def compute_momentum(df, window=20):
    result = df.copy()
    result[f'mom_{window}'] = result['close'] / result['close'].shift(window) - 1
    return result


def compute_ma_lines(df, ma60=60):
    result = df.copy()
    result['ma60'] = result['close'].rolling(window=ma60).mean()
    return result


def get_daily_factors(code, df, fund_map, date):
    """获取某股票某日的所有因子值"""
    if date not in df.index:
        return None
    
    i = df.index.get_loc(date)
    cur = df.iloc[i]
    
    factors = {}
    
    # DMI
    if 'pdi' in df.columns and 'ndi' in df.columns:
        pdi, ndi, adx = cur.get('pdi', 0), cur.get('ndi', 0), cur.get('adx', 0)
        if pdi > ndi and adx > 20:
            factors['dmi_strength'] = min(100, (pdi - ndi) * 2)
            factors['dmi_adx'] = adx
        else:
            factors['dmi_strength'] = 0
            factors['dmi_adx'] = adx
    
    # MACD
    if 'DIF' in df.columns and 'DEA' in df.columns:
        factors['macd_dif'] = cur['DIF']
        factors['macd_bar'] = cur['MACD_bar']
        factors['macd_golden'] = 1 if cur['DIF'] > cur['DEA'] else 0
        # MACD金叉信号（DIF上穿DEA）
        prev = df.iloc[i-1] if i > 0 else cur
        factors['macd_cross'] = 1 if prev['DIF'] <= prev['DEA'] and cur['DIF'] > cur['DEA'] else 0
    
    # 动量
    if 'mom_20' in df.columns:
        factors['mom_20'] = cur['mom_20']
    if 'mom_60' in df.columns:
        factors['mom_60'] = cur['mom_60']
    
    # MA60位置
    if 'ma60' in df.columns:
        factors['close_ma60_ratio'] = cur['close'] / cur['ma60'] if cur['ma60'] else 1
        factors['above_ma60'] = 1 if cur['close'] > cur['ma60'] else 0
    
    # PE估值
    if code in fund_map and date in fund_map[code].index:
        pe = fund_map[code].get('pe')
        pb = fund_map[code].get('pb')
        if pd.notna(pe.get(date, np.nan)):
            factors['pe'] = pe[date]
        if pd.notna(pb.get(date, np.nan)):
            factors['pb'] = pb[date]
    
    return factors if factors else None


def main():
    logger.info("=" * 60)
    logger.info("📊 因子相关性分析")
    logger.info("=" * 60)
    
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
    
    logger.info(f"数据: {len(data_map)}只K线, {len(fund_map)}只基本面")
    
    # 计算技术指标
    for code, df in data_map.items():
        data_map[code] = compute_dmi(df, 14)
        data_map[code] = compute_macd(data_map[code])
        data_map[code] = compute_momentum(data_map[code], 20)
        data_map[code] = compute_momentum(data_map[code], 60)
        data_map[code] = compute_ma_lines(data_map[code])
    
    # 采样日期（每10天取一天，取100个采样点）
    all_dates = sorted(set().union(*(df.index for df in data_map.values())))
    sample_dates = all_dates[::10][:100]
    logger.info(f"采样 {len(sample_dates)} 个交易日")
    
    # 收集因子值
    factor_data = {}  # {factor_name: [values]}
    
    for date in sample_dates:
        for code in data_map:
            factors = get_daily_factors(code, data_map[code], fund_map, date)
            if factors:
                for name, val in factors.items():
                    if name not in factor_data:
                        factor_data[name] = []
                    factor_data[name].append(val)
    
    # 确保所有因子等长
    min_len = min(len(v) for v in factor_data.values())
    for k in factor_data:
        factor_data[k] = factor_data[k][:min_len]
    
    df_factors = pd.DataFrame(factor_data)
    corr_matrix = df_factors.corr()
    
    # 输出
    print("\n" + "=" * 70)
    print("📊 因子相关性矩阵（Pearson相关系数）")
    print("=" * 70)
    print(corr_matrix.round(3).to_string())
    
    # 高相关因子对
    print("\n⚠️  高相关性因子对（|r| > 0.7）:")
    high_corr = []
    cols = corr_matrix.columns
    for i, f1 in enumerate(cols):
        for j, f2 in enumerate(cols):
            if i < j:
                r = corr_matrix.loc[f1, f2]
                if abs(r) > 0.7:
                    high_corr.append((f1, f2, r))
                    print(f"  {f1:20s} <-> {f2:20s}: r = {r:.3f}")
    
    if not high_corr:
        print("  无高相关性因子对 ✓")
    
    # 独立性评分
    print("\n📈 因子独立性评分 (1-平均相关性，越接近1越独立):")
    for f in cols:
        others = [corr_matrix.loc[f, c] for c in cols if c != f]
        avg_abs = np.mean(np.abs(others))
        independence = 1 - avg_abs
        status = "⚠️" if independence < 0.5 else "✓"
        print(f"  {f:20s}: {independence:.3f} {status}")
    
    # 建议
    print("\n💡 优化建议:")
    if high_corr:
        print("  以下因子对相关性过高，建议:")
        for f1, f2, r in high_corr:
            # 判断保留哪个
            keep = f1 if corr_matrix.loc[f1, f1] >= corr_matrix.loc[f2, f2] else f2
            print(f"    - 保留 {keep}，移除另一个（r={r:.3f}）")
    else:
        print("  各因子独立性良好，无需调整 ✓")
        print("  建议维持现有6因子结构")
    
    # 保存
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_correlation_report.txt")
    with open(result_path, 'w') as f:
        f.write(corr_matrix.round(3).to_string())
    print(f"\n报告保存: {result_path}")


if __name__ == "__main__":
    main()
