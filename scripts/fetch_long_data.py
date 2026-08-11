"""
拉取更长期 K 线数据（2022-01-01 ~ 2026-08-11），用于诚实回测
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jesse/dmi-multifactor-stock/scripts')
import akshare as ak

CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"
os.makedirs(CACHE_DIR, exist_ok=True)

# 现有股票列表
stock_files = [f.replace('_daily.csv', '') for f in os.listdir(CACHE_DIR) if f.endswith('_daily.csv')]
print(f"需要拉取 {len(stock_files)} 只股票的长期数据 (2022-01-01 ~ 2026-08-11)")

# 格式转换: '000963' -> 'sh000963'
def _ak_code(code):
    if code.startswith(('6', '9')):
        return 'sh' + code
    else:
        return 'sz' + code

# 拉取数据
for i, code in enumerate(stock_files[:10]):  # 先试 10 只
    ak_code = _ak_code(code)
    cache_file = os.path.join(CACHE_DIR, f'{code}_daily.csv')
    
    try:
        print(f"  [{i+1}/{len(stock_files)}] 拉取 {code} ({ak_code})...")
        df = ak.stock_zh_a_daily(symbol=ak_code, start_date="20220101", end_date="20260811", adjust="hfq")
        
        if 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # 保存
        df.to_csv(cache_file)
        print(f"    ✅ {len(df)} 行, {df.index.min().date()} ~ {df.index.max().date()}")
        
    except Exception as e:
        print(f"    ❌ {code}: {e}")
        
print("\n完成！")
