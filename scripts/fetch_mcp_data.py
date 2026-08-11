"""
MCP 连接器数据获取脚本
使用 workbuddy-stock-mcp 服务器获取沪深300成分股数据
替代 akshare/baostock，直接从通达信和腾讯自选股获取数据
"""
import sys
import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta

# 导入 MCP 服务器函数
sys.path.insert(0, '/home/jesse/workbuddy-stock-mcp')
from server import get_hs300_constituents, get_realtime_quotes, get_historical_kline

CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print("🚀 MCP 连接器 - 沪深300数据获取")
print("=" * 60)

# ============================================================
# 步骤 1: 获取沪深300成分股
# ============================================================
print("\n📊 步骤 1: 获取沪深300成分股...")
try:
    hs300_result = get_hs300_constituents()
    hs300 = json.loads(hs300_result)
    stocks = hs300['data']
    print(f"✅ 获取到 {len(stocks)} 只股票")
    
    # 保存成分股列表
    df_hs300 = pd.DataFrame(stocks)
    df_hs300.to_csv(os.path.join(CACHE_DIR, "hs300_constituents.csv"), index=False)
    print(f"✅ 已保存: hs300_constituents.csv")
except Exception as e:
    print(f"❌ 获取失败: {e}")
    sys.exit(1)

# ============================================================
# 步骤 2: 获取实时行情数据
# ============================================================
print("\n📈 步骤 2: 获取实时行情...")
try:
    # 分批获取，每批20只
    batch_size = 20
    all_quotes = []
    for i in range(0, len(stocks), batch_size):
        batch = stocks[i:i+batch_size]
        codes = [s['stock_code'] for s in batch]
        codes_str = ",".join(codes)
        
        result = get_realtime_quotes(codes_str)
        data = json.loads(result)
        
        if data['status'] == 'success':
            all_quotes.extend(data['data'])
            print(f"  批次 {i//batch_size + 1}: {len(data['data'])} 只")
        
        time.sleep(0.1)  # 避免请求过快
    
    # 保存行情数据
    df_quotes = pd.DataFrame(all_quotes)
    df_quotes.to_csv(os.path.join(CACHE_DIR, "quotes_latest.csv"), index=False)
    print(f"✅ 已保存: quotes_latest.csv ({len(all_quotes)} 只)")
except Exception as e:
    print(f"❌ 获取失败: {e}")

# ============================================================
# 步骤 3: 获取历史K线数据（分批拉取）
# ============================================================
print("\n📉 步骤 3: 获取历史K线数据...")
print("   注意: 每只股票最多拉取500条日K线")

success_count = 0
fail_count = 0
total = min(50, len(stocks))  # 先测试前50只

for idx, stock in enumerate(stocks[:total]):
    code = stock['stock_code']
    market = 'sz' if code.startswith('0') or code.startswith('3') else 'sh'
    
    try:
        # 获取K线数据
        kline_result = get_historical_kline(
            code, 
            market=market, 
            period="daily", 
            count=500
        )
        kline_data = json.loads(kline_result)
        
        if kline_data['status'] == 'success' and kline_data['count'] > 0:
            # 转换数据格式，适配策略需求
            df_kline = pd.DataFrame(kline_data['data'])
            
            # 列名映射
            df_kline.rename(columns={
                'datetime': 'date',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'vol': 'volume'
            }, inplace=True)
            
            # 确保日期格式正确
            df_kline['date'] = pd.to_datetime(df_kline['date'])
            df_kline = df_kline.sort_values('date')
            
            # 保存
            file_path = os.path.join(CACHE_DIR, f"{code}_daily.csv")
            df_kline.to_csv(file_path, index=False)
            success_count += 1
            
            if (idx + 1) % 10 == 0:
                print(f"  进度: {idx+1}/{total} ({success_count} 成功)")
        else:
            fail_count += 1
            
    except Exception as e:
        fail_count += 1
        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx+1}/{total} (失败: {fail_count})")
    
    time.sleep(0.2)  # 避免请求过快

print(f"\n✅ K线数据获取完成:")
print(f"   成功: {success_count} 只")
print(f"   失败: {fail_count} 只")

# ============================================================
# 步骤 4: 数据统计
# ============================================================
print("\n" + "=" * 60)
print("📊 数据获取统计")
print("=" * 60)
print(f"沪深300成分股: {len(stocks)} 只")
print(f"实时行情: {len(all_quotes)} 只")
print(f"历史K线: {success_count} 只 ({success_count/total*100:.1f}% 成功率)")

# 检查数据质量
if success_count > 0:
    sample_file = os.path.join(CACHE_DIR, f"{stocks[0]['stock_code']}_daily.csv")
    if os.path.exists(sample_file):
        df_sample = pd.read_csv(sample_file)
        print(f"\n📅 数据日期范围:")
        print(f"   起始: {df_sample['date'].min()}")
        print(f"   结束: {df_sample['date'].max()}")
        print(f"   总条数: {len(df_sample)}")

print("\n🎉 数据获取完成！")
print("=" * 60)
