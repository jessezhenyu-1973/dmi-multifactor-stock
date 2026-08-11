"""
用pytdx拉取沪深300成分股历史数据(2023-2026)
过滤损坏数据,保存为csv
"""
from pytdx.hq import TdxHq_API
import pandas as pd
import os, time, sys

CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"
UNIVERSE_FILE = "/home/jesse/dmi-multifactor-stock/scripts/universe_hs300_latest.txt"

SERVERS = [
    ('115.238.56.198', 7709),
    ('115.238.90.165', 7709),
    ('180.153.18.170', 7709),
    ('218.75.126.9', 7709),
    ('60.12.136.250', 7709),
    ('60.191.117.167', 7709),
]

def connect_api():
    for ip, port in SERVERS:
        try:
            api = TdxHq_API()
            connected = api.connect(ip, port)
            if connected:
                print(f'✅ 连接成功: {ip}:{port}')
                return api
            else:
                print(f'❌ 连接失败: {ip}:{port}')
        except Exception as e:
            print(f'⚠️ 异常: {ip}:{port} -> {str(e)[:40]}')
    return None

def fetch_stock_bars(api, code, market, start_page=0, page_size=800):
    """拉取一只股票的历史K线数据,返回有效数据"""
    all_bars = []
    
    # 最多拉取100页(80000条,约300天)
    for page in range(100):
        try:
            df = api.get_security_bars(4, market, code, start_page + page * page_size, page_size)
            if df is None or len(df) == 0:
                break
            
            # 转换为DataFrame
            if isinstance(df, list):
                df = pd.DataFrame(df)
            
            # 过滤损坏数据: 年份应在2020-2027之间
            if 'year' in df.columns:
                valid = df[(df['year'] >= 2020) & (df['year'] <= 2027)]
                corrupted = len(df) - len(valid)
                if corrupted > 0:
                    df = valid
            
            if len(df) > 0:
                all_bars.append(df)
                if len(df) < page_size:
                    break
        except:
            break
    
    if not all_bars:
        return None
    
    result = pd.concat(all_bars, ignore_index=True)
    return result

def main():
    print("=" * 70)
    print("📊 用pytdx拉取沪深300成分股历史数据 (2023-2026)")
    print("=" * 70)
    
    # 连接服务器
    api = connect_api()
    if not api:
        print("❌ 无法连接任何服务器!")
        sys.exit(1)
    
    # 读取沪深300成分股
    with open(UNIVERSE_FILE) as f:
        codes = [l.strip() for l in f if l.strip()]
    print(f"\n沪深300成分股: {len(codes)}只")
    
    # 检查已有缓存
    existing = set()
    for fn in os.listdir(CACHE_DIR):
        if fn.endswith('_hfq_kline.csv'):
            existing.add(fn.replace('_hfq_kline.csv', ''))
    print(f"已有缓存: {len(existing)}只(2025年起)")
    
    # 拉取前50只测试
    print(f"\n开始拉取(最多50只测试)...")
    success_count = 0
    fail_count = 0
    t0 = time.time()
    
    test_results = []
    for i, code in enumerate(codes[:50]):
        try:
            # 解析代码和市场
            parts = code.split('.')
            if len(parts) != 2:
                fail_count += 1
                continue
            
            symbol = parts[0]  # 6位代码
            market = 0 if parts[1] == 'SH' else 1  # 0=上海, 1=深圳
            
            # 拉取数据
            bars = fetch_stock_bars(api, symbol, market)
            
            if bars is not None and len(bars) > 100:
                # 保存为csv
                out_file = os.path.join(CACHE_DIR, f"{symbol}_hfq_kline_full.csv")
                bars.to_csv(out_file)
                success_count += 1
                test_results.append({
                    'code': code,
                    'rows': len(bars),
                    'start': str(bars['datetime'].min())[:10],
                    'end': str(bars['datetime'].max())[:10],
                })
                print(f"  [{i+1}] {code}: {len(bars)}条, {bars['datetime'].min()}~{bars['datetime'].max()}")
            else:
                fail_count += 1
                print(f"  [{i+1}] {code}: 数据不足({len(bars) if bars is not None else 0}条)")
            
            if (i+1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  ...进度: {i+1}/50, 耗时{elapsed:.1f}s")
                
        except Exception as e:
            fail_count += 1
            print(f"  [{i+1}] {code}: 失败 {str(e)[:40]}")
    
    print(f"\n完成: 成功{success_count}只, 失败{fail_count}只, 总耗时{time.time()-t0:.1f}s")
    
    if test_results:
        # 汇总
        print(f"\n数据汇总:")
        print(f"  总记录数: {sum(r['rows'] for r in test_results)}")
        print(f"  平均记录数: {sum(r['rows'] for r in test_results)/len(test_results):.0f}/只")
        print(f"  最早日期: {min(r['start'] for r in test_results)}")
        print(f"  最晚日期: {max(r['end'] for r in test_results)}")
        
        # 检查是否有2023-2024数据
        count_2023 = sum(1 for r in test_results if r['start'] < '2024-01-01')
        count_2025 = sum(1 for r in test_results if r['start'] < '2025-01-01')
        print(f"  有2023年数据: {count_2023}/{len(test_results)}只")
        print(f"  有2024年数据: {count_2025}/{len(test_results)}只")
        print(f"  有2025年数据: {count_2025}/{len(test_results)}只")

if __name__ == "__main__":
    main()
