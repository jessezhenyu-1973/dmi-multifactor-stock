"""
用baostock拉取沪深300成分股历史数据(2023-2026)
补充akshare无法拉取的早期数据
"""
import baostock as bs
import pandas as pd
import os, time, sys

CACHE_DIR = "/home/jesse/dmi-multifactor-stock/scripts/cache_latest"
UNIVERSE_FILE = "/home/jesse/dmi-multifactor-stock/scripts/universe_hs300_latest.txt"

def get_hs300_codes():
    """获取沪深300成分股列表"""
    lg = bs.login()
    rs = bs.query_hs300_stocks()
    codes = []
    while (rs.error_code == '0') and rs.next():
        codes.append(rs.get_row_data())
    bs.logout()
    return [c[1] for c in codes if c[1]]  # code_ext

def fetch_baostock_data(code, start_date="2023-01-01", end_date="2026-08-10"):
    """用baostock拉取一只股票数据"""
    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 后复权
        )
        if rs.error_code != '0':
            return None
        
        data = []
        while (rs.error_code == '0') and rs.next():
            data.append(rs.get_row_data())
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=['date','open','high','low','close','volume','amount','turn'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df[df['close'] > 0]  # 过滤停牌日
        return df
    except:
        return None

def main():
    print("=" * 70)
    print("📊 用baostock补充历史数据 (2023-2026)")
    print("=" * 70)
    
    # 获取沪深300成分股
    print("\n获取沪深300成分股...")
    hs300 = get_hs300_codes()
    print(f"共{len(hs300)}只")
    
    # 读取已存在的universe
    with open(UNIVERSE_FILE) as f:
        existing_codes = [l.strip() for l in f if l.strip()]
    print(f"已有缓存: {len(existing_codes)}只(2025年起)")
    
    # 检查哪些股票已有数据
    existing_klines = [f.replace('_hfq_kline.csv', '') for f in os.listdir(CACHE_DIR) 
                       if f.endswith('_hfq_kline.csv')]
    print(f"已有K线文件: {len(existing_klines)}个")
    
    # 拉取数据(先测几只)
    print(f"\n开始拉取数据(最多20只测试)...")
    success_count = 0
    fail_count = 0
    t0 = time.time()
    
    for i, code in enumerate(hs300[:20]):  # 先测20只
        try:
            df = fetch_baostock_data(code)
            if df is not None and len(df) > 100:
                # 保存为csv
                symbol = code.replace('.SH', '').replace('.SZ', '')
                out_file = os.path.join(CACHE_DIR, f"{symbol}_hfq_kline_baostock.csv")
                df.to_csv(out_file)
                success_count += 1
                print(f"  [{i+1}] {code}: {len(df)}条, {df.index.min().strftime('%Y-%m-%d')}~{df.index.max().strftime('%Y-%m-%d')}")
            else:
                fail_count += 1
                print(f"  [{i+1}] {code}: 数据不足")
        except Exception as e:
            fail_count += 1
            print(f"  [{i+1}] {code}: 失败 {str(e)[:30]}")
        
        if (i+1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  ...进度: {i+1}/20, 耗时{elapsed:.1f}s, 平均{(elapsed/(i+1))*60:.0f}分钟/300只")
    
    print(f"\n完成: 成功{success_count}只, 失败{fail_count}只, 总耗时{time.time()-t0:.1f}s")
    
    if success_count > 0:
        # 检查数据质量
        test_file = os.path.join(CACHE_DIR, f"{hs300[0].replace('.SH', '').replace('.SZ', '')}_hfq_kline_baostock.csv")
        if os.path.exists(test_file):
            test_df = pd.read_csv(test_file, index_col=0, parse_dates=True)
            print(f"\n数据质量检查:")
            print(f"  日期范围: {test_df.index.min()} ~ {test_df.index.max()}")
            print(f"  总行数: {len(test_df)}")
            print(f"  2023-01-01前数据: {(test_df.index < '2023-01-01').sum()}行")
            print(f"  2025-01-01前数据: {(test_df.index < '2025-01-01').sum()}行")
            print(f"  列名: {list(test_df.columns)}")

if __name__ == "__main__":
    main()
