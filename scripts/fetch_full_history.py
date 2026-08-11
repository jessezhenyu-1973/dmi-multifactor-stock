"""
重新拉取沪深300成分股完整历史数据（2023-2026）
用于回测区间扩展
"""
import os, sys, time, logging
import pandas as pd
import numpy as np
import akshare as ak

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("full_data_fetch")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest")
UNIVERSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt")

os.makedirs(CACHE_DIR, exist_ok=True)

# 读取沪深300成分股
with open(UNIVERSE_FILE, "r") as f:
    codes = [line.strip() for line in f if line.strip()]
logger.info(f"沪深300成分股: {len(codes)} 只")

# 拉取K线数据
for idx, code in enumerate(codes):
    if (idx + 1) % 50 == 0:
        logger.info(f"进度: {idx+1}/{len(codes)}")
    
    symbol = code.split('.')[0]
    kline_file = os.path.join(CACHE_DIR, f"{symbol}_hfq_kline.csv")
    
    # 如果已有数据且覆盖到2023年，跳过
    if os.path.exists(kline_file):
        try:
            df = pd.read_csv(kline_file, index_col=0, parse_dates=True)
            if df.index.min() < pd.Timestamp("2023-06-01"):
                continue  # 数据已完整
        except:
            pass
    
    try:
        # 获取后复权日线数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="hfq")
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '最高': 'high',
            '最低': 'low', '收盘': 'close', '成交量': 'volume',
            '成交额': 'amount', '振幅': 'amplitude', 
            '涨跌幅': 'pct_change', '涨跌额': 'change',
            '换手率': 'turnover'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
        df = df[~df.index.duplicated(keep='first')]  # 去重
        
        # 保存
        df.to_csv(kline_file)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"已保存 {idx+1} 只K线")
            
    except Exception as e:
        logger.warning(f"{code} 拉取失败: {e}")
    
    # 避免请求过快
    time.sleep(0.2)

logger.info("K线数据拉取完成！")

# 拉取基本面数据
for idx, code in enumerate(codes):
    if (idx + 1) % 50 == 0:
        logger.info(f"基本面进度: {idx+1}/{len(codes)}")
    
    symbol = code.split('.')[0]
    fund_file = os.path.join(CACHE_DIR, f"{symbol}_fund.csv")
    
    if os.path.exists(fund_file):
        try:
            df = pd.read_csv(fund_file, index_col=0, parse_dates=True)
            if df.index.min() < pd.Timestamp("2023-06-01"):
                continue
        except:
            pass
    
    try:
        # 获取财务指标
        df_fin = ak.stock_financial_analysis_indicator(symbol=symbol)
        df_pe = ak.stock_a_indicator_lg(symbol=symbol)
        
        # 合并
        if not df_fin.empty and not df_pe.empty:
            df_fin.index = pd.to_datetime(df_fin['日期'])
            df_fin = df_fin.drop(columns=['日期'], errors='ignore')
            
            # PE/PB
            pe_pb = df_pe[['pe', 'pb', 'ps', 'pe_ttm', 'pb'] if 'pb' in df_pe.columns else ['pe', 'ps']]
            pe_pb.index = pd.to_datetime(pe_pb.index) if hasattr(pe_pb.index, 'dtype') else pe_pb.index
            
            # 保存基本财务指标
            df_fin.to_csv(fund_file)
            
        if (idx + 1) % 100 == 0:
            logger.info(f"已保存 {idx+1} 只基本面")
            
    except Exception as e:
        logger.warning(f"{code} 基本面拉取失败: {e}")
    
    time.sleep(0.2)

logger.info("全部数据拉取完成！")
