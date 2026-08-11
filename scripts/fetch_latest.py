"""
按今天的收盘数据跑一次：抓取最新沪深300成分股行情+基本面
"""
import os, time, logging, pandas as pd
from datetime import datetime

try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    ak = None
    AK_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fetch_latest")

# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_latest")
os.makedirs(CACHE_DIR, exist_ok=True)

START = "20250101"  # 从2025年开始，覆盖近期数据
END = "20260810"     # 到今天

from data_source import DataSourceManager

SUFFIX = lambda code: ".SH" if code.startswith("6") else ".SZ"

def get_hs300_codes():
    if not AK_AVAILABLE:
        raise RuntimeError("akshare 未安装")
    df = ak.index_stock_cons_csindex(symbol="000300")
    codes = []
    for raw in df["成分券代码"].astype(str):
        code = raw.strip().zfill(6)
        if len(code) == 6 and code.isdigit():
            codes.append(code + SUFFIX(code))
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def main():
    logger.info("开始抓取沪深300成分股最新数据...")
    codes = get_hs300_codes()
    logger.info(f"沪深300 成分股数量: {len(codes)}")

    # 保存成分股清单
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300_latest.txt"), "w") as f:
        f.write("\n".join(codes))

    mgr = DataSourceManager(use_real_data=True, cache_dir=CACHE_DIR)

    ok_k, ok_f = 0, 0
    t0 = time.time()
    for i, code in enumerate(codes):
        try:
            k = mgr.fetch_kline(code, START, END, "daily", "hfq")
            if k is not None and not k.empty:
                ok_k += 1
        except Exception as e:
            logger.warning(f"{code} K线失败: {e}")

        try:
            fnd = mgr.fetch_fundamental(code, START, END)
            if fnd is not None and not fnd.empty:
                ok_f += 1
        except Exception as e:
            logger.warning(f"{code} 基本面失败: {e}")

        time.sleep(0.4)

        if (i + 1) % 50 == 0:
            logger.info(f"进度 {i + 1}/{len(codes)} | K线成功 {ok_k} 基本面成功 {ok_f} | 用时 {time.time() - t0:.0f}s")

    logger.info(f"完成：K线成功 {ok_k}/{len(codes)}，基本面成功 {ok_f}/{len(codes)}，总用时 {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
