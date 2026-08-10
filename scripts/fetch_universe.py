"""
抓取并缓存沪深300成分股的真实行情与基本面数据
================================================
- 成分股列表：AKShare index_stock_cons_csindex(symbol="000300")
- 区间：2023-01-01 ~ 2024-12-31（留足指标预热）
- 落盘：./cache/<code>_kline.csv 与 ./cache/<code>_fund.csv
- 可重入：已缓存的标的自动跳过，断点续传
- 用法：python fetch_universe.py
"""
import os
import time
import logging
import pandas as pd

# akshare 延迟导入：离线回测（已有缓存）无需 akshare，避免硬依赖
try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    ak = None
    AK_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fetch_universe")

# 缓存目录：可用环境变量 QF_CACHE_DIR 覆盖（便于复用已抓取的沪深300缓存）
CACHE_DIR = os.environ.get(
    "QF_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"),
)
START = "20230101"
END = "20241231"
os.makedirs(CACHE_DIR, exist_ok=True)

# 复用策略内的数据源（自带缓存 + 重试退避）
from data_source import DataSourceManager

SUFFIX = lambda code: ".SH" if code.startswith("6") else ".SZ"


def get_hs300_codes() -> list:
    if not AK_AVAILABLE:
        raise RuntimeError("akshare 未安装，无法抓取沪深300成分股；请先 `pip install akshare` 或复用已有的 QF_CACHE_DIR 缓存。")
    df = ak.index_stock_cons_csindex(symbol="000300")
    codes = []
    for raw in df["成分券代码"].astype(str):
        code = raw.strip().zfill(6)
        if len(code) == 6 and code.isdigit():
            codes.append(code + SUFFIX(code))
    # 去重保序
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def main():
    codes = get_hs300_codes()
    logger.info(f"沪深300 成分股数量: {len(codes)}")

    # 保存成分股清单（供调参脚本参考）
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe_hs300.txt"), "w") as f:
        f.write("\n".join(codes))

    mgr = DataSourceManager(use_real_data=True, cache_dir=CACHE_DIR)

    ok_k, ok_f = 0, 0
    t0 = time.time()
    for i, code in enumerate(codes):
        try:
            k = mgr.fetch_kline(code, START, END)
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

        # 节流，避免密集请求触发远端限流/断连
        time.sleep(0.4)

        if (i + 1) % 25 == 0:
            logger.info(f"进度 {i + 1}/{len(codes)} | K线成功 {ok_k} 基本面成功 {ok_f} | 用时 {time.time() - t0:.0f}s")

    logger.info(f"完成：K线成功 {ok_k}/{len(codes)}，基本面成功 {ok_f}/{len(codes)}，总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
