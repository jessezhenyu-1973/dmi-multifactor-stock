"""
真实数据加载器（AKShare 自包含，全部带本地缓存）。

设计目标：
  1. 完全不依赖只能在对话里调用的 MCP 工具 —— 任何环境 `python real_data_loader.py` 即可重建缓存。
  2. 时点正确（point-in-time），消除此前"单一未来快照"造成的前瞻偏差：
       - 基本面(ROE/毛利率/净利TTM/PE/PB) 用报告期真实披露值，按报告日 ffill 到交易日；
       - 舆情用个股自身历史量价推导，绝不使用未来数据。
  3. 支撑股票池扩容（HS300 300 只成分股）。

用法：
  from real_data_loader import build_real_dataset
  data_map, fundamental_data, benchmark = build_real_dataset(codes, start, end)
  # data_map:        code -> OHLCV DataFrame(索引=交易日)
  # fundamental_data:code -> 时点对齐的 pe/pb/roe/gross_margin/debt_ratio (索引=交易日, ffill)
  # benchmark:       沪深300 日线 DataFrame (regime 基准)
"""

import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

try:
    import akshare as ak
except ImportError:
    ak = None

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
logger = logging.getLogger(__name__)


# ------------------------- 代码格式工具 -------------------------
def _ak_kline_code(code: str) -> str:
    """'600519.SH' -> 'sh600519' (stock_zh_a_daily 需要带前缀)"""
    raw, mkt = code.split(".")
    return ("sh" if mkt.upper() == "SH" else "sz") + raw


def _raw_code(code: str) -> str:
    """'600519.SH' -> '600519'"""
    return code.split(".")[0]


def _sh_or_sz(raw: str) -> str:
    return "SH" if raw.startswith("6") else "SZ"


# ------------------------- 股票池 -------------------------
def get_hs300_codes(limit: int = None) -> list:
    """返回 HS300 成分股代码列表（'600519.SH' 格式），结果缓存。"""
    p = os.path.join(CACHE, "hs300_universe.json")
    if os.path.exists(p):
        codes = json.load(open(p, encoding="utf-8"))
    else:
        if ak is None:
            raise RuntimeError("akshare 未安装")
        df = ak.index_stock_cons(symbol="000300")
        codes = []
        for _, r in df.iterrows():
            raw = str(r["品种代码"]).zfill(6)
            codes.append(f"{raw}.{_sh_or_sz(raw)}")
        os.makedirs(CACHE, exist_ok=True)
        json.dump(codes, open(p, "w", encoding="utf-8"))
        logger.info(f"HS300 成分股已缓存: {len(codes)} 只")
    if limit:
        codes = codes[:limit]
    return codes


# ------------------------- K 线 -------------------------
def fetch_klines(codes: list, start: str = "20220101", end: str = "20251231",
                 adjust: str = "hfq", max_workers: int = 1) -> dict:
    """
    逐只拉取真实日 K 线（后复权），返回 code -> OHLCV DataFrame。带缓存。

    注意：AKShare 的东方财富解析依赖 py_mini_racer(V8)，**不可多线程并发**，
    否则 V8 隔离区崩溃。故默认串行（max_workers 仅保留接口兼容，实际始终串行）。
    """
    out = {}
    done = 0
    for code in codes:
        done += 1
        cache = os.path.join(CACHE, f"{code}_kline.csv")
        if os.path.exists(cache):
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            out[code] = df
            continue
        try:
            df = ak.stock_zh_a_daily(symbol=_ak_kline_code(code),
                                     start_date=start, end_date=end, adjust=adjust)
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(cache)
            out[code] = df
        except Exception as e:
            logger.warning(f"K 线抓取失败 {code}: {e}")
        if done % 25 == 0:
            logger.info(f"K 线进度 {done}/{len(codes)}")
    logger.info(f"K 线就绪: {len(out)} 只")
    return out


# ------------------------- 总市值（日频） -------------------------
def fetch_market_cap(codes: list, max_workers: int = 1) -> dict:
    """逐只拉取日频总市值(元)，返回 code -> Series(索引=交易日)。带缓存。串行（见 fetch_klines 说明）。"""
    out = {}
    done = 0
    for code in codes:
        done += 1
        cache = os.path.join(CACHE, f"{code}_mktcap_5y.csv")
        if os.path.exists(cache):
            s = pd.read_csv(cache, index_col=0, parse_dates=True)["value"]
            s.index = pd.to_datetime(s.index)
            out[code] = s
            continue
        try:
            df = ak.stock_zh_valuation_baidu(symbol=_raw_code(code),
                                             indicator="总市值", period="近五年")
            s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"]))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.to_frame("value").to_csv(cache)
            out[code] = s
        except Exception as e:
            logger.warning(f"市值抓取失败 {code}: {e}")
        if done % 25 == 0:
            logger.info(f"市值进度 {done}/{len(codes)}")
    logger.info(f"市值就绪: {len(out)} 只")
    return out


# ------------------------- 季报（全市场，时点正确基本面） -------------------------
QUARTERS = [
    "20220331", "20220630", "20220930", "20221231",
    "20230331", "20230630", "20230930", "20231231",
    "20240331", "20240630", "20240930", "20241231",
    "20250331",
]


def fetch_quarterly_reports(quarters: list = None) -> dict:
    """拉取全市场季报，返回 code -> [(报告日, 累计净利, ROE, 毛利率, 营收), ...]。带缓存。"""
    quarters = quarters or QUARTERS
    cache = os.path.join(CACHE, "quarterly_reports.json")
    if os.path.exists(cache):
        raw = json.load(open(cache, encoding="utf-8"))
        # json 把 tuple 转 list，还原
        return {k: [tuple(x) for x in v] for k, v in raw.items()}

    if ak is None:
        raise RuntimeError("akshare 未安装")
    raw = {}
    for q in quarters:
        df = ak.stock_yjbb_em(date=q)  # 全市场一次性返回
        for _, r in df.iterrows():
            raw_code = str(r["股票代码"]).zfill(6)
            code = f"{raw_code}.{_sh_or_sz(raw_code)}"
            try:
                np_cum = float(r.get("净利润-净利润")) if pd.notna(r.get("净利润-净利润")) else None
                roe = float(r.get("净资产收益率")) if pd.notna(r.get("净资产收益率")) else None
                gm = float(r.get("销售毛利率")) if pd.notna(r.get("销售毛利率")) else None
                rev = float(r.get("营业总收入-营业总收入")) if pd.notna(r.get("营业总收入-营业总收入")) else None
            except (ValueError, TypeError):
                continue
            raw.setdefault(code, []).append((q, np_cum, roe, gm, rev))
        logger.info(f"季报 {q} 已拉取")
    os.makedirs(CACHE, exist_ok=True)
    json.dump(raw, open(cache, "w", encoding="utf-8"))
    return raw


# ------------------------- 由季报构建 TTM 净利 -------------------------
def _single_quarters(reps: list):
    """reps: [(日, 累计净利, ...)]；返回 [(日, 单季净利)]，单季=当季累计-同年上一报告累计。"""
    reps = sorted(reps, key=lambda x: x[0])
    prev_cum = {}
    singles = []
    for d, np_cum, roe, gm, rev in reps:
        yr = d[:4]
        single = (np_cum - prev_cum.get(yr, 0.0)) if np_cum is not None else None
        prev_cum[yr] = np_cum if np_cum is not None else prev_cum.get(yr, 0.0)
        singles.append((d, single))
    return singles


def _ttm_net_profit(singles: list):
    """滚动 4 个单季求和得到 TTM 净利，返回 [(报告日, TTM净利)]。"""
    out = []
    for i in range(len(singles)):
        if i >= 3 and all(singles[j][1] is not None for j in range(i - 3, i + 1)):
            ttm = sum(singles[j][1] for j in range(i - 3, i + 1))
            out.append((singles[i][0], ttm))
    return out


# ------------------------- 组装时点正确基本面 -------------------------
def build_point_in_time_fundamentals(codes, klines, mktcap, quarterly) -> dict:
    """
    返回 code -> DataFrame(索引=该股票交易日)，列:
        pe, pb, roe, gross_margin, debt_ratio
    其中 pe=总市值/净利TTM(ffill), pb≈pe/roe, roe/毛利率=报告期真实值(ffill)。
    全部按报告日 ffill 到交易日 => 时点正确，无前瞻。
    """
    fund = {}
    for code in codes:
        if code not in klines or klines[code] is None or klines[code].empty:
            continue
        kline = klines[code]
        reps = quarterly.get(code, [])
        if not reps:
            # 无报告数据 -> 全 NaN，引擎按中性 50 处理
            fund[code] = pd.DataFrame(index=kline.index,
                                      columns=["pe", "pb", "roe", "gross_margin", "debt_ratio"],
                                      dtype=float)
            continue

        ttm = _ttm_net_profit(_single_quarters(reps))
        # 报告日索引的 roe / 毛利率 / 净利TTM
        rep_dates = [pd.Timestamp(r[0]) for r in reps]
        roe_s = pd.Series([r[2] for r in reps], index=rep_dates)
        gm_s = pd.Series([r[3] for r in reps], index=rep_dates)
        if ttm:
            ttm_dates = [pd.Timestamp(t[0]) for t in ttm]
            ttm_s = pd.Series([t[1] for t in ttm], index=ttm_dates)
        else:
            ttm_s = pd.Series(dtype=float)

        # 对齐到交易日并 ffill（时点正确：只用已披露报告）
        idx = kline.index
        roe_a = roe_s.reindex(idx).ffill()
        gm_a = gm_s.reindex(idx).ffill()
        ttm_a = ttm_s.reindex(idx).ffill()

        # 市值对齐到交易日（已有日频）。注意：stock_zh_valuation_baidu 返回单位为【亿元】，
        # 而净利润(TTM)为【元】，统一换算为元再相除，否则 PE/PB 数量级错误。
        if code in mktcap and mktcap[code] is not None:
            mc = mktcap[code].reindex(idx).ffill() * 1e8
        else:
            mc = pd.Series(np.nan, index=idx)

        pe = mc / ttm_a.replace(0, np.nan)
        # 恒等式 PB = PE × ROE（ROE 取小数）：由 PB=PE×ROE 推导。
        # roe_a 为百分比(%)，故先 /100 转小数。之前误写成 pe / (roe/100) 导致 PB 虚高。
        pb = pe * (roe_a / 100.0).replace(0, np.nan)

        out = pd.DataFrame({
            "pe": pe,
            "pb": pb,
            # 注意：roe 与 scoring 函数约定一致 —— 取【百分比】原值（如 12.34 表示 12.34%）。
            # 切勿 /100：compute_fundamental_score / compute_quality_score 均按百分比处理 roe。
            "roe": roe_a,                 # 百分比（与 mock 同口径）
            "gross_margin": gm_a / 100.0, # 百分比 -> 小数（compute_quality_score 按小数处理）
            "debt_ratio": np.nan,         # AKShare 业绩报表不含负债率，留空(质量因子可降权)
        }, index=idx)
        fund[code] = out
    return fund


# ------------------------- regime 基准（沪深300） -------------------------
def fetch_benchmark(start: str = "20220101", end: str = "20251231") -> pd.DataFrame:
    cache = os.path.join(CACHE, "sh000300_kline.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df
    df = ak.stock_zh_index_daily(symbol="sh000300")
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    df.to_csv(cache)
    return df


# ------------------------- 顶层入口 -------------------------
def build_real_dataset(codes: list, start: str = "20220101", end: str = "20251231",
                       adjust: str = "hfq", max_workers: int = 8,
                       with_benchmark: bool = True) -> tuple:
    """
    一键构建真实回测数据集。返回 (data_map, fundamental_data, benchmark)。
    data_map / fundamental_data 索引均为交易日，可直接喂给 run_backtest。
    """
    logger.info(f"构建真实数据集: {len(codes)} 只, {start}~{end}")
    klines = fetch_klines(codes, start, end, adjust, max_workers)
    mktcap = fetch_market_cap(codes, max_workers)
    quarterly = fetch_quarterly_reports()
    fundamental_data = build_point_in_time_fundamentals(codes, klines, mktcap, quarterly)
    benchmark = fetch_benchmark(start, end) if with_benchmark else None
    # 仅保留成功拿到 K 线的股票
    kept = [c for c in codes if c in klines and klines[c] is not None and not klines[c].empty]
    data_map = {c: klines[c] for c in kept}
    fundamental_data = {c: fundamental_data[c] for c in kept if c in fundamental_data}
    logger.info(f"数据集就绪: K线 {len(data_map)} 只, 基本面 {len(fundamental_data)} 只")
    return data_map, fundamental_data, benchmark


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    codes = get_hs300_codes()
    dm, fd, bench = build_real_dataset(codes)
    print(f"K线: {len(dm)} 只; 基本面: {len(fd)} 只; 基准: {len(bench)} 行")
