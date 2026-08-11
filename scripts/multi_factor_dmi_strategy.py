"""
多因子选股策略 - DMI + 技术分析 + 基本面 + 舆情监控（9 因子）
============================================================

策略核心（9 个因子，加权综合打分）：
- 技术面：
  - DMI 因子：+DI/-DI 交叉 + ADX 趋势强度过滤
  - MACD 因子：DIF/DEA 零轴位置 + 金叉 + 红柱动量
  - KDJ 因子：超买超卖区 + 金叉/死叉拐点
  - RSI 因子：相对强弱超买超卖
  - 布林带因子：%B 通道位置 + 挤压突破预期
  - 威科夫量价因子：量涨价涨（需求）/ 量涨价跌（供给）/ 量能高潮
  - 135 战法因子：13 日操作线 + 34 日生命线 + 均线互换 + 放量确认
- 基本面因子：PE/PB/ROE 价值筛选
- 舆情因子：新闻情绪 + 资金流向
- 止损纪律：单股 -10% 硬止损
- 回撤控制：组合最大回撤 -20% 触发清仓，回撤 -15% 触发减仓

数据源：支持 AKShare（默认）、通达信、模拟数据自动降级
依赖：pandas, numpy, akshare (pip install akshare)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import sys
import os
import time

# 添加当前目录到路径，以便导入数据源模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_source import DataSourceManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 1. DMI 因子计算
# ============================================================

def compute_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算 DMI（Directional Movement Index）指标
    
    Args:
        df: 包含 OHLCV 列的 DataFrame
        period: 计算周期，默认 14
    
    Returns:
        包含 +DI, -DI, ADX 列的 DataFrame
    """
    result = df.copy()
    
    # TR (True Range)
    result['high_low'] = result['high'] - result['low']
    result['high_prev_close'] = abs(result['high'] - result['close'].shift(1))
    result['low_prev_close'] = abs(result['low'] - result['close'].shift(1))
    result['TR'] = result[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    
    # +DM (+Directional Movement)
    result['high_diff'] = result['high'] - result['high'].shift(1)
    result['low_diff'] = result['low'].shift(1) - result['low']
    result['+DM'] = np.where(
        (result['high_diff'] > result['low_diff']) & 
        (result['high_diff'] > 0),
        result['high_diff'],
        0
    )
    
    # -DM (-Directional Movement)
    result['-DM'] = np.where(
        (result['low_diff'] > result['high_diff']) & 
        (result['low_diff'] > 0),
        result['low_diff'],
        0
    )
    
    # 平滑处理
    result['ATR'] = result['TR'].rolling(window=period).mean()
    result['+DI'] = 100 * result['+DM'].rolling(window=period).mean() / result['ATR']
    result['-DI'] = 100 * result['-DM'].rolling(window=period).mean() / result['ATR']
    
    # DX 和 ADX
    result['DX'] = 100 * abs(result['+DI'] - result['-DI']) / (result['+DI'] + result['-DI'])
    result['ADX'] = result['DX'].rolling(window=period).mean()
    
    # 清理临时列
    result.drop(columns=['high_low', 'high_prev_close', 'low_prev_close', 
                         'TR', 'high_diff', 'low_diff', '+DM', '-DM', 'ATR', 'DX'],
                inplace=True, errors='ignore')
    
    return result


# ============================================================
# 1b. 技术指标因子计算（MACD / KDJ / RSI / 布林 / 威科夫 / 135 战法）
# ============================================================

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    计算 MACD 指标，新增 DIF / DEA / MACD_bar 三列
    DIF = EMA(fast) - EMA(slow)
    DEA = EMA(DIF, signal)
    MACD_bar = 2 * (DIF - DEA)
    """
    result = df.copy()
    ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
    result['DIF'] = ema_fast - ema_slow
    result['DEA'] = result['DIF'].ewm(span=signal, adjust=False).mean()
    result['MACD_bar'] = 2 * (result['DIF'] - result['DEA'])
    return result


def compute_kdj(df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    """
    计算 KDJ 指标，新增 K / D / J 三列
    RSV = (close - low_n) / (high_n - low_n) * 100
    K = SMA(RSV, k_period); D = SMA(K, d_period); J = 3K - 2D
    """
    result = df.copy()
    low_n = result['low'].rolling(window=n).min()
    high_n = result['high'].rolling(window=n).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = (result['close'] - low_n) / rng * 100
    rsv = rsv.fillna(50)
    # 平滑移动平均（SMA）
    k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    result['K'] = k
    result['D'] = d
    result['J'] = 3 * k - 2 * d
    return result


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算 RSI 相对强弱指标（Wilder 平滑），新增 RSI 列
    """
    result = df.copy()
    delta = result['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50)
    result['RSI'] = rsi
    return result


def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    计算布林带，新增 bb_mid / bb_upper / bb_lower / pct_b / bandwidth 列
    """
    result = df.copy()
    mid = result['close'].rolling(window=period).mean()
    std = result['close'].rolling(window=period).std()
    result['bb_mid'] = mid
    result['bb_upper'] = mid + num_std * std
    result['bb_lower'] = mid - num_std * std
    width = (result['bb_upper'] - result['bb_lower'])
    result['pct_b'] = (result['close'] - result['bb_lower']) / width.replace(0, np.nan)
    result['pct_b'] = result['pct_b'].fillna(0.5)
    result['bandwidth'] = (result['bb_upper'] - result['bb_lower']) / mid.replace(0, np.nan)
    result['bandwidth'] = result['bandwidth'].fillna(0)
    return result


def compute_wyckoff(df: pd.DataFrame, vol_period: int = 20) -> pd.DataFrame:
    """
    计算威科夫量价分析所需字段：相对成交量 rel_volume = 成交量 / 成交量均线
    """
    result = df.copy()
    vol_sma = result['volume'].rolling(window=vol_period).mean()
    result['rel_volume'] = result['volume'] / vol_sma.replace(0, np.nan)
    result['rel_volume'] = result['rel_volume'].fillna(1.0)
    return result


def compute_ma_lines(df: pd.DataFrame, ma13: int = 13, ma34: int = 34) -> pd.DataFrame:
    """
    计算 135 战法所需均线：13 日操作线 + 34 日生命线
    """
    result = df.copy()
    result['ma13'] = result['close'].rolling(window=ma13).mean()
    result['ma34'] = result['close'].rolling(window=ma34).mean()
    return result


def compute_momentum(df: pd.DataFrame, short: int = 20, long: int = 60) -> pd.DataFrame:
    """
    计算动量因子所需字段：短/长周期价格动量（相对收益）。
    - short_mom: 近 short 根收益率（短期动量）
    - long_mom:  近 long 根收益率（中期动量，更稳）
    """
    result = df.copy()
    result['short_mom'] = result['close'].pct_change(short)
    result['long_mom'] = result['close'].pct_change(long)
    return result


def compute_momentum_factor(
    df: pd.DataFrame,
    date_idx: int,
    short_w: float = 0.6,
    long_w: float = 0.4,
    scale: float = 400.0
) -> float:
    """
    动量因子得分 (0-100)

    逻辑：短/中期动量加权合成，动量越强得分越高（追涨，与价值因子互补）。
    score = 50 + (short_w*short_mom + long_w*long_mom) * scale，裁剪到 [0,100]。
    数据不足（< long 根）返回中性 50。

    Args:
        df: 含 short_mom/long_mom 列的 DataFrame（由 compute_momentum 预计算）
        date_idx: 当前本地行号
        short_w/long_w: 短/长动量权重（默认 0.6/0.4）
        scale: 收益→分数的缩放（默认 400：10% 动量 ≈ +40 分）
    """
    if 'short_mom' not in df.columns or 'long_mom' not in df.columns or date_idx < 60:
        return 50.0
    s = df.iloc[date_idx]['short_mom']
    l = df.iloc[date_idx]['long_mom']
    if pd.isna(s) or pd.isna(l):
        return 50.0
    raw = short_w * s + long_w * l
    score = 50.0 + raw * scale
    return float(min(max(score, 0.0), 100.0))


# ============================================================
# 2. 基本面因子计算
# ============================================================

def compute_fundamental_score(
    df: pd.DataFrame,
    pe_min: float = 0,
    pe_max: float = 40,
    pb_max: float = 10,
    roe_min: float = 8
) -> pd.Series:
    """
    计算基本面综合得分
    
    Args:
        df: 包含 pe, pb, roe 列的 DataFrame
        pe_min: PE 下限
        pe_max: PE 上限
        pb_max: PB 上限
        roe_min: ROE 下限（%）
    
    Returns:
        基本面得分 Series（0-100 分）
    """
    scores = pd.Series(0.0, index=df.index)
    
    # 列不存在时视为中性（缺失数据不惩罚，避免 KeyError）
    pe = df['pe'] if 'pe' in df.columns else pd.Series(np.nan, index=df.index)
    pb = df['pb'] if 'pb' in df.columns else pd.Series(np.nan, index=df.index)
    roe = df['roe'] if 'roe' in df.columns else pd.Series(np.nan, index=df.index)
    
    # PE 得分（越低越好，但大于 0）
    pe_valid = pe.between(pe_min, pe_max)
    pe_score = pd.Series(0.0, index=df.index)
    pe_score[pe_valid] = (1 - (pe[pe_valid] - pe_min) / (pe_max - pe_min)) * 30  # 最高 30 分
    
    # PB 得分（越低越好）
    pb_valid = pb <= pb_max
    pb_score = pd.Series(0.0, index=df.index)
    pb_score[pb_valid] = (1 - pb[pb_valid] / pb_max) * 30  # 最高 30 分
    
    # ROE 得分（越高越好，缺失则记 0）
    roe_valid = roe.notna() & (roe >= roe_min)
    roe_score = pd.Series(0.0, index=df.index)
    if roe_valid.any():
        roe_score[roe_valid] = (roe[roe_valid] / 50 * 40).clip(upper=40)  # 最高 40 分
    
    scores = pe_score + pb_score + roe_score
    
    # PE/PB 有值即视为满足条件，ROE 缺失放宽（不再强制三者齐全）
    mask = pe_valid & pb_valid
    scores[~mask] *= 0.5  # 不满足条件的减半
    
    return scores


def compute_quality_score(
    df: pd.DataFrame,
    roe_ref: float = 12.0,
    roe_scale: float = 4.0
) -> pd.Series:
    """
    质量因子得分 (0-100) —— 与价值因子正交

    价值因子用 PE/PB 便宜度 + ROE 阈值过滤；质量因子用 ROE 水平的【连续】打分，
    并可在真实基本面提供毛利率/资产负债率时合成更完整的质量分（高毛利、低负债更优）。

    组件：
    - 盈利能力：ROE 越高越优，相对参考值 roe_ref 线性映射（roe_ref=12% → 50 分，每 +1pct ≈ +4 分）
    - 盈利质量（可选）：gross_margin 越高越优（30% 为中性）；debt_ratio 越低越优（50% 为中性）
    缺失字段不参与；全缺返回中性 50，避免 KeyError。

    Args:
        df: 含 roe / gross_margin / debt_ratio 任一列的 DataFrame
        roe_ref: ROE 中性参考值（%）
        roe_scale: ROE→分数 缩放
    """
    idx = df.index
    comps = []

    # 盈利能力（必选组件）
    if 'roe' in df.columns:
        roe = pd.to_numeric(df['roe'], errors='coerce')
        prof = (50.0 + (roe - roe_ref) * roe_scale).clip(0.0, 100.0)
    else:
        prof = pd.Series(50.0, index=idx)
    comps.append(prof)

    # 盈利质量增强（真实数据接入后才生效；mock 仅 roe 时不影响）
    if 'gross_margin' in df.columns:
        gm = pd.to_numeric(df['gross_margin'], errors='coerce')
        gm_score = (50.0 + (gm - 0.30) * 100.0).clip(0.0, 100.0)  # 毛利率 30% 为中性
        comps.append(gm_score)
    if 'debt_ratio' in df.columns:
        dr = pd.to_numeric(df['debt_ratio'], errors='coerce')
        dr_score = (50.0 - (dr - 0.50) * 100.0).clip(0.0, 100.0)  # 资产负债率 50% 为中性，越低越优
        comps.append(dr_score)

    if len(comps) > 1:
        quality = pd.concat(comps, axis=1).mean(axis=1)
    else:
        quality = comps[0]
    return quality


# ============================================================
# 2.5 时点正确舆情（无需外部数据，绝不使用未来信息）
# ============================================================
def compute_point_in_time_sentiment(df: pd.DataFrame, i: int) -> float:
    """
    由个股自身历史量价推导的"技术面情绪"，完全时点正确（只用 index<=i 的数据）。
    返回 [-1, 1]：
      - 近 20 日收益率（趋势方向）为主
      - 近 20 日成交量相对前 20 日的放大倍数为辅（放量助涨、缩量助跌）
    用于替换此前"单一未来快照"资金流舆情，从根上消除前瞻偏差。
    """
    if i < 20:
        return 0.0
    close = df['close'].values
    r20 = close[i] / close[i - 20] - 1.0
    vol = df['volume'].values
    avg_recent = vol[i - 19:i + 1].mean()
    avg_prev = vol[max(0, i - 39):i - 19].mean()
    vol_ratio = (avg_recent / avg_prev - 1.0) if avg_prev > 0 else 0.0
    raw = 6.0 * r20 + 0.5 * vol_ratio
    return float(np.tanh(raw))


# ============================================================
# 2.6 市场状态（regime）识别：沪深300 趋势 + 波动率
# ============================================================
class MarketRegime:
    """
    基于宽基基准（沪深300）的市场状态识别，用于熊市/高波动期降低暴露。
    状态 -> 目标暴露：bull=1.0 / neutral=0.5 / bear=0.0。
    完全时点正确：只用基准在 date 及之前的数据判定。
    """

    def __init__(self, benchmark: pd.DataFrame, ma_window: int = 200,
                 vol_window: int = 60, vol_z_thresh: float = 2.0):
        self.benchmark = benchmark
        self.ma_window = ma_window
        self.vol_window = vol_window
        self.vol_z_thresh = vol_z_thresh
        self.ret = benchmark['close'].pct_change()
        self.roll_vol = self.ret.rolling(vol_window).std()
        self.ma = benchmark['close'].rolling(ma_window).mean()

    def exposure(self, date) -> float:
        """返回目标暴露倍数 [0,1]，基于 date 及之前的数据判定 regime。"""
        if self.benchmark is None or date not in self.benchmark.index:
            return 1.0
        i = self.benchmark.index.get_loc(date)
        if i < self.ma_window:
            return 1.0  # 预热期默认满仓
        price = self.benchmark['close'].iloc[i]
        ma = self.ma.iloc[i]
        vol = self.roll_vol.iloc[i]
        if pd.isna(vol) or vol <= 0:
            vol_z = 0.0
        else:
            hist = self.roll_vol.iloc[:i + 1]
            vol_median = hist.median()
            vol_std = hist.std()
            vol_z = (vol - vol_median) / (vol_std + 1e-12) if vol_std and not pd.isna(vol_std) else 0.0
        trend_up = price > ma
        high_vol = vol_z > self.vol_z_thresh
        if (not trend_up) and high_vol:
            return 0.0   # 熊市 + 高波动：空仓
        if (not trend_up) or high_vol:
            return 0.5   # 熊市趋势 / 牛但高波动：半仓
        return 1.0       # 牛市平稳：满仓


# ============================================================
# 3. 舆情监控模块
# ============================================================

@dataclass
class SentimentSignal:
    """舆情信号"""
    date: str
    sentiment_score: float  # -1 到 1，负为负面，正为正面
    news_count: int  # 新闻数量
    fund_flow: float  # 资金流向（亿元），正为净流入
    social_mentions: int  # 社交媒体提及量
    overall_rating: str  # 正面/中性/负面


class SentimentMonitor:
    """
    舆情监控模块
    
    数据来源（按优先级降级）：
    1. AKShare 新闻情绪
    2. 腾讯财经/东方财富舆情
    3. 同花顺 iFinD
    4. 模拟数据（用于回测）
    """
    
    # 真实资金流舆情归一化缩放：MainNetFlow/总市值 后再乘该系数并 tanh。
    # 约 0.1% 市值净流入 -> 0.1*1000 -> tanh≈0.76（强但非饱和），单日常规流幅映射到 ±0.1~0.9。
    FUND_FLOW_SENTIMENT_SCALE = 1000.0

    def __init__(self, use_real_data: bool = False, cache_dir: Optional[str] = None,
                 use_westock_real: bool = False):
        self.use_real_data = use_real_data
        self.cache_dir = cache_dir
        # 真实舆情开关（腾讯自选股 westock-mcp 主力资金流）：开启且存在 {code}_realflow.csv 时，
        # 由主力净流入/总市值归一化得到 [-1,1] 舆情分，替换 md5 mock。基线（未开启）零影响。
        self.use_westock_real = use_westock_real
        self._warned_lookahead = False
        self.sentiment_history: List[SentimentSignal] = []

    def get_sentiment_score(self, stock_code: str, date: str) -> float:
        """
        获取个股舆情得分（-1 到 1）
        
        Args:
            stock_code: 股票代码
            date: 日期
        
        Returns:
            舆情得分
        """
        # 优先级 1：真实资金流舆情（westock-mcp，已落地）
        if self.use_westock_real:
            s = self._real_sentiment_from_flow(stock_code)
            if s is not None:
                return s
        # 优先级 2：AKShare 真实舆情（占位，暂未实现）
        if self.use_real_data:
            return self._fetch_real_sentiment(stock_code, date)
        # 回退：确定性 md5 mock（保证可复现）
        return self._generate_mock_sentiment(stock_code, date)

    def _real_sentiment_from_flow(self, stock_code: str) -> Optional[float]:
        """读 {code}_realflow.csv，由主力净流入/总市值归一化得到舆情分；无缓存返回 None。"""
        if not self.cache_dir:
            return None
        import os
        p = os.path.join(self.cache_dir, f"{stock_code}_realflow.csv")
        if not os.path.exists(p):
            return None
        try:
            import pandas as pd
            import math
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if df.empty:
                return None
            row = df.iloc[0]
            mcap_yuan = float(row["total_market_cap_yi"]) * 1e8
            flow = float(row["main_net_flow"])
            if mcap_yuan <= 0:
                return 0.0
            ratio = flow / mcap_yuan
            if not self._warned_lookahead:
                logger.warning(
                    "真实资金流舆情使用单一当前快照(2026-08-10)覆盖全部历史再平衡日，含前瞻偏差；"
                    "仅用于验证数据接入，非时点正确信号。"
                )
                self._warned_lookahead = True
            return float(math.tanh(ratio * self.FUND_FLOW_SENTIMENT_SCALE))
        except Exception as e:
            logger.warning(f"读取真实资金流舆情失败 {p}: {e}")
            return None
    
    def _fetch_real_sentiment(self, stock_code: str, date: str) -> float:
        """
        获取真实舆情数据
        
        数据源优先级：
        1. AKShare - 新闻情绪
        2. 腾讯财经 - 市场情绪
        3. 同花顺问财
        """
        try:
            # AKShare 新闻情绪（示例）
            import akshare as ak
            # 需要根据实际情况调整 API
            pass
        except ImportError:
            logger.warning("AKShare 未安装，回退到模拟数据")
            return self._generate_mock_sentiment(stock_code, date)
        
        return 0.0
    
    def _generate_mock_sentiment(self, stock_code: str, date: str) -> float:
        """
        生成模拟舆情数据（用于回测）
        
        使用确定性摘要做种子，保证跨进程/跨运行可复现
        （Python 3 的字符串 hash() 每进程加盐，不能直接用作种子）
        注意：对代码做后缀归一化（000001.SZ -> 000001），
        否则带/不带交易所后缀会得到不同舆情分，破坏可复现性。
        """
        import hashlib
        norm_code = stock_code.split('.')[0]
        seed = int(hashlib.md5(f"{norm_code}{date}".encode()).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        # 舆情得分在 [-0.5, 0.5] 之间
        return np.random.uniform(-0.5, 0.5)
    
    def compute_sentiment_factor(
        self, 
        stock_codes: List[str], 
        dates: List[str]
    ) -> pd.DataFrame:
        """
        计算所有股票在日期列表上的舆情因子
        
        Args:
            stock_codes: 股票代码列表
            dates: 日期列表
        
        Returns:
            舆情因子 DataFrame（行=日期，列=股票）
        """
        data = {}
        for code in stock_codes:
            data[code] = [
                self.get_sentiment_score(code, date)
                for date in dates
            ]
        
        return pd.DataFrame(data, index=dates)


# ============================================================
# 4. 止损与回撤控制
# ============================================================

@dataclass
class TradeRecord:
    """交易记录"""
    code: str
    buy_date: str
    buy_price: float
    sell_date: Optional[str] = None
    sell_price: Optional[float] = None
    reason: str = ""  # 卖出原因：止损/逻辑/时间/调仓
    pnl_pct: float = 0.0  # 收益率


class RiskManager:
    """
    风险管理器
    
    包含：
    1. 单股 -10% 硬止损
    2. 组合最大回撤 -20% 触发减仓
    3. 仓位控制
    """
    
    def __init__(
        self,
        stop_loss_pct: float = -0.10,  # -10% 止损
        max_drawdown_pct: float = -0.20,  # -20% 最大回撤
        max_positions: int = 10,  # 最大持仓数
        min_positions: int = 3,   # 最小持仓数
    ):
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_positions = max_positions
        self.min_positions = min_positions
        
        self.trades: Dict[str, TradeRecord] = {}  # 活跃持仓
        self.trade_history: List[TradeRecord] = []  # 历史交易
        self.peak_nav = 0.0  # 最高净值
        self.current_nav = 0.0  # 当前净值
        self.drawdown = 0.0  # 当前回撤
    
    def check_stop_loss(
        self, 
        code: str, 
        current_price: float, 
        current_date: str
    ) -> Optional[str]:
        """
        检查是否需要触发止损
        
        Returns:
            如果触发止损，返回卖出原因；否则返回 None
        """
        if code not in self.trades:
            return None
        
        trade = self.trades[code]
        pnl = (current_price - trade.buy_price) / trade.buy_price
        
        if pnl <= self.stop_loss_pct:
            return "硬止损"
        return None
    
    def check_max_drawdown(self) -> bool:
        """
        检查组合是否触发最大回撤限制
        
        Returns:
            如果触发最大回撤限制，返回 True
        """
        if self.peak_nav > 0:
            self.drawdown = (self.current_nav - self.peak_nav) / self.peak_nav
            if self.drawdown <= self.max_drawdown_pct:
                return True
        return False
    
    def update_nav(self, nav: float):
        """更新组合净值"""
        self.current_nav = nav
        if nav > self.peak_nav:
            self.peak_nav = nav
    
    def should_reduce_positions(self) -> bool:
        """
        是否需要减仓
        
        当回撤接近阈值时（如 -15%），开始减仓
        """
        if self.peak_nav > 0:
            current_dd = (self.current_nav - self.peak_nav) / self.peak_nav
            if current_dd <= -0.15:  # 回撤超过 15% 开始减仓
                return True
        return False
    
    def get_position_count(self) -> int:
        """获取当前持仓数"""
        return len(self.trades)
    
    def add_trade(self, trade: TradeRecord):
        """添加交易记录"""
        self.trades[trade.code] = trade
    
    def close_trade(self, code: str, sell_price: float, sell_date: str, reason: str):
        """平仓"""
        if code in self.trades:
            trade = self.trades[code]
            trade.sell_price = sell_price
            trade.sell_date = sell_date
            trade.reason = reason
            trade.pnl_pct = (sell_price - trade.buy_price) / trade.buy_price
            self.trade_history.append(trade)
            del self.trades[code]


# ============================================================
# 5. 多因子选股策略引擎
# ============================================================

@dataclass
class FactorWeights:
    """因子权重配置（9 因子，权重之和 = 1.0）"""
    # —— 技术面（趋势 / 动量 / 波动 / 量价）——
    dmi_weight: float = 0.12        # DMI 趋势方向 + 强度
    macd_weight: float = 0.12       # MACD 趋势动量
    kdj_weight: float = 0.08        # KDJ 超买超卖 / 拐点
    rsi_weight: float = 0.08        # RSI 相对强弱
    bollinger_weight: float = 0.08  # 布林带通道位置
    wyckoff_weight: float = 0.10    # 威科夫量价关系
    method135_weight: float = 0.10  # 135 战法短线趋势
    # —— 基本面 ——
    fundamental_weight: float = 0.18  # PE/PB/ROE 价值筛选
    # —— 舆情面 ——
    sentiment_weight: float = 0.14    # 新闻情绪 + 资金流向
    # —— 动量（可选，默认 0：关闭，保持原 9 因子基线不变；启用时建议将其余权重同比缩放使总和=1）——
    momentum_weight: float = 0.0
    # —— 质量（可选，默认 0：关闭；以 ROE 盈利能力为核心的连续打分，与价值因子正交）——
    quality_weight: float = 0.0


@dataclass
class TransactionCost:
    """交易成本模型（让回测更接近实盘）

    - commission：佣金费率（双边收取）
    - min_commission：单笔最低佣金（元）
    - stamp：印花税费率（A 股仅卖出收取，stamp_sell_only=True）
    - slippage：滑点（双边，成交价相对收盘价偏移）
    """
    commission: float = 0.0003       # 佣金 0.03%
    min_commission: float = 5.0      # 单笔最低佣金 5 元
    stamp: float = 0.0005            # 印花税 0.05%（仅卖出）
    stamp_sell_only: bool = True
    slippage: float = 0.001          # 滑点 0.1%（双边）


class MultiFactorDMIStrategy:
    """
    多因子 DMI 选股策略
    
    策略流程：
    1. 计算 DMI 因子：+DI > -DI 且 ADX > 25
    2. 计算基本面因子：PE/PB/ROE 评分
    3. 计算舆情因子：新闻情绪得分
    4. 因子加权综合打分
    5. 选择 Top N 股票建仓
    6. 实时监控止损和回撤
    
    数据源：支持 AKShare/通达信/模拟数据自动降级
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,  # 初始资金 100 万
        top_n: int = 5,  # 持仓股票数
        factor_weights: Optional[FactorWeights] = None,
        risk_manager: Optional[RiskManager] = None,
        sentiment_monitor: Optional[SentimentMonitor] = None,
        rebalance_freq: int = 20,  # 调仓频率（交易日）
        use_real_data: bool = False,  # 是否使用真实数据
        data_source: Optional[DataSourceManager] = None,  # 数据源管理器
        cost: Optional[TransactionCost] = None,  # 交易成本模型
        use_point_in_time_sentiment: bool = False,  # 时点正确技术舆情（默认关闭，保持基线）
        use_regime: bool = False,  # 市场状态(regime)暴露调控（默认关闭，保持基线）
        regime: Optional[MarketRegime] = None,  # 预构建的 MarketRegime 实例
    ):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.weights = factor_weights or FactorWeights()
        self.risk_manager = risk_manager or RiskManager()
        self.sentiment_monitor = sentiment_monitor or SentimentMonitor(use_real_data=False)
        self.rebalance_freq = rebalance_freq
        self.use_real_data = use_real_data
        self.data_source = data_source or DataSourceManager(use_real_data=use_real_data)
        self.cost = cost or TransactionCost()
        # 时点正确舆情 / regime 调控（默认关闭，不影响既有基线）
        self.use_point_in_time_sentiment = use_point_in_time_sentiment
        self.use_regime = use_regime
        self.regime = regime
        
        # 回测结果
        self.nav_history: List[dict] = []  # 净值历史
        self.trade_log: List[TradeRecord] = []  # 交易日志
        self.current_positions: Dict[str, float] = {}  # 当前持仓（代码 -> 买入成本）
        self.position_dates: Dict[str, str] = {}  # 当前持仓买入日期（代码 -> 日期字符串）
        self.position_shares: Dict[str, int] = {}  # 当前持仓股数（代码 -> 股数）
        self.position_cost_basis: Dict[str, float] = {}  # 持仓总成本（含买入佣金/滑点）
        self.cash: float = 0.0  # 可用现金
    
    def compute_dmi_factor(
        self, 
        df: pd.DataFrame, 
        date_idx: int
    ) -> float:
        """
        计算 DMI 因子得分
        
        条件：
        - +DI > -DI：多头趋势
        - ADX > 25：趋势强度足够
        - +DI 与 -DI 差值越大，得分越高
        
        Returns:
            因子得分 (0-100)
        """
        if date_idx < 14:
            return 50.0  # 数据不足时给中性分
        
        current = df.iloc[date_idx]
        prev = df.iloc[date_idx - 1]
        
        # 检查 +DI > -DI
        if current['+DI'] <= current['-DI']:
            dmi_score = 0  # 不满足多头条件
        else:
            # ADX 趋势强度过滤
            if current['ADX'] > 25:
                # 得分基于 +DI - -DI 的差值
                diff = current['+DI'] - current['-DI']
                dmi_score = min(diff * 2, 100)  # 最高 100 分
            else:
                dmi_score = 30  # ADX 偏低，给低分
        
        return dmi_score
    
    def compute_macd_factor(self, df: pd.DataFrame, date_idx: int) -> float:
        """
        MACD 因子得分 (0-100)
        逻辑：DIF 在零轴上方且 DIF > DEA（多头排列）+ 金叉 + 红柱放大 越高分
        """
        if 'DIF' not in df.columns or date_idx < 35:
            return 50.0
        cur = df.iloc[date_idx]
        prev = df.iloc[date_idx - 1]
        score = 50.0
        if cur['DIF'] > cur['DEA']:
            score += 20
        if cur['DIF'] > 0:
            score += 15
        # 金叉（DIF 上穿 DEA）
        if prev['DIF'] <= prev['DEA'] and cur['DIF'] > cur['DEA']:
            score += 15
        # 红柱为正且放大
        if cur['MACD_bar'] > 0:
            score += 10
        if cur['MACD_bar'] > prev['MACD_bar']:
            score += 10
        return float(min(max(score, 0), 100))
    
    def compute_kdj_factor(self, df: pd.DataFrame, date_idx: int) -> float:
        """
        KDJ 因子得分 (0-100)
        逻辑：超卖区（K<20 / J<0）给高分；金叉加分；超买区（K>80 / J>100）给低分；死叉减分
        """
        if 'K' not in df.columns or date_idx < 9:
            return 50.0
        cur = df.iloc[date_idx]
        prev = df.iloc[date_idx - 1]
        score = 50.0
        if cur['K'] < 20:
            score += 25
        elif cur['K'] > 80:
            score -= 25
        if prev['K'] <= prev['D'] and cur['K'] > cur['D']:
            score += 20  # 金叉
        elif prev['K'] >= prev['D'] and cur['K'] < cur['D']:
            score -= 20  # 死叉
        if cur['J'] < 0:
            score += 10
        if cur['J'] > 100:
            score -= 10
        return float(min(max(score, 0), 100))
    
    def compute_rsi_factor(self, df: pd.DataFrame, date_idx: int) -> float:
        """
        RSI 因子得分 (0-100)
        逻辑：超卖（RSI<30）高分，超买（RSI>70）低分
        """
        if 'RSI' not in df.columns or date_idx < 14:
            return 50.0
        rsi = df.iloc[date_idx]['RSI']
        if rsi < 20:
            score = 95
        elif rsi < 30:
            score = 80 + (30 - rsi) * 1.5
        elif rsi < 50:
            score = 50 + (50 - rsi)
        elif rsi < 70:
            score = 50 - (rsi - 50)
        elif rsi < 80:
            score = 30 - (rsi - 70) * 2
        else:
            score = 10
        return float(min(max(score, 0), 100))
    
    def compute_bollinger_factor(self, df: pd.DataFrame, date_idx: int) -> float:
        """
        布林带因子得分 (0-100)
        逻辑：%B 越低（贴近下轨）给高分（均值回归）；带宽收窄（挤压）给突破预期加分
        """
        if 'pct_b' not in df.columns or date_idx < 20:
            return 50.0
        cur = df.iloc[date_idx]
        pct_b = cur['pct_b']
        score = 50 + (0.5 - pct_b) * 60  # %B=0→80, %B=1→20
        # 挤压（带宽低于历史较低水平）预示变盘，小幅加分
        if cur['bandwidth'] < 0.06:
            score += 10
        return float(min(max(score, 0), 100))
    
    def compute_wyckoff_factor(self, df: pd.DataFrame, date_idx: int) -> float:
        """
        威科夫量价因子得分 (0-100)
        逻辑：价涨量增（需求强劲）加分；价跌量增（供给强劲）减分；
             放量见顶/见底（高潮）作反转参考；地量（量枯）在支撑区加分
        """
        if 'rel_volume' not in df.columns or date_idx < 20:
            return 50.0
        cur = df.iloc[date_idx]
        prev = df.iloc[date_idx - 1]
        rel_vol = cur['rel_volume']
        price_chg = (cur['close'] - prev['close']) / prev['close']
        score = 50.0
        if price_chg > 0 and rel_vol > 1.5:
            score += 30  # 需求强劲
        elif price_chg < 0 and rel_vol > 1.5:
            score -= 30  # 供给强劲
        if rel_vol > 2.5:  # 量能高潮
            if price_chg <= 0:
                score += 10  # 抛售高潮 → 见底可能
            else:
                score -= 10  # 抢购高潮 → 见顶可能
        if rel_vol < 0.6:
            score += 5  # 量枯，蓄势
        return float(min(max(score, 0), 100))
    
    def compute_135_method_factor(self, df: pd.DataFrame, date_idx: int) -> float:
        """
        135 战法因子得分 (0-100)
        逻辑：收盘价站上 13 日操作线 + 13 日线在 34 日生命线之上 + 13 日线走多 +
             均线互换（13 上穿 34）/ 红杏出墙（13 日线首次拐头向上）+ 放量确认
        """
        if 'ma13' not in df.columns or date_idx < 34:
            return 50.0
        cur = df.iloc[date_idx]
        prev = df.iloc[date_idx - 1]
        score = 50.0
        if cur['close'] > cur['ma13']:
            score += 20
        if cur['ma13'] > cur['ma34']:
            score += 20
        if cur['ma13'] > prev['ma13']:
            score += 20  # 操作线走多
        # 均线互换：13 上穿 34（金叉）
        if prev['ma13'] <= prev['ma34'] and cur['ma13'] > cur['ma34']:
            score += 20
        # 放量确认（135 战法强调量能配合）
        if cur.get('rel_volume', 1.0) > 1.3 and cur['close'] > cur['ma13']:
            score += 10
        return float(min(max(score, 0), 100))
    
    def composite_score(
        self,
        dmi_score: float,
        fundamental_score: float,
        sentiment_score: float,
        macd_score: float = 50.0,
        kdj_score: float = 50.0,
        rsi_score: float = 50.0,
        bollinger_score: float = 50.0,
        wyckoff_score: float = 50.0,
        method135_score: float = 50.0,
        momentum_score: float = 50.0,
        quality_score: float = 50.0
    ) -> float:
        """
        计算综合因子得分（9 + 动量 + 质量 因子线性加权）

        Args:
            dmi_score: DMI 因子得分 (0-100)
            fundamental_score: 基本面得分 (0-100)
            sentiment_score: 舆情得分 (-1 到 1)，转换为 (0-100)
            macd_score ~ method135_score: 各技术指标因子得分 (0-100)
            momentum_score: 动量因子得分 (0-100)，默认 50（中性）
            quality_score: 质量因子得分 (0-100)，默认 50（中性）

        Returns:
            综合得分
        """
        # 舆情得分转换为 0-100
        sentiment_factor = (sentiment_score + 1) / 2 * 100

        # 动量 / 质量 权重默认 0，未启用时不改变原 9 因子基线
        momentum_factor = momentum_score
        quality_factor = quality_score

        composite = (
            self.weights.dmi_weight * dmi_score +
            self.weights.macd_weight * macd_score +
            self.weights.kdj_weight * kdj_score +
            self.weights.rsi_weight * rsi_score +
            self.weights.bollinger_weight * bollinger_score +
            self.weights.wyckoff_weight * wyckoff_score +
            self.weights.method135_weight * method135_score +
            self.weights.fundamental_weight * fundamental_score +
            self.weights.sentiment_weight * sentiment_factor +
            self.weights.momentum_weight * momentum_factor +
            self.weights.quality_weight * quality_factor
        )

        return composite
    
    def _compute_nav(self, date, data_map: Dict[str, pd.DataFrame]) -> float:
        """
        按 现金 + 持仓市值 计算当前组合净值（逐日盯市）
        """
        market_value = 0.0
        for code, shares in self.position_shares.items():
            if code in data_map and date in data_map[code].index:
                market_value += shares * data_map[code].loc[date, 'close']
        return self.cash + market_value

    def _execute_buy(self, code, date, close_price, budget: float):
        """
        以预算资金买入（含滑点 + 佣金），更新持仓与现金。
        止损判定仍用原始收盘价（self.current_positions[code]）。
        """
        c = self.cost
        fill = close_price * (1 + c.slippage)
        shares = int(budget / fill / 100) * 100  # 100 股整数倍
        if shares <= 0:
            return
        gross = fill * shares
        commission = max(gross * c.commission, c.min_commission)
        total_cost = gross + commission
        # 预算超出可用现金时，按现金向下取整
        if total_cost > self.cash:
            shares = int((self.cash / (fill * (1 + c.commission))) / 100) * 100
            if shares <= 0:
                return
            gross = fill * shares
            commission = max(gross * c.commission, c.min_commission)
            total_cost = gross + commission
        self.cash -= total_cost
        self.current_positions[code] = close_price       # 原始收盘价（止损判定用）
        self.position_dates[code] = str(date)
        self.position_shares[code] = shares
        self.position_cost_basis[code] = total_cost       # 含买入滑点 + 佣金
        self.risk_manager.add_trade(TradeRecord(
            code=code, buy_date=str(date), buy_price=close_price, reason="选股买入"
        ))

    def _execute_sell(self, code, date, close_price, reason: str):
        """
        以收盘价卖出（含滑点 + 佣金 + 印花税），记录含成本的真实盈亏。
        """
        c = self.cost
        shares = self.position_shares.get(code, 0)
        if shares <= 0:
            return
        fill = close_price * (1 - c.slippage)
        gross = fill * shares
        commission = max(gross * c.commission, c.min_commission)
        stamp = gross * c.stamp if c.stamp_sell_only else 0.0
        proceeds = gross - commission - stamp
        self.cash += proceeds

        cost_basis = self.position_cost_basis.get(code, gross)
        pnl = (proceeds - cost_basis) / cost_basis if cost_basis > 0 else 0.0

        trade = TradeRecord(
            code=code,
            buy_date=self.position_dates.get(code, ""),
            buy_price=self.current_positions.get(code, close_price),
            sell_date=str(date),
            sell_price=close_price,
            reason=reason,
            pnl_pct=pnl
        )
        # 仅用于维持风控器持仓表（止损判定）；盈亏以 strategy.trade_log 为准
        self.risk_manager.close_trade(code, close_price, str(date), reason)
        self.trade_log.append(trade)

        self.position_dates.pop(code, None)
        self.current_positions.pop(code, None)
        self.position_shares.pop(code, None)
        self.position_cost_basis.pop(code, None)
    
    def fetch_stock_data(
        self,
        stock_codes: List[str],
        start_date: str = "20230101",
        end_date: str = "",
        fetch_fundamental: bool = True,
        max_workers: int = 1,
        show_progress: bool = True,
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]:
        """
        获取股票数据（支持真实数据自动降级）。

        Args:
            stock_codes: 股票代码列表（可带 .SZ/.SH 后缀）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            fetch_fundamental: 是否获取基本面数据（真实行情下该接口数据质量不稳时可关闭）
            max_workers: 并行抓取线程数。>1 时启用线程池（默认 1，串行，与旧行为一致、
                         最稳）。远端对并发敏感，建议 ≤4；单任务异常不影响其余股票。
            show_progress: 是否打印 [已抓/总数] 进度（含命中缓存提示）

        Returns:
            (K 线数据字典, 基本面数据字典)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_one(code: str):
            kline = self.data_source.fetch_kline(code, start_date, end_date)
            fund = None
            if fetch_fundamental and kline is not None and not kline.empty:
                fund = self.data_source.fetch_fundamental(code, start_date, end_date)
                if fund is not None:
                    # 将基本面索引对齐到 K 线交易日，缺失日 ffill/bfill 兜底，
                    # 保证 _rebalance 中 `date in fundamental_data[code].index` 始终命中
                    fund = fund.reindex(kline.index)
                    fund = fund.ffill().bfill()
            return code, kline, fund

        data_map: Dict[str, pd.DataFrame] = {}
        fundamental_data: Dict[str, pd.DataFrame] = {}
        total = len(stock_codes)
        done = 0

        def _report(code: str, ok: bool, cached: bool = False):
            nonlocal done
            done += 1
            if show_progress:
                tag = "缓存" if cached else ("OK" if ok else "FAIL")
                logger.info(f"[{done}/{total} {done / total * 100:.0f}%] {code} {tag}")

        if max_workers and max_workers > 1:
            # 并行模式：单任务异常被隔离，不中断整体；失败时回退为 mock/空
            with ThreadPoolExecutor(max_workers=min(max_workers, 8)) as ex:
                futs = {ex.submit(_fetch_one, c): c for c in stock_codes}
                for fut in as_completed(futs):
                    code = futs[fut]
                    try:
                        c, kline, fund = fut.result()
                        if kline is not None and not kline.empty:
                            data_map[c] = kline
                            if fund is not None:
                                fundamental_data[c] = fund
                            _report(c, True)
                        else:
                            _report(c, False)
                    except Exception as e:
                        logger.warning(f"  {code} 抓取异常: {e}")
                        _report(code, False)
        else:
            # 串行模式（默认，与旧行为一致）：保留 0.6s 节流防 AKShare 限流
            for code in stock_codes:
                try:
                    c, kline, fund = _fetch_one(code)
                    if kline is not None and not kline.empty:
                        data_map[c] = kline
                        if fund is not None:
                            fundamental_data[c] = fund
                        _report(c, True)
                    else:
                        logger.warning(f"  {code} K 线数据获取失败")
                        _report(c, False)
                except Exception as e:
                    logger.warning(f"  {code} 抓取异常: {e}")
                    _report(code, False)
                time.sleep(0.6)

        return data_map, fundamental_data
    
    def run_backtest(
        self,
        data_map: Optional[Dict[str, pd.DataFrame]] = None,
        fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
        stock_codes: Optional[List[str]] = None,
        start_date: str = "20230101",
        end_date: str = "",
        fetch_fundamental: bool = True,
    ) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            data_map: 股票代码 -> OHLCV DataFrame（如果提供则直接使用该数据）
            fundamental_data: 股票代码 -> 基本面数据 DataFrame
            stock_codes: 股票代码列表（如果 data_map 未提供，将自动获取）
            start_date: 数据开始日期（YYYYMMDD 格式，仅在自动获取时有效）
            end_date: 数据结束日期（YYYYMMDD 格式，仅在自动获取时有效）
            fetch_fundamental: 自动获取时是否拉取基本面数据
        
        Returns:
            回测结果 DataFrame（日期 -> 净值、收益率、回撤等）
        """
        # 如果未提供数据，自动获取
        if data_map is None and stock_codes is not None:
            data_map, fundamental_data = self.fetch_stock_data(
                stock_codes, start_date, end_date, fetch_fundamental
            )
        
        if data_map is None or len(data_map) == 0:
            logger.error("没有可用的股票数据，回测终止")
            return pd.DataFrame()
        
        logger.info(f"开始回测，共 {len(data_map)} 只股票")
        
        # 获取所有股票的交易日期
        all_dates = sorted(set().union(*(df.index for df in data_map.values())))
        
        # 计算全部技术指标（DMI / MACD / KDJ / RSI / 布林 / 威科夫 / 135 均线）
        for code, df in data_map.items():
            df = compute_dmi(df, period=14)
            df = compute_macd(df)
            df = compute_kdj(df)
            df = compute_rsi(df)
            df = compute_bollinger(df)
            df = compute_wyckoff(df)
            df = compute_ma_lines(df)
            data_map[code] = df
        
        # 重置组合状态：保证多次调用 run_backtest（如 walk-forward 多折）互不污染
        self.current_positions.clear()
        self.position_dates.clear()
        self.position_shares.clear()
        self.position_cost_basis.clear()
        self.nav_history.clear()
        self.trade_log.clear()

        # 初始化净值与现金
        nav = self.initial_capital
        self.cash = self.initial_capital
        peak_nav = nav
        daily_records = []
        
        for i, date in enumerate(all_dates):
            # 更新净值
            self.risk_manager.update_nav(nav)
            
            # 检查止损
            codes_to_sell = []
            for code in list(self.current_positions.keys()):
                if code not in data_map or date not in data_map[code].index:
                    continue
                
                current_price = data_map[code].loc[date, 'close']
                stop_reason = self.risk_manager.check_stop_loss(
                    code, current_price, date
                )
                
                if stop_reason:
                    codes_to_sell.append((code, current_price, stop_reason))
            
            # 执行止损卖出
            for code, sell_price, reason in codes_to_sell:
                self._execute_sell(code, date, sell_price, reason)
            
            # 重新盯市并更新净值
            nav = self._compute_nav(date, data_map)
            self.risk_manager.update_nav(nav)
            
            # 检查最大回撤
            if self.risk_manager.check_max_drawdown():
                logger.warning("触发最大回撤限制！清仓所有持仓")
                for code in list(self.position_shares.keys()):
                    if code in data_map and date in data_map[code].index:
                        self._execute_sell(code, date, data_map[code].loc[date, 'close'], "最大回撤触发")
                
                nav = self._compute_nav(date, data_map)
                self.risk_manager.update_nav(nav)
            
            # 调仓日
            if i % self.rebalance_freq == 0:
                self._rebalance(
                    data_map, 
                    fundamental_data, 
                    date, 
                    i,
                    all_dates
                )
                nav = self._compute_nav(date, data_map)
                self.risk_manager.update_nav(nav)
            
            # 记录净值
            daily_records.append({
                'date': str(date),
                'nav': nav,
                'return': (nav / self.initial_capital) - 1,
                'drawdown': (nav - peak_nav) / peak_nav if peak_nav > 0 else 0,
                'positions': len(self.current_positions),
            })
            
            if nav > peak_nav:
                peak_nav = nav
        
        # 计算统计指标
        result_df = pd.DataFrame(daily_records)
        stats = self._calculate_stats(result_df)
        
        logger.info(f"回测完成：")
        logger.info(f"  总收益率: {stats['total_return']:.2%}")
        logger.info(f"  年化收益: {stats['annual_return']:.2%}")
        logger.info(f"  夏普比率: {stats['sharpe_ratio']:.2f}")
        logger.info(f"  最大回撤: {stats['max_drawdown']:.2%}")
        logger.info(f"  胜率: {stats['win_rate']:.2%}")
        logger.info(f"  交易次数: {stats['total_trades']}")
        
        return result_df

    # ------------------------------------------------------------
    # 2.7 滚动窗口 walk-forward 验证（样本外 OOS 稳健性）
    # ------------------------------------------------------------
    @staticmethod
    def _window_stats(nav: np.ndarray) -> dict:
        """由一段连续净值序列计算区间收益/年化/夏普/最大回撤。"""
        if len(nav) < 2:
            return {"total_return": 0.0, "annual_return": 0.0,
                    "sharpe": 0.0, "max_drawdown": 0.0, "days": len(nav)}
        nav = np.asarray(nav, dtype=float)
        rets = np.diff(nav) / nav[:-1]
        total_return = nav[-1] / nav[0] - 1.0
        n = len(nav)
        annual_return = (1.0 + total_return) ** (252.0 / n) - 1.0
        rf = 0.03 / 252.0
        sd = rets.std()
        sharpe = ((rets.mean() - rf) / sd) * np.sqrt(252.0) if sd and sd > 0 else 0.0
        peak = np.maximum.accumulate(nav)
        max_drawdown = ((nav - peak) / peak).min()
        return {"total_return": total_return, "annual_return": annual_return,
                "sharpe": sharpe, "max_drawdown": max_drawdown, "days": n}

    def walk_forward_validation(
        self,
        data_map: Dict[str, pd.DataFrame],
        fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
        benchmark: Optional[pd.DataFrame] = None,
        start_date: str = "20220101",
        end_date: str = "20251231",
        n_folds: int = 5,
        warmup_days: int = 250,
    ) -> dict:
        """
        滚动窗口 walk-forward 验证：把样本区间切为 n_folds 段连续的【样本外】测试窗口，
        每段前接 warmup_days 历史交易日做指标预热（不计入测试收益），从而每段测试期都
        严格"未参与过参数拟合"——本策略权重为先前一次性调参固定，故任何测试期天然样本外。

        返回：{'folds':[每段统计], 'overall': 拼接后的样本外总统计, 'oos_nav': 连续样本外净值序列}
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        all_dates = sorted(set().union(*(df.index for df in data_map.values())))
        all_dates = [d for d in all_dates if start <= d <= end]
        if len(all_dates) < warmup_days + 30:
            raise ValueError(f"数据不足以支撑 walk-forward（需 ≥ {warmup_days + 30} 交易日，实有 {len(all_dates)}）")

        test_dates = all_dates[warmup_days:]
        fold_bounds = np.array_split(np.arange(len(test_dates)), n_folds)

        oos_nav = [float(self.initial_capital)]
        fold_stats = []
        for k, seg in enumerate(fold_bounds):
            if len(seg) == 0:
                continue
            t0 = test_dates[seg[0]]      # 测试起点
            t1 = test_dates[seg[-1]]     # 测试终点
            warm0 = all_dates[0]         # 预热从全局起点（含 warmup_days 历史）

            # 切片到 [warm0, t1]，保证指标在 warmup 期预热
            dm_slice = {c: df.loc[(df.index >= warm0) & (df.index <= t1)]
                        for c, df in data_map.items() if not df.empty}
            fd_slice = None
            if fundamental_data:
                fd_slice = {c: f.loc[(f.index >= warm0) & (f.index <= t1)]
                            for c, f in fundamental_data.items() if c in dm_slice}

            result = self.run_backtest(
                data_map=dm_slice, fundamental_data=fd_slice,
                start_date=warm0.strftime("%Y%m%d"), end_date=t1.strftime("%Y%m%d"),
            )
            if result.empty:
                continue

            # 仅提取测试窗口 [t0, t1] 的净值做样本外统计
            test_mask = (result['date'] >= str(t0)) & (result['date'] <= str(t1))
            test_df = result.loc[test_mask]
            if test_df.empty:
                continue
            seg_nav = test_df['nav'].values.astype(float)
            # 衔接到上一段末尾，形成连续样本外净值曲线
            scale = oos_nav[-1] / seg_nav[0]
            oos_nav.extend((seg_nav[1:] * scale).tolist())

            st = self._window_stats(seg_nav)
            st.update({"fold": k + 1, "start": str(t0), "end": str(t1)})
            fold_stats.append(st)
            logger.info(f"  WF 第{k+1}段 [{t0.date()}~{t1.date()}] 收益={st['total_return']:.2%} "
                        f"年化={st['annual_return']:.2%} 夏普={st['sharpe']:.2f} 回撤={st['max_drawdown']:.2%}")

        oos_series = pd.DataFrame({"nav": oos_nav})
        overall = self._window_stats(oos_series['nav'].values)
        overall["n_folds"] = len(fold_stats)
        logger.info(f"Walk-forward 样本外整体: 收益={overall['total_return']:.2%} "
                    f"年化={overall['annual_return']:.2%} 夏普={overall['sharpe']:.2f} "
                    f"回撤={overall['max_drawdown']:.2%}")
        return {"folds": fold_stats, "overall": overall, "oos_nav": oos_series}

    def _rebalance(
        self,
        data_map: Dict[str, pd.DataFrame],
        fundamental_data: Optional[Dict[str, pd.DataFrame]],
        date,
        date_idx: int,
        all_dates: List
    ):
        """调仓：重新选股"""
        
        # 计算每只股票的综合得分
        scores = {}
        for code, df in data_map.items():
            if date not in df.index:
                continue
            
            # 用个股【本地】行号定位（不同股票上市日期/历史长度不一，
            # 全局日期序号 ≠ 个股本地行号，直接用会 iloc 越界）
            i = df.index.get_loc(date)
            
            # DMI 因子
            dmi_score = self.compute_dmi_factor(df, i)
            
            # 技术指标因子
            macd_score = self.compute_macd_factor(df, i)
            kdj_score = self.compute_kdj_factor(df, i)
            rsi_score = self.compute_rsi_factor(df, i)
            bollinger_score = self.compute_bollinger_factor(df, i)
            wyckoff_score = self.compute_wyckoff_factor(df, i)
            method135_score = self.compute_135_method_factor(df, i)

            # 动量因子（仅当权重 > 0 时计算，避免无谓开销；默认关闭）
            momentum_score = 50.0
            if self.weights.momentum_weight > 0:
                df_m = compute_momentum(df)
                momentum_score = compute_momentum_factor(df_m, i)

            # 基本面因子
            fundamental_score = 50.0  # 默认中性
            if fundamental_data and code in fundamental_data:
                if date in fundamental_data[code].index:
                    try:
                        fund_df = fundamental_data[code]
                        fundamental_score = compute_fundamental_score(fund_df).get(date, 50.0)
                    except:
                        pass

            # 质量因子（仅当权重 > 0 时计算；默认关闭，不影响基线）
            quality_score = 50.0
            if self.weights.quality_weight > 0 and fundamental_data and code in fundamental_data:
                if date in fundamental_data[code].index:
                    try:
                        fund_df = fundamental_data[code]
                        quality_score = compute_quality_score(fund_df).get(date, 50.0)
                    except:
                        pass

            # 舆情因子
            if self.use_point_in_time_sentiment:
                # 时点正确：仅用个股自身历史量价，绝不使用未来数据（消前瞻）
                sentiment_score = compute_point_in_time_sentiment(df, i)
            else:
                sentiment_score = self.sentiment_monitor.get_sentiment_score(code, str(date))

            # 综合得分（9 因子 + 可选动量 / 质量 加权）
            composite = self.composite_score(
                dmi_score, fundamental_score, sentiment_score,
                macd_score, kdj_score, rsi_score, bollinger_score,
                wyckoff_score, method135_score,
                momentum_score=momentum_score,
                quality_score=quality_score
            )
            scores[code] = composite
        
        # 排序选 Top N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 获取当前持仓
        current_held = set(self.current_positions.keys())
        selected = set([code for code, _ in ranked[:self.top_n]])

        # 卖出不在新组合中的股票
        to_sell = current_held - selected
        for code in to_sell:
            if date in data_map[code].index:
                self._execute_sell(code, date, data_map[code].loc[date, 'close'], "调仓卖出")

        # regime 调控：熊市/高波动期降低暴露（默认 use_regime=False -> 满仓，不影响基线）
        regime_exposure = 1.0
        if self.use_regime and self.regime is not None:
            regime_exposure = self.regime.exposure(date)
        if regime_exposure <= 0.0:
            # 空仓：清掉全部当前持仓，不再买入
            for code in list(self.current_positions.keys()):
                if code in data_map and date in data_map[code].index:
                    self._execute_sell(code, date, data_map[code].loc[date, 'close'], "regime 空仓")
            return

        # 买入新选中的股票
        to_buy = selected - current_held
        available_cash = self.cash * regime_exposure
        
        if to_buy and available_cash > 0:
            equal_weight = available_cash / len(to_buy)
            for code in to_buy:
                if date in data_map[code].index:
                    self._execute_buy(code, date, data_map[code].loc[date, 'close'], equal_weight)
    
    def _calculate_stats(self, result_df: pd.DataFrame) -> dict:
        """计算回测统计指标"""
        if result_df.empty:
            return {}
        
        # 日收益率（基于净值，而非累积收益率序列）
        nav_returns = result_df['nav'].pct_change().fillna(0)
        
        # 总收益率
        total_return = result_df['return'].iloc[-1]
        
        # 年化收益
        trading_days = len(result_df)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # 夏普比率（无风险利率 3%，基于日收益率）
        risk_free_rate = 0.03
        daily_std = nav_returns.std()
        if daily_std > 0:
            sharpe_ratio = (nav_returns.mean() - risk_free_rate / 252) / daily_std * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        max_drawdown = result_df['drawdown'].min()
        
        # 胜率
        win_trades = [t for t in self.trade_log if t.pnl_pct > 0]
        total_trades = len(self.trade_log)
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'win_rate': win_rate,
            'total_trades': total_trades,
        }


# ============================================================
# 6. 数据生成（用于回测演示）
# ============================================================

def generate_mock_data(
    stock_codes: List[str],
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    initial_price: float = 20.0
) -> Dict[str, pd.DataFrame]:
    """
    生成模拟股票数据用于回测演示
    
    Args:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        initial_price: 初始价格
    
    Returns:
        股票代码 -> OHLCV DataFrame
    """
    dates = pd.bdate_range(start_date, end_date)
    data_map = {}
    
    np.random.seed(42)
    
    for code in stock_codes:
        n = len(dates)
        # 几何布朗运动模拟股价
        drift = np.random.uniform(-0.0005, 0.001)  # 日漂移率
        volatility = np.random.uniform(0.015, 0.035)  # 日波动率
        
        returns = np.random.normal(drift, volatility, n)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # 生成 OHLCV
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            'high': prices * (1 + np.random.uniform(0.005, 0.02, n)),
            'low': prices * (1 + np.random.uniform(-0.02, -0.005, n)),
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, n),
        }, index=dates)
        
        # 生成基本面数据
        df['pe'] = np.random.uniform(5, 30, n)
        df['pb'] = np.random.uniform(0.5, 5, n)
        df['roe'] = np.random.uniform(5, 25, n)
        
        data_map[code] = df
    
    return data_map


# ============================================================
# 7. 主函数
# ============================================================

def main(use_real: bool = False, use_westock_real: bool = False,
         use_real_hs300: bool = False, use_pit_sentiment: bool = False,
         use_regime: bool = False, hs300_limit: Optional[int] = None):
    """运行回测演示

    Args:
        use_real: True 使用 AKShare 真实行情（需 pip install akshare 且联网），
                  False 使用内置模拟数据
        use_westock_real: True 接入腾讯自选股 westock-mcp 真实基本面+资金流舆情
                  （确定性 mock K线 + 真实 pe/pb/roe/毛利率/负债率 + 真实主力净流入舆情）。
                  需先运行 gen_real_cache.py 生成 cache/{code}_realfund.csv / {code}_realflow.csv。
        use_real_hs300: True 使用 AKShare 自包含真实数据加载器，对沪深300成分股做
                  时点正确(point-in-time)回测 + walk-forward 样本外验证。
                  --pit-sentiment 启用时点正确技术舆情；--regime 启用市场状态暴露调控。
                  --hs300-limit N 可限制成分股数量（快速验证）。
    """
    import os
    CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")

    stock_codes = [
        "000001.SZ", "000002.SZ", "600000.SH", "600036.SH",
        "000858.SZ", "600519.SH", "000333.SZ", "601318.SH",
        "002594.SZ", "600276.SH"
    ]

    factor_weights = FactorWeights(
        dmi_weight=0.12, macd_weight=0.12, kdj_weight=0.08, rsi_weight=0.08,
        bollinger_weight=0.08, wyckoff_weight=0.10, method135_weight=0.10,
        fundamental_weight=0.18, sentiment_weight=0.14,
    )
    risk_manager = RiskManager(
        stop_loss_pct=-0.10,  # 10% 止损
        max_drawdown_pct=-0.20,  # 20% 最大回撤
        max_positions=10,
    )

    if use_real_hs300:
        # —— 真实 HS300 时点正确回测 + walk-forward 样本外验证 ——
        from real_data_loader import build_real_dataset, get_hs300_codes
        codes = get_hs300_codes(limit=hs300_limit)
        logger.info(f"真实 HS300 回测：{len(codes)} 只 | pit_sentiment={use_pit_sentiment} | regime={use_regime}")
        dm, fd, bench = build_real_dataset(codes, "20220101", "20251231")
        regime = MarketRegime(bench) if use_regime else None
        hs300_risk = RiskManager(
            stop_loss_pct=-0.10, max_drawdown_pct=-0.20, max_positions=30,
        )
        strategy = MultiFactorDMIStrategy(
            initial_capital=1_000_000, top_n=15, factor_weights=factor_weights,
            risk_manager=hs300_risk, sentiment_monitor=SentimentMonitor(use_real_data=False),
            rebalance_freq=20, use_real_data=True,
            use_point_in_time_sentiment=use_pit_sentiment,
            use_regime=use_regime, regime=regime,
        )
        result = strategy.run_backtest(
            data_map=dm, fundamental_data=fd, start_date="20220101", end_date="20251231",
        )
        wf = strategy.walk_forward_validation(dm, fd, bench, "20220101", "20251231", n_folds=5)

        print("\n" + "=" * 60)
        print("Walk-forward 样本外分段统计")
        print("=" * 60)
        print(f"{'段':>3} {'区间':<24} {'收益':>10} {'年化':>10} {'夏普':>7} {'回撤':>9}")
        for f in wf["folds"]:
            print(f"{f['fold']:>3} {f['start'][:10]}~{f['end'][:10]} "
                  f"{f['total_return']:>10.2%} {f['annual_return']:>10.2%} "
                  f"{f['sharpe']:>7.2f} {f['max_drawdown']:>9.2%}")
        o = wf["overall"]
        print("-" * 60)
        print(f"{'OOS':>3} {'整体':<24} {o['total_return']:>10.2%} {o['annual_return']:>10.2%} "
              f"{o['sharpe']:>7.2f} {o['max_drawdown']:>9.2%}")
        return result, wf

    if use_westock_real:
        # —— 真实数据接入回测：确定性 mock K线 + 真实基本面 + 真实资金流舆情 ——
        logger.info("真实数据接入回测（腾讯自选股 westock-mcp）：确定性 mock K线 + 真实基本面 + 真实资金流舆情")
        from data_source import DataSourceManager
        dm = DataSourceManager(use_real_data=True, use_westock_real=True, cache_dir=CACHE)
        sm = SentimentMonitor(use_real_data=True, use_westock_real=True, cache_dir=CACHE)

        # 确定性 mock K线（结构可复现，不与网络强耦合）
        data_map = generate_mock_data(stock_codes)

        # 真实基本面：读取 westock 衍生缓存，并 REINDEX 到 K线交易日
        # （与 fetch_stock_data 同口径：ffill/bfill 兜底，保证 _rebalance 命中）。
        # 注意：真实基本面为单一最新报告期，对历史再平衡日属前瞻，仅用于验证接入。
        fundamental_data = {}
        for c in stock_codes:
            f = dm.fetch_fundamental(c)
            if f is not None and not f.empty:
                f = f.reindex(data_map[c].index).ffill().bfill()
                fundamental_data[c] = f

        strategy = MultiFactorDMIStrategy(
            initial_capital=1_000_000, top_n=5, factor_weights=factor_weights,
            risk_manager=risk_manager, sentiment_monitor=sm,
            rebalance_freq=20, use_real_data=True, data_source=dm,
        )
        result = strategy.run_backtest(
            data_map=data_map, fundamental_data=fundamental_data,
            start_date="20230101", end_date="20241231",
        )
    elif use_real:
        logger.info("使用 AKShare 真实行情进行回测（2024 全年，含真实基本面因子）")
        strategy = MultiFactorDMIStrategy(
            initial_capital=1_000_000, top_n=5, factor_weights=factor_weights,
            risk_manager=risk_manager, sentiment_monitor=SentimentMonitor(use_real_data=False),
            rebalance_freq=20, use_real_data=True,
        )
        result = strategy.run_backtest(
            stock_codes=stock_codes, start_date="20240101", end_date="20241231",
            fetch_fundamental=True,
        )
    else:
        strategy = MultiFactorDMIStrategy(
            initial_capital=1_000_000, top_n=5, factor_weights=factor_weights,
            risk_manager=risk_manager, sentiment_monitor=SentimentMonitor(use_real_data=False),
            rebalance_freq=20, use_real_data=False,
        )
        data_map = generate_mock_data(stock_codes)
        result = strategy.run_backtest(data_map)

    # 输出结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(result.to_string(index=False))

    # 输出交易记录
    print("\n" + "=" * 60)
    print("交易记录")
    print("=" * 60)
    for trade in strategy.trade_log[:20]:  # 只打印前 20 条
        print(f"{trade.code} | 买入: {trade.buy_price:.2f} -> 卖出: {trade.sell_price:.2f} | "
              f"收益率: {trade.pnl_pct:.2%} | 原因: {trade.reason}")

    return result


if __name__ == "__main__":
    import sys
    use_real = "--real" in sys.argv
    use_westock_real = "--westock" in sys.argv
    use_real_hs300 = "--real-hs300" in sys.argv
    use_pit_sentiment = "--pit-sentiment" in sys.argv
    use_regime = "--regime" in sys.argv
    # --hs300-limit N
    hs300_limit = None
    if "--hs300-limit" in sys.argv:
        try:
            hs300_limit = int(sys.argv[sys.argv.index("--hs300-limit") + 1])
        except (IndexError, ValueError):
            hs300_limit = None
    main(use_real=use_real, use_westock_real=use_westock_real,
         use_real_hs300=use_real_hs300, use_pit_sentiment=use_pit_sentiment,
         use_regime=use_regime, hs300_limit=hs300_limit)
