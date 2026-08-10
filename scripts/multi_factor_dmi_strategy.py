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
    
    def __init__(self, use_real_data: bool = False):
        self.use_real_data = use_real_data
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
        if self.use_real_data:
            return self._fetch_real_sentiment(stock_code, date)
        else:
            # 回测时使用模拟数据
            return self._generate_mock_sentiment(stock_code, date)
    
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
        method135_score: float = 50.0
    ) -> float:
        """
        计算综合因子得分（9 因子线性加权）
        
        Args:
            dmi_score: DMI 因子得分 (0-100)
            fundamental_score: 基本面得分 (0-100)
            sentiment_score: 舆情得分 (-1 到 1)，转换为 (0-100)
            macd_score ~ method135_score: 各技术指标因子得分 (0-100)
        
        Returns:
            综合得分
        """
        # 舆情得分转换为 0-100
        sentiment_factor = (sentiment_score + 1) / 2 * 100
        
        composite = (
            self.weights.dmi_weight * dmi_score +
            self.weights.macd_weight * macd_score +
            self.weights.kdj_weight * kdj_score +
            self.weights.rsi_weight * rsi_score +
            self.weights.bollinger_weight * bollinger_score +
            self.weights.wyckoff_weight * wyckoff_score +
            self.weights.method135_weight * method135_score +
            self.weights.fundamental_weight * fundamental_score +
            self.weights.sentiment_weight * sentiment_factor
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
        fetch_fundamental: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]:
        """
        获取股票数据（支持真实数据自动降级）
        
        Args:
            stock_codes: 股票代码列表（可带 .SZ/.SH 后缀）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            fetch_fundamental: 是否获取基本面数据（真实行情下该接口数据质量不稳时可关闭）
        
        Returns:
            (K 线数据字典, 基本面数据字典)
        """
        data_map = {}
        fundamental_data = {}
        
        for code in stock_codes:
            logger.info(f"正在获取 {code} 数据...")
            
            # 获取 K 线
            kline = self.data_source.fetch_kline(code, start_date, end_date)
            if kline is not None and not kline.empty:
                data_map[code] = kline
                logger.info(f"  {code} K 线数据: {len(kline)} 条")
            else:
                logger.warning(f"  {code} K 线数据获取失败")
            
            # 获取基本面数据
            if fetch_fundamental:
                fund = self.data_source.fetch_fundamental(code, start_date, end_date)
                if fund is not None:
                    # 将基本面索引对齐到 K 线交易日，缺失日 ffill/bfill 兜底，
                    # 保证 _rebalance 中 `date in fundamental_data[code].index` 始终命中
                    fund = fund.reindex(kline.index)
                    fund = fund.ffill().bfill()
                    fundamental_data[code] = fund
                    logger.info(f"  {code} 基本面数据: {len(fund)} 条")
            
            # 节流：避免密集请求触发 AKShare 远端限流/断连
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
            
            # 基本面因子
            fundamental_score = 50.0  # 默认中性
            if fundamental_data and code in fundamental_data:
                if date in fundamental_data[code].index:
                    try:
                        fund_df = fundamental_data[code]
                        fundamental_score = compute_fundamental_score(fund_df).get(date, 50.0)
                    except:
                        pass
            
            # 舆情因子
            sentiment_score = self.sentiment_monitor.get_sentiment_score(code, str(date))
            
            # 综合得分（9 因子加权）
            composite = self.composite_score(
                dmi_score, fundamental_score, sentiment_score,
                macd_score, kdj_score, rsi_score, bollinger_score,
                wyckoff_score, method135_score
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
        
        # 买入新选中的股票
        to_buy = selected - current_held
        available_cash = self.cash
        
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

def main(use_real: bool = False):
    """运行回测演示
    
    Args:
        use_real: True 使用 AKShare 真实行情（需 pip install akshare 且联网），
                  False 使用内置模拟数据
    """
    
    stock_codes = [
        "000001.SZ", "000002.SZ", "600000.SH", "600036.SH",
        "000858.SZ", "600519.SH", "000333.SZ", "601318.SH",
        "002594.SZ", "600276.SH"
    ]
    
    # 初始化策略
    strategy = MultiFactorDMIStrategy(
        initial_capital=1_000_000,
        top_n=5,
        factor_weights=FactorWeights(
            dmi_weight=0.12,
            macd_weight=0.12,
            kdj_weight=0.08,
            rsi_weight=0.08,
            bollinger_weight=0.08,
            wyckoff_weight=0.10,
            method135_weight=0.10,
            fundamental_weight=0.18,
            sentiment_weight=0.14
        ),
        risk_manager=RiskManager(
            stop_loss_pct=-0.10,  # 10% 止损
            max_drawdown_pct=-0.20,  # 20% 最大回撤
            max_positions=10,
        ),
        sentiment_monitor=SentimentMonitor(use_real_data=False),
        rebalance_freq=20,
        use_real_data=use_real,
    )
    
    # 运行回测
    if use_real:
        logger.info("使用 AKShare 真实行情进行回测（2024 全年，含真实基本面因子）")
        result = strategy.run_backtest(
            stock_codes=stock_codes,
            start_date="20240101",
            end_date="20241231",
            # 基本面因子接入真实数据：PE/PB 来自 stock_value_em，ROE 来自同花顺财务摘要
            fetch_fundamental=True,
        )
    else:
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
    main(use_real=use_real)
