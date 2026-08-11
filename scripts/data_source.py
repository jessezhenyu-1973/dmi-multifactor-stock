"""
数据源适配层
============

支持多种 A 股数据源，按优先级自动降级：
1. AKShare - 开源免费，覆盖全面
2. 通达信 pytdx - 本地行情数据
3. 腾讯财经 - 备用接口
4. 模拟数据 - 兜底方案

依赖：pip install akshare requests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
import os
import zlib

logger = logging.getLogger(__name__)


def _stable_seed(s: str) -> int:
    """
    跨进程稳定的随机种子。

    Python 内置 hash() 对 str/bytes 默认带随机盐（PYTHONHASHSEED），
    同一段字符串在不同进程里 hash 值不同，不能用于复现性。这里改用
    zlib.crc32（Python3 中保证返回无符号 32 位整数），保证 mock 数据
    跨进程/跨运行完全一致，符合策略「可复现性铁律」。
    """
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


# ============================================================
# 1. AKShare 数据源
# ============================================================

class AKShareDataSource:
    """
    AKShare 数据源
    
    功能：
    - 历史 K 线数据（日线/分钟线）
    - 实时行情（PE/PB/换手率等）
    - 财务数据（ROE/营收/净利润等）
    - 舆情/新闻情绪（部分支持）
    
    文档：https://akshare.akfamily.xyz/
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.ak_available = False
        self.ak = None
        self.cache_dir = cache_dir
        try:
            import akshare as ak
            self.ak = ak
            self.ak_available = True
            logger.info("AKShare 加载成功")
        except ImportError:
            logger.warning("AKShare 未安装，请运行: pip install akshare")

    # ---------- 本地磁盘缓存（让大股票池只抓一次，后续离线回测） ----------
    def _cache_path(self, kind: str, symbol: str, adjust: Optional[str] = None) -> Optional[str]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        # K 线缓存文件名须含复权方式，避免 qfq/hfq 串味、缓存命中短路不重抓
        if kind == "kline" and adjust:
            return os.path.join(self.cache_dir, f"{symbol}_{adjust}_{kind}.csv")
        return os.path.join(self.cache_dir, f"{symbol}_{kind}.csv")

    def _load_cache(self, kind: str, symbol: str, adjust: Optional[str] = None) -> Optional[pd.DataFrame]:
        p = self._cache_path(kind, symbol, adjust)
        if p and os.path.exists(p):
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"读取缓存失败 {p}: {e}")
        return None

    def _save_cache(self, kind: str, symbol: str, df: Optional[pd.DataFrame], adjust: Optional[str] = None) -> None:
        p = self._cache_path(kind, symbol, adjust)
        if p and df is not None and not df.empty:
            try:
                df.to_csv(p)
            except Exception as e:
                logger.warning(f"写入缓存失败 {p}: {e}")

    def _call_ak(self, func, *args, retries: int = 4, delay: float = 1.5, **kwargs):
        """
        带重试/退避的 AKShare 调用，缓解远端连接中断与限流
        （东财接口对密集请求会 RemoteDisconnected）
        """
        last_exc = None
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                logger.warning(f"AKShare 调用失败 (第 {attempt + 1}/{retries} 次): {e}")
                time.sleep(delay * (attempt + 1))
        raise last_exc
    
    def fetch_kline(
        self,
        stock_code: str,
        start_date: str = "20230101",
        end_date: str = "",
        period: str = "daily",
        adjust: str = "hfq"
    ) -> Optional[pd.DataFrame]:
        """
        获取个股 K 线数据（腾讯接口为主，东财接口兜底）
        
        Args:
            stock_code: 股票代码（可带 .SZ/.SH 后缀）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            period: K 线周期 'daily'/'weekly'/'monthly'（腾讯接口仅日线）
            adjust: 复权方式 'hfq'(后复权，跨运行稳定，推荐用于回测)/'qfq'(前复权，锚定最新价会漂移)/''(不复权)
        
        Returns:
            DataFrame 包含 columns: open, high, low, close, volume（日期为索引）
        """
        # 去除交易所后缀（AKShare 需要纯数字代码，如 000001）
        symbol = stock_code.split('.')[0]

        # 优先读本地缓存（命中则跳过网络，避免大股票池重复抓取被限流）
        cached = self._load_cache('kline', symbol, adjust)
        if cached is not None:
            cached = cached[cached.index >= pd.Timestamp(start_date)]
            if end_date:
                cached = cached[cached.index <= pd.Timestamp(end_date)]
            if not cached.empty:
                logger.info(f"缓存命中 K 线({adjust}): {stock_code}, 共 {len(cached)} 条")
                return cached

        if not self.ak_available:
            return None
        
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        
        df = None
        # 主源：腾讯接口（仅支持日线；周/月线腾讯无对应接口，直接走东财兜底）
        tcode = ('sh' if symbol.startswith('6') else 'sz') + symbol
        if period == "daily":
            try:
                df = self._call_ak(self.ak.stock_zh_a_daily, symbol=tcode, adjust=adjust)
            except Exception as e:
                logger.warning(f"腾讯接口获取 {stock_code} 失败: {e}")
        
        # 兜底：东财接口
        if df is None or df.empty:
            try:
                df = self._call_ak(
                    self.ak.stock_zh_a_hist,
                    symbol=symbol,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
            except Exception as e:
                logger.warning(f"东财接口获取 {stock_code} 失败: {e}")
        
        if df is None or df.empty:
            return None
        
        # 归一化：日期索引 + 仅保留 OHLCV
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['date'] <= pd.Timestamp(end_date)]
        df.set_index('date', inplace=True)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available = [c for c in required_cols if c in df.columns]
        if not available:
            return None
        df = df[available]
        self._save_cache('kline', symbol, df, adjust)
        return df
    
    def fetch_fundamental(
        self,
        stock_code: str,
        start_date: str = "",
        end_date: str = ""
    ) -> Optional[pd.DataFrame]:
        """
        获取个股财务数据（PE/PB/ROE 等）—— 使用可靠的历史估值源
        
        数据源选择（已实地验证主机可解析、数值合理）：
        - PE/PB：东财历史逐日估值接口 stock_value_em（返回 数据日期/PE(TTM)/市净率，
                  自 2018 起逐交易日，数值可靠；原 stock_a_indicator_lg 在本 akshare
                  版本数值错乱，已弃用）
        - ROE：同花顺财务摘要 stock_financial_abstract_ths（按报告期，含 净资产收益率），
               回填（ffill）到每个交易日，实现 point-in-time 语义
        
        Args:
            stock_code: 股票代码（可带 .SZ/.SH 后缀）
            start_date: 开始日期 YYYYMMDD（可选，用于裁剪）
            end_date: 结束日期 YYYYMMDD（可选，用于裁剪）
        
        Returns:
            DataFrame 索引为日期（Timestamp），含 pe, pb, roe 列；失败返回 None
        """
        # 去除交易所后缀（AKShare 需要纯数字代码）
        symbol = stock_code.split('.')[0]

        # 优先读本地缓存（命中则跳过网络）
        cached = self._load_cache('fund', symbol)
        if cached is not None:
            if start_date:
                cached = cached[cached.index >= pd.Timestamp(start_date)]
            if end_date:
                cached = cached[cached.index <= pd.Timestamp(end_date)]
            if not cached.empty:
                logger.info(f"缓存命中基本面: {stock_code}, 共 {len(cached)} 条")
                return cached

        if not self.ak_available:
            return None

        try:
            # 1) 历史逐日 PE/PB（东财估值接口，数值可靠）
            val = self._call_ak(self.ak.stock_value_em, symbol=symbol)
            if val is None or val.empty:
                logger.warning(f"{stock_code} 估值接口返回空")
                return None

            val['date'] = pd.to_datetime(val['数据日期'])
            val = val.rename(columns={'PE(TTM)': 'pe', '市净率': 'pb'})
            val = val.set_index('date')[['pe', 'pb']]
            val = val[~val.index.duplicated(keep='last')]

            # 2) ROE（同花顺财务摘要，按报告期）—— 独立 try，失败不阻断 PE/PB
            roe = None
            try:
                ths = self._call_ak(
                    self.ak.stock_financial_abstract_ths,
                    symbol=symbol,
                    indicator="按报告期"
                )
                if ths is not None and not ths.empty and '净资产收益率' in ths.columns:
                    ths['报告期'] = pd.to_datetime(ths['报告期'])
                    roe = ths.set_index('报告期')['净资产收益率']
                    roe = pd.to_numeric(roe, errors='coerce')
                    roe = roe[~roe.index.duplicated(keep='last')]
                    logger.info(f"{stock_code} ROE 报告期数: {len(roe)}")
            except Exception as e:
                logger.warning(f"{stock_code} ROE 获取失败（降级为 NaN）: {e}")

            # 合并：以逐日估值为底，ROE 按报告期回填到交易日
            out = val.copy()
            if roe is not None and len(roe) > 0:
                # reindex(method='ffill')：取不晚于该交易日的最新一期报告
                out['roe'] = roe.reindex(out.index, method='ffill')
            else:
                out['roe'] = np.nan

            # 日期裁剪
            if start_date:
                out = out[out.index >= pd.Timestamp(start_date)]
            if end_date:
                out = out[out.index <= pd.Timestamp(end_date)]

            # 类型与异常值清洗（估值接口偶发脏数据）
            out['pe'] = pd.to_numeric(out['pe'], errors='coerce')
            out['pb'] = pd.to_numeric(out['pb'], errors='coerce')
            out['roe'] = pd.to_numeric(out['roe'], errors='coerce')
            out = out[(out['pe'] > 0) & (out['pe'] < 1000)]
            out = out[(out['pb'] > 0) & (out['pb'] < 100)]

            out = out[['pe', 'pb', 'roe']]

            if out.empty:
                logger.warning(f"{stock_code} 基本面清洗后为空")
                return None

            self._save_cache('fund', symbol, out)
            logger.info(f"基本面数据获取成功: {stock_code}, 共 {len(out)} 条")
            return out

        except Exception as e:
            logger.error(f"AKShare 获取 {stock_code} 基本面失败: {e}")
            return None
    
    def fetch_sentiment(self, stock_code: str, date: str) -> float:
        """
        获取个股舆情得分
        
        Args:
            stock_code: 股票代码
            date: 日期 YYYY-MM-DD
        
        Returns:
            舆情得分 (-1 到 1)
        """
        if not self.ak_available:
            return 0.0
        
        try:
            # 获取个股新闻情绪（示例）
            # AKShare 的情绪数据接口有限，可能需要结合其他数据源
            pass
        except Exception as e:
            logger.warning(f"舆情获取失败 {stock_code}: {e}")
        
        return 0.0
    
    def fetch_realtime_quote(self, stock_codes: List[str]) -> Dict[str, dict]:
        """
        获取实时行情
        
        Args:
            stock_codes: 股票代码列表
        
        Returns:
            字典 {code: {price, pe, pb, turnover, ...}}
        """
        if not self.ak_available:
            return {}
        
        try:
            # 获取 A 股实时行情
            df = self.ak.stock_zh_a_spot_em()
            
            results = {}
            for code in stock_codes:
                # 匹配代码（AKShare 返回格式）
                match = df[df['代码'] == code]
                if not match.empty:
                    row = match.iloc[0]
                    results[code] = {
                        'price': float(row['最新价']),
                        'pe': float(row['市盈率-动态']) if '市盈率-动态' in row else None,
                        'pb': float(row['市净率']) if '市净率' in row else None,
                        'turnover': float(row['换手率']) if '换手率' in row else None,
                        'volume': float(row['成交量']),
                        'amount': float(row['成交额']),
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"实时行情获取失败: {e}")
            return {}


# ============================================================
# 2. 通达信数据源（备选）
# ============================================================

class TDXDataSource:
    """
    通达信 pytdx 数据源
    
    功能：
    - 历史 K 线数据
    - 实时行情
    - 分笔成交
    
    依赖：pip install pytdx
    """
    
    def __init__(self):
        self.tdx_available = False
        self._connected = False
        self.api = None
        try:
            from pytdx.hq import TdxHq_API
            self.api = TdxHq_API()
            self.tdx_available = True
            logger.info("通达信 pytdx 已加载（懒连接：首次抓取时才连服务器）")
        except ImportError:
            logger.warning("pytdx 未安装，请运行: pip install pytdx")

    def _ensure_connected(self) -> bool:
        """
        懒连接：仅在实际需要抓取 K 线时才连服务器，避免无谓阻塞/报错。
        多服务器轮询，提升连通率；连接成功置 _connected 避免重复连。
        """
        if not self.tdx_available or self.api is None:
            return False
        if self._connected:
            return True
        for host, port in [
            ('119.147.212.81', 7709),
            ('120.24.0.77', 7709),
            ('47.107.75.142', 7709),
            ('101.227.73.20', 7709),
        ]:
            try:
                if self.api.connect(host, port):
                    self._connected = True
                    logger.info(f"通达信已连接 {host}:{port}")
                    return True
            except Exception as e:
                logger.warning(f"通达信连接 {host}:{port} 失败: {e}")
        logger.warning("通达信所有服务器均连接失败")
        return False
    
    def fetch_kline(
        self,
        stock_code: str,
        start_date: str = "20230101",
        end_date: str = "",
        period: str = "daily",
        adjust: str = ""
    ) -> Optional[pd.DataFrame]:
        """
        获取个股 K 线数据

        Args:
            stock_code: 完整代码（如 '000001' 或 '600000'）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            period: K 线周期 'daily'/'weekly'/'monthly'（其余映射为日线）
            adjust: 复权方式。注意 pytdx 原生不支持复权，非 '' 时仅警告并
                    返回原始（未复权）数据，可能破坏 hfq 复现纪律；
                    回测请优先用 AKShare 的 hfq 源。

        Returns:
            DataFrame
        """
        if not self._ensure_connected():
            return None

        # 周期映射（pytdx 不支持复权，adjust 非空时退化为原始数据）
        kline_type_map = {"daily": 9, "weekly": 8, "monthly": 6,
                          "60min": 3, "30min": 2, "15min": 1, "5min": 0}
        kline_type = kline_type_map.get(period, 9)
        if adjust in ("hfq", "qfq"):
            logger.warning(
                f"通达信不支持复权，{stock_code} 返回原始（未复权）K线，"
                f"可能破坏 hfq 复现纪律；建议优先使用 AKShare 源"
            )

        try:
            # 解析市场（去除交易所后缀）
            raw_code = stock_code.split('.')[0]
            if raw_code.startswith('6'):
                market = 1  # 上海
            else:
                market = 0  # 深圳

            # 获取 K 线（每次最多 800 条，向后翻页直到取完）
            all_data = []
            start_pos = 0

            while True:
                data = self.api.get_security_bars(
                    category=kline_type,
                    market=market,
                    code=raw_code,
                    start=start_pos,
                    count=800
                )

                if not data or len(data) == 0:
                    break

                all_data.extend(data)
                start_pos += len(data)

                if len(data) < 800:
                    break

            if not all_data:
                return None

            df = pd.DataFrame(all_data)
            df['datetime'] = pd.to_datetime(df['datetime'])

            # 日期过滤（start_date/end_date 为 YYYYMMDD 字符串，pandas 可比较）
            if start_date:
                df = df[df['datetime'] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df['datetime'] <= pd.Timestamp(end_date)]

            df.set_index('datetime', inplace=True)

            # 重命名列
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount',
            })

            return df

        except Exception as e:
            logger.error(f"通达信获取 {stock_code} K 线失败: {e}")
            return None

    def close(self):
        """关闭连接"""
        if self._connected and self.api is not None:
            try:
                self.api.disconnect()
            except Exception:
                pass
            self._connected = False


# ============================================================
# 3. 数据源管理器（自动降级）
# ============================================================

class DataSourceManager:
    """
    数据源管理器
    
    优先级：AKShare > 通达信 > 模拟数据
    """
    
    def __init__(self, use_real_data: bool = True, cache_dir: Optional[str] = None,
                 use_westock_real: bool = False):
        self.use_real_data = use_real_data
        self.cache_dir = cache_dir
        # 真实基本面/资金流接入开关（腾讯自选股 westock-mcp）：仅当显式开启且本地缓存
        # {code}_realfund.csv 存在时才读真实数据，否则走 AKShare/通达信/mock，基线零风险。
        self.use_westock_real = use_westock_real
        self.ak_source = AKShareDataSource(cache_dir=cache_dir) if use_real_data else None
        self.tdx_source = TDXDataSource() if use_real_data else None
        self.primary_source = self.ak_source or self.tdx_source

    # ---------- 真实基本面/资金流 本地缓存读取（westock-mcp 写入，引擎只读） ----------
    def _load_real_cache(self, kind: str, symbol: str) -> Optional[pd.DataFrame]:
        """读取真实数据缓存 CSV（{symbol}_{kind}.csv），索引为日期；不存在返回 None。"""
        if not self.cache_dir:
            return None
        import os
        p = os.path.join(self.cache_dir, f"{symbol}_{kind}.csv")
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"读取真实缓存失败 {p}: {e}")
        return None

    def fetch_kline(
        self,
        stock_code: str,
        start_date: str = "20230101",
        end_date: str = "",
        period: str = "daily",
        adjust: str = "hfq"
    ) -> Optional[pd.DataFrame]:
        """
        获取 K 线数据（自动降级）
        """
        if not self.use_real_data:
            return self._generate_mock_kline(stock_code)
        if self.ak_source:
            df = self.ak_source.fetch_kline(stock_code, start_date, end_date, period, adjust)
            if df is not None and not df.empty:
                logger.info(f"AKShare 获取 {stock_code} K 线成功，共 {len(df)} 条")
                return df
        
        # 尝试通达信
        if self.tdx_source:
            df = self.tdx_source.fetch_kline(stock_code, start_date, end_date, period, adjust)
            if df is not None and not df.empty:
                logger.info(f"通达信获取 {stock_code} K 线成功，共 {len(df)} 条")
                return df
        
        # 降级到模拟数据
        logger.warning(f"所有数据源失败，降级到模拟数据: {stock_code}")
        return self._generate_mock_kline(stock_code)
    
    def fetch_fundamental(
        self,
        stock_code: str,
        start_date: str = "",
        end_date: str = ""
    ) -> Optional[pd.DataFrame]:
        """获取基本面数据（自动降级）"""
        if not self.use_real_data:
            return self._generate_mock_fundamental(stock_code)

        # 真实基本面（腾讯自选股 westock-mcp）：本地缓存 {code}_realfund.csv
        # 含 pe/pb/roe/gross_margin/debt_ratio，优先于 AKShare 使用（已联网验证真实）
        if self.use_westock_real:
            real = self._load_real_cache('realfund', stock_code)
            if real is not None and not real.empty:
                logger.info(f"真实基本面(westock)命中: {stock_code}")
                return real

        if self.ak_source:
            df = self.ak_source.fetch_fundamental(stock_code, start_date, end_date)
            if df is not None:
                return df

        return self._generate_mock_fundamental(stock_code)
    
    def fetch_sentiment(self, stock_code: str, date: str) -> float:
        """获取舆情得分（自动降级）"""
        if not self.use_real_data:
            return self._generate_mock_sentiment(stock_code, date)
        
        if self.ak_source:
            return self.ak_source.fetch_sentiment(stock_code, date)
        
        return self._generate_mock_sentiment(stock_code, date)
    
    def fetch_realtime_quotes(self, stock_codes: List[str]) -> Dict[str, dict]:
        """获取实时行情"""
        if not self.use_real_data:
            return {}
        
        if self.ak_source:
            return self.ak_source.fetch_realtime_quote(stock_codes)
        
        return {}
    
    def _generate_mock_kline(self, stock_code: str) -> pd.DataFrame:
        """生成模拟 K 线数据（兜底）"""
        np.random.seed(_stable_seed(stock_code))
        dates = pd.bdate_range("2023-01-01", "2024-12-31")
        n = len(dates)
        
        drift = np.random.uniform(-0.0005, 0.001)
        volatility = np.random.uniform(0.015, 0.035)
        returns = np.random.normal(drift, volatility, n)
        prices = 20.0 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            'high': prices * (1 + np.random.uniform(0.005, 0.02, n)),
            'low': prices * (1 + np.random.uniform(-0.02, -0.005, n)),
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, n),
        }, index=dates)
        
        return df
    
    def _generate_mock_fundamental(self, stock_code: str) -> pd.DataFrame:
        """生成模拟基本面数据（兜底）"""
        np.random.seed(_stable_seed(stock_code + "fund"))
        dates = pd.bdate_range("2023-01-01", "2024-12-31")
        n = len(dates)
        
        return pd.DataFrame({
            'pe': np.random.uniform(5, 30, n),
            'pb': np.random.uniform(0.5, 5, n),
            'roe': np.random.uniform(5, 25, n),
        }, index=dates)
    
    def _generate_mock_sentiment(self, stock_code: str, date: str) -> float:
        """生成模拟舆情数据（兜底）"""
        np.random.seed(_stable_seed(stock_code + date))
        return np.random.uniform(-0.5, 0.5)
    
    def close(self):
        """关闭所有数据源连接"""
        if self.tdx_source:
            self.tdx_source.close()


# ============================================================
# 4. 便捷函数
# ============================================================

def get_stock_data(
    stock_code: str,
    start_date: str = "20230101",
    end_date: str = "",
    use_real_data: bool = True
) -> Optional[pd.DataFrame]:
    """
    便捷函数：获取股票数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        use_real_data: 是否使用真实数据
    
    Returns:
        K 线 DataFrame
    """
    manager = DataSourceManager(use_real_data=use_real_data)
    try:
        return manager.fetch_kline(stock_code, start_date, end_date)
    finally:
        manager.close()


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)
    
    manager = DataSourceManager(use_real_data=True)
    
    # 测试获取平安银行 K 线
    df = manager.fetch_kline("000001", "20240101", "20241231")
    if df is not None:
        print(f"获取成功，共 {len(df)} 条记录")
        print(df.head())
    else:
        print("获取失败")
    
    manager.close()
