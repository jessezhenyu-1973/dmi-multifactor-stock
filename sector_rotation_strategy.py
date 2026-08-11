"""
龙头战法选股系统 v1.0
====================
多策略量化交易模型 - 市场状态识别 → 板块轮动 → 龙头股右侧交易

核心策略：
1. 市场状态识别（牛市/熊市/震荡市）
2. 板块轮动分析（基于涨停池资金流向）
3. 龙头股筛选（右侧交易，不做反弹）

数据源：同花顺金融数据服务 (Fuyao MCP)
"""

import json
import os
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from hermes_tools import tool_call


# ==============================================================================
# 数据模型
# ==============================================================================

class MarketRegime(Enum):
    BULL = "牛市"
    BEAR = "熊市"
    SIDEWAYS = "震荡市"


def sma(values: list[float], period: int) -> float | None:
    """计算简单移动平均"""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: list[float], period: int) -> float | None:
    """计算指数移动平均"""
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    e = sum(values[:period]) / period
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def ret(values: list[float], n: int) -> float:
    """计算n日收益率"""
    if len(values) < n + 1:
        return 0.0
    return (values[-1] / values[-n] - 1) * 100


def parse_kline(items: list[dict]) -> list[dict]:
    """解析K线数据"""
    return [{
        "date": i.get("trade_time", ""),
        "open": float(i.get("open", 0)),
        "high": float(i.get("high", 0)),
        "low": float(i.get("low", 0)),
        "close": float(i.get("close", 0)),
        "volume": float(i.get("volume", 0)),
        "amount": float(i.get("amount", 0)),
    } for i in items]


def ms(date_str: str) -> int:
    """日期字符串转毫秒时间戳"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def now_ms() -> int:
    """当前毫秒时间戳"""
    return int(datetime.now().timestamp() * 1000)


def days_ago_ms(n: int) -> int:
    """n天前的毫秒时间戳"""
    return int((datetime.now() - timedelta(days=n)).timestamp() * 1000)


# ==============================================================================
# 阶段1：市场状态识别
# ==============================================================================

def detect_regime() -> dict:
    """
    识别市场状态（牛市/熊市/震荡市）
    
    判断逻辑：
    - 牛市：核心指数均线多头排列，价格站上20日线，MACD为正
    - 熊市：核心指数均线空头排列，价格跌破20日线，MACD为负
    - 震荡市：信号混合，无明显单边趋势
    """
    print("\n" + "=" * 70)
    print("  [阶段1] 市场状态识别")
    print("=" * 70)

    indices = {
        "000001.SH": "上证综指",
        "399001.SZ": "深证成指",
        "000300.SH": "沪深300",
    }

    results = {}
    for code, name in indices.items():
        try:
            end = now_ms()
            start = days_ago_ms(120)
            r = tool_call(
                name="mcp__fuyao_a_share_index__get_a_share_index_prices_historical",
                arguments={"thscode": code, "interval": "1d", "start": start, "end": end}
            )
            items = r.get("data", {}).get("items", [])
            bars = parse_kline(items)

            if len(bars) < 60:
                results[code] = {
                    "trend": "数据不足", "ma20": 0, "ma60": 0, "ma120": 0,
                    "pvs_ma20": 0, "pvs_ma60": 0, "macd": 0
                }
                continue

            closes = [b["close"] for b in bars]
            price = closes[-1]
            ma20 = sma(closes, 20)
            ma60 = sma(closes, 60)
            ma120 = sma(closes, 120)

            pvs20 = (price / ma20 - 1) * 10 if ma20 and ma20 > 0 else 0
            pvs60 = (price / ma60 - 1) * 10 if ma60 and ma60 > 0 else 0
            ma20_vs_60 = (ma20 / ma60 - 1) * 10 if ma20 and ma60 and ma60 > 0 else 0

            # MACD计算
            macd = 0
            if len(closes) >= 26:
                e12 = ema(closes, 12)
                e26 = ema(closes, 26)
                if e12 and e26 and e26 > 0:
                    macd = ((e12 - e26) / e26) * 100

            # 趋势判断
            if ma20 and ma60 and ma120:
                if ma20 > ma60 > ma120 and price > ma20:
                    trend = "强势上涨"
                elif ma20 > ma60 and price > ma20:
                    trend = "上升趋势"
                elif ma20 < ma60 < ma120 and price < ma20:
                    trend = "强势下跌"
                elif ma20 < ma60 and price < ma20:
                    trend = "下降趋势"
                else:
                    trend = "震荡整理"
            else:
                if pvs20 > 0 and pvs60 > 0:
                    trend = "偏强"
                elif pvs20 < 0 and pvs60 < 0:
                    trend = "偏弱"
                else:
                    trend = "震荡"

            results[code] = {
                "trend": trend, "ma20": ma20 or 0, "ma60": ma60 or 0,
                "ma120": ma120 or 0, "pvs20": pvs20, "pvs60": pvs60, "macd": macd,
            }
        except Exception as e:
            print(f"  [WARN] {name}({code}): {e}")
            results[code] = {
                "trend": "错误", "ma20": 0, "ma60": 0, "ma120": 0,
                "pvs20": 0, "pvs60": 0, "macd": 0
            }

    # 综合判断
    bullish = sum(
        1 for r in results.values()
        if r["pvs20"] > 1.0 and r["pvs20"] > 0 and r["ma20"] > r["ma60"]
    )
    bearish = sum(
        1 for r in results.values()
        if r["pvs20"] < -1.0 and r["ma20"] < r["ma60"]
    )
    avg_macd = sum(r["macd"] for r in results.values()) / max(len(results), 1)

    if bullish >= 2 and avg_macd > 0:
        regime = MarketRegime.BULL
        conf = min(bullish / 3.0, 1.0)
        desc = f"市场处于牛市状态。{bullish}/3核心指数处于上升通道，均线多头排列。"
    elif bearish >= 2 and avg_macd < 0:
        regime = MarketRegime.BEAR
        conf = min(bearish / 3.0, 1.0)
        desc = f"市场处于熊市状态。{bearish}/3核心指数处于下降通道，均线空头排列。"
    else:
        regime = MarketRegime.SIDEWAYS
        conf = 0.5
        desc = "市场处于震荡市。核心指数信号混合，无明显单边趋势。"

    print(f"\n  市场状态: {regime.value}")
    print(f"  置信度: {conf:.0%}")
    for code, name in indices.items():
        r = results.get(code, {})
        print(f"  {name}: {r.get('trend', 'N/A')} (MACD: {r.get('macd', 0):+.2f})")
    print(f"  {desc}")

    return {
        "regime": regime.value, "confidence": conf, "description": desc,
        "indices": {name: results[code] for code, name in indices.items()},
    }


# ==============================================================================
# 阶段2：板块轮动分析
# ==============================================================================

def analyze_sectors() -> list[dict]:
    """
    板块轮动分析
    
    方法：
    1. 获取全市场行业板块和概念板块列表
    2. 获取板块实时行情快照
    3. 统计涨停池个股的行业分布，识别资金流向
    4. 综合打分排序
    """
    print("\n" + "=" * 70)
    print("  [阶段2] 板块轮动分析")
    print("=" * 70)

    # 核心跟踪板块
    core_sectors = {
        "881121.TI": "半导体", "881124.TI": "消费电子", "881125.TI": "汽车整车",
        "881126.TI": "汽车零部件", "881281.TI": "电池", "881279.TI": "光伏设备",
        "881280.TI": "风电设备", "881065.TI": "低空经济", "881066.TI": "商业航天",
        "881122.TI": "光学光电子", "881171.TI": "自动化设备", "881270.TI": "元件",
        "881144.TI": "医疗器械", "881141.TI": "中药", "881157.TI": "证券",
        "881155.TI": "银行", "881273.TI": "白酒", "881131.TI": "白色家电",
        "881129.TI": "通信设备", "881272.TI": "软件开发", "881063.TI": "算力",
        "881064.TI": "光模块", "881072.TI": "HBM", "881073.TI": "先进封装",
        "881067.TI": "人形机器人", "881068.TI": "合成生物", "881074.TI": "华为概念",
        "881075.TI": "特斯拉概念", "881051.TI": "军工",
    }

    # 同时获取概念板块
    try:
        cat_r = tool_call(
            name="mcp__fuyao_a_share_index__get_a_share_index_catalog_ths_index_list",
            arguments={"tag": "cn_concept"}
        )
        for item in cat_r.get("data", {}).get("items", [])[:60]:
            code = item.get("thscode", "")
            name = item.get("name", "")
            if code and name and code not in core_sectors:
                core_sectors[code] = name
    except:
        pass

    # 快照所有板块
    all_codes = list(core_sectors.keys())
    sector_data = {}

    # 分批获取快照
    for i in range(0, len(all_codes), 50):
        chunk = all_codes[i:i+50]
        try:
            r = tool_call(
                name="mcp__fuyao_a_share_index__get_a_share_index_prices_snapshot",
                arguments={"thscodes": ",".join(chunk)}
            )
            for item in r.get("data", {}).get("items", []):
                code = item.get("thscode", "")
                sector_data[code] = {
                    "price": float(item.get("last_price", 0)),
                    "change_pct": float(item.get("price_change_ratio_pct", 0)),
                    "turnover": float(item.get("turnover", 0)),
                    "volume": float(item.get("volume", 0)),
                }
        except Exception as e:
            print(f"  [WARN] Snapshot batch {i//50}: {e}")

    # 获取涨停池分析资金流向
    limit_up_sectors = {}
    try:
        lu_r = tool_call(
            name="mcp__fuyao_a_share__get_a_share_special_data_limit_up_pool",
            arguments={"size": 200, "page": 1}
        )
        for item in lu_r.get("data", {}).get("items", []):
            reason = item.get("limit_up_reason", "")
            # 从涨停原因中提取板块关键词
            for sector_name in core_sectors.values():
                if sector_name in reason or reason in sector_name:
                    limit_up_sectors[sector_name] = limit_up_sectors.get(sector_name, 0) + 1
                    break
    except:
        pass

    # 给板块打分
    scored = []
    for code, name in core_sectors.items():
        snap = sector_data.get(code, {})
        chg = snap.get("change_pct", 0)

        base = max(-10, min(chg / 5, 10))

        # 资金流向分数
        lu = limit_up_sectors.get(name, 0)
        cap = min(lu / 5.0, 1.0)

        mom = min(max(chg / 3, -1), 1)
        composite = 0.4 * base + 0.3 * mom + 0.3 * cap

        scored.append({
            "thscode": code, "name": name,
            "price": snap.get("price", 0),
            "change_pct": chg, "capital_score": cap,
            "momentum_score": mom, "composite_score": composite,
        })

    scored.sort(key=lambda s: s["composite_score"], reverse=True)

    print(f"\n  分析板块数: {len(scored)}")
    print(f"\n  {'排名':>4} | {'板块名称':<15} | {'涨跌幅':>8} | {'综合得分':>8} | {'资金分':>6}")
    print(f"  {'-' * 60}")
    for i, s in enumerate(scored[:20], 1):
        print(f"  {i:>4} | {s['name']:<15} | {s['change_pct']:>+7.2f}% | {s['composite_score']:>8.3f} | {s['capital_score']:>6.3f}")

    return scored


# ==============================================================================
# 阶段3：龙头股筛选（右侧交易）
# ==============================================================================

def scan_leading_stocks(sectors: list[dict], regime: str) -> list[dict]:
    """
    龙头股筛选 - 只做右侧交易
    
    右侧交易核心规则：
    1. 股价 > 20日均线 > 60日均线（多头排列）
    2. 近期放量（量比 > 1.5）
    3. 板块为当前最强主线
    4. 非ST、非退市整理、非次新股(<60天)
    5. 非当日涨停（避免追高）
    """
    print("\n" + "=" * 70)
    print("  [阶段3] 龙头股筛选（右侧交易）")
    print("=" * 70)

    # 取前8个最强板块
    hot = sectors[:8] if sectors else []

    candidates = []

    for sector in hot:
        code = sector["thscode"]
        name = sector["name"]

        # 获取成分股
        try:
            const_r = tool_call(
                name="mcp__fuyao_a_share_index__get_a_share_index_constituents_ths_stock_list",
                arguments={"thscode": code}
            )
            stocks = const_r.get("data", {}).get("items", [])
        except Exception as e:
            print(f"  [WARN] {name} constituents: {e}")
            continue

        if not stocks:
            continue

        # 快照获取行情
        stock_codes = [s.get("thscode", "") for s in stocks if s.get("thscode")]
        stock_codes = [c for c in stock_codes if c]

        snapshots = {}
        for i in range(0, len(stock_codes), 50):
            chunk = stock_codes[i:i+50]
            try:
                r = tool_call(
                    name="mcp__fuyao_a_share__get_a_share_prices_snapshot",
                    arguments={"thscodes": ",".join(chunk)}
                )
                for item in r.get("data", {}).get("items", []):
                    snapshots[item.get("thscode", "")] = item
            except:
                pass

        # 处理每只股票
        for stock in stocks:
            sc = stock.get("thscode", "")
            sn = stock.get("name", "")
            snap = snapshots.get(sc, {})

            if not snap:
                continue

            chg = float(snap.get("price_change_ratio_pct", 0))
            price = float(snap.get("last_price", 0))

            # 基础过滤
            if "ST" in sn or "退" in sn:
                continue
            if price < 5:
                continue
            if abs(chg) > 19.5:  # 排除已涨停
                continue

            # 获取历史数据
            try:
                r = tool_call(
                    name="mcp__fuyao_a_share__get_a_share_prices_historical",
                    arguments={"thscode": sc, "interval": "1d",
                               "start": days_ago_ms(60), "end": now_ms()}
                )
                bars = parse_kline(r.get("data", {}).get("items", []))
            except:
                continue

            if len(bars) < 60:
                continue

            closes = [b["close"] for b in bars]
            ma20 = sma(closes, 20)
            ma60 = sma(closes, 60)

            # 右侧核心过滤
            if not (ma20 and ma60 and ma20 > 0 and ma60 > 0):
                continue
            if not (price > ma20 > ma60):  # 右侧：P > MA20 > MA60
                continue

            r5 = ret(closes, 5)
            r10 = ret(closes, 10)
            r20 = ret(closes, 20)

            # 技术分
            tech = 0.5  # 右侧已确认
            ma_spread = (ma20 / ma60 - 1) * 100
            if ma_spread > 3:
                tech += 0.3
            elif ma_spread > 1:
                tech += 0.2

            mom_ratio = (r5 / max(r20, 0.01)) if r20 > 0 and r5 > 0 else (r10 / 0.01 if r10 > 0 else 0)
            tech += min(mom_ratio * 0.1, 0.2)

            # 动量分
            mom_score = min(r5 / 5, 3)
            if r5 > r10 > r20 > 0:
                mom_score += 2
            elif r5 > r10 > 0:
                mom_score += 1
            elif r10 > r20 > 0:
                mom_score += 0.5
            mom_score = min(mom_score, 10)

            # 综合得分
            comp = (0.35 * tech + 0.30 * min(mom_score, 10) +
                    0.20 * sector["composite_score"] + 0.15 * min(chg / 5, 5))

            candidates.append({
                "thscode": sc, "name": sn, "sector": name,
                "price": price, "change_pct": chg,
                "return_5d": r5, "return_10d": r10, "return_20d": r20,
                "ma20": ma20, "ma60": ma60,
                "technical_score": round(tech, 3),
                "momentum_score": round(mom_score, 3),
                "sector_score": round(sector["composite_score"], 3),
                "composite_score": round(comp, 3),
            })

    candidates.sort(key=lambda s: s["composite_score"], reverse=True)

    print(f"\n  筛选到右侧标的: {len(candidates)} 只")

    if candidates:
        print(f"\n  {'排名':>4} | {'代码':<12} | {'名称':<10} | {'板块':<12} | {'价格':>8} | "
              f"{'5日%':>6} | {'10日%':>7} | {'20日%':>7} | {'综合分':>6}")
        print(f"  {'-' * 95}")
        for i, s in enumerate(candidates[:30], 1):
            print(f"  {i:>4} | {s['thscode']:<12} | {s['name']:<10} | {s['sector']:<12} | "
                  f"{s['price']:>8.2f} | {s['return_5d']:>+5.1f}% | {s['return_10d']:>+6.1f}% | "
                  f"{s['return_20d']:>+6.1f}% | {s['composite_score']:>6.3f}")

    return candidates


# ==============================================================================
# 主执行函数
# ==============================================================================

def run() -> dict:
    """执行完整策略"""
    print("=" * 70)
    print("  龙头战法选股系统 - 板块轮动 + 龙头右侧交易系统")
    print("  执行时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # 阶段1：市场状态识别
    regime = detect_regime()

    # 阶段2：板块轮动分析
    sectors = analyze_sectors()

    # 阶段3：龙头股筛选
    stocks = scan_leading_stocks(sectors, regime["regime"])

    # 阶段4：策略建议
    print("\n" + "=" * 70)
    print("  [阶段4] 策略建议")
    print("=" * 70)

    regime_name = regime["regime"]
    hot = sectors[:5] if sectors else []

    print(f"\n  当前市场状态: {regime_name}")

    if regime_name == "牛市":
        print("  【牛市策略】")
        print("  - 仓位建议：70-100%")
        print("  - 选股策略：重点追击板块龙头，可适度追涨")
        print("  - 板块轮动：关注资金流入最快的前3个板块")
        print("  - 持股策略：趋势不破持有，放量突破加仓")
    elif regime_name == "熊市":
        print("  【熊市策略】")
        print("  - 仓位建议：0-20%")
        print("  - 选股策略：严格右侧，仅做超强势龙头")
        print("  - 持股策略：快进快出，破20日线立即止损")
    else:
        print("  【震荡市策略】")
        print("  - 仓位建议：30-50%")
        print("  - 选股策略：精选右侧标的，避免追高")
        print("  - 板块轮动：关注资金持续流入的板块，做波段")
        print("  - 持股策略：20日线附近低吸，突破前高加仓")

    if hot:
        print(f"\n  当前最强板块:")
        for i, s in enumerate(hot[:5], 1):
            print(f"    {i}. {s['name']} ({s['change_pct']:+.2f}%) 综合分: {s['composite_score']:.3f}")

    if stocks:
        print(f"\n  TOP 5 龙头候选:")
        for i, s in enumerate(stocks[:5], 1):
            print(f"    {i}. {s['name']} ({s['thscode']}) - {s['sector']}")
            print(f"       价格: {s['price']:.2f} | 5日: {s['return_5d']:+.1f}% | "
                  f"10日: {s['return_10d']:+.1f}% | 20日: {s['return_20d']:+.1f}%")
            print(f"       综合分: {s['composite_score']:.3f} | "
                  f"技术分: {s['technical_score']:.3f} | 动量分: {s['momentum_score']:.3f}")

    # 保存报告
    os.makedirs("output", exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "regime": regime,
        "top_sectors": sectors[:20],
        "leading_stocks": stocks[:30],
    }

    path = "output/sector_rotation_report.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存至: {path}")

    return report


if __name__ == "__main__":
    run()
