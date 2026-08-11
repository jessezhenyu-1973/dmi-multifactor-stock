"""
批量把 westock-mcp 抓到的真实财务/资金流，落盘为引擎可消费的缓存 CSV。

输入：本脚本内嵌的 RAW 字典，来自对 10 只宇宙股票调用
      mcp__westock-mcp__data_finance(income/balance) + data_quote + data_fund_flow
      的真实返回（快照日 2026-03-31 财报 / 2026-08-10 资金流）。

输出（写入 ../cache/）：
  {code}_realfund.csv  列: pe, pb, roe, gross_margin, debt_ratio   (index=报告期)
  {code}_realflow.csv  列: main_net_flow(元), total_market_cap_yi(亿元), close  (index=资金流日)

文件命名带交易所后缀（如 000001.SZ_realfund.csv），与 MultiFactorDMIStrategy
传入的股票代码格式一致，确保 DataSourceManager._load_cache 能命中。

派生字段：
  roe          = 归母净利TTM / 股东权益
  gross_margin = (营收TTM - 成本TTM) / 营收TTM
  debt_ratio   = 总负债 / 总资产

重新运行即可刷新；若后续连接更多连接器（东方财富妙想/Wind/恒生聚源），
可替换 RAW 来源而不动引擎。
"""

import os
import sys
import math
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
REPORT_DATE = "2026-03-31"
FLOW_DATE = "2026-08-10"

# 引擎股票代码 -> 真实提取字段
# rev/cost/np 为 TTM；equity 股东权益；liab 总负债；assets 总资产；
# pe/pb 来自行情快照；mcap 总市值(亿元)；flow 主力净流入(元，正=流入)；close 现价
RAW = {
    "000001.SZ": dict(rev=133010000000.0, cost=81123000000.0, np=43060000000.0,
                       equity=544083000000.0, liab=5489879000000.0, assets=6033962000000.0,
                       pe=5.1, pb=0.48, mcap=2194.81, flow=-102449818.02, close=11.31),
    "000002.SZ": dict(rev=224366008400.61, cost=201952750878.24, np=-88262418179.95,
                       equity=228061984693.62, liab=769344739328.43, assets=997406724022.05,
                       pe=-0.44, pb=0.35, mcap=388.94, flow=149689.88, close=3.26),
    "600000.SH": dict(rev=174615000000.0, cost=119382000000.0, np=50280000000.0,
                       equity=841399000000.0, liab=9464247000000.0, assets=10305646000000.0,
                       pe=6.15, pb=0.42, mcap=3094.11, flow=34439905.33, close=9.29),
    "600036.SH": dict(rev=340721000000.0, cost=161066000000.0, np=150747000000.0,
                       equity=1290585000000.0, liab=12194297000000.0, assets=13484882000000.0,
                       pe=6.51, pb=0.89, mcap=9815.56, flow=-146883792.72, close=38.92),
    "000858.SZ": dict(rev=46280768276.55, cost=9766010718.12, np=12600709094.88,
                       equity=130478844013.87, liab=68159329198.55, assets=198638173212.42,
                       pe=23.47, pb=2.51, mcap=2957.79, flow=177882579.17, close=76.2),
    "600519.SH": dict(rev=172146396849.52, cost=16351576220.80, np=82715105749.37,
                       equity=281135886435.69, liab=38782958469.89, assets=319918844905.58,
                       pe=20.39, pb=7.24, mcap=16861.98, flow=705758304.55, close=1348.87),
    "000333.SZ": dict(rev=459711794000.0, cost=338257442000.0, np=44197734000.0,
                       equity=245441131000.0, liab=368440440000.0, assets=613881571000.0,
                       pe=14.72, pb=3.19, mcap=6507.8, flow=-125467531.93, close=85.31),
    "601318.SH": dict(rev=1036110000000.0, cost=848827000000.0, np=132784000000.0,
                       equity=1434290000000.0, liab=12735657000000.0, assets=14169947000000.0,
                       pe=7.27, pb=0.98, mcap=9653.18, flow=-407489453.50, close=53.31),
    "002594.SZ": dict(rev=783829824000.0, cost=647103396000.0, np=27548588000.0,
                       equity=262120423000.0, liab=639956542000.0, assets=902076965000.0,
                       pe=30.17, pb=3.63, mcap=8311.24, flow=179188107.06, close=91.16),
    "600276.SH": dict(rev=32564370391.88, cost=4384942437.44, np=8119267525.13,
                       equity=64147086277.95, liab=7146776162.93, assets=71293862440.88,
                       pe=44.04, pb=5.74, mcap=3576.12, flow=-479546257.62, close=53.88),
}


def derive(d: dict) -> dict:
    """由原始字段派生 roe / gross_margin / debt_ratio。"""
    roe = d["np"] / d["equity"] if d["equity"] else float("nan")
    gm = (d["rev"] - d["cost"]) / d["rev"] if d["rev"] else float("nan")
    dr = d["liab"] / d["assets"] if d["assets"] else float("nan")
    return dict(pe=d["pe"], pb=d["pb"], roe=roe, gross_margin=gm, debt_ratio=dr)


def main():
    os.makedirs(CACHE, exist_ok=True)
    print(f"{'code':12} {'pe':>7} {'pb':>6} {'roe%':>7} {'gross%':>7} {'debt%':>7} {'flow亿':>9} {'sent':>6}")
    print("-" * 78)
    for code, d in RAW.items():
        f = derive(d)
        # 基本面缓存（带后缀文件名，匹配引擎代码）
        fund = pd.DataFrame([f], index=pd.to_datetime([REPORT_DATE]))
        fund.index.name = "date"
        fund.to_csv(os.path.join(CACHE, f"{code}_realfund.csv"))

        # 资金流缓存（含市值，供舆情归一化）
        flow = pd.DataFrame(
            [{"main_net_flow": d["flow"],
              "total_market_cap_yi": d["mcap"],
              "close": d["close"]}],
            index=pd.to_datetime([FLOW_DATE]),
        )
        flow.index.name = "date"
        flow.to_csv(os.path.join(CACHE, f"{code}_realflow.csv"))

        # 舆情预览（与 SentimentMonitor 同口径：MainNetFlow/总市值 归一化后 tanh）
        SCALE = 1000.0
        ratio = d["flow"] / (d["mcap"] * 1e8)
        sent = math.tanh(ratio * SCALE)
        print(f"{code:12} {d['pe']:>7.2f} {d['pb']:>6.2f} {f['roe']*100:>7.2f} "
              f"{f['gross_margin']*100:>7.2f} {f['debt_ratio']*100:>7.2f} "
              f"{d['flow']/1e8:>9.3f} {sent:>6.3f}")
    print(f"\n已写入 {len(RAW)} 只股票的真实缓存到: {os.path.abspath(CACHE)}")


if __name__ == "__main__":
    main()
