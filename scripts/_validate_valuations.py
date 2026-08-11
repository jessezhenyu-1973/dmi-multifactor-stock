"""临时校验：用已缓存的 15 只子集，确认 PB = PE × ROE 正确且 roe 为百分比口径。"""
import logging
logging.basicConfig(level=logging.WARNING)
from real_data_loader import build_real_dataset

CODES = [
    "000657.SZ", "000988.SZ", "002202.SZ", "002532.SZ", "002558.SZ",
    "002602.SZ", "002837.SZ", "301165.SZ", "301308.SZ", "600118.SH",
    "600221.SH", "600549.SH", "688072.SH", "688183.SH", "688521.SH",
]

dm, fd, bench = build_real_dataset(CODES, "20220101", "20251231", with_benchmark=False)
print(f"K线 {len(dm)} 只, 基本面 {len(fd)} 只\n")

bad = 0
for code in CODES:
    if code not in fd:
        print(f"  {code}: 无基本面"); continue
    f = fd[code].dropna(subset=["pe", "pb", "roe"])
    if f.empty:
        print(f"  {code}: PE/PB 全 NaN"); continue
    row = f.iloc[len(f) // 2]
    pe, pb, roe_pct = row["pe"], row["pb"], row["roe"]
    expected_pb = pe * (roe_pct / 100.0)   # roe 为百分比 -> 转小数
    ok = abs(pb - expected_pb) < max(0.05, abs(expected_pb) * 0.02)
    if not ok: bad += 1
    print(f"  {code} pe={pe:8.2f} pb={pb:7.2f} roe={roe_pct:5.2f}%  PB≈PE×ROE={expected_pb:7.2f}  {'OK' if ok else 'BAD'}")

# 抽查 value 因子是否真正区分（roe 为百分比时 fundamental_score 应正常工作）
from multi_factor_dmi_strategy import compute_fundamental_score
print("\n== 抽查 fundamental_score 在真实数据上是否生效（roe 百分比口径）==")
for code in ["600118.SH", "002532.SZ", "000988.SZ"]:
    f = fd[code].dropna(subset=["pe", "pb", "roe"])
    s = compute_fundamental_score(f)
    print(f"  {code}: fundamental_score 均值={s.mean():.1f} 范围[{s.min():.1f},{s.max():.1f}]")
print(f"\nPB 公式+roe口径校验: {'全部通过' if bad == 0 else f'{bad} 只异常'}")
