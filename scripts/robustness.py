"""
过拟合 / 样本外检验（Robustness / Out-of-sample validation）
==========================================================
- 训练窗口 2023（样本内）跑网格选冠军
- 测试窗口 2024（样本外，从未用于选参）验证冠军是否存活
- 稳定性矩阵：6 个权重预设 x (2023 / 2024 / 2年) 表现
- 2 年全样本排名，给出更鲁棒的生产默认权重建议

协议（避免前视）：
- 每个窗口的数据在传入回测前独立裁剪，技术指标仅在该窗口内计算
- 2023 训练 / 2024 测试 完全隔离
- 复用 tune.load_universe / tune.run_one（与已有网格搜索同一套管线，保证可比性）
"""
import os
import json
import numpy as np
import pandas as pd

import tune
from tune import (
    WEIGHT_PRESETS, TOP_NS, FREQS, TransactionCost, load_universe, run_one
)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness.log")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness_result.json")

WINDOWS = {
    "train_2023": ("20230101", "20231231"),
    "test_2024":  ("20240101", "20241231"),
    "full_2y":    ("20230101", "20241231"),
}


def crop(data_map, fundamental_data, start, end):
    """把数据裁剪到目标窗口（回测前独立裁剪，确保无前视）"""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    dm, fd = {}, {}
    for code, df in data_map.items():
        d = df[(df.index >= lo) & (df.index <= hi)]
        if not d.empty:
            dm[code] = d
    if fundamental_data:
        for code, f in fundamental_data.items():
            f2 = f[(f.index >= lo) & (f.index <= hi)]
            if not f2.empty:
                fd[code] = f2
    return dm, fd


def run_window(full_dm, full_fd, start, end, weights_tuple, top_n, freq, cost):
    dm, fd = crop(full_dm, full_fd, start, end)
    return run_one(dm, fd, weights_tuple, top_n, freq, cost)


def main():
    cost = TransactionCost()
    # 加载全量 2 年窗口缓存（load_universe 会按 START/END 裁剪，这里设为全区间）
    tune.START = "20230101"
    tune.END = "20241231"
    full_dm, full_fd = load_universe(limit=None)

    lines = []
    def log(s=""):
        lines.append(str(s))
        print(s)

    log("=" * 100)
    log("过拟合 / 样本外检验  (沪深300 · 真实行情+基本面 · 含交易成本)")
    log("=" * 100)

    # ---------- 1) 训练窗口 2023 网格 ----------
    log("\n[1] 训练窗口 2023 网格搜索 (样本内)")
    train_rows = []
    for pname, wt in WEIGHT_PRESETS.items():
        for top_n in TOP_NS:
            for freq in FREQS:
                try:
                    s = run_window(full_dm, full_fd, *WINDOWS["train_2023"], wt, top_n, freq, cost)
                    train_rows.append({
                        "preset": pname, "top_n": top_n, "freq": freq,
                        "total_return": s['total_return'], "annual_return": s['annual_return'],
                        "sharpe": s['sharpe_ratio'], "max_drawdown": s['max_drawdown'],
                        "win_rate": s['win_rate'], "trades": s['total_trades'],
                    })
                    log(f"  {pname:10s} top_n={top_n} -> 收益 {s['total_return']:.2%} "
                        f"夏普 {s['sharpe_ratio']:.2f} 回撤 {s['max_drawdown']:.2%}")
                except Exception as e:
                    log(f"  {pname} top_n={top_n} 失败: {e}")
    train_df = pd.DataFrame(train_rows)
    feasible = train_df[train_df['max_drawdown'] <= 0.25].copy()
    pool = feasible if not feasible.empty else train_df.copy()
    pool = pool.sort_values("sharpe", ascending=False).reset_index(drop=True)
    champ = pool.iloc[0]
    log(f"\n  2023 训练冠军: {champ['preset']} top_n={int(champ['top_n'])} -> "
        f"夏普 {champ['sharpe']:.2f} 收益 {champ['total_return']:.2%} 回撤 {champ['max_drawdown']:.2%}")

    # ---------- 2) 稳定性矩阵：6 预设(top5) x 3 窗口 ----------
    log("\n[2] 稳定性矩阵 (各预设 top_n=5, freq=20)")
    presets_list = list(WEIGHT_PRESETS.keys())
    stab = {}
    for pname in presets_list:
        wt = WEIGHT_PRESETS[pname]
        stab[pname] = {}
        for wkey, (ws, we) in WINDOWS.items():
            s = run_window(full_dm, full_fd, ws, we, wt, 5, 20, cost)
            stab[pname][wkey] = {
                "total_return": s['total_return'], "sharpe": s['sharpe_ratio'],
                "max_drawdown": s['max_drawdown'],
            }
        log(f"  {pname:10s} | 2023 收益 {stab[pname]['train_2023']['total_return']:.2%} "
            f"夏普 {stab[pname]['train_2023']['sharpe']:.2f}"
            f" | 2024 收益 {stab[pname]['test_2024']['total_return']:.2%} "
            f"夏普 {stab[pname]['test_2024']['sharpe']:.2f}"
            f" | 2年 收益 {stab[pname]['full_2y']['total_return']:.2%} "
            f"夏普 {stab[pname]['full_2y']['sharpe']:.2f}")

    # 2 年全样本冠军（鲁棒性参考）
    twoyr = [(p, stab[p]['full_2y']['sharpe'], stab[p]['full_2y']['total_return']) for p in presets_list]
    twoyr_sorted = sorted(twoyr, key=lambda x: x[1], reverse=True)
    twoyr_champ = twoyr_sorted[0][0]
    log(f"\n  2 年全样本夏普冠军: {twoyr_champ} (夏普 {twoyr_sorted[0][1]:.2f}, "
        f"收益 {twoyr_sorted[0][2]:.2%})")

    # 哪些预设两年都正收益（鲁棒）
    robust_both = [p for p in presets_list
                   if stab[p]['train_2023']['total_return'] > 0
                   and stab[p]['test_2024']['total_return'] > 0]
    log(f"  两年均正收益的预设: {robust_both}")

    # ---------- 3) 样本外测试 2024：多个候选配置 ----------
    log("\n[3] 样本外测试 2024 (候选配置对比)")
    candidates = {
        "train_champion(2023冠军)": (champ['preset'], int(champ['top_n'])),
        "sent_low(2024样本内冠军)": ("sent_low", 5),
        "even(近等权鲁棒)": ("even", 5),
        f"{twoyr_champ}(2年冠军)": (twoyr_champ, 5),
    }
    test_rows = []
    for label, (pname, top_n) in candidates.items():
        wt = WEIGHT_PRESETS[pname]
        s = run_window(full_dm, full_fd, *WINDOWS["test_2024"], wt, top_n, 20, cost)
        test_rows.append({
            "config": label, "preset": pname, "top_n": top_n,
            "total_return": s['total_return'], "sharpe": s['sharpe_ratio'],
            "max_drawdown": s['max_drawdown'], "win_rate": s['win_rate'], "trades": s['total_trades'],
        })
        log(f"  {label:30s} -> 收益 {s['total_return']:.2%} 夏普 {s['sharpe_ratio']:.2f} "
            f"回撤 {s['max_drawdown']:.2%} 胜率 {s['win_rate']:.2%}")

    # ---------- 4) 结论 ----------
    log("\n[4] 结论")
    tc = next(r for r in test_rows if r['config'].startswith("train_champion"))
    sl = next(r for r in test_rows if r['config'].startswith("sent_low"))
    ev = next(r for r in test_rows if r['config'].startswith("even"))
    log(f"  - 2023 训练冠军 = {champ['preset']} (top_n={int(champ['top_n'])})")
    log(f"      样本内 2023 夏普 {champ['sharpe']:.2f} / 收益 {champ['total_return']:.2%}")
    log(f"      样本外 2024 夏普 {tc['sharpe']:.2f} / 收益 {tc['total_return']:.2%} "
        f"(OOS 夏普衰减 {champ['sharpe']-tc['sharpe']:.2f})")
    log(f"  - 对比：2024 样本内冠军 sent_low 在 2024 = 夏普 {sl['sharpe']:.2f} / 收益 {sl['total_return']:.2%}")
    log(f"  - 近等权 even 在 2024 = 夏普 {ev['sharpe']:.2f} / 收益 {ev['total_return']:.2%}")
    if tc['total_return'] > 0:
        log(f"  - 冠军样本外仍盈利 ({tc['total_return']:.2%})，但夏普从 {champ['sharpe']:.2f} "
            f"降至 {tc['sharpe']:.2f} —— 存在过拟合衰减但不致命，可作为候选之一。")
    else:
        log(f"  - 冠军样本外转亏 ({tc['total_return']:.2%}) —— 确认严重过拟合，2023 冠军不可外推！")
    log(f"  - 推荐生产默认: 优先 {twoyr_champ} (2年全样本最稳) 或 even (近等权、历次实验稳定居前)；"
        f"避免使用单年冠军直接上线。")

    # 写 JSON
    result = {
        "train_champion": {
            "preset": champ['preset'], "top_n": int(champ['top_n']),
            "train_sharpe": champ['sharpe'], "train_return": champ['total_return'],
        },
        "oos_2024": {r['config']: {
            "preset": r['preset'], "top_n": r['top_n'],
            "sharpe": r['sharpe'], "total_return": r['total_return'],
            "max_drawdown": r['max_drawdown'], "win_rate": r['win_rate'], "trades": r['trades'],
        } for r in test_rows},
        "stability": stab,
        "twoyr_champion": twoyr_champ,
        "twoyr_ranking": [{"preset": p, "sharpe": sh, "total_return": ret} for p, sh, ret in twoyr_sorted],
        "robust_both_years": robust_both,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(lines))
    log(f"\n报告已写入: {OUT_LOG}")
    log(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
