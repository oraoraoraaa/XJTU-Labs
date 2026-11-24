#!/usr/bin/env python3
"""Generate 300-DPI publication figures for Lab 3 report from results/*.csv."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.28
plt.rcParams["grid.linestyle"] = "--"

BASE_DIR = "/Users/rinalic/Local/Github/XJTU-Labs/Computer-System/lab-3"
RES_DIR = os.path.join(BASE_DIR, "results")
IMG_DIR = os.path.join(BASE_DIR, "lab-report", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def load_csv(name):
    with open(os.path.join(RES_DIR, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# 1. Task 1a: L1D 64KiB/4-way all 15 policies (Victim-selection vs Insertion-position)
def plot_l1_repl():
    rows = {r["l1d-repl"]: r for r in load_csv("lab3_repl.csv")}
    group_a = ["lru", "fifo", "secondchance", "treeplru", "random", "mru", "lfu"]
    group_b = ["lip", "bip", "rrip", "brrip", "nru", "drrip", "ship", "ship_pc"]
    order = group_a + group_b

    misses = [float(rows[p]["l1d_mshr_misses"]) for p in order]
    cyc = [float(rows[p]["cycles"]) / 1e6 for p in order]
    colors = [
        ("#1f77b4" if p in ("lfu", "mru", "random") else "#7f8c8d")
        if p in group_a else "#d62728"
        for p in order
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    x = np.arange(len(order))

    b1 = ax1.bar(x, misses, width=0.62, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.5)
    ax1.axhline(12820, color="#333333", linestyle="--", linewidth=1.2, label="LRU 基线 (12,820 块)")
    ax1.axhline(4352, color="#2ca02c", linestyle="-.", linewidth=1.5, label="OPT 理论下界 (4,352 块)")
    ax1.axvline(len(group_a) - 0.5, color="#555555", linestyle=":", linewidth=1.4)
    ax1.bar_label(b1, fmt="%.0f", fontsize=7.5, padding=2)
    ax1.set_ylabel("L1D MSHR 失效数 (块)", fontsize=10.5)
    ax1.set_title("任务 1a：64 KiB / 4 路 L1D 下 15 种替换策略对比 (qnn, O3)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 15200)
    ax1.text(3.0, 14100, "（甲）牺牲者选择型", ha="center", fontsize=10, fontweight="bold", color="#1f4e78")
    ax1.text(10.5, 14100, "（乙）插入位置型（全部失效）", ha="center", fontsize=10, fontweight="bold", color="#a61c1c")

    p_eff = mpatches.Patch(color="#1f77b4", alpha=0.88, label="甲类有效策略 (LFU / MRU / Random)")
    p_ineff_a = mpatches.Patch(color="#7f8c8d", alpha=0.88, label="甲类无效策略 (LRU / FIFO / 2ndChance / TreePLRU)")
    p_b = mpatches.Patch(color="#d62728", alpha=0.88, label="乙类插入位置策略 (LIP~SHiP-PC)")
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles + [p_eff, p_ineff_a, p_b], fontsize=8.0, loc="lower left", ncol=2, frameon=True)

    b2 = ax2.bar(x, cyc, width=0.62, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.5)
    ax2.axhline(float(rows["lru"]["cycles"]) / 1e6, color="#333333", linestyle="--", linewidth=1.2)
    ax2.axvline(len(group_a) - 0.5, color="#555555", linestyle=":", linewidth=1.4)
    ax2.bar_label(b2, fmt="%.3f", fontsize=7.5, padding=2)
    ax2.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=10.5)
    ax2.set_ylim(0.95, 1.21)
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=25, ha="right", fontsize=9.5)
    ax2.set_xlabel("L1D 替换策略", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig1_l1_repl.png"))
    plt.close(fig)


# 2. Task 1b & 1c: Associativity self-check + Non-pathological 128 KiB control
def plot_assoc_nopath():
    assoc_rows = load_csv("lab3_assoc.csv")
    assocs = [2, 4, 8, 16]
    lru_m = [float(r["l1d_mshr_misses"]) for r in assoc_rows if r["l1d-repl"] == "lru"]
    mru_m = [float(r["l1d_mshr_misses"]) for r in assoc_rows if r["l1d-repl"] == "mru"]

    nopath_rows = {r["l1d-repl"]: r for r in load_csv("lab3_nopath.csv")}
    np_order = ["lru", "bip", "random", "mru"]
    np_cyc = [float(nopath_rows[p]["cycles"]) / 1e6 for p in np_order]
    np_miss = [int(float(nopath_rows[p]["l1d_mshr_misses"])) for p in np_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))

    ax1.plot(assocs, lru_m, "o-", lw=2.2, ms=7, color="#d62728", label="LRU（与相联度无关）")
    ax1.plot(assocs, mru_m, "s-", lw=2.2, ms=7, color="#1f77b4", label="MRU")
    ax1.axhline(4352, color="#2ca02c", linestyle="--", lw=1.4, label="OPT 下界 (4,352)")
    for a, y in zip(assocs, lru_m):
        ax1.annotate(f"{int(y):,}", (a, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#d62728")
    for a, y in zip(assocs, mru_m):
        ax1.annotate(f"{int(y):,}", (a, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color="#1f77b4")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(assocs)
    ax1.set_xticklabels([f"{a} 路" for a in assocs])
    ax1.set_ylim(3200, 14500)
    ax1.set_xlabel("L1D 相联度 (64 KiB 固定容量)", fontsize=10.5)
    ax1.set_ylabel("L1D MSHR 失效数 (块)", fontsize=10.5)
    ax1.set_title("任务 1b：相联度自检 (LRU 恒为 12,820)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="center", bbox_to_anchor=(0.62, 0.62), frameon=True)

    np_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax2.bar(np.arange(len(np_order)), np_cyc, width=0.52, color=np_colors, alpha=0.88, edgecolor="#222222", linewidth=0.5)
    for bar, c, m in zip(bars, np_cyc, np_miss):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            c + 0.0012,
            f"{c:.4f}M\n({m} misses)",
            ha="center", va="bottom", fontsize=8.0
        )
    ax2.set_xticks(np.arange(len(np_order)))
    ax2.set_xticklabels(np_order, fontsize=9.5)
    ax2.set_ylim(0.895, 0.932)
    ax2.set_xlabel("L1D 替换策略 (128 KiB / 4 路)", fontsize=10.5)
    ax2.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=10.5)
    ax2.set_title("任务 1c：对照组（病态消除后 MRU 最差）", fontsize=11, fontweight="bold")

    fig.suptitle("任务 1b/1c：相联度无关性与 128 KiB 无病态对照", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig2_assoc_nopath.png"))
    plt.close(fig)


# 3. Task 3c: L1 vs L2 rank inversion (64KiB/4-way)
def plot_l1_vs_l2():
    l1 = {r["l1d-repl"]: r for r in load_csv("lab3_repl.csv")}
    l2 = {r["l2-repl"]: r for r in load_csv("lab3_l2repl.csv")}
    l1_lru = float(l1["lru"]["l1d_mshr_misses"])
    l2_lru = float(l2["lru"]["l2_mshr_misses"])

    policies = [
        "lfu", "mru", "random", "treeplru", "drrip", "ship", "ship_pc",
        "brrip", "rrip", "nru", "lip", "bip", "secondchance", "fifo", "lru"
    ]
    l1_red = [(l1_lru - float(l1[p]["l1d_mshr_misses"])) / l1_lru * 100.0 for p in policies]
    l2_red = [(l2_lru - float(l2[p]["l2_mshr_misses"])) / l2_lru * 100.0 for p in policies]

    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    x = np.arange(len(policies))
    w = 0.37
    b1 = ax.bar(x - w / 2 - 0.02, l1_red, w, label="L1D (64 KiB/4 路, refsPerInsertion = 112.27)", color="#1f77b4", alpha=0.88)
    b2 = ax.bar(x + w / 2 + 0.02, l2_red, w, label="L2 (64 KiB/4 路, refsPerInsertion = 0.62)", color="#d62728", alpha=0.88)

    ax.axhline(0, color="#333333", linewidth=0.9)
    for idx, (bar, v) in enumerate(zip(b1, l1_red)):
        if abs(v) >= 2.0:
            dy = 1.1 if idx != 2 else 1.1
            ax.text(bar.get_x() + bar.get_width() / 2 - (0.04 if idx == 2 else 0.0), v + dy, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.2, color="#1f77b4")
    for idx, (bar, v) in enumerate(zip(b2, l2_red)):
        if abs(v) >= 2.0:
            dx = 0.06 if idx in (0, 2) else 0.0
            ax.text(bar.get_x() + bar.get_width() / 2 + dx, v + 1.1, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.2, color="#d62728")

    ax.annotate(
        "排名反转：\n• L1 最优：LFU (+45.8%) / MRU (+40.0%)\n• L2 最优：SHiP-PC (+43.1%) / BIP (+39.5%) / BRRIP (+39.4%)\n• L2 下 MRU 失效，仅改善 3.1%",
        xy=(6, 43.1), xytext=(8.2, 47.5),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8dc", ec="#c0a040", lw=1.0),
        fontsize=8.5, ha="left"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=28, ha="right", fontsize=9.5)
    ax.set_ylabel("相对同级 LRU 的 MSHR 失效降幅 (%)", fontsize=10.5)
    ax.set_ylim(-4, 61)
    ax.set_title("任务 3c 判决实验：相同 64 KiB/4 路下 L1D 与 L2 的策略效力反转", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig3_l1_vs_l2_inversion.png"))
    plt.close(fig)


# 4. Task 4: L1D prefetchers (cycles/accuracy & demand misses vs mshr_mpki)
def plot_prefetchers():
    rows = load_csv("lab3_pf.csv")
    names = [r["l1d-pf"] for r in rows]
    cyc = [float(r["cycles"]) / 1e6 for r in rows]
    acc = [float(r["l1d_pf_accuracy"]) * 100.0 for r in rows]
    misses = [float(r["l1d_misses"]) / 1000.0 for r in rows]
    mshr_mpki = [float(r["l1d_mshr_mpki"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.6))
    x = np.arange(len(names))

    bars1 = ax1.bar(x - 0.19, cyc, 0.35, color="#1f77b4", alpha=0.88, label="ROI 周期数 (×10⁶)")
    ax1.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=10, color="#1f77b4")
    ax1.set_ylim(0, 1.48)
    ax1.bar_label(bars1, fmt="%.3f", fontsize=7.5, padding=2, color="#1f77b4")

    ax1r = ax1.twinx()
    ax1r.grid(False)
    bars1r = ax1r.bar(x + 0.19, acc, 0.35, color="#ff7f0e", alpha=0.88, label="预取准确率 (%)")
    ax1r.set_ylabel("预取准确率 accuracy (%)", fontsize=10, color="#c85a00")
    ax1r.set_ylim(0, 134)
    ax1r.bar_label(bars1r, fmt="%.1f%%", fontsize=7.5, padding=2, color="#c85a00")

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9.5)
    ax1.set_xlabel("L1D 预取器", fontsize=10)
    ax1.set_title("周期数 vs. 预取准确率（代价指标）", fontsize=11, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h1r, l1r = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h1r, l1 + l1r, fontsize=8.0, loc="upper center", ncol=2)

    bars2 = ax2.bar(x - 0.19, misses, 0.35, color="#2ca02c", alpha=0.88, label="需求失效 l1d_misses (千次)")
    ax2.set_ylabel("需求失效 l1d_misses (×10³)", fontsize=10, color="#2ca02c")
    ax2.set_ylim(0, 275)
    ax2.bar_label(bars2, fmt="%.1fk", fontsize=7.5, padding=2, color="#1f7a1f")

    ax2r = ax2.twinx()
    ax2r.grid(False)
    ax2r.plot(x + 0.19, mshr_mpki, "D-", lw=2.0, ms=6.0, color="#d62728", label="真实下行流量 l1d_mshr_mpki")
    for xi, v in zip(x + 0.19, mshr_mpki):
        ax2r.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.8, color="#d62728")
    ax2r.set_ylabel("l1d_mshr_mpki", fontsize=10, color="#d62728")
    ax2r.set_ylim(0, 4.4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.5)
    ax2.set_xlabel("L1D 预取器", fontsize=10)
    ax2.set_title("需求失效骤降 vs. 下行流量不变", fontsize=11, fontweight="bold")
    h2, l2 = ax2.get_legend_handles_labels()
    h2r, l2r = ax2r.get_legend_handles_labels()
    ax2.legend(h2 + h2r, l2 + l2r, fontsize=8.0, loc="upper center")

    fig.suptitle("任务 4：32 KiB / 2 路 L1D 下五种硬件预取器对比 (qnn, O3)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig4_prefetchers.png"))
    plt.close(fig)


# 5. Task 5 & 6: Prefetch vs Capacity + pchase bandwidth blowup
def plot_pf_vs_size_pchase():
    t5 = {(r["l1d-size"], r["l1d-pf"]): r for r in load_csv("lab3_pf_vs_size.csv")}
    sizes = ["32kB", "64kB", "128kB"]
    none_cyc = [float(t5[(s, "none")]["cycles"]) / 1e6 for s in sizes]
    str_cyc = [float(t5[(s, "stride")]["cycles"]) / 1e6 for s in sizes]
    none_mpki = [float(t5[(s, "none")]["l1d_mshr_mpki"]) for s in sizes]
    str_mpki = [float(t5[(s, "stride")]["l1d_mshr_mpki"]) for s in sizes]

    t6 = load_csv("lab3_pf_pchase.csv")
    t6_names = [r["l1d-pf"] for r in t6]
    t6_cyc = [float(r["cycles"]) / 1e6 for r in t6]
    t6_mpki = [float(r["l1d_mshr_mpki"]) for r in t6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.7))

    x = np.arange(len(sizes))
    w = 0.34
    b1 = ax1.bar(x - w / 2, none_cyc, w, color="#7f8c8d", alpha=0.88, label="无预取 (cycles)")
    b2 = ax1.bar(x + w / 2, str_cyc, w, color="#1f77b4", alpha=0.88, label="Stride 预取 (cycles)")
    ax1.bar_label(b1, fmt="%.3fM", fontsize=7.5, padding=2)
    ax1.bar_label(b2, fmt="%.3fM", fontsize=7.5, padding=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("kB", " KiB") for s in sizes], fontsize=9.5)
    ax1.set_xlabel("L1D 容量", fontsize=10)
    ax1.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=10)
    ax1.set_ylim(0, 1.88)

    ax1r = ax1.twinx()
    ax1r.grid(False)
    ax1r.plot(x - w / 2, none_mpki, "o--", lw=1.8, ms=6, color="#2ca02c", label="无预取 mshr_mpki")
    ax1r.plot(x + w / 2, str_mpki, "s-", lw=1.8, ms=6, color="#d62728", label="Stride mshr_mpki")
    for idx, (xi, v) in enumerate(zip(x - w / 2, none_mpki)):
        dy = 8 if idx < 2 else 10
        ax1r.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5, color="#1f7a1f")
    for idx, (xi, v) in enumerate(zip(x + w / 2, str_mpki)):
        dy = 8 if idx < 2 else 10
        ax1r.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5, color="#d62728")
    ax1r.set_ylabel("l1d_mshr_mpki", fontsize=10, color="#d62728")
    ax1r.set_ylim(-0.25, 3.15)
    ax1.set_title("任务 5：预取 vs. 加容量 (qnn)", fontsize=11, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h1r, l1r = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h1r, l1 + l1r, fontsize=7.5, loc="upper center", ncol=2)

    x2 = np.arange(len(t6_names))
    b3 = ax2.bar(x2 - 0.19, t6_cyc, 0.35, color="#7f8c8d", alpha=0.88, label="ROI 周期数 (×10⁶)")
    ax2.bar_label(b3, fmt="%.2fM", fontsize=7.8, padding=2)
    ax2.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=10)
    ax2.set_ylim(0, 5.95)

    ax2r = ax2.twinx()
    ax2r.grid(False)
    b4 = ax2r.bar(x2 + 0.19, t6_mpki, 0.35, color="#d62728", alpha=0.88, label="l1d_mshr_mpki")
    ax2r.bar_label(b4, fmt="%.1f", fontsize=7.8, padding=2, color="#a61c1c")
    ax2r.set_ylabel("l1d_mshr_mpki", fontsize=10, color="#d62728")
    ax2r.set_ylim(0, 600)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(t6_names, fontsize=9.5)
    ax2.set_xlabel("L1D 预取器 (pchase, 32 KiB)", fontsize=10)
    ax2.set_title("任务 6：无局部性负载下的带宽膨胀 (pchase)", fontsize=11, fontweight="bold")
    h2, l2 = ax2.get_legend_handles_labels()
    h2r, l2r = ax2r.get_legend_handles_labels()
    ax2.legend(h2 + h2r, l2 + l2r, fontsize=8.0, loc="upper center", ncol=2)

    fig.suptitle("任务 5 与任务 6：预取器的收益机制差异及无局部性代价", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig5_pf_vs_size_pchase.png"))
    plt.close(fig)


if __name__ == "__main__":
    plot_l1_repl()
    plot_assoc_nopath()
    plot_l1_vs_l2()
    plot_prefetchers()
    plot_pf_vs_size_pchase()
    print("Generated 5 figures in", IMG_DIR)
