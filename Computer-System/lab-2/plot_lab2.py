#!/usr/bin/env python3
"""Generate publication-quality figures for Lab 2 report from results/*.csv."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"

BASE_DIR = "/Users/rinalic/Local/Github/XJTU-Labs/Computer-System/lab-2"
RES_DIR = os.path.join(BASE_DIR, "results")
IMG_DIR = os.path.join(BASE_DIR, "lab-report", "images")
os.makedirs(IMG_DIR, exist_ok=True)


def load_csv(name):
    with open(os.path.join(RES_DIR, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def size_kb(s):
    s = s.strip().lower()
    if s.endswith("kb"):
        return float(s[:-2])
    if s.endswith("mb"):
        return float(s[:-2]) * 1024.0
    return float(s)


# 1. Task 1: Capacity curve (auto-latency on vs off)
def plot_capacity():
    on_rows = load_csv("task1_size.csv")
    off_rows = load_csv("task1_size_nolat.csv")
    xs = [size_kb(r["l1d-size"]) for r in on_rows]
    y_on = [float(r["cycles"]) / 1e6 for r in on_rows]
    y_off = [float(r["cycles"]) / 1e6 for r in off_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(xs, y_on, "o-", linewidth=2.2, markersize=7, color="#1f77b4", label="auto-latency on（默认延迟缩放）")
    ax.plot(xs, y_off, "s--", linewidth=2.0, markersize=6.5, color="#d62728", label="auto-latency off（固定 3 周期）")

    ax.axvline(100.7, color="#2ca02c", linestyle=":", linewidth=1.6, label="qnn 工作集大小 (≈100.7 KiB)")
    ax.annotate("拐点 128 KiB\n0.906M cycles", xy=(128, y_on[4]), xytext=(145, 0.96),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2), fontsize=9, color="#1f77b4")
    ax.annotate("64 KiB 最慢\n1.143M cycles", xy=(64, y_on[3]), xytext=(26, 1.155),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2), fontsize=9, color="#1f77b4")

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)} KiB" for x in xs])
    ax.set_xlabel("L1D Cache 容量（对数刻度）", fontsize=11)
    ax.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=11)
    ax.set_title("任务 1：L1D 容量扫描与延迟模型对比 (qnn, O3)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.84, 1.20)
    ax.legend(frameon=True, fontsize=9.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig1_capacity.png"))
    plt.close(fig)


# 2. Task 2: Associativity x Replacement Policy (64 KiB)
def plot_assoc_repl():
    rows = load_csv("task2_repl.csv")
    assocs = [1, 2, 4, 8]
    lru_mpki = [float(r["l1d_mshr_mpki"]) for r in rows if r["l1d-repl"] == "lru"]
    rnd_mpki = [float(r["l1d_mshr_mpki"]) for r in rows if r["l1d-repl"] == "random"]
    lru_cyc = [float(r["cycles"]) / 1e6 for r in rows if r["l1d-repl"] == "lru"]
    rnd_cyc = [float(r["cycles"]) / 1e6 for r in rows if r["l1d-repl"] == "random"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    x = np.arange(len(assocs))
    w = 0.34

    b1 = ax1.bar(x - w/2, lru_mpki, w, label="LRU", color="#d62728", alpha=0.88)
    b2 = ax1.bar(x + w/2, rnd_mpki, w, label="Random", color="#1f77b4", alpha=0.88)
    ax1.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax1.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{a} 路" for a in assocs])
    ax1.set_xlabel("相联度", fontsize=10.5)
    ax1.set_ylabel("l1d_mshr_mpki", fontsize=10.5)
    ax1.set_title("64 KiB L1D：真实失效率 (mshr_mpki)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 2.55)
    ax1.legend(fontsize=9)

    ax2.plot(assocs, lru_cyc, "o-", linewidth=2.2, markersize=7, color="#d62728", label="LRU")
    ax2.plot(assocs, rnd_cyc, "s-", linewidth=2.2, markersize=7, color="#1f77b4", label="Random")
    for a, c in zip(assocs, lru_cyc):
        ax2.annotate(f"{c:.3f}M", (a, c), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#d62728")
    for a, c in zip(assocs, rnd_cyc):
        off = -13 if a > 1 else -13
        ax2.annotate(f"{c:.3f}M", (a, c), textcoords="offset points", xytext=(0, off), ha="center", fontsize=8, color="#1f77b4")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(assocs)
    ax2.set_xticklabels([f"{a} 路" for a in assocs])
    ax2.set_xlabel("相联度", fontsize=10.5)
    ax2.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=10.5)
    ax2.set_title("64 KiB L1D：执行周期数对比", fontsize=11, fontweight="bold")
    ax2.set_ylim(1.04, 1.175)
    ax2.legend(fontsize=9)

    fig.suptitle("任务 2：相联度 × 替换策略交互效应 (qnn, 64 KiB L1D)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig2_assoc_repl.png"))
    plt.close(fig)


# 3. Task 3: Cacheline Size (Predicted vs Measured mshr_mpki & Cycles)
def plot_cacheline():
    rows = load_csv("task3a_line.csv")
    lines = [int(r["cacheline"]) for r in rows]
    meas_mpki = [float(r["l1d_mshr_mpki"]) for r in rows]
    pred_mpki = [(103117.0 / b) * 8.0 * 1000.0 / 5800000.0 for b in lines]
    cyc = [float(r["cycles"]) / 1e6 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))

    ax1.plot(lines, pred_mpki, "o--", linewidth=2.0, markersize=7, color="#ff7f0e", label="理论预测 (100.7 KiB / B)")
    ax1.plot(lines, meas_mpki, "s-", linewidth=2.0, markersize=6, color="#1f77b4", label="gem5 实测")
    for b, m in zip(lines, meas_mpki):
        ax1.annotate(f"{m:.2f}", (b, m), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(lines)
    ax1.set_xticklabels([f"{b} B" for b in lines])
    ax1.set_xlabel("Cache 块大小", fontsize=10.5)
    ax1.set_ylabel("l1d_mshr_mpki", fontsize=10.5)
    ax1.set_title("预测 vs. 实测 mshr_mpki (误差 ≈0.5%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 10.2)
    ax1.legend(fontsize=9)

    bars = ax2.bar([str(b) + " B" for b in lines], cyc, color=["#d62728", "#1f77b4", "#1f77b4", "#1f77b4", "#2ca02c"], alpha=0.88)
    ax2.bar_label(bars, fmt="%.3fM", fontsize=8.5, padding=3)
    ax2.annotate("I-Cache 取指崩塌\n(停顿占 66.6%)", xy=(0, cyc[0]), xytext=(0.9, 4.1),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3), fontsize=9, color="#d62728", fontweight="bold")
    ax2.set_xlabel("Cache 块大小", fontsize=10.5)
    ax2.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=10.5)
    ax2.set_title("块大小对执行周期数的影响", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 5.6)

    fig.suptitle("任务 3：Cache 块大小扫描 (qnn, 32 KiB L1D)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig3_cacheline.png"))
    plt.close(fig)


# 4. Section IV Q2: Latency sensitivity
def plot_latsens():
    rows = load_csv("q2_latsens.csv")
    sizes = [8, 32, 128, 256]
    lats = [2, 3, 5]
    colors = {2: "#2ca02c", 3: "#1f77b4", 5: "#d62728"}
    markers = {2: "^-", 3: "o-", 5: "s-"}

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    for lat in lats:
        ys = [float(r["cycles"]) / 1e6 for r in rows if int(r["l1-latency"]) == lat]
        ax.plot(sizes, ys, markers[lat], linewidth=2.1, markersize=7, color=colors[lat], label=f"基准延迟 l1-latency = {lat} 周期")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s} KiB" for s in sizes])
    ax.set_xlabel("L1D Cache 容量（对数刻度）", fontsize=11)
    ax.set_ylabel("ROI 执行周期数 (×10⁶ cycles)", fontsize=11)
    ax.set_title("思考题 2：基准命中延迟敏感性与结论稳健性", fontsize=12, fontweight="bold")
    ax.set_ylim(0.85, 1.23)
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "fig4_latsens.png"))
    plt.close(fig)


if __name__ == "__main__":
    plot_capacity()
    plot_assoc_repl()
    plot_cacheline()
    plot_latsens()
    print("Generated 4 figures in", IMG_DIR)
