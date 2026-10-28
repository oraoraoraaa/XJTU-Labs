"""
stats.py —— 解析 gem5 的 stats.txt

gem5 每次遇到 dump_stats 伪指令都会追加一段统计块。基准程序在 ROI 的入口和
出口各发射一次，程序退出时 gem5 自动再来一次，所以 stats.txt 里有三段：

    第 0 段 = 程序开始 -> ROI_BEGIN    启动段（libc 初始化等）
    第 1 段 = ROI_BEGIN -> ROI_END     ★ 内核，默认取这一段
    第 2 段 = ROI_END -> 程序退出      收尾段

默认取第 1 段。这一点很重要：静态链接的 libc 启动要跑掉上百万条指令，
而且 IPC 极低，混进来会把测量结果彻底带偏。

用 -DNO_M5OPS 编译的版本没有标记，整个文件只有一段，此时 block 参数越界
会自动回退到第 0 段（即整个程序）。
"""

import re

BLOCK_START = "---------- Begin Simulation Statistics ----------"

ROI_BLOCK = 1  # 三段布局里 ROI 的下标

_LINE = re.compile(r"^(\S+)\s+([-\d.eE+naninf]+)")


def parse_blocks(path):
    """把 stats.txt 拆成若干个 {统计名: 数值} 字典，按出现顺序返回。"""
    blocks, current = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if BLOCK_START in line:
                if current is not None:
                    blocks.append(current)
                current = {}
                continue
            if current is None:
                continue
            m = _LINE.match(line)
            if not m:
                continue
            name, raw = m.group(1), m.group(2)
            try:
                current[name] = float(raw)
            except ValueError:
                pass
    if current:
        blocks.append(current)
    return blocks


def load(path, block=ROI_BLOCK):
    blocks = parse_blocks(path)
    if not blocks:
        raise ValueError(f"{path} 中没有找到任何统计块")
    if block >= len(blocks):
        block = 0  # 没有 ROI 标记的版本只有一段
    return blocks[block]


def get(stats, *names, default=0.0):
    """按顺序尝试多个统计名，返回第一个存在的。gem5 不同版本/不同 CPU 模型
    的统计名不完全一致，用这个函数避免到处写 if。"""
    for n in names:
        if n in stats:
            return stats[n]
    return default


def _ratio(num, den):
    return (num / den) if den else 0.0


def summarize(stats):
    """把一次仿真压缩成一行关键指标。这是所有实验共用的评价口径。"""
    insts = get(stats, "simInsts")
    cycles = get(stats, "system.cpu.numCycles")
    seconds = get(stats, "simSeconds")

    ipc = get(stats, "system.cpu.ipc")
    if not ipc and cycles:
        ipc = insts / cycles

    bp_lookups = get(stats, "system.cpu.branchPred.lookups_0::total",
                     "system.cpu.branchPred.lookups::total")
    bp_miss = get(stats, "system.cpu.branchPred.mispredicted_0::total",
                  "system.cpu.branchPred.mispredicted::total")
    # 分母必须用"最终提交的分支数"，不能用 lookups。
    # lookups 把错误路径上取出来又被squash掉的预测也算了进去，误预测越多分母
    # 涨得越快，算出来的"误预测率"反而越低——一个越差越好看的指标。
    bp_committed = get(stats, "system.cpu.branchPred.committed_0::total",
                       "system.cpu.branchPred.committed::total")

    row = {
        "insts": insts,
        "cycles": cycles,
        "sim_seconds": seconds,
        "ipc": ipc,
        "cpi": (1.0 / ipc) if ipc else 0.0,
        "host_seconds": get(stats, "hostSeconds"),
        "host_inst_rate": get(stats, "hostInstRate"),
    }

    for level, prefix in (
        ("l1i", "system.cpu.icache"),
        ("l1d", "system.cpu.dcache"),
        ("l2", "system.l2cache"),
    ):
        acc = get(stats, f"{prefix}.overallAccesses::total")
        miss = get(stats, f"{prefix}.overallMisses::total")
        # MSHR 失效才是真正发往下一级的请求数：一次失效如果发现同一块已经
        # 有请求在路上，就只是挂到已有的 MSHR 上（MshrHit），不产生新流量。
        # 乱序核上这两个数能差十几倍，只看 miss_rate 会得出错误结论。
        mshr_miss = get(stats, f"{prefix}.overallMshrMisses::total")
        row[f"{level}_accesses"] = acc
        row[f"{level}_misses"] = miss
        row[f"{level}_miss_rate"] = (miss / acc) if acc else 0.0
        row[f"{level}_mpki"] = (miss * 1000.0 / insts) if insts else 0.0
        row[f"{level}_mshr_misses"] = mshr_miss
        row[f"{level}_mshr_miss_rate"] = (mshr_miss / acc) if acc else 0.0
        row[f"{level}_mshr_mpki"] = (mshr_miss * 1000.0 / insts) if insts else 0.0
        row[f"{level}_mlp"] = (miss / mshr_miss) if mshr_miss else 0.0
        row[f"{level}_avg_miss_lat_ticks"] = get(
            stats, f"{prefix}.overallAvgMissLatency::total"
        )
        row[f"{level}_avg_mshr_lat_ticks"] = get(
            stats, f"{prefix}.overallAvgMshrMissLatency::total"
        )

    row["bp_lookups"] = bp_lookups
    row["bp_committed"] = bp_committed
    row["bp_mispredicts"] = bp_miss
    # ★ 主指标：误预测数 / 提交的分支数
    row["bp_miss_rate"] = _ratio(bp_miss, bp_committed)
    # 保留旧口径只是为了让学生看到"换个分母，同一次运行可以给出两个数"
    row["bp_miss_rate_lookups"] = _ratio(bp_miss, bp_lookups)
    row["bp_mpki"] = (bp_miss * 1000.0 / insts) if insts else 0.0

    # 方向预测错和目标预测错是两件独立的事，硬件上也是两个部件（预测器 / BTB）。
    # 第 4 次实验要用这两个数说明"换更强的预测器"和"加大 BTB"解决的不是同一个问题。
    row["bp_miss_predictor"] = get(
        stats, "system.cpu.branchPred.mispredictDueToPredictor_0::total")
    row["bp_miss_btb"] = get(
        stats, "system.cpu.branchPred.mispredictDueToBTBMiss_0::total")
    row["btb_hit_rate"] = _ratio(
        get(stats, "system.cpu.branchPred.targetProvider_0::BTB"),
        get(stats, "system.cpu.branchPred.targetProvider_0::total"))

    # ------------------------------------------------------------------
    # 乱序核内部：谁把流水线堵住了（第 4 次实验的主指标）
    #
    # rename 级是唯一同时看得见全部后端资源的地方：一条指令要进入后端，
    # 必须同时拿到 ROB 表项、IQ 表项、LQ/SQ 表项和物理寄存器。任何一个拿不到，
    # rename 就停住并记一次对应的 FullEvent。所以"哪个 FullEvent 最多"
    # 就是"该给哪个部件加资源"的直接答案——不需要猜。
    # ------------------------------------------------------------------
    cycles_nz = cycles or 1.0
    row["rob_full_events"] = get(stats, "system.cpu.rename.ROBFullEvents")
    row["iq_full_events"] = get(stats, "system.cpu.rename.IQFullEvents")
    row["lq_full_events"] = get(stats, "system.cpu.rename.LQFullEvents")
    row["sq_full_events"] = get(stats, "system.cpu.rename.SQFullEvents")
    row["rename_block_cycles"] = get(stats, "system.cpu.rename.blockCycles")
    row["rename_block_frac"] = row["rename_block_cycles"] / cycles_nz

    # 推测执行的代价：取出来、算了、最后因为猜错而被丢掉的指令。
    # squash_ratio = 白做的功 / 有用的功。
    squashed = get(stats, "system.cpu.commit.commitSquashedInsts")
    row["squashed_insts"] = squashed
    row["squash_ratio"] = _ratio(squashed, insts)
    row["commit_branch_mispredicts"] = get(
        stats, "system.cpu.commit.branchMispredicts")

    # 取指侧停顿：第 2 次实验发现"数据 Cache 的块大小拖垮取指"就是靠这个。
    row["icache_stall_cycles"] = get(
        stats, "system.cpu.fetchStats0.icacheStallCycles")
    row["icache_stall_frac"] = row["icache_stall_cycles"] / cycles_nz
    row["fetch_cache_lines"] = get(stats, "system.cpu.fetch.cacheLines")

    # load 从发出到数据可用的平均周期数。加大 ROB 能不能起作用，看的就是
    # "有没有足够长的延迟值得去隐藏"。
    row["load_to_use"] = get(stats, "system.cpu.lsq0.loadToUse::mean")
    row["mem_order_violations"] = get(
        stats, "system.cpu.iew.memOrderViolationEvents")
    row["iq_full_iew"] = get(stats, "system.cpu.iew.iqFullEvents")
    row["lsq_full_iew"] = get(stats, "system.cpu.iew.lsqFullEvents")

    # ------------------------------------------------------------------
    # 预取器（第 3 次实验）
    #
    # 预取的两个代价必须分开看：
    #   准确率 accuracy = 有用的预取 / 发出的预取   —— 白发的预取浪费带宽和能量
    #   覆盖率 coverage = 被预取挡掉的失效 / 原本的失效 —— 没挡住的还是要等
    # 一个"准确率高但覆盖率低"的预取器很省电但没用；反过来则很有用但很费电。
    # ------------------------------------------------------------------
    for level, prefix in (("l1d", "system.cpu.dcache"),
                          ("l2", "system.l2cache")):
        issued = get(stats, f"{prefix}.prefetcher.pfIssued")
        useful = get(stats, f"{prefix}.prefetcher.pfUseful")
        row[f"{level}_pf_issued"] = issued
        row[f"{level}_pf_useful"] = useful
        row[f"{level}_pf_unused"] = get(stats, f"{prefix}.prefetcher.pfUnused")
        row[f"{level}_pf_late"] = get(stats, f"{prefix}.prefetcher.pfLate")
        # gem5 自己也算了 accuracy/coverage，优先用它的；缺失时用比值兜底
        row[f"{level}_pf_accuracy"] = get(
            stats, f"{prefix}.prefetcher.accuracy", default=_ratio(useful, issued))
        row[f"{level}_pf_coverage"] = get(stats, f"{prefix}.prefetcher.coverage")

    # 主存流量。第 2 次实验用它看"块变大 → 失效变少但搬的字节变多"，
    # 第 3 次看预取器的带宽代价，第 6 次算 Roofline 的操作强度。
    dram_r = get(stats, "system.mem_ctrl.dram.bytesRead::total")
    dram_w = get(stats, "system.mem_ctrl.dram.bytesWritten::total")
    row["dram_read_bytes"] = dram_r
    row["dram_write_bytes"] = dram_w
    row["dram_bytes"] = dram_r + dram_w
    row["dram_bytes_per_ki"] = (
        (dram_r + dram_w) * 1000.0 / insts) if insts else 0.0
    row["dram_bw_MBps"] = (
        (dram_r + dram_w) / seconds / 1e6) if seconds else 0.0

    # ------------------------------------------------------------------
    # 加速器（第 6 次实验）
    #
    # 这一组数就是 Roofline 图的三个坐标：
    #   accel_macs_per_byte   横轴，算术强度。只跟算法和 batch 有关，跟硬件无关
    #   accel_macs_per_cycle  纵轴，实测算力。上界是 pe
    #   accel_bytes_per_cycle 斜边，实测带宽。上界由挂载点决定（L2 总线 or 内存总线）
    #
    # 判断"这次跑在屋顶的哪一段"只要一句话：
    #   stall_frac 接近 0 → 撞到了算力顶（pe 不够）
    #   stall_frac 很大   → 撞到了带宽斜边（挂错地方了 / outstanding 太小）
    # 没有加速器时这些统计名不存在，get() 全部返回 0，CSV 里就是一列 0。
    # ------------------------------------------------------------------
    a_jobs = get(stats, "system.accel.jobs")
    a_macs = get(stats, "system.accel.macs")
    a_bytes = get(stats, "system.accel.bytesRead")
    a_job_cyc = get(stats, "system.accel.jobCycles")
    a_busy = get(stats, "system.accel.busyCycles")
    a_stall = get(stats, "system.accel.stallCycles")
    row["accel_jobs"] = a_jobs
    row["accel_macs"] = a_macs
    row["accel_bytes"] = a_bytes
    row["accel_line_reads"] = get(stats, "system.accel.lineReads")
    row["accel_job_cycles"] = a_job_cyc
    row["accel_busy_cycles"] = a_busy
    row["accel_stall_cycles"] = a_stall
    # 优先用 gem5 自己算的 Formula；老 stats.txt 里没有就用比值兜底
    row["accel_bytes_per_cycle"] = get(
        stats, "system.accel.bytesPerCycle", default=_ratio(a_bytes, a_job_cyc))
    row["accel_macs_per_cycle"] = get(
        stats, "system.accel.macsPerCycle", default=_ratio(a_macs, a_job_cyc))
    row["accel_macs_per_byte"] = get(
        stats, "system.accel.macsPerByte", default=_ratio(a_macs, a_bytes))
    # 停顿占比：一眼看出是算力受限还是带宽受限
    row["accel_stall_frac"] = _ratio(a_stall, a_job_cyc)
    # 加速器忙的时候 CPU 在干什么？accel_job_frac 小 = Amdahl 的另一半在拖后腿
    row["accel_job_frac"] = _ratio(a_job_cyc, cycles)

    # 把"tick"换算成周期，方便学生直接跟 Cache 配置里的延迟参数对照。
    tpc = (get(stats, "simTicks") / cycles) if cycles else 0.0
    row["ticks_per_cycle"] = tpc
    if tpc:
        for level in ("l1i", "l1d", "l2"):
            row[f"{level}_avg_miss_lat_cycles"] = (
                row[f"{level}_avg_miss_lat_ticks"] / tpc)
            row[f"{level}_avg_mshr_lat_cycles"] = (
                row[f"{level}_avg_mshr_lat_ticks"] / tpc)
    else:
        for level in ("l1i", "l1d", "l2"):
            row[f"{level}_avg_miss_lat_cycles"] = 0.0
            row[f"{level}_avg_mshr_lat_cycles"] = 0.0

    return row


def summarize_file(path, block=ROI_BLOCK):
    return summarize(load(path, block))


if __name__ == "__main__":
    import json
    import sys

    blk = int(sys.argv[2]) if len(sys.argv) > 2 else ROI_BLOCK
    print(json.dumps(summarize_file(sys.argv[1], blk), indent=2))
