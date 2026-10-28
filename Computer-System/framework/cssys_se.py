"""
cssys_se.py —— 计算机系统综合设计实验 统一 gem5 SE 模式配置脚本

七次实验共用这一个脚本，区别只在命令行参数。这样做的好处是：你每次实验
调整的东西，都能直接对应到一条命令行参数，也就对应到真实处理器里的一个
设计决策。

用法示例：
  gem5.opt --outdir=out/base framework/cssys_se.py \
      --cmd bench/bin/riscv/qnn --options "16" --cpu o3 \
      --l1d-size 32kB --l2-size 256kB

重要设计：Cache 的访问延迟默认随容量自动增长（--auto-latency）。
真实芯片里"容量翻倍"必然带来"访问变慢"，如果延迟固定不变，
设计空间探索就会退化成"越大越好"这种没有意义的结论。
"""

import argparse

import m5
from m5.objects import *

# ---------------------------------------------------------------------------
# 1. 可选组件表：把 gem5 的类名映射成简短的命令行取值
# ---------------------------------------------------------------------------

CPU_MODELS = {
    "atomic": RiscvAtomicSimpleCPU,  # 功能级，不建模时序，最快
    "timing": RiscvTimingSimpleCPU,  # 简单时序，单发射顺序，访存阻塞
    "minor": RiscvMinorCPU,  # 顺序流水线，建模到流水级
    "o3": RiscvO3CPU,  # 乱序超标量，最接近真实高性能核
}

BRANCH_PREDICTORS = {
    "none": None,
    "local": LocalBP,  # 局部历史，两位饱和计数器
    "bimode": BiModeBP,  # 双模式，缓解别名冲突
    "tournament": TournamentBP,  # 竞赛式，局部+全局动态选择
    "tage": TAGE,  # 多长度几何历史
    "ltage": LTAGE,  # TAGE + 循环预测器
    "tage_sc_l": TAGE_SC_L_64KB,  # 业界最强之一
    "perceptron": MultiperspectivePerceptron64KB,
}

PREFETCHERS = {
    "none": None,
    "tagged": TaggedPrefetcher,  # 下一行预取
    "stride": StridePrefetcher,  # 按 PC 识别定步长
    "bop": BOPPrefetcher,  # Best-Offset
    "spp": SignaturePathPrefetcher,  # 基于访问签名的路径预取
    "ampm": AMPMPrefetcher,
    "isb": IrregularStreamBufferPrefetcher,  # 面向不规则访存
}

REPLACEMENT_POLICIES = {
    "lru": LRURP,
    "random": RandomRP,
    "fifo": FIFORP,
    "lfu": LFURP,
    "lip": LIPRP,
    "mru": MRURP,
    "nru": NRURP,
    "bip": BIPRP,
    "brrip": BRRIPRP,
    # DRRIP 用 set dueling 在 SRRIP 和 BRRIP 之间动态选，两个尺寸参数
    # gem5 没给默认值，不填会直接 fatal。取值只要保证 team_size <= 相联度即可。
    "drrip": lambda: DRRIPRP(constituency_size=16, team_size=2),
    "rrip": RRIPRP,
    # SHiPRP 本身是抽象类，必须挑一个具体的索引方式
    "ship": SHiPMemRP,  # 用内存地址索引 SHCT
    "ship_pc": SHiPPCRP,  # 用 PC 索引 SHCT（论文原版）
    "treeplru": TreePLRURP,
    "secondchance": SecondChanceRP,
}

# 实验三：学生自己写进 gem5 的替换策略。只有重新编译过 gem5 的人才有这个类，
# 所以用 try 包起来——没做实验三的人照样能跑实验一、二。
try:
    REPLACEMENT_POLICIES["cssys"] = CssysRP  # 自己写的，LRU 口径
    REPLACEMENT_POLICIES["cssys_mru"] = lambda: CssysRP(evict_mru=True)
except NameError:
    pass

# 实验六：学生自己写进 gem5 的加速器。和上面的 CssysRP 同理——
# 没编进去的人照样能跑前五次实验，只是不能用 --accel。
try:
    HAVE_ACCEL = Dotp8Accel is not None
except NameError:
    HAVE_ACCEL = False

MEM_TYPES = {
    "DDR3_1600_8x8": DDR3_1600_8x8,
    "DDR3_2133_8x8": DDR3_2133_8x8,
    "DDR4_2400_8x8": DDR4_2400_8x8,
    "DDR5_4400_4x8": DDR5_4400_4x8,
    "LPDDR3_1600_1x32": LPDDR3_1600_1x32,
    "HBM_1000_4H_1x128": HBM_1000_4H_1x128,
}


# ---------------------------------------------------------------------------
# 2. 命令行参数
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="计算机系统综合设计实验 gem5 配置")

    w = p.add_argument_group("工作负载")
    w.add_argument("--cmd", required=True, help="RISC-V 静态可执行文件路径")
    w.add_argument("--options", default="", help="传给被测程序的参数，用引号包起来")
    w.add_argument("--maxinsts", type=int, default=0, help="最大模拟指令数，0 表示不限制")

    c = p.add_argument_group("处理器核")
    c.add_argument("--cpu", choices=CPU_MODELS, default="timing")
    c.add_argument("--freq", default="2GHz")

    o3 = p.add_argument_group("乱序核参数（仅 --cpu o3 生效）")
    o3.add_argument("--width", type=int, default=0, help="一次性设定取指/译码/重命名/发射/提交宽度")
    o3.add_argument("--fetch-width", type=int, default=0)
    o3.add_argument("--issue-width", type=int, default=0)
    o3.add_argument("--commit-width", type=int, default=0)
    o3.add_argument("--wb-width", type=int, default=0,
                    help="写回宽度。调小于 4 会触发 gem5 的级间缓冲断言，见 configure_o3")
    o3.add_argument("--rob", type=int, default=0, help="重排序缓冲(ROB)表项数")
    o3.add_argument("--iq", type=int, default=0, help="发射队列表项数")
    o3.add_argument("--lq", type=int, default=0, help="load 队列表项数")
    o3.add_argument("--sq", type=int, default=0, help="store 队列表项数")
    o3.add_argument("--int-regs", type=int, default=0, help="物理整数寄存器数")
    o3.add_argument("--bp", choices=BRANCH_PREDICTORS, default="tournament")
    o3.add_argument("--btb-entries", type=int, default=0)
    o3.add_argument("--dotp8-lat", type=int, default=0,
                    help="自定义指令 dotp8 的执行延迟(周期)。0 表示用 gem5 默认的 1 拍——"
                         "那等于宣称 8 路 int8 乘法加一棵加法树一拍就能算完")
    o3.add_argument("--dotp8-units", type=int, default=0,
                    help="能执行 dotp8 的功能单元个数，0 表示保持默认 4")
    o3.add_argument("--fetch-buffer", type=int, default=0,
                    help="取指缓冲大小(字节)，0 表示跟随 --cacheline。"
                         "必须 ≤ cacheline 且能整除它（gem5 会 fatal）")

    m = p.add_argument_group("存储层次")
    m.add_argument("--no-caches", action="store_true", help="完全不接 Cache，CPU 直连内存")
    m.add_argument("--no-l2", action="store_true", help="只有 L1，没有 L2")
    m.add_argument("--cacheline", type=int, default=64, help="Cache 块大小(字节)")
    m.add_argument("--l1i-size", default="32kB")
    m.add_argument("--l1i-assoc", type=int, default=2)
    m.add_argument("--l1d-size", default="32kB")
    m.add_argument("--l1d-assoc", type=int, default=2)
    m.add_argument("--l1d-mshrs", type=int, default=4)
    m.add_argument("--l2-size", default="256kB")
    m.add_argument("--l2-assoc", type=int, default=8)
    m.add_argument("--l1d-repl", choices=REPLACEMENT_POLICIES, default="lru")
    m.add_argument("--l2-repl", choices=REPLACEMENT_POLICIES, default="lru")
    m.add_argument("--l1d-pf", choices=PREFETCHERS, default="none")
    m.add_argument("--l2-pf", choices=PREFETCHERS, default="none")
    m.add_argument(
        "--auto-latency",
        choices=["on", "off"],
        default="on",
        help="Cache 延迟随容量自动增长；关掉它会得到'越大越好'的假结论",
    )
    # 基准点取真实芯片的量级：32kB 的 L1D 大约 3~4 周期，256kB 的 L2 大约 12 周期
    m.add_argument("--l1-latency", type=int, default=3, help="32kB 基准点的 L1 延迟(周期)")
    m.add_argument("--l2-latency", type=int, default=12, help="256kB 基准点的 L2 延迟(周期)")

    a = p.add_argument_group("实验六：dotp8 加速器")
    a.add_argument(
        "--accel",
        choices=["off", "l2bus", "membus"],
        default="off",
        help="加速器的取数端口接在哪一级。这是本次实验最重要的一个决策："
             "l2bus 表示和 L1 同级、共享 L2；membus 表示旁路整个 Cache 层次"
             "直连内存（真实 SoC 里挂在 NoC 上的加速器就是这样）",
    )
    a.add_argument("--accel-pe", type=int, default=8,
                   help="加速器每周期能做多少次 int8 乘加(算力)")
    a.add_argument("--accel-outstanding", type=int, default=8,
                   help="最多允许多少个访存请求同时在路上(访存并行度)")
    a.add_argument("--accel-start-lat", type=int, default=10,
                   help="收到 GO 到发出第一个访存请求的固定开销(周期)")
    a.add_argument("--accel-addr", type=lambda s: int(s, 0), default=0x40000000,
                   help="MMIO 寄存器的物理地址。必须和编译 qnn_accel 时的 "
                        "ACCEL_BASE 一致，且落在主存范围之外")

    d = p.add_argument_group("主存")
    d.add_argument("--mem-type", choices=MEM_TYPES, default="DDR3_1600_8x8")
    d.add_argument("--mem-size", default="512MB")
    d.add_argument("--mem-channels", type=int, default=1)

    return p.parse_args()


# ---------------------------------------------------------------------------
# 3. 延迟模型：容量越大，访问越慢
# ---------------------------------------------------------------------------


def size_to_kb(text):
    t = text.strip().lower()
    if t.endswith("kb"):
        return float(t[:-2])
    if t.endswith("mb"):
        return float(t[:-2]) * 1024
    if t.endswith("b"):
        return float(t[:-1]) / 1024
    return float(t)


def scaled_latency(size_text, ref_kb, base_cycles, cycles_per_doubling):
    """容量每翻一倍，访问延迟增加 cycles_per_doubling 个周期。

    这是对 CACTI 那类面积/延迟模型的极简近似，量级上是对的：
    真实设计中 32kB->64kB 的 L1 通常要多花 1 个周期。
    """
    import math

    kb = size_to_kb(size_text)
    doublings = math.log2(max(kb, 1.0) / ref_kb)
    return max(1, int(round(base_cycles + cycles_per_doubling * doublings)))


# ---------------------------------------------------------------------------
# 4. Cache 类定义
# ---------------------------------------------------------------------------


class L1ICache(Cache):
    mshrs = 4
    tgts_per_mshr = 20
    writeback_clean = True
    is_read_only = True


class L1DCache(Cache):
    tgts_per_mshr = 20
    writeback_clean = False


class L2Cache(Cache):
    mshrs = 20
    tgts_per_mshr = 12
    write_buffers = 8
    writeback_clean = False


def make_component(table, key):
    cls = table[key]
    return None if cls is None else cls()


def make_repl(key, assoc):
    """替换策略工厂。

    大部分策略无参可构造，但 DRRIP 用 set dueling，gem5 硬性要求
    team_size 必须正好等于相联度（dueling_rp.cc:115 的 panic），
    所以它必须知道这一级 Cache 有几路。
    """
    if key == "drrip":
        return DRRIPRP(constituency_size=8 * assoc, team_size=assoc)
    return make_component(REPLACEMENT_POLICIES, key)


# ---------------------------------------------------------------------------
# 5. 组装系统
# ---------------------------------------------------------------------------


def build_system(args):
    system = System()
    system.clk_domain = SrcClockDomain(
        clock=args.freq, voltage_domain=VoltageDomain()
    )
    system.mem_mode = "atomic" if args.cpu == "atomic" else "timing"
    system.mem_ranges = [AddrRange(args.mem_size)]
    system.cache_line_size = args.cacheline

    system.cpu = CPU_MODELS[args.cpu]()

    if args.cpu == "o3":
        configure_o3(system.cpu, args)
        # O3 的取指缓冲不能大于一个 Cache 块，否则 gem5 直接 fatal。
        # 扫描块大小时会扫到 16B/32B，这里自动跟着调小。
        fb = args.fetch_buffer or min(64, args.cacheline)
        if fb > args.cacheline or args.cacheline % fb:
            raise SystemExit(
                "--fetch-buffer %d 非法：必须 <= --cacheline %d 且能整除它"
                % (fb, args.cacheline))
        system.cpu.fetchBufferSize = fb

    system.membus = SystemXBar()

    if args.no_caches:
        system.cpu.icache_port = system.membus.cpu_side_ports
        system.cpu.dcache_port = system.membus.cpu_side_ports
        system.cpu.mmu.connectWalkerPorts(
            system.membus.cpu_side_ports, system.membus.cpu_side_ports
        )
    else:
        attach_caches(system, args)

    system.cpu.createInterruptController()
    system.system_port = system.membus.cpu_side_ports

    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = MEM_TYPES[args.mem_type](range=system.mem_ranges[0])
    system.mem_ctrl.port = system.membus.mem_side_ports

    if args.accel != "off":
        attach_accel(system, args)

    return system


def attach_accel(system, args):
    """把 dotp8 加速器接进系统。

    它有两个端口，接的地方完全不同，这一点是本次实验的第一个知识点：

      cpu_side —— 一组 MMIO 寄存器，永远挂在 membus 上。CPU 通过它下命令、
                  等结果。一条命令只有 6 次 8 字节访问，接在哪里都无所谓。
      mem_side —— 加速器自己的取数端口。98 KB 权重全从这里进来，
                  **接在哪一级决定了本次实验的全部结论**。
    """
    if not HAVE_ACCEL:
        raise SystemExit(
            "gem5 里没有 Dotp8Accel。请先把 src/cssys/ 拷进 gem5 源码树"
            "并重新编译（见实验六指导书任务 1）"
        )

    from m5.util import convert

    mem_end = convert.toMemorySize(args.mem_size)
    if args.accel_addr < mem_end:
        raise SystemExit(
            "--accel-addr %#x 落在主存范围 [0, %#x) 里，会和内存控制器抢地址。"
            "请把它放到主存之上，或者调小 --mem-size"
            % (args.accel_addr, mem_end)
        )

    system.accel = Dotp8Accel(
        pio_addr=args.accel_addr,
        pio_size=0x1000,
        pe=args.accel_pe,
        outstanding=args.accel_outstanding,
        start_latency=args.accel_start_lat,
    )

    # 命令通道：CPU --(uncacheable store)--> membus --> 加速器寄存器
    system.accel.cpu_side = system.membus.mem_side_ports

    # 数据通道：本次实验要做的那个决策
    if args.accel == "l2bus":
        if not hasattr(system, "l2bus"):
            raise SystemExit("--accel l2bus 需要有 L2，不能和 --no-l2 / "
                             "--no-caches 一起用")
        # 和 L1 同级挂在 L2 总线上：能命中 L2（98 KB 权重正好放得进 256 KB），
        # 也能被 L1 的脏数据 snoop 到。代价是加速器的流量会冲刷 L2。
        system.accel.mem_side = system.l2bus.cpu_side_ports
    else:
        # 挂在系统总线上：旁路整个 Cache 层次，每一个字节都从 DRAM 来。
        # 真实 SoC 里挂在 NoC 上的加速器就是这个位置。
        system.accel.mem_side = system.membus.cpu_side_ports


def configure_o3(cpu, args):
    """把命令行里的乱序核参数写进 CPU 对象。0 表示保持 gem5 默认值。"""
    if args.width:
        cpu.fetchWidth = args.width
        cpu.decodeWidth = args.width
        cpu.renameWidth = args.width
        cpu.dispatchWidth = args.width
        cpu.issueWidth = args.width
        cpu.commitWidth = args.width
        cpu.squashWidth = args.width
        # 注意：这里故意不动 wbWidth。
        # gem5 的 IEW 把写回结果放进一个深度为 forwardComSize(=5) 的级间时间缓冲，
        # 放不下当前拍就顺延到下一拍（iew.cc:577-588）。wbWidth 一旦调小，
        # wbCycle 会越过缓冲深度，直接触发 TimeBuffer 断言崩溃（退出码 -6）。
        # 而且从设计上讲也不该调：窄核已经在发射级被卡住了，
        # 再把写回总线一起收窄并不会让它更"窄"，只会让模型崩溃。
    if args.wb_width:
        cpu.wbWidth = args.wb_width

    if args.fetch_width:
        cpu.fetchWidth = args.fetch_width
    if args.issue_width:
        cpu.issueWidth = args.issue_width
    if args.commit_width:
        cpu.commitWidth = args.commit_width
    if args.rob:
        cpu.numROBEntries = args.rob
    if args.iq:
        cpu.numIQEntries = args.iq
    if args.lq:
        cpu.LQEntries = args.lq
    if args.sq:
        cpu.SQEntries = args.sq
    if args.int_regs:
        cpu.numPhysIntRegs = args.int_regs

    bp = make_component(BRANCH_PREDICTORS, args.bp)
    if bp is not None:
        if args.btb_entries:
            bp.btb.numEntries = args.btb_entries
        cpu.branchPred = bp

    if args.dotp8_lat or args.dotp8_units:
        cpu.fuPool = make_fu_pool(args.dotp8_lat, args.dotp8_units)


# dotp8 在 decoder.isa 里挂的 OpClass。它决定这条指令去哪个功能单元执行、
# 因而也决定它的延迟——这是"自定义指令要付多少硬件代价"唯一的旋钮。
DOTP8_OPCLASS = "SimdMultAcc"


def make_fu_pool(lat, units):
    """复制一份默认功能单元池，只把 dotp8 所在的那一类改掉。

    本课程的所有基准程序都不含向量指令（编译目标是 rv64gc，没有 V 扩展），
    所以改 SIMD 单元等价于只改 dotp8，不会污染别的指令。
    照抄默认表而不是自己重写，是为了保证除了这一个参数以外一切都没变。
    """
    simd = SIMD_Unit()
    op_names = [str(od.opClass) for od in simd.opList]
    simd.opList = [
        OpDesc(opClass=n, opLat=lat if (n == DOTP8_OPCLASS and lat) else 1)
        for n in op_names
    ]
    if units:
        simd.count = units

    return FUPool(FUList=[IntALU(), IntMultDiv(), FP_ALU(), FP_MultDiv(),
                          ReadPort(), simd, Matrix_Unit(), PredALU(),
                          WritePort(), RdWrPort(), IprPort()])


def attach_caches(system, args):
    cpu = system.cpu

    if args.auto_latency == "on":
        l1i_lat = scaled_latency(args.l1i_size, 32.0, args.l1_latency, 1)
        l1d_lat = scaled_latency(args.l1d_size, 32.0, args.l1_latency, 1)
        l2_lat = scaled_latency(args.l2_size, 256.0, args.l2_latency, 2)
    else:
        l1i_lat = l1d_lat = args.l1_latency
        l2_lat = args.l2_latency

    cpu.icache = L1ICache(
        size=args.l1i_size,
        assoc=args.l1i_assoc,
        tag_latency=l1i_lat,
        data_latency=l1i_lat,
        response_latency=l1i_lat,
        replacement_policy=make_component(REPLACEMENT_POLICIES, "lru"),
    )
    cpu.dcache = L1DCache(
        size=args.l1d_size,
        assoc=args.l1d_assoc,
        mshrs=args.l1d_mshrs,
        tag_latency=l1d_lat,
        data_latency=l1d_lat,
        response_latency=l1d_lat,
        replacement_policy=make_repl(args.l1d_repl, args.l1d_assoc),
    )
    l1d_pf = make_component(PREFETCHERS, args.l1d_pf)
    if l1d_pf is not None:
        cpu.dcache.prefetcher = l1d_pf

    cpu.icache_port = cpu.icache.cpu_side
    cpu.dcache_port = cpu.dcache.cpu_side

    if args.no_l2:
        cpu.icache.mem_side = system.membus.cpu_side_ports
        cpu.dcache.mem_side = system.membus.cpu_side_ports
        cpu.mmu.connectWalkerPorts(
            system.membus.cpu_side_ports, system.membus.cpu_side_ports
        )
        return

    system.l2bus = L2XBar()
    cpu.icache.mem_side = system.l2bus.cpu_side_ports
    cpu.dcache.mem_side = system.l2bus.cpu_side_ports
    cpu.mmu.connectWalkerPorts(
        system.l2bus.cpu_side_ports, system.l2bus.cpu_side_ports
    )

    system.l2cache = L2Cache(
        size=args.l2_size,
        assoc=args.l2_assoc,
        tag_latency=l2_lat,
        data_latency=l2_lat,
        response_latency=l2_lat,
        replacement_policy=make_repl(args.l2_repl, args.l2_assoc),
    )
    l2_pf = make_component(PREFETCHERS, args.l2_pf)
    if l2_pf is not None:
        system.l2cache.prefetcher = l2_pf

    system.l2cache.cpu_side = system.l2bus.mem_side_ports
    system.l2cache.mem_side = system.membus.cpu_side_ports


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    system = build_system(args)

    process = Process()
    process.cmd = [args.cmd] + args.options.split()
    system.workload = SEWorkload.init_compatible(args.cmd)
    system.cpu.workload = process
    system.cpu.createThreads()

    if args.maxinsts:
        system.cpu.max_insts_any_thread = args.maxinsts

    root = Root(full_system=False, system=system)
    m5.instantiate()

    # 运行期输出保持 ASCII，避免不同终端 locale 下出现乱码
    accel = (
        f" accel={args.accel}@{args.accel_pe}pe/"
        f"{args.accel_outstanding}out"
        if args.accel != "off"
        else ""
    )
    print(f"[cssys] start: cpu={args.cpu} freq={args.freq} "
          f"l1d={args.l1d_size}/{args.l1d_assoc}way "
          f"l2={'none' if args.no_l2 else args.l2_size} bp={args.bp}{accel}")
    event = m5.simulate()
    print(f"[cssys] done @ tick {m5.curTick()}, cause: {event.getCause()}")


main()
