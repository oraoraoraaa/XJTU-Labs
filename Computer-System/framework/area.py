#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""area.py —— 给一次 gem5 运行称重：面积、静态功耗、动态能量。

前六次实验只回答了"快了多少"。第七次要回答"值不值"，
所以需要一把秤。这就是那把秤。

    python3 framework/area.py out/lab6/verify
    python3 framework/area.py out/sweeps/lab7_budget/*/ --csv
    python3 framework/area.py --model          # 打印系数表和每一条的依据

╔══════════════════════════════════════════════════════════════════════╗
║  这是一个**模型**，不是测量值。                                       ║
║                                                                      ║
║  它读的是 gem5 输出目录里的 config.ini（实际跑的配置）和 stats.txt    ║
║  （实际的活动量），按下面那张系数表算出面积和能量。                   ║
║  系数全部写在本文件顶部，**每一条都可以被质疑**——                     ║
║  实验七的思考题就是要你质疑其中一条，并说明它错了会怎样。             ║
║                                                                      ║
║  它不是 CACTI，也不是 McPAT。它的目标只有一个：                       ║
║  **让"性能提升"和"付出的代价"能放在同一张表里比较。**                 ║
║  绝对值不可信，**同一模型下的相对比较是可信的**。                     ║
╚══════════════════════════════════════════════════════════════════════╝

单位
----
面积  1 A  = 1 KB SRAM 数据阵列的面积。
             （量级参考：7nm 下约 400 µm²，含外围电路）
能量  1 E  = 访问一次 1 KB SRAM 的能量。
             （量级参考：约 1 pJ）
这样定义的好处：**两个单位都锚在"一小块 SRAM"上**，
学生不需要相信任何绝对数字，只需要相信"一个 int8 乘法器大约相当于
1 KB SRAM"这类相对关系——而这类关系比绝对值稳健得多。
"""
import configparser
import glob
import io
import os
import re
import sys

# =====================================================================
# 系数表 —— 全部依据写在这里，欢迎质疑
# =====================================================================

MODEL = {
    # ---- SRAM 阵列 ---------------------------------------------------
    "sram_per_kb": (1.0, "定义：1 A = 1 KB 数据阵列"),
    "tag_frac_64B": (0.12, "64B 行的 tag+状态位约占数据阵列的 12%："
                           "48 位物理地址去掉块内偏移和组索引，再加 valid/dirty/"
                           "一致性状态，一行约 40 位 tag 对 512 位数据"),
    "assoc_per_way": (0.02, "每多一路要多一个比较器和一路数据 mux，"
                            "约为数据阵列的 2%/路（8 路 → +16%）"),
    "mshr": (1.5, "一个 MSHR = 地址 CAM 项 + 目标队列(tgts_per_mshr 项)，"
                  "比一个 Cache 行贵，取 1.5 A"),
    # 替换策略：每行需要的状态位
    "repl_bits": {
        "lru":    ("log2(assoc)", "真 LRU 要给每行存一个序号"),
        "random": (0,             "不需要任何状态"),
        "rrip":   (2,             "RRPV 是 2 位"),
        "brrip":  (2,             "同 RRIP"),
        "drrip":  (2,             "同 RRIP，另加一个 32 项的 PSEL 计数器(忽略)"),
        "ship":   (2,             "RRPV 2 位；另有签名表，见 ship_table"),
        "fifo":   (0,             "组内一个插入指针即可，按 0 计"),
        "lip":    ("log2(assoc)", "同 LRU"),
        "mru":    ("log2(assoc)", "同 LRU"),
        "secondchance": (1,       "每行一个引用位"),
        "treeplru": (1,           "树形伪 LRU 每行约 1 位"),
        "weightedlru": ("log2(assoc)", "同 LRU"),
        "dueling": (2,            "按 RRIP 计"),
    },
    "ship_table": (8.0, "SHiP 的签名历史表 (SHCT)，约 16k 项 × 3 位 ≈ 6 KB → 8 A"),
    # 预取器
    "prefetcher": {
        "none":      (0.0,  "没有就是没有"),
        "nextline":  (0.5,  "只要一个加法器和一个队列"),
        "stride":    (4.0,  "PC 索引的步长表，约 64 项 × (PC+lastAddr+stride+conf)"),
        "tagged":    (1.0,  "next-line 加一个 tag 位"),
        "ampm":      (16.0, "AMPM 要一张访问位图，是这几个里最贵的"),
        "signature": (12.0, "SPP 的签名表 + 模式表"),
        "indirect":  (10.0, "间接预取器要存基址和步幅候选"),
    },

    # ---- 乱序核 ------------------------------------------------------
    "rob_per_entry": (0.09, "一个 ROB 项存 PC、目的寄存器、异常状态等约 8 字节，"
                            "外加 commitWidth 个读端口的开销"),
    "iq_coef": (0.035, "发射队列是 CAM：每周期要拿 issueWidth 个结果标签去"
                       "唤醒所有项。面积按 n^1.5 增长(唤醒网络是超线性的)"),
    "iq_exp": (1.5, "同上"),
    "lsq_coef": (0.05, "LQ/SQ 要做地址比较(访存序违例检查)，也是 CAM"),
    "lsq_exp": (1.5, "同上"),
    "prf_per_reg": (0.05, "一个 64 位物理寄存器 + 重命名表项"),
    "prf_port_coef": (0.10, "寄存器堆面积随端口数超线性增长，"
                            "端口数正比于发射宽度"),
    "width_coef": (3.0, "取指/译码/重命名/旁路网络。**旁路网络是 n² 结构**，"
                        "整体按 width^1.8 建模——这是宽发射最大的一笔账"),
    "width_exp": (1.8, "同上"),
    "btb_per_entry": (0.010, "一个 BTB 项 = 部分 tag + 目标地址 ≈ 8 字节"),
    "bp_per_kbit": (0.125, "预测器的计数器表，按位数折算成 SRAM"),
    "ras_per_entry": (0.02, "返回地址栈"),
    "core_fixed": (120.0, "译码器、异常/中断逻辑、TLB、时钟树、"
                          "以及所有没被单独建模的东西。"
                          "**这是本模型最粗的一项**，见 --model 的告诫"),

    # ---- 实验五：dotp8 功能单元 ---------------------------------------
    "dotp8_unit": (6.0, "一个 4 路 int8 点积单元：4 个 8×8 乘法器 + 加法树 + "
                        "累加器。约等于 6 KB SRAM 的面积"),
    "dotp8_lat_penalty": (1.0, "同样一棵乘加树，要求 1 拍出结果就得加大驱动、"
                               "把加法树整个展开；要求 3 拍就能流水切开、用小门。"
                               "面积按 (1 + 1/opLat) 缩放：1 拍→2 倍，3 拍→1.33 倍。"
                               "**实验五那次 --dotp8-lat 扫描的意义在实验七这里才兑现**"),

    # ---- 实验六：加速器 -----------------------------------------------
    "accel_mac": (1.0, "一个 int8 乘累加单元 ≈ 1 KB SRAM 的面积。"
                       "**这是全表最值得质疑的一条**，见思考题"),
    "accel_fixed": (20.0, "命令寄存器、地址生成、行缓冲、状态机"),

    # ---- 能量 --------------------------------------------------------
    "e_sram_ref_kb": (1.0, "定义：访问 1 KB SRAM = 1 E"),
    "e_sram_exp": (0.5, "访问能量主要是位线/字线的充放电，"
                        "随阵列边长增长 → 约按容量的 0.5 次方"),
    "e_mac": (0.05, "一次 int8 乘加约为访问 1 KB SRAM 的 1/20"),
    "e_core_per_cycle": (3.0, "核心每个活动周期的动态能量，正比于发射宽度。"
                              "定成这个值，是为了让基线(8 发射)的核心动态能量约为"
                              "L1D 访问能量的 3 倍——真实芯片里 L1 Cache 大约占核心"
                              "动态功耗的 20~25%，这个比例要对，否则能量指标会退化成"
                              "'指令数指标'，选出来的冠军和纯性能一模一样"),
    "e_dram_per_line": (25.0, "一次 DRAM 行访问(64B)的能量远高于片上，取 25 E。"
                              "**片外访存比片内贵一个量级**是本模型想传达的关键量级"),
    "leak_per_area_cycle": (8.0e-3, "泄漏功耗正比于面积、也正比于时间——"
                                    "**跑得慢的设计要多漏这么久**。系数定得让基线的"
                                    "泄漏约占总能量的 15~20%。这一条会让'又大又慢'的"
                                    "配置在能量上被双重惩罚，这是对的"),
}

CACHES = [
    # (config.ini 里的节名, 显示名)
    ("system.cpu.icache", "L1I"),
    ("system.cpu.dcache", "L1D"),
    ("system.l2cache",    "L2"),
]


# =====================================================================
# 工具
# =====================================================================

def _f(cfg, section, key, default=0.0):
    try:
        return float(cfg[section][key])
    except Exception:
        return float(default)


def _s(cfg, section, key, default=""):
    try:
        return cfg[section][key]
    except Exception:
        return default


def _log2(x):
    import math
    return math.log2(x) if x > 1 else 0.0


def coef(name):
    return MODEL[name][0]


def read_config(outdir):
    path = os.path.join(outdir, "config.ini")
    if not os.path.isfile(path):
        raise SystemExit("找不到 %s —— 参数应当是 gem5 的 --outdir 目录" % path)
    cfg = configparser.ConfigParser(strict=False)
    cfg.read(path)
    return cfg


def read_roi_stats(outdir):
    """读 stats.txt 的 ROI 统计块（第 2 个 Begin Simulation Statistics）。

    和 framework/stats.py 的口径保持一致：ROI 是第二块。
    找不到第二块就退回用第一块，并在返回值里标记。
    """
    path = os.path.join(outdir, "stats.txt")
    if not os.path.isfile(path):
        return {}, False
    blocks, cur = [], None
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Begin Simulation Statistics" in line:
                cur = {}
                blocks.append(cur)
                continue
            if cur is None or not line.strip() or line.startswith("-"):
                continue
            m = re.match(r"^(\S+)\s+([-\d.naif]+)", line)
            if m:
                try:
                    cur[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    if not blocks:
        return {}, False
    if len(blocks) >= 2:
        return blocks[1], True
    return blocks[0], False


# =====================================================================
# 面积
# =====================================================================

def cache_area(cfg, section, line_bytes):
    """一级 Cache 的面积：数据 + tag + 相联度 + 替换状态 + MSHR + 预取器。"""
    if not cfg.has_section(section):
        return None
    kb = _f(cfg, section, "size") / 1024.0
    assoc = max(1.0, _f(cfg, section, "assoc", 1))
    lines = (kb * 1024.0) / line_bytes if line_bytes else 0.0

    data = kb * coef("sram_per_kb")
    tag = data * coef("tag_frac_64B") * (64.0 / line_bytes if line_bytes else 1.0)
    ways = data * coef("assoc_per_way") * assoc

    # 替换策略的每行状态位
    rp = _s(cfg, section, "replacement_policy")
    rp_type = _s(cfg, rp, "type", "LRURP") if rp else "LRURP"
    key = rp_type.replace("RP", "").lower()
    spec = MODEL["repl_bits"].get(key, ("log2(assoc)", "未知策略，按 LRU 计"))[0]
    bits = _log2(assoc) if spec == "log2(assoc)" else float(spec)
    repl = lines * bits / 8.0 / 1024.0          # 位 → KB → A
    if key == "ship":
        repl += coef("ship_table")

    mshr = _f(cfg, section, "mshrs") * coef("mshr")

    pf = _s(cfg, section, "prefetcher")
    pf_area = 0.0
    pf_name = "none"
    if pf and pf.lower() != "null":
        pf_type = _s(cfg, pf, "type", "")
        pf_name = pf_type
        for k, (v, _) in MODEL["prefetcher"].items():
            if k in pf_type.lower():
                pf_area = v
                break
        else:
            pf_area = 4.0                        # 认不出来的按 stride 计

    return {
        "kB": kb, "assoc": assoc, "repl": rp_type, "pf": pf_name,
        "数据": data, "tag": tag, "相联": ways,
        "替换状态": repl, "MSHR": mshr, "预取器": pf_area,
        "小计": data + tag + ways + repl + mshr + pf_area,
    }


def core_area(cfg):
    """乱序核的面积。非 o3 的核按一个固定的小值计。"""
    sec = "system.cpu"
    cpu_type = _s(cfg, sec, "type", "")
    if "O3" not in cpu_type:
        # 顺序核：没有 ROB、没有发射队列、没有寄存器重命名、没有旁路网络。
        # 本模型是照着乱序核标定的，这里只给一个**量级正确**的粗估：
        #   Atomic / Timing —— 连流水线都没有建模，按最小核算
        #   Minor          —— 4 级顺序流水，有分支预测器，贵一些
        # ⚠️ 实验七只用 o3，这条路径存在只是为了不让实验一的目录报错。
        base = coef("core_fixed") * 0.5
        out = {"_type": cpu_type, "_width": 1.0, "_inorder": True,
               "固定": base}
        if "Minor" in cpu_type:
            out["顺序流水线"] = coef("core_fixed") * 0.25
            bp = _s(cfg, sec, "branchpred")
            if bp and bp.lower() != "null":
                btb = _s(cfg, bp, "btb")
                if btb and btb.lower() != "null":
                    out["BTB"] = _f(cfg, btb, "numentries") * coef("btb_per_entry")
        out["小计"] = sum(v for k, v in out.items() if not k.startswith("_"))
        return out

    rob = _f(cfg, sec, "numrobentries")
    iq = _f(cfg, sec, "numiqentries")
    lq = _f(cfg, sec, "lqentries")
    sq = _f(cfg, sec, "sqentries")
    regs = _f(cfg, sec, "numphysintregs")
    iw = max(1.0, _f(cfg, sec, "issuewidth", 1))
    fw = max(1.0, _f(cfg, sec, "fetchwidth", 1))
    cw = max(1.0, _f(cfg, sec, "commitwidth", 1))
    width = (iw + fw + cw) / 3.0                # 三个宽度的均值代表"核有多宽"

    a_rob = rob * coef("rob_per_entry") * (1 + 0.1 * cw)
    a_iq = coef("iq_coef") * (iq ** coef("iq_exp")) * (1 + 0.15 * iw) / 2.2
    a_lq = coef("lsq_coef") * (lq ** coef("lsq_exp"))
    a_sq = coef("lsq_coef") * (sq ** coef("lsq_exp"))
    a_prf = regs * coef("prf_per_reg") * (1 + coef("prf_port_coef") * iw)
    a_width = coef("width_coef") * (width ** coef("width_exp"))

    # 分支预测器：把每张表折算成"多少位"，再按 bp_per_kbit 换成面积。
    #  (表名, 每项位数的来源)。每项位数取不到时的默认值写在第三项。
    #  注意 localHistoryTable 存的是"历史"不是"计数器"，每项 log2(表项数) 位，
    #  gem5 里这个值不在 config.ini 里(是 Python 侧算出来的)，这里现算。
    bp = _s(cfg, sec, "branchpred")
    a_bp, a_btb, a_ras = 0.0, 0.0, 0.0
    bp_type = ""
    bp_bits = 0.0
    if bp and bp.lower() != "null":
        bp_type = _s(cfg, bp, "type", "")
        for size_key, bit_key, dflt in (
                ("globalpredictorsize", "globalctrbits", 2.0),
                ("choicepredictorsize", "choicectrbits", 2.0),
                ("localpredictorsize", "localctrbits", 2.0),
                ("predictorsize", "ctrbits", 2.0),
                ("localhistorytablesize", None, None),   # 每项 log2(localPredictorSize) 位
        ):
            n = _f(cfg, bp, size_key)
            if not n:
                continue
            if bit_key is None:
                per = _log2(_f(cfg, bp, "localpredictorsize", 1)) or 8.0
            else:
                per = _f(cfg, bp, bit_key) or dflt
            bp_bits += n * per
        # TAGE/LTAGE 之类多表结构：把所有 tagtablesize 之类也算上（有就算，没有就算了）
        for k in ("tagtablesize", "loopTableEntries"):
            n = _f(cfg, bp, k.lower())
            if n:
                bp_bits += n * 12.0
        a_bp = bp_bits / 1024.0 * coef("bp_per_kbit")
        btb = _s(cfg, bp, "btb")
        if btb and btb.lower() != "null":
            a_btb = _f(cfg, btb, "numentries") * coef("btb_per_entry")
        ind = _s(cfg, bp, "indirectbranchpred")
        if ind and ind.lower() != "null":
            ent = _f(cfg, ind, "indirectsets") * _f(cfg, ind, "indirectways")
            a_btb += ent * coef("btb_per_entry")     # 间接预测器并进 BTB 一栏
        ras = _s(cfg, bp, "ras")
        if ras and ras.lower() != "null":
            a_ras = _f(cfg, ras, "numentries") * coef("ras_per_entry")

    fixed = coef("core_fixed")
    out = {
        "_type": cpu_type, "_width": width,
        "ROB": a_rob, "发射队列": a_iq, "LQ": a_lq, "SQ": a_sq,
        "物理寄存器": a_prf, "发射/旁路网络": a_width,
        "分支预测器": a_bp, "BTB": a_btb, "RAS": a_ras, "固定开销": fixed,
    }
    out["小计"] = sum(v for k, v in out.items() if not k.startswith("_"))
    out["_bp"] = bp_type
    out["_bp_kbit"] = bp_bits / 1024.0
    return out


def accel_area(cfg):
    if not cfg.has_section("system.accel"):
        return None
    pe = _f(cfg, "system.accel", "pe")
    outs = _f(cfg, "system.accel", "outstanding")
    mac = pe * coef("accel_mac")
    mshr = outs * coef("mshr")
    fixed = coef("accel_fixed")
    return {"pe": pe, "outstanding": outs,
            "MAC 阵列": mac, "访存队列": mshr, "控制/缓冲": fixed,
            "小计": mac + mshr + fixed}


DOTP8_OPCLASS = "SimdMultAcc"      # 和 framework/cssys_se.py 里的定义保持一致


def dotp8_area(cfg, st):
    """实验五的 dotp8 功能单元。

    dotp8 在 decoder.isa 里挂的 OpClass 是 SimdMultAcc，它复用 gem5 默认
    FUPool 里的 SIMD 单元 —— 所以 config.ini 里**根本没有 "dotp8" 这个词**。
    只能这样找：

      1) 在 system.cpu.fuPool.FUList*.opList* 里找 opClass=SimdMultAcc，
         回到它的父 FUList，读 count（单元数）和 opLat（延迟）。
      2) 再看 ROI 里 committedInstType::SimdMultAcc 是不是真的大于 0。
         **没执行过就不收面积** —— 基线核里那个 SIMD 单元本来就存在，
         它的面积已经含在 core_fixed 里了；这里收的是"为 dotp8 额外加的
         那条 4 路 int8 乘加数据通路"。

    返回 None 表示这次运行没用到 dotp8。
    """
    used = 0.0
    for k in ("system.cpu.commitStats0.committedInstType::" + DOTP8_OPCLASS,
              "system.cpu.commit.committedInstType_0::" + DOTP8_OPCLASS):
        used = max(used, st.get(k, 0.0))
    if used <= 0:
        return None

    count, lat = 0, 1.0
    for s in cfg.sections():
        if not re.match(r"^system\.cpu\.fuPool\.FUList\d+\.opList", s):
            continue
        if _s(cfg, s, "opclass", "") != DOTP8_OPCLASS:
            continue
        parent = s.rsplit(".opList", 1)[0]
        count = int(_f(cfg, parent, "count", 1)) or 1
        lat = max(1.0, _f(cfg, s, "oplat", 1))
        break
    if not count:
        return None

    # 延迟压得越低，同样的乘加树越要靠加大门/展开来赶上一拍，面积越贵。
    lat_factor = 1.0 + coef("dotp8_lat_penalty") / lat
    return {"单元数": count, "opLat": lat, "延迟系数": round(lat_factor, 3),
            "dotp8 指令数": used,
            "小计": count * coef("dotp8_unit") * lat_factor}


# =====================================================================
# 能量
# =====================================================================

def sram_access_energy(kb):
    """访问一次 kB 大小的 SRAM 的能量（单位 E）。"""
    return (max(kb, 0.25) / coef("e_sram_ref_kb")) ** coef("e_sram_exp")


def energy(cfg, st, areas, line_bytes):
    """动态能量 + 泄漏。返回 dict（单位 E）。

    动态部分全部由 stats.txt 里的**实际活动次数**驱动，
    所以它是"这次运行"的能量，不是"这个设计"的能量。
    """
    if not st:
        return None
    cycles = st.get("system.cpu.numCycles", 0.0)
    out = {}

    # Cache 访问
    for sec, name in CACHES:
        if not cfg.has_section(sec):
            continue
        kb = _f(cfg, sec, "size") / 1024.0
        acc = st.get("%s.overallAccesses::total" % sec.replace("system.", "system."), 0.0)
        if not acc:
            acc = st.get("%s.demandAccesses::total" % sec, 0.0)
        out["%s 访问" % name] = acc * sram_access_energy(kb)

    # DRAM
    dram_lines = st.get("system.mem_ctrl.dram.bytesRead::total", 0.0) / line_bytes
    dram_lines += st.get("system.mem_ctrl.dram.bytesWritten::total", 0.0) / line_bytes
    out["DRAM"] = dram_lines * coef("e_dram_per_line")

    # 核心
    width = areas["核心"].get("_width", 4.0)
    out["核心"] = cycles * coef("e_core_per_cycle") * width

    # 加速器
    macs = st.get("system.accel.macs", 0.0)
    if macs:
        out["加速器 MAC"] = macs * coef("e_mac")

    # 泄漏：正比于总面积 × 周期数
    total_area = sum(a["小计"] for a in areas.values() if a)
    out["泄漏"] = total_area * cycles * coef("leak_per_area_cycle")

    out["合计"] = sum(out.values())
    return out


# =====================================================================
# 汇总一次运行
# =====================================================================

def weigh(outdir):
    cfg = read_config(outdir)
    st, roi_ok = read_roi_stats(outdir)
    line_bytes = _f(cfg, "system", "cache_line_size", 64) or 64.0

    areas = {}
    for sec, name in CACHES:
        a = cache_area(cfg, sec, line_bytes)
        if a:
            areas[name] = a
    areas["核心"] = core_area(cfg)
    d8 = dotp8_area(cfg, st)
    if d8:
        areas["dotp8 单元"] = d8
    ac = accel_area(cfg)
    if ac:
        areas["加速器"] = ac

    total = sum(a["小计"] for a in areas.values())
    e = energy(cfg, st, areas, line_bytes)
    cycles = st.get("system.cpu.numCycles", 0.0)
    insts = (st.get("system.cpu.commitStats0.numInsts", 0.0)
             or st.get("system.cpu.commit.committedInsts", 0.0))

    return {
        "outdir": outdir, "areas": areas, "总面积": total,
        "energy": e, "cycles": cycles, "insts": insts, "roi_ok": roi_ok,
        "accel": _s(cfg, "system.accel", "type", "") and
                 ("l2bus" if "l2bus" in _s(cfg, "system.accel", "mem_side") else "membus"),
    }


# =====================================================================
# 输出
# =====================================================================

def print_one(r, detail=False):
    print("=" * 66)
    print(r["outdir"])
    if not r["roi_ok"]:
        print("  ⚠️ stats.txt 里没有第二个统计块，用的是全程数据（不是 ROI）")
    if r["areas"].get("核心", {}).get("_inorder"):
        print("  ⚠️ 这是顺序核。本模型是照乱序核标定的，核心面积只是量级估计，"
              "不要拿它和 o3 做精细比较。")
    print("=" * 66)
    print("%-16s %10s   %s" % ("部件", "面积(A)", "占比"))
    print("-" * 66)
    for name, a in r["areas"].items():
        pct = 100.0 * a["小计"] / r["总面积"] if r["总面积"] else 0
        bar = "█" * int(pct / 2)
        print("%-16s %10.1f   %5.1f%% %s" % (name, a["小计"], pct, bar))
        if detail:
            for k, v in a.items():
                if k.startswith("_") or k == "小计" or not isinstance(v, float):
                    continue
                if v:
                    print("      %-14s %8.1f" % (k, v))
    print("-" * 66)
    print("%-16s %10.1f" % ("总计", r["总面积"]))

    if r["cycles"]:
        print()
        print("%-16s %10.0f 周期" % ("ROI 时间", r["cycles"]))
        if r["insts"]:
            print("%-16s %10.0f 条  (IPC %.3f)"
                  % ("ROI 指令", r["insts"], r["insts"] / r["cycles"]))
    e = r["energy"]
    if e:
        print()
        print("%-16s %10s   %s" % ("能量来源", "能量(E)", "占比"))
        print("-" * 66)
        for k, v in e.items():
            if k == "合计" or not v:
                continue
            pct = 100.0 * v / e["合计"] if e["合计"] else 0
            print("%-16s %10.0f   %5.1f%% %s" % (k, v, pct, "█" * int(pct / 2)))
        print("-" * 66)
        print("%-16s %10.0f" % ("总能量", e["合计"]))

    if r["cycles"] and r["总面积"]:
        m = metrics(r)
        print()
        print("综合指标")
        print("-" * 66)
        print("  三个基本量（都是越小越好）")
        print("    T  时间          %14.0f  周期" % m["T"])
        print("    A  面积          %14.1f  A" % m["A"])
        if m["E"]:
            print("    E  能量          %14.0f  E" % m["E"])
        print("  组合指标（越小越好）")
        print("    ADP = A·T        %14.2f   (A×1e6 周期)" % m["ADP"])
        if m["E"]:
            print("    EDP = E·T        %14.2f   (1e3 E×1e6 周期)" % m["EDP"])
            print("    ED²P = E·T²      %14.2f   (1e3 E×1e12 周期²)" % m["ED2P"])
        print("  注：常说的\"性能/面积\"就是 1/ADP，\"性能/能量\"就是 1/EDP——")
        print("      它们和上面两个是同一个指标的倒数，不是新指标。")


def metrics(r):
    """把一次运行压成一行数字。

    只有 T / A / E 三个是**独立**的，其余全是它们的乘积。
    ADP、EDP、ED²P 的区别只是"多看重速度一点还是少看重一点"：
      ADP  面积和时间一比一        —— 你在乎单位硅面积的吞吐
      EDP  能量和时间一比一        —— 你在乎电池
      ED²P 时间的权重是能量的两倍  —— 你在乎响应时间，愿意为它多花电
    """
    T = r["cycles"] or 1.0
    A = r["总面积"] or 1.0
    E = (r["energy"] or {}).get("合计", 0.0)
    return {
        "T": T, "A": A, "E": E,
        "ADP": A * T / 1e6,
        "EDP": (E / 1e3) * (T / 1e6),
        "ED2P": (E / 1e3) * (T / 1e6) ** 2,
    }


def print_csv(rows):
    cols = ["outdir", "cycles", "insts", "area", "energy",
            "adp", "edp", "ed2p",
            "area_core", "area_l1i", "area_l1d", "area_l2",
            "area_accel", "area_dotp8"]
    print(",".join(cols))
    for r in rows:
        m = metrics(r)
        g = lambda n: r["areas"].get(n, {}).get("小计", 0.0)
        vals = [
            os.path.basename(os.path.normpath(r["outdir"])),
            "%.0f" % r["cycles"], "%.0f" % r["insts"],
            "%.1f" % r["总面积"], "%.0f" % m["E"],
            "%.4f" % m["ADP"],
            "%.4f" % m["EDP"] if m["E"] else "",
            "%.4f" % m["ED2P"] if m["E"] else "",
            "%.1f" % g("核心"), "%.1f" % g("L1I"), "%.1f" % g("L1D"),
            "%.1f" % g("L2"), "%.1f" % g("加速器"), "%.1f" % g("dotp8 单元"),
        ]
        print(",".join(vals))


def print_model():
    print("面积/能量模型的全部系数与依据")
    print("=" * 70)
    print("单位： 1 A = 1 KB SRAM 数据阵列的面积（≈400 µm² @7nm）")
    print("       1 E = 访问一次 1 KB SRAM 的能量（≈1 pJ）")
    print()
    for k, v in MODEL.items():
        if isinstance(v, dict):
            print("%-22s（分项）" % k)
            for kk, (vv, why) in v.items():
                print("    %-14s %-10s %s" % (kk, vv, why))
        else:
            val, why = v
            print("%-22s %-10s %s" % (k, val, why))
    print()
    print("=" * 70)
    print("""⚠️ 这个模型最粗的三个地方（实验七思考题要你挑一个来质疑）：

  1) core_fixed = 120 A 是一个**拍脑袋的常数**。它把译码器、TLB、时钟树
     等等全都塞进一个数里。它占了基线核心面积的三分之一，
     **却完全不随任何参数变化**——这意味着本模型系统性地低估了
     "把核做窄"能省下的面积。

  2) accel_mac = 1.0 A（一个 int8 MAC ≈ 1 KB SRAM）。这个比值在不同工艺、
     不同实现（阵列 vs 独立单元）下可以差 3 倍以上。
     **加速器的性价比结论对这一条最敏感。**

  3) 能量模型里没有时钟树、没有片上互连、没有电压/频率的关系。
     所有"每周期"的项都假设活动因子恒定。

  **但是**：这三条都是**单调的**——它们会让某一类设计整体偏贵或偏便宜，
  不会改变同一类设计内部的排序。所以"L2 从 256 kB 砍到 128 kB 省多少"
  这类同类比较是可信的；"加速器 vs 加宽核心哪个划算"这类跨类比较
  **必须在报告里注明它依赖于系数 2**。""")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if "--model" in flags:
        print_model()
        return

    if not args:
        print(__doc__)
        print("用法: python3 framework/area.py OUTDIR [OUTDIR ...] "
              "[--detail] [--csv]")
        print("      python3 framework/area.py --model")
        return

    dirs = []
    for a in args:
        if any(ch in a for ch in "*?["):
            dirs.extend(sorted(glob.glob(a)))
        else:
            dirs.append(a)
    dirs = [d for d in dirs if os.path.isdir(d)]

    rows = [weigh(d) for d in dirs]
    if "--csv" in flags:
        print_csv(rows)
    else:
        for r in rows:
            print_one(r, detail="--detail" in flags)
            print()


if __name__ == "__main__":
    main()
