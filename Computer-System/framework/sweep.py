"""
sweep.py —— 设计空间并行扫描器

体系结构实验的核心动作是"改一个参数，看性能怎么变"。手工一条条敲命令行
既慢又容易出错，而且 gem5 是单线程的，不并行就是浪费机器。这个脚本把
"参数取值的笛卡尔积"变成一批并行的 gem5 进程，最后汇总成一张 CSV 表。

例子——扫 L1 数据 Cache 的容量：

  python3 framework/sweep.py --name lab2_l1d \\
      --bench qnn --bench-args 8 \\
      --base "--cpu o3" \\
      --vary l1d-size=8kB,16kB,32kB,64kB,128kB \\
      --jobs 16

例子——二维扫描（容量 x 相联度），会跑 5x4=20 个点：

  python3 framework/sweep.py --name lab2_l1d_2d \\
      --bench qnn --bench-args 8 --base "--cpu o3" \\
      --vary l1d-size=8kB,16kB,32kB,64kB,128kB \\
      --vary l1d-assoc=1,2,4,8 --jobs 20

例子——把"换二进制"也当成一个维度（bench 是唯一一个不传给 gem5 的键）：

  python3 framework/sweep.py --name lab5_cmp \\
      --bench qnn --bench-args "8 0x01352505" --base "--cpu o3" \\
      --vary bench=qnn,qnn_dotp8x4c --jobs 2

结果写到 results/<name>.csv，每行一个配置点。
"""

import argparse
import csv
import itertools
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as statsmod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEM5 = os.environ.get("GEM5_BIN", os.path.join(ROOT, "gem5", "build", "RISCV", "gem5.opt"))
CONFIG = os.path.join(ROOT, "framework", "cssys_se.py")
BENCH_DIR = os.path.join(ROOT, "bench", "bin", "riscv")


def parse_args():
    p = argparse.ArgumentParser(description="gem5 设计空间并行扫描")
    p.add_argument("--name", required=True, help="本次扫描的名字，决定输出文件名")
    p.add_argument("--bench", required=True,
                   help="基准程序名，如 qnn / gemm / pchase。"
                        "也可以写成 --vary bench=a,b 来把不同二进制放进扫描")
    p.add_argument("--bench-args", default="", help="传给基准程序的参数")
    p.add_argument("--base", default="", help="所有配置点共用的 gem5 参数")
    p.add_argument(
        "--vary",
        action="append",
        default=[],
        metavar="KEY=V1,V2,...",
        help="要扫描的参数，可重复给出，多个 --vary 之间取笛卡尔积",
    )
    p.add_argument("--jobs", type=int, default=os.cpu_count() // 2 or 1)
    p.add_argument("--block", type=int, default=statsmod.ROI_BLOCK,
                   help="取第几个统计块：0=启动段 1=ROI(默认) 2=收尾段")
    p.add_argument("--timeout", type=int, default=3600, help="单点超时秒数")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def expand(vary):
    """把 ['a=1,2', 'b=x,y'] 展开成 [{'a':'1','b':'x'}, {'a':'1','b':'y'}, ...]"""
    if not vary:
        return [{}]
    keys, value_lists = [], []
    for spec in vary:
        if "=" not in spec:
            raise SystemExit(f"--vary 格式应为 KEY=V1,V2,...，收到: {spec}")
        key, values = spec.split("=", 1)
        keys.append(key.strip())
        value_lists.append([v.strip() for v in values.split(",") if v.strip()])
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def point_label(point, index):
    if not point:
        return f"p{index:03d}"
    parts = [f"{k}-{v}".replace("/", "_") for k, v in point.items()]
    return f"p{index:03d}_" + "_".join(parts)


def build_command(args, point, outdir):
    # bench 是个特殊的"参数"：它不传给 gem5，而是换掉被测的可执行文件。
    # 这样 "换二进制" 和 "换硬件参数" 可以放进同一个笛卡尔积里，例如
    #   --vary bench=qnn,qnn_dotp8x4c --vary cpu=timing,o3
    # 会跑出 2x2 四个点，软硬件两个维度一次扫完。
    binary = os.path.join(BENCH_DIR, point.get("bench", args.bench))
    cmd = [GEM5, f"--outdir={outdir}", CONFIG, "--cmd", binary]
    if args.bench_args:
        cmd += ["--options", args.bench_args]
    cmd += shlex.split(args.base)
    for key, value in point.items():
        if key == "bench":
            continue
        cmd += [f"--{key}", value]
    return cmd


def run_point(args, point, index, total, state):
    label = point_label(point, index)
    outdir = os.path.join(ROOT, "out", "sweeps", args.name, label)
    os.makedirs(outdir, exist_ok=True)
    cmd = build_command(args, point, outdir)

    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cmd))
        return None

    started = time.time()
    row = dict(point)
    row["label"] = label
    try:
        with open(os.path.join(outdir, "run.log"), "w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
            )
        if proc.returncode != 0:
            row["error"] = f"gem5 退出码 {proc.returncode}"
        else:
            row.update(statsmod.summarize_file(
                os.path.join(outdir, "stats.txt"), args.block))
    except subprocess.TimeoutExpired:
        row["error"] = f"超时 (>{args.timeout}s)"
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)

    row["wall_seconds"] = round(time.time() - started, 1)

    state["done"] += 1
    flag = "FAIL" if "error" in row else "ok"
    print(f"[{state['done']:3d}/{total}] {label:<44s} {flag:>4s} "
          f"{row['wall_seconds']:6.1f}s  ipc={row.get('ipc', 0):.3f}", flush=True)
    return row


def main():
    args = parse_args()
    points = expand(args.vary)
    total = len(points)

    if not os.path.exists(GEM5):
        raise SystemExit(f"找不到 gem5 可执行文件: {GEM5}\n可用 GEM5_BIN 环境变量指定")

    print(f"扫描 '{args.name}': {total} 个配置点, 并行度 {args.jobs}")
    state = {"done": 0}
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(run_point, args, pt, i, total, state)
            for i, pt in enumerate(points)
        ]
        rows = [f.result() for f in futures]

    if args.dry_run:
        return

    rows = [r for r in rows if r]
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"{args.name}.csv")

    columns = []
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    failed = sum(1 for r in rows if "error" in r)
    print(f"\n完成: {total - failed}/{total} 成功, 总耗时 {time.time() - started:.1f}s")
    print(f"结果: {out_csv}")


if __name__ == "__main__":
    main()
