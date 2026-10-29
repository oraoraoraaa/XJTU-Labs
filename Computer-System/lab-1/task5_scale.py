"""Run the workload-size experiment, or plot its saved measurements."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR.parent / "framework"))
from stats import parse_blocks, summarize


def run(args):
    rows = []
    for samples in (2, 4, 8, 16, 32):
        name = f"task5_o3_n{samples}_seed{args.seed}"
        out_dir = Path("out") / name
        (LAB_DIR / out_dir).mkdir(parents=True, exist_ok=True)
        command = [
            args.gem5, f"--outdir={out_dir}", "framework/cssys_se.py",
            "--cmd", "bench/bin/riscv/qnn", "--options",
            f"{samples} {int(args.seed, 10)}", "--cpu", "o3",
        ]
        print(f"Running {samples} samples (sequential)...", flush=True)
        started = time.perf_counter()
        with (LAB_DIR / out_dir / "run.log").open("w") as log:
            subprocess.run(command, cwd=LAB_DIR, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        wall_seconds = time.perf_counter() - started
        stats_path = LAB_DIR / out_dir / "stats.txt"
        blocks = parse_blocks(stats_path)
        if len(blocks) != 3:
            raise ValueError(f"{stats_path}: expected three statistics blocks")
        startup, roi, shutdown = [summarize(block) for block in blocks]
        rows.append({
            "samples": samples,
            "seed": args.seed,
            "insts": int(roi["insts"]),
            "cycles": int(roi["cycles"]),
            "ipc": roi["ipc"],
            "cycles_per_sample": roi["cycles"] / samples,
            "roi_host_seconds": roi["host_seconds"],
            "startup_host_seconds": startup["host_seconds"],
            "shutdown_host_seconds": shutdown["host_seconds"],
            "total_host_seconds": sum(block["host_seconds"] for block in
                                      (startup, roi, shutdown)),
            "wall_seconds": round(wall_seconds, 6),
            "l1i_misses": int(roi["l1i_misses"]),
            "bp_mpki": roi["bp_mpki"],
            "l1d_mshr_mpki": roi["l1d_mshr_mpki"],
            "stats_path": str(stats_path),
        })
        print(f"  IPC={roi['ipc']:.6f}; whole-run elapsed={wall_seconds:.3f}s",
              flush=True)
    # A partial/failed sweep must not replace a previous complete CSV.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved measurements: {args.output}")
    print("samples       IPC     ROI host (s)    whole-run elapsed (s)")
    for row in rows:
        print(f"{row['samples']:7d} {row['ipc']:9.6f} "
              f"{row['roi_host_seconds']:16.3f} {row['wall_seconds']:24.3f}")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    samples = [int(row["samples"]) for row in rows]
    seed = rows[0]["seed"]
    plt.rcParams.update({"font.size": 11})

    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    ipc = [float(row["ipc"]) for row in rows]
    ax.plot(samples, ipc, "o-", color="#2463a5")
    ax.set(xlabel="Samples (count)", ylabel="ROI IPC (instructions/cycle)",
           title=f"Task 5: O3 throughput vs workload size (seed {seed})",
           xticks=samples)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=0.25)
    for x, y in zip(samples, ipc):
        ax.annotate(f"{y:.6f}", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.margins(x=0.1, y=0.25)
    ax.text(0.02, 0.02, "Y-axis zoomed to show small IPC differences",
            transform=ax.transAxes, fontsize=9, color="#555555")
    path = args.input.with_name(args.input.stem + "_ipc.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {path}")

    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    for key, label, color in (
        ("wall_seconds", "Whole-run elapsed", "#2463a5"),
        ("roi_host_seconds", "ROI host time", "#ca6824"),
    ):
        ax.plot(samples, [float(row[key]) for row in rows], "o-",
                label=label, color=color)
    ax.set(xlabel="Samples (count)", ylabel="Host time (seconds)",
           title=f"Task 5: sequential simulation cost (seed {seed})",
           xticks=samples, ylim=(0, None))
    ax.grid(alpha=0.25)
    ax.legend()
    path = args.input.with_name(args.input.stem + "_host_time.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--seed", required=True)
    runner.add_argument("--gem5", required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.set_defaults(action=run)
    plotter = commands.add_parser("plot")
    plotter.add_argument("--input", type=Path, required=True)
    plotter.set_defaults(action=plot)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
