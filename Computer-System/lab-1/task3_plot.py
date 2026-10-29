"""Plot Task 3 CPU throughput against host simulation rate from its CSV."""

import argparse
import csv
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    models = ("atomic", "timing", "minor", "o3")
    if len(rows) != 4 or {row["cpu"] for row in rows} != set(models):
        raise ValueError("Expected exactly one CSV row for each of the four CPU models")
    points = {}
    for row in rows:
        if row.get("error"):
            raise ValueError(f"Simulation failed for {row['cpu']}: {row['error']}")
        rate, ipc = float(row["host_inst_rate"]), float(row["ipc"])
        if rate <= 0 or ipc <= 0:
            raise ValueError(f"Expected positive rate and IPC for {row['cpu']}")
        points[row["cpu"]] = (rate / 1e6, ipc)

    colors = {"atomic": "#c27528", "timing": "#6b7280",
              "minor": "#33846a", "o3": "#2463a5"}
    offsets = {"atomic": (-8, 10), "timing": (0, 12),
               "minor": (-5, -20), "o3": (15, -3)}
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    for model in models:
        x, y = points[model]
        ax.scatter(x, y, s=65, color=colors[model], zorder=3)
        ax.annotate(model, (x, y), xytext=offsets[model],
                    textcoords="offset points", ha="center", fontsize=11)
    ax.set(
        xlabel="Simulation rate (million target instructions / host second)",
        ylabel="ROI IPC (instructions / cycle)",
        title=f"CPU models: qnn, 8 samples, seed {args.input.stem.rsplit('_seed', 1)[-1]}",
        xlim=(0, max(x for x, _ in points.values()) * 1.15),
        ylim=(0, max(y for _, y in points.values()) * 1.15),
    )
    ax.grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() != args.report_output.resolve():
        shutil.copy2(args.output, args.report_output)
    print(f"Saved scatter plot: {args.output}")
    print(f"Report image: {args.report_output}")


if __name__ == "__main__":
    main()
