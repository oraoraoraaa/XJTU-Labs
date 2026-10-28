"""
report.py —— 把扫描结果 CSV 渲染成对齐的表格

  python3 framework/report.py results/lab2_l1d.csv
  python3 framework/report.py results/lab2_l1d.csv ipc l1d_miss_rate l1d_mpki

不指定列名时会自动挑常用的几列。百分比类的列会自动乘 100。
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as statsmod  # noqa: E402

DEFAULT_COLUMNS = [
    "insts", "cycles", "ipc", "cpi",
    "l1d_miss_rate", "l1d_mpki", "l1d_mshr_mpki",
    "l2_miss_rate", "l2_mpki",
    "bp_miss_rate", "bp_mpki",
    "wall_seconds",
]

RATE_COLUMNS = {
    "l1i_miss_rate", "l1d_miss_rate", "l2_miss_rate", "bp_miss_rate",
    "l1i_mshr_miss_rate", "l1d_mshr_miss_rate", "l2_mshr_miss_rate",
}

# summarize() 产出的所有列都是因变量；CSV 里剩下的才是这次扫描的自变量。
METRIC_COLUMNS = set(statsmod.summarize({})) | {"label", "error", "wall_seconds"}


def is_swept(rows, key):
    """判断某列是不是这次扫描真正变化的自变量。"""
    return len({r.get(key, "") for r in rows}) > 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    path = sys.argv[1]
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("空结果")
        return 1

    knobs = [k for k in rows[0] if k not in METRIC_COLUMNS and is_swept(rows, k)]
    if not knobs:
        knobs = ["label"]

    if len(sys.argv) > 2:
        metrics = sys.argv[2:]
    else:
        metrics = [c for c in DEFAULT_COLUMNS if c in rows[0] and any(rows[0].get(c))]

    header = knobs + metrics
    widths = {c: max(len(c), 10) for c in header}

    def fmt(row, col):
        raw = row.get(col, "")
        if col in knobs:
            return str(raw)
        if row.get("error"):
            return "ERR"
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return str(raw)
        if col in RATE_COLUMNS:
            return f"{100 * v:.2f}%"
        if abs(v) >= 100000:
            return f"{v:.4g}"
        return f"{v:.3f}"

    print("  ".join(c.rjust(widths[c]) for c in header))
    print("  ".join("-" * widths[c] for c in header))
    for row in rows:
        print("  ".join(fmt(row, c).rjust(widths[c]) for c in header))

    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n{len(errs)} 个点失败:")
        for r in errs:
            print(f"  {r.get('label', '?')}: {r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
