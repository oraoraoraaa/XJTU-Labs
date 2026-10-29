"""Export per-region and whole-program instruction-fetch counters from gem5."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "framework"))
from stats import parse_blocks


COUNTERS = {
    "instructions": "simInsts",
    "cycles": "system.cpu.numCycles",
    "fetch_block_requests": "system.cpu.fetch.cacheLines",
    "icache_accesses": "system.cpu.icache.overallAccesses::total",
    "icache_misses": "system.cpu.icache.overallMisses::total",
    "icache_stall_cycles": "system.cpu.fetchStats0.icacheStallCycles",
}


def rows_for(path):
    blocks = parse_blocks(path)
    if len(blocks) not in (1, 3):
        raise ValueError(f"{path}: expected 1 or 3 statistics blocks, got {len(blocks)}")
    totals = dict.fromkeys(COUNTERS, 0)
    regions = ["whole_program"] if len(blocks) == 1 else ["startup", "roi", "shutdown"]
    for index, (block, region) in enumerate(zip(blocks, regions)):
        # Missing counters are errors, rather than misleading zero measurements.
        values = {column: int(block[key]) for column, key in COUNTERS.items()}
        for column, value in values.items():
            totals[column] += value
        yield make_row(path, region, index, values)
    if len(blocks) == 3:
        yield make_row(path, "whole_program", "all", totals)


def make_row(path, region, block, values):
    cycles = values["cycles"]
    accesses = values["icache_accesses"]
    return {
        "run": path.parent.name,
        "region": region,
        "block": block,
        **values,
        "icache_stall_percent": round(100 * values["icache_stall_cycles"] / cycles, 6) if cycles else "",
        "icache_miss_percent": round(100 * values["icache_misses"] / accesses, 6) if accesses else "",
        "stats_path": str(path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("stats", type=Path, nargs="+")
    args = parser.parse_args()
    # Parse every input before replacing an existing report.
    rows = [row for path in args.stats for row in rows_for(path)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved I-Cache report: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
