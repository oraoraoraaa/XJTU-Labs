#!/usr/bin/env bash
# ============================================================================
#  Sweep the cache simulator over all required parameter combinations and
#  collect the results into results/results.csv
#
#    traces        : 022.li  047.tomcatv  078.swm256  085.gcc   (4)
#    cache sizes   : 8K 16K 32K 64K                              (4)
#    associativity : 1 2 4 8                                     (4)
#    block sizes   : 16 32 64 128                                (4)
#  -> 4 * 4 * 4 * 4 = 256 runs
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SIM=./cache_sim
TRACE_DIR="trace files"
OUT=results/results.csv

# Build the simulator if it is missing or out of date.
if [[ ! -x "$SIM" || src/cache_sim.cpp -nt "$SIM" ]]; then
    echo "Building simulator..."
    g++ -O2 -std=c++17 -o "$SIM" src/cache_sim.cpp
fi

TRACES=("022.li.din" "047.tomcatv.din" "078.swm256.din" "085.gcc.din")
SIZES=("8K" "16K" "32K" "64K")
ASSOCS=(1 2 4 8)
BLOCKS=(16 32 64 128)

mkdir -p results
echo "trace,cache_size,assoc,block_size,reads,writes,read_miss,write_miss,replacements,read_miss_rate,write_miss_rate,total_miss_rate" > "$OUT"

total=$(( ${#TRACES[@]} * ${#SIZES[@]} * ${#ASSOCS[@]} * ${#BLOCKS[@]} ))
n=0
for t in "${TRACES[@]}"; do
    for s in "${SIZES[@]}"; do
        for a in "${ASSOCS[@]}"; do
            for b in "${BLOCKS[@]}"; do
                n=$((n+1))
                printf "\r[%3d/%3d] %-16s size=%-4s assoc=%s block=%-3s" "$n" "$total" "$t" "$s" "$a" "$b"
                "$SIM" --trace "$TRACE_DIR/$t" --cache_size "$s" --assoc "$a" --block_size "$b" --csv >> "$OUT"
            done
        done
    done
done
printf "\nDone. Results written to %s (%d rows).\n" "$OUT" "$total"
