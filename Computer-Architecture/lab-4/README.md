# Lab 4 — Cache Basic Performance Analysis (Simulator A)

A self-written, configurable **set-associative cache simulator** used to study how
cache size, associativity, and block size affect the miss rate.

- **Replacement policy:** LRU
- **Write-miss policy:** write-allocate (write-back on hit)
- **Modeled accesses:** only `load` (type 0) and `store` (type 1); instruction
  fetches (type 2) are ignored, as permitted by the lab.

## Files

```
src/cache_sim.cpp     # the simulator (C++17, single file)
run_experiments.sh    # sweeps all 256 parameter combinations -> results/results.csv
results/results.csv   # generated experiment data
trace files/*.din     # the 4 provided trace files
```

## Build

```bash
g++ -O2 -std=c++17 -o cache_sim src/cache_sim.cpp
```

## Run a single configuration

```bash
./cache_sim --trace "trace files/022.li.din" \
            --cache_size 32K --assoc 4 --block_size 64
```

The four required parameters:

| Flag            | Meaning                | Accepted values                |
|-----------------|------------------------|--------------------------------|
| `--trace`       | trace input file       | any `.din` file                |
| `--cache_size`  | cache size in bytes    | `8K 16K 32K 64K` (`K`/`M` ok)  |
| `--assoc`       | associativity (ways)   | `1 2 4 8`                      |
| `--block_size`  | block size in bytes    | `16 32 64 128`                 |

Optional: `--csv` prints one machine-readable line instead of the table; `--help`.

All three sizes must be powers of two and satisfy
`block_size ≤ cache_size` and `(cache_size / block_size) % assoc == 0`.

### Example output

```
==================== Cache Simulation Result ====================
 Trace file       : trace files/022.li.din
 Cache size       : 32768 B (32 KB)
 Block size       : 64 B
 Associativity    : 4-way
 Number of sets   : 128
 Replacement      : LRU      Write-miss: write-allocate
-----------------------------------------------------------------
 Data accesses    : 257749  (read 155399, write 102350)
 Read  misses     : 452
 Write misses     : 872
 Replaced blocks  : 812
 Read  miss rate  : 0.2909 %
 Write miss rate  : 0.8520 %
 Total miss rate  : 0.5137 %
=================================================================
```

## Run the full experiment sweep

Sweeps **4 traces × 4 cache sizes × 4 associativities × 4 block sizes = 256 runs**
and writes a CSV for the report (auto-builds the simulator if needed):

```bash
./run_experiments.sh
```

Output `results/results.csv` columns:

```
trace, cache_size, assoc, block_size,
reads, writes, read_miss, write_miss, replacements,
read_miss_rate, write_miss_rate, total_miss_rate
```

## Trace format

Each line is `access_type address [size/data]`, address in hex:
`0` = load data, `1` = store data, `2` = instruction fetch (ignored).
