#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
CSSYS_SETUP_FOR_BUILD=1 source ../setup.sh

# Task 1a: L1D replacement policies (64 KiB / 4-way cyclic scan pathology)
python3 framework/sweep.py --name lab3_repl --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l1d-size 64kB --l1d-assoc 4" \
  --vary l1d-repl=lru,fifo,secondchance,treeplru,lfu,random,mru,lip,bip,rrip,brrip,nru,drrip,ship,ship_pc \
  --jobs 8
python3 framework/report.py results/lab3_repl.csv cycles ipc l1d_misses l1d_mshr_misses l1d_mshr_mpki > results/lab3_repl_report.txt

grep -A3 'replacement_policy' out/sweeps/lab3_repl/p007_l1d-repl-lip/config.ini > results/task3_config_check.txt
grep -A6 'replacement_policy' out/sweeps/lab3_repl/p010_l1d-repl-brrip/config.ini >> results/task3_config_check.txt

# Task 1b: Associativity self-check (64 KiB, 2/4/8/16 ways)
python3 framework/sweep.py --name lab3_assoc --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l1d-size 64kB" \
  --vary l1d-assoc=2,4,8,16 --vary l1d-repl=lru,mru --jobs 8
python3 framework/report.py results/lab3_assoc.csv cycles ipc l1d_mshr_misses l1d_mshr_mpki > results/lab3_assoc_report.txt

# Task 1c: Control group without pathology (128 KiB / 4-way)
python3 framework/sweep.py --name lab3_nopath --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l1d-size 128kB --l1d-assoc 4" \
  --vary l1d-repl=lru,random,mru,bip --jobs 4
python3 framework/report.py results/lab3_nopath.csv cycles ipc l1d_mshr_misses l1d_mshr_mpki > results/lab3_nopath_report.txt

# Task 2c: Four-way verification of CssysRP (requires rebuilt gem5.opt)
python3 framework/sweep.py --name lab3_mine3 --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l1d-size 64kB --l1d-assoc 4" \
  --vary l1d-repl=lru,cssys,mru,cssys_mru --jobs 4
python3 framework/report.py results/lab3_mine3.csv cycles ipc l1d_misses l1d_mshr_misses l1d_mshr_mpki > results/lab3_mine3_report.txt

# Task 3b: L1D CssysStats instrumentation (ROI block 1)
python3 framework/sweep.py --name lab3_mine4 --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l1d-size 64kB --l1d-assoc 4" \
  --vary l1d-repl=cssys --jobs 1

# Task 3c: L2 replacement policies (64 KiB / 4-way L2) + CssysStats
python3 framework/sweep.py --name lab3_l2repl --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l2-size 64kB --l2-assoc 4" \
  --vary l2-repl=lru,fifo,secondchance,treeplru,lfu,random,mru,lip,bip,rrip,brrip,nru,drrip,ship,ship_pc \
  --jobs 8
python3 framework/report.py results/lab3_l2repl.csv cycles ipc l2_accesses l2_mshr_misses l2_mshr_mpki > results/lab3_l2repl_report.txt

python3 framework/sweep.py --name lab3_l2cssys --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l2-size 64kB --l2-assoc 4" \
  --vary l2-repl=cssys --jobs 1

# Task 4: L1D prefetchers on 32 KiB / 2-way baseline
python3 framework/sweep.py --name lab3_pf --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" --vary l1d-pf=none,tagged,stride,bop,ampm,isb --jobs 6
python3 framework/report.py results/lab3_pf.csv cycles ipc l1d_misses l1d_mshr_mpki \
  l1d_pf_issued l1d_pf_useful l1d_pf_accuracy l1d_pf_coverage l1d_mlp > results/lab3_pf_report.txt

# Task 5: Prefetcher vs capacity + area/energy evaluation
python3 framework/sweep.py --name lab3_pf_vs_size --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" --vary l1d-size=32kB,64kB,128kB \
  --vary l1d-pf=none,stride --jobs 6
python3 framework/report.py results/lab3_pf_vs_size.csv cycles ipc l1d_misses l1d_mshr_mpki \
  l1d_pf_issued l1d_pf_useful l1d_pf_accuracy l1d_mlp > results/lab3_pf_vs_size_report.txt
python3 framework/area.py out/sweeps/lab3_pf_vs_size/*/ > results/lab3_pf_vs_size_area.txt

# Task 6: Prefetcher cost on random pointer chasing (pchase)
python3 framework/sweep.py --name lab3_pf_pchase --bench pchase \
  --bench-args "256 200000 2799" \
  --base "--cpu o3" --vary l1d-pf=none,tagged,stride,ampm --jobs 4
python3 framework/report.py results/lab3_pf_pchase.csv cycles ipc l1d_mshr_mpki \
  l1d_pf_issued l1d_pf_useful l1d_pf_accuracy dram_bytes > results/lab3_pf_pchase_report.txt
