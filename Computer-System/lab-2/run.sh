# Task 1
python3 framework/sweep.py --name task1_size \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" \
  --vary l1d-size=8kB,16kB,32kB,64kB,128kB,256kB --jobs 6

python3 framework/sweep.py --name task1_size_nolat \
--bench qnn --bench-args "8 2799" \
--base "--cpu o3 --auto-latency off" \
--vary l1d-size=8kB,16kB,32kB,64kB,128kB,256kB --jobs 6

echo "size report:" >> results/task1_size_report.txt
python3 framework/report.py results/task1_size.csv \
  cycles ipc l1d_miss_rate l1d_mpki l1d_mshr_mpki l1d_mlp >> results/task1_size_report.txt

echo "" >> results/task1_size_report.txt
echo "size_nolat report:" >> results/task1_size_report.txt
python3 framework/report.py results/task1_size_nolat.csv \
  cycles ipc l1d_miss_rate l1d_mpki l1d_mshr_mpki l1d_mlp >> results/task1_size_report.txt

# Task 2
python3 framework/sweep.py --name task2_assoc \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" \
  --vary l1d-size=32kB,64kB,128kB --vary l1d-assoc=1,2,4,8 --jobs 12

python3 framework/sweep.py --name task2_repl \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l1d-size 64kB" \
  --vary l1d-assoc=1,2,4,8 --vary l1d-repl=lru,random --jobs 8

echo "assoc report:" >> results/task2_assoc_report.txt
python3 framework/report.py results/task2_assoc.csv \
  cycles ipc l1d_miss_rate l1d_mshr_mpki >> results/task2_assoc_report.txt
echo "repl report:" >> results/task2_repl_report.txt
python3 framework/report.py results/task2_repl.csv \
  cycles ipc l1d_miss_rate l1d_mshr_mpki >> results/task2_repl_report.txt

# Task 3(a)
python3 framework/sweep.py --name task3a_line \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" \
  --vary cacheline=16,32,64,128,256 --jobs 5

echo "line report" >> results/task3a_line_report.txt
python3 framework/report.py results/task3a_line.csv \
  cycles ipc l1d_mshr_mpki l1d_mlp >> results/task3a_line_report.txt

# Task 3(b): MSHR hypothesis test
python3 framework/sweep.py --name task3b_line_mshr \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" \
  --vary cacheline=16,32,64 --vary l1d-mshrs=4,16,64 --jobs 9

echo "line_mshr report:" > results/task3b_line_mshr_report.txt
python3 framework/report.py results/task3b_line_mshr.csv \
  cycles ipc l1d_mshr_mpki l1d_mlp >> results/task3b_line_mshr_report.txt

echo "" >> results/task3b_line_mshr_report.txt
echo "no_mshrs blockedCycles:" >> results/task3b_line_mshr_report.txt
grep 'dcache.blockedCycles::no_mshrs' out/sweeps/task3b_line_mshr/*/stats.txt >> results/task3b_line_mshr_report.txt

# Task 3(b) Q5: I-Cache fetch-side check (16B vs 64B)
echo "" >> results/task3b_line_mshr_report.txt
echo "fetch stats (16B):" >> results/task3b_line_mshr_report.txt
grep -E 'fetch.cacheLines|icacheStallCycles|cpu.numCycles ' \
  out/sweeps/task3a_line/p000_cacheline-16/stats.txt >> results/task3b_line_mshr_report.txt

echo "" >> results/task3b_line_mshr_report.txt
echo "fetch stats (64B):" >> results/task3b_line_mshr_report.txt
grep -E 'fetch.cacheLines|icacheStallCycles|cpu.numCycles ' \
  out/sweeps/task3a_line/p002_cacheline-64/stats.txt >> results/task3b_line_mshr_report.txt

# Task 3(b) Q6: pchase contrast (no spatial locality)
python3 framework/sweep.py --name task3b_line_pchase \
  --bench pchase --bench-args "256 200000 2799" \
  --base "--cpu o3" \
  --vary l1d-size=32kB,128kB --vary cacheline=64,256 --jobs 4

echo "line_pchase report:" > results/task3b_line_pchase_report.txt
python3 framework/report.py results/task3b_line_pchase.csv \
  cycles ipc l1d_mshr_mpki l1d_mlp >> results/task3b_line_pchase_report.txt


# Task 4: MSHR depth (miss_rate vs mshr_mpki)
python3 framework/sweep.py --name task4_mshr \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" \
  --vary l1d-mshrs=1,2,4,8,16 --jobs 5

echo "mshr report:" > results/task4_mshr_report.txt
python3 framework/report.py results/task4_mshr.csv \
  cycles ipc l1d_miss_rate l1d_mshr_mpki l1d_mlp >> results/task4_mshr_report.txt

# Task 5: L2 size & DRAM traffic
python3 framework/sweep.py --name task5_l2 \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" \
  --vary l2-size=128kB,256kB,512kB,1MB,2MB --jobs 5

echo "l2 report:" > results/task5_l2_report.txt
python3 framework/report.py results/task5_l2.csv \
  cycles ipc l2_miss_rate l2_accesses dram_bytes >> results/task5_l2_report.txt


# Task 6: Final configuration proposal on qnn
python3 framework/sweep.py --name task6_final \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l1d-size 128kB --l1d-assoc 2 --cacheline 128 --l2-size 256kB" \
  --vary l1d-repl=lru,random --jobs 2

echo "task6_final (qnn) report:" > results/task6_final_report.txt
python3 framework/report.py results/task6_final.csv \
  cycles ipc l1d_miss_rate l1d_mshr_mpki l2_accesses dram_bytes >> results/task6_final_report.txt

# Task 6 Q3: Cross-workload validation on pchase
python3 framework/sweep.py --name task6_final_x \
  --bench pchase --bench-args "256 200000 2799" \
  --base "--cpu o3 --l1d-assoc 2 --cacheline 128 --l2-size 256kB" \
  --vary l1d-size=32kB,128kB --jobs 2

echo "" >> results/task6_final_report.txt
echo "task6_final_x (pchase) report:" >> results/task6_final_report.txt
python3 framework/report.py results/task6_final_x.csv \
  cycles ipc l1d_mshr_mpki l1d_mlp l2_accesses >> results/task6_final_report.txt


# Section IV Q2: Latency model sensitivity (conclusion robustness)
python3 framework/sweep.py --name q2_latsens \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3" \
  --vary l1d-size=8kB,32kB,128kB,256kB --vary l1-latency=2,3,5 --jobs 12

echo "q2_latsens report:" > results/q2_latsens_report.txt
python3 framework/report.py results/q2_latsens.csv \
  cycles ipc l1d_mshr_mpki >> results/q2_latsens_report.txt

# Section IV Q3: Replacement policy family (RRIP/BRRIP vs LRU/Random/FIFO/NRU)
python3 framework/sweep.py --name q3_replfam \
  --bench qnn --bench-args "8 2799" \
  --base "--cpu o3 --l1d-size 64kB --l1d-assoc 4" \
  --vary l1d-repl=lru,random,fifo,nru,rrip,brrip --jobs 6

echo "q3_replfam report:" > results/q3_replfam_report.txt
python3 framework/report.py results/q3_replfam.csv \
  cycles ipc l1d_miss_rate l1d_mshr_mpki >> results/q3_replfam_report.txt
