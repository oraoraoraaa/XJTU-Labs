#!/usr/bin/env bash
set -euo pipefail

# Keep paths (including the simulated argv) consistent from any working directory.
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$LAB_DIR"
source ../setup.sh

../bench/build.sh bin/riscv/qnn bin/riscv/qnn_noroi noalign

OUT_DIR=out/task1_o3_n8_seed2799
mkdir -p "$OUT_DIR"
"$GEM5_BIN" \
  --outdir="$OUT_DIR" \
  framework/cssys_se.py \
  --cmd bench/bin/riscv/qnn \
  --options "8 2799" \
  --cpu o3 \
  2>&1 | tee "$OUT_DIR/run.log"

# Task 1: startup, inference ROI, and shutdown, respectively.
for block in 0 1 2; do
  printf '\nStatistics block %s:\n' "$block"
  python framework/stats.py "$OUT_DIR/stats.txt" "$block"
done
