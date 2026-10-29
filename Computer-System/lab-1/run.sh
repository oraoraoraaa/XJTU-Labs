#!/usr/bin/env bash
set -euo pipefail

# Keep paths (including the simulated argv) consistent from any working directory.
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RESULT_DIR="$LAB_DIR/results"

usage() {
  cat<<'EOF'
Usage:
  ./run.sh TASK [--seed NUMBER]

Tasks:
  task1     Run qnn with 8 samples
  task2b    Run qnn with 64 samples
  task2c    Compare 4 binaries with 8 samples
  task3     Sweep four CPU models; save ROI metrics and scatter plot
  task3plot Plot the existing Task 3 CSV without rerunning simulations
  task4     Compare 16kB and 128kB L1D on timing, minor, and o3
  task5     Run 2, 4, 8, 16, 32 samples sequentially; save CSV and plots
  task6     Compare 16kB and 128kB L1D using dependent pointer chasing
  symbols   Inspect dot_i8 in the existing binaries; no simulation
  icache    Export I-Cache counters from existing runs for this seed
  all       Run task1, task2b, task2c, task3, task4, task5, and task6

Options:
  --seed NUMBER     Random seed; default: 2799
  -h, --help        Show this help

Results:
  out/      Raw simulator outputs and logs
  results/   Readable statistics summaries

Rerunning the same task and seed replaces its previous output.
EOF
}

TASK=""
SEED=2799

while (( $# > 0 )); do
  case "$1" in
    task1|task2b|task2c|task3|task3plot|task4|task5|task6|symbols|icache|all)
      if [[ -n "$TASK" ]]; then
        printf 'ERROR: Choose only one task.\n' >&2
        exit 2
      fi
      TASK="$1"
      shift
      ;;

    --seed)
      if (( $# < 2 )); then
        printf 'ERROR: --seed requires a number.\n' >&2
        exit 2
      fi
      SEED="$2"
      shift 2
      ;;

    -h|--help)
      usage
      exit 0
      ;;

    *)
      printf 'ERROR: Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK" ]]; then
  usage >&2
  exit 2
fi

if [[ ! "$SEED" =~ ^[0-9]{4}$ ]]; then
  printf 'ERROR: seed must be an 4 digits integer number.\n' >&2
  exit 2
fi

SEED_DECIMAL=$((10#$SEED))

cd "$LAB_DIR"
source ../setup.sh

run_case() {
  mkdir -p "$RESULT_DIR"
  local label="$1"
  local binary="$2"
  local samples="$3"
  local blocks="$4"

  local block
  local name="$label"
  if [[ "$label" == task2c ]]; then
    name="${label}_${binary}"
  fi

  local out_dir="out/${name}_o3_n${samples}_seed${SEED}"
  local summary="results/${name}_o3_n${samples}_seed${SEED}.txt"

  mkdir -p "$out_dir"
  "$GEM5_BIN" \
    --outdir="$out_dir" \
    framework/cssys_se.py \
    --cmd "bench/bin/riscv/$binary" \
    --options "$samples $SEED_DECIMAL" \
    --cpu o3 \
    2>&1 | tee "$out_dir/run.log"

  {
    for (( block=0; block<blocks; block++ )); do
      if (( block > 0 )); then
        printf '\n'
      fi
      printf 'Statistics block %s:\n' "$block"
      python framework/stats.py "${out_dir}/stats.txt" "$block"
    done
  } > "$summary"

  printf 'Saved summary: %s\n' "$summary"
}

export_icache() {
  local -a stats_files=()
  local path
  for path in out/*_seed"$SEED"/stats.txt; do
    [[ -f "$path" ]] && stats_files+=("$path")
  done
  if (( ${#stats_files[@]} == 0 )); then
    printf 'ERROR: no statistics found for seed %s.\n' "$SEED" >&2
    return 1
  fi
  python "$LAB_DIR/icache_report.py" \
    --output "$RESULT_DIR/task2c_icache_seed${SEED}.csv" "${stats_files[@]}"
}

task1() {
  run_case task1 qnn 8 3
}

task2b() {
  run_case task2b qnn 64 3
}

task2c() {
  local binary
  local blocks

  for binary in qnn qnn_noroi qnn_noalign qnn_noalign_noroi; do
    case "$binary" in
      *_noroi)    blocks=1 ;;
      *)          blocks=3 ;;
    esac
    run_case task2c "$binary" 8 "$blocks"
  done

  inspect_dot_i8
}

task3() {
  local name="task3_lab1_cpu_seed${SEED}"
  python framework/sweep.py \
    --name "$name" \
    --bench qnn --bench-args "8 $SEED_DECIMAL" \
    --vary cpu=atomic,timing,minor,o3 --jobs 4

  python framework/report.py "$RESULT_DIR/${name}.csv" \
    insts cycles ipc host_seconds host_inst_rate \
    l1d_miss_rate l1d_mshr_mpki \
    | tee "$RESULT_DIR/${name}.txt"

  plot_task3
}

plot_task3() {
  local name="task3_lab1_cpu_seed${SEED}"
  uv run --no-project --with matplotlib python "$LAB_DIR/task3_plot.py" \
    --input "$RESULT_DIR/${name}.csv" \
    --output "$RESULT_DIR/${name}.png" \
    --report-output "$LAB_DIR/lab-report/images/${name}.png"
}

task4() {
  local name="task4_lab1_fid_seed${SEED}"
  python framework/sweep.py \
    --name "$name" \
    --bench qnn --bench-args "8 $SEED_DECIMAL" \
    --vary cpu=timing,minor,o3 --vary l1d-size=16kB,128kB --jobs 6

  python framework/report.py "$RESULT_DIR/${name}.csv" \
    cycles ipc l1d_miss_rate l1d_mshr_mpki \
    | tee "$RESULT_DIR/${name}.txt"
}

task5() {
  local table="$RESULT_DIR/task5_lab1_scale_seed${SEED}.csv"
  python "$LAB_DIR/task5_scale.py" run \
    --seed "$SEED" --gem5 "$GEM5_BIN" --output "$table"
  # uv keeps the plotting dependency separate from the gem5 build environment.
  uv run --no-project --with matplotlib python "$LAB_DIR/task5_scale.py" plot \
    --input "$table"
}

task6() {
  local name="task6_lab1_fid_pchase_seed${SEED}"
  python framework/sweep.py \
    --name "$name" \
    --bench pchase --bench-args "256 200000 $SEED_DECIMAL" \
    --vary cpu=timing,minor,o3 --vary l1d-size=16kB,128kB --jobs 6

  python framework/report.py "$RESULT_DIR/${name}.csv" \
    insts cycles ipc l1d_miss_rate l1d_mshr_mpki \
    | tee "$RESULT_DIR/${name}.txt"
}

# Run a matching RISC-V utility locally, or use the benchmark Docker image.
riscv_tool() {
  local utility="$1"
  shift
  local prefix
  local -a docker_cmd=(docker)

  if [[ -n "${CROSS:-}" ]]; then
    "${CROSS}${utility}" "$@"
    return
  fi

  for prefix in riscv64-linux-gnu- riscv64-unknown-linux-gnu-; do
    if command -v "${prefix}${utility}" >/dev/null 2>&1; then
      "${prefix}${utility}" "$@"
      return
    fi
  done

  if [[ "$(uname -s)" == Darwin ]] && command -v colima >/dev/null 2>&1; then
    docker_cmd+=(--context colima-cssys)
  fi

  "${docker_cmd[@]}" run --rm \
    --mount "type=bind,source=$LAB_DIR/../bench,target=/bench,readonly" \
    --workdir /bench \
    cssys-riscv-toolchain:bookworm \
    "riscv64-linux-gnu-${utility}" "$@"
}

inspect_dot_i8() {
  local binary symbols address
  local table="$RESULT_DIR/task2c_dot_i8-addresses.csv"
  local disassembly="$RESULT_DIR/task2c_dot_i8-disassembly.txt"

  mkdir -p "$RESULT_DIR"
  # Check all inputs before replacing the previous inspection results.
  for binary in qnn qnn_noroi qnn_noalign qnn_noalign_noroi; do
    if [[ ! -f "bench/bin/riscv/$binary" ]]; then
      printf 'ERROR: missing %s; run task2c to build the binaries.\n' "$binary" >&2
      return 1
    fi
  done

  printf 'binary,function_address_hex,function_address_mod64\n' > "$table"
  : > "$disassembly"
  for binary in qnn qnn_noroi qnn_noalign qnn_noalign_noroi; do
    # Utilities use paths relative to bench, both locally and in Docker.
    symbols=$(cd bench && riscv_tool nm "bin/riscv/$binary")
    address=$(awk '$3 == "dot_i8" { print $1 }' <<< "$symbols")
    if [[ ! "$address" =~ ^[0-9a-fA-F]+$ ]]; then
      printf 'ERROR: could not find one dot_i8 symbol in %s.\n' "$binary" >&2
      return 1
    fi
    printf '%s,0x%s,%d\n' "$binary" "$address" "$((16#$address % 64))" >> "$table"
    printf '\n=== %s ===\n' "$binary" >> "$disassembly"
    (cd bench && riscv_tool objdump -d --disassemble=dot_i8 "bin/riscv/$binary") \
      >> "$disassembly"
  done
  cat "$table"
  printf 'Saved disassembly: %s\n' "$disassembly"
}

# Build the binaries needed by the selected task.
case "$TASK" in
  task1|task2b|task3|task4|task5)
    ../bench/build.sh bin/riscv/qnn
    ;;
  task2c)
    ../bench/build.sh bin/riscv/qnn bin/riscv/qnn_noroi noalign
    ;;
  task6)
    ../bench/build.sh bin/riscv/pchase
    ;;
  all)
    ../bench/build.sh bin/riscv/qnn bin/riscv/qnn_noroi noalign bin/riscv/pchase
    ;;
esac

# Execute the selected task.
case "$TASK" in
  task1)  task1 ;;
  task2b) task2b ;;
  task2c) task2c ;;
  task3) task3 ;;
  task3plot) plot_task3 ;;
  task4) task4 ;;
  task5) task5 ;;
  task6) task6 ;;
  symbols) inspect_dot_i8 ;;
  icache) export_icache ;;
  all)
    task1
    task2b
    task2c
    task3
    task4
    task5
    task6
    ;;
esac

case "$TASK" in
  task1|task2b|task2c|all) export_icache ;;
esac

rm -rf m5out

