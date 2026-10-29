#!/usr/bin/env bash
set -euo pipefail

# Keep paths (including the simulated argv) consistent from any working directory.
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RESULT_DIR="$LAB_DIR/result"

usage() {
  cat<<'EOF'
Usage:
  ./run.sh TASK [--seed NUMBER]

Tasks:
  task1     Run qnn with 8 samples
  task2b    Run qnn with 64 samples
  task2c    Compare 4 binaries with 8 samples
  all       Run task1, task2b, and task2c

Options:
  --seed NUMBER     Random seed; default: 2799
  -h, --help        Show this help

Results:
  out/      Raw simulator outputs and logs
  result/   Readable statistics summaries

Rerunning the same task and seed replaces its previous output.
EOF
}

TASK=""
SEED=2799

while (( $# > 0 )); do
  case "$1" in
    task1|task2b|task2c|all)
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
  local summary="result/${name}_o3_n${samples}_seed${SEED}.txt"

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
    run_case task2c $binary 8 $blocks
  done
}

# Build the binaries needed by the selected task.
case "$TASK" in
  task1|task2b)
    ../bench/build.sh bin/riscv/qnn
    ;;
  task2c|all)
    ../bench/build.sh bin/riscv/qnn bin/riscv/qnn_noroi noalign
    ;;
esac

# Execute the selected task.
case "$TASK" in
  task1)  task1 ;;
  task2b) task2b ;;
  task2c) task2c ;;
  all)
    task1
    task2b
    task2c
    ;;
esac

rm -rf m5out
