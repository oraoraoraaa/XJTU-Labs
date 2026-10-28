#!/usr/bin/env bash
# Explicit build entry point. Daily environment loading belongs to setup.sh.
if [ -n "${ZSH_VERSION-}" ]; then
    printf 'ERROR: run this script with Bash: bash /path/to/Computer-System/init.sh\n' >&2
    return 1 2>/dev/null || exit 1
fi
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'ERROR: execute init.sh with Bash; use source setup.sh to load your environment.\n' >&2
    return 1
fi
set -euo pipefail

if [[ "${1-}" == --help ]]; then
    printf 'Usage: bash init.sh\nOptional: CSSYS_JOBS=4 bash init.sh (default: 2 build jobs).\nBuilds gem5 only; does not run benchmarks or lab experiments.\n'
    exit 0
fi
if (( $# != 0 )); then
    printf 'ERROR: unexpected arguments. Use bash init.sh --help.\n' >&2
    exit 1
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
build_jobs="${CSSYS_JOBS:-2}"
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: CSSYS_JOBS must be a positive integer; received %s.\n' "$build_jobs" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    printf 'ERROR: uv is not on PATH. Install it or restore its PATH entry.\n' >&2
    exit 1
fi
if [[ ! -f "$project_root/gem5/SConstruct" ]]; then
    printf 'ERROR: missing gem5 sources at %s/gem5.\n' "$project_root" >&2
    exit 1
fi

cd -- "$project_root"
if ! uv sync --locked; then
    printf 'ERROR: uv sync failed. Check the error above; the build has not started.\n' >&2
    exit 1
fi
CSSYS_SETUP_FOR_BUILD=1 source "$project_root/setup.sh"

# The teacher snapshot omitted execute permission on this build-time generator.
chmod u+x "$project_root/gem5/util/cpt_upgrader.py"
log_dir="$project_root/lab-1/out/setup"
mkdir -p "$log_dir"
log_file="$log_dir/gem5-build-$(date +%Y%m%d-%H%M%S)-$$.log"

cd -- "$project_root/gem5"
printf 'Building gem5 with %s jobs. Log: %s\n' "$build_jobs" "$log_file"
if ! scons build/RISCV/gem5.opt -j"$build_jobs" 2>&1 | tee "$log_file"; then
    printf 'ERROR: gem5 build failed. Inspect the first error in %s.\nCompleted work is retained; rerun this script after fixing the cause.\n' "$log_file" >&2
    exit 1
fi
if ! build_info="$("$GEM5_BIN" --build-info)"; then
    printf 'ERROR: build finished but gem5 --build-info failed.\n' >&2
    exit 1
fi
if [[ "$build_info" != *'USE_RISCV_ISA = True'* ]]; then
    printf 'ERROR: the built gem5 does not report RISC-V support.\n' >&2
    exit 1
fi
printf '%s\n' "$build_info" > "$project_root/build-info.txt"
printf 'Build verified. Information saved to %s/build-info.txt\n' "$project_root"
printf 'To load this environment in your current shell:\n  source "%s/setup.sh"\n' "$project_root"
