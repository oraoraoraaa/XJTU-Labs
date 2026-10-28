#!/usr/bin/env bash
# Build Linux/glibc RISC-V benchmarks; leave simulation on the host.
set -euo pipefail
BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -n "${CROSS:-}" ]]; then
  if ! command -v "${CROSS}gcc" >/dev/null 2>&1; then
    printf 'ERROR: compiler not found: %sgcc\n' "$CROSS" >&2
    exit 1
  fi
  exec make -C "$BENCH_DIR" "CROSS=$CROSS" "$@"
fi
for prefix in riscv64-linux-gnu- riscv64-unknown-linux-gnu-; do
  if command -v "${prefix}gcc" >/dev/null 2>&1; then
    exec make -C "$BENCH_DIR" "CROSS=$prefix" "$@"
  fi
done

if ! command -v docker >/dev/null 2>&1; then
  printf 'ERROR: install a RISC-V Linux/glibc compiler (riscv64-linux-gnu-gcc), or Docker.\nThe riscv64-unknown-elf compiler targets bare metal and cannot replace it.\n' >&2
  exit 1
fi

# Use a separate Colima profile on macOS without changing the active context.
DOCKER=(docker)
if [[ "$(uname -s)" == Darwin ]] && command -v colima >/dev/null 2>&1; then
  DOCKER+=(--context colima-cssys)
  if ! "${DOCKER[@]}" info >/dev/null 2>&1; then
    colima start --profile cssys --cpu 2 --memory 2 --disk 10 \
      --vm-type vz --mount-type virtiofs --activate=false
  fi
fi
if ! "${DOCKER[@]}" info >/dev/null 2>&1; then
  printf 'ERROR: Docker is not running. Start Docker and retry.\n' >&2
  exit 1
fi

IMAGE=cssys-riscv-toolchain:bookworm
if ! "${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
  "${DOCKER[@]}" build -t "$IMAGE" - < "$BENCH_DIR/toolchain.Dockerfile"
fi
exec "${DOCKER[@]}" run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$BENCH_DIR,target=/bench" --workdir /bench \
  "$IMAGE" make "CROSS=riscv64-linux-gnu-" "$@"
