# Computer System labs

Shared benchmark sources, framework scripts, gem5, and the uv environment live
in this directory. Each lab keeps its own report, results, and raw simulation
output. `lab-1` links to the shared `bench`, `framework`, and `gem5` directories.

## Load the environment

From this directory, in Bash or Zsh:

```sh
source setup.sh
cd lab-1
```

From another directory, source the script by its full path. It leaves your
working directory and shell options unchanged. It checks the pinned Python
version, development headers, shared library, and any existing gem5 executable.
It sets `PYTHON_CONFIG`, `LIBRARY_PATH`, and `GEM5_BIN` on Linux and macOS,
plus `LD_LIBRARY_PATH` for runtime library loading on Linux. `LIBRARY_PATH`
lets the linker find uv's Python library even when `python-config` omits its
directory on macOS. Sourcing it repeatedly does not duplicate library paths.
The build script skips checking an older simulator executable while loading
the environment, allowing a broken or missing executable to be rebuilt.

The script must be **sourced** so its exports reach your current shell.
`bash setup.sh` deliberately reports an error.

## Build or resume building gem5

Only run this when you intend to build:

```sh
bash init.sh
# Or explicitly choose build parallelism:
CSSYS_JOBS=4 bash init.sh
```

The script can be invoked by its full path from any directory. It synchronizes
the uv environment using `uv.lock`, loads the environment, repairs the known
checkpoint-generator permission issue, and builds `gem5/build/RISCV/gem5.opt`.
It defaults to two concurrent jobs, retains completed work, and stops on failure.
Build logs are kept separately in `lab-1/out/setup/`. A successful executable
check refreshes `build-info.txt`. The script does not compile benchmarks or run
lab experiments. Afterward, source `setup.sh` in your terminal to load its exports.

## Run lab commands

Build the lab 1 benchmarks and run Task 1:

```sh
cd lab-1
./run.sh
```

The runner uses seed `2799`, stops if compilation fails, saves simulation output
under `out/task1_o3_n8_seed2799/`, and prints all three statistics blocks.
It also works when invoked by its full path from another directory.

`bench/build.sh` uses a local `riscv64-linux-gnu-gcc` or
`riscv64-unknown-linux-gnu-gcc` when available (`CROSS` can select another prefix).
The Homebrew `riscv64-unknown-elf-gcc` is a bare-metal compiler; this lab needs
Linux/glibc binaries. Without a local Linux compiler, the helper builds them
with GCC 12 in a Debian Bookworm Docker image. On macOS with Colima installed,
it starts a separate `cssys` profile (2 CPUs, 2 GiB RAM, 10 GiB virtual disk),
without changing your active Docker context. The first build needs internet
access; subsequent builds reuse the image. gem5 still runs natively on the Mac.
Use the same compiler for all compared binaries because compiler versions and
libc change code layout and measured performance.

For other benchmark targets, pass the normal Makefile targets to the helper:

```sh
./bench/build.sh all noroi noalign
# Stop the compiler VM when finished (gem5 does not need it):
colima stop --profile cssys
```

Run simulations and invoke `framework/sweep.py` through the links inside the
lab directory. This keeps that lab's `out/` and `results/` files together.
The shared `bench/bin/` binaries and simulator build are ignored by Git;
keep `results/*.csv` for submission. Lab 1 uses seed `2799`.
