#!/usr/bin/env bash
# QCompiler end-to-end build & run (Task step 5: integration test).
#
# Compiles tests/qsort.ql through every stage, assembles the generated ARM64,
# links it with the C test harness, and runs it -- both the unoptimised and
# the peephole-optimised assembly.
#
# Auto-detects the host: Apple Silicon (Darwin) uses the macOS target and
# `cc`; Linux/Kunpeng (aarch64) uses the linux target and `gcc`.

set -e
cd "$(dirname "$0")"

SRC=tests/qsort.ql
BUILD=build
STEM=qsort

# -- pick target + compiler for the host ------------------------------------
case "$(uname -s)" in
    Darwin) TARGET=macos ; CC=${CC:-cc}  ;;
    *)      TARGET=linux ; CC=${CC:-gcc} ;;
esac
echo "host    : $(uname -s) $(uname -m)"
echo "target  : $TARGET   (compiler: $CC)"
echo

# -- 1. compile QL -> tokens/AST/QTAC/ARM -----------------------------------
echo "== Stage 1-4: QL -> Token -> AST -> QTAC -> ARM (+opt) =="
python3 src/qcc.py "$SRC" -o "$BUILD" --target "$TARGET"
echo

# -- helper: assemble .s + C harness, link, run -----------------------------
build_and_run () {
    local s="$1" tag="$2" exe="$3"
    echo "== $tag =="
    echo "$CC -c $s -o $BUILD/$exe.o"
    $CC -c "$s" -o "$BUILD/$exe.o"
    echo "$CC -c runtime/main.c -o $BUILD/main.o"
    $CC -c runtime/main.c -o "$BUILD/main.o"
    echo "$CC $BUILD/$exe.o $BUILD/main.o -o $BUILD/$exe"
    $CC "$BUILD/$exe.o" "$BUILD/main.o" -o "$BUILD/$exe"
    echo "./$BUILD/$exe"
    "./$BUILD/$exe"
    echo
}

# -- 2. unoptimised --------------------------------------------------------
build_and_run "$BUILD/$STEM.s"     "Assemble + link + run (unoptimised)" "${STEM}_run"

# -- 3. optimised ----------------------------------------------------------
build_and_run "$BUILD/$STEM.opt.s" "Assemble + link + run (optimised)"   "${STEM}_opt_run"

echo "All artefacts are in $BUILD/:"
ls -1 "$BUILD"
