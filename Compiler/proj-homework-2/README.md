# QCompiler — QL → ARM64 (Homework 2, Path A: LL(1))

A complete QL compiler: **lexical analysis → LL(1) recursive-descent parsing →
QAST → QTAC three-address code → ARM64 assembly → peephole optimisation**, with
end-to-end verification on the `qsort` test program.

This implements **Path A** of the lab:
Task 1 (lexer) → Task 2A (LL(1) parser → QAST) → Task 2AB (QAST → QTAC) →
Task 3 (QTAC → ARM64, reusing/generalising Homework 1) → Task 4 (optimisation) →
Task 5 (integration test).

## Layout

```
proj-homework-2/
├── src/
│   ├── lexer.py        Task 1   QL source  -> token stream
│   ├── qast.py         Task 2AB QAST node definitions + pretty printer
│   ├── parser_ll1.py   Task 2A  LL(1) recursive descent -> QAST
│   ├── irgen.py        Task 2AB QAST -> QTAC + symbol table
│   ├── codegen.py      Task 3   QTAC -> ARM64 (Q2ARM templates)
│   ├── optimize.py     Task 4   ARM64 peephole optimisation
│   └── qcc.py          driver tying all stages together
├── runtime/main.c      C test harness (fills the array, calls the QL entry,
│                        prints + checks the result)
├── tests/qsort.ql      the qsort test program (QL source)
├── build/              generated artefacts (see below)
└── run.sh              one-command build + run, auto-detecting the host
```

## Quick start

```bash
./run.sh
```

`run.sh` auto-detects the host (Apple Silicon macOS vs Linux/Kunpeng aarch64),
compiles `tests/qsort.ql` through every stage, assembles the ARM64, links it
with the C harness, and runs **both** the unoptimised and optimised builds.

Expected output:

```
before: 5 3 8 1 9 2 7 4 6 0
after : 0 1 2 3 4 5 6 7 8 9
OK: array is sorted
```

## Running locally (Apple Silicon macOS)

The lab targets Huawei Kunpeng (ARM64 Linux), but Apple Silicon is **also
arm64**, so the whole pipeline runs natively with `cc` (Apple clang). `run.sh`
already does this automatically; the manual steps are:

```bash
# 1. compile QL -> tokens/AST/QTAC/ARM, emitting the macOS ARM64 variant
python3 src/qcc.py tests/qsort.ql -o build --target macos

# 2. assemble + link with the C harness, then run
cc -c build/qsort.s   -o build/qsort.o
cc -c runtime/main.c  -o build/main.o
cc build/qsort.o build/main.o -o build/qsort_run
./build/qsort_run
```

The only platform difference is how a global's address is materialised:
`--target macos` emits the Mach-O `adrp …@PAGE / add …@PAGEOFF` pair, while
`--target linux` (default) emits the plain `ADR` from the Q2ARM template.
Everything else — frame layout, calling convention, control flow — is identical.

## Running on Kunpeng / ARM64 Linux

```bash
# generate the Linux/Kunpeng assembly (default target)
python3 src/qcc.py tests/qsort.ql -o build --target linux

# assemble + link with the C harness and run
gcc -c build/qsort.s  -o build/qsort.o
gcc -c runtime/main.c -o build/main.o
gcc build/qsort.o build/main.o -o build/qsort_run
./build/qsort_run
```

The generated `.s` uses only standard GNU-as directives
(`.data`/`.text`/`.global`/`.balign`/`.skip`/`ADR`). You can also assemble the
object file with the bare assembler — `as -o build/qsort.o build/qsort.s` — and
link it together with the harness via `gcc`/`ld`.

> **Note on the harness.** The QL program (`qsort(a[], 0, 9)`) only *sorts* the
> global array; it never initialises or prints it, so a bare `_start`/`ld`
> executable would sort uninitialised memory and exit silently. To make the
> result observable, we link the generated functions with a small C driver
> (`runtime/main.c`) that fills `ql_a`, calls the QL entry point `ql_entry`, and
> prints/checks the sorted array. QL function/global names are prefixed with
> `ql_` in the assembly so they never clash with libc (e.g. `qsort`).

## Compile a single program / inspect stage outputs

```bash
python3 src/qcc.py tests/qsort.ql -o build           # all stages
# or run a single stage from stdin:
python3 src/lexer.py       < tests/qsort.ql          # token stream
python3 src/parser_ll1.py  < tests/qsort.ql          # QAST dump
python3 src/irgen.py       < tests/qsort.ql          # QTAC
python3 src/codegen.py     < build/qsort.qtac        # ARM64 (--macos for mac)
```

`qcc.py` writes these artefacts to `build/` (stem = source base name):

| file            | stage   | contents                                   |
|-----------------|---------|--------------------------------------------|
| `qsort.tokens`  | Task 1  | `(TYPE, value)` token stream               |
| `qsort.ast`     | Task 2A | QAST dump                                  |
| `qsort.qtac`    | Task 2AB| QTAC three-address code                    |
| `qsort.sym`     | Task 2AB| symbol table (globals, functions, locals)  |
| `qsort.s`       | Task 3  | unoptimised ARM64                          |
| `qsort.opt.s`   | Task 4  | peephole-optimised ARM64                   |

## Design notes

- **Word size = 8 bytes.** Ints are 64-bit (`X` registers), so `int a[10]`
  becomes the 80-byte global `i80 a`, matching the reference QTAC. Array element
  addresses are computed as `base + index*8` (the reference omits the `*8`; we
  make it explicit so the output is directly executable).
- **Register strategy (Task 3).** Every QTAC value that is not a global,
  immediate, or label gets a stack slot in the current frame; each instruction
  loads operands into scratch registers `X9`/`X10`, computes, and stores back.
  This is the "leave the virtual registers / spill everything" approach the lab
  explicitly permits, and it runs unmodified — no manual register substitution
  needed. The optimiser then removes the obvious redundancy.
- **Calling convention.** AAPCS64: arguments in `X0–X7`, return value in `X0`,
  prologue/epilogue exactly per the Q2ARM template
  (`stp/mov/sub … add/ldp/ret`).
- **Optimisation (Task 4).** Three safe, adjacent-instruction peephole passes:
  redundant-branch removal (`B Lx` immediately before `Lx:`), store/load and
  load/load forwarding, and self-move removal. On `qsort` it removes 18
  instructions; both builds produce identical, correct results.
- **Grammar transforms (Task 2A).** The raw QL grammar is left-recursive and
  ambiguous; `parser_ll1.py` documents the left-recursion elimination, left
  factoring after an identifier (disjoint SELECT sets with one token of
  look-ahead), and the single precedence ladder that folds `E` and `C` together
  to remove the `(E)` vs `(C)` ambiguity. The `arr[]` array-decay argument form
  used by the test program is handled as a special case.
