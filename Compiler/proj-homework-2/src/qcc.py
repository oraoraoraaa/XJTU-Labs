"""QCompiler driver: QL source -> tokens -> QAST -> QTAC -> ARM (+opt).

Usage:
    python3 qcc.py <source.ql> [-o <outdir>] [--target linux|macos]

    --target linux   ARM64 for Linux/Kunpeng (default; ADR addressing,
                     `as`/`ld` or gcc).  This is the deliverable .s.
    --target macos   ARM64 for Apple Silicon (ADRP/@PAGE addressing) so the
                     same program can be assembled and run locally with cc.

Writes the following artefacts into <outdir> (default: ../build), using the
source file's base name as the stem:

    <stem>.tokens   token stream            (Task 1)
    <stem>.ast      QAST dump               (Task 2A)
    <stem>.qtac     QTAC three-address code (Task 2AB)
    <stem>.sym      symbol table            (Task 2AB)
    <stem>.s        unoptimised ARM64        (Task 3)
    <stem>.opt.s    peephole-optimised ARM64 (Task 4)
"""

import os
import sys

from lexer import tokenize, dump_tokens
from parser_ll1 import parse
import qast
from irgen import generate as ir_generate, emit_qtac, emit_symtab
from codegen import generate as arm_generate
from optimize import optimize


def compile_file(path, outdir, target="linux"):
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        src = f.read()

    os.makedirs(outdir, exist_ok=True)

    def write(ext, text):
        p = os.path.join(outdir, stem + ext)
        with open(p, "w") as f:
            f.write(text)
        return p

    # 1. lexical analysis
    tokens = tokenize(src)
    write(".tokens", dump_tokens(tokens))

    # 2A. parsing -> QAST
    tree = parse(tokens)
    write(".ast", qast.dump(tree))

    # 2AB. QAST -> QTAC + symbol table
    g = ir_generate(tree)
    qtac_text = emit_qtac(g)
    write(".qtac", qtac_text)
    write(".sym", emit_symtab(g))

    # 3. QTAC -> ARM64
    asm = arm_generate(qtac_text, target)
    write(".s", asm)

    # 4. peephole optimisation
    opt_asm, stats = optimize(asm)
    write(".opt.s", opt_asm)

    print(f"compiled {path} -> {outdir}/{stem}.*")
    print(f"  optimiser removed {stats['removed']} instruction(s) "
          f"({stats['branches']} redundant branch, "
          f"{stats['reloads']} redundant reload, "
          f"{stats['selfmoves']} self-move)")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    path = argv[1]
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "build")
    if "-o" in argv:
        outdir = argv[argv.index("-o") + 1]
    target = "linux"
    if "--target" in argv:
        target = argv[argv.index("--target") + 1]
    print(f"target = {target}")
    compile_file(path, outdir, target)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
