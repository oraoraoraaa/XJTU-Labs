"""QTAC -> ARM64 assembly (Task step 3, common to A/B/C).

This is the homework-1 QTAC->ARM stage generalised to the full QTAC language
(multiple functions, calls, memory access, control flow) using the Q2ARM
instruction templates from materials.md.

Register / storage strategy
---------------------------
Every QTAC value that is not a global, an immediate, or a label (i.e. every
parameter, local and temporary `t_i`) is given a slot in the current
function's stack frame.  Each instruction loads its operands into scratch
registers, computes, and stores the result back.  This is the "no register
allocator -> everything spilled" strategy explicitly permitted by the lab
(materials: virtual registers may be left for a later pass; here we lower
them straight to stack slots so the output runs unmodified).  The optional
peephole pass in optimize.py cleans up the resulting redundancy.

Calling convention: AAPCS64.  Arguments in X0..X7, return value in X0,
frame set up exactly as the Q2ARM prologue/epilogue template prescribes:

    stp x29, x30, [sp, #-16]!
    mov x29, sp
    sub sp, sp, #<width>
    ...
    add sp, sp, #<width>
    ldp x29, x30, [sp], #16
    ret

Portability: symbols the C driver references (function entries and the
global array) are emitted under both `name:` and `_name:` so the file
assembles and links on Linux/Kunpeng (ELF, no underscore) *and* macOS
(Mach-O, leading underscore) -- the same trick homework-1 used.
"""

import re
import sys

WORD = 8

# QL function names are prefixed so they never clash with libc (e.g. qsort).
PREFIX = "ql_"

AOP = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "SDIV"}
JROP = {"<": "B.LT", "<=": "B.LE", ">": "B.GT", ">=": "B.GE",
        "==": "B.EQ", "!=": "B.NE"}

SCRATCH_A = "X9"
SCRATCH_B = "X10"


class Func:
    def __init__(self, name, params):
        self.name = name
        self.params = params          # ordered param names
        self.insts = []               # list of raw QTAC instruction strings
        self.slots = {}               # var name -> byte offset
        self.width = 0


def parse_qtac(text):
    """Parse a QTAC program into (globals, functions)."""
    globals_ = []                     # (name, size_bytes)
    funcs = []
    cur = None
    for raw in text.splitlines():
        line = raw.strip().rstrip(";").strip()
        if not line:
            continue
        m = re.match(r"^i(\d+)\s+(\w+)$", line)
        if m and cur is None:
            globals_.append((m.group(2), int(m.group(1))))
            continue
        m = re.match(r"^define\s+(\w+)\s*\(([^)]*)\)\s*\{$", line)
        if m:
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            cur = Func(m.group(1), params)
            continue
        if line == "}":
            funcs.append(cur)
            cur = None
            continue
        if cur is not None:
            cur.insts.append(line)
    return globals_, funcs


class CodeGen:
    def __init__(self, globals_, funcs, target="linux"):
        self.globals = {n for n, _ in globals_}
        self.global_sizes = dict(globals_)
        self.funcs = funcs
        self.func_names = {f.name for f in funcs}
        self.target = target          # "linux" (Kunpeng) | "macos"
        self.out = []

    # -- operand classification -------------------------------------------
    def is_imm(self, x):
        return x.lstrip("-").isdigit()

    def is_global(self, x):
        return x in self.globals

    def label(self, name):
        """Map a QTAC label/func name to its assembly label."""
        if name in self.func_names:
            return PREFIX + name
        return name

    # -- frame layout ------------------------------------------------------
    def layout(self, fn):
        order = list(fn.params)        # params occupy the first slots
        seen = set(order)

        def consider(tok):
            if tok in seen:
                return
            if self.is_imm(tok) or self.is_global(tok):
                return
            seen.add(tok)
            order.append(tok)

        for inst in fn.insts:
            for tok in self.operands_of(inst):
                consider(tok)

        for idx, name in enumerate(order):
            fn.slots[name] = idx * WORD
        n = len(order)
        fn.width = ((n * WORD + 15) // 16) * 16   # 16-byte aligned

    def operands_of(self, inst):
        """Value operands appearing in an instruction (no labels/opcodes)."""
        m = re.match(r"^LABEL\s+\w+$", inst)
        if m:
            return []
        m = re.match(r"^GOTO\s+\w+$", inst)
        if m:
            return []
        m = re.match(r"^IF\s+(\S+)\s+\S+\s+(\S+)\s+THEN\s+\w+\s+ELSE\s+\w+$",
                     inst)
        if m:
            return [m.group(1), m.group(2)]
        m = re.match(r"^RETURN\s+(\S+)$", inst)
        if m:
            return [m.group(1)]
        m = re.match(r"^PAR\s+(\S+)$", inst)
        if m:
            return [m.group(1)]
        m = re.match(r"^(\w+)\s*=\s*CALL\s+\w+\s*,\s*\d+$", inst)
        if m:
            return [m.group(1)]
        m = re.match(r"^(\w+)\s*=\s*M\[(\w+)\]$", inst)
        if m:
            return [m.group(1), m.group(2)]
        m = re.match(r"^M\[(\w+)\]\s*=\s*(\S+)$", inst)
        if m:
            return [m.group(1), m.group(2)]
        m = re.match(r"^(\w+)\s*=\s*(\S+)\s*([\+\-\*\/])\s*(\S+)$", inst)
        if m:
            return [m.group(1), m.group(2), m.group(4)]
        m = re.match(r"^(\w+)\s*=\s*(\S+)$", inst)
        if m:
            return [m.group(1), m.group(2)]
        return []

    # -- emission helpers --------------------------------------------------
    def emit(self, s):
        self.out.append(s)

    def load(self, fn, operand, reg):
        """Load a value operand into `reg`."""
        if self.is_imm(operand):
            self.emit(f"\tMOV {reg}, #{operand}")
        elif self.is_global(operand):
            self.emit_addr_of(reg, PREFIX + operand)
        else:
            off = fn.slots[operand]
            self.emit(f"\tLDR {reg}, [sp, #{off}]")

    def emit_addr_of(self, reg, sym):
        """Materialise the address of a global (Q2ARM `ADR` template).

        Linux/Kunpeng accept a plain ADR (within +/-1MB).  macOS Mach-O needs
        the ADRP + ADD page/page-offset relocation pair instead.
        """
        if self.target == "macos":
            self.emit(f"\tadrp {reg}, {sym}@PAGE")
            self.emit(f"\tadd {reg}, {reg}, {sym}@PAGEOFF")
        else:
            self.emit(f"\tADR {reg}, {sym}")

    def store(self, fn, dest, reg):
        off = fn.slots[dest]
        self.emit(f"\tSTR {reg}, [sp, #{off}]")

    # -- program ----------------------------------------------------------
    def generate(self):
        self.emit_data()
        self.emit("\t.text")
        for fn in self.funcs:
            self.layout(fn)
            self.gen_func(fn)
        return "\n".join(self.out) + "\n"

    def emit_data(self):
        self.emit("\t.data")
        self.emit("\t.balign 8")
        for name, size in self.global_sizes.items():
            g = PREFIX + name
            self.emit(f"\t.global {g}")
            self.emit(f"\t.global _{g}")
            self.emit(f"{g}:")
            self.emit(f"_{g}:")
            self.emit(f"\t.skip {size}")

    def gen_func(self, fn):
        asm_name = PREFIX + fn.name
        self.emit("")
        self.emit(f"\t.global {asm_name}")
        self.emit(f"\t.global _{asm_name}")
        self.emit(f"{asm_name}:")
        self.emit(f"_{asm_name}:")
        # prologue (Q2ARM template)
        self.emit("\tstp x29, x30, [sp, #-16]!")
        self.emit("\tmov x29, sp")
        if fn.width:
            self.emit(f"\tsub sp, sp, #{fn.width}")
        # spill incoming parameters X0.. into their slots
        for idx, p in enumerate(fn.params):
            self.store(fn, p, f"X{idx}")

        pending = []                   # buffered PAR operands
        for inst in fn.insts:
            self.gen_inst(fn, inst, pending)

        # epilogue
        self.emit(f"{asm_name}_exit:")
        if fn.width:
            self.emit(f"\tadd sp, sp, #{fn.width}")
        self.emit("\tldp x29, x30, [sp], #16")
        self.emit("\tret")

    def gen_inst(self, fn, inst, pending):
        # LABEL l
        m = re.match(r"^LABEL\s+(\w+)$", inst)
        if m:
            name = m.group(1)
            if name in self.func_names:
                return                 # function entry label already emitted
            self.emit(f"{name}:")
            return
        # GOTO l
        m = re.match(r"^GOTO\s+(\w+)$", inst)
        if m:
            self.emit(f"\tB {self.label(m.group(1))}")
            return
        # IF q rop a THEN l1 ELSE l2
        m = re.match(r"^IF\s+(\S+)\s+(\S+)\s+(\S+)\s+THEN\s+(\w+)\s+ELSE\s+(\w+)$",
                     inst)
        if m:
            q, rop, a, l1, l2 = m.groups()
            self.load(fn, q, SCRATCH_A)
            self.load(fn, a, SCRATCH_B)
            self.emit(f"\tCMP {SCRATCH_A}, {SCRATCH_B}")
            self.emit(f"\t{JROP[rop]} {self.label(l1)}")
            self.emit(f"\tB {self.label(l2)}")
            return
        # RETURN a
        m = re.match(r"^RETURN\s+(\S+)$", inst)
        if m:
            self.load(fn, m.group(1), "X0")
            self.emit(f"\tB {PREFIX + fn.name}_exit")
            return
        # PAR a
        m = re.match(r"^PAR\s+(\S+)$", inst)
        if m:
            pending.append(m.group(1))
            return
        # q = CALL d, k
        m = re.match(r"^(\w+)\s*=\s*CALL\s+(\w+)\s*,\s*(\d+)$", inst)
        if m:
            q, callee, k = m.group(1), m.group(2), int(m.group(3))
            args = pending[-k:] if k else []
            for idx, a in enumerate(args):
                self.load(fn, a, f"X{idx}")
            del pending[len(pending) - k:]
            self.emit(f"\tBL {self.label(callee)}")
            self.store(fn, q, "X0")
            return
        # q = M[q2]   (load)
        m = re.match(r"^(\w+)\s*=\s*M\[(\w+)\]$", inst)
        if m:
            q, q2 = m.groups()
            self.load(fn, q2, SCRATCH_A)
            self.emit(f"\tLDR {SCRATCH_A}, [{SCRATCH_A}]")
            self.store(fn, q, SCRATCH_A)
            return
        # M[q1] = a   (store)
        m = re.match(r"^M\[(\w+)\]\s*=\s*(\S+)$", inst)
        if m:
            q1, a = m.groups()
            self.load(fn, q1, SCRATCH_A)
            self.load(fn, a, SCRATCH_B)
            self.emit(f"\tSTR {SCRATCH_B}, [{SCRATCH_A}]")
            return
        # q = a aop b
        m = re.match(r"^(\w+)\s*=\s*(\S+)\s*([\+\-\*\/])\s*(\S+)$", inst)
        if m:
            q, a, op, b = m.groups()
            self.load(fn, a, SCRATCH_A)
            self.load(fn, b, SCRATCH_B)
            self.emit(f"\t{AOP[op]} {SCRATCH_A}, {SCRATCH_A}, {SCRATCH_B}")
            self.store(fn, q, SCRATCH_A)
            return
        # q = a   (move)
        m = re.match(r"^(\w+)\s*=\s*(\S+)$", inst)
        if m:
            q, a = m.groups()
            self.load(fn, a, SCRATCH_A)
            self.store(fn, q, SCRATCH_A)
            return
        self.emit(f"\t// UNKNOWN: {inst}")


def generate(text, target="linux"):
    globals_, funcs = parse_qtac(text)
    return CodeGen(globals_, funcs, target).generate()


if __name__ == "__main__":
    tgt = "macos" if (len(sys.argv) > 1 and sys.argv[1] == "--macos") else "linux"
    sys.stdout.write(generate(sys.stdin.read(), tgt))
