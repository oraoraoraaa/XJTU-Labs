"""Simple translator from quadruples to ARM64 assembly for a small subset.

Supported ops: =, +, -, *, /, neg, ret, print, input, param, call, j<, j<=, j>, j>=, j==, j!=, j

This is intentionally minimal and emits simple assembly using a stack-based
convention. It is an educational translator, not a production backend.
"""
import re
from typing import List, Tuple


def parse_quads(text: str) -> List[Tuple[str,str,str,str]]:
    quads = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+\. \(([^,]+), ?([^,]*), ?([^,]*), ?([^)]*)\)", line)
        if m:
            op = m.group(1).strip()
            a1 = m.group(2).strip()
            a2 = m.group(3).strip()
            res = m.group(4).strip()
            quads.append((op,a1,a2,res))
    return quads


def emit_header() -> List[str]:
    return [
        "    .text",
        "    .global main",
        "main:",
        "    stp x29, x30, [sp, #-16]!  // prologue",
        "    mov x29, sp",
        "",
    ]


def emit_footer() -> List[str]:
    return [
        "    mov x0, #0",
        "    ldp x29, x30, [sp], #16",
        "    ret",
    ]


def quad_to_arm(quads: List[Tuple[str,str,str,str]]) -> str:
    asm: List[str] = []
    asm.extend(emit_header())
    temp_map = {}
    next_local = 0

    def get_loc(name: str) -> str:
        # maps temporaries and variables to stack offsets
        nonlocal next_local
        if name == "" or name == " ":
            return None
        if re.match(r"T\d+", name):
            if name not in temp_map:
                next_local += 8
                temp_map[name] = -next_local
            return f"[sp, #{temp_map[name]}]"
        if re.match(r"^-?\d+(\.\d+)?$", name):
            return name
        # variables or arr[...] -> allocate as locals
        key = name
        if key not in temp_map:
            next_local += 8
            temp_map[key] = -next_local
        return f"[sp, #{temp_map[key]}]"

    for op,a1,a2,res in quads:
        if op == "=":
            src = get_loc(a1)
            dst = get_loc(res)
            if re.match(r"^-?\d+(\.\d+)?$", a1):
                asm.append(f"    mov x9, #{int(float(a1))}")
                asm.append(f"    str x9, {dst}")
            else:
                asm.append(f"    ldr x9, {src}")
                asm.append(f"    str x9, {dst}")
        elif op in ['+','-','*','/']:
            l = get_loc(a1)
            r = get_loc(a2)
            d = get_loc(res)
            asm.append(f"    ldr x9, {l}")
            asm.append(f"    ldr x10, {r}")
            if op == '+':
                asm.append(f"    add x11, x9, x10")
            elif op == '-':
                asm.append(f"    sub x11, x9, x10")
            elif op == '*':
                asm.append(f"    mul x11, x9, x10")
            elif op == '/':
                asm.append(f"    sdiv x11, x9, x10")
            asm.append(f"    str x11, {d}")
        elif op == 'neg':
            s = get_loc(a1)
            d = get_loc(res)
            asm.append(f"    ldr x9, {s}")
            asm.append(f"    neg x9, x9")
            asm.append(f"    str x9, {d}")
        elif op.startswith('j'):
            # conditional and unconditional jumps; res contains target index
            if op == 'j':
                asm.append(f"    b .L{res}")
            else:
                l = get_loc(a1)
                r = get_loc(a2)
                asm.append(f"    ldr x9, {l}")
                asm.append(f"    ldr x10, {r}")
                if op == 'j<':
                    asm.append(f"    cmp x9, x10")
                    asm.append(f"    blt .L{res}")
                elif op == 'j<=':
                    asm.append(f"    cmp x9, x10")
                    asm.append(f"    ble .L{res}")
                elif op == 'j>':
                    asm.append(f"    cmp x9, x10")
                    asm.append(f"    bgt .L{res}")
                elif op == 'j>=':
                    asm.append(f"    cmp x9, x10")
                    asm.append(f"    bge .L{res}")
                elif op == 'j==':
                    asm.append(f"    cmp x9, x10")
                    asm.append(f"    beq .L{res}")
                elif op == 'j!=':
                    asm.append(f"    cmp x9, x10")
                    asm.append(f"    bne .L{res}")
        elif op == 'ret':
            if a1:
                # move return value to x0
                if re.match(r"^-?\d+(\.\d+)?$", a1):
                    asm.append(f"    mov x0, #{int(float(a1))}")
                else:
                    src = get_loc(a1)
                    asm.append(f"    ldr x0, {src}")
            asm.append("    ldp x29, x30, [sp], #16")
            asm.append("    ret")
        elif op == 'print':
            # simple: ignore and leave a nop comment
            asm.append(f"    // print {a1}")
        elif op == 'input':
            asm.append(f"    // input -> {res}")
        elif op == 'call':
            asm.append(f"    // call {a1} with {a2} args -> {res}")
        elif op == 'param':
            asm.append(f"    // param {a1}")
        else:
            asm.append(f"    // unsupported op: {op} {a1} {a2} {res}")

    # Add labels for jump targets (simple approach: place labels before corresponding quad indices)
    # We'll convert to final output by inserting labels at appropriate quad positions.

    # Prepend stack allocation if any locals used
    if next_local > 0:
        alloc = [f"    sub sp, sp, #{next_local + 16}  // allocate locals"]
        asm = asm[:0] + alloc + asm

    # Append epilogue if not already returned
    if not any(line.strip().endswith('ret') for line in asm):
        asm.extend(emit_footer())

    return '\n'.join(asm)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: quad_to_arm.py <quads.txt>')
        sys.exit(1)
    text = open(sys.argv[1]).read()
    quads = parse_quads(text)
    asm = quad_to_arm(quads)
    print(asm)
