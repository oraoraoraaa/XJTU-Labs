"""ARM64 peephole optimisation (Task step 4, optional).

The "everything spilled" code generator produces a lot of obvious local
redundancy.  Three safe, language-independent peephole passes clean it up
without changing program behaviour:

  1. redundant-branch removal
        B Lx
     Lx:                      -> drop the branch (jump to the next line)

  2. redundant-reload removal (store/load forwarding)
        STR Xr, [sp, #k]
        LDR Xr, [sp, #k]      -> drop the LDR (value already in Xr)
     and
        LDR Xr, [sp, #k]
        LDR Xr, [sp, #k]      -> drop the second LDR

  3. self-move removal
        MOV Xr, Xr            -> drop

The passes only look at adjacent instructions and never move code across a
label or branch, so liveness across basic blocks is unaffected.
"""

import re


def _is_label(line):
    return bool(re.match(r"^\w+:\s*$", line)) or line.rstrip().endswith(":")


def optimize(asm_text):
    lines = asm_text.splitlines()
    stats = {"branches": 0, "reloads": 0, "selfmoves": 0, "removed": 0}

    # pass 1: redundant branch to the immediately following label
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^\s*B\s+(\w+)$", line)
        if m and i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if nxt == f"{m.group(1)}:":
                stats["branches"] += 1
                i += 1
                continue
        out.append(line)
        i += 1
    lines = out

    # pass 2: store/load and load/load forwarding (adjacent, same slot/reg)
    out = []
    i = 0
    str_re = re.compile(r"^\s*STR\s+(X\w+),\s*\[sp, #(\d+)\]$")
    ldr_re = re.compile(r"^\s*LDR\s+(X\w+),\s*\[sp, #(\d+)\]$")
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            s = str_re.match(line)
            l2 = ldr_re.match(lines[i + 1])
            if s and l2 and s.group(1) == l2.group(1) and s.group(2) == l2.group(2):
                out.append(line)          # keep STR, drop the LDR
                stats["reloads"] += 1
                i += 2
                continue
            l1 = ldr_re.match(line)
            if l1 and l2 and l1.group(1) == l2.group(1) and l1.group(2) == l2.group(2):
                out.append(line)          # keep first LDR, drop duplicate
                stats["reloads"] += 1
                i += 2
                continue
        out.append(line)
        i += 1
    lines = out

    # pass 3: self-moves
    out = []
    for line in lines:
        m = re.match(r"^\s*MOV\s+(X\w+),\s*(X\w+)$", line)
        if m and m.group(1) == m.group(2):
            stats["selfmoves"] += 1
            continue
        out.append(line)
    lines = out

    stats["removed"] = stats["branches"] + stats["reloads"] + stats["selfmoves"]
    return "\n".join(lines) + "\n", stats


if __name__ == "__main__":
    import sys
    text, stats = optimize(sys.stdin.read())
    sys.stderr.write(f"removed {stats['removed']} instructions: {stats}\n")
    sys.stdout.write(text)
