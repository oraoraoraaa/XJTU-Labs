"""QAST -> QTAC three-address code + symbol table (Task 2AB).

Walks the QAST and emits QTAC following the grammar in materials.md:

    instruction forms emitted
        LABEL l                         basic-block entry
        GOTO l                          unconditional jump
        IF q rop a THEN l ELSE l        conditional jump
        q = a                           move
        q = a aop a                     binary arithmetic
        q = M[q]                        load from memory
        M[q] = a                        store to memory
        PAR a                           pass an argument
        q = CALL d, k                   call with k args, result in q
        RETURN a                        return
        define d(a1,..,an){ ... }       function wrapper

Word size is 8 bytes (64-bit int / long), matching the reference QTAC where
`int a[10]` becomes the 80-byte global `i80 a`.  Array element addresses are
therefore computed as  base + index*8  (the reference text omits the *8
scaling; we make it explicit so the generated code is directly executable).
"""

import qast

WORD = 8


class Sym:
    """A name visible during code generation."""
    def __init__(self, name, kind, is_array=False):
        self.name = name        # source name
        self.kind = kind        # 'global' | 'param' | 'local'
        self.is_array = is_array


class IRGen:
    def __init__(self):
        self.code = []          # list of QTAC instruction strings (no ';')
        self.globals = {}       # name -> (size_bytes, n_elems)
        self.functions = []     # (name, ret, [Param])
        self.tcount = 0
        self.lcount = 0
        self.scope = {}         # name -> Sym (current function scope)
        self.symtab_lines = []

    # -- helpers -----------------------------------------------------------
    def new_temp(self):
        t = f"t{self.tcount}"
        self.tcount += 1
        return t

    def new_label(self):
        l = f"l{self.lcount}"
        self.lcount += 1
        return l

    def emit(self, s):
        self.code.append(s)

    def lookup(self, name):
        return self.scope.get(name)

    def is_array_name(self, name):
        s = self.lookup(name)
        if s is not None:
            return s.is_array
        if name in self.globals:
            return self.globals[name][1] is not None
        return False

    # -- program -----------------------------------------------------------
    def gen_program(self, prog):
        # 1. globals
        for d in prog.decls:
            if d.kind == "VarDecl":
                n_elems = d.dims[0] if d.dims else None
                size = (n_elems if n_elems else 1) * WORD
                self.globals[d.name] = (size, n_elems)
                self.emit(f"i{size} {d.name}")
                t = f"int[{n_elems}]" if n_elems else "int"
                self.symtab_lines.append(f"[GLOBAL] {d.name} : {t}  size={size}")
        # 2. functions
        for d in prog.decls:
            if d.kind == "FuncDecl":
                self.gen_func(d)
        # 3. top-level statements -> synthetic `entry` function
        self.functions.append(("entry", "int", []))
        self.symtab_lines.append("[FUNCTION] entry() -> int  (program entry)")
        self.scope = {}
        self.emit("")
        self.emit("define entry(){")
        self.emit("LABEL entry")
        for st in prog.stmts:
            self.gen_stmt(st)
        self.emit("RETURN 0")
        self.emit("}")

    # -- function ----------------------------------------------------------
    def gen_func(self, fn):
        self.functions.append((fn.name, fn.ret_type, fn.params))
        self.scope = {}
        for p in fn.params:
            self.scope[p.name] = Sym(p.name, "param", p.is_array)

        pdesc = ", ".join((p.name + ":int[]" if p.is_array else p.name + ":int")
                          for p in fn.params)
        self.symtab_lines.append(
            f"[FUNCTION] {fn.name}({pdesc}) -> {fn.ret_type}")
        # register locals declared in the body block (listed under this header)
        self._register_locals(fn.body)

        ps = ", ".join(p.name for p in fn.params)
        self.emit("")
        self.emit(f"define {fn.name}({ps}){{")
        self.emit(f"LABEL {fn.name}")
        self.gen_block(fn.body)
        # void / fall-through functions return 0
        self.emit("RETURN 0")
        self.emit("}")

    def _register_locals(self, block):
        for d in block.decls:
            if d.kind == "VarDecl":
                is_arr = bool(d.dims)
                self.scope[d.name] = Sym(d.name, "local", is_arr)
                t = f"int[{d.dims[0]}]" if d.dims else "int"
                self.symtab_lines.append(f"    local {d.name} : {t}")

    # -- statements --------------------------------------------------------
    def gen_block(self, block):
        for st in block.stmts:
            self.gen_stmt(st)

    def gen_stmt(self, st):
        k = st.kind
        if k == "Nop":
            return
        if k == "Block":
            self.gen_block(st)
            return
        if k == "ExprStmt":
            self.gen_expr(st.expr)
            return
        if k == "Return":
            if st.expr is not None:
                v = self.gen_expr(st.expr)
                self.emit(f"RETURN {v}")
            else:
                self.emit("RETURN 0")
            return
        if k == "If":
            l_then = self.new_label()
            l_end = self.new_label()
            if st.els is not None:
                l_else = self.new_label()
                self.gen_cond(st.cond, l_then, l_else)
                self.emit(f"LABEL {l_then}")
                self.gen_stmt(st.then)
                self.emit(f"GOTO {l_end}")
                self.emit(f"LABEL {l_else}")
                self.gen_stmt(st.els)
                self.emit(f"LABEL {l_end}")
            else:
                self.gen_cond(st.cond, l_then, l_end)
                self.emit(f"LABEL {l_then}")
                self.gen_stmt(st.then)
                self.emit(f"LABEL {l_end}")
            return
        if k == "While":
            l_top = self.new_label()
            l_body = self.new_label()
            l_end = self.new_label()
            self.emit(f"LABEL {l_top}")
            self.gen_cond(st.cond, l_body, l_end)
            self.emit(f"LABEL {l_body}")
            self.gen_stmt(st.body)
            self.emit(f"GOTO {l_top}")
            self.emit(f"LABEL {l_end}")
            return
        raise RuntimeError(f"cannot generate statement {k}")

    # -- conditions: emit branching code ----------------------------------
    REL = {"<", "<=", ">", ">=", "==", "!="}

    def gen_cond(self, c, l_true, l_false):
        if c.kind == "Binary" and c.op in self.REL:
            a = self.gen_expr(c.left)
            b = self.gen_expr(c.right)
            a = self._as_register(a)
            self.emit(f"IF {a} {c.op} {b} THEN {l_true} ELSE {l_false}")
            return
        if c.kind == "Binary" and c.op == "&&":
            mid = self.new_label()
            self.gen_cond(c.left, mid, l_false)
            self.emit(f"LABEL {mid}")
            self.gen_cond(c.right, l_true, l_false)
            return
        if c.kind == "Binary" and c.op == "||":
            mid = self.new_label()
            self.gen_cond(c.left, l_true, mid)
            self.emit(f"LABEL {mid}")
            self.gen_cond(c.right, l_true, l_false)
            return
        if c.kind == "Unary" and c.op == "!":
            self.gen_cond(c.operand, l_false, l_true)
            return
        # bare expression  E  used as a condition: E != 0
        v = self.gen_expr(c)
        v = self._as_register(v)
        self.emit(f"IF {v} != 0 THEN {l_true} ELSE {l_false}")

    def _as_register(self, v):
        """IF requires its first operand to be a register; box immediates."""
        if str(v).lstrip("-").isdigit():
            t = self.new_temp()
            self.emit(f"{t} = {v}")
            return t
        return v

    # -- expressions: return an operand (temp / var / global / immediate) --
    def gen_expr(self, e):
        k = e.kind
        if k == "Num":
            return str(e.value)
        if k == "Id":
            # an array name evaluates to its base address; scalars to value
            return e.name
        if k == "Index":
            addr = self.gen_addr(e)
            t = self.new_temp()
            self.emit(f"{t} = M[{addr}]")
            return t
        if k == "Call":
            return self.gen_call(e)
        if k == "Unary":
            if e.op == "-":
                v = self.gen_expr(e.operand)
                t = self.new_temp()
                self.emit(f"{t} = 0 - {v}")
                return t
            raise RuntimeError(f"unary {e.op} not valid in value context")
        if k == "Binary":
            if e.op in ("=", "+="):
                return self.gen_assign(e)
            if e.op in ("+", "-", "*", "/"):
                a = self.gen_expr(e.left)
                b = self.gen_expr(e.right)
                a = self._as_register(a)
                t = self.new_temp()
                self.emit(f"{t} = {a} {e.op} {b}")
                return t
            # relational / logical evaluated as a value -> materialise 0/1
            t = self.new_temp()
            l_true = self.new_label()
            l_false = self.new_label()
            l_end = self.new_label()
            self.gen_cond(e, l_true, l_false)
            self.emit(f"LABEL {l_true}")
            self.emit(f"{t} = 1")
            self.emit(f"GOTO {l_end}")
            self.emit(f"LABEL {l_false}")
            self.emit(f"{t} = 0")
            self.emit(f"LABEL {l_end}")
            return t
        raise RuntimeError(f"cannot generate expression {k}")

    # -- address of an lvalue (array element) ------------------------------
    def gen_addr(self, e):
        if e.kind == "Index":
            base = self.gen_expr(e.base)          # base address operand
            idx = self.gen_expr(e.index)
            idx = self._as_register(idx)
            off = self.new_temp()
            self.emit(f"{off} = {idx} * {WORD}")
            addr = self.new_temp()
            self.emit(f"{addr} = {base} + {off}")
            return addr
        raise RuntimeError("address of non-lvalue")

    # -- assignment --------------------------------------------------------
    def gen_assign(self, e):
        lhs = e.left
        if e.op == "+=":
            # lhs += rhs   ==>   lhs = lhs + rhs
            rhs = qast.Binary("+", lhs, e.right)
        else:
            rhs = e.right

        if lhs.kind == "Id":
            v = self.gen_expr(rhs)
            self.emit(f"{lhs.name} = {v}")
            return lhs.name
        if lhs.kind == "Index":
            addr = self.gen_addr(lhs)
            v = self.gen_expr(rhs)
            self.emit(f"M[{addr}] = {v}")
            return v
        raise RuntimeError("invalid assignment target")

    # -- call --------------------------------------------------------------
    def gen_call(self, c):
        arg_ops = [self.gen_expr(a) for a in c.args]
        for op in arg_ops:
            self.emit(f"PAR {op}")
        t = self.new_temp()
        self.emit(f"{t} = CALL {c.name}, {len(arg_ops)}")
        return t


def generate(prog):
    g = IRGen()
    g.gen_program(prog)
    return g


def emit_qtac(g):
    """Serialise the QTAC program to text (one instruction per line)."""
    out = []
    for line in g.code:
        if line == "" or line.endswith("{") or line == "}":
            out.append(line)
        else:
            out.append(line + ";")
    return "\n".join(out) + "\n"


def emit_symtab(g):
    return "; QTAC symbol table\n" + "\n".join(g.symtab_lines) + "\n"


if __name__ == "__main__":
    import sys
    from lexer import tokenize
    from parser_ll1 import parse
    tree = parse(tokenize(sys.stdin.read()))
    g = generate(tree)
    sys.stdout.write(emit_qtac(g))
