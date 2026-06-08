"""QAST abstract syntax tree node definitions (Task 2AB).

The materials describe a *flattened* AST: precedence is carried by tree
shape rather than by extra non-terminal nodes.  We use one lightweight
``Node`` class with a ``kind`` tag and named fields, which keeps the tree
easy to traverse during QTAC generation and easy to pretty-print for the
deliverable ``.ast`` file.

Mapping to the QASTNode catalogue in materials.md:

    Program       <- P -> B               (globals + top-level statements)
    VarDecl       <- NodeDVar
    FuncDecl      <- NodeDFunc
    Param         <- NodeAVar
    Block         <- NodeB
    ExprStmt      <- NodeSExpr
    If            <- NodeIF
    While         <- NodeWHILE
    Return        <- NodeRETURN
    Nop           <- NodeNOP
    Num           <- NodeNUM
    Id            <- NodeID
    Index         <- NodeArrayAccess
    Call          <- NodeCall
    Unary         <- NodeUOP
    Binary        <- NodeAOP / NodeROP / NodeAND / NodeOR / assignment
"""


class Node:
    def __init__(self, kind, **fields):
        self.kind = kind
        self.fields = fields

    def __getattr__(self, name):
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name)

    def __repr__(self):
        return f"Node({self.kind}, {self.fields})"


# Convenience constructors -------------------------------------------------

def Program(decls, stmts):
    return Node("Program", decls=decls, stmts=stmts)


def VarDecl(name, dims):
    # dims = [] for a scalar, [n] for a 1-D array of n elements
    return Node("VarDecl", name=name, dims=dims)


def FuncDecl(name, ret_type, params, body):
    return Node("FuncDecl", name=name, ret_type=ret_type, params=params, body=body)


def Param(name, is_array):
    return Node("Param", name=name, is_array=is_array)


def Block(decls, stmts):
    return Node("Block", decls=decls, stmts=stmts)


def ExprStmt(expr):
    return Node("ExprStmt", expr=expr)


def If(cond, then, els):
    return Node("If", cond=cond, then=then, els=els)


def While(cond, body):
    return Node("While", cond=cond, body=body)


def Return(expr):
    return Node("Return", expr=expr)


def Nop():
    return Node("Nop")


def Num(value):
    return Node("Num", value=value)


def Id(name):
    return Node("Id", name=name)


def Index(base, index):
    return Node("Index", base=base, index=index)


def Call(name, args):
    return Node("Call", name=name, args=args)


def Unary(op, operand):
    return Node("Unary", op=op, operand=operand)


def Binary(op, left, right):
    return Node("Binary", op=op, left=left, right=right)


# Pretty-printer for the .ast deliverable ----------------------------------

def dump(node, indent=0):
    pad = "  " * indent
    if node is None:
        return f"{pad}<null>\n"

    k = node.kind
    if k == "Program":
        s = f"{pad}Program\n"
        s += f"{pad}  decls:\n"
        for d in node.decls:
            s += dump(d, indent + 2)
        s += f"{pad}  stmts:\n"
        for st in node.stmts:
            s += dump(st, indent + 2)
        return s
    if k == "VarDecl":
        t = f"int[{node.dims[0]}]" if node.dims else "int"
        return f"{pad}VarDecl {node.name}: {t}\n"
    if k == "FuncDecl":
        ps = ", ".join((p.name + "[]" if p.is_array else p.name) for p in node.params)
        s = f"{pad}FuncDecl {node.ret_type} {node.name}({ps})\n"
        s += dump(node.body, indent + 1)
        return s
    if k == "Block":
        s = f"{pad}Block\n"
        for d in node.decls:
            s += dump(d, indent + 1)
        for st in node.stmts:
            s += dump(st, indent + 1)
        return s
    if k == "ExprStmt":
        s = f"{pad}ExprStmt\n"
        return s + dump(node.expr, indent + 1)
    if k == "If":
        s = f"{pad}If\n{pad}  cond:\n" + dump(node.cond, indent + 2)
        s += f"{pad}  then:\n" + dump(node.then, indent + 2)
        if node.els is not None:
            s += f"{pad}  else:\n" + dump(node.els, indent + 2)
        return s
    if k == "While":
        s = f"{pad}While\n{pad}  cond:\n" + dump(node.cond, indent + 2)
        s += f"{pad}  body:\n" + dump(node.body, indent + 2)
        return s
    if k == "Return":
        s = f"{pad}Return\n"
        return s + (dump(node.expr, indent + 1) if node.expr else "")
    if k == "Nop":
        return f"{pad}Nop\n"
    if k == "Num":
        return f"{pad}Num {node.value}\n"
    if k == "Id":
        return f"{pad}Id {node.name}\n"
    if k == "Index":
        s = f"{pad}Index\n{pad}  base:\n" + dump(node.base, indent + 2)
        s += f"{pad}  index:\n" + dump(node.index, indent + 2)
        return s
    if k == "Call":
        s = f"{pad}Call {node.name}\n"
        for a in node.args:
            s += dump(a, indent + 1)
        return s
    if k == "Unary":
        return f"{pad}Unary {node.op}\n" + dump(node.operand, indent + 1)
    if k == "Binary":
        s = f"{pad}Binary {node.op}\n"
        s += dump(node.left, indent + 1)
        s += dump(node.right, indent + 1)
        return s
    return f"{pad}{k}\n"
