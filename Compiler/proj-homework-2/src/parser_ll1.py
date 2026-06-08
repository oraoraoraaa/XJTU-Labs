"""QL syntactic analysis -- LL(1) recursive descent (Task step 2A, path A).

The raw QL grammar in materials.md is left-recursive and ambiguous w.r.t.
operator precedence, so it is first transformed for predictive parsing:

1.  Left recursion removed:
       V -> d | V[E]          ==>  V -> d ('[' E ']')*
       E -> E o E             ==>  precedence ladder (see below)
       C -> C && C | C || C   ==>  folded into the same ladder
       Ď -> Ď D; , Ǎ -> Ǎ A;  ==>  iterative list parsing
       Ř -> Ř E,              ==>  iterative argument parsing
2.  Left factoring after an identifier (the only place needing >1 token of
    distinction) -- SELECT sets are disjoint with one token of look-ahead:
       primary -> d              SELECT = { ID not followed by '(' or '[' }
       primary -> d '(' Ř ')'    SELECT = { ID followed by '(' }
       postfix -> ... '[' E ']'  SELECT = { ID/postfix followed by '[' }
3.  Expressions and conditions share one precedence ladder so that the
    "( E )" vs "( C )" ambiguity disappears:

       assign      -> logic_or ( ('=' | '+=') assign )?      right assoc
       logic_or    -> logic_and ( '||' logic_and )*
       logic_and   -> equality  ( '&&' equality  )*
       equality    -> relational ( ('==' | '!=') relational )*
       relational  -> additive  ( ('<'|'<='|'>'|'>=') additive )*
       additive    -> mul ( ('+'|'-') mul )*
       mul         -> unary ( ('*'|'/') unary )*
       unary       -> ('-'|'!') unary | postfix
       postfix     -> primary ( '[' E ']' | '[' ']' )*
       primary     -> NUM | '(' assign ')' | ID | ID '(' args ')'

The declaration-vs-statement decision inside a block is made with one
token of look-ahead: a type keyword (int/void) starts a declaration,
anything else starts a statement -- disjoint SELECT sets.
"""

import qast


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0

    # -- token helpers -----------------------------------------------------
    @property
    def cur(self):
        return self.toks[self.pos]

    def peek(self, k=1):
        j = self.pos + k
        return self.toks[j] if j < len(self.toks) else self.toks[-1]

    def at(self, value):
        return self.cur.value == value

    def at_type(self, ttype):
        return self.cur.type == ttype

    def advance(self):
        t = self.cur
        self.pos += 1
        return t

    def expect(self, value):
        if self.cur.value != value:
            raise ParseError(
                f"line {self.cur.line}: expected {value!r} but found "
                f"{self.cur.value!r}")
        return self.advance()

    def expect_type(self, ttype):
        if self.cur.type != ttype:
            raise ParseError(
                f"line {self.cur.line}: expected {ttype} but found "
                f"{self.cur.value!r}")
        return self.advance()

    # -- entry: P -> B -----------------------------------------------------
    def parse_program(self):
        self.expect("{")
        decls, stmts = self.parse_decls_and_stmts()
        self.expect("}")
        if not self.at_type("EOF"):
            raise ParseError(f"line {self.cur.line}: trailing input "
                             f"{self.cur.value!r}")
        return qast.Program(decls, stmts)

    # -- B body: Ď then Š --------------------------------------------------
    def parse_decls_and_stmts(self):
        decls = []
        while self.cur.value in ("int", "void"):
            decls.append(self.parse_decl())
        stmts = []
        while not self.at("}"):
            stmts.append(self.parse_stmt())
        return decls, stmts

    # -- D -> T V ; | H B  (var or function) -------------------------------
    def parse_decl(self):
        ret_type = self.advance().value          # int | void
        name = self.expect_type("ID").value
        if self.at("("):                         # function: H -> T d ( Ǎ )
            params = self.parse_params()
            body = self.parse_block()
            return qast.FuncDecl(name, ret_type, params, body)
        # variable: optional [ NUM ]
        dims = []
        if self.at("["):
            self.expect("[")
            dims.append(int(self.expect_type("NUM").value))
            self.expect("]")
        self.expect(";")
        return qast.VarDecl(name, dims)

    # -- ( Ǎ )  where A -> T d | T d [] , separated/terminated by ';' ------
    def parse_params(self):
        self.expect("(")
        params = []
        while not self.at(")"):
            self.advance()                       # type (int)
            pname = self.expect_type("ID").value
            is_array = False
            if self.at("["):
                self.expect("[")
                self.expect("]")
                is_array = True
            params.append(qast.Param(pname, is_array))
            self.expect(";")
        self.expect(")")
        return params

    # -- B -> { Ď Š } ------------------------------------------------------
    def parse_block(self):
        self.expect("{")
        decls, stmts = self.parse_decls_and_stmts()
        self.expect("}")
        return qast.Block(decls, stmts)

    # -- S ----------------------------------------------------------------
    def parse_stmt(self):
        v = self.cur.value
        if v == ";":
            self.advance()
            return qast.Nop()
        if v == "{":
            return self.parse_block()
        if v == "if":
            return self.parse_if()
        if v == "while":
            return self.parse_while()
        if v == "return":
            self.advance()
            expr = None
            if not self.at(";"):
                expr = self.parse_expr()
            self.expect(";")
            return qast.Return(expr)
        # E ;
        expr = self.parse_expr()
        self.expect(";")
        return qast.ExprStmt(expr)

    def parse_if(self):
        self.expect("if")
        self.expect("(")
        cond = self.parse_expr()
        self.expect(")")
        then = self.parse_stmt()
        els = None
        if self.at("else"):
            self.advance()
            els = self.parse_stmt()
        return qast.If(cond, then, els)

    def parse_while(self):
        self.expect("while")
        self.expect("(")
        cond = self.parse_expr()
        self.expect(")")
        body = self.parse_stmt()
        return qast.While(cond, body)

    # -- expression precedence ladder -------------------------------------
    def parse_expr(self):
        return self.parse_assign()

    def parse_assign(self):
        left = self.parse_logic_or()
        if self.cur.value in ("=", "+="):
            op = self.advance().value
            right = self.parse_assign()          # right associative
            return qast.Binary(op, left, right)
        return left

    def _left_assoc(self, sub, ops):
        node = sub()
        while self.cur.value in ops:
            op = self.advance().value
            node = qast.Binary(op, node, sub())
        return node

    def parse_logic_or(self):
        return self._left_assoc(self.parse_logic_and, ("||",))

    def parse_logic_and(self):
        return self._left_assoc(self.parse_equality, ("&&",))

    def parse_equality(self):
        return self._left_assoc(self.parse_relational, ("==", "!="))

    def parse_relational(self):
        return self._left_assoc(self.parse_additive, ("<", "<=", ">", ">="))

    def parse_additive(self):
        return self._left_assoc(self.parse_mul, ("+", "-"))

    def parse_mul(self):
        return self._left_assoc(self.parse_unary, ("*", "/"))

    def parse_unary(self):
        if self.cur.value in ("-", "!"):
            op = self.advance().value
            return qast.Unary(op, self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()
        while self.at("["):
            self.expect("[")
            if self.at("]"):                     # array-decay form  arr[]
                self.expect("]")
                # leave `node` unchanged: an array name already denotes its
                # base address when used as a value / argument
            else:
                idx = self.parse_expr()
                self.expect("]")
                node = qast.Index(node, idx)
        return node

    def parse_primary(self):
        t = self.cur
        if t.type == "NUM":
            self.advance()
            return qast.Num(int(t.value))
        if t.value == "(":
            self.advance()
            e = self.parse_expr()
            self.expect(")")
            return e
        if t.type == "ID":
            self.advance()
            if self.at("("):                     # call: d ( Ř )
                args = self.parse_args()
                return qast.Call(t.value, args)
            return qast.Id(t.value)
        raise ParseError(f"line {t.line}: unexpected token {t.value!r}")

    # -- ( Ř )  arguments, comma separated/terminated ---------------------
    def parse_args(self):
        self.expect("(")
        args = []
        while not self.at(")"):
            args.append(self.parse_expr())
            if self.at(","):
                self.advance()
            else:
                break
        self.expect(")")
        return args


def parse(tokens):
    return Parser(tokens).parse_program()


if __name__ == "__main__":
    import sys
    from lexer import tokenize
    tree = parse(tokenize(sys.stdin.read()))
    sys.stdout.write(qast.dump(tree))
