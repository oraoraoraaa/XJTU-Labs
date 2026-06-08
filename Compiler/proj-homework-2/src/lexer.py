"""QL lexical analyzer (Task step 1, common to A/B/C).

Reads a QL source program and produces a token stream in (TYPE, value)
format.  QL lexical conventions (materials.md):

    i = [0-9]+            integer literal
    d = [a-zA-Z]+         identifier / keyword
    o = + - * / = +=      arithmetic / assignment operators
    r = < <= > >= == !=   relational operators
    logical: ! && ||
    delimiters: ( ) { } [ ] ; ,

The PDF source uses a few "typeset" glyphs that we normalise to ASCII:
    ×  -> *      (multiplication)
    –  -> -      (en-dash used as minus)
"""

import re

KEYWORDS = {"int", "void", "if", "else", "while", "return"}

# Multi-character operators must be tried before single-character ones.
MULTI_OPS = ["<=", ">=", "==", "!=", "&&", "||", "+="]
SINGLE_OPS = set("+-*/=<>!")
DELIMS = set("(){}[];,")

# Normalisation of non-ASCII glyphs coming from the PDF/Word source.
GLYPH_FIX = {
    "×": "*",   # ×  multiply
    "–": "-",   # –  en dash
    "—": "-",   # —  em dash
    "−": "-",   # −  minus sign
}


class Token:
    def __init__(self, type_, value, line, col):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"({self.type}, {self.value})"


class LexError(Exception):
    pass


def normalise(text):
    for bad, good in GLYPH_FIX.items():
        text = text.replace(bad, good)
    return text


def tokenize(text):
    text = normalise(text)
    tokens = []
    line, col = 1, 1
    i, n = 0, len(text)

    def adv(k=1):
        nonlocal i, col
        i += k
        col += k

    while i < n:
        c = text[i]

        # newline / whitespace
        if c == "\n":
            line += 1
            col = 1
            i += 1
            continue
        if c in " \t\r":
            adv()
            continue

        # line comment  //...
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue

        start_col = col

        # integer literal
        if c.isdigit():
            m = re.match(r"[0-9]+", text[i:])
            tokens.append(Token("NUM", m.group(0), line, start_col))
            adv(len(m.group(0)))
            continue

        # identifier / keyword
        if c.isalpha() or c == "_":
            m = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", text[i:])
            word = m.group(0)
            ttype = "KEYWORD" if word in KEYWORDS else "ID"
            tokens.append(Token(ttype, word, line, start_col))
            adv(len(word))
            continue

        # multi-char operator
        two = text[i:i + 2]
        if two in MULTI_OPS:
            tokens.append(Token("OP", two, line, start_col))
            adv(2)
            continue

        # single-char operator
        if c in SINGLE_OPS:
            tokens.append(Token("OP", c, line, start_col))
            adv()
            continue

        # delimiter
        if c in DELIMS:
            tokens.append(Token("DELIM", c, line, start_col))
            adv()
            continue

        raise LexError(f"illegal character {c!r} at line {line}, col {col}")

    tokens.append(Token("EOF", "$", line, col))
    return tokens


def dump_tokens(tokens):
    """Render the token list in the spec's (TYPE, value) format."""
    out = []
    for t in tokens:
        if t.type == "EOF":
            continue
        out.append(f"({t.type}, {t.value})")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    import sys
    src = sys.stdin.read()
    sys.stdout.write(dump_tokens(tokenize(src)))
