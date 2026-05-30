# Lab 6 - Intermediate Code Generator

This program reads a small C-like language and emits quadruples (four-address code).
It includes a lexer, recursive-descent parser, basic symbol table checks, and
quadruple generation for expressions, assignments, control flow, and simple I/O.

## Input
- Keywords: int, float, void, if, else, while, return, input, print
- Expressions: +, -, *, /, assignment, relational operators
- Blocks: { ... }
- Arrays: name[index]
- Functions and function calls

See instruction/grammar.txt and the example images in instruction/.

## Output
Quadruples printed as:

  N. (op, arg1, arg2, res)

## Build
From this folder:

  cc -std=c11 -O2 -Wall -Wextra -o icg main.c

## Run
Read from a file:

  ./lab6 ../test-set/1.src

Or from stdin:

  ./lab6 < ../test-set/1.src

## Test All Cases

Run `test.sh`.
