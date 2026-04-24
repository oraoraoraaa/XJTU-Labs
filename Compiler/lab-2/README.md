# Lab 2 Scanner (C-like Language)

This lab implements a scanner using DFA-style state transitions in C.

## Build

```bash
gcc -std=c11 -Wall -Wextra -O2 main.c -o scanner
```

## Run

Default input/output:

```bash
./scanner
```

Custom input/output:

```bash
./scanner test.c tokens.out
```

The scanner prints tokens to terminal and also writes them to the output file.
