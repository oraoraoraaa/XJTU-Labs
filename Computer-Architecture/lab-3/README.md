# Lab 3 Tomasulo Simulator

This directory contains a PyQt5-based Tomasulo simulator for the Computer Architecture lab.

## Features

- Simulates the Tomasulo execution flow
- GUI interaction with step, multi-step, and run-to-end controls
- Instruction status, reservation stations, register status, and memory history views
- Cycle history browsing for past states
- Execution statistics after the program finishes
- Supports `L.D`, `S.D`, `ADD.D`, `SUB.D`, `MUL.D`, and `DIV.D`
- Program input can be pasted directly or loaded from a file

## Run

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then start the simulator:

```bash
python tomasulo_simulator.py
```

You can also start with a file:

```bash
python tomasulo_simulator.py --file your_program.asm
```

## Input format

The editor accepts a simple MIPS-style subset. Optional initialization directives are also supported:

- `R1 = 100`
- `F2 = 3.5`
- `MEM[100] = 8.0`

Example instructions:

```asm
L.D F6, 0(R1)
ADD.D F8, F6, F2
MUL.D F10, F8, F2
S.D F10, 8(R1)
```

## Sample programs

Three ready-to-run examples are provided in `examples/`:

- `no_hazard.asm`: independent instruction mix with no structural or data hazard pressure
- `raw_hazard.asm`: contains at least one RAW dependence
- `war_hazard.asm`: contains a classic WAR pattern for scheduling demonstration
