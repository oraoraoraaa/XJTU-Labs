# Pipeline Scenario Analysis

## A. No Conflict Scenario (`no_hazard.s`)

- Key sequence: `LW -> LW -> LW -> ADD -> ADD -> SW -> SW`
- Design idea: each `ADD` reads registers whose values are already written back with enough spacing from previous `LW` instructions.
- Expected behavior: no RAW dependency that forces a stall, and no control hazard.

## B. RAW Conflict Scenario (`raw_hazard.s`)

- Key sequence: `LW $r1,VAL($r0)` followed immediately by `ADD $r2,$r1,$r3`.
- Conflict type: RAW (Read After Write).
- Reason: `ADD` needs `$r1` before `LW` has fully completed write-back.
- Expected behavior: pipeline inserts a bubble/stall (or relies on forwarding rules, depending on simulator settings).

## C. Branch Jump Scenario (`branch_jump.s`)

- Key sequence: `BEQZ $r1,DO_ADD`, and `FLAG` is initialized to `0`.
- Branch result: branch is taken at least once, control flow jumps to label `DO_ADD`.
- Expected behavior: one control-hazard case appears (possible flush or delay-slot effect based on simulator model).

## D. Summary

- `no_hazard.s`: demonstrates a smooth pipeline with no intentional hazard.
- `raw_hazard.s`: demonstrates at least one RAW data hazard.
- `branch_jump.s`: demonstrates at least one branch jump/control hazard.
- Instruction set used for demos focuses on `LW`, `SW`, `ADD`, `BEQZ` (plus `TEQ` as stop marker in this simulator).
