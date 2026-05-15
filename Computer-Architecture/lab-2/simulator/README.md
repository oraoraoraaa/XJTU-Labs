MIPS 5-Stage Pipeline Simulator

Features

- 5-stage pipeline (IF/ID/EX/MEM/WB)
- CLI interaction
- Step, run, breakpoints, run-to
- View pipeline registers and GPRs
- Optional data forwarding
- Basic performance stats
- Supported ops: lw, sw, add, beqz, nop

Usage

- Start CLI: python main.py
- Load from file: python main.py -f examples/sample.asm
- Toggle forwarding: python main.py --forwarding off

CLI Commands

- load <file>
- input (end with .end)
- step [n]
- run
- runto <pc|label>
- break <pc|label>
- breaks
- clear <pc|all>
- regs
- mem <addr> <count>
- memset <addr> <value>
- pipe
- stats
- forwarding on|off
- reset
- quit

Notes

- Memory is word-addressed via byte addresses; lw/sw require 4-byte alignment.
- PC and labels are instruction indices, not byte addresses.
- For examples, initialize memory with memset before running.
