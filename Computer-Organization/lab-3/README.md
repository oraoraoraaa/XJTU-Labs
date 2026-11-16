# Lab 3: adders and 32-bit ALU

All four XSim testbenches passed. See `output.txt`.

Design sources and testbenches are in `lab3-verilog/lab3-verilog.srcs`.
Open `lab3-verilog/lab3-verilog.xpr` in Vivado 2026.1, or source
`setup_lab3.tcl` in a project and run `lab3_sim tb_alu32`.
Run all tests from PowerShell with `.\run_all.ps1`.

Saved WDB files and simulation logs are in `lab3-verilog/.verification`.
To review without rerunning, source `view_results.tcl` in Vivado and use
`lab3_view 3-tb-alu32-flags`. The available views match PNG names without
extensions in `lab-report/images` (0 through 5).

ALU semantics: unsigned SLT; B shifted left by full A (A>=32 gives zero);
subtraction CF=borrow; OF only for add/sub; PF=odd parity over all 32 bits.
Disabled result and flags are high impedance. Reserved opcodes 8 through 15
produce zero with ZF=1 and other flags=0 when enabled.
The bus implementation uses one-hot operation selection and separate tri-state
drivers, with a structural ripple adder for add/sub.

`lab-report/report.tex` contains Chinese explanations, all six real Vivado
screenshots, the actual PASS summary, and embedded complete HDL sources.
It has not been compiled. No ZIP was created.
