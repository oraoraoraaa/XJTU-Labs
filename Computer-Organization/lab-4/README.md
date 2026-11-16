# Lab 4: memory arrays

All five XSim testbenches passed; actual summary: `output.txt`.
Open `lab4-verilog/lab4-verilog.xpr` in Vivado 2026.1, or source
`setup_lab4.tcl` and call `lab4_sim tb_register_file`.
Run all tests from this directory using `.\run_all.ps1`.

RTL: `lab4-verilog/lab4-verilog.srcs/sources_1/new`.
Testbenches: `lab4-verilog/lab4-verilog.srcs/sim_1/new`.
The 64-word ROM initialization file is `imem.hex`; the runner copies it into
its simulation working directory. It is registered as a memory initialization
file in the Vivado project.

Implemented: 1Kx16 RAM, 4Kx32 RAM using eight 1Kx16 chips, optional bidirectional
1Kx16 and 4Kx32 RAM, 64x32 instruction ROM, 32x32 two-read/one-write register file.
Writes occur on rising edges; reads are asynchronous. Reset is active high and
clears all words, regardless of chip selection. In split-data RAM, re=1/we=0
reads, re=0/we=1 writes; neither or both releases the output and does not write.
Bidirectional RAM uses wr=1 read, wr=0 write. Host must release the bus for reads.
Register zero ignores writes and always reads zero; nonzero registers update
after the write edge, with no pre-edge bypass.

Six actual Vivado screenshots are in `lab-report/images`, numbered 0 through 5.
Saved WDB files and full logs are in `lab4-verilog/.verification`.
To view saved results, source `view_results.tcl` and call
`lab4_view 2-tb-ram-bank-boundary` (or any other image stem).
Generated simulator files are ignored by the repository and can be regenerated.

The Chinese report is `lab-report/report.tex`, with figures, actual test output,
and embedded complete source. No LaTeX compilation or ZIP creation was performed.
