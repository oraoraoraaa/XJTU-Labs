# Lab 2: sequential logic (Verilog / Vivado 2026.1)

Student-ID suffix: **99**. All nine self-checking testbenches passed in XSim.
The actual PASS summary is in `output.txt`. The report is in
`lab-report/report.tex` and has **not** been compiled.

## Run or review

In an existing Vivado project, load the sources with:

```tcl
source D:/Local/Github/XJTU-Labs/Computer-Organization/lab-2/setup_lab2.tcl
lab2_sim tb_counter_mod99
```

Replace the simulation top using the table below. Each testbench ends with
`$finish` and prints a PASS line only after its checks succeed.

To review the already-recorded waveforms without rerunning:

```tcl
source D:/Local/Github/XJTU-Labs/Computer-Organization/lab-2/view_results.tcl
lab2_view 6-tb-counter-mod99
```

Use any view name in the screenshot column (without `.png`). Wave configurations
are saved in `lab2-verilog/lab2-verilog.srcs/sim_1/wave`; WDB files and complete
compiler/elaboration/simulation logs are in `lab2-verilog/.verification`.
The repository ignores generated Vivado output; run `run_all.ps1` to regenerate it.

From PowerShell, all tests can also be run using:

```powershell
.\run_all.ps1
# If Vivado is installed elsewhere:
.\run_all.ps1 -VivadoBin 'D:\Software\AMDDesignTools\2026.1\Vivado\bin'
```

## Screenshots and coverage

All image paths below are relative to `lab-report/images`. The existing image 0
was preserved; images 1–9 were captured from the native Vivado waveform viewer.

| Simulation top | Screenshot | Main observation |
|---|---|---|
| `tb_latch_vs_ff` | `0-tb-latch-vs-ff.png` | Latch transparency versus edge sampling |
| `tb_sr_latch` | `1-tb-sr-latch.png` | Set, reset, hold, forbidden S=R=1 |
| `tb_d_latch_reset` | `2-tb-d-latch-reset.png` | Transparency, hold, reset priority |
| `tb_d_ff_control` | `3-tb-d-ff-control.png` | Enable; reset at 32 ns; sync reset at 35 ns |
| `tb_register_n` | `4-tb-register-n.png` | Parallel load and OE-controlled high impedance |
| `tb_shift_register` | `5-tb-shift-register.png` | Four modes; active-high asynchronous clear |
| `tb_counter_mod99` | `6-tb-counter-mod99.png` | 900–1040 ns: 98 → 0 at 995 ns |
| `tb_counter_mod99` | `7-tb-counter-mod99-controls.png` | 1970–2090 ns: enable, load, reset priority |
| `tb_fsm_div3` | `8-tb-fsm-div3.png` | Case FSM and three-DFF ring states agree |
| `tb_sequence_01` | `9-tb-sequence-01.png` | Sampled 1001101 produces 0001001 |

Additional automated checks include 198 consecutive counter increments,
128 shift-register mode/data/serial cases, and all 64 six-bit detector streams.

## Implementation notes

- Flip-flop and latch resets are active high; counter `cr_n` and `pe_n` are active low.
  The shift-register reset follows the slide's Verilog example (active high).
- The parallel register captures on each rising edge. `oe=1` makes its external
  output high impedance while retaining its internal storage.
- FSM next-state logic is combinational; the state register uses nonblocking
  assignments. This avoids the clocked-next-state race in the slide examples.
- The modulo-99 counter uses the requested asynchronous feedback clear decoded
  at 99. In zero-delay RTL its clear pulse has delta-cycle duration; `wraps` counts
  those events so the screenshots can show that feedback occurred. The stable
  sequence is 0 through 98. `tc` indicates 98 when `cet=1`.
- This is a behavioral-simulation lab. The asynchronous clear feedback is not
  a pulse-width-qualified physical FPGA reset network.

The report contains the source code and actual output directly, so it needs only
`report.tex` and the `images` folder when moved to another machine.
