`timescale 1ns / 1ps
// Cross-coupled NOR latch. S=R=1 is forbidden (Q=QN=0).
module sr_latch(input wire s, r, output wire q, qn);
    nor (q, r, qn);
    nor (qn, s, q);
endmodule
